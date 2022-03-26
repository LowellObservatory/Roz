# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 19-Oct-2021
#
#  @author: tbowers

"""Module for gathering and moving around whatever frames are required by Roz.

This module is part of the Roz package, written at Lowell Observatory.

This module buttles its own data from the site NAS (which will be mounted in
the container running this package) and to the MH storage location (also
mounted as a local directory).  The paths for these local mount points are
specified in `roz.conf`.

This module primarily trades in CCDPROC Image File Collections
(`ccdproc.ImageFileCollection`).
"""

# Built-In Libraries
import os
import pathlib
import re
import shutil
import tarfile
import warnings

# 3rd Party Libraries
from astropy.io.fits import getheader
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
import ccdproc as ccdp
import numpy as np
from tqdm import tqdm

# Internal Imports
from roz import send_alerts as sa
from roz import utils

# Currently Supported Frameclasses
FRAMECLASSES = ["calibration", "science"]


class Dumbwaiter:
    """dumbwaiter Class for moving data between floors (servers)

    It seemed easier to contain in one place all of the data and methods
    related to identifying the appropriate frames, copying them to a processing
    location, and packaging them for cold storage.

    Roz moves its own data around, without the help of other LIG workers, so
    the concept of a dumbwaiter seemed appropriate.

    NOTE: 'calibration' is the ONLY type of frame currently supported,
    but the `frameclass` keyword is included for future expansions of the
    package.

    Parameters
    ----------
    data_dir : `str` or `pathlib.Path`
        The directory to search for appropriate files
    frameclass : `str`, optional
        Class of frame to collect for processing.  [Default: 'calibration']
    """

    def __init__(self, data_dir, frameclass="calibration"):

        # Check that `frameclass` is the currently supported
        if frameclass not in FRAMECLASSES:
            sa.send_alert("Incorrect frameclass specified", "Dumbwaiter.__init__()")
            return

        # Recheck that the `data_dir` is, in fact, okay, just to be sure
        if not check_directory_okay(data_dir, "Dumbwaiter.__init__()"):
            return

        # Initialize attributes
        self.frameclass = frameclass
        self.locations = utils.read_ligmos_conffiles("rozSetup")
        self.data_dir = pathlib.Path(data_dir).resolve()
        self.proc_dir = pathlib.Path(self.locations.processing_dir).resolve()
        self.instrument = divine_instrument(self.data_dir)
        # e.g., `lmi/20210107b` or `deveny/20220221a`
        self.nightname = os.sep.join(self.data_dir.parts[-2:])

        # If the directory is completely empty: send alert, set empty, return
        if not self.instrument:
            sa.send_alert(
                f"Directory {self.data_dir} is empty", "Dumbwaiter.__init__()"
            )
            self.empty = True
            return

        self.inst_flags = set_instrument_flags(self.instrument)

        # Based on the `frameclass`, call the appropriate `gather_*_frames()`
        if self.frameclass == "calibration":
            self.frames = gather_cal_frames(
                self.data_dir, self.inst_flags, fnames_only=True
            )
        else:
            sa.send_alert(
                f"Unsupported frameclass {self.frameclass}", "Dumbwaiter.__init__()"
            )

        # Make an attribute specifying whether the dumbwaiter is empty
        self.empty = not self.frames

    def copy_frames_to_processing(self, keep_existing=False):
        """copy_frames_to_processing Copy data frames to local processing dir

        This method copies the identified frames from the original data
        directory to a local processing directory.

        First, unless `keep_existing = True`, clear out any existing files in
        the processing directory (to keep from builing up cruft).  Then, copy
        over all of the data files in self.frames.

        Parameters
        ----------
        keep_existing : `bool`, optional
            Keep the existing files in the processing directory?
            [Default: False]
        """
        # If empty, dont' do anything
        if self.empty:
            return

        if not keep_existing:
            print("Clearing cruft from processing directory " f"{self.proc_dir}")
            for entry in os.scandir(self.proc_dir):
                if entry.is_file():
                    os.remove(self.proc_dir.joinpath(entry))

        print(f"Copying data from {self.data_dir} to {self.proc_dir} for processing...")
        # Show progress bar for copying files
        progress_bar = tqdm(
            total=len(self.frames), unit="file", unit_scale=False, colour="#2a52be"
        )
        for frame in self.frames:
            shutil.copy2(self.data_dir.joinpath(frame), self.proc_dir)
            progress_bar.update(1)
        progress_bar.close()

    def cold_storage(self, testing=True):
        """cold_storage Put the dumbwaited frames into cold strage

        This method takes the frames contained internally and packages them up
        for long-term cold storage.  The location of the cold storage (and any)
        associated login credentials) are contained in the self.locations
        attribute.

        The storage (compressed) tarball filename will consist of the
        instrument, UT date, and frameclass.

        Parameters
        ----------
        testing : `bool`, optional
            If testing, don't commit to cold storage  [Default: True]
        """
        # If empty, dont' do anything
        if self.empty:
            return

        # First, check to see if the UT Date is encoded in the source `data_dir`
        #  (8 consecutive digits) using regex negative lookbehind / lookahead
        #  but also includes the lowercase letter sub-night designation
        if result := re.search(r"(?<!\d)\d{8}[a-z](?!\d)", str(self.data_dir)):
            utdate = result.group(0)

        # Otherwise, grab the header of the LAST file in self.frames (as this
        #  is most likely to be taken AFTER 00:00UT) and extract from DATE-OBS
        else:
            utdate = (
                getheader(self.proc_dir.joinpath(self.frames[-1]))["DATE-OBS"]
                .split("T")[0]
                .replace("-", "")
            )

        # Build the tar filename
        tarbase = f"{self.instrument}_{utdate}_{self.frameclass}.tar.bz2"
        tarname = self.proc_dir.joinpath(tarbase)

        # Just return now
        if testing:
            return

        # Create a summary table to include in the tarball
        self._make_summary_table()

        # Tar up the files!
        print("Creating the compressed tar file for cold storage...")
        with tarfile.open(tarname, "w:bz2") as tar:
            tar.add(self.proc_dir.joinpath("README.txt"), arcname="README.txt")
            # Show progress bar for processing the tarball
            progress_bar = tqdm(
                total=len(self.frames), unit="file", unit_scale=False, colour="#00ff7f"
            )
            for name in self.frames:
                tar.add(self.proc_dir.joinpath(name), arcname=name)
                progress_bar.update(1)
            progress_bar.close()

        # Next, set up for copying the tarball over to cold storage
        # The requisite cold storage directories will be mounted locally and
        #  be of form ".../dataquality/{site}/{instrument}"
        cold_dir = pathlib.Path(self.locations.coldstorage_dir).joinpath(
            self.inst_flags["site"], self.instrument
        )
        if not cold_dir.is_dir():
            sa.send_alert(
                f"Woah!  No cold storage directory at {cold_dir} on `{sa.MACHINE}`",
                "Dumbwaiter.cold_storage()",
            )
            return
        print(f"Copying {tarbase} to {cold_dir}...")
        # NOTE: Using this lower-level function to avoid chmod() errors
        shutil.copyfile(tarname, cold_dir.joinpath(tarbase))

    def _make_summary_table(self, debug=False):
        """_make_summary_table Create and write to disk a summary table

        Add summary table to the tarball for future reference
        Tags: obserno, frametype, filter, binning, numamp, ampid
        """
        icl = ccdp.ImageFileCollection(location=self.proc_dir, filenames=self.frames)
        # Pull the subtable based on FITS header keywords
        summary = icl.summary[
            "obserno", "imagetyp", "filters", "ccdsum", "numamp", "ampid"
        ]
        # Convert those to the InfluxDB tags used with Roz
        summary.rename_columns(
            ["imagetyp", "filters", "ccdsum"], ["frametype", "filter", "binning"]
        )
        # Write it out!
        summary.write(self.proc_dir.joinpath("README.txt"), format="ascii.fixed_width")
        if debug:
            summary.pprint()


# Non-Class Functions ========================================================#
def check_directory_okay(directory, caller=None):
    """check_directory_okay Check `directory` is okay to proceed

    Check that directory is, indeed, a directory and contains FITS files

    NOTE: This is the first function called, and should have robust alerting if
          things aren't up to snuff!

    Parameters
    ----------
    directory : `str` or `Pathlib.path`
        The directory to check
    caller : `str`, optional
        The name of the calling function, to be printed in the alert  [Default: None]

    Returns
    -------
    `bool`
        True if OK, False otherwise
    """
    # Check that `directory` is, in fact, a directory
    if not os.path.isdir(directory):
        sa.send_alert(
            f"Directory Issue: {utils.subpath(directory)} is not a valid directory",
            caller,
        )
        return False

    # Get the list of normal FITS files in the directory
    fits_files = get_sequential_fitsfiles(directory)

    # Check if there's anything useful
    if not fits_files:
        sa.send_alert(
            f"Empty Directory: `{utils.subpath(directory)}` does not contain "
            "any sequential FITS files",
            caller,
        )
        return False

    # If we get here, we're clear to proceed!
    return True


def divine_instrument(directory):
    """divine_instrument Divine the instrument whose data is in this directory

    This function emulates Carnac the Magnificent, where it holds a sealed
    envelope (FITS header) to its forehead and divines the answer to the
    question contained inside (what is the instrument?).  Finally, it rips
    open the envelope, and reads the index card (INSTRUME keyword) inside.

    NOTE: For proper functioning, the FITS headers must be kept in a
          mayonnaise jar on Funk and Wagnalls' porch since noon UT.

    TODO: As we bring the Anderson Mesa instruments into Roz, this function
          may need significant overhaul.

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        The directory for which to divine the instrument

    Returns
    -------
    `str`
        Lowercase string of the contents of the FITS `INSTRUME` keyword
    """
    # Get the list of normal FITS files in the directory
    fits_files = get_sequential_fitsfiles(directory)

    # Loop through the files, looking for a valid INSTRUME keyword
    for fitsfile in fits_files:
        try:
            # If we're good to go...
            if fitsfile.is_file():
                return getheader(fitsfile)["instrume"].lower()
        except KeyError:
            continue
    # Otherwise...
    sa.send_alert(
        f"No Instrument found in {utils.subpath(directory)}", "divine_instrument()"
    )
    return None


def gather_cal_frames(directory, inst_flag, fnames_only=False):
    """gather_cal_frames Gather calibration frames from specified directory

    [extended_summary]

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        Directory name to search for calibration files
    inst_flag : `dict`
        Dictionary of instrument flags
    fnames_only : `bool`, optional
        Only return a concatenated list of filenames instead of the IFCs
        [Default: False]

    Returns
    -------
    return_object : `dict`
        Dictionary containing the various filename lists, ImageFileCollections,
        and/or binning lists, as specified by `inst_flag`.
    -- OR --
    fnames : `list`
        List of calibration filenames (returned when `fnames_only = True`)
    """
    # Silence the AstropyUserWarning from CCDPROC
    warnings.simplefilter("ignore", AstropyUserWarning)

    # Because over-the-network reads can take a while, say something!
    print(f"Reading the files in {directory}...")

    # Create an ImageFileCollection for the specified directory
    icl = ccdp.ImageFileCollection(
        location=directory,
        glob_include=f"{inst_flag['prefix']}*.fits",
        glob_exclude="test.fits",
    )

    if not icl.files:
        print("There ain't nothin' here that meets my needs!")
        sa.send_alert(
            f"Empty Directory: No matching files in {utils.subpath(directory)}",
            "gather_cal_frames()",
        )
        return None

    return_object = {}

    # Keep these items separate for now, in case future instruments need one
    #  but not the others
    if inst_flag["get_bias"]:
        # Gather any bias frames (OBSTYPE=`bias` or EXPTIME=0) FULL FRAME ONLY
        bias_fns = icl.files_filtered(obstype="bias", subarrno=0)
        zero_fns = icl.files_filtered(exptime=0, subarrno=0)
        biases = list(np.unique(np.concatenate([bias_fns, zero_fns])))
        # NOTE: We sometimes get weird IFC cant' find file warnings with this line:
        bias_cl = ccdp.ImageFileCollection(location=directory, filenames=biases)
        return_object["bias_fn"] = bias_cl.files
        return_object["bias_cl"] = bias_cl

    if inst_flag["get_dark"]:
        # Gather DARK frames; FULL FRAME ONLY
        dark_cl = icl.filter(obstype="dark", subarrno=0)
        return_object["dark_fn"] = dark_cl.files
        return_object["dark_cl"] = dark_cl

    if inst_flag["get_flat"]:
        # Gather DOME FLAT frames; FULL FRAME ONLY
        domeflat_cl = icl.filter(obstype="dome flat", subarrno=0)
        return_object["domeflat_fn"] = domeflat_cl.files
        return_object["domeflat_cl"] = domeflat_cl
        # TODO: SKY FLATs returned separately -- will need to deal with them elswehere
        skyflat_cl = icl.filter(obstype="sky flat", subarrno=0)
        return_object["skyflat_fn"] = skyflat_cl.files
        return_object["skyflat_cl"] = skyflat_cl

    if inst_flag["check_bin"]:
        # Get the complete list of binnings used -- but clear out `None` entries
        bin_list = icl.values("ccdsum", unique=True)
        bin_list = sorted(list(filter(None, bin_list)))
        return_object["bin_list"] = bin_list

    # ===============================================================#
    # If we only want the filenames, flatten out the fn lists and return
    if fnames_only:
        # Append all the filename lists onto `fn_list`
        fn_list = []
        print(f"{'*'*19}\n* -Frame Summary- *")
        for key, val in return_object.items():
            if key.find("_fn") != -1:
                print(f"* {key.split('_')[0].upper():10s}: {len(val):3d} *")
                fn_list.append(val)
        print("*" * 19)
        # Flatten and return the BASENAME only
        return [os.path.basename(fn) for fn in list(np.concatenate(fn_list).flat)]

    # Otherwise, return the accumulated dictionary
    return return_object


def gather_other_frames():
    """gather_other_frames Stub for additional functionality

    [extended_summary]
    """


def get_sequential_fitsfiles(directory):
    """get_sequential_fitsfiles Get the sequential FITS files in a directory

    Since we do this several times, pull it out to a separate function.  This
    function returns a list of the non-test (assumed sequential) FITS files in
    `directory`.

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        Directory name to search for FITS files

    Returns
    -------
    `list`
        List of the non-test (i.e. sequential) FITS files in `directory`
    """
    # Make sure directory is a pathlib.Path
    if isinstance(directory, str):
        directory = pathlib.Path(directory)

    # Get a sorted list of all the FITS files
    fits_files = sorted(directory.glob("*.fits"))

    # Remove `test.fits` because we just don't care about it.
    try:
        fits_files.remove(directory.joinpath("test.fits"))
    except ValueError:
        pass

    return fits_files


def set_instrument_flags(inst):
    """set_instrument_flags Set the global instrument flags for processing

    These instrument-specific flags are used throughout the code.  As more
    instruments are added to Roz, this function will grow commensurately.

    Alternatively, this information could be placed in an XML VOTABLE that
    could simply be read in -- to eliminiate one more hard-coded thing.

    Parameters
    ----------
    instrument : `str`
        Name of the instrument to use

    Returns
    -------
    `dict`
        Dictionary of instrument flags.
    """
    # Read in the instrument flag table
    instrument_table = Table.read(utils.Paths.data.joinpath("instrument_flags.ecsv"))

    # Check that the instrument is in the table
    if (inst := inst.upper()) not in instrument_table["instrument"]:
        sa.send_alert(
            f"Instrument {inst} not yet supported; update instrument_flags.ecsv",
            "set_instrument_flags()",
        )
        return None

    # Extract the row, and convert it to a dictionary
    for row in instrument_table:
        if row["instrument"] == inst:
            return dict(zip(row.colnames, row))

    raise utils.DeveloperWarning("Error: this line should never run.")
