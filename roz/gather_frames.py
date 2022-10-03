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
specified in :obj:`~roz.config`.

This module primarily trades in CCDPROC Image File Collections
(`ccdproc.ImageFileCollection`_).

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

# Built-In Libraries
import os
import pathlib
import re
import shutil
import tarfile
import warnings

# 3rd Party Libraries
import astropy.io.fits
from astropy.utils.exceptions import AstropyUserWarning
import ccdproc
import numpy as np
from tqdm import tqdm

# Internal Imports
from roz import alerting
from roz import msgs
from roz import utils

# Set API Components
__all__ = [
    "Dumbwaiter",
    "gather_calibration_frames",
    "gather_science_frames",
    "gather_allsky_frames",
    "FRAMECLASSES",
]

# Currently Supported Frameclasses
FRAMECLASSES = ["calibration", "science", "allsky"]

# Silence the AstropyUserWarning from CCDPROC
warnings.simplefilter("ignore", AstropyUserWarning)


class Dumbwaiter:
    """Class for moving data between floors (servers)

    This class contains all of the data and methods related to identifying the
    frames associated with a given frameclass, copying them to a processing
    location, and packaging them for cold storage.

    Roz does not need LIG data butlers for data movement, as it does everything
    itself.  Therefore, the concept of a dumbwaiter seemed appropriate.

    .. note::

        The Dumbwaiter ingests all information about the specified data
        directory, and creates internal lists of frames for each of the
        ``FRAMECLASSES`` that Roz currently knows about.  Each list is used
        by :func:`~roz.main_driver.main` to process the frameclass
        appropriately.

    Parameters
    ----------
    data_dir : str or :obj:`pathlib.Path`:
        The directory to search for appropriate files
    proc_args : dict
        Dictionary of processing arguments from :func:`~rox.main_driver.main`
        to be cross-referenced with the instrument flags to determine the
        processing behavior.
    """

    def __init__(self, data_dir, proc_args):

        # Initialize locations
        locations = utils.read_ligmos_conffiles("rozSetup")
        self.dirs = {
            "data": pathlib.Path(data_dir).resolve(),
            "proc": pathlib.Path(locations.processing_dir).resolve(),
            "cold": pathlib.Path(locations.coldstorage_dir).resolve(),
        }

        # Divine the instrument and set flags
        fitsfiles = self.get_sequential_fitsfiles(self.dirs["data"])
        self.instrument = self.divine_instrument(fits_files=fitsfiles)
        self.flags = self.set_instrument_flags(self.instrument)

        # Set `process_frameclass` attribute as a list if all 3 clauses are True
        self.process_frameclass = [
            fclass
            for fclass in FRAMECLASSES
            if (
                proc_args[fclass]
                and self.flags[f"has_{fclass}"]
                and self.flags[f"proc_{fclass}"]
            )
        ]

        # If no FITS files, no instrument, or nothing to process, set "empty"
        if not all([fitsfiles, self.instrument, self.process_frameclass]):
            self._empty = {fclass: True for fclass in FRAMECLASSES}
            return

        # Set nightname (e.g., `lmi/20210107b` or `deveny/20220221a`)
        self.nightname = os.sep.join(self.dirs["data"].parts[-2:])

        # Based on the `frameclass`, call the appropriate `gather_*_frames()`
        # NOTE: self.frames is a dictionary of frame lists by frame class
        #       If not processing a frameclass, use an empty list
        self.frames = {
            fclass: globals()[f"gather_{fclass}_frames"](
                self.dirs["data"],
                self.flags,
                fitsfiles=fitsfiles,
                fnames_only=True,
            )
            if fclass in self.process_frameclass
            else []
            for fclass in FRAMECLASSES
        }

        # Make a dictionary specifying whether the dumbwaiter is empty of fclass
        self._empty = {fclass: not self.frames[fclass] for fclass in FRAMECLASSES}

    def empty(self, frameclass):
        """Is the Dumbwaiter empty of this frameclass?

        Parameters
        ----------
        frameclass : str
            The frameclass, from ``FRAMECLASSES``

        Returns
        -------
        bool
            Is this frameclass empty?
        """
        return self._empty[frameclass]

    def serve_frames(self, frameclass, keep_existing=False):
        """Copy data frames to a local processing dir

        This method copies the identified frames from the original data
        directory to a local processing directory.

        First, unless ``keep_existing = True``, clear out any existing files in
        the processing directory (to keep from builing up cruft).  Then, copy
        over all of the data files in ``self.frames[frameclass]``.

        Parameters
        ----------
        frameclass : str
            The frameclass, from ``FRAMECLASSES``
        keep_existing : bool, optional
            Keep the existing files in the processing directory?
            (Default: False)
        """
        # If empty, dont' do anything
        if self.empty(frameclass):
            return

        if not keep_existing:
            msgs.info(f"Clearing cruft from processing directory {self.dirs['proc']}")
            for entry in self.dirs["proc"].glob("*"):
                if entry.is_file():
                    self.dirs["proc"].joinpath(entry).unlink()

        msgs.info(
            f"Copying data from {self.dirs['data']} to {self.dirs['proc']} for processing..."
        )
        # Show progress bar for copying files (COLOR = "Cherry Red")
        progress_bar = tqdm(
            total=len(self.frames[frameclass]),
            unit="file",
            unit_scale=False,
            colour="#D2042D",
        )
        for frame in self.frames[frameclass]:
            try:
                shutil.copy2(self.dirs["data"].joinpath(frame), self.dirs["proc"])
            except FileNotFoundError as err:
                msgs.error(
                    f"Expected file was not found in the data directory:"
                    f"{self.dirs['data'].joinpath(frame)}.{msgs.newline()}"
                    f"{err}"
                )
            progress_bar.update(1)
        progress_bar.close()

    def cold_storage(self, frameclass, skip_cold=False, **kwargs):
        """Put the dumbwaited frames into cold strage

        This method takes the frames contained internally and packages them up
        for long-term cold storage.  The location of the cold storage (and any
        associated login credentials) are contained in the ``self.locations``
        attribute.

        The storage (compressed) tarball filename will consist of the
        instrument, UT date, and frameclass.

        Parameters
        ----------
        frameclass : str
            The frameclass, from ``FRAMECLASSES``
        skip_cold : bool, optional
            Don't commit to cold storage  (Default: False)
        """
        # If empty, dont' do anything
        if self.empty(frameclass):
            return

        # First, check to see if the UT Date is encoded in the source `data_dir`
        #  (8 consecutive digits) using regex negative lookbehind / lookahead
        #  but also includes the lowercase letter sub-night designation
        if result := re.search(r"(?<!\d)\d{8}[a-z](?!\d)", str(self.dirs["data"])):
            utdate = result.group(0)

        # Otherwise, grab the header of the LAST file in this self.frames (as this
        #  is most likely to be taken AFTER 00:00UT) and extract from DATE-OBS
        else:
            utdate = (
                astropy.io.fits.getheader(
                    self.dirs["proc"].joinpath(self.frames[frameclass][-1])
                )["DATE-OBS"]
                .split("T")[0]
                .replace("-", "")
            )

        # Build the tar filename
        tarbase = f"{self.instrument}_{utdate}_{frameclass}.tar.bz2"
        tarname = self.dirs["proc"].joinpath(tarbase)

        # Just return now, if commanded
        if skip_cold:
            return

        # Create a summary table (README.txt) to include in the tarball
        self._make_summary_table(frameclass)

        # Tar up the files!
        msgs.info("Creating the compressed tar file for cold storage...")
        with tarfile.open(tarname, "w:bz2") as tar:
            tar.add(self.dirs["proc"].joinpath("README.txt"), arcname="README.txt")
            # Show progress bar for processing the tarball (COLOR = "Forest Green")
            progress_bar = tqdm(
                total=len(self.frames[frameclass]),
                unit="file",
                unit_scale=False,
                colour="#228B22",
            )
            for name in self.frames[frameclass]:
                tar.add(self.dirs["proc"].joinpath(name), arcname=name)
                progress_bar.update(1)
            progress_bar.close()

        # Next, set up for copying the tarball over to cold storage
        # The requisite cold storage directories will be mounted locally and
        #  be of form ".../dataquality/{site}/{instrument}"
        cold_dir = self.dirs["cold"].joinpath(self.flags["site"], self.instrument)
        if not cold_dir.is_dir():
            alerting.send_alert("dir_not_found", dirname=cold_dir, **kwargs)
            return
        msgs.info(f"Copying {tarbase} to {cold_dir}...")
        # NOTE: Using this lower-level function to avoid chmod() errors
        shutil.copyfile(tarname, cold_dir.joinpath(tarbase))

    @staticmethod
    def divine_instrument(directory=None, fits_files=None):
        """Divine the instrument whose data is in this directory

        This function emulates Carnac the Magnificent, where it holds a sealed
        envelope (FITS header) to its forehead and divines the answer to the
        question contained inside (what is the instrument?).  Finally, it rips
        open the envelope, and reads the index card (INSTRUME keyword) inside.

        .. note::

            For proper functioning, the FITS headers must be kept in a
            mayonnaise jar on Funk and Wagnalls' porch since noon UT.

        .. note::

            As we bring the Anderson Mesa instruments into Roz, this function
            may need significant overhaul.

        Parameters
        ----------
        directory : str or :obj:`pathlib.Path`, optional
            The directory for which to divine the instrument
        fits_files : list, optional
            The list of FITS files from which to divine the instrument
            This parameter has priority over ``directory``

        Returns
        -------
        str
            Lowercase string of the contents of the FITS ``INSTRUME`` keyword
        """
        if not any([directory, fits_files]):
            msgs.error("Either `directory` or `fits_files` is required.")

        if directory and not fits_files:
            # Get the list of normal FITS files in the directory
            fits_files = Dumbwaiter.get_sequential_fitsfiles(directory)

        # Loop through the files, looking for a valid INSTRUME keyword
        for fitsfile in fits_files:
            try:
                # If we're good to go...
                if fitsfile.is_file():
                    return astropy.io.fits.getheader(fitsfile)["instrume"].lower()
            except KeyError:
                continue

        # Otherwise...
        alerting.send_alert(
            "no_inst_found",
            dirname=utils.subpath(directory) if directory else "this directory",
        )
        return None

    @staticmethod
    def set_instrument_flags(inst):
        """Set the global instrument flags for processing

        These instrument-specific flags are used throughout the code.  As more
        instruments are added to Roz, this function will grow commensurately.

        Alternatively, this information could be placed in an XML VOTABLE that
        could simply be read in -- to eliminiate one more hard-coded thing.

        Parameters
        ----------
        instrument : str
            Name of the instrument to use

        Returns
        -------
        dict
            Dictionary of instrument flags.
        """
        # Read in the instrument flag table
        instrument_table = utils.read_instrument_table()

        # Check that the instrument is in the table
        if (inst := inst.upper()) not in instrument_table["instrument"]:
            alerting.send_alert("inst_not_support", inst=inst)
            return None

        # Extract the row, and convert it to a dictionary
        for row in instrument_table:
            if row["instrument"] == inst:
                return dict(zip(row.colnames, row))

        msgs.error("Error: this line should never run.")

    def _make_summary_table(self, frameclass, debug=False):
        """Create and write to disk a summary table

        Add summary table to the tarball for future reference
        Tags: obserno, frametype, filter, binning, numamp, ampid

        Parameters
        ----------
        frameclass : str
            The frameclass, from ``FRAMECLASSES``
        debug : bool, optional
            ``.pprint()`` the Table for debug?  (Default: False)
        """
        # List of possible keywords to include in the summary table:
        keywords = [
            "obserno",
            "seqnum",
            "imagetyp",
            "imgtype",
            "filters",
            "ccdsum",
            "numamp",
            "creator",
        ]
        # Column rename dictionary:
        renames = {
            "imagetyp": "frametype",
            "imgtype": "frametype",
            "filters": "filter",
            "ccdsum": "binning",
            "seqnum": "obserno",
        }

        # If the FITS files are compressed, emit a warning about slowness
        exts = sorted(list({pathlib.Path(fn).suffix for fn in self.frames[frameclass]}))
        if ".gz" in exts or ".bz2" in exts:
            msgs.warn("The FITS files in this directory are compressed; be patient.")

        # Load in the Image File Collection
        icl = ccdproc.ImageFileCollection(
            location=self.dirs["proc"], filenames=self.frames[frameclass]
        )

        # Pull the README subtable based on FITS header keywords present
        readme = icl.summary[list(set(icl.summary.colnames) & set(keywords))]

        if "ampid" in icl.summary.colnames:
            # For single-amplifier readouts
            readme["ampid"] = icl.summary["ampid"]
        elif readme["creator"][0] == "LOIS":
            # For LOIS multi-amplifier readouts
            readme["ampid"] = [utils.parse_lois_ampids(hdr) for hdr in icl.headers()]

        # Convert column names to the InfluxDB tags used with Roz
        for oldname, newname in renames.items():
            if oldname in readme.colnames:
                readme.rename_column(oldname, newname)

        # Write it out!
        readme.write(
            self.dirs["proc"].joinpath("README.txt"), format="ascii.fixed_width"
        )
        if debug:
            readme.pprint()

    @staticmethod
    def get_sequential_fitsfiles(directory, prefix=""):
        """Get the sequential FITS files in a directory

        Since we do this several times, pull it out to a separate function.  This
        function returns a list of the non-test (assumed sequential) FITS files in
        ``directory``.

        Parameters
        ----------
        directory : :obj:`str` or :obj:`pathlib.Path`
            Directory name to search for FITS files
        prefix : str, optional
            The file prefix from ``instrument_flags.ecsv``  (Default: ``""``)

        Returns
        -------
        list
            List of the sequentially numbered FITS files in ``directory``
        """
        # Make sure directory is a pathlib.Path
        if isinstance(directory, str):
            directory = pathlib.Path(directory)

        # Check that `directory` is, in fact, a directory
        if not directory.is_dir():
            alerting.send_alert("dir_not_found", dirname=utils.subpath(directory))
            return None

        # Get a sorted list of all the FITS files (inluding compressed formats)
        fits_files = []
        for file_ext in ["fits", "fit", "fits.gz", "fit.gz", "fits.bz2", "fit.bz2"]:
            fits_files.extend(list(directory.glob(f"{prefix}*.{file_ext}")))
        fits_files = sorted(fits_files)

        # Remove `test.fits` because we just don't care about it.
        return [
            file
            for file in fits_files
            if file.name not in ["test.fits", "final.fit.bz2"]
        ]


# Non-Class Functions ========================================================#
def gather_calibration_frames(directory, inst_flag, fitsfiles=None, fnames_only=False):
    """Gather calibration frames from the specified ``directory``

    [extended_summary]

    Parameters
    ----------
    directory : :obj:`str` or :obj:`pathlib.Path`
        Directory name to search for calibration files
    inst_flag : dict
        Dictionary of instrument flags
    fitsfiles : list, optional
        The list of FITS files in this directory  (Default: None)
    fnames_only : bool, optional
        Only return a concatenated list of filenames instead of the IFCs
        (Default: False)

    Returns
    -------
    return_object : dict
        Dictionary containing the various filename lists, ImageFileCollections,
        and/or binning lists, as specified by ``inst_flag``.
    -- OR --
    fnames : list
        List of calibration filenames (returned when ``fnames_only = True``)
    """
    # Because over-the-network reads can take a while, say something!
    msgs.info(f"Reading the files in {directory}...")

    # Create an ImageFileCollection for the specified directory
    if not fitsfiles:
        fitsfiles = Dumbwaiter.get_sequential_fitsfiles(directory, inst_flag["prefix"])
    icl = ccdproc.ImageFileCollection(location=directory, filenames=fitsfiles)

    if not icl.files:
        return None

    return_object = {}

    # Process each instrument flag separately:
    if inst_flag["get_bias"]:
        # Gather any bias frames (OBSTYPE=`bias` or EXPTIME=0) FULL FRAME ONLY
        fn_list = []
        fn_list.append(icl.files_filtered(obstype="bias", subarrno=0))
        fn_list.append(icl.files_filtered(exptime=0, subarrno=0))
        fn_list = list(np.unique(np.concatenate(fn_list)))
        # Do this `location` thing to work around how IFC deals with empty lists
        bias_cl = ccdproc.ImageFileCollection(
            location=directory if fn_list else None, filenames=fn_list
        )
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
        # Gather SKY FLAT frames; FULL FRAME ONLY
        skyflat_cl = icl.filter(obstype="sky flat", subarrno=0)
        return_object["skyflat_fn"] = skyflat_cl.files
        return_object["skyflat_cl"] = skyflat_cl

    if inst_flag["check_bin"]:
        # Get the complete list of binnings used -- but clear out `None` entries
        bin_list = icl.values("ccdsum", unique=True)
        bin_list = sorted(list(filter(None, bin_list)))
        return_object["bin_list"] = bin_list

    if inst_flag["check_amp"]:
        # Get the complete list of amplifier configurations
        amp_configs = [utils.parse_lois_ampids(hdr) for hdr in icl.headers()]
        # Return a sorted list of the unique configurations
        return_object["amp_config"] = sorted(list(set(amp_configs)))

    # Concatenate the filename list, as we'll need it regardless
    all_fns = [
        os.path.basename(fn)
        for fn in list(
            np.concatenate(
                [val for key, val in return_object.items() if "_fn" in key]
            ).flat
        )
    ]

    # ===============================================================#
    # If we only want the filenames, print a summary and return the names
    if fnames_only:
        # Print out Frame Summary
        msgs.table("*" * 19)
        msgs.table("* -Frame Summary- *")
        for key, val in return_object.items():
            if "_fn" in key:
                msgs.table(f"* {key.split('_')[0].upper():10s}: {len(val):3d} *")
        msgs.table("*" * 19)
        # Flatten and return the BASENAME only
        return all_fns

    # Otherwise, create a combined IFC, and return the accumulated dictionary
    return_object["calibration_cl"] = ccdproc.ImageFileCollection(
        location=directory if all_fns else None, filenames=all_fns
    )
    return return_object


def gather_science_frames(directory, inst_flag, fitsfiles=None, fnames_only=False):
    """Gather sciencd frames from the specified ``directory``

    [extended_summary]

    Parameters
    ----------
    directory : :obj:`str` or :obj:`pathlib.Path`
        Directory name to search for calibration files
    inst_flag : dict
        Dictionary of instrument flags
    fitsfiles : list, optional
        The list of FITS files in this directory  (Default: None)
    fnames_only : bool, optional
        Only return a concatenated list of filenames instead of the IFCs
        (Default: False)

    Returns
    -------
    return_object : dict
        Dictionary containing the various filename lists, ImageFileCollections,
        and/or binning lists, as specified by ``inst_flag``.
    -- OR --
    fnames : list
        List of calibration filenames (returned when ``fnames_only = True``)
    """
    # Because over-the-network reads can take a while, say something!
    msgs.info(f"Reading the files in {directory}...")

    # Create an ImageFileCollection for the specified directory
    if not fitsfiles:
        fitsfiles = Dumbwaiter.get_sequential_fitsfiles(directory, inst_flag["prefix"])
    icl = ccdproc.ImageFileCollection(location=directory, filenames=fitsfiles)

    if not icl.files:
        return None

    return_object = {}

    # Gather OBJECT frames; FULL FRAME ONLY
    object_cl = icl.filter(obstype="object", subarrno=0)
    return_object["object_fn"] = object_cl.files
    return_object["object_cl"] = object_cl

    if inst_flag["check_bin"]:
        # Get the complete list of binnings used -- but clear out `None` entries
        bin_list = icl.values("ccdsum", unique=True)
        bin_list = sorted(list(filter(None, bin_list)))
        return_object["bin_list"] = bin_list

    if inst_flag["check_amp"]:
        # Get the complete list of amplifier configurations
        amp_configs = [utils.parse_lois_ampids(hdr) for hdr in icl.headers()]
        # Return a sorted list of the unique configurations
        return_object["amp_config"] = sorted(list(set(amp_configs)))

    # ===============================================================#
    # If we only want the filenames, print a summary and return the names
    if fnames_only:
        # Print out Frame Summary
        msgs.table("*" * 19)
        msgs.table("* -Frame Summary- *")
        for key, val in return_object.items():
            if "_fn" in key:
                msgs.table(f"* {key.split('_')[0].upper():10s}: {len(val):3d} *")
        msgs.table("*" * 19)
        # Flatten and return the BASENAME only
        return return_object["object_fn"]

    # Otherwise, create a combined IFC, and return the accumulated dictionary
    return_object["science_cl"] = ccdproc.ImageFileCollection(
        location=directory if return_object["object_fn"] else None,
        filenames=return_object["object_fn"],
    )
    return return_object


def gather_allsky_frames(directory, inst_flag, fitsfiles=None, fnames_only=False):
    """Gather allsky frames from the specified ``directory``

    .. note::

        This is identical to :func:`gather_science_frames` at the moment, but
        could potentially have different requirements as things evolve.

    Parameters
    ----------
    directory : :obj:`str` or :obj:`pathlib.Path`
        Directory name to search for calibration files
    inst_flag : dict
        Dictionary of instrument flags
    fitsfiles : list, optional
        The list of FITS files in this directory  (Default: None)
    fnames_only : bool, optional
        Only return a concatenated list of filenames instead of the IFCs
        (Default: False)

    Returns
    -------
    return_object : dict
        Dictionary containing the various filename lists, ImageFileCollections,
        and/or binning lists, as specified by ``inst_flag``.
    -- OR --
    fnames : list
        List of calibration filenames (returned when ``fnames_only = True``)
    """
    # Because over-the-network reads can take a while, say something!
    msgs.info(f"Reading the files in {directory}...")

    # Create an ImageFileCollection for the specified directory
    if not fitsfiles:
        fitsfiles = Dumbwaiter.get_sequential_fitsfiles(directory, inst_flag["prefix"])

    # If the FITS files are compressed, emit a warning about slowness
    exts = sorted(list({pathlib.Path(fn).suffix for fn in fitsfiles}))
    if ".gz" in exts or ".bz2" in exts:
        msgs.warn("The FITS files in this directory are compressed; be patient.")

    icl = ccdproc.ImageFileCollection(location=directory, filenames=fitsfiles)

    if not icl.files:
        return None

    return_object = {}

    # Gather OBJECT frames
    object_cl = icl.filter(imgtype="object")
    return_object["object_fn"] = object_cl.files
    return_object["object_cl"] = object_cl

    if inst_flag["check_bin"]:
        # Get the complete list of binnings used -- but clear out `None` entries
        bin_list = icl.values("ccdsum", unique=True)
        bin_list = sorted(list(filter(None, bin_list)))
        return_object["bin_list"] = bin_list

    if inst_flag["check_amp"]:
        # Get the complete list of amplifier configurations
        amp_configs = [utils.parse_lois_ampids(hdr) for hdr in icl.headers()]
        # Return a sorted list of the unique configurations
        return_object["amp_config"] = sorted(list(set(amp_configs)))

    # ===============================================================#
    # If we only want the filenames, print a summary and return the names
    if fnames_only:
        # Print out Frame Summary
        msgs.table("*" * 19)
        msgs.table("* -Frame Summary- *")
        for key, val in return_object.items():
            if "_fn" in key:
                msgs.table(f"* {key.split('_')[0].upper():10s}: {len(val):3d} *")
        msgs.table("*" * 19)
        # Flatten and return the BASENAME only
        return return_object["object_fn"]

    # Otherwise, create a combined IFC, and return the accumulated dictionary
    return_object["science_cl"] = ccdproc.ImageFileCollection(
        location=directory if return_object["object_fn"] else None,
        filenames=return_object["object_fn"],
    )
    return return_object
