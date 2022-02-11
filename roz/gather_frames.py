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

This module will need to interface at some level with Wadsworth (the LIG Data
Butler) to either buttle data to whatever machine is running Roz, or to let Roz
know that data has been buttled to the proper location (laforge?) and provide
the proper directory information.

This module primarily trades in CCDPROC Image File Collections
(`ccdproc.ImageFileCollection`), along with whatever data structures are
sent to or recieved by Wadsworth.
"""

# Built-In Libraries
import glob
import os
from pathlib import Path
import re
import shutil
import tarfile
import warnings

# 3rd Party Libraries
from astropy.io.fits import getheader
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
import numpy as np
from tqdm import tqdm

# Internal Imports
from roz import send_alerts as sa
from roz import utils


# Create an error class to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


class Dumbwaiter():
    """dumbwaiter Class for moving data between floors (servers)

    It seemed easier to contain in one place all of the data and methods related to
    identifying the appropriate frames, copying them to a processing
    location, and packaging them for cold storage.
    """

    def __init__(self, data_dir, frameclass='calibration'):
        """__init__ Initialize the Dumbwaiter class

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
        # Check that the (presumably remote) directory is, in fact, a directory
        if not os.path.isdir(data_dir):
            sa.send_alert('BadDirectoryAlert : Dumbwaiter.__init__()')
            return

        # Initialize attributes
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.frameclass = frameclass
        self.locations = utils.read_ligmos_conffiles('rozSetup')
        self.proc_dir = Path(self.locations.processing_dir)
        self.instrument = divine_instrument(self.data_dir)

        # If the directory is completely empty: send alert, set empty, return
        if not self.instrument:
            sa.send_alert('EmptyDirectoryAlert : Dumbwaiter.__init__()')
            self.empty = True
            return

        self.inst_flags = utils.set_instrument_flags(self.instrument)

        # Based on the `frameclass`, call the appropriate `gather_*_frames()`
        if self.frameclass == 'calibration':
            self.frames = gather_cal_frames(self.data_dir,
                                            self.inst_flags,
                                            fnames_only=True)
        else:
            sa.send_alert('BadFrameclassAlert : Dumbwaiter.__init__()')

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
            print("Cleaning out previous cruft in processing directory "
                  f"{self.proc_dir}")
            for entry in os.scandir(self.proc_dir):
                if entry.is_file():
                    os.remove(self.proc_dir.joinpath(entry))

        print(f"Copying data from {self.data_dir} to {self.proc_dir} "
              "for processing...")
        # Show progress bar for copying files
        progress_bar = tqdm(total=len(self.frames), unit='file',
                            unit_scale=False, colour='cyan')
        for frame in self.frames:
            shutil.copy(self.data_dir.joinpath(frame), self.proc_dir)
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
        if result := re.search(r'(?<!\d)\d{8}(?!\d)', str(self.data_dir)):
            utdate = result.group(0)

        # Otherwise, grab the header of the LAST file in self.frames (as this
        #  is most likely to be taken AFTER 00:00UT) and extract from DATE-OBS
        else:
            utdate = getheader(self.proc_dir.joinpath(
                self.frames[-1]))['DATE-OBS'].split('T')[0].replace('-','')

        # Build the tar filename
        tarbase = f"{self.instrument}_{utdate}_{self.frameclass}.tar.bz2"
        tarname = self.proc_dir.joinpath(tarbase)

        # Just return now
        if testing:
            return

        # Tar up the files!
        print("Creating the compressed tar file for cold storage...")
        with tarfile.open(tarname, "w:bz2") as tar:
            # Show progress bar for processing the tarball
            progress_bar = tqdm(total=len(self.frames), unit='file',
                                unit_scale=False, colour='green')
            for name in self.frames:
                tar.add(self.proc_dir.joinpath(name))
                progress_bar.update(1)
            progress_bar.close()


        # Next, set up for copying the tarball over to cold storage
        # TODO: Need to confer with Ryan about how this step will be done.
        #       For instance, will the storage directories be mounted on the
        #       processing machine, or will the files be copied over via scp
        #       or similar protocol?


# Non-Class Functions ========================================================#
def divine_instrument(directory):
    """divine_instrument Divine the instrument whose data is in this directory

    Opens one of the FITS files and reads in the INSTRUME header keyword,
    returns as a lowercase string.

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        The directory for which to divine the instrument

    Returns
    -------
    `str`
        Lowercase string of the contents of the FITS `INSTRUME` keyword
    """
    # Get a sorted list of all the FITS files
    fitsfiles = sorted(glob.glob(f"{directory}/*.fits"))

    # Loop through the files, looking for a valid INSTRUME keyword
    for fitsfile in fitsfiles:
        # print(f"Attempting to find INSTRUME in {fitsfile}")
        try:
            # If we're good to go...
            if os.path.isfile(fitsfile):
                header = getheader(fitsfile)
                return header['instrume'].lower()
        except KeyError:
            continue
    # Otherwise...
    sa.send_alert('BadDirectoryAlert : divine_instrument()')
    return None


def gather_cal_frames(directory, inst_flag, fnames_only=False):
    """gather_cal_frames Gather calibration frames from specified directory

    [extended_summary]

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        Directory name to search for calibration files
    instrument : `str`, optional
        Name of the instrument to gather calibration frames for [Default: LMI]
    fnames_only : `bool`, optional
        Only return a concatenated list of filenames instead of the IFCs
        [Default: False]

    Returns
    -------
    bias_cl : `ccdproc.ImageFileCollection`
        ImageFileColleciton containing the BIAS frames from the directory
    domeflat_cl : `ccdproc.ImageFileCollection`, optional (LMI only)
        ImageFileCollection containing the FLAT frames from the directory
    bin_list : `list`, optional (LMI only)
        List of binning setups found in this directory
    -- OR --
    fnames, `list`
        List of calibration filenames (returned when `fnames_only = True`)
    """
    # Silence Superflous AstroPy Warnings from CCDPROC routines
    warnings.simplefilter('ignore', AstropyWarning)

    # Because over-the-network reads can take a while, say something!
    print(f"Reading the files in {directory}...")

    # Create an ImageFileCollection for the specified directory
    icl = ccdp.ImageFileCollection(
            directory, glob_include=f"{inst_flag['prefix']}*.fits")

    if not icl.files:
        print("There ain't nothin' here that meets my needs!")
        sa.send_alert('EmptyDirectoryAlert : gather_cal_frames()')
        return None

    return_object = []

    # Keep these items separate for now, in case future instruments need one
    #  but not the others
    if inst_flag['get_bias']:
        # Gather any bias frames (OBSTYPE=`bias` or EXPTIME=0) FULL FRAME ONLY
        bias_fns = icl.files_filtered(obstype='bias', subarrno=0)
        zero_fns = icl.files_filtered(exptime=0, subarrno=0)
        biases = list(np.unique(np.concatenate([bias_fns, zero_fns])))
        bias_cl = ccdp.ImageFileCollection(filenames=biases)
        return_object.append(bias_cl.files if fnames_only else bias_cl)

    if inst_flag['get_flats']:
        # Gather DOME FLAT frames; FULL FRAME ONLY
        # TODO: SKY FLAT not supported at this time -- Add support
        domeflat_cl = icl.filter(obstype='dome flat', subarrno=0)
        return_object.append(domeflat_cl.files if fnames_only else domeflat_cl)

    if inst_flag['check_binning'] and not fnames_only:
        # Get the complete list of binnings used -- but clear out `None` entries
        bin_list = icl.values('ccdsum', unique=True)
        bin_list = sorted(list(filter(None, bin_list)))
        return_object.append(bin_list)

    #===============================================================#
    # If we only want the filenames, flatten out the list and return
    if fnames_only:
        return list(np.concatenate(return_object).flat)

    # Otherwise, return the accumulated objects as a tuple
    return tuple(return_object)


def gather_other_frames():
    """gather_other_frames Stub for additional functionality

    [extended_summary]
    """
