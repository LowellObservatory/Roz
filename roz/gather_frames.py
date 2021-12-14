# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 19-Oct-2021
#
#  @author: tbowers

"""Module for gathering whatever frames are required by Roz.

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
import shutil
import warnings

# 3rd Party Libraries
from astropy.io.fits import getheader
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
import numpy as np

# Internal Imports
from .send_alerts import send_alert, BadDirectoryAlert, BadFrametypeAlert
from .utils import roz_config, set_instrument_flags

# Silence Superflous AstroPy Warnings
warnings.simplefilter('ignore', AstropyWarning)


# Create an error class to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


def butler_bell():
    """butler_bell Ring for the data butler

    This function will interact with Wadsworth (or appropriate successor) to
    have the proper data buttled to a location suitable for processing and
    analysis.

    This function should receive from the Butler the location of the data,
    which is returned.

    Returns
    -------
    `str` or `pathlib.Path`
        Directory name containing the files to be processed and analyzed.
    """
    directory = ''
    return directory


def dumbwaiter(data_dir, frametype='calibration'):
    """dumbwaiter Carry the data to a processable location

    This function scans the (presumably remote) directory for suitable
    files, and copies them to the local processing directory.

    NOTE: 'calibration' is the ONLY type of frame currently supported,
    but the `frametype` keyword is included for future expansions of the
    package.

    Parameters
    ----------
    data_dir : `str` or `pathlib.Path`
        The directory to search for appropriate files
    frametype : `str`, optional
        Type of frame to collect for processing.  [Default: 'calibration']

    Returns
    -------
    instrument : `str`
        The name of the instrument, as returned by divine_instrument()
    frames : `list`
        List of pathless filenames copied into the processing directory
    proc_dir : `pathlib.Path`
        The Path of the processing directory, as extracted from the
        configuration file
    """
    # Check that the (presumably remote) directory is, in fact, a directory
    if not os.path.isdir(data_dir):
        send_alert(BadDirectoryAlert)

    # Path-ify `directory`, if necessary
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    # Determine the instrument in question and set the flags
    instrument = divine_instrument(data_dir)
    inst_flags = set_instrument_flags(instrument)

    # Based on the `frametype`, call the appropriate `gather_*_frames()`
    if frametype == 'calibration':
        frames = gather_cal_frames(data_dir, inst_flags, fnames_only=True)
    else:
        send_alert(BadFrametypeAlert)

    # Copy the calibration frames to a local processing directory -- but first
    #  clear out the processing directory to have a fresh start
    proc_dir = Path(roz_config().processing_dir)
    print(f"Cleaning out previous cruft in processing directory {proc_dir}")
    for entry in os.scandir(proc_dir):
        if entry.is_file():
            os.remove(proc_dir.joinpath(entry))

    # Then copy
    print(f"Copying data from {data_dir} to {proc_dir} for processing...")
    for frame in frames:
        shutil.copy(data_dir.joinpath(frame), proc_dir)

    # Return the instrument name and the list of frames (sans path)
    return instrument, frames, proc_dir


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
    send_alert(BadDirectoryAlert)
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
    `ccdproc.ImageFileCollection`
        ImageFileColleciton containing the BIAS frames from the directory
    `ccdproc.ImageFileCollection`, optional (LMI only)
        ImageFileCollection containing the FLAT frames from the directory
    `list`, optional (LMI only)
        List of binning setups found in this directory
    """
    # Create an ImageFileCollection for the specified directory
    icl = ccdp.ImageFileCollection(
        directory, glob_include=f"{inst_flag['prefix']}*.fits")
    return_object = []

    # Keep these items separate for now, in case future instruments need one
    #  but not the others
    if inst_flag['get_bias']:
        # Gather any bias frames (OBSTYPE=`bias` or EXPTIME=0)
        bias_fns = icl.files_filtered(obstype='bias')
        zero_fns = icl.files_filtered(exptime=0)
        biases = list(np.unique(np.concatenate([bias_fns, zero_fns])))
        bias_cl = ccdp.ImageFileCollection(filenames=biases)
        return_object.append(biases if fnames_only else bias_cl)

    if inst_flag['get_flats']:
        # Gather DOME FLAT frames -- SKY FLAT not supported at this time
        # TODO: Figure out how to add support for SKY FLAT
        domeflat_cl = icl.filter(obstype='dome flat')
        return_object.append(domeflat_cl.files if fnames_only else domeflat_cl)

    if inst_flag['check_binning'] and not fnames_only:
        # Get the complete list of binnings used -- but clear out `None` entries
        bin_list = icl.values('ccdsum', unique=True)
        bin_list = sorted(list(filter(None, bin_list)))
        return_object.append(bin_list)

    # If we only want the filenames, flatten out the list and return
    if fnames_only:
        return list(np.concatenate(return_object).flat)

    # Otherwise, return the accumulated objects as a tuple
    return tuple(return_object)


def gather_other_frames():
    """gather_other_frames Stub for additional functionality

    [extended_summary]
    """
