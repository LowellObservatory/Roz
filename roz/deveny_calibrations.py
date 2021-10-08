# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 08-Oct-2021
#
#  @author: tbowers

"""Analyze DeVeny Calibration Frames for 1 Night

Further description.
"""

# Built-In Libraries
import os
import warnings

# 3rd Party Libraries
from astropy.stats import mad_std
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string
import numpy as np

# Internal Imports
from .database_manager import CalibrationDatabase
from .utils import trim_oscan


# Silence Superflous AstroPy Warnings
warnings.simplefilter('ignore', AstropyWarning)


# Create an error class to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


def gather_cal_frames(directory):
    """gather_cal_frames Gather calibration frames from this directory

    [extended_summary]

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        Directory name to search for calibration files

    Returns
    -------
    `ccdproc.ImageFileCollection`
        ImageFileColleciton containing the BIAS frames from the directory
    `ccdproc.ImageFileCollection`
        ImageFileCollection containing the FLAT frames from the directory
    `list`
        List of binning setups found in this directory

    Raises
    ------
    ValueError
        Temporary bug, will need to expand this to handle multiple binnings
    """
    # Create an ImageFileCollection for the specified directory
    icl = ccdp.ImageFileCollection(directory, glob_include="20*.fits")

    # Gather any bias frames (OBSTYPE=`bias` or EXPTIME=0)
    bias_fns = icl.files_filtered(obstype='bias')
    zero_fns = icl.files_filtered(exptime=0)
    biases = np.unique(np.concatenate([bias_fns, zero_fns]))
    bias_cl = ccdp.ImageFileCollection(filenames=biases.tolist())

    return bias_cl


def process_bias(bias_cl, debug=True, mem_limit=8.192e9):
    """process_bias Process and combine available bias frames

    [extended_summary]

    Parameters
    ----------
    bias_cl : `ccdproc.ImageFileCollection`
        IFC containing bias frames to be combined
    debug : `bool`, optional
        Print debugging statements? [Default: True]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]

    Returns
    -------
    `astropy.nddata.CCDData`
        The combined, overscan-subtracted bias frame
    `astropy.table.Table`
        A table containing information about the bias frames for analysis

    Raises
    ------
    InputError
        Raised if the binning is not set.
    """
    # Last check to ensure there are bias frames
    if not bias_cl.files:
        return None

    bias_ccds = []
    bias_temp = []
    # Loop through files
    for ccd in bias_cl.ccds(bitpix=16):

        hdr = ccd.header
        bias_data = ccd.data[slice_from_string(hdr['TRIMSEC'], fits_convention=True)]

        # For posterity, gather the mount temperature and mean bias level
        bias_temp.append({'utdate': hdr['DATE-OBS'].split('T')[0],
                          'utcstart': hdr['UTCSTART'],
                          'frametyp': hdr['OBSTYPE'],
                          'obserno': hdr['OBSERNO'],
                          'binning': hdr['CCDSUM'],
                          'numamp': hdr['NUMAMP'],
                          'ampid': hdr['AMPID'],
                          'mnttemp': hdr['MNTTEMP'],
                          'tempamb': hdr['TEMPAMB'],
                          'biasavg': np.mean(bias_data),
                          'biasmed': np.median(bias_data)})

        # Fit the overscan section, subtract it, then trim the image
        # Append this to a list
        bias_ccds.append(trim_oscan(ccd, hdr['BIASSEC'], hdr['TRIMSEC']))

    if debug:
        print("Doing median combine of biases now...")

    return ccdp.combine(bias_ccds, method='median', sigma_clip=True,
                             sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                             sigma_clip_func=np.ma.median, mem_limit=mem_limit,
                             sigma_clip_dev_func=mad_std), Table(bias_temp)


def produce_database_object(bias_meta):
    """produce_database_object [summary]

    [extended_summary]

    Parameters
    ----------
    bias_meta : `astropy.table.Table`
        Table containing the metadata and statistics for BIAS frames
    flat_meta : `astropy.table.Table`
        Table containing the metadata and statistics for FLAT frames

    Returns
    -------
    `database_manager.CalibrationDatabase`
        Database object for use with... something?
    """
    database = CalibrationDatabase()

    # First analyze the data in the bias_meta table
    database.bias = validate_bias_table(bias_meta)

    return database


def validate_bias_table(bias_meta):
    return None


def validate_flat_table(flat_meta):
    return None


#=============================================================================#
def main(args=None, directory=None, mem_limit=8.192e9):
    """main This is the main body function

    Collect the DeVeny calibration frames, produce statistics, and return a
    `CalibrationDatabase` object for this one directory.

    Parameters
    ----------
    args : `list`, optional
        If this file is called from the command line, these are the command
        line arguments.  [Default: None]
    directory : `str` or `pathlib.Path`, optional
        The directory to operate on [Default: None]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]

    Returns
    -------
    `database_manager.CalibrationDatabase`
        Database object to be fed into... something?
    """

    # Parse command-line arguments, if called that way
    if args is not None:
        if len(args) == 1:
            print("ERROR: Must specify a directory to process.")
            return None

        # If not passed a directory, exit
        if not os.path.isdir(args[1]):
            print("ERROR: Must specify a directory to process.")
            return None
        directory = args[1]

    # Collect the BIAS & FLAT frames for this directory
    bias_cl = gather_cal_frames(directory)

    # Process the BIAS frames to produce a reduced frame and statistics
    _, bias_meta = process_bias(bias_cl, mem_limit=mem_limit)

    # Take the metadata from the BAIS and FLAT frames and produce something
    return produce_database_object(bias_meta)


if __name__ == "__main__":
    import sys
    main(sys.argv)
