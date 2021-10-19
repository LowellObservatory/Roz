# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Module for gathering whatever frames are required by Roz.

This module will need to interface at some level with Wadsworth to either
buttle data to whatever machine is running Roz, or to let Roz know that
data has been buttled to the proper location (laforge?) and provide the
proper directory information.
"""

# Built-In Libraries
import warnings

# 3rd Party Libraries
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
import numpy as np

# Internal Imports
from .utils import INSTRUMENTS


# Silence Superflous AstroPy Warnings
warnings.simplefilter('ignore', AstropyWarning)


# Create an error class to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


def gather_cal_frames(directory, instrument='lmi'):
    """gather_cal_frames Gather calibration frames from specified directory

    [extended_summary]

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        Directory name to search for calibration files
    instrument : `str`, optional
        Name of the instrument to gather calibration frames for [Default: LMI]

    Returns
    -------
    `ccdproc.ImageFileCollection`
        ImageFileColleciton containing the BIAS frames from the directory
    `ccdproc.ImageFileCollection`, optional (LMI only)
        ImageFileCollection containing the FLAT frames from the directory
    `list`, optional (LMI only)
        List of binning setups found in this directory

    Raises
    ------
    ValueError
        Temporary bug, will need to expand this to handle multiple binnings
    """
    # Check the instrument
    if (instrument := instrument.upper()) not in INSTRUMENTS:
        raise InputError(f"Instrument {instrument} not supported!")

    # Set various flags by instrument
    if instrument == 'LMI':
        prefix = 'lmi'
        get_bias = True
        get_flats = True
        check_binning = True
        # Other flags...
    elif instrument == 'DEVENY':
        prefix = '20'
        get_bias = True
        get_flats = False
        check_binning = False
        # Other flags...
    else:
        raise InputError(f"Developer: Add {instrument} to gather_frames.py")

    # Create an ImageFileCollection for the specified directory
    icl = ccdp.ImageFileCollection(directory, glob_include=f"{prefix}*.fits")
    return_object = []

    # Keep these separate for now, in case future instruments need one but
    #  not the other
    if get_bias:
        # Gather any bias frames (OBSTYPE=`bias` or EXPTIME=0)
        bias_fns = icl.files_filtered(obstype='bias')
        zero_fns = icl.files_filtered(exptime=0)
        biases = np.unique(np.concatenate([bias_fns, zero_fns]))
        bias_cl = ccdp.ImageFileCollection(filenames=biases.tolist())
        return_object.append(bias_cl)

    if get_flats:
        # Gather any FLAT frames (OBSTYPE=`SKY FLAT` or OBSTYPE=`DOME FLAT`)
        flat_cl = icl.filter(obstype='[a-z]+ flat', regex_match=True)
        return_object.append(flat_cl)

    if check_binning:
        # Get the complete list of binnings used -- but clear out `None` entries
        # TODO: Clear out any `None` entries
        bin_list = icl.values('ccdsum', unique=True)
        if len(bin_list) > 1:
            print(f"This is the bin_list: {bin_list}")
            raise ValueError("More than one binning exists in this directory!")
        return_object.append(bin_list)

    # Return the accumulated objects as a tuple
    return tuple(return_object)


def gather_other_frames():
    """gather_other_frames Stub for additional functionality

    [extended_summary]
    """