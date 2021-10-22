# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Process the Calibration Frames for 1 Night for specified instrument

This module is part of the Roz package, written at Lowell Observatory.

This module takes the gathered calibration frames from a night (as collected by
roz.gather_frames) and performs basic data processing (bias & overscan
subtraction) before gathering statistics.  The statistics are then stuffed into
a database object (from roz.database_manager) for later use.

This module primarily trades in AstroPy Table objects (`astropy.table.Table`)
and CCDPROC Image File Collections (`ccdproc.ImageFileCollection`), along with
the odd AstroPy CCDData object (`astropy.nddata.CCDData`) and basic python
dictionaries (`dict`).
"""

# Built-In Libraries
import warnings

# 3rd Party Libraries
from astropy.stats import mad_std
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string as get_slice
import numpy as np
from tqdm import tqdm

# Internal Imports
from .analyze_alert import validate_bias_table, validate_flat_table
from .database_manager import CalibrationDatabase
from .utils import fit_quadric_surface, trim_oscan, LMI_FILTERS, FMS


# Silence Superflous AstroPy Warnings
warnings.simplefilter('ignore', AstropyWarning)


# Create an error class to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


def process_bias(bias_cl, binning=None, debug=True, mem_limit=8.192e9):
    """process_bias Process and combine available bias frames

    [extended_summary]

    Parameters
    ----------
    bias_cl : `ccdproc.ImageFileCollection`
        IFC containing bias frames to be combined
    binning : `str`, optional
        Binning of the CCD -- must be specified by the caller [Default: None]
    debug : `bool`, optional
        Print debugging statements? [Default: True]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]

    Returns
    -------
    `astropy.table.Table`
        A table containing information about the bias frames for analysis
    `astropy.nddata.CCDData`
        The combined, overscan-subtracted bias frame

    Raises
    ------
    InputError
        Raised if the binning is not set.
    """
    # Last check to ensure there are bias frames
    if not bias_cl.files:
        return None, None

    # Error checking for binning
    if binning is None:
        raise InputError('Binning not set.')
    if debug:
        print(f"Combining bias frames with binning {binning}...")

    # Double-check that we're combining bias frames of identical binning
    bias_cl = bias_cl.filter(ccdsum=binning)

    # Show progress bar for processing bias frames
    progress_bar = tqdm(total=len(bias_cl.files), unit='frame',
                        unit_scale=False, colour='yellow')

    # Loop through files
    bias_ccds, metadata, coord_arrays = [], [], None
    for ccd in bias_cl.ccds(bitpix=16):

        hdr = ccd.header
        data = ccd.data[get_slice(hdr['TRIMSEC'], fits_convention=True)]

        # Statistics, statistics, statistics!!!!
        quadsurf, coord_arrays = fit_quadric_surface(data, coord_arrays)
        metadata.append({'utdate': hdr['DATE-OBS'].split('T')[0],
                         'utcstart': hdr['UTCSTART'],
                         'frametyp': hdr['OBSTYPE'],
                         'obserno': hdr['OBSERNO'],
                         'binning': hdr['CCDSUM'],
                         'numamp': hdr['NUMAMP'],
                         'ampid': hdr['AMPID'],
                         'mnttemp': hdr['MNTTEMP'],
                         'tempamb': hdr['TEMPAMB'],
                         'biasavg': np.mean(data),
                         'biasmed': np.ma.median(data),
                         'cen_avg': np.mean(data[100:-100,100:-100]),
                         'cen_med': np.ma.median(data[100:-100,100:-100]),
                         'cen_std': np.std(data[100:-100,100:-100]),
                         'quadsurf': quadsurf})

        # Fit the overscan section, subtract it, then trim the image
        #  Append this to a list, update the progress bar and repeat!
        bias_ccds.append(trim_oscan(ccd, hdr['BIASSEC'], hdr['TRIMSEC']))
        progress_bar.update(1)

    progress_bar.close()

    if debug:
        print("Doing median combine of biases now...")

    # Convert the list of dicts into a Table and return, plus combined bias
    return Table(metadata), \
        ccdp.combine(bias_ccds, method='median', sigma_clip=True,
                     sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                     sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                     mem_limit=mem_limit)


def process_flats(flat_cl, bias_frame, binning=None, debug=True):
    """process_flats Process the flat fields and return statistics

    [extended_summary]

    Parameters
    ----------
    flat_cl : `ccdproc.ImageFileCollection`
        The ImageFileCollection of FLAT frames to process
    bias_frame : `astropy.nddata.CCDData`
        The combined, overscan-subtracted bias frame
    binning : `str`, optional
        The binning to use for this routine [Default: None]
    debug : `bool`, optional
        Print debugging statements? [Default: True]

    Returns
    -------
    `astropy.table.Table`
        The table of relevant metadata and statistics for each frame

    Raises
    ------
    InputError
        Raised if the binning is not set.
    """
    # Error checking for binning
    if binning is None:
        raise InputError('Binning not set.')
    if debug:
        print(f"Processing flat frames with binning {binning}...")

    # Show progress bar for processing flat frames
    progress_bar = tqdm(total=len(flat_cl.files), unit='frame',
                        unit_scale=False, colour='yellow')

    # Loop through flat frames, subtracting bias and gathering statistics
    metadata, coord_arrays = [], None
    for ccd in flat_cl.ccds(ccdsum=binning, bitpix=16):

        hdr = ccd.header
        # Fit & subtract the overscan section, trim the image, subtract bias
        ccd = trim_oscan(ccd, hdr['BIASSEC'], hdr['TRIMSEC'])
        ccd = ccdp.subtract_bias(ccd, bias_frame)

        # Work entirely in COUNT RATE -- ergo divide by exptime
        count_rate = ccd.divide(hdr['EXPTIME'])

        # Statistics, statistics, statistics!!!!
        quadsurf, coord_arrays = fit_quadric_surface(count_rate, coord_arrays)
        metadata.append({'utdate': hdr['DATE-OBS'].split('T')[0],
                         'utcstart': hdr['UTCSTART'],
                         'frametyp': hdr['OBSTYPE'],
                         'obserno': hdr['OBSERNO'],
                         'binning': hdr['CCDSUM'],
                         'numamp': hdr['NUMAMP'],
                         'ampid': hdr['AMPID'],
                         'mnttemp': hdr['MNTTEMP'],
                         'tempamb': hdr['TEMPAMB'],
                         'filter': hdr['FILTERS'],
                         'exptime': hdr['EXPTIME'],
                         'rc1pos': [hdr['P1X'], hdr['P1Y']],
                         'rc2pos': [hdr['P2X'], hdr['P2Y']],
                         'icstat': hdr['ICSTAT'],
                         'icpos': hdr['ICPOS'],
                         'fmstat': [hdr[f"FM{x}STAT"] for x in FMS],
                         'fmpos': [hdr[f"FM{x}POS"] for x in FMS],
                         'flatavg': np.mean(count_rate),
                         'flatmed': np.ma.median(count_rate),
                         'flatstd': np.std(count_rate),
                         'quadsurf': quadsurf})
        progress_bar.update(1)

    progress_bar.close()

    # Convert the list of dicts into a Table and return
    return Table(metadata)


def produce_database_object(bias_meta, flat_meta, inst_flags):
    """produce_database_object Stuff the metadata tables into a database object

    [extended_summary]

    Parameters
    ----------
    bias_meta : `astropy.table.Table`
        Table containing the metadata and statistics for BIAS frames
    flat_meta : `astropy.table.Table`
        Table containing the metadata and statistics for FLAT frames
    inst_flags : `dict`
        Dictionary of instrument flags from .utils.set_instrument_flags()

    Returns
    -------
    `database_manager.CalibrationDatabase`
        Database object for use with... something?
    """
    # Instantiate the database
    database = CalibrationDatabase(inst_flags)

    # Analyze the bias_meta table, and insert it into the database
    database.bias = validate_bias_table(bias_meta)

    if inst_flags['get_flats']:
        # Analyze the flat_meta table, sorted by LMI_FILTERS, and insert
        for lmi_filt in LMI_FILTERS:
            database.flat[lmi_filt] = validate_flat_table(flat_meta, lmi_filt)

    # Return the filled database
    return database
