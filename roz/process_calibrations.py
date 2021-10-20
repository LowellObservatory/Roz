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
from tqdm import tqdm

# Internal Imports
from .database_manager import CalibrationDatabase
from .gather_frames import gather_cal_frames
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
        return None

    # Error checking for binning
    if binning is None:
        raise InputError('Binning not set.')
    if debug:
        print(f"Combining bias frames with binning {binning}...")

    # Double-check that we're combining bias frames of identical binning
    bias_cl = bias_cl.filter(ccdsum=binning)

    # Show progress bar for processing flats
    progress_bar = tqdm(total=len(bias_cl.files), unit='frame',
                        unit_scale=False)

    bias_ccds = []
    b_meta = []
    # Loop through files
    for ccd in bias_cl.ccds(bitpix=16):

        hdr = ccd.header
        bias_data = ccd.data[slice_from_string(hdr['TRIMSEC'],
                                               fits_convention=True)]

        # For posterity, gather the mount temperature and mean bias level
        b_meta.append({'utdate': hdr['DATE-OBS'].split('T')[0],
                       'utcstart': hdr['UTCSTART'],
                       'frametyp': hdr['OBSTYPE'],
                       'obserno': hdr['OBSERNO'],
                       'binning': hdr['CCDSUM'],
                       'numamp': hdr['NUMAMP'],
                       'ampid': hdr['AMPID'],
                       'mnttemp': hdr['MNTTEMP'],
                       'tempamb': hdr['TEMPAMB'],
                       'biasavg': np.mean(bias_data),
                       'biasmed': np.ma.median(bias_data),
                       'cen_avg': np.mean(bias_data[100:-100,100:-100]),
                       'cen_med': np.ma.median(bias_data[100:-100,100:-100]),
                       'cen_std': np.std(bias_data[100:-100,100:-100]),
                       'quadsurf': fit_quadric_surface(bias_data)})

        # Fit the overscan section, subtract it, then trim the image
        # Append this to a list
        bias_ccds.append(trim_oscan(ccd, hdr['BIASSEC'], hdr['TRIMSEC']))

        progress_bar.update(1)
    progress_bar.close()

    if debug:
        print("Doing median combine of biases now...")

    # Convert the list of dicts into a Table and return, plus combined bias
    return Table(b_meta), \
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

    # Show progress bar for processing flats
    progress_bar = tqdm(total=len(flat_cl.files), unit='frame',
                        unit_scale=False)

    # Loop through flats, subtracting bias and gathering statistics
    flat_meta = []
    for ccd in flat_cl.ccds(ccdsum=binning, bitpix=16):

        hdr = ccd.header
        # Fit the overscan section, subtract it, then trim the image
        ccd = trim_oscan(ccd, hdr['BIASSEC'], hdr['TRIMSEC'])
        # Subtract master bias
        ccd = ccdp.subtract_bias(ccd, bias_frame)

        # Work entirely in COUNT RATE -- ergo divide by exptime
        count_rate_img = ccd.divide(hdr['EXPTIME'])

        # Statistics, statistics, statistics!!!!
        flat_meta.append({'utdate': hdr['DATE-OBS'].split('T')[0],
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
                            'flatavg': np.mean(count_rate_img),
                            'flatmed': np.ma.median(count_rate_img),
                            'flatstd': np.std(count_rate_img),
                            'quadsurf': fit_quadric_surface(count_rate_img)})
        progress_bar.update(1)
    progress_bar.close()

    # Convert the list of dicts into a Table and return
    return Table(flat_meta)


def produce_database_object(bias_meta, flat_meta):
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

    # Next analyze the data in the flat_meta table, sorted by LMI_FILTERS
    for lmi_filt in LMI_FILTERS:
        database.flat[lmi_filt] = validate_flat_table(flat_meta, lmi_filt)

    return database


def validate_bias_table(bias_meta):
    """validate_bias_table [summary]

    [extended_summary]

    Parameters
    ----------
    bias_meta : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    print("\nIn validate_bias_table():")
    print(np.mean(bias_meta['cen_avg']), np.median(bias_meta['cen_med']),
          np.mean(bias_meta['cen_std']))

    return bias_meta


def validate_flat_table(flat_meta, lmi_filt):
    """validate_flat_table [summary]

    Does nothing at the moment -- just returns the input metadata table

    Parameters
    ----------
    flat_meta : [type]
        [description]
    lmi_filt : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Find the rows of the table corresponding to this filter, return if 0
    idx = np.where(flat_meta['filter'] == lmi_filt)
    if len(idx[0]) == 0:
        return flat_meta

    # For ease, pull these rows into a subtable
    subtable = flat_meta[idx]

    # Make sure 'flats' have a reasonable flat countrate, or total counts
    #  in the range 1,500 - 52,000 ADU above bias.  (Can get from countrate *
    #  exptime).

    # Do something...
    print("\nIn validate_flat_table():")
    print(lmi_filt)
    subtable.pprint()
    print(np.mean(subtable['flatavg']), np.median(subtable['flatmed']))

    # Find the mean quadric surface for this set of flats
    quadsurf = np.mean(np.asarray(subtable['quadsurf']), axis=0)
    print(quadsurf)

    return flat_meta


#=============================================================================#
def main(args=None, directory=None, mem_limit=8.192e9):
    """main This is the main body function

    Collect the LMI calibration frames, produce statistics, and return a
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
            return

        # If not passed a directory, exit
        if not os.path.isdir(args[1]):
            print("ERROR: Must specify a directory to process.")
            return
        directory = args[1]

    # Collect the BIAS & FLAT frames for this directory
    bias_cl, flat_cl, bin_list = gather_cal_frames(directory)

    # Process the BIAS frames to produce a reduced frame and statistics
    bias_meta, bias_frame = process_bias(bias_cl, binning=bin_list[0],
                                                mem_limit=mem_limit)

    # Process the FLAT frames to produce statistics
    flat_meta = process_flats(flat_cl, bias_frame, binning=bin_list[0])

    # Take the metadata from the BAIS and FLAT frames and produce something
    database = produce_database_object(bias_meta, flat_meta)

    # Write to InfluxDB
    database.write_to_influxdb()

    # OR --- Could return the database to calling routine and have that call
    #  the .write_to_influxdb() method.

    return database

if __name__ == "__main__":
    import sys
    main(sys.argv)
