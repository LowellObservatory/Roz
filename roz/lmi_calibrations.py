# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Analyze LMI Calibration Frames for 1 Night

Further description.
"""

# Built-In Libraries
import os
import warnings

# 3rd Party Libraries
from astropy.modeling import models
from astropy.stats import mad_std
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string
import numpy as np

# Internal Imports
from .database_manager import CalibrationDatabase


# List of LMI Filters
LMI_FILTERS = ['U', 'B', 'V', 'R', 'I',
               'SDSS-U', 'SDSS-G', 'SDSS-R', 'SDSS-I', 'SDSS-Z',
               'VR', 'YISH', 'OIII', 'HALPHAON', 'HALPHAOFF',
               'WR-WC', 'WR-WN', 'WR-CT',
               'UC','BC','GC','RC','C2','C3','CN','CO+','H2O+','OH','NH']
# Fold Mirror Names
FMS = ['A', 'B', 'C', 'D']

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
    icl = ccdp.ImageFileCollection(directory, glob_include="lmi*.fits")

    # Gather any bias frames (OBSTYPE=`bias` or EXPTIME=0)
    bias_fns = icl.files_filtered(obstype='bias')
    zero_fns = icl.files_filtered(exptime=0)
    biases = np.unique(np.concatenate([bias_fns, zero_fns]))
    bias_cl = ccdp.ImageFileCollection(filenames=biases.tolist())

    # Gather any FLAT frames (OBSTYPE=`SKY FLAT` or OBSTYPE=`DOME FLAT`)
    flat_cl = icl.filter(obstype='[a-zA-Z]+ flat', regex_match=True)

    # Get the complete list of binnings used -- but clear out "None" entries
    bin_list = icl.values('ccdsum', unique=True)
    if len(bin_list) > 1:
        print(f"This is the bin_list: {bin_list}")
        raise ValueError("More than one binning exists in this directory!")

    return bias_cl, flat_cl, bin_list


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

    # Error checking for binning
    if binning is None:
        raise InputError('Binning not set.')
    if debug:
        print(f"Combining bias frames with binning {binning}...")

    # Double-check that we're combining bias frames of identical binning
    bias_cl = bias_cl.filter(ccdsum=binning)

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

    # Loop through flats, subtracting bias and gathering statistics
    flat_meta = []
    for ccd in flat_cl.ccds(ccdsum=binning, bitpix=16):

        hdr = ccd.header
        # Fit the overscan section, subtract it, then trim the image
        ccd = trim_oscan(ccd, hdr['BIASSEC'], hdr['TRIMSEC'])
        # Subtract master bias
        ccd = ccdp.subtract_bias(ccd, bias_frame)

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
                            'flatavg': np.mean(ccd.data),
                            'flatmed': np.median(ccd.data),
                            'flatstd': np.std(ccd.data),
                            'quadsurf': fit_quadric_surface(ccd.data)})

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
        database.flat['lmi_filt'] = validate_flat_table(flat_meta, lmi_filt)

    return database


def validate_bias_table(bias_meta):
    return None


def validate_flat_table(flat_meta):
    return None


# Helper Functions ===========================================================#
def trim_oscan(ccd, biassec, trimsec):
    """trim_oscan Subtract the overscan region and trim image to desired size

    The CCDPROC function subtract_overscan() expects the TRIMSEC of the image
    (the part you want to keep) to span the entirety of one dimension, with the
    BIASSEC (overscan section) being at the end of the other dimension.
    Both LMI and DeVeny have edge effects on all sides of their respective
    chips, and so the TRIMSEC and BIASSEC do not meet the expectations of
    subtract_overscan().

    Therefore, this function is a wrapper to first remove the undesired ROWS
    from top and bottom, then perform the subtract_overscan() fitting and
    subtraction, followed by trimming off the now-spent overscan region.

    At present, the overscan region is modeled with a first-order Chebyshev
    one-dimensional polynomial.  The model used can be changed in the future
    or allowed as a input, as desired.

    Parameters
    ----------
    ccd : `astropy.nddata.CCDData`
        The CCDData object upon which to operate
    biassec : `str`
        String containing the FITS-convention overscan section coordinates
    trimsec : `str`
        String containing the FITS-convention data section coordinates

    Returns
    -------
    `astropy.nddata.CCDData`
        The properly trimmed and overscan-subtracted CCDData object
    """
    # Convert the FITS bias & trim sections into slice classes for use
    _, xb = slice_from_string(biassec, fits_convention=True)
    yt, xt = slice_from_string(trimsec, fits_convention=True)

    # First trim off the top & bottom rows
    ccd = ccdp.trim_image(ccd[yt.start : yt.stop, :])

    # Model & Subtract the overscan
    ccd = ccdp.subtract_overscan(ccd, overscan=ccd[: , xb.start : xb.stop],
                                 median=True, model=models.Chebyshev1D(1))

    # Trim the overscan & return
    return ccdp.trim_image(ccd[:, xt.start:xt.stop])


def fit_quadric_surface(data, fit_quad=True, return_surface=False):
    """fit_quadric_surface Fit a quadric surface to an image array

    Performs a **LEAST SQUARES FIT** of a (plane or) quadric surface to an
    input image array.  The basic equation is:
            matrix ## fit_coeff = right_hand_side

    This routine computes the `matrix` needed, as well as the right_hand_side.
    The fit coefficients are found by miltiplying matrix^(-1) by the RHS.

    Fit coefficients are:
        coeff[0] = Baseline offset
        coeff[1] = Linear term in x
        coeff[2] = Linear term in y
        coeff[3] = Quadratic term in x
        coeff[4] = Quadratic term in y
        coeff[5] = Quadratic cross-term in xy
    where the last three are only returned if `fit_quad == True`

    Parameters
    ----------
    data : `numpy.ndarray`
        The image (as a 2D array) to be fit with a surface
    fit_quad : `bool`, optional
        Fit a quadric surface, rather than a plane, to the data [Default: True]
    return_surface : `bool`, optional
        Return the model surface, built up from the fit coefficients?
        [Default: False]

    Returns
    -------
    `numpy.ndarray`
        Array of 3 (plane) or 6 (quadric surface) fit coefficients
    `numpy.ndarray` (if `return_surface == True`)
        The 2D array modeling the surface ensconced in the first return.  Array
        is of same size as the input `data`.
    """
    # Construct the arrays for doing the matrix magic
    n_y, n_x = data.shape
    x_coord_arr = np.tile(np.arange(n_x), (n_y,1))
    y_coord_arr = np.transpose(np.tile(np.arange(n_y), (n_x,1)))

    # Construct the matrix for use with the LEAST SQUARES FIT
    #  np.dot(mat, fitvec) = RHS
    n_terms = 6 if fit_quad else 3
    matrix = np.empty((n_terms, n_terms))

    # Compute the terms needed for the matrix
    n_pixels = x_coord_arr.size
    sum_x = np.sum(x_coord_arr)
    sum_y = np.sum(y_coord_arr)
    sum_x2 = np.sum(x_coord_arr * x_coord_arr)
    sum_xy = np.sum(x_coord_arr * y_coord_arr)
    sum_y2 = np.sum(y_coord_arr * y_coord_arr)
    sum_x3 = np.sum(x_coord_arr * x_coord_arr * x_coord_arr)
    sum_x2y = np.sum(x_coord_arr * x_coord_arr * y_coord_arr)
    sum_xy2 = np.sum(x_coord_arr * y_coord_arr * y_coord_arr)
    sum_y3 = np.sum(y_coord_arr * y_coord_arr * y_coord_arr)
    sum_x4 = np.sum(x_coord_arr * x_coord_arr * x_coord_arr * x_coord_arr)
    sum_x3y = np.sum(x_coord_arr * x_coord_arr * x_coord_arr * y_coord_arr)
    sum_x2y2 = np.sum(x_coord_arr * x_coord_arr * y_coord_arr * y_coord_arr)
    sum_xy3 = np.sum(x_coord_arr * y_coord_arr * y_coord_arr * y_coord_arr)
    sum_y4 = np.sum(y_coord_arr * y_coord_arr * y_coord_arr * y_coord_arr)

    # Fill in the matrix elements
    #  Upper left quadrant (or only quadrant, if fitting linear):
    matrix[:3,:3] = [[n_pixels, sum_x, sum_y],
                     [sum_x, sum_x2, sum_xy],
                     [sum_y, sum_xy, sum_y2]]

    # And the other 3 quadrants, if fitting a quadric surface
    if fit_quad:
        # Lower left quadrant:
        matrix[3:,:3] = [[sum_x2, sum_x3, sum_x2y],
                         [sum_y2, sum_xy2, sum_y3],
                         [sum_xy, sum_x2y, sum_xy2]]
        # Right half:
        matrix[:,3:] = [[sum_x2, sum_y2, sum_xy],
                        [sum_x3, sum_xy2, sum_x2y],
                        [sum_x2y, sum_y3, sum_xy2],
                        [sum_x4, sum_x2y2, sum_x3y],
                        [sum_x2y2, sum_y4, sum_xy3],
                        [sum_x3y, sum_xy3, sum_x2y2]]

    # The right-hand side of the matrix equation:
    right_hand_side = np.empty(n_terms)

    # Top half:
    right_hand_side[:3] = [np.sum(data),
                           np.sum(x_coord_arr * data),
                           np.sum(y_coord_arr * data)]

    if fit_quad:
        # Bottom half:
        right_hand_side[3:] = [np.sum(x_coord_arr * x_coord_arr * data),
                               np.sum(y_coord_arr * y_coord_arr * data),
                               np.sum(x_coord_arr * y_coord_arr * data)]

    # Here's where the magic of matrix multiplication happens!
    fit_coefficients = np.dot(np.linalg.inv(matrix), right_hand_side)

    # If not returning the model surface, go ahead and return now
    if not return_surface:
        return fit_coefficients

    # Build the model fit from the coefficients
    model_fit = fit_coefficients[0] + \
                fit_coefficients[1] * x_coord_arr + \
                fit_coefficients[2] * y_coord_arr

    if fit_quad:
        model_fit += fit_coefficients[3] * x_coord_arr * x_coord_arr + \
                     fit_coefficients[4] * y_coord_arr * y_coord_arr + \
                     fit_coefficients[5] * x_coord_arr * y_coord_arr

    return fit_coefficients, model_fit


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
            return None

        # If not passed a directory, exit
        if not os.path.isdir(args[1]):
            print("ERROR: Must specify a directory to process.")
            return None
        directory = args[1]

    # Collect the BIAS & FLAT frames for this directory
    bias_cl, flat_cl, bin_list = gather_cal_frames(directory)

    # Process the BIAS frames to produce a reduced frame and statistics
    bias_frame, bias_meta = process_bias(bias_cl, binning=bin_list[0],
                                                mem_limit=mem_limit)

    # Process the FLAT frames to produce statistics
    flat_meta = process_flats(flat_cl, bias_frame, binning=bin_list[0])

    # Take the metadata from the BAIS and FLAT frames and produce something
    return produce_database_object(bias_meta, flat_meta)


if __name__ == "__main__":
    import sys
    main(sys.argv)
