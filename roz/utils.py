# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Utility Functions and Variables

Further description.
"""

# Built-In Libraries

# 3rd Party Libraries
from astropy.modeling import models
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string
import numpy as np

# Internal Imports


# List of LMI Filters
LMI_FILTERS = ['U', 'B', 'V', 'R', 'I',
               'SDSS-U', 'SDSS-G', 'SDSS-R', 'SDSS-I', 'SDSS-Z',
               'VR', 'YISH', 'OIII', 'HALPHAON', 'HALPHAOFF',
               'WR-WC', 'WR-WN', 'WR-CT',
               'UC','BC','GC','RC','C2','C3','CN','CO+','H2O+','OH','NH']
# Fold Mirror Names
FMS = ['A', 'B', 'C', 'D']


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


