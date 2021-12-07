# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 08-Oct-2021
#
#  @author: tbowers

"""Utility Functions and Variables

This module is part of the Roz package, written at Lowell Observatory.

This module contains various utility routines and global variables from across
the package.

This module primarily trades in... utility?
"""

# Built-In Libraries

# 3rd Party Libraries
from astropy.io.votable import parse
from astropy.modeling import models
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string
from importlib_resources import files as pkg_files
import numpy as np

# Internal Imports

# Data & config directories
ROZ_CONFIG = pkg_files('Roz.config')
ROZ_DATA = pkg_files('Roz.data')
XML_TABLE = ROZ_DATA.joinpath('lmi_filter_table.xml')
HTML_TABLE_FN = 'lmi_filter_table.html'
LMI_DYNTABLE = ROZ_DATA.joinpath('lmi_dynamic_filter.fits')

# List of supported instruments
INSTRUMENTS = ['LMI', 'DEVENY']  # Could add RC1/2 at some point?

# List of LMI Filters
LMI_FILTERS = parse(XML_TABLE).get_first_table().to_table()['FITS_Header_Value']

# Fold Mirror Names
FMS = ['A', 'B', 'C', 'D']

# Create an error class to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


def set_instrument_flags(instrument='lmi'):
    """set_instrument_flags Set the global instrument flags for processing

    These instrument-specific flags are used throughout the code.  As more
    instruments are added to Roz, this function will grow commensurately.

    Alternatively, this information could be placed in an XML VOTABLE that
    could simply be read in -- to eliminiate one more hard-coded thing.

    Parameters
    ----------
    instrument : `str`, optional
        Name of the instrument to use.  [Default: LMI]

    Returns
    -------
    `dict`
        Dictionary of instrument flags.

    Raises
    ------
    InputError
        If the input instrument is not in the list, raise error.
    """
    # Check the instrument
    if (instrument := instrument.upper()) not in INSTRUMENTS:
        raise InputError(f"Instrument {instrument} not supported!")

    if instrument == 'LMI':
        inst_flag = {'instrument': instrument,
                     'prefix': 'lmi',
                     'get_bias': True,
                     'get_flats': True,
                     'check_binning': True}
                     # Other flags...
    elif instrument == 'DEVENY':
        inst_flag = {'instrument': instrument,
                     'prefix': '20',
                     'get_bias': True,
                     'get_flats': False,
                     'check_binning': True}
                     # Other flags...
    else:
        raise InputError(f"Developer: Add {instrument} to utils.py")

    return inst_flag


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


# Quadric Surface Functions ==================================================#
def fit_quadric_surface(data, c_arr=None, fit_quad=True, return_surface=False):
    """fit_quadric_surface Fit a quadric surface to an image array

    Performs a **LEAST SQUARES FIT** of a (plane or) quadric surface to an
    input image array.  The basic equation is:
            matrix ## fit_coeff = right_hand_side

    In specific, the quadric surface fit is either an elliptic or hyperbolic
    paraboloid (of arbitrary orientation), since the resulting equation is:
        z = a0 + a1•x + a2•y + a3•x^2 + a4•y^2 + a5•xy
    https://en.wikipedia.org/wiki/Quadric
    https://en.wikipedia.org/wiki/Paraboloid

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
    ca : `dict`, optional
        Dictionary of coefficient arrays needed for creating the matrix
        [Default: None]
    fit_quad : `bool`, optional
        Fit a quadric surface, rather than a plane, to the data [Default: True]
    return_surface : `bool`, optional
        Return the model surface, built up from the fit coefficients?
        [Default: False]

    Returns
    -------
    `numpy.ndarray`
        Array of 3 (plane) or 6 (quadric surface) fit coefficients
    `dict`
        Dictionary of coefficient arrays needed for creating the matrix
    `numpy.ndarray` (if `return_surface == True`)
        The 2D array modeling the surface ensconced in the first return.  Array
        is of same size as the input `data`.
    """
    # Construct the matrix for use with the LEAST SQUARES FIT
    #  np.dot(mat, fitvec) = RHS
    n_terms = 6 if fit_quad else 3
    matrix = np.empty((n_terms, n_terms))

    # Produce the coordinate arrays, if not fed an existing dict
    c_arr = produce_coordinate_arrays(data.shape) if c_arr is None else c_arr

    # Fill in the matrix elements
    #  Upper left quadrant (or only quadrant, if fitting linear):
    matrix[:3,:3] = [[c_arr['n_pixels'], c_arr['sum_x'], c_arr['sum_y']],
                     [c_arr['sum_x'], c_arr['sum_x2'], c_arr['sum_xy']],
                     [c_arr['sum_y'], c_arr['sum_xy'], c_arr['sum_y2']]]

    # And the other 3 quadrants, if fitting a quadric surface
    if fit_quad:
        # Lower left quadrant:
        matrix[3:,:3] = [[c_arr['sum_x2'], c_arr['sum_x3'], c_arr['sum_x2y']],
                         [c_arr['sum_y2'], c_arr['sum_xy2'], c_arr['sum_y3']],
                         [c_arr['sum_xy'], c_arr['sum_x2y'], c_arr['sum_xy2']]]
        # Right half:
        matrix[:,3:] = [[c_arr['sum_x2'], c_arr['sum_y2'], c_arr['sum_xy']],
                        [c_arr['sum_x3'], c_arr['sum_xy2'], c_arr['sum_x2y']],
                        [c_arr['sum_x2y'], c_arr['sum_y3'], c_arr['sum_xy2']],
                        [c_arr['sum_x4'], c_arr['sum_x2y2'], c_arr['sum_x3y']],
                        [c_arr['sum_x2y2'], c_arr['sum_y4'], c_arr['sum_xy3']],
                        [c_arr['sum_x3y'], c_arr['sum_xy3'], c_arr['sum_x2y2']]]

    # The right-hand side of the matrix equation:
    right_hand_side = np.empty(n_terms)

    # Top half:
    right_hand_side[:3] = [np.sum(data),
                           np.sum(xd := np.multiply(c_arr['x_coord_arr'], data)),
                           np.sum(yd := np.multiply(c_arr['y_coord_arr'], data))]

    if fit_quad:
        # Bottom half:
        right_hand_side[3:] = [np.sum(np.multiply(c_arr['x_coord_arr'], xd)),
                               np.sum(np.multiply(c_arr['y_coord_arr'], yd)),
                               np.sum(np.multiply(c_arr['x_coord_arr'], yd))]

    # Here's where the magic of matrix multiplication happens!
    fit_coefficients = np.dot(np.linalg.inv(matrix), right_hand_side)

    # If not returning the model surface, go ahead and return now
    if not return_surface:
        return fit_coefficients, c_arr

    # Build the model fit from the coefficients
    model_fit = fit_coefficients[0] + \
                fit_coefficients[1] * c_arr['x_coord_arr'] + \
                fit_coefficients[2] * c_arr['y_coord_arr']

    if fit_quad:
        model_fit += fit_coefficients[3] * c_arr['x2'] + \
                     fit_coefficients[4] * c_arr['y2'] + \
                     fit_coefficients[5] * c_arr['xy']

    return fit_coefficients, c_arr, model_fit


def produce_coordinate_arrays(shape):
    """produce_coordinate_arrays Produce the dictionary of coordinate arrays

    Since these coordinate arrays are dependent ONLY upon the SHAPE of the
    input array, when doing multiple fits of data arrays with the same size, it
    greatly speeds things up to compute these arrays once and reuse them.

    Parameters
    ----------
    shape : `tuple`
        The .shape of the data (numpy ndarray)

    Returns
    -------
    `dict`
        Dictionary of coefficient arrays needed for creating the matrix
    """
    # Construct the arrays for doing the matrix magic -- origin in center
    n_y, n_x = shape
    x_arr = np.tile(np.arange(n_x), (n_y,1)) - (n_x / 2.)
    y_arr = np.transpose(np.tile(np.arange(n_y), (n_x,1))) - (n_y / 2.)

    # Compute the terms needed for the matrix
    return {'n_x': n_x, 'n_y': n_y,
            'x_coord_arr': x_arr,
            'y_coord_arr': y_arr,
            'n_pixels': x_arr.size,
            'sum_x': np.sum(x_arr),
            'sum_y': np.sum(y_arr),
            'sum_x2': np.sum(x2 := np.multiply(x_arr, x_arr)),
            'sum_xy': np.sum(xy := np.multiply(x_arr, y_arr)),
            'sum_y2': np.sum(y2 := np.multiply(y_arr, y_arr)),
            'sum_x3': np.sum(np.multiply(x2, x_arr)),
            'sum_x2y': np.sum(np.multiply(x2, y_arr)),
            'sum_xy2': np.sum(np.multiply(x_arr, y2)),
            'sum_y3': np.sum(np.multiply(y2, y_arr)),
            'sum_x4': np.sum(np.multiply(x2, x2)),
            'sum_x3y': np.sum(np.multiply(x2, xy)),
            'sum_x2y2': np.sum(np.multiply(x2, y2)),
            'sum_xy3': np.sum(np.multiply(xy, y2)),
            'sum_y4': np.sum(np.multiply(y2, y2)),
            'x2': x2,
            'xy': xy,
            'y2': y2}


def compute_human_readable_surface(coefficients):
    """compute_human_readable_surface Rotate surface into standard-ish form

    Use the standard form of:
        z = Ax^2 + Bxy + Cy^ + Dx + Ey + F

    Find the rotation when the axes of the surface are along x' and y':
        x = x'*cos(th) - y'*sin(th)
        y = x'*sin(th) + y'*cos(th)

    Rotate this into standard form of:
        z = x'^2/a^2 + y'^2/b^2 + x'/c + y'/d + F
    where a = semimajor axis (slower-changing direction) -> x'
          b = semiminor axis (faster-changing direction) -> y'
          c = 1 / slope along x'  (Scale of the change, like a,b)
          d = 1 / slope along y'  (Scale of the change, like a,b)

    https://courses.lumenlearning.com/ivytech-collegealgebra/chapter/
    writing-equations-of-rotated-conics-in-standard-form/

    Parameters
    ----------
    coefficients : `numpy.ndarray`
        Coefficients output from fit_quadric_surface()

    Returns
    -------
    `dict`
        Dictionary of human-readable quantities
    """

    # Parse the coefficients from the quadric surface into standard form
    F, D, E, A, C, B = tuple(coefficients)

    # Compute the rotation of the axes of the surface away from x-y
    theta = 0.5 * np.arctan2(B, A-C)

    # Use a WHILE loop to check for orientation issues
    good_orient = False
    while not good_orient:

        # Always use a theta between 0º and 180º:
        theta = theta + np.pi if theta < 0 else theta
        theta = theta - np.pi if theta > np.pi else theta

        # Define sine and cosine for ease of typing and reading
        costh = np.cos(theta)
        sinth = np.sin(theta)

        # Compute the rotated coefficients
        #  xpxp == coefficient on x'^2 in Standard Form
        #  ypyp == coefficient on y'^2 in Standard Form
        xpxp = A*costh**2 + B*sinth*costh + C*sinth**2
        ypyp = A*sinth**2 - B*sinth*costh + C*costh**2

        # Convert to "semimajor" and "semiminor" axes from Standard Form
        semimaj = 1/np.sqrt(np.absolute(xpxp))
        semimin = 1/np.sqrt(np.absolute(ypyp))

        # Check orientation (s.t. semimajor axis is larger than semiminor)
        if semimaj > semimin:
            good_orient = True
        else:
            theta += np.pi/2.

    # Along the native axes, the coefficient on x'y' == 0
    #  Compute as a check
    #xpyp = 2*(C-A)*sinth*costh + B*(costh**2 - sinth**2)

    # Convert values into human-readable things
    return {'rot': np.rad2deg(theta),
            'maj': semimaj,
            'min': semimin,
            'bma': 1./(D*costh + E*sinth),
            'bmi': 1./(-D*sinth + E*costh),
            'zpt': F,
            'oma': int(np.sign(xpxp)),
            'omi': int(np.sign(ypyp)),
            'typ': f"Elliptic Paraboloid {'Up' if np.sign(xpxp) == 1 else 'Down'}" \
                if np.sign(xpxp) == np.sign(ypyp) else "Hyperbolic Paraboloid"}


def compute_flatness(human, shape, stddev):
    """compute_flatness Compute "flatness" statistics

    This function computes a pair of "flatness" statistics for calibration
    frames.  These are used as both a measure in themselves and as a marker
    for investigating change over time.  Changes in "flatness" are used as one
    of the alerting criteria.

    The statistics are basically computed as how fast the large-scale shape
    changes compared to both the smaller dimension of the image and the
    variability within the image, as defined by the "cropped" standard
    deviation.

    For each of the linear (plane) and quadratic portions of the quadric
    surface fit, the flatness statistic is computed as:
        flatness = (smaller dimension, pix) / (change scale, pix per ADU) /
                   (standard deviation, ADU)

    The resulting statistic is unitless, and always positive.  A value of 1
    implies that the fit surface changes by 1 standard deviation over the
    length of the image's smaller dimension.  A perfectly flat image would
    have a value of zero -- highly curved or tilted images would have values
    much larger than 1.

    Parameters
    ----------
    human : `dict`
        Dictionary of human-readable quantities from
        compute_human_readable_surface()
    shape : `int`,`int`
        Tuple of frame sizes (nx, ny)
    stddev : `float`
        Standard deviation of the "crop" section of the frame, used as a scale
        aganist which the tilt or curvature nonflatness is measured.

    Returns
    -------
    `float`, `float`
        Tuple of flatness stat for linear tilt, flatness stat for quadratic
        curvature.
    """
    # Frame minimum dimension
    dim_min = np.minimum(shape[0], shape[1])

    # The keys 'maj' and 'min' refer to the # of pixels until the quadradic
    #  changes by 1 sigma from the center of the paraboloid.  Multiply by the
    #  standard deviation to yield a value for use.
    # NOTE: This is not strictly correct, since the quadratic will actually
    #       reach the value of the standard deviation much faster than this
    #       linear approximation.  Need to think about how to handle this
    #       more correctly.
    pix_quad = human['min'] * stddev

    # Get the lower pixel count of the two linear axes (distance to 1 sigma)
    pix_lin = np.minimum(np.absolute(human['bma']),
                         np.absolute(human['bmi'])) * stddev

    return dim_min/pix_lin, dim_min/pix_quad
