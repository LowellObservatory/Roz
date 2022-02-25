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
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.table import Table
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string
from importlib_resources import files as pkg_files
import numpy as np
from numpy.ma.core import MaskedConstant

# Lowell Libraries
from ligmos import utils as lig_utils, workers as lig_workers

# Internal Imports


# Classes to hold useful information
class Paths:
    """ Paths

    [extended_summary]
    """

    # Main data & config directories
    data = pkg_files("Roz.data")
    thumbnail = pkg_files("Roz.thumbnails")
    config = pkg_files("Roz.config")

    # Particular filenames needed by various routines
    xml_table = data.joinpath("lmi_filter_table.xml")
    ecsv_filters = data.joinpath("lmi_filter_table.ecsv")
    ecsv_sechead = data.joinpath("lmi_table_sechead.ecsv")
    html_table_fn = "lmi_filter_table.html"
    lmi_dyntable = data.joinpath("lmi_dynamic_filter.ecsv")
    local_html_table_fn = data.joinpath("lmi_filter_table.html")
    css_table = data.joinpath("lmi_filter_table.css")

    def __init__(self):
        pass


# List of LMI Filters
LMI_FILTERS = list(Table.read(Paths.ecsv_filters)["FITS Header Value"])

# Fold Mirror Names
FMS = ["A", "B", "C", "D"]

# Create an error class to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


def load_saved_bias(instrument, binning):
    """load_saved_bias Load a saved (canned) bias frame

    In the event that a data set does not contain a concomitant bias frame(s),
    load in a saved (canned) frame for use with processing the flat frames.

    Parameters
    ----------
    instrument : `str`
        Instrument name from instrument_flags()
    binning : `str`
        Instrument binning from CCDSUM

    Returns
    -------
    `astropy.nddata.CCDData`
        The (canned) combined, overscan-subtracted bias frame

    Raises
    ------
    FileNotFoundError
        If the desired canned frame does not exist in Paths.data, raise this
        error with a note to the Developer to add said file.
    """
    # Build bias filename
    fn = f"bias_{instrument.lower()}_{binning.replace(' ','x')}.fits"

    print(f"Reading in saved file {fn}...")
    try:
        return CCDData.read(Paths.data.joinpath(fn))
    except Exception as e:
        print(e)
        raise FileNotFoundError(f"Developer: Add {fn} to Paths.data") from e


def write_saved_bias(ccd, instrument, binning):
    """write_saved_bias Write a saved (canned) bias frame

    Write a bias frame to disk for use with other nights' data that has
    no bias.

    Parameters
    ----------
    ccd : `astropy.nddata.CCDData`
        The (canned) combined, overscan-subtracted bias frame to write
    instrument : `str`
        Instrument name from instrument_flags()
    binning : `str`
        Instrument binning from CCDSUM
    """
    # Build bias filename
    fn = f"bias_{instrument.lower()}_{binning.replace(' ','x')}.fits"
    ccd.write(Paths.data.joinpath(fn), overwrite=True)


def read_ligmos_conffiles(confname, conffile="roz.conf"):
    """read_ligmos_conffiles Read a configuration file using LIGMOS

    Having this as a separate function may be a bit of an overkill, but it
    makes it easier to keep the ligmos imports only in one place, and
    simplifies the code elsewhere.

    Parameters
    ----------
    confname : `str`
        Name of the table within the configuration file to parse
    conffile : `str`, optional
        Name of the configuration file to parse  [Default: 'roz.conf']

    Returns
    -------
    `ligmos.utils.classes.baseTarget`
        An object with arrtibutes matching the keys in the associated
        configuration file.
    """
    ligconf = lig_utils.confparsers.rawParser(Paths.config.joinpath(conffile))
    ligconf = lig_workers.confUtils.assignConf(
        ligconf[confname], lig_utils.classes.baseTarget, backfill=True
    )
    return ligconf


def set_instrument_flags(inst="lmi"):
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
    # Read in the instrument flag table
    instrument_table = Table.read(Paths.data.joinpath("instrument_flags.ecsv"))

    # Check that the instrument is in the table
    if (inst := inst.upper()) not in instrument_table["instrument"]:
        raise InputError(
            f"Instrument {inst} not yet supported; " "update instrument_flags.ecsv"
        )

    # Extract the row , and convert it to a dictionary
    for row in instrument_table:
        if row["instrument"] == inst:
            return dict(zip(row.colnames, row))

    raise InputError("Developer error... this line should never run.")


def table_sort_on_list(table, colname, sort_list):
    """table_sort_on_list Sort an AstroPy Table according to a list

    The actual sorting of the table is code taken directly from Astropy v5.0
    (astropy.table.table.py).  This function does an arbitrary sort based on
    an input list.

    Parameters
    ----------
    table : `astropy.table.Table`
        The table to sort
    colname : `str`
        The column name to sort on
    sort_list : `list`
        The list of values to sort with such that table[colname] == sort_list

    Returns
    -------
    `astropy.table.Table`
        The sorted table

    Raises
    ------
    TypeError
        If the input table is not really a table
    ValueError
        If the `sort_list` is not the same length as the table
    """
    # Check that the input parameters are of the proper type
    if not isinstance(table, Table):
        raise TypeError(
            "table must be of type astropy.table.Table not " f"{type(table)}"
        )
    sort_list = list(sort_list)

    # Check that `sort_list` is the same length as the table
    if len(sort_list) != len(table[colname]):
        raise ValueError(
            f"Sorting list and table column {colname} " "must be the same length."
        )

    # Find the indices that sort the table by sort_list
    table.add_index(colname)
    indices = []
    for sort_item in sort_list:
        indices.append(table.loc_indices[sort_item])

    # NOTE: This code paraphrased directly from astropy.table.table.py (v5.0)
    with table.index_mode("freeze"):
        for _, col in table.columns.items():
            # Make a new sorted column.
            new_col = col.take(indices, axis=0)
            # Do the substitution
            try:
                col[:] = new_col
            except Exception:
                table[col.info.name] = new_col

    return table


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
    ccd = ccdp.subtract_overscan(
        ccd,
        overscan=ccd[:, xb.start : xb.stop],
        median=True,
        model=models.Chebyshev1D(1),
    )

    # Trim the overscan & return
    return ccdp.trim_image(ccd[:, xt.start : xt.stop])


def two_sigfig(value):
    """two_sigfig String representation of a float at 2 significant figures

    Simple utility function to return a 2-sigfig representation of a float.

    Limitation: At present, at most 2 decimal places are shown.  Therefore,
                this function will not work as expected for values < 0.1

                If I can figure out dynamic format specifiers, this limitation
                can be removed.

                Also, zero is represented as '-----' rather than numerically.

    Parameters
    ----------
    value : `float`
        Input value to be stringified

    Returns
    -------
    `str`
        String representation of `value` at two significant figures
    """
    # If zero, return a 'N/A' type string
    if value <= 0 or isinstance(value, MaskedConstant):
        return "-----"
    # Compute the number of decimal places using the log10.  The way
    #  np.around() works is that +decimal is to the RIGHT, hence the
    #  negative sign on log10.  The "+1" gives the second sig fig.
    try:
        decimal = -int(np.floor(np.log10(value))) + 1
    except ValueError:
        decimal = 0

    # Choose the output specification
    if decimal <= 0:
        return f"{np.around(value, decimals=decimal):.0f}"
    if decimal == 1:
        return f"{np.around(value, decimals=decimal):.1f}"
    return f"{np.around(value, decimals=decimal):.2f}"


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

    # Produce the coordinate arrays, if not fed an existing dict OR if the
    #  array size is different (occasional edge case)
    reproduce = (c_arr is None) or (data.shape != c_arr["x_coord_arr"].shape)
    c_arr = produce_coordinate_arrays(data.shape) if reproduce else c_arr

    # Fill in the matrix elements
    #  Upper left quadrant (or only quadrant, if fitting linear):
    matrix[:3, :3] = [
        [c_arr["n_pixels"], c_arr["sum_x"], c_arr["sum_y"]],
        [c_arr["sum_x"], c_arr["sum_x2"], c_arr["sum_xy"]],
        [c_arr["sum_y"], c_arr["sum_xy"], c_arr["sum_y2"]],
    ]

    # And the other 3 quadrants, if fitting a quadric surface
    if fit_quad:
        # Lower left quadrant:
        matrix[3:, :3] = [
            [c_arr["sum_x2"], c_arr["sum_x3"], c_arr["sum_x2y"]],
            [c_arr["sum_y2"], c_arr["sum_xy2"], c_arr["sum_y3"]],
            [c_arr["sum_xy"], c_arr["sum_x2y"], c_arr["sum_xy2"]],
        ]
        # Right half:
        matrix[:, 3:] = [
            [c_arr["sum_x2"], c_arr["sum_y2"], c_arr["sum_xy"]],
            [c_arr["sum_x3"], c_arr["sum_xy2"], c_arr["sum_x2y"]],
            [c_arr["sum_x2y"], c_arr["sum_y3"], c_arr["sum_xy2"]],
            [c_arr["sum_x4"], c_arr["sum_x2y2"], c_arr["sum_x3y"]],
            [c_arr["sum_x2y2"], c_arr["sum_y4"], c_arr["sum_xy3"]],
            [c_arr["sum_x3y"], c_arr["sum_xy3"], c_arr["sum_x2y2"]],
        ]

    # The right-hand side of the matrix equation:
    right_hand_side = np.empty(n_terms)

    # Top half:
    right_hand_side[:3] = [
        np.sum(data),
        np.sum(xd := np.multiply(c_arr["x_coord_arr"], data)),
        np.sum(yd := np.multiply(c_arr["y_coord_arr"], data)),
    ]

    if fit_quad:
        # Bottom half:
        right_hand_side[3:] = [
            np.sum(np.multiply(c_arr["x_coord_arr"], xd)),
            np.sum(np.multiply(c_arr["y_coord_arr"], yd)),
            np.sum(np.multiply(c_arr["x_coord_arr"], yd)),
        ]

    # Here's where the magic of matrix multiplication happens!
    fit_coefficients = np.dot(np.linalg.inv(matrix), right_hand_side)

    # If not returning the model surface, go ahead and return now
    if not return_surface:
        return fit_coefficients, c_arr

    # Build the model fit from the coefficients
    model_fit = (
        fit_coefficients[0]
        + fit_coefficients[1] * c_arr["x_coord_arr"]
        + fit_coefficients[2] * c_arr["y_coord_arr"]
    )

    if fit_quad:
        model_fit += (
            fit_coefficients[3] * c_arr["x2"]
            + fit_coefficients[4] * c_arr["y2"]
            + fit_coefficients[5] * c_arr["xy"]
        )

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
    x_arr = np.tile(np.arange(n_x), (n_y, 1)) - (n_x / 2.0)
    y_arr = np.transpose(np.tile(np.arange(n_y), (n_x, 1))) - (n_y / 2.0)

    # Compute the terms needed for the matrix
    return {
        "n_x": n_x,
        "n_y": n_y,
        "x_coord_arr": x_arr,
        "y_coord_arr": y_arr,
        "n_pixels": x_arr.size,
        "sum_x": np.sum(x_arr),
        "sum_y": np.sum(y_arr),
        "sum_x2": np.sum(x2 := np.multiply(x_arr, x_arr)),
        "sum_xy": np.sum(xy := np.multiply(x_arr, y_arr)),
        "sum_y2": np.sum(y2 := np.multiply(y_arr, y_arr)),
        "sum_x3": np.sum(np.multiply(x2, x_arr)),
        "sum_x2y": np.sum(np.multiply(x2, y_arr)),
        "sum_xy2": np.sum(np.multiply(x_arr, y2)),
        "sum_y3": np.sum(np.multiply(y2, y_arr)),
        "sum_x4": np.sum(np.multiply(x2, x2)),
        "sum_x3y": np.sum(np.multiply(x2, xy)),
        "sum_x2y2": np.sum(np.multiply(x2, y2)),
        "sum_xy3": np.sum(np.multiply(xy, y2)),
        "sum_y4": np.sum(np.multiply(y2, y2)),
        "x2": x2,
        "xy": xy,
        "y2": y2,
    }


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
    theta = 0.5 * np.arctan2(B, A - C)

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
        xpxp = A * costh ** 2 + B * sinth * costh + C * sinth ** 2
        ypyp = A * sinth ** 2 - B * sinth * costh + C * costh ** 2

        # Convert to "semimajor" and "semiminor" axes from Standard Form
        semimaj = 1 / np.sqrt(np.absolute(xpxp))
        semimin = 1 / np.sqrt(np.absolute(ypyp))

        # Check orientation (s.t. semimajor axis is larger than semiminor)
        if semimaj > semimin:
            good_orient = True
        else:
            theta += np.pi / 2.0

    # Along the native axes, the coefficient on x'y' == 0
    #  Compute as a check
    # xpyp = 2*(C-A)*sinth*costh + B*(costh**2 - sinth**2)

    # Convert values into human-readable things
    return {
        "rot": np.rad2deg(theta),
        "maj": semimaj,
        "min": semimin,
        "bma": 1.0 / (D * costh + E * sinth),
        "bmi": 1.0 / (-D * sinth + E * costh),
        "zpt": F,
        "oma": int(np.sign(xpxp)),
        "omi": int(np.sign(ypyp)),
        "typ": f"Elliptic Paraboloid {'Up' if np.sign(xpxp) == 1 else 'Down'}"
        if np.sign(xpxp) == np.sign(ypyp)
        else "Hyperbolic Paraboloid",
    }


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
    # TODO: This is not strictly correct, since the quadratic will actually
    #       reach the value of the standard deviation much faster than this
    #       linear approximation.  Need to think about how to handle this
    #       more correctly.
    pix_quad = human["min"] * stddev

    # Get the lower pixel count of the two linear axes (distance to 1 sigma)
    pix_lin = np.minimum(np.absolute(human["bma"]), np.absolute(human["bmi"])) * stddev

    return dim_min / pix_lin, dim_min / pix_quad
