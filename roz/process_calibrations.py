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
import os
import warnings

# 3rd Party Libraries
from astropy.stats import mad_std
from astropy.table import Table
from astropy.wcs import FITSFixedWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string as get_slice
import numpy as np
from tqdm import tqdm

# Internal Imports
from roz import utils

# Silence Superflous AstroPy FITS Header Warnings
warnings.simplefilter("ignore", FITSFixedWarning)


# Narrative Functions ========================================================#
def process_bias(
    bias_cl, binning=None, debug=True, mem_limit=8.192e9, produce_combined=True
):
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
    produce_combined : `bool`, optional
        Produce and return a combined bais image?  [Default: True]

    Returns
    -------
    `astropy.table.Table`
        A table containing information about the bias frames for analysis
    `astropy.nddata.CCDData` or `NoneType`
        The combined, overscan-subtracted bias frame (if
        `produce_combined == True` else None)
    """
    bias_cl = check_processing_ifc(bias_cl, binning)
    if not bias_cl.files:
        return (Table(), None) if produce_combined else Table()
    if debug:
        print("Processing bias frames...")

    # Show progress bar for processing bias frames
    progress_bar = tqdm(
        total=len(bias_cl.files), unit="frame", unit_scale=False, colour="yellow"
    )

    # Loop through files
    bias_ccds, metadata, coord_arrays = [], [], None
    for ccd, fname in bias_cl.ccds(bitpix=16, return_fname=True):

        hdr = ccd.header
        # For BIAS set header FILTERS keyword to "DARK"
        hdr["FILTERS"] = "DARK"
        hdr["SHORT_FN"] = fname.split(os.sep)[-1]
        data = ccd.data[get_slice(hdr["TRIMSEC"], fits_convention=True)]

        # Statistics, statistics, statistics!!!!
        quadsurf, coord_arrays = utils.fit_quadric_surface(data, coord_arrays)
        metadata.append(base_metadata_dict(hdr, data, quadsurf))

        # Fit the overscan section, subtract it, then trim the image
        #  Append this to a list, update the progress bar and repeat!
        bias_ccds.append(utils.trim_oscan(ccd, hdr["BIASSEC"], hdr["TRIMSEC"]))
        progress_bar.update(1)

    progress_bar.close()

    # Convert the list of dicts into a Table and return, plus combined bias
    combined = None
    if produce_combined:
        if debug:
            print("Doing median combine of biases now...")
        combined = ccdp.combine(
            bias_ccds,
            method="median",
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            mem_limit=mem_limit,
            sigma_clip_dev_func=mad_std,
        )
    return Table(metadata), combined


def process_dark(
    dark_cl, binning=None, debug=True, mem_limit=8.192e9, produce_combined=True
):
    """process_dark Process and combine available dark frames

    NOTE: Not yet implemented -- Boilerplate below is from process_bias
    """
    dark_cl = check_processing_ifc(dark_cl, binning)
    dark_ccds, metadata, _ = [], [], None
    # Convert the list of dicts into a Table and return, plus combined bias
    combined = None
    if produce_combined:
        if debug:
            print("Doing median combine of darks now...")
        combined = ccdp.combine(
            dark_ccds,
            method="median",
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            mem_limit=mem_limit,
            sigma_clip_dev_func=mad_std,
        )
    return Table(metadata), combined


def process_domeflat(
    flat_cl, bias_frame=None, dark_frame=None, binning=None, instrument=None, debug=True
):
    """process_flats Process the flat fields and return statistics

    [extended_summary]

    Parameters
    ----------
    flat_cl : `ccdproc.ImageFileCollection`
        The ImageFileCollection of FLAT frames to process
    bias_frame : `astropy.nddata.CCDData`, optional
        The combined, overscan-subtracted bias frame  [Default: None]
        If None, the routine will load in a saved bias
    dark_frame : `astropt.nddata.CCDData`, optional
        The combined, bias-subtracted dark frame [Default: None]
        If None, the routine will load in a saved dark, if necessary
    binning : `str`, optional
        The binning to use for this routine [Default: None]
    instrument : `str`, optional
        For the case of no bias frame of the proper binning, load in a saved
        bias frame for this instrument  [Default: None]
    debug : `bool`, optional
        Print debugging statements? [Default: True]

    Returns
    -------
    `astropy.table.Table`
        The table of relevant metadata and statistics for each frame
    """
    # Check for existance of flats with this binning, else retun empty Table()
    flat_cl = check_processing_ifc(flat_cl, binning)
    if not flat_cl.files:
        return Table()

    # Check for actual bias frame, else make something up
    if not bias_frame:
        print("No appropriate bias frames passed; loading saved BIAS...")
        bias_frame = utils.load_saved_bias(instrument, binning)
    else:
        # Write this bias to disk for future use
        utils.write_saved_bias(bias_frame, instrument, binning)

    if debug:
        print("Processing flat frames...")

    # Show progress bar for processing flat frames
    progress_bar = tqdm(
        total=len(flat_cl.files), unit="frame", unit_scale=False, colour="yellow"
    )

    # Loop through flat frames, subtracting bias and gathering statistics
    metadata, coord_arrays = [], None
    for ccd, fname in flat_cl.ccds(bitpix=16, return_fname=True):

        hdr = ccd.header
        # Add a "short filename" to the header for use further along
        hdr["SHORT_FN"] = fname.split(os.sep)[-1]

        # Fit & subtract the overscan section, trim the image, subtract bias
        ccd = utils.trim_oscan(ccd, hdr["BIASSEC"], hdr["TRIMSEC"])
        ccd = ccdp.subtract_bias(ccd, bias_frame)

        # Work entirely in COUNT RATE -- ergo divide by exptime
        count_rate = ccd.divide(hdr["EXPTIME"])

        # Statistics, statistics, statistics!!!!
        quadsurf, coord_arrays = utils.fit_quadric_surface(count_rate, coord_arrays)

        metadict = base_metadata_dict(hdr, count_rate, quadsurf)

        # Additional fields for flats: Stuff that can block the light path
        #  Do type-forcing to make InfluxDB happy
        for rc_num in [1, 2]:
            for axis in ["x", "y"]:
                metadict[f"rc{rc_num}pos_{axis.lower()}"] = float(
                    hdr[f"P{rc_num}{axis.upper()}"]
                )
        metadict["icpos"] = float(hdr["ICPOS"])
        for axis in utils.FMS:
            metadict[f"fmpos_{axis.lower()}"] = float(hdr[f"FM{axis.upper()}POS"])

        metadata.append(metadict)
        progress_bar.update(1)

    progress_bar.close()

    # Convert the list of dicts into a Table and return
    return Table(metadata)


def process_skyflat(
    flat_cl, bias_frame=None, dark_frame=None, binning=None, instrument=None, debug=True
):
    """process_flats Process the flat fields and return statistics

    NOTE: Not yet implemented --
    """
    return Table()


def validate_bias_table(bias_meta):
    """validate_bias_table Analyze and validate the bias frame metadata table

    [extended_summary]

    Parameters
    ----------
    bias_meta : `astropy.table.Table`
        A table containing information about the bias frames for analysis

    Returns
    -------
    `astropy.table.Table`
        The, um, validated table?  This may change.
    """
    # If there were no biases at all (blank Table), return None
    if not bias_meta:
        return None

    # For now, just print some stats and return the table.
    print("\nIn validate_bias_table():")
    print(
        f"Mean: {np.mean(bias_meta['crop_avg']):.2f}  "
        f"Median: {np.median(bias_meta['crop_med']):.2f}  "
        f"Stddev: {np.mean(bias_meta['crop_std']):.2f}"
    )

    # Add logic checks for header datatypes (edge cases)

    return bias_meta


def validate_dark_table(dark_meta):
    """validate_dark_table Analyze and validate the dark frame metadata table

    NOTE: Not yet implemented
    """
    return dark_meta


def validate_flat_table(flat_meta, flat_filter):
    """validate_flat_table Analyze and validate the flat frame metadata table

    Separates the wheat from the chaff -- returning a subtable for the
    specified filter, or None.

    Parameters
    ----------
    flat_meta : `astropy.table.Table`
        Table containing the flat frame metadata
    flat_filter : `str`
        Flatfield filter to validate

    Returns
    -------
    `astropy.table.Table` or `None`
        If the `flat_filter` was used in this set, return the subtable of
        `flat_meta` containing that filter.  Otherwise, return `None`.
    """
    # If there were no flats at all (blank Table), return None
    if not flat_meta:
        return None

    # Find the rows of the table corresponding to this filter, return if none
    subtable = flat_meta[flat_meta["filter"] == flat_filter]
    if not subtable:
        return None

    # Make sure 'flats' have a reasonable flat countrate, or total counts
    #  in the range 1,500 - 52,000 ADU above bias.  (Can get from countrate *
    #  exptime).

    # Do something...
    print(f"\nValidating {flat_filter} in validate_flat_table():")
    print(
        f"Mean: {np.mean(subtable['frame_avg']):.2f}  "
        f"Median:  {np.median(subtable['frame_med']):.2f}  "
        f"Stddev:  {np.mean(subtable['frame_std']):.2f}"
    )

    # Find the mean quadric surface for this set of flats
    # quadsurf = np.mean(np.asarray(subtable['quadsurf']), axis=0)
    # print(quadsurf)

    return subtable


def validate_skyf_table(skyf_meta):
    """validate_dark_table Analyze and validate the sky flat frame metadata table

    NOTE: Not yet implemented
    """
    return skyf_meta


# Helper Functions (Alphabetical) ============================================#
def base_metadata_dict(hdr, data, quadsurf, crop=100):
    """base_metadata_dict Create the basic metadata dictionary

    [extended_summary]

    Parameters
    ----------
    hdr : `astropy.io.fits.Header`
        FITS header for this frame
    data : `numpy.ndarray` or `astropy.nddata.CCDData`
        FITS image data for this frame
    crop : `int`, optional
        Size of the border around the edge of the frame to crop off
        [Default: 100]

    Returns
    -------
    `dict`
        The base metadata dictionary
    """
    # Make things easier by creating a slice for cropping
    allslice = np.s_[:, :]
    cropslice = np.s_[crop:-crop, crop:-crop]
    human_readable = utils.compute_human_readable_surface(quadsurf)
    human_readable.pop("typ")
    shape = (hdr["naxis1"], hdr["naxis2"])

    # TODO: Add error checking here to keep InfluxDB happy
    metadict = {
        "dateobs": f"{hdr['DATE-OBS'].strip()}",
        "instrument": f"{hdr['INSTRUME'].strip()}",
        "frametype": f"{hdr['OBSTYPE'].strip()}",
        "obserno": int(hdr["OBSERNO"]),
        "filename": f"{hdr['SHORT_FN'].strip()}",
        "binning": "x".join(hdr["CCDSUM"].split()),
        "filter": f"{hdr['FILTERS'].strip()}",
        "numamp": int(hdr["NUMAMP"]),
        "ampid": f"{hdr['AMPID'].strip()}",
        "exptime": float(hdr["EXPTIME"]),
        "mnttemp": float(hdr["MNTTEMP"]),
        "tempamb": float(hdr["TEMPAMB"]),
        "cropsize": int(crop),
    }
    for name, the_slice in zip(["frame", "crop"], [allslice, cropslice]):
        metadict[f"{name}_avg"] = np.mean(data[the_slice])
        metadict[f"{name}_med"] = np.ma.median(data[the_slice])
        metadict[f"{name}_std"] = np.std(data[the_slice])
    for key, val in human_readable.items():
        metadict[f"qs_{key}"] = val
    lin_flat, quad_flat = utils.compute_flatness(
        human_readable, shape, metadict["crop_std"]
    )
    metadict["lin_flat"] = lin_flat
    metadict["quad_flat"] = quad_flat

    # for i, m in enumerate(['b','x','y','xx','yy','xy']):
    #     metadict[f"qs_{m}"] = quadsurf[i]

    return metadict


def check_processing_ifc(ifc, binning):
    """check_processing_ifc Check the IFC being processed

    This is a DRY block, used in both process_bias and process_flats.  It
    does the various checks for existance of files, and making sure binning
    is uniform and FULL FRAME.

    Parameters
    ----------
    ifc : `ccdproc.ImageFileCollection`
        The ImageFileCollection to check
    binning : `str`
        The binning to use for this routine

    Returns
    -------
    `ccdproc.ImageFileCollection`
        Filtered ImageFileCollection, ready for processing

    Raises
    ------
    InputError
        Raised if the binning is not set.
    """
    # Error checking for binning
    if not binning:
        raise utils.InputError("Binning not set.")

    # If IFC is empty already, just return it
    if not ifc.files:
        return ifc

    # Double-check that we're processing FULL FRAMEs of identical binning only
    return ifc.filter(ccdsum=binning, subarrno=0)
