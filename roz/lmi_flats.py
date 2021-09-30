# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Analyze LMI Flat Field Frames for 1 Night

Further description.
"""

# Built-In Libraries
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


# List of LMI Filters
LMI_FILTERS = ['U', 'B', 'V', 'R', 'I',
               'SDSS-U', 'SDSS-G', 'SDSS-R', 'SDSS-I', 'SDSS-Z',
               'VR', 'YISH', 'OIII', 'HALPHAON', 'HALPHAOFF',
               'WR-WC', 'WR-WN', 'WR-CT',
               'UC','BC','GC','RC','C2','C3','CN','CO+','H2O+','OH','NH']

# Silence Superflous AstroPy Warnings
warnings.simplefilter('ignore', AstropyWarning)


# Create an error class to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


def collect_flats(directory, mem_limit=8.192e9):
    """collect_flats Collect the flat field frames

    [extended_summary]

    Parameters
    ----------
    directory : `str`
        The directory to scan for flats
     mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]

    Raises
    ------
    ValueError
        [description]
    """
    # Create an ImageFileCollection
    icl = ccdp.ImageFileCollection(directory)

    # Make bias collection for bias-subtraction
    bias_cl = icl.filter(obstype='bias')
    bias_frame = None

    for flat_type in ['DOME FLAT', 'SKY FLAT']:
        # Grab all of the flats of this type
        flat_cl = icl.filter(obstype=flat_type)

        # If no files, move along, move along
        if not flat_cl.files:
            continue

        # Get the complete list of binnings used
        bin_list = icl.values('ccdsum', unique=True)
        if len(bin_list) > 1:
            raise ValueError("More than one binning exists in this directory!")

        # Before doing stuff with the flats, make sure we have a bias_frame
        if bias_frame is None:
            bias_frame, bias_meta = combine_bias_frames(bias_cl, 
                                                        binning=bin_list[0],
                                                        mem_limit=mem_limit)

        # Retreive the list of unique filters for this set
        flat_filt = sorted(flat_cl.values('filters', unique=True))

        # I want to go through the filters in the same order as LMI_FILTERS...
        for lmi_filt in LMI_FILTERS:
            # If this filter wasn't used, move along, move along
            if lmi_filt not in flat_filt:
                continue
            
            print(lmi_filt)





def combine_bias_frames(bias_cl, binning=None, debug=True, mem_limit=8.192e9):
    """combine_bias_frames Combine available bias frames

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
    bias_cl = bias_cl.filter(obstype='bias', ccdsum=binning)

    bias_ccds = []
    bias_temp = []
    # Loop through files
    for ccd in bias_cl.ccds(ccdsum=binning, bitpix=16, imagetyp='bias'):

        hdr = ccd.header
        bias_data = ccd.data[slice_from_string(hdr['TRIMSEC'], fits_convention=True)]

        # For posterity, gather the mount temperature and mean bias level
        bias_temp.append({'utdate': hdr['DATE-OBS'].split('T')[0],
                          'utcstart': hdr['UTCSTART'],
                          'binning': hdr['CCDSUM'],
                          'mnttemp': hdr['MNTTEMP'],
                          'tempamb': hdr['TEMPAMB'],
                          'biaslev': np.mean(bias_data)})

        # Fit the overscan section, subtract it, then trim the image
        # Append this to a list
        bias_ccds.append(trim_oscan(ccd, ccd.header['BIASSEC'],
                                    ccd.header['TRIMSEC']))

    if debug:
        print("Doing median combine of biases now...")

    return ccdp.combine(bias_ccds, method='median', sigma_clip=True,
                             sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                             sigma_clip_func=np.ma.median, mem_limit=mem_limit,
                             sigma_clip_dev_func=mad_std), Table(bias_temp)


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



#=============================================================================#
def main(args):
    """
    This is the main body function.
    """


if __name__ == "__main__":
    import sys
    main(sys.argv)
