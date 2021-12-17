# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 08-Oct-2021
#
#  @author: tbowers

"""Make graphs desired for inclusion on the Confluence page

This module is part of the Roz package, written at Lowell Observatory.

This module will eventually create graphs for investigative purposes and/or
for upload to Confluence.  Alternatively, what could be produced here could
also be produced by the Grafana interface with the InfluxDB database that will
house the actual data from Roz.

This module primarily trades in... hope?
"""

# Built-In Libraries
import os
from pathlib import Path

# 3rd Party Libraries
from astropy.nddata import CCDData
from astropy.visualization import AsymmetricPercentileInterval
import matplotlib.pyplot as plt

# Internal Imports
from .utils import ROZ_THUMB


def plot_lmi_bias_temp():
    """plot_lmi_bias_temp [summary]

    [extended_summary]
    """


def make_png_thumbnail(img_fn, inst_flags, latest=True):
    """make_png_thumbnail Make PNG thumbnails of calibration frames

    These thumbnails will be uploaded to the Confluence page and will be
    linked to from the table created in Roz and uploaded.

    Parameters
    ----------
    img_fn : `str` or `pathlib.Path`
        The filename or path to the image for which a PNG will be created.
    inst_flags : `dict`
        The instrument flags dictionary.
    latest : `bool`, optional
        Label this image as a 'Latest' image rather than a 'Nominal' image.
        [Default: True]

    Returns
    -------
    `str`
        The filename (without path) of the PNG created.
    """
    # Read in the image, with error checking
    try:
        ccd = CCDData.read(img_fn)
    except Exception as exception:
        print(f"Could not open {img_fn} because of {exception}.")
        return None

    # Since we use the filename (sans path) in the graphic title...
    if isinstance(img_fn, str):
        img_fn = img_fn.split(os.path.sep)[-1]
    elif isinstance(img_fn, Path):
        img_fn = img_fn.name

    # Construct the output filename from the image header
    hdr = ccd.header
    png_fn = [hdr['INSTRUME'].lower()]
    # TODO: Not strictly correct, if we want this routine to also make
    #       thumbnails of bais frames... needs thought.  For now, though...
    png_fn.append(filt := hdr['FILTERS'] if inst_flags['get_flats'] else '')
    png_fn.append(hdr['DATE-OBS'].split('T')[0].replace('-',''))
    png_fn.append(f"{hdr['OBSERNO']:04d}")
    png_fn.append('png')
    png_fn = '.'.join(png_fn)
    print(f"This is the PNG filename!  {png_fn}")

    # Set up the plot environment
    _, ax = plt.subplots(figsize=(5,5.2))
    tsz=10

    # Plotting percentile limits -- convert to image intensity limits
    vmin, vmax = get_image_intensity_limits(ccd)

    # Show the data on the plot, using the limits computed above
    ax.imshow(ccd.data, vmin=vmin, vmax=vmax, origin='lower', cmap='gist_gray')

    # Set the title and don't draw any axes
    title = ['*Latest*' if latest else '*Nominal*', hdr['INSTRUME'].upper(),
             hdr['OBSTYPE'], filt, hdr['DATE-OBS'].split('T')[0], img_fn]
    ax.set_title('   '.join(title), y=-0.00, pad=-14, fontsize=tsz)
    ax.axis('off')

    # Clean up the plot and save
    plt.tight_layout()
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99)
    plt.savefig(ROZ_THUMB.joinpath(png_fn))

    # Return the filename we just saved
    return png_fn


def get_image_intensity_limits(ccd):
    """get_image_intensity_limits Return appropriate plot intensity limits

    Compute appropraite intensity ranges for plotting images based on the
    FITS header keyword 'OBSTYPE'.

    Parameters
    ----------
    image : `astropy.nddata.CCDData`
        The CCDData object for a frame

    Returns
    -------
    `float`, `float`
        The minimum and maximum values in the data image that correspond to
        the percentiles assigned by `OBSTYPE`.
    """
    # Get the image type from the FITS header, and select the percentile range
    if ccd.header['OBSTYPE'] == 'OBJECT':
        pmin, pmax = 25, 99.75
    elif ccd.header['OBSTYPE'] in ['DOME FLAT', 'SKY FLAT']:
        pmin, pmax = 3, 99
    elif ccd.header['OBSTYPE'] == 'BIAS':
        pmin, pmax = 5, 95
    else:
        pmin, pmax = 0, 100

    # Compute the iterval and return
    interval = AsymmetricPercentileInterval(pmin, pmax, n_samples=10000)
    return interval.get_limits(ccd.data)
