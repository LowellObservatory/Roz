# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 05-Oct-2022
#
#  @author: tbowers

"""Module containing toolbox utilities for processing All-Sky Images

This module is part of the Roz package, written at Lowell Observatory.

Images from the All-Sky Cameras require aditional specialized processing in
order to make use of the data for sky quality monitoring purposes.  The
routines here are based on algorithms from the `kpno-allsky package
<https://github.com/dylanagreen/kpno-allsky>`_.

This module primarily trades in, um, stuff?

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

# Built-In Libraries
import os
import pathlib
import warnings

# 3rd Party Libraries
import astropy.coordinates
import astropy.convolution
import astropy.io.fits
import astropy.nddata
import astropy.stats
import astropy.table
import astropy.time
import astropy.units as u
import astropy.visualization
import astropy.wcs
import ccdproc
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import photutils.aperture
import photutils.background
import photutils.detection
import scipy.ndimage
import scipy.optimize
from tqdm import tqdm


# Internal Imports
from roz import graphics_maker
from roz import msgs
from roz import utils

# Set API Components
# __all__ = ['generate_mask','gener']

# Module Constants
LDT_ASC = {
    "xcen": 684,
    "ycen": 484.5,
    "mrad": 505,
    "earthloc": astropy.coordinates.EarthLocation.of_site("DCT"),
    "a0": -86.2,
    "F": 2.33,
    "R": 800,
}
SIXTEEN_BIT = 2**16 - 1
DDIR = pathlib.Path("/Users/tbowers/sandbox/")

# Silence Superflous AstroPy FITS Header Warnings
warnings.simplefilter("ignore", astropy.wcs.FITSFixedWarning)


def generate_radius_mask(input_image: np.ndarray):
    """generate_radius_mask _summary_

    _extended_summary_

    Parameters
    ----------
    input_image : `numpy.ndarray`_
        The input image for which the mask will be generated.

    Returns
    -------
    `numpy.ndarray`_
        The BAD PIXEL MASK for radius from the center of the all-sky image,
        of the same shape as ``input_image``.  Good pixels have a value of
        0, and bad pixels have a value of 1.
    """
    # Construct the arrays for doing the matrix magic -- origin in center
    n_y, n_x = input_image.shape
    x_arr = np.tile(np.arange(n_x), (n_y, 1)) - LDT_ASC["xcen"]
    y_arr = np.transpose(np.tile(np.arange(n_y), (n_x, 1))) - LDT_ASC["ycen"]

    # Return the mask identifying pixels outside the specified radius
    return (np.hypot(x_arr, y_arr) > LDT_ASC["mrad"]).astype(int)


def generate_hotpixel_mask(icl: ccdproc.ImageFileCollection, hot_lim=0.25, f_dark=0.75):
    """Generate a hot pixel mask for a night's worth of AllSky Data

    After playing around with this, it seems the best criteria for identifying
    hot pixels in a night's ASC data is to find all pixels whose value >= 25%
    of saturation in 3/4 of the 60-second (i.e., DARK) frames.

    For the LDT all-sky camera, this should identify something in the range of
    50 hot pixels.

    Returns
    -------
    `numpy.ndarray`_
        The BAD PIXEL MASK of identified hot pixels, of the same shape as the
        images in ``icl``.  Good pixels have a value of 0, and bad pixels
        have a value of 1.
    """

    # Show progress bar for processing ASC frames
    msgs.info("Processing frames to generate a hot pixel mask...")
    progress_bar = tqdm(
        total=len(icl.files), unit="frame", unit_scale=False, colour="#eab676"
    )

    # Set empty return item and DARK counter
    hotpix, n_dark = None, 0

    for ccd in icl.ccds(ccd_kwargs={"unit": u.adu}):

        if float(ccd.header["exptime"]) < 60:
            continue

        # LR flip the image and convert to float
        ccd.data = np.fliplr(ccd.data.astype(float))

        # Find the "HOT PIXELS" in this image, and add them to the mask
        hpm = (ccd.data > hot_lim * SIXTEEN_BIT).astype(int)
        hotpix = hpm if hotpix is None else hotpix + hpm

        n_dark += 1
        progress_bar.update(1)

    progress_bar.close()

    # Identify pixels marked as HOT in `f_dark` of the DARK frames
    return (hotpix >= f_dark * n_dark).astype(int)


def make_animation(icl: ccdproc.ImageFileCollection):
    """Create an animation from an ImageFileCollection of ASC frames

    _extended_summary_

    Parameters
    ----------
    icl : `ccdproc.ImageFileCollection`_
        The ImageFileCollection from which to make the animation.
    """
    # Get data directory, for later use
    ddir = pathlib.Path(icl.location)

    # Get masks and median flat
    hotpix_mask = generate_hotpixel_mask(icl)
    radius_mask = generate_radius_mask(next(icl.ccds(ccd_kwargs={"unit": u.adu})).data)
    median_flat = astropy.nddata.CCDData.read(
        utils.Paths.data.joinpath("ldt_asc_median_flat.fits"), unit=u.adu
    )

    # Show progress bar for processing ASC frames
    msgs.info("Processing frames...")
    progress_bar = tqdm(
        total=len(icl.files), unit="frame", unit_scale=False, colour="#eab676"
    )

    # Loop over the frames in the collection
    for ccd in icl.ccds(ccd_kwargs={"unit": u.adu}):

        # Turn the observation time into an object for ALT-AZ conversion
        obstime = astropy.time.Time(utils.scrub_isot_dateobs(ccd.header["DATE-OBS"]))

        # LR flip the image and convert to float
        ccd.data = np.fliplr(ccd.data.astype(float))

        # Mask Hot pixels by NaN -> interpolate over NaN
        ccd.data[hotpix_mask.astype(bool)] = np.nan
        ccd.data = astropy.convolution.interpolate_replace_nans(
            ccd.data, astropy.convolution.Gaussian2DKernel(x_stddev=1)
        )

        # Divide by the normalized median flat:
        ccd.data /= median_flat.data

        # Mask by radius
        ccd.data[radius_mask.astype(bool)] = np.nan

        #
        # Do some other stuff here, maybe?
        #

        # Set up the plotting environment
        _, axis = plt.subplots(figsize=(16, 12))

        # Show the data on the plot, using ZScale
        interval = astropy.visualization.ZScaleInterval(nsamples=10000)
        vmin, vmax = interval.get_limits(ccd.data)
        axis.imshow(ccd.data, vmin=vmin, vmax=vmax, origin="lower")
        axis.axis("off")

        # Add interesting things to the plot!
        n_points = 500
        # Start with Alt/Az lines
        for alt_line in np.arange(0, 90, 20):
            altaz = astropy.coordinates.AltAz(
                alt=np.full(n_points, alt_line) * u.deg,
                az=np.linspace(0, 360, n_points) * u.deg,
            )
            xpl, ypl = skycoord2xy(altaz, obstime, LDT_ASC["earthloc"])
            axis.plot(xpl, ypl, "-", color="black", alpha=0.25)
        for az_line in np.arange(0, 360, 30):
            altaz = astropy.coordinates.AltAz(
                alt=np.linspace(0, 90, n_points) * u.deg,
                az=np.full(n_points, az_line) * u.deg,
            )
            xpl, ypl = skycoord2xy(altaz, obstime, LDT_ASC["earthloc"])
            axis.plot(xpl, ypl, "-", color="black", alpha=0.25)

        # Add RA lines:
        for ra_line in np.arange(0, 360, 30):
            sc = astropy.coordinates.SkyCoord(
                ra=np.full(n_points, ra_line) * u.deg,
                dec=np.linspace(-90, 90, n_points) * u.deg,
                frame="icrs",
            )
            xpl, ypl = skycoord2xy(sc, obstime, LDT_ASC["earthloc"])
            axis.plot(xpl, ypl, "-", color="white", alpha=0.5)

        # Add DEC lines:
        for dec_line in np.arange(-80, 81, 20):
            sc = astropy.coordinates.SkyCoord(
                ra=np.linspace(0, 360, n_points) * u.deg,
                dec=np.full(n_points, dec_line) * u.deg,
                frame="icrs",
            )
            xpl, ypl = skycoord2xy(sc, obstime, LDT_ASC["earthloc"])
            axis.plot(xpl, ypl, "-", color="white", alpha=0.5)

        # Add Ecliptic:
        sc = astropy.coordinates.SkyCoord(
            lon=np.linspace(0, 360, n_points) * u.deg,
            lat=np.full(n_points, 0) * u.deg,
            frame="geocentricmeanecliptic",
        )
        xpl, ypl = skycoord2xy(sc, obstime, LDT_ASC["earthloc"])
        axis.plot(xpl, ypl, "-", color="orange", alpha=0.5)

        # Add Galactic Plane:
        sc = astropy.coordinates.SkyCoord(
            l=np.linspace(0, 360, n_points) * u.deg,
            b=np.full(n_points, 0) * u.deg,
            frame="galactic",
        )
        xpl, ypl = skycoord2xy(sc, obstime, LDT_ASC["earthloc"])
        axis.plot(xpl, ypl, "-", color="pink", alpha=0.5)

        # Finish up
        axis.set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(ddir.joinpath(f"asc_{ccd.header['seqnum']:05d}.png"))
        plt.close()
        progress_bar.update(1)

    progress_bar.close()

    # Create the MP4 animation; the operable command should be like:
    # ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
    #   -c:v libx264 -pix_fmt yuv420p out.mp4
    msgs.info("Creating the MP4 animation for this night...")
    stream = ffmpeg.input(
        str(ddir.joinpath("asc_*.png")), framerate=30, pattern_type="glob"
    )
    stream = ffmpeg.output(
        stream,
        str(ddir.joinpath("asc_night.mp4")),
        pix_fmt="yuv420p",
        vcodec="libx264",
    )
    msgs.info(f"{ffmpeg.compile(stream)}")
    ffmpeg.run(stream, overwrite_output=True)

    # Open the animation
    os.system(f"/usr/bin/open {ddir}/asc_night.mp4")


def skycoord2xy(
    coords,
    obstime: astropy.time.Time,
    location: astropy.coordinates.EarthLocation,
):
    """Convert SkyCoord coordinates into CCD positions

    Use the simplified 5-parameter model (plus location & obstime) to
    convert SkyCoord coordinates into location on the CCD.

    Parameters
    ----------
    coords : `astropy.coordinates.SkyCoord`_ or `astropy.coordinates.AltAz`_
        Input SkyCoord or AltAz object
    obstime : `astropy.time.Time`_
        The time of the observation, for conversion from ``SkyCoord`` -> ``AltAz``
    location : `astropy.coordinates.EarthLocation`_
        The location of the observation, for conversion from ``SkyCoord`` -> ``AltAz``

    Returns
    -------
    tuple
        (xcat, ycat) positions for the objects provided in the catalog
    """
    # Check the input type
    if isinstance(coords, astropy.coordinates.SkyCoord):
        altaz = coords.transform_to(
            astropy.coordinates.AltAz(obstime=obstime, location=location)
        )
    elif isinstance(coords, astropy.coordinates.AltAz):
        altaz = coords
    else:
        msgs.error(f"coords type {type(coords)} not recognized.")

    # Pull the zenith distance and azimuth variables separately
    zcat = 90.0 * u.deg - altaz.alt
    acat = altaz.az

    # Set anything below the horizon to NaN; will propagate
    zcat[zcat > 90 * u.deg] = np.nan

    # Compute the CCD catalog positions
    rcat = LDT_ASC["R"] * np.sin(np.radians(zcat) / LDT_ASC["F"])
    xcat = LDT_ASC["xcen"] + rcat * np.cos(np.radians(acat - LDT_ASC["a0"] * u.deg))
    ycat = LDT_ASC["ycen"] + rcat * np.sin(np.radians(acat - LDT_ASC["a0"] * u.deg))

    # Return the CCD positions of the catalog objects
    return xcat, ycat


def main():
    """Typical Driver Function

    _extended_summary_
    """
    # # First, compute radius of horizon:
    # n_points = 100
    # altaz = astropy.coordinates.AltAz(
    #     alt=np.full(n_points, 0) * u.deg,
    #     az=np.linspace(0, 360, n_points) * u.deg,
    # )
    # # Pull the zenith distance and azimuth variables separately
    # zcat = 90.0 * u.deg - altaz.alt
    # acat = altaz.az

    # # Set anything below the horizon to NaN; will propagate
    # zcat[zcat > 90 * u.deg] = np.nan

    # # Compute the CCD catalog positions
    # rcat = LDT_ASC["R"] * np.sin(np.radians(zcat) / LDT_ASC["F"])

    # print(rcat)

    msgs.info("Reading in the ImageFileCollection...")
    icl = ccdproc.ImageFileCollection(DDIR, glob_include="TARGET*.fit")
    make_animation(icl)


# =============================================================================#
# Cruft... just, cruft
def generate_mask(icl: ccdproc.ImageFileCollection, forcenew=False):
    """Generate a complete mask for Lowell All-Sky Images

    The mask generated by this function is a BAD PIXEL MASK, in that a value
    of 0 indicates the pixel should remain visible, and masked pixels are
    marked by bitwise value:

    1. Hot pixels
    2. Horizon objects
    3. Pixels outside the circular All-Sky Image.

    Parameters
    ----------
    icl : `ccdproc.ImageFileCollection`_
        The Image File Collection for this night, used to generate masks
    forcenew : bool, optional
        Force the generation of a mask completely from scratch rather than use
        a saved mask?  (Default: False)

    Returns
    -------
    `numpy.ndarray`_
        The mask array where 0 represents pixels that are to be masked and 1
        represents pixels that should remain visible.

    See Also
    --------
    generate_object_mask : Used by generate_mask to generate the hot pixel
                    and horizon masks.

    Notes
    -----
    generate_mask calls generate_object_mask, which requires there to be
    median images in Images/mask/ but also additionally requires an image
    named Ignore.png in Images/ that deliniates the horizon objects to be
    ignored. These images can be downloaded from the kpno-allsky github.
    """

    msgs.info("Generating masks...")

    # The other two masks
    mask1 = generate_hotpixel_mask(icl)
    # mask2 = generate_object_mask(forcenew)
    mask3 = generate_radius_mask(next(icl.ccds(ccd_kwargs={"unit": u.adu})).data)

    return mask1 | mask3


def generate_object_mask(forcenew=False):
    """Generate a mask for KPNO images.

    Generates a masking array for KPNO images that masks out not only hot
    pixels, but also the horizon objects.

    Parameters
    ----------
    forcenew : bool, optional
        Whether or not this method should load a previously saved mask or if it
        should generate it completely from scratch.

    Returns
    -------
    `numpy.ndarray`_
        The mask array where 1 represents pixels that are to be masked and 0
        represents pixels that should remain visible.

    See Also
    --------
    generate_clean_mask : Used by generate_mask to generate the hot pixel
                          mask.

    Notes
    -----
    generate_mask requires there to be median images in Images/mask/ but also
    additionally requires an image named Ignore.png in Images/ that
    deliniates the horizon objects to be ignored.
    These images can be downloaded from the kpno-allsky github or may be
    generated by median.median_all_date and moved.

    """
    # center = (256, 252)

    # # Read in the ignore image.
    # # I read this in first to make sure the Mask.png is the correct dimensions.
    # ignore_loc = os.path.join(os.path.dirname(__file__), *["Images", "Ignore.png"])
    # ignore = np.asarray(PIL.Image.open(ignore_loc).convert("RGB"))

    # # If we"ve already generated and saved a mask, load that one.
    # # This speeds up code execution by a lot, otherwise we loop through 512x512
    # # pixels 6 times! With this we don"t have to even do it once, we just load
    # # and go.
    # maskloc = os.path.join(os.path.dirname(__file__), *["Images", "Mask.png"])
    # if os.path.isfile(maskloc) and not forcenew:
    #     mask = np.asarray(PIL.Image.open(maskloc).convert("L"))
    #     # Converts the 255 bit loaded image to binary 1-0 image.
    #     mask = np.where(mask == 255, 1, 0)

    #     # Have to compare these two separately, since ignore has a third color
    #     # dimensions and mask.shape == ignore.shape would therefore always be
    #     # False.
    #     if mask.shape[0] == ignore.shape[0] and mask.shape[1] == ignore.shape[1]:
    #         return mask

    # # Get the "clean" mask, i.e. the pixels only ignore mask.
    # mask = generate_hotpixel_mask()

    # hyp = np.hypot
    # for y in range(0, ignore.shape[1]):
    #     for x in range(0, ignore.shape[0]):
    #         x1 = x - center[0]
    #         y1 = center[1] - y
    #         r = hyp(x1, y1)

    #         # Ignore horizon objects (which have been painted pink)
    #         # Only want the horizon objects actually in the circle.
    #         # Avoids unnecessary pixels.
    #         if r < 242 and np.array_equal(ignore[y, x], [244, 66, 235]):
    #             mask[y, x] = 1

    # # If we've made a new mask, save it so we can skip the above steps later.
    # save_mask(mask)

    # return mask


def save_mask(mask):
    """Save a masking image.

    Parameters
    ----------
    mask : `numpy.ndarray`_
        The mask to save.

    See Also
    --------
    image.save_image : Save an image.

    """


def test_animations():
    """Test creating animations

    _extended_summary_
    """
    msgs.info("Reading in the ImageFileCollection...")
    icl = ccdproc.ImageFileCollection(DDIR, glob_include="TARGET*.fit")

    # Show progress bar for processing ASC frames
    msgs.info("Processing frames...")
    progress_bar = tqdm(
        total=len(icl.files), unit="frame", unit_scale=False, colour="#eab676"
    )

    sobel_mask = None

    for ccd in icl.ccds(ccd_kwargs={"unit": u.adu}):

        # Set FITS keyword for image scaling
        ccd.header["OBSTYPE"] = "DOME FLAT"  # "OBJECT"

        # LR flip the image and convert to float
        ccd.data = np.fliplr(ccd.data.astype(float))

        # Generate the mask(s)

        # Convert the image into a masked array
        m_idx = generate_radius_mask(ccd.data) != 0
        # ccd.data = np.ma.array(ccd.data, mask=m_idx)
        ccd.data = scipy.ndimage.sobel(ccd.data)

        # Add to the Sobel Mask
        sobel_mask = ccd.data if sobel_mask is None else sobel_mask + ccd.data

        # Set up the plotting environment
        _, axis = plt.subplots(figsize=(16, 12))

        # Get image limits
        vmin, vmax = graphics_maker.get_image_intensity_limits(ccd)

        # Show the data on the plot, using the limits computed above
        axis.imshow(ccd.data, vmin=vmin, vmax=vmax, origin="lower")
        axis.axis("off")

        # # Show the pixel histogram on the plot, marking the limits from above
        # axis.hist(ccd.data.flatten(), bins=100, range=(0,65535), histtype='step')
        # axis.vlines([vmin, vmax], 0, 1, transform=axis.get_xaxis_transform())
        # axis.set_yscale('log')
        # axis.set_ylim(0.6, 1e6)

        # Finish up
        plt.tight_layout()
        plt.savefig(DDIR.joinpath(f"asc_{ccd.header['seqnum']:05d}.png"))
        plt.close()
        progress_bar.update(1)

    progress_bar.close()

    # Create the MP4 animation; the operable command should be like:
    # ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
    #   -c:v libx264 -pix_fmt yuv420p out.mp4
    stream = ffmpeg.input(
        str(DDIR.joinpath("asc_*.png")), framerate=30, pattern_type="glob"
    )
    stream = ffmpeg.output(
        stream,
        str(DDIR.joinpath("asc_night.mp4")),
        pix_fmt="yuv420p",
        vcodec="libx264",
    )
    msgs.info(f"{ffmpeg.compile(stream)}")
    ffmpeg.run(stream, overwrite_output=True)

    # Open the animation
    os.system(f"/usr/bin/open {DDIR}/asc_night.mp4")

    # Set up the plotting environment for the final SOBEL mask
    sccd = astropy.nddata.CCDData(sobel_mask, unit=u.adu)
    sccd.write(DDIR.joinpath("sobel_sum.fits"), overwrite=True)

    _, axis = plt.subplots()

    # Compute the iterval
    pmin, pmax = 0, 100
    interval = astropy.visualization.AsymmetricPercentileInterval(
        pmin, pmax, n_samples=10000
    )
    vmin, vmax = interval.get_limits(sobel_mask)
    # Show the data on the plot, using the limits computed above
    axis.imshow(sobel_mask, vmin=vmin, vmax=vmax, origin="lower")
    axis.axis("off")
    # Finish up
    plt.tight_layout()
    plt.show()
    plt.close()


def test_hotpix(hot_lim):
    """Test finding hot pixels

    _extended_summary_
    """
    msgs.info("Reading in the ImageFileCollection...")
    icl = ccdproc.ImageFileCollection(DDIR, glob_include="TARGET*.fit")

    # Show progress bar for processing ASC frames
    msgs.info("Processing frames...")
    progress_bar = tqdm(
        total=len(icl.files), unit="frame", unit_scale=False, colour="#eab676"
    )

    hotpix = None
    i = 0
    for ccd in icl.ccds(ccd_kwargs={"unit": u.adu}):

        # Set FITS keyword for image scaling
        ccd.header["OBSTYPE"] = "DOME FLAT"  # "OBJECT"

        if float(ccd.header["exptime"]) < 60:
            continue

        # LR flip the image and convert to float
        ccd.data = np.fliplr(ccd.data.astype(float))
        hpm = (ccd.data > hot_lim * 65535).astype(int)

        hotpix = hpm if hotpix is None else hotpix + hpm

        i += 1
        progress_bar.update(1)

    progress_bar.close()
    msgs.info(f"Total Number of frames: {len(icl.files)}")
    msgs.info(f"Maximum value in hotpix: {hotpix.max()}")
    msgs.info(
        f"Number of pixels at the max value: {np.count_nonzero(hotpix[hotpix == hotpix.max()])}"
    )
    n_hot = []
    for maxval in (maxvals := np.arange(200, len(icl.files), 10)):
        msgs.info(
            f"Number of pixels exceeding {maxval} frames: "
            f"{np.count_nonzero(hotpix[hotpix > maxval])}"
        )
        n_hot.append(np.count_nonzero(hotpix[hotpix > maxval]))

    # Set up the plotting environment for the final SOBEL mask
    sccd = astropy.nddata.CCDData(hotpix, unit=u.adu)
    sccd.write(DDIR.joinpath("hotpix_sum.fits"), overwrite=True)

    # _, axis = plt.subplots()
    # # axis.hist(hotpix.flatten(), bins=np.arange(len(icl.files)) + 1, histtype="step")
    # # axis.set_yscale("log")
    # axis.imshow(hotpix, origin="lower", vmin=400, vmax=len(icl.files))
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    return maxvals, np.array(n_hot)


def run_test_hotpix():
    """run_test_hotpix _summary_

    _extended_summary_
    """
    res = {}

    for hot_lim in np.arange(0.2, 0.601, 0.05):
        msgs.info(f"\n  Testing HOT LIMIT ({hot_lim:.2f}) {hot_lim * 65535:.0f}...")
        res[f"{hot_lim:.2f}"] = test_hotpix(hot_lim)

    _, axis = plt.subplots()

    for k, v in res.items():
        axis.plot(v[0], v[1], "o-", label=k)

    axis.legend(loc="upper right")
    axis.set_xlabel("N (60s) Frames this pixel identified in")
    axis.set_ylabel("N pixles identified")
    axis.set_title("Hot Pixel Identification Criteria")
    plt.tight_layout()
    plt.savefig(DDIR.joinpath("hotpix_identification_criteria.png"))
    plt.savefig(DDIR.joinpath("hotpix_identification_criteria.pdf"))
    plt.close()


def test_masking():
    """test_masking _summary_

    _extended_summary_
    """
    msgs.info("Reading in the ImageFileCollection...")
    icl = ccdproc.ImageFileCollection(DDIR, glob_include="TARGET*.fit")

    hotpix = generate_hotpixel_mask(icl)
    radius = generate_radius_mask(next(icl.ccds(ccd_kwargs={"unit": u.adu})).data)

    # Show progress bar for processing ASC frames
    msgs.info("Processing frames...")
    progress_bar = tqdm(
        total=len(icl.files), unit="frame", unit_scale=False, colour="#eab676"
    )

    for ccd in icl.ccds(ccd_kwargs={"unit": u.adu}):

        # Set FITS keyword for image scaling
        ccd.header["OBSTYPE"] = "DOME FLAT"  # "OBJECT"

        # LR flip the image and convert to float
        ccd.data = np.fliplr(ccd.data.astype(float))

        # Mask hot pixels, then interpolate
        ccd.data[hotpix.astype(bool)] = np.nan
        ccd.data = astropy.convolution.interpolate_replace_nans(
            ccd.data, astropy.convolution.Gaussian2DKernel(x_stddev=1)
        )
        # Mask by radius
        ccd.data[radius.astype(bool)] = np.nan

        # Set up the plotting environment
        _, axis = plt.subplots(figsize=(16, 12))

        # Get image limits
        vmin, vmax = graphics_maker.get_image_intensity_limits(ccd)

        # Show the data on the plot, using the limits computed above
        axis.imshow(
            ccd.data, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest"
        )
        axis.axis("off")

        # # Show the pixel histogram on the plot, marking the limits from above
        # axis.hist(ccd.data.flatten(), bins=100, range=(0,65535), histtype='step')
        # axis.vlines([vmin, vmax], 0, 1, transform=axis.get_xaxis_transform())
        # axis.set_yscale('log')
        # axis.set_ylim(0.6, 1e6)

        # Finish up
        plt.tight_layout()
        plt.savefig(DDIR.joinpath(f"asc_{ccd.header['seqnum']:05d}.png"))
        plt.close()
        progress_bar.update(1)

        # if (i+1) % 20 == 0:
        #     break

    progress_bar.close()

    # Create the MP4 animation; the operable command should be like:
    # ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
    #   -c:v libx264 -pix_fmt yuv420p out.mp4
    stream = ffmpeg.input(
        str(DDIR.joinpath("asc_*.png")), framerate=30, pattern_type="glob"
    )
    stream = ffmpeg.output(
        stream,
        str(DDIR.joinpath("asc_night.mp4")),
        pix_fmt="yuv420p",
        vcodec="libx264",
    )
    msgs.info(f"{ffmpeg.compile(stream)}")
    ffmpeg.run(stream, overwrite_output=True)

    # Open the animation
    os.system(f"/usr/bin/open {DDIR}/asc_night.mp4")


def make_clean_sobel_map():
    """Make a clean Sobel map

    First mask and interpolate over hotpixels, then Sobel filter and build up
    a sum sobel map.
    """
    msgs.info("Reading in the ImageFileCollection...")
    icl = ccdproc.ImageFileCollection(DDIR, glob_include="TARGET*.fit")

    hp_mask = generate_hotpixel_mask(icl)
    msgs.bug(f"Number of marked HOT PIXELS: {np.sum(hp_mask)}")

    # Show progress bar for processing ASC frames
    msgs.info("Processing frames...")
    progress_bar = tqdm(
        total=len(icl.files), unit="frame", unit_scale=False, colour="#eab676"
    )

    sobel_mask = None

    for ccd in icl.ccds(ccd_kwargs={"unit": u.adu}):

        # Set FITS keyword for image scaling
        ccd.header["OBSTYPE"] = "DOME FLAT"  # "OBJECT"

        # LR flip the image and convert to float
        ccd.data = np.fliplr(ccd.data.astype(float))

        # Apply the hot pixel mask, and interpolate
        msgs.bug(
            f"Number of NaN pixels in unmasked image: {np.sum(np.isnan(ccd.data))}"
        )
        ccd.data[hp_mask.astype(bool)] = np.nan
        msgs.bug(f"Number of NaN pixels in masked image: {np.sum(np.isnan(ccd.data))}")

        ccd.data = astropy.convolution.interpolate_replace_nans(
            ccd.data, astropy.convolution.Gaussian2DKernel(x_stddev=1)
        )
        msgs.bug(
            f"Number of NaN pixels in interpolated image: {np.sum(np.isnan(ccd.data))}"
        )

        # Sobel filter the data
        ccd.data = scipy.ndimage.sobel(ccd.data)

        # Add to the Sobel Mask
        sobel_mask = ccd.data if sobel_mask is None else sobel_mask + ccd.data

        progress_bar.update(1)
    progress_bar.close()

    # Set up the plotting environment for the final SOBEL mask
    sccd = astropy.nddata.CCDData(sobel_mask, unit=u.adu)
    sccd.write(DDIR.joinpath("sobel_sum.fits"), overwrite=True)

    _, axis = plt.subplots()

    # Compute the iterval
    interval = astropy.visualization.ZScaleInterval(nsamples=10000)
    vmin, vmax = interval.get_limits(sobel_mask)
    # Show the data on the plot, using the limits computed above
    axis.imshow(sobel_mask, vmin=vmin, vmax=vmax, origin="lower")
    axis.axis("off")
    # Finish up
    plt.tight_layout()
    plt.show()
    plt.close()


def make_hpm_fits():
    """Create a Hot Pixel Mask for this data set

    _extended_summary_
    """
    msgs.info("Reading in the ImageFileCollection...")
    icl = ccdproc.ImageFileCollection(DDIR, glob_include="TARGET*.fit")

    hotpix = generate_hotpixel_mask(icl)

    # Set up the plotting environment for the final SOBEL mask
    sccd = astropy.nddata.CCDData(hotpix, unit=u.adu)
    sccd.write(DDIR.joinpath("hotpix_sum.fits"), overwrite=True)


def make_nightly_median_flat():
    """Build a nightly median flat

    _extended_summary_
    """
    msgs.info("Reading in the ImageFileCollection...")
    icl = ccdproc.ImageFileCollection(DDIR, glob_include="TARGET*.fit")
    hotpix = generate_hotpixel_mask(icl)

    # Show progress bar for processing ASC frames
    msgs.info("Processing frames...")
    progress_bar = tqdm(
        total=len(icl.files), unit="frame", unit_scale=False, colour="#eab676"
    )

    img_list = []

    for ccd in icl.ccds(ccd_kwargs={"unit": u.adu}):

        # Skip images with exptime less than a minute
        if ccd.header["EXPTIME"] < 60:
            progress_bar.update(1)
            continue

        # LR flip the image and convert to float
        ccd.data = np.fliplr(ccd.data.astype(float))

        # Mask hot pixels, then interpolate
        ccd.data[hotpix.astype(bool)] = np.nan
        ccd.data = astropy.convolution.interpolate_replace_nans(
            ccd.data, astropy.convolution.Gaussian2DKernel(x_stddev=1)
        )

        img_list.append(ccd)

        progress_bar.update(1)
    progress_bar.close()

    return ccdproc.combine(
        img_list,
        output_file=DDIR.joinpath("median_flat.fits"),
        method="median",
        mem_limit=8.192e9,
        ccd_kwargs={"unit": u.adu},
    )


def plot_medflat():
    """Plot the median flat stats

    _extended_summary_
    """
    med_flat = astropy.nddata.CCDData.read(DDIR.joinpath("median_flat.fits"))

    _, axis = plt.subplots()
    axis.hist(med_flat.data.flatten(), histtype="step", bins=100)

    axis.set_yscale("log")

    plt.tight_layout()
    plt.savefig(DDIR.joinpath("medflat_hist.pdf"))
    plt.savefig(DDIR.joinpath("medflat_hist.png"))
    plt.close()

    radius = generate_radius_mask(med_flat.data)

    comppix = med_flat.data.copy()
    comppix[radius.astype(bool)] = np.nan
    mean_value = np.nanmean(comppix)
    median_value = np.nanmedian(comppix)
    msgs.info(f"Mean value is: {mean_value}")
    msgs.info(f"Median value is: {median_value}")

    med_flat.data = med_flat.divide(mean_value)

    med_flat.write(DDIR.joinpath("median_flat.fits"), overwrite=True)


def find_stars_asc(find_algorithm=photutils.detection.DAOStarFinder):
    """Find stars in the ALL-SKY IMAGE

    _extended_summary_
    """

    def get_xcat_ycat(params, cat_coords=None):
        """Convert RA/Dec coordinates into CCD positions

        Use the simplified 5-parameter model (plus location & obstime) to
        convert catalog coordinates (J2000 RA/Dec) into location on the CCD.

        Parameters
        ----------
        params : :obj:`list` or :obj:`tuple`
            Model parameters to be used in the conversion

        Returns
        -------
        tuple
            (xcat, ycat) positions for the objects provided in the catalog
        """
        # Unpack the parameters
        a0, xc, yc, F, R = params

        # The `cat_coords` are AltAz coordinates already
        zcat = 90.0 * u.deg - cat_coords.alt
        acat = cat_coords.az

        # Set anything below the horizon to NaN; will propagate
        zcat[zcat > 90 * u.deg] = np.nan

        # Compute the CCD catalog positions
        rcat = R * np.sin(np.radians(zcat) / F)
        xcat = xc + rcat * np.cos(np.radians(acat - a0 * u.deg))
        ycat = yc + rcat * np.sin(np.radians(acat - a0 * u.deg))

        # Return the CCD positions of the catalog objects
        return xcat, ycat

    def lsq_minfunc(params):
        """Minimization function

        _extended_summary_

        Parameters
        ----------
        params : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        xcat, ycat = get_xcat_ycat(params, cat_coords=sc_cat)

        return np.hypot(
            xcat - lim_sources["xcentroid"], ycat - lim_sources["ycentroid"]
        )

    msgs.info("Reading in the ImageFileCollection...")
    icl = ccdproc.ImageFileCollection(DDIR, glob_include="TARGET*.fit")

    # Load in the requisite files
    hpm = astropy.nddata.CCDData.read(DDIR.joinpath("hotpix_sum.fits"), unit=u.adu)
    mfl = astropy.nddata.CCDData.read(DDIR.joinpath("median_flat.fits"), unit=u.adu)

    msgs.info("Reading in the Tycho-2 Catalog to match stars")
    tycho2 = astropy.table.Table.read(utils.Paths.data.joinpath("hipparcos_vmag6.fits"))

    # The array to place fit parameters into
    par_array = []

    # Simple Optics Model Parameters -- Starting Point
    a0, xc, yc, F, R = -90.0 + 4, LDT_ASC["xcen"], LDT_ASC["ycen"], 1.9, 667.0

    # LOOP!!!!!!!
    for ccd in icl.ccds(exptime=60, ccd_kwargs={"unit": u.adu}):

        msgs.info("")
        msgs.info(f"Processing sequence number: {ccd.header['seqnum']}")

        # Turn the time into an object
        obstime = astropy.time.Time(utils.scrub_isot_dateobs(ccd.header["DATE-OBS"]))

        # LR flip the image and convert to float
        ccd.data = np.fliplr(ccd.data.astype(float))

        # Mask Hot pixels by NaN -> interpolate over NaN
        ccd.data[hpm.data.astype(bool)] = np.nan
        ccd.data = astropy.convolution.interpolate_replace_nans(
            ccd.data, astropy.convolution.Gaussian2DKernel(x_stddev=1)
        )

        # Divide by the normalized median flat:
        ccd = ccd.divide(mfl.data)

        # Mask by radius
        radius = generate_radius_mask(ccd.data)
        ccd.data[radius.astype(bool)] = np.nan

        # =======================#
        # At this point, we have the flattened, cleaned circular image.
        # It is ready for star finding, then matching to the Hipparcos Catalog
        # Make some background estimates, do some subtraction, and find stars
        _, _, std = astropy.stats.sigma_clipped_stats(ccd.data, sigma=3.0)
        bkg = photutils.background.Background2D(
            ccd.data,
            (50, 50),
            filter_size=(3, 3),
            sigma_clip=astropy.stats.SigmaClip(sigma=3.0),
            bkg_estimator=photutils.background.MedianBackground(),
        )
        starfind = find_algorithm(fwhm=1.5, threshold=5.0 * std, brightest=200)
        sources = starfind(ccd.data - bkg.background)

        # Turn these into plotable positions for the individual frames
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        apertures = photutils.aperture.CircularAperture(positions, r=4.0)

        # Add the planets to Hipparcos-2 table -- use generic built-in ephemerides
        with astropy.coordinates.solar_system_ephemeris.set("builtin"):
            mer = astropy.coordinates.get_body("mercury", obstime, LDT_ASC["earthloc"])
            ven = astropy.coordinates.get_body("venus", obstime, LDT_ASC["earthloc"])
            mar = astropy.coordinates.get_body("mars", obstime, LDT_ASC["earthloc"])
            jup = astropy.coordinates.get_body("jupiter", obstime, LDT_ASC["earthloc"])
            sat = astropy.coordinates.get_body("saturn", obstime, LDT_ASC["earthloc"])
        planet_table = astropy.table.Table(
            [
                [mer.ra, ven.ra, mar.ra, jup.ra, sat.ra],
                [mer.dec, ven.dec, mar.dec, jup.dec, sat.dec],
                [0] * 5,
                [0] * 5,
                [-1, -4, -1, -3, 1],
                [0] * 5,
                [0] * 5,
                [0] * 5,
                [""] * 5,
                ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"],
            ],
            names=tycho2.colnames,
        )
        cat_table = astropy.table.vstack([tycho2, planet_table])
        # Sort on visual magnitude
        cat_table.sort("Vmag")
        cat_table.pprint()

        # Convert Hipparcos table columns into SkyCoord --> AltAz
        tyc2_coords = astropy.coordinates.SkyCoord(
            ra=cat_table["ra"], dec=cat_table["dec"], frame="icrs"
        ).transform_to(
            astropy.coordinates.AltAz(obstime=obstime, location=LDT_ASC["earthloc"])
        )
        msgs.info(f"Obstime: {obstime}")
        msgs.info(f"Location: {LDT_ASC['earthloc'].geodetic}")

        # ===================#
        # Define the overlap set between the found objects and the Hipparcos
        #  catalog.  This may include non-unique matches, but those should be
        #  minimal (I hope).

        # Get nominal catalog positions with the initial parameter guesses
        xycatalog = cat_table.copy()
        xycatalog["xcat"], xycatalog["ycat"] = get_xcat_ycat(
            [a0, xc, yc, F, R], cat_coords=tyc2_coords
        )

        # Make an initial cut on the catalog magnitude
        initial_mag_limit = 3
        xcat = xycatalog["xcat"][xycatalog["Vmag"] < initial_mag_limit]
        ycat = xycatalog["ycat"][xycatalog["Vmag"] < initial_mag_limit]
        sc_cat = tyc2_coords[xycatalog["Vmag"] < initial_mag_limit]

        # Print out the photometry table for inspection
        msgs.info("Photometry Table:")
        sources.sort("mag")
        sources["mag"] += 4
        for col in sources.colnames:
            sources[col].info.format = "%.8g"

        # Perform the actual matching between the sky and catalog
        all_distances = np.hypot(
            sources["xcentroid"] - np.transpose(np.atleast_2d(xcat)),
            sources["ycentroid"] - np.transpose(np.atleast_2d(ycat)),
        )
        sources["r_to_cat"] = np.nanmin(all_distances, axis=0)
        sources["idx_in_cat"] = np.nanargmin(all_distances, axis=0)

        # Figure out which `sources` are within the matching radius
        initial_matching_radius = 15
        sources_idx = sources["r_to_cat"] < initial_matching_radius

        # Figure out which Hipparcos catalog objects are matched
        close_idx = sources["idx_in_cat"][sources_idx]
        xcat = xcat[close_idx]
        ycat = ycat[close_idx]
        sc_cat = sc_cat[close_idx]

        # For plotting purposes only
        apertures = apertures[sources_idx]

        # Print to screen for sanity
        for col in sources.colnames:
            sources[col].info.format = "%.8g"
        lim_sources = sources[sources_idx].copy()
        lim_sources.pprint()

        # =======================#
        # Now, for the fun part: doing the parameter optimization!
        msgs.info("Least Squares Time!")

        msgs.bug(f"{type(sc_cat)}")
        x0 = np.array([a0, xc, yc, F, R])
        res = scipy.optimize.least_squares(
            lsq_minfunc,
            x0,
            # bounds=([0, 600, 400, 1, 100], [360, 800, 600, 3, 500]),
            # args=sources,
            verbose=1,
            method="lm",
        )
        if res.success:
            par_array.append(res.x)

        msgs.info(f"These were the starting parameters: {pprint_params(x0)}")
        msgs.info(f"These are the resulting parameters: {pprint_params(res.x)}")
        msgs.info(
            f"Final residuals: min = {np.min(res.fun):.1f} pix, "
            f"max = {np.max(res.fun):.1f} pix, "
            f"mean = {np.mean(res.fun):.1f} pix, "
            f"median = {np.median(res.fun):.1f} pix"
        )

        # Final stuff for plotting
        xcat, ycat = get_xcat_ycat(res.x, cat_coords=sc_cat)

        # Plot for fun!
        _, axis = plt.subplots(figsize=(16, 12))

        interval = astropy.visualization.ZScaleInterval(nsamples=10000)
        vmin, vmax = interval.get_limits(ccd.data - bkg.background)
        # Show the data on the plot, using the limits computed above
        axis.imshow(ccd.data - bkg.background, vmin=vmin, vmax=vmax, origin="lower")
        axis.axis("off")

        apertures.plot(color="red", lw=1.5, alpha=0.5)
        axis.plot(xcat, ycat, "x", color="white", markersize=6)

        # Finish up
        plt.tight_layout()
        plt.savefig(DDIR.joinpath("finding_stars.png"))
        plt.savefig(DDIR.joinpath("finding_stars.pdf"))
        plt.close()

        ccd.write(DDIR.joinpath("flattened_image.fits"), overwrite=True)

    # Run some stats on the accumulated parameters
    par_array = np.array(par_array)
    print(par_array)
    par_array.tofile(DDIR.joinpath("optical_params.dat"))


def analyze_asc_optics():
    """Do some analysis of the stuff from :func:`find_stars_asc`

    _extended_summary_
    """
    msgs.info("Reading in optical_params.dat for analysis...")
    par_array = np.fromfile(DDIR.joinpath("optical_params.dat"))

    par_array = par_array.reshape(len(par_array) // 5, 5)

    # print(par_array)

    msgs.info(f"Mean parameter values: {np.mean(par_array, axis=0)}")
    msgs.info(f"Stddev of parameters: {np.std(par_array, axis=0)}")


def rejigger_tycho2():
    """Rebuild and limit the Tycho-2 Catalog in magnitude and columns
    Or, rather, the Hipparcos catalog

    _extended_summary_
    """
    msgs.info("Playing with Hipparcos now!")
    t = astropy.table.Table.read(utils.Paths.data.joinpath("hipparcos.fits"))
    print(t.colnames)

    # Pull only the desired columns, and rename some of them
    t = t[
        "RAICRS",
        "DEICRS",
        "pmRA",
        "pmDE",
        "Vmag",
        "BTmag",
        "VTmag",
        "HD",
        "BD",
        "SpType",
    ]
    t.rename_columns(
        ["RAICRS", "DEICRS"],
        ["ra", "dec"],
    )
    print(t.colnames)

    # Limit the catalog to Vmag <= 6
    t = t[t["Vmag"] <= 6.0]

    _, axis = plt.subplots()
    axis.hist(t["Vmag"], bins=np.arange(8), histtype="step")
    axis.set_xlabel("Magnitude")
    axis.set_ylabel("N per bin")
    axis.set_yscale("log")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Write it out to disk
    t.pprint()
    t.write(utils.Paths.data.joinpath("hipparcos_vmag6.fits"), overwrite=True)


def pprint_params(pars):
    """Simple Pretty Print for the optimization parameters

    Parameters
    ----------
    pars : array-like
        The parameters to print

    Returns
    -------
    str
        The pretty-print
    """
    return (
        f"a0 = {pars[0]:.2f}, xc = {pars[1]:.1f}, yc = {pars[2]:.1f}, "
        f"F = {pars[3]:.2f}, R = {pars[4]:.1f}"
    )


# Testing CLI
if __name__ == "__main__":

    # test_animations()
    # run_test_hotpix()
    # test_masking()
    # make_clean_sobel_map()
    # make_hpm_fits()
    # make_nightly_median_flat()
    # plot_medflat()
    # rejigger_tycho2()
    # find_stars_asc()
    # analyze_asc_optics()

    main()
