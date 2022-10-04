"""A module containing various methods for loading, saving and modifying images.

This module is designed to facilitate image processing. A class is provided
that defines a custom AllSkyImage object that contains all necessary details
about an image when it is loaded. This AllSkyImage object is used in all
image processing methods throughout the other modules. Methods are provided
to load and save an image in this format. Additional methods are provided to
draw analysis objects on top of the images.
"""

import os
import math

from PIL import Image as pil_image
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import coordinates


class AllSkyImage():
    """An all-sky image taken at a certain point in time.

    Attributes
    ----------
    name : str
        The name of the image.
    date : str
        The date on which the image was taken.
    camera : str
        The camera used to take the image. Either `KPNO` for the all-sky
        camera at Kitt-Peak or `MMTO` for the all-sky camera at the MMT
        Observatory.
    data : numpy.ndarray
        The image data.
    formatdate : str
        The date and time the image was taken in yyyy-mm-dd hh:mm:ss format.
    time : astropy.time.Time
        The time at which the image was taken.
    """
    def __init__(self, name, date, camera, data):
        self.name = name
        self.date = date
        self.camera = camera
        self.data = data

        if date is not None:
            format1 = date[:4] + "-" + date[4:6] + "-" + date[6:8]
            format2 = name[4:6] + ":" + name[6:8] + ":" + name[8:10]
            self.formatdate = format1 + " " + format2
            self.time = Time(self.formatdate)
        else:
            self.formatdate = None
            self.time = None


def load_image(name, date, camera, mode="L"):
    """Load an image.

    Parameters
    ----------
    name : str
        The name of the image.
    date : str
        The date on which the image was taken.
    camera : {"KPNO", "MMTO", "SW"}
        The camera used to take the image. "KPNO" represents the all-sky
        camera at Kitt-Peak. "MMTO" represents the all-sky camera at the MMT
        Observatory. "SW" represents the spacewatch all-sky camera.
    mode : {"L", "RGB", "RGBA"}, optional
        The color mode to load the image in. Defaults to "L" for greyscale.
        Use "RGB" for color and "RGBA" for color with an alpha layer.

    Returns
    -------
    img : image.AllSkyImage
        The image.

    """
    # If the name was passed without .png at the end append it so we know what
    # format this bad boy is in.
    if camera == "KPNO" and not name[-4:] == ".png":
        name = name + ".png"
    elif camera == "SW" and not name[-4:] == ".jpg":
        name = name + ".jpg"

    # Loads the image using Pillow and converts it to given mode.
    loc = os.path.join("Images", *["Original", camera, date, name])
    img = np.asarray(pil_image.open(loc).convert(mode))
    return AllSkyImage(name, date, camera, img)


def save_image(img, location, cmap="gray"):
    """Save an image.

    Save an image passed in `img` with the name `img.name` into the location in
    `location`. `cmap` provides an option to save the image in greyscale.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.
    location : str
        The relative path to save the image to. If the path does not exist,
        it is created.
    cmap : str, optional
        A colormap to use when saving the image. Supports any matplotlib
        supported colormap. Defaults to "gray" to save in grayscale.

    Notes
    -----
    See https://matplotlib.org/tutorials/colors/colormaps.html for more detail
    on matplotlib colormaps.

    """
    if not os.path.exists(location):
        os.makedirs(location)

    dpi = 128
    y = img.data.shape[0] / dpi
    x = img.data.shape[1] / dpi

    # Generate Figure and Axes objects.
    fig = plt.figure()
    fig.set_size_inches(x, y)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    ax.set_axis_off()
    fig.add_axes(ax)

    # Adds the image into the axes and displays it
    # Then saves
    ax.imshow(img.data, cmap=cmap, vmin=0, vmax=255)

    # If location was passed with / on the end, don"t append another one.
    # I changed this since the above comment, now I just slice it off if you
    # added a /.
    if not location[-1:] == "/":
        name = os.path.join(location, img.name)
    else:
        name = os.path.join(location[:-1], img.name)

    # Print "saved" after saving, in case saving messes up.
    plt.savefig(name, dpi=dpi)
    print("Saved: " + name)

    # Close the plot in case you"re running multiple saves.
    plt.close()


def draw_patch(img, patch):
    """Draw a given patch on an image.

    Patches will be drawn in lime green.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.
    patch : matplotlib.patches.Patch
        The patch.

    Returns
    -------
    img : image.AllSkyImage
        The image with patches drawn on top.

    Notes
    -----
    This method draws patches directly onto an image. The returned image will
    be in the same color mode as the input image, i.e. if the input image is
    in RGB color, then the returned image will be also. If the input image is
    greyscale, then the returned image will be also.
    """
    # Scale in inches
    scale = 4
    dpi = img.data.shape[0] / scale

    greyscale = True
    if len(img.data.shape) == 3:
        greyscale = False

    # Generate Figure and Axes objects.
    fig = plt.figure()
    fig.set_size_inches(scale, scale)
    fig.set_dpi(dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    ax.set_axis_off()
    fig.add_axes(ax)

    # Adds the image into the axes and displays it
    ax.imshow(img.data, cmap="gray")
    ax.set_aspect("equal")
    ax.add_patch(patch)

    width = int(scale * dpi)
    height = width

    # Extracts the figure into a numpy array and then converts it to greyscale.
    canvas = FigureCanvas(fig)
    canvas.draw()
    data = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape((height, width, 3))

    # Slices out the RGB components and then multiplies them by RGB conversion.
    if greyscale:
        data = np.dot(data[...,:3], [0.299, 0.587, 0.114])

    plt.close()

    return AllSkyImage(img.name, img.date, img.camera, data)


def draw_celestial_horizon(img):
    """Draw a path representing where the declination angle is zero.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    img : image.AllSkyImage
        A greyscale image with a pink path representing the celestial horizon.
    """

    # So that we don"t modify in place.
    data = np.copy(img.data)

    dec = 0
    ra = 0
    while ra <= 360:
        xy = coordinates.radec_to_xy(ra, dec, img.time)
        xy = (round(xy[0]), round(xy[1]))

        # Remember y is first, then x
        # Also make sure it"s on the image at all.
        if xy[1] < 512 and xy[0] < 512:
            data[xy[1], xy[0]] = (244, 66, 229)

        ra += 0.5

    return AllSkyImage(img.name, img.date, img.camera, data)


def draw_contours(img):
    """Draw three angular contours on an image.

    Contours will be drawn in lime green.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    img : image.AllSkyImage
        The image with contours drawn on top.

    Notes
    -----
    This method draws contours directly onto an image using matplotlib patches.
    These contours represent altitude angles of 0, 30, and 60 degrees up
    from the horizon. The returned image will be in the same color mode as the
    input image, i.e. if the input image is in RGB color, then the returned
    image will be also. If the input image is greyscale, then the returned
    image will be also.
    """
    if img.camera.lower() == "kpno":
        theta = coordinates.theta_kpno
        rp = coordinates.r_kpno
        center = coordinates.center_kpno
    else:
        theta = coordinates.theta_sw
        rp = coordinates.r_sw
        center = coordinates.center_sw

    for alt in range(0, 100, 30):
        r = np.interp(90 - alt, xp=theta, fp=rp)
        if img.camera.lower() == "kpno":
            r = r * 240 / 11.6  # mm to pixel rate

        circ = Circle(center, radius=r, fill=False, edgecolor="green")
        img = draw_patch(img, circ)

    return img


def draw_square(x, y, img):
    """Draw squares centered at the given coordinates.

    Drawn squares will be cyan in color and will have a side length of 10
    pixels.

    Parameters
    ----------
    x : array_like
        The x coordinates of the centers of each square.
    y : array_like
        The x coordinates of the centers of each square.
    img : image.AllSkyImage
        The image.
    rgb : bool, optional
        If the returned image should be in RGB color or greyscale. Defaults to
        False, representing greyscale.

    Returns
    -------
    img : image.AllSkyImage
        The image with squares drawn on top.

    Notes
    -----
    The returned image will be in the same color mode as the input
    image, i.e. if the input image is in RGB color, then the returned image
    will be also. If the input image is greyscale, then the returned image will
    be also.

    """
    # Scale in inches
    scale = 4
    dpi = img.data.shape[0] / scale

    greyscale = True
    if len(img.data.shape) == 3:
        greyscale = False

    # Generate Figure and Axes objects.
    fig = plt.figure()
    fig.set_size_inches(scale, scale)  # 4 inches by 4 inches
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    ax.set_axis_off()
    fig.add_axes(ax)

    # Adds the image into the axes and displays it
    ax.imshow(img, cmap="gray")

    ax.set_aspect("equal")
    for i, val in enumerate(x):
        rect = Rectangle((x[i]-5, y[i]-5), 11, 11, fill=False, ec="c")
        ax.add_patch(rect)

    width = int(scale * dpi)
    height = width

    # Extracts the figure into a numpy array and then converts it to greyscale.
    canvas = FigureCanvas(fig)
    canvas.draw()
    data = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape((height, width, 3))

    # Slices out the RGB components and then multiplies them by RGB conversion.
    # Only done if returning an image in greyscale.
    if greyscale:
        data = np.dot(data[...,:3], [0.299, 0.587, 0.114])

    plt.close()

    return AllSkyImage(img.name, img.date, img.camera, data)


def get_exposure(img):
    """Get the exposure time of an image.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    float or int
        The exposure time in seconds of the provided image.
        Possible values are 0.3, 0.02 or 6.

    Notes
    -----
    get_exposure works by looking at two specific pixels in an image taken on
    the KPNO camera. The first pixel is at (174, 19) in (x, y) coordinates,
    where (0, 0) is the top left corner of the image. This pixel appears as
    gray in images taken at 0.3s or 0.02s exposure times, but as
    black in images taken in 6s exposure times. In order to differentiate
    between 0.3s and 0.02s a second pixel at (119, 17) is used, which appears
    as gray in images taken at 0.02s exposure time but as black in images taken
    in 0.3s exposure time.

    """
    pix1 = img.data[19, 174]
    pix2 = img.data[17, 119]

    # Handles separate cases for greyscale and RGB images.
    # Greyscale conversion below is the same one used by imread.
    if len(img.data.shape) == 3:
        pix1 = pix1[0] * 299/1000 + pix1[1] * 587/1000 + pix1[2] * 114/1000
        pix1 = math.floor(pix1)

        pix2 = pix2[0] * 299/1000 + pix2[1] * 587/1000 + pix2[2] * 114/1000
        pix2 = math.floor(pix2)

    if pix1 == 225:
        return 0.3
    if pix2 == 225:
        return 0.02
    return 6

if __name__ == "__main__":
    test = load_image("r_ut005728s27480", "20160101", "KPNO", "RGB")
    test = draw_contours(test)
    save_image(test, "Test")
