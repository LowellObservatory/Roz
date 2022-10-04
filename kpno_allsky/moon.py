"""A module providing facilities to analyze the moon in an image.

Methods in this module predominantly deal with determining the relationship
between the phase of the moon and the apparent size of the moon in an all-sky
image. The phase of the moon is provided on a scale from 0.0 (a new moon) to
1.0 (a full moon). Methods are provided to find the position of the moon and
sun. The phase of the moon in an eclipse and outside of an eclipse are
separated into their own methods. There is also a method provided to plot this
data and the model linking the apparent size of the image to the moon phase.
"""

import os
import math
import warnings
import ephem
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.modeling import models, fitting

import coordinates
import image


# Sets up a pyephem object for the camera.
camera = ephem.Observer()
camera.lat = "31.959417"
camera.lon = "-111.598583"
camera.elevation = 2120


def eclipse_phase(d):
    """Calculate the proportion of the moon that is lit up during an eclipse.

    Parameters
    ----------
    d : float
        The distance between the center of Earth"s shadow and the center of the
        moon in kilometers.

    Returns
    -------
    float
        The phase of the moon, ranging between 0.0 and 1.0, where 0.0 is a new
        moon and 1.0 is a full moon.
    """

    # In kilometers.
    r = 1737 # R_moon
    R = 4500  #R_earth

    # This makes addition work as addition and not concatenates
    d = np.asarray(d)
    d = np.abs(d)  # Required as for after totality times d < 0

    r2 = r * r
    R2 = R * R
    d2 = np.square(d)

    # Part 1 of the shaded area equation
    a = (d2 + r2 - R2)
    b = d * 2 * r
    p1 = r2 * np.arccos(a / b)

    # Part 2 of the shaded area equation
    a = (d2 + R2 - r2)
    b = d * 2 * R
    p2 = R2 * np.arccos(a / b)

    # Part 3 of the shaded area equation
    a1 = (r + R - d)
    a2 = (d + r - R)
    a3 = (d - r + R)
    a4 = (d + r + R)
    p3 = (0.5) * np.sqrt(a1 * a2 * a3 * a4)

    # Add them together to get the shaded area
    A = p1 + p2 - p3

    # Get the shaded proportion by divding the shaded area by the total area
    # Assumes r is the radius of the moon being shaded.
    P = A / (np.pi * r2)

    # P is the shaded area, so 1-P is the lit up area.
    P = 1 - P

    return P


# 1.0 = Full moon, 0.0 = New Moon
def moon_phase(img):
    """Calculate the proportion of the moon that is lit up for non-eclipse
    nights.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    float
        The phase of the moon, ranging between 0.0 and 1.0, where 0.0 is a new
        moon and 1.0 is a full moon.
    """
    # Sets the calculation date.
    camera.date = img.formatdate

    # Makes a moon object and calculates it for the observation location/time
    moon = ephem.Moon()
    moon.compute(camera)

    return moon.moon_phase


def moon_size(img):
    """Calculate the area of the moon in pixels in a given image.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    int
        The size of the moon in pixels.

    Notes
    -----
    This method first converts the image to a black and white two-tone image
    where pixels with greyscale values above or equal to 250 is set to
    white and everything below is set to black. A binary closing is performed
    on the image, which smooths over any small black pixel regions within the
    moon. These black regions are created when the pixels are brighter than the
    maximum of 255 for white pixels and overflow back to 0.
    The white regions are labeled and their sizes are found using
    ndimage.label. Then, the approximate position of the center of the moon
    is found using find_moon.

    If the pixel at the moon"s center is black (due to aforementioned pixel
    value overflow), the nearest white region along the x axis is found and
    the size of this region is returned.

    If this pixel is white, the size of this white region is returned.
    """
    thresh = 5
    dat = np.where(img.data >= 255 - thresh, 1, 0)

    # Runs a closing to smooth over local minimums (which are mainly caused by
    # a rogue antenna). Then labels the connected white regions. Structure s
    # Makes it so that regions connected diagonally are counted as 1 region.
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    dat = ndimage.morphology.binary_closing(dat, structure=s)
    labeled, nums = ndimage.label(dat, structure=s)

    # Want to find the size of each labeled region.
    sizes = [0] * (nums + 1)
    for row in labeled:
        for val in row:
            sizes[val] = sizes[val] + 1

    # We want to exclude the background from the "biggest region" calculation.
    # It"s quicker to just set the background (0) region to 0 than to check
    # every single value above before adding it to the array.
    sizes[0] = 0

    # Following code calculates d, the distance between the center of
    # Earth"s shadow and the center of the moon. Basically just d = v*t.

    # Use pyephem to find the labeled region that the moon is in.
    posx, posy, _ = find_moon(img)
    posx = math.floor(posx)
    posy = math.floor(posy)

    reg = labeled[posy, posx]

    # Very large and bright moons have a dark center (region 0) but I want the
    # region of the moon.
    while reg == 0 and posx < 511:
        posx = posx + 1
        reg = labeled[posy, posx]

    biggest = sizes[reg]

    return biggest


def find_moon(img):
    """Find the (x, y, alt) coordinate of the moon"s center in a given image.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    x : float
        The x coordinate of the moon"s center.
    y : float
        The y coordinate of the moon"s center.
    alt : float
        The altitude angle of the moon"s center.

    Notes
    -----
    The x and y coordinates are corrected for irregularities in the lens using
    coordinates.galactic_conv.
    """

    # Sets the date of calculation.
    camera.date = img.formatdate

    # Calculates the moon position.
    moon = ephem.Moon()
    moon.compute(camera)

    # Conversion to x,y positions on the image.
    alt = np.degrees(moon.alt)
    az = np.degrees(moon.az)
    x, y = coordinates.altaz_to_xy(alt, az)
    x, y = coordinates.galactic_conv(x, y, az)

    return (x, y, alt)


def find_sun(img):
    """Find the (alt, az) coordinate of the sun"s center in a given image.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    alt : float
        The altitude coordinate of the sun"s center.
    az : float
        The azimuth coordinate of the sun"s center.
    """

    # Sets the date of calculation.
    camera.date = img.formatdate

    # Calculates the sun position.
    sun = ephem.Sun()
    sun.compute(camera)

    # Conversion to x,y positions on the image.
    alt = np.degrees(sun.alt)
    az = np.degrees(sun.az)

    return (alt, az)


# Fits a Moffat fit to the moon and returns the estimated radius of the moon.
# Radius of the moon is the FWHM of the fitting function.
def fit_moon(img, x, y):
    """Fit a Moffat function to the moon in a given image.

    Parameters
    ---------
    img : image.AllSkyImage
        The image.
    x : float
        The x coordinate of the moon"s center.
    y : float
        The y coordinate of the moon"s center.

    Returns
    -------
    float
        The Full Width at Half Maximum of the Moffat function.
    """
    # This block of code runs straight vertical from the center of the moon
    # It gives a predicted rough radius of the moon, it starts counting at the
    # first white pixel it encounters (the center may be black)
    # and stops at the last white pixel. White here defined as > 250 greyscale.
    yfloor = math.floor(y)
    count = False
    size = 0
    xfloor = math.floor(x)
    start = xfloor

    # The only reason we have this if block is to ensure we don"t run for
    # moon radii greater than 35 in this case.
    for i in range(0, 35):
        start += 1

        # Breaks if it reaches the edge of the image.
        if start == img.data.shape[1]:
            break
        if not count and img.data[yfloor, start] >= 250:
            count = True
        elif count and img.data[yfloor, start] >= 250:
            size += 1
        elif count and img.data[yfloor, start] < 250:
            break

    # Add some buffer pixels in case the center is black and the edges of the
    # moon are fuzzed and then convert radius to diameter.
    size = (size + 10) * 2

    # Makes sure the lower/upper slices don"t out of bounds error.
    lowerx = xfloor - size if (xfloor - size > 0) else 0
    lowery = yfloor - size if (yfloor - size > 0) else 0
    upperx = xfloor + size if (xfloor + size < 511) else 511
    uppery = yfloor + size if (yfloor + size < 511) else 511

    # Size of the moon enclosing square.
    deltax = (upperx - lowerx)
    deltay = (uppery - lowery)

    # Creates two arrays, with the array values being the x or y coordinate of
    # that location in the array.
    y, x = np.mgrid[0:deltay, 0:deltax]

    # Slices out the moon square and finds center coords.
    z = img.data[lowery:uppery, lowerx:upperx]
    midy = deltay / 2
    midx = deltax / 2

    # Moffat fit, centered in square, stdev of 20 as a start.
    stddev = 20
    model_init = models.Moffat2D(amplitude=200, x_0=midx, y_0=midy,
                                 gamma=stddev)
    fit = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter("ignore")
        model = fit(model_init, x, y, z)

    # /2 is average FWHM but FWHM = diameter, so divide by two again.
    #fwhm = (model.x_fwhm + model.y_fwhm) / 4
    fwhm = model.fwhm / 2

    return fwhm


# Generates the size vs illuminated fraction for the two eclipse nights.
def generate_eclipse_data(regen=False):
    """Generate moon phase data for eclipses.

    Parameters
    ----------
    regen : bool, optional
        If True, regen from scratch. Otherwise, load it from a file.
        Defaults to False.

    Returns
    -------
    truevis : list of lists
        A list containing one or more lists where each list contains values
        corresponding to an eclipse. Each value is the phase of the moon
        ranging from 0.0 to 1.0, in an image taken during that night.
        0.0 corresponds to a new moon, while 1.0 corresponds to a full moon.
        The list is ordered in chronological order; that is, the first value
        within the list is calculated from an image that was taken at the
        beginning of the eclipse, and the last value is taken from the end of
        the eclipse. Currently, this is hardcoded such that the first list
        represents the eclipse on 2018/01/31, and the second list represents
        the eclipse on 2015/04/04.
    imvis : list of lists
        A list containing one or more lists where each list contains values
        corresponding to an eclipse. Each value is the area of the moon,
        in pixels, in an image taken during that night. The list is ordered in
        chronological order; that is, the first value within the list is
        calculated from an image that was taken at the beginning of the eclipse,
        and the last value is taken from the end of the eclipse.
        Currently, this is hardcoded such that the first list represents
        the eclipse on 2018/01/31, and the second list represents the eclipse
        on 2015/04/04.
    """
    dates = ["20180131", "20150404"]

    # Function within a function to avoid code duplication.
    def data(date):
        # Necessary lists
        distances = []
        imvis = []
        truevis = []

        # Check to see if the data has been generated already.
        # If it has then read it from the file.
        save = os.path.join(os.path.dirname(__file__), *["data", "eclipse-" + date + ".txt"])
        if os.path.isfile(save) and not regen:
            f = open(save)
            for line in f:
                line = line.rstrip().split(",")
                truevis.append(float(line[0]))
                imvis.append(float(line[1]))
            f.close()
            return (truevis, imvis)

        # If we're regenerating the data we do it here.
        directory = os.path.join(os.path.dirname(__file__), *["Images", "Original", "KPNO", date])
        images = sorted(os.listdir(directory))

        # I need a better way to check this.
        if ".DS_Store" in images:
            images.remove(".DS_Store")

        # Finds the size of the moon in each image.
        for name in images:
            print(name)
            print(date)
            print(directory)
            img = image.load_image(name, date, "KPNO")

            # This basically hacks us to use the center of the earth as our
            # observation point.
            camera.elevation = - ephem.earth_radius
            camera.date = img.formatdate

            # Calculates the sun and moon positions.
            moon = ephem.Moon()
            sun = ephem.Sun()
            moon.compute(camera)
            sun.compute(camera)

            # Finds the angular separation between the sun and the moon.
            sep = ephem.separation((sun.az, sun.alt), (moon.az, moon.alt))

            # Radius of moon orbit to convert angular separation -> distance
            R = 385000

            # For angles this small theta ~ sin(theta), so I dropped the sine
            # to save computation time.
            # Angle between moon and earth"s shadow + angle between moon and sun
            # should ad d to pi, i.e. the earth"s shadow is across from the sun.
            d = R * (np.pi - sep)

            size = moon_size(img)
            imvis.append(size)
            distances.append(d)

            print("Processed: " + date + "/" + name)

        # Calculates the proportion of visible moon for the given distance
        # between the centers.
        truevis = eclipse_phase(distances)

        imvis = np.asarray(imvis)

        # If the moon is greater than 40,000 pixels then I know that the moon
        # has merged with the light that comes from the sun and washes out the
        # horizon.
        imvis = np.where(imvis < 80000, imvis, float("NaN"))

        f = open(save, "w")

        # Writes the data to a file so we can read it later for speed.
        for i in range(0, len(truevis)):
            f.write(str(truevis[i]) + "," + str(imvis[i]) + "\n")
        f.close()

        return (truevis, imvis)

    trues = []
    ims = []
    for date in dates:
        true, im = data(date)
        trues.append(true)
        ims.append(im)

    return (trues, ims)


def moon_circle(frac):
    """Calculate the estimated pixel radius of the moon based on the
    fraction of the moon that is illuminated.

    Parameters
    ----------
    frac : float
        The proportion of the moon that is illuminated by sunlight.

    Returns
    -------
    float
        The estimated radius of the moon.

    Notes
    -----
    The model used in this method to convert between the fraction of the moon
    that is illuminated to the estimated pixel area was found by plotting the
    moon pixel area versus the moon phase and picking representative points
    to model the relation. The
    model is designed to always overestimate the size of the moon. The model
    is defined by interpolating from the following table of representative
    points:

    =========================   ==================
    Moon fraction illuminated   Moon area (pixels)
    -------------------------   ------------------
    0                            650
    0.345                        4000
    0.71                         10500
    0.88                         18000
    0.97                         30000
    1.0                          35000
    =========================   ==================

    """
    illuminated = [0, 0.345, 0.71, 0.88, 0.97, 1.0]
    size = [650, 4000, 10500, 18000, 30000, 35000]

    A = np.interp(frac, illuminated, size)
    return np.sqrt(A/np.pi)


def moon_mask(img):
    """Generate a masking array that covers the moon in a given image.

    Parameters
    ---------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    numpy.ndarray
        An array where pixels inside the moon are marked with False and those
        outside the moon are marked with True.
    """
    # Get the fraction visible for interpolation and find the
    # location of the moon.
    vis = moon_phase(img)
    x, y, _ = find_moon(img)

    # Creates the circle patch we use.
    r = moon_circle(vis)
    circ = Circle((x, y), r, fill=False)

    # The following code converts the patch to a 512x512 mask array, with True
    # for values outside the circle and False for those inside.
    # This is the same syntax as np.ma.make_mask returns.

    # This section of code generates an 262144x2 array of the
    # 512x512 pixel locations. 262144 = 512^2
    points = np.zeros((512**2, 2))
    index = 0
    for i in range(0, 512):
        for j in range(0, 512):
            # These are backwards as expected due to how reshape works later.
            # Points is in x,y format, but reshape reshapes such that
            # it needs to be in y,x format.
            points[index, 0] = j
            points[index, 1] = i
            index += 1

    # Checks all the points are inside the circle, then reshapes it to the
    # 512x512 size.
    mask = circ.contains_points(points)
    mask = mask.reshape(512, 512)

    return mask


def generate_plots():
    """Generate a plot of illuminated fraction versus apparent moon size.

    Notes
    -----
    The eclipse dataset is loaded using generate_eclipse_data and then plotted.
    The file images.txt contains the illuminated fraction of the moon
    and the pixel area of the moon in that image. This data is plotted on top
    of the eclipse data. The illuminated fraction of the moon is found using
    moon_phase, and the size of the moon in the image is found using moon_size.
    On top of this data the theoretical moon size model used in moon_circle is
    plotted. Once all of these are plotted, two versions of the plot are saved,
    one with a standard y and x axis, and one with a logarithmic y axis.

    These plots are saved directly to Images/ under the names "moon-size.png"
    and "moon-size-log.png."

    """
    # Loads the eclipse data
    vis, found = generate_eclipse_data()
    print("Eclipse data loaded!")

    # Eclipse normalization code.
    #found[0] = np.asarray(found[0]) / np.nanmax(found[0])
    #found[1] = np.asarray(found[1]) / np.nanmax(found[1])

    # Plots the two eclipses, the first in blue (default), the second in green
    plt.scatter(vis[0], found[0], label="2018/01/31 Eclipse", s=7)
    #plt.scatter(vis[1], found[1], label="2015/04/04 Eclipse", s=7, c="g")
    plt.ylabel("Approx Moon Size (pixels)")
    plt.xlabel("Illuminated Fraction")

    # Vis is the portion of the moon illuminated by the sun that night
    # Found is the approximate size of the moon in the image
    vis = []
    found = []
    loc = os.path.join(os.path.dirname(__file__), *["data", "images.txt"])
    with open(loc, "r") as f:
        for line in f:
            line = line.rstrip()
            info = line.split(",")
            img = image.load_image(info[1], info[0], "KPNO")
            vis.append(moon_phase(img))
            found.append((moon_size(img)))
            print("Processed: " + info[0] + "/" + info[1] + ".png")

    # Removes out any moons that appear too large in the images to be
    # considered valid.
    found = np.asarray(found)
    found = np.where(found < 40000, found, float("NaN"))

    # Normalizes the non eclipse data.
    #found1 = found / np.nanmax(found)

    # Adds the noneclipse data to the plot.
    plt.scatter(vis, found, label="Regular", s=7)

    # This plots the estimated model of moon size on top of the graph.
    vis2 = [0, 0.345, 0.71, 0.88, 0.97, 1.0]
    found2 = [650, 4000, 10500, 18000, 30000, 35000]
    plt.plot(vis2, found2, label="Model", c="r")

    # Interpolation estimate for the moon size in the image based on the
    # illuminated fractions.
    found3 = np.interp(vis, vis2, found2)
    plt.scatter(vis, found3, label="Interpolated", s=7)
    plt.legend()

    # Saves the figure, and then saves the same figure with a log scale.
    plt.savefig("Images/moon-size.png", dpi=256)

    ax = plt.gca()
    ax.set_yscale("log")
    plt.savefig("Images/moon-size-log.png", dpi=256)

    plt.close()


if __name__ == "__main__":
    # 20160326/r_ut071020s70020
    generate_plots()
