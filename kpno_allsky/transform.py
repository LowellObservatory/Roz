"""A module providing facilities for converting all-sky images into map-style
projections of the visible night sky.

Methods in this module are designed to build a projection of the visible night
sky from an all-sky image. The predominant method, :func:~`transform`, performs
the transformation from an all-sky image to an Eckert-IV projection. Additional
methods are included that convert the all-sky image coordinates to a Mollweide
projection. There are also methods that draw defining contours on top of the
transformed image. One method draws altitude contours at 0, 30, and 60 degrees
of altitude angle, and one method draws an outline of the DESI survey area.
"""

import math
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import coordinates
import mask
import image


center_kpno = (256, 252)
center_sw = (512, 512)


# This takes the file and the given date and then transforms it
# from the circle into an eckert-iv projected ra-dec map.
def transform(img):
    """Transform a circular all-sky image into an Eckert-IV projection of the
    visible night sky.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    See Also
    --------
    eckertiv : Define the projection method.

    Notes
    -----
    First applies a mask generated from mask.generate_mask().
    From there, lists of x and y pixels inside each image is built.
    These lists are converted to right ascension and declination
    representations of each pixel. These are passed to eckertiv(), which
    converts these points to x and y positions on the Eckert-IV projection.
    The map is then built as a scatter plot using these x and y positions,
    where the color of each dot is taken from the pixel originally used.
    Each point is the same size, which is a valid assumption since the
    Eckert-IV projection is an equal area projection.
    The plot is then saved to Images/transform/`img.date`/`img.name`.

    """
    time = img.time

    # Find the mask and black out those pixels.
    # Contrasting the clouds already masks.
    if img.camera == "KPNO":
        masking = mask.generate_full_mask()
        img = mask.apply_mask(masking, img)

    # Sets up the figure and axes objects
    fig = plt.figure(frameon=False)
    fig.set_size_inches(12, 6)

    # We just want the globe to be centered in an image so we turn off the axis
    ax1 = plt.Axes(fig, [0., 0., 1., 1.])
    ax1.set_axis_off()

    # This is black background stuff
    rapoints = []
    decpoints = []

    # Just a bunch of ra-dec points for the background.
    ra = 0
    while ra <= 360:
        dec = -90
        while dec <= 90:
            rapoints.append(ra)
            decpoints.append(dec)

            dec += .5
        ra += .5

    # Scatter for the background
    # (i.e. fills in the rest of the globular shape with black)
    x, y = eckertiv(rapoints, decpoints)
    ax1.scatter(x, y, s=2, color="black")

    # This is the image conversion
    xpoints = []
    ypoints = []

    center = center_kpno if img.camera == "KPNO" else center_sw
    max_r = 241 if img.camera == "KPNO" else 510
    for row in range(0, img.data.shape[0]):
        for column in range(0, img.data.shape[1]):

            x = column - center[0]
            y = center[1] - row
            r = math.hypot(x, y)

            # Only want points in the circle to convert
            if r <= max_r:
                xpoints.append(column)
                ypoints.append(row)

    # We need to add 0.5 to the x,y coords to get the center of the pixel
    # rather than the top left corner.
    # Convert the alt az to x,y
    x = np.add(np.asarray(xpoints), 0.5)
    y = np.add(np.asarray(ypoints), 0.5)
    rapoints, decpoints = coordinates.xy_to_radec(x, y, time, img.camera)


    # This block changes the ra so that the projection is centered at
    # ra = 360-rot.
    # The reason for this is so the outline survey area is 2 rather than 3
    # polygons.
    rot = 60
    rapoints = np.where(rapoints > (360 - rot), rapoints + rot - 360, rapoints + rot)

    # Finds colors for dots.
    colors = []
    for i, _ in enumerate(rapoints):
        x = xpoints[i]
        y = ypoints[i]

        if img.data.shape[-1] == 3:
            colors.append(img.data[y, x] / 255)
        else:
            colors.append(img.data[y, x])

    # Scatter for the image conversion
    x, y = eckertiv(rapoints, decpoints)
    ax1.scatter(x, y, s=1, c=colors, cmap="gray")

    # Add the contours
    ax1 = contours(ax1, time)

    # These coord: -265.300085635, -132.582101423 are the minimum x and y of
    # the projection.
    ax1.text(-290, -143, img.formatdate, style="italic")

    patches = desi_patch()
    for patch in patches:
        ax1.add_patch(patch)

    # Add the axes to the fig so it gets saved.
    fig.add_axes(ax1)

    # Make sure the folder location exists
    directory = os.path.join("Images", *["transform", img.date])
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save name.
    conv = os.path.join(directory, img.name)

    # Want it to be 1920 wide.
    dpi = 1920 / (fig.get_size_inches()[0])
    plt.savefig(conv, dpi=dpi)

    print("Saved: " + conv)

    # Gotta close the plot so we don't memory overflow lol.
    plt.close()


def contours(axis, time):
    """Add three altitude contours to an axis based on a given time.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis
        An axis to add the contours to.
    time : astropy.time.Time
        A time and date.

    Returns
    -------
    matplotlib.pyplot.axis
        New axis where the contours have been overlaid on top.

    Notes
    -----
    The contours added to the axis correspond to altitude angles of 0, 30 and
    60 degrees.
    """
    # Loop runs over all the alts.
    # Resets the arrays at the start, creates alt/az for that alt value.
    for alt in range(0, 90, 30):
        # We need it to not connect the different contours so they have to be
        # added seperately.
        altpoints = []
        azpoints = []
        for az in range(0, 360, 1):
            altpoints.append(alt)
            azpoints.append(az)

        rapoints, decpoints = coordinates.altaz_to_radec(altpoints, azpoints, time)

        # Rotation block
        # Centers contours at 60 degrees ra.
        for i, ra in enumerate(rapoints):
            rot = 60
            if ra > (360 - rot):
                ra = ra + rot - 360
            else:
                ra = ra + rot

        # Don"t sort the 60 contour since it"s a complete circle.
        if not alt == 60:
            # Sorting by ra so that the left and right edges don't connect.
            points = []
            for i, ra in enumerate(rapoints):
                points.append((ra, decpoints[i]))

            points = sorted(points)

            # Condensing lines using magic python list comprehension.
            rapoints = [point[0] for point in points]
            decpoints = [point[1] for point in points]

            x, y = eckertiv(rapoints, decpoints)

            # 42f44e is super bright green.
            axis.plot(x, y, c="#42f44e")

        # The 60 contour needs to be two plots if it gets seperated by the edge.
        else:
            temp = sorted(rapoints)
            # Basically if the difference in the least and most is almost the
            # entire image then seperate
            # "Lower" = leftside, "Upper" = rightside
            if temp[-1] - temp[0] > 350:
                lowerra = []
                lowerdec = []
                upperra = []
                upperdec = []
                for i, ra in enumerate(rapoints):
                    if rapoints[i] < 180:
                        lowerra.append(ra)
                        lowerdec.append(decpoints[i])
                    else:
                        upperra.append(ra)
                        upperdec.append(decpoints[i])

                # Clockwise sorting is necessary here to prevent the top and
                # Bottom ends on either edge from joining.
                # Left needs to be sorted from negative x.
                lowerra, lowerdec = clockwise_sort(lowerra, lowerdec)
                x, y = eckertiv(lowerra, lowerdec)
                axis.plot(x, y, c="#42f44e")

                # Right needs to be sorted from the positive x.
                upperra, upperdec = clockwise_sort(upperra, upperdec, True)
                x, y = eckertiv(upperra, upperdec)
                axis.plot(x, y, c="#42f44e")

            else:
                x, y = eckertiv(rapoints, decpoints)
                axis.plot(x, y, c="#42f44e")

    return axis


def mollweide_findtheta(dec, n):
    """Find the auxiliary latitude (theta) value that defines a given
    latitude in the Mollweide projection.

    Parameters
    ----------
    dec : array_like
        The declination (or latitude) angular coordinates of the data set in
        radians.
    n : int
        The number of iterations to use in Newton's method.

    Returns
    -------
    np.ndarray
        Array of auxiliary latitude values corresponding to the
        input latitude values.

    Notes
    -----
    This method finds the auxiliary latitude values using Newton's method, and
    is thus recursive.
    Wikipedia provides a simple form of the equation that is iterated upon
    in this method. See here for more details:
    https://en.wikipedia.org/wiki/Mollweide_projection
    """
    # This is here in case dec is a list rather than a numpy array.
    dec = np.asarray(dec)

    # First short circuit
    if n == 0:
        return np.arcsin(2 * dec / math.pi)

    # Array literally just filled with half pis.
    halfpi = np.empty(len(dec))
    halfpi.fill(math.pi / 2)

    theta = mollweide_findtheta(dec, n-1)

    cond1 = np.equal(theta, halfpi)
    cond2 = np.equal(theta, -1 * halfpi)
    cond = np.logical_or(cond1, cond2)

    # Choose the original value (pi/2 or neg pi/2) if its true for equality
    # Otherwise use that value"s thetanew.
    num = (2 * theta + np.sin(2 * theta) - math.pi * np.sin(dec))
    thetanew = theta - num / (2 + 2 * np.cos(2 * theta))
    thetanew = np.where(cond, dec, thetanew)

    return thetanew


def mollweide(ra, dec):
    """Find a Mollweide representation of the given coordinates.

    Parameters
    ----------
    ra : array_like
        The right ascension (or longitude) coordinates of the data set.
    dec : array_like
        The declination (or latitude) coordinates of the data set.

    Returns
    -------
    x : array_like
        The x coordinates corresponding to the given points.
    y : array_like
        The y coordinates corresponding to the given points.

    See Also
    --------
    mollweide_findtheta : Newton's method for finding each point"s auxiliary
                          latitdue (theta) value.

    Notes
    -----
    This method defines the x,y and latitude and longitude using the standard
    Mollweide definition. Wikipedia provides a simple form of the equations
    used in this method, including a defnition of the theta value that is
    found using Newton's method. See here for more details:
    https://en.wikipedia.org/wiki/Mollweide_projection
    """

    # Center latitude
    center = math.radians(180)

    theta = mollweide_findtheta(np.radians(dec), 2)

    R = 100

    # Mollweide conversion functions.
    # a is the minor axis of an ellipse, hence the variable.
    a = R * math.sqrt(2)
    x = (2 * a / math.pi)*(np.subtract(np.radians(ra), center))*np.cos(theta)
    y = a * np.sin(theta)

    return(x, y)


def eckertiv_findtheta(dec, n):
    """Find the auxiliary latitude (theta) value that defines a given
    latitude in the Eckert-IV projection.

    Parameters
    ----------
    dec : array_like
        The declination (or latitude) angular coordinates of the data set in
        radians.
    n : int
        The number of iterations to use in Newton's method.

    Returns
    -------
    numpy.ndarray
        Array of auxiliary latitude values corresponding to the
        input latitude values.

    Notes
    -----
    This method finds the auxiliary latitude values using Newton's method, and
    is thus recursive.
    Wikipedia provides a simple form of the equation that is iterated upon
    in this method. See here for more details:
    https://en.wikipedia.org/wiki/Eckert_IV_projection
    """
    # This is here in case dec is a list rather than a numpy array.
    dec = np.asarray(dec)

    # First short circuit
    if n == 0:
        return dec / 2

    pi = math.pi

    # Array literally just filled with half pis.
    halfpi = np.empty(len(dec))
    halfpi.fill(pi / 2)

    theta = eckertiv_findtheta(dec, n-1)

    # Condition for the angle is pi/2 OR -pi/2
    cond1 = np.equal(theta, halfpi)
    cond2 = np.equal(theta, -1 * halfpi)
    cond = np.logical_or(cond1, cond2)

    # Choose the original value (pi/2 or -pi/2) if its true for equality
    # Otherwise use that value"s thetanew.
    # This is the eckertiv theta finding Newton's method.
    # It"s been broken up for style.
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    num = theta + np.multiply(s_theta, c_theta) + 2 * s_theta - (2 + pi/2) * np.sin(dec)
    denom = 2 * c_theta * (1 + c_theta)
    thetanew = theta - num / denom
    thetanew = np.where(cond, dec, thetanew)

    return thetanew


def eckertiv(ra, dec):
    """Find an Eckert-IV representation of the given coordinates.

    Parameters
    ----------
    ra : array_like
        The right ascension (or longitude) coordinates of the data set.
    dec : array_like
        The declination (or latitude) coordinates of the data set.

    Returns
    -------
    x : array_like
        The x coordinates corresponding to the given points.
    y : array_like
        The y coordinates corresponding to the given points.

    See Also
    --------
    eckertiv_findtheta : Newton's method for finding each point"s
                         auxiliary latitude value.

    Notes
    -----
    This method defines the x,y and latitude and longitude using the standard
    Eckert-IV definition. Wikipedia provides a simple form of the equations
    used in this method, including a defnition of the theta value that is
    found using Newton's method. See here for more details:
    https://en.wikipedia.org/wiki/Eckert_IV_projection

    """
    # Center latitude
    center = math.radians(180)

    # n = 5 seems to be sufficient for the shape.
    # This doesn"t converge as quickly as Mollweide
    theta = eckertiv_findtheta(np.radians(dec), 5)

    R = 100

    # For readability sake
    coeff = 1 / math.sqrt(math.pi * (4 + math.pi))

    # Eckert IV conversion functions.
    x = 2 * R * coeff * np.subtract(np.radians(ra), center) * (1 + np.cos(theta))
    y = 2 * R * math.pi * coeff * np.sin(theta)

    return(x, y)


def desi_patch():
    """Create axis patches corresponding to the DESI survey areas.

    Returns
    -------
    list
        List of matplotlib.patches.Patch objects representing the two DESI survey
        areas.

    Notes
    -----
    This method requires the file hull.txt to be in the module"s directory.
    This file can be downloaded from the kpno-allsky GitHub.

    """
    hull_loc = os.path.join(os.path.dirname(__file__), *["data", "hull.txt"])
    f = open(hull_loc, "r")

    # Converts the string representation of the list to a list of points.
    left = f.readline()
    left = ast.literal_eval(left)
    right = f.readline()
    right = ast.literal_eval(right)

    # Zorder parameter ensures the patches are on top of everything.
    patch1 = Polygon(left, closed=True, fill=False,
                     edgecolor="red", lw=2, zorder=4)
    patch2 = Polygon(right, closed=True, fill=False,
                     edgecolor="red", lw=2, zorder=4)

    f.close()

    return [patch1, patch2]


# This function sorts a set of values clockwise from the center.
# Pos defines whether or not the sort sorts anticlockwise from the positive x
# Or clockwise from the negative x.
# Anticlockwise = True, clockwise = False
def clockwise_sort(x, y, clockwise=True):
    """Sort a set of coordinates clockwise, element-wise.

    Parameters
    ----------
    x : array_like
        The set of x coordinates.
    y : array_like
        The set of y coordinates.
    clockwise : bool, optional
        If True, sorts clockwise, otherwise sorts anti-clockwise. Defaults to
        True.

    Returns
    -------
    x : array_like
        The sorted set of x coordinates.
    y : array_like
        The sorted set of y coordinates.

    Notes
    -----
    This method sorts a data set clockwise from the calculated center of the
    data. The center is found by taking the maximum of the sorted x and
    y values and then finding the midpoint between the two. While for some
    strange dataset this may not be the actual center (for example, a crescent
    moon), it is a reasonably fast approximation. The dataset will always
    be sorted by theta first, then radius. Points with the same angular
    distance from the sorting axis will be sorted by their radial distance.

    The sort is done clockwise starting from the negative x axis, or
    anticlockwise from the positive x axis. The reason for this is due to
    how atan2 returns angles as between -pi and pi instead of between pi and
    2pi.

    """
    # Finds the center of the circle ish object
    centerx = (np.min(x) + np.max(x))/2
    centery = (np.min(y) + np.max(y))/2

    x = np.subtract(x, centerx)
    y = np.subtract(y, centery)

    # Creates polar nonsense
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    # Reshape to stack
    r = np.reshape(r, (len(r), 1))
    theta = np.reshape(theta, (len(theta), 1))

    # If we want to sort from pos x, we need to ensure that the negative angles
    # Are actually big positive angles.
    if not clockwise:
        cond = np.less(theta, 0)
        theta = np.where(cond, theta + 2 * np.pi, theta)

    # Stack into list of form (theta,r) (we want to sort theta first)
    stack = np.hstack((theta, r))

    stack2 = []
    for i in stack:
        stack2.append(tuple(i))

    # Standard python sort by theta
    stack2 = sorted(stack2)

    # Now we just have to convert back!
    # Slice out theta and r
    stack2 = np.array(stack2)
    theta = stack2[:, 0]
    r = stack2[:, 1]

    x = r * np.cos(theta) + centerx
    y = r * np.sin(theta) + centery
    return (x, y)


if __name__ == "__main__":
    date = "20160220"
    img = image.load_image("r_ut020509s43380.png", date, "KPNO")
    transform(img)
