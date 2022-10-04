"""A module providing facilities for coordinate conversion.

This module provides methods that allow for conversion between the Cartesian
(x, y) image coordinates and two different astrophysical coordinate systems:
Horizontal (alt, az) and Equatorial (ra, dec).
Two methods account for irregularities in the lens during these
conversions. One method finds stars, and another finds the
difference between the expected and actual locations of an star in an image.
These two methods are used to verify the methods that correct for
irregularities in the lens.
"""

import math
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.time.core as aptime
from astropy import units as u
import numpy as np

# Globals
# Center of the circle found using super accurate photoshop layering technique
center_kpno = (256, 252)
center_sw = (512, 512)

# r - theta tables.
r_kpno = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
           5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
           10.5, 11, 11.5, 11.6]
theta_kpno = [0, 3.58, 7.17, 10.76, 14.36, 17.98, 21.62, 25.27,
               28.95, 32.66, 36.40, 40.17, 43.98, 47.83, 51.73,
               55.67, 59.67, 63.72, 67.84, 72.03, 76.31, 80.69,
               85.21, 89.97, 90]

r_sw = [0, 55, 110, 165, 220, 275, 330, 385, 435, 480, 510]
theta_sw = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

# Radec
stars = {"Polaris": (37.9461429,  89.2641378),
         "Altair": (297.696, 8.86832),
         "Vega": (279.235, 38.7837),
         "Arcturus": (213.915, 19.1822),
         "Alioth": (193.507, 55.9598),
         "Spica": (201.298, -11.1613),
         "Sirius": (101.2875, -16.7161)}


def xy_to_altaz(x, y, camera="KPNO"):
    """Convert a set of (x, y) coordinates to (alt, az) coordinates,
    element-wise.

    Parameters
    ----------
    x : array_like
        The x coordinates.
    y : array_like
        The y coordinates.
    camera : {"KPNO", "SW"}
        The camera used to take the image. "KPNO" represents the all-sky
        camera at Kitt-Peak. "SW" represents the spacewatch all-sky camera.
        Defaults to "KPNO".

    Returns
    -------
    alt : array_like
        The altitude coordinates. This is a scalar if x and y are scalars.
    az : array_like
        The azimuth coordinates. This is a scalar if x and y are scalars.

    Notes
    -----
    The altitude and azimuthal angles corresponding to each (x, y) position
    are determined using the position of the all-sky camera at the Kitt Peak
    National Observatory.
    """
    # Converts lists/numbers to np ndarrays for vectorwise math.
    x = np.asarray(x)
    y = np.asarray(y)

    # Point adjusted based on the center being at... well... the center.
    # And not the top left. In case you were confused.
    # Y is measured from the top stop messing it up.
    center  = center_sw if camera == "SW" else center_kpno
    # Spacewatch camera isn't perfectly flat, true zenith is 2 to the right
    # and 3 down from center. I tried doing the full geometric conversion for
    # this, rotating the camera plane across the axis that this corresponds to
    # and basically the real correction is so close to x -= 2 and y -=3 that
    # there's not really a point to doing expensive mathematics computations
    # that are going to simplify to this anyway.
    if camera == "SW":
        x -= 2
        y -= 3
    pointadjust = (x - center[0], center[1] - y)

    # We use -x here because the E and W portions of the image are flipped
    az = np.arctan2(-pointadjust[0], pointadjust[1])

    # atan2 ranges from -pi to pi but we need 0 to 2 pi.
    # So values with alt < 0 need to actually be the range from pi to 2 pi
    cond = np.less(pointadjust[0], 0)
    az = np.where(cond, az + 2 * np.pi, az)
    az = np.degrees(az)

    # Pythagorean thereom boys.
    r = np.hypot(pointadjust[0], pointadjust[1])

    # 90- turns the angle from measured from the vertical
    # to measured from the horizontal.
    # This interpolates the value from the two on either side of it.
    if camera == "SW":
        alt = 90 - np.interp(r, xp=r_sw, fp=theta_sw)
        az = az - 0.1 # Camera rotated 0.1 degrees.
    else:
        r = r * 11.6 / 240  # Magic pixel to mm conversion rate

        alt = 90 - np.interp(r, xp=r_kpno, fp=theta_kpno)
        # For now if r is on the edge of the circle or beyond
        # we'll have it just be 0 degrees. (Up from horizontal)
        cond = np.greater(r, 240)
        alt = np.where(cond, 0, alt)

        # Az correction
        az = az + .94444

    return (alt.tolist(), az.tolist())


def altaz_to_xy(alt, az, camera="KPNO"):
    """Convert a set of (alt, az) coordinates to (x, y) coordinates,
    element-wise.

    Parameters
    ----------
    alt : array_like
        The altitude coordinates.
    az : array_like
        The azimuth coordinates.
    camera : {"KPNO" "SW"}
        The camera used to take the image. "KPNO" represents the all-sky
        camera at Kitt-Peak. "SW" represents the spacewatch all-sky camera.
        Defaults to "KPNO".

    Returns
    -------
    x : array_like
        The x coordinates. This is a scalar if alt and az are scalars.
    y : array_like
        The y coordinates. This is a scalar if alt and az are scalars.

    Notes
    -----
    The altitude and azimuthal angles corresponding to each (x, y) position
    are determined using the position of the all-sky camera at the Kitt Peak
    National Observatory.
    """
    alt = np.asarray(alt)
    az = np.asarray(az)

    if camera == "SW":
        # Reverse of r interpolation
        r = np.interp(90 - alt, xp=theta_sw, fp=r_sw)
        az = az + 0.1 # Camera rotated 0.1 degrees.
    else:
        # Approximate correction (due to distortion of lens?)
        az = az - .94444

        # Reverse of r interpolation
        r = np.interp(90 - alt, xp=theta_kpno, fp=r_kpno)
        r = r * 240 / 11.6  # mm to pixel rate

    # Angle measured from vertical so sin and cos are swapped from usual polar.
    # These are x,ys with respect to a zero.
    x = -1 * r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))

    # y is measured from the top!
    center  = center_sw if camera == "SW" else center_kpno
    x = x + center[0]
    y = center[1] - y

    # Spacewatch camera isn't perfectly flat, true zenith is 2 to the right
    # and 3 down from center.
    if camera == "SW":
        x += 2
        y += 3

    pointadjust = (x.tolist(), y.tolist())

    return pointadjust


def altaz_to_radec(alt, az, time, camera="KPNO"):
    """Convert a set of (alt, az) coordinates to (ra, dec) coordinates,
    element-wise.

    Parameters
    ----------
    alt : array_like
        The altitude coordinates.
    az : array_like
        The azimuth coordinates.
    time : astropy.time.core.aptime.Time
        The time and date to use in the conversion.

    Returns
    -------
    ra : array_like
        The right-ascension coordinates. This is a scalar if alt and az are
        scalars.
    dec : array_like
        The declination coordinates. This is a scalar if alt and az are
        scalars.

    See Also
    --------
    timestring_to_obj : Convert a date and filename to an astropy.Time object.

    Notes
    -----
    The `time` parameter is used for the mapping from altitude and azimuth to
    right ascension and declination. Astropy is used to perform this conversion.

    """
    assert isinstance(time, aptime.Time), "Time should be an astropy Time Object."

    # This is the latitude/longitude of the camera
    if camera =="KPNO":
        camera_loc = (31.959417 * u.deg, -111.598583 * u.deg)
    else:
        camera_loc = (31.96164 * u.deg, -111.60022 * u.deg)

    cameraearth = EarthLocation(lat=camera_loc[0], lon=camera_loc[1],
                                height=2070 * u.meter)

    alt = np.asarray(alt)
    az = np.asarray(az)

    alt = alt * u.deg
    az = az * u.deg
    altazcoord = SkyCoord(alt=alt, az=az, frame="altaz",
                          obstime=time, location=cameraearth)
    radeccoord = altazcoord.icrs

    return (radeccoord.ra.degree, radeccoord.dec.degree)


def radec_to_altaz(ra, dec, time):
    """Convert a set of (ra, dec) coordinates to (alt, az) coordinates,
    element-wise.

    Parameters
    ----------
    ra : array_like
        The right ascension coordinates.
    dec : array_like
        The declination coordinates.
    time : astropy.time.core.aptime.Time
        The time and date to use in the conversion.

    Returns
    -------
    alt : array_like
        The altitude coordinates. This is a scalar if ra and dec are scalars.
    az : array_like
        The azimuth coordinates. This is a scalar if ra and dec are scalars.

    See Also
    --------
    timestring_to_obj : Convert a date and filename to an astropy.Time object.

    Notes
    -----
    The `time` parameter is used for the mapping from altitude and azimuth to
    right ascension and declination. Astropy is used to perform this conversion.
    """
    assert isinstance(time, aptime.Time), "Time should be an astropy Time Object."

    # This is the latitude/longitude of the camera
    camera = (31.959417 * u.deg, -111.598583 * u.deg)

    cameraearth = EarthLocation(lat=camera[0], lon=camera[1],
                                height=2120 * u.meter)

    # Creates the SkyCoord object
    radeccoord = SkyCoord(ra=ra, dec=dec, unit="deg", obstime=time,
                          location=cameraearth, frame="icrs",
                          temperature=5 * u.deg_C, pressure=78318 * u.Pa)

    # Transforms
    altazcoord = radeccoord.transform_to("altaz")

    return (altazcoord.alt.degree, altazcoord.az.degree)


def radec_to_xy(ra, dec, time, camera="KPNO"):
    """Convert a set of (ra, dec) coordinates to (x, y) coordinates,
    element-wise.

    Parameters
    ----------
    ra : array_like
        The right ascension coordinates.
    dec : array_like
        The declination coordinates.
    camera : {"KPNO" "SW"}
        The camera used to take the image. "KPNO" represents the all-sky
        camera at Kitt-Peak. "SW" represents the spacewatch all-sky camera.
        Defaults to "KPNO".
    time : astropy.time.core.aptime.Time
        The time and date to use in the conversion.

    Returns
    -------
    x : array_like
        The x coordinates. This is a scalar if ra and dec are scalars.
    y : array_like
        The y coordinates. This is a scalar if ra and dec are scalars.

    See Also
    --------
    timestring_to_obj : Convert a date and filename to an astropy.Time object.
    galactic_conv : Convert from expected galactic coordinates to image
                    coordinates accounting for distortions in the camera lens.

    Notes
    -----
    The `time` parameter is used for the mapping from altitude and azimuth to
    right ascension and declination.

    This method first converts the right ascension and declination coordinates
    to altitude and azimuth using radec_to_altaz. It then converts the altitude
    and azimuth coordinates to x and y using altaz_to_xy. From there, lens
    distortion is corrected for by using galactic_conv, and the final x and
    y coordinates are returned.
    """
    alt, az = radec_to_altaz(ra, dec, time)
    x, y = altaz_to_xy(alt, az, camera)
    if camera == "KPNO":
        return galactic_conv(x, y, az)
    else:
        return (x, y)


def xy_to_radec(x, y, time, camera="KPNO"):
    """Convert a set of (x, y) coordinates to (ra, dec) coordinates,
    element-wise.

    Parameters
    ----------
    x : array_like
        The x coordinates.
    y : array_like
        The y coordinates.
    camera : {"KPNO" "SW"}
        The camera used to take the image. "KPNO" represents the all-sky
        camera at Kitt-Peak. "SW" represents the spacewatch all-sky camera.
        Defaults to "KPNO".
    time : astropy.time.core.aptime.Time
        The time and date at which the image was taken.

    Returns
    -------
    ra : array_like
        The right-ascension coordinates. This is a scalar if x and y are
        scalars.
    dec : array_like
        The declination coordinates. This is a scalar if x and y are
        scalars.

    See Also
    --------
    timestring_to_obj : Convert a date and filename to an astropy.Time object.
    camera_conv : Convert from image coordinates to expected galactic
                  coordinates accounting for distortions in the camera lens.

    Notes
    -----
    The `time` parameter is used for the mapping from altitude and azimuth to
    right ascension and declination.

    This method first converts the x and y coordinates
    to altitude and azimuth using xy_to_altaz. It then converts the altitude
    and azimuth coordinates to x and y using camera_conv to correct for
    distortions in the fisheye lens. The x and y coordinates are converted once
    again to altitude and azimuth using xy_to_altaz, which are then converted
    final to right ascension and declination using altaz_to_radec.
    """
    alt, az = xy_to_altaz(x, y, camera)

    if camera == "KPNO":
        x, y = camera_conv(x, y, az)
        alt, az = xy_to_altaz(x, y)

    return altaz_to_radec(alt, az, time, camera)


# Converts a file name to a time object.
def timestring_to_obj(date, filename):
    """Convert a date and filename to an astropy Time object.

    Parameters
    ----------
    date : str
        The date on which the image was taken in yyyymmdd format.
    filename : str
        The image"s filename.

    Returns
    -------
    astropy.time.core.aptime.Time
        When the image was taken.
    """
    # Add the dashes
    formatted = date[:4] + "-" + date[4:6] + "-" + date[6:]

    # Extracts the time from the file name.
    # File names seem to be machine generated so this should not break.
    # Hopefully.
    time = filename[4:6] + ":" + filename[6:8] + ":" + filename[8:10]
    formatted = formatted + " " + time

    return Time(formatted)


def galactic_conv(x, y, az):
    """Convert from expected galactic coordinates to image coordinates
    accounting for distortions in the camera lens.

    Parameters
    ---------
    x : array_like
        The x coordinates.
    y : array_like
        The y coordiantes.
    az : array_like
        The azimuth coordinates.

    Returns
    -------
    x : array_like
        The x coordinates. This is a scalar if x, y, and az are scalars.
    y : array_like
        The y coordinates. This is a scalar if x, y, and az are scalars.

    Notes
    -----
    This method uses a model that depends on x, y and radial distance to
    correct for distortions in the fisheye lens. The radial distance is
    calculated from the x and y positions.
    Before the computation is performed, the azimuthal angle is
    decreased by 0.94444 to account for a mismatch between true north and
    north on the image.

    The exact mathematical model is as follows:

    .. math:: r_{new} = r + 2.369*\cos(0.997*(az - 42.088)) + 0.699
    .. math:: az_{new} = az + 0.716*\cos(1.015*(az + 31.358)) - 0.181

    where the azimuthal angle is in radians.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    az = np.asarray(az)

    # Convert to center relative coords.
    x = x - center_kpno[0]
    y = center_kpno[1] - y

    r = np.hypot(x, y)
    az = az - .94444

    # This was the best model I came up with.
    r = np.add(r, 2.369 * np.cos(np.radians(0.997 * (az - 42.088)))) + 0.699
    az = np.add(az, 0.716 * np.cos(np.radians(1.015 * (az + 31.358)))) - 0.181

    x = -1 * r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))

    # Convert to top left relative coords.
    x = x + center_kpno[0]
    y = center_kpno[1] - y

    return (x.tolist(), y.tolist())


# Converts from camera r,az to galactic r,az
def camera_conv(x, y, az):
    """Convert from image coordinates to expected galactic coordinates
    accounting for distortions in the camera lens.

    Parameters
    ---------
    x : array_like
        The x coordinates.
    y : array_like
        The y coordiantes.
    az : array_like
        The azimuth coordinates.

    Returns
    -------
    x : array_like
        The x coordinates. This is a scalar if x, y, and az are scalars.
    y : array_like
        The y coordinates. This is a scalar if x, y, and az are scalars.

    Notes
    -----
    This method uses a model that depends on x, y and radial distance to
    correct for distortions in the fisheye lens. The radial distance is
    calculated from the x and y positions.
    Before the computation is performed, the azimuthal angle is
    decreased by 0.94444 to account for a mismatch between true north and
    north on the image.

    The exact mathematical model is as follows:

    .. math:: az_{new} = az - 0.731*\cos(0.993*(az + 34.5)) + 0.181
    .. math:: r_{new} = r + 2.358*\cos(0.99*(az - 40.8)) - 0.729

    where the azimuthal angle is in radians. This conversion is performed in
    this order, and the corrected aziumthal angle is used in calculating the
    new radial distance.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    az = np.asarray(az)

    # Convert to center relative coords.
    x = x - center_kpno[0]
    y = center_kpno[1] - y

    r = np.hypot(x, y)

    # You might think that this should be + but actually no.
    # Mostly due to math down below in the model this works better as -.
    az = az - .94444  # - 0.286375

    az = np.subtract(az, 0.731 * np.cos(np.radians(0.993 * (az + 34.5)))) + 0.181
    r = np.subtract(r, 2.358 * np.cos(np.radians(0.99 * (az - 40.8)))) - 0.729

    x = -1 * r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))

    # Convert to top left relative coords.
    x = x + center_kpno[0]
    y = center_kpno[1] - y

    return (x.tolist(), y.tolist())


def find_star(img, centerx, centery, square=6):
    """Find a star near a given (x, y) coordinate.

    The search area for this method is defined as a square, centered at
    `centerx` and `centery`, with a sidelength of double the `square` parameter.

    Parameters
    ----------
    img : numpy.ndarray
        A greyscale image.
    centerx : float
        The x coordinate of the center of the search box.
    centery : float
        The y coordinate of the center of the search box.
    square : int, optional
        Half of the side length of the square to search within. Defaults to 6.

    Returns
    -------
    x : float
        The x coordinate of the star.
    y : float
        The y coordinate of the star.

    Notes
    -----
    This method uses a center of mass formula to search for
    a star. The search area is defined as a square of size 2* `square`
    centered on `centerx` and `centery`.
    The pixel values within this square are increased according to the formula:

    .. math:: w = \exp(v/10)

    where w is defined as the "weight" of the pixel and v is the greyscale
    pixel value. The average weight of all the pixels within the search square
    is calculated. Any weights that fall below the average
    are set to 0. Finally, the method finds the center of mass of the weights,
    which approximates the center of the star very well.

    In order to increase the method"s accuracy, it runs
    recursively. The discovered position of the star is used as the `centerx`
    and `centery` guesses for the next iteration. The size of the square is
    decreased by 2 with each iteration until it is less than 2. At this point
    the final position of the star is returned.

    """

    # We need to round these to get the center pixel as an int.
    centerx = np.int(round(centerx))
    centery = np.int(round(centery))
    # Just setting up some variables.
    R = (0, 0)
    M = 0

    # Fudge factor exists because I made a math mistake and somehow
    # it worked better than the correct mean.
    fudge = ((2 * square + 1)/(2 * square)) ** 2

    lowery = centery - square
    uppery = centery + square + 1
    lowerx = centerx - square
    upperx = centerx + square + 1

    temp = np.array(img[lowery: uppery, lowerx: upperx], copy=True)
    temp = temp.astype(np.float32)

    for x in range(0, len(temp[0])):
        for y in range(0, len(temp[0])):
            temp[y, x] = math.exp(temp[y, x]/10)

    averagem = np.mean(temp) * fudge
    # This is a box from -square to square in both directions
    # Range is open on the upper bound.
    for x in range(-square, square + 1):
        for y in range(-square, square + 1):
            m = img[(centery + y), (centerx + x)]

            m = math.exp(m/10)
            # Ignore the "mass" of that pixel
            # if it"s less than the average of the stamp
            if m < averagem:
                m = 0

            R = (m * x + R[0], m * y + R[1])
            M += m

    # Avoids divide by 0 errors.
    if M == 0:
        M = 1

    R = (R[0] / M, R[1] / M)
    star = (centerx + R[0], centery + R[1])

    # For some reason de-incrementing by 2 is more accurate than 1.
    # Don't ask me why, I don't understand it either.
    if square > 2:
        return find_star(img, star[0], star[1], square - 2)
    return star


# Returns a tuple of the form (rexpected, ractual, deltar)
# Deltar = ractual - rexpected
def delta_r(img, centerx, centery):
    """Find the difference between the calculated radial position of a star
    and the true radial position of the star.

    Parameters
    ----------
    img : numpy.ndarray
        A greyscale image.
    centerx : float
        The x coordinate of the star's center.
    centery : float
        The y coordinate of the star's center.

    Returns
    -------
    rexpected : float
        The radial position of the star as calculated from its true position.
    ractual : float
        The radial position of the star as it appears in the image.
    deltar : float
        The difference between rexpected and ractual.

    See Also
    --------
    find_star : Find a star near a given coordinate.

    Notes
    -----
    find_star is used to find the position of the star in the image using the
    true position of the star as the initial guess.

    This method is useful when performing a chi-squared analysis. Modifications
    to the model representing the irregularities in the lens will change
    the difference between the true radial distance and the radial distance
    in the saved image.

    """

    adjust1 = (centerx - center_kpno[0], center_kpno[1] - centery)

    rexpected = math.sqrt(adjust1[0] ** 2 + adjust1[1] ** 2)

    # If we think it's outside the circle then bail on all the math.
    # R of circle is 240, but sometimes the r comes out as 239.9999999
    if rexpected > 239:
        return (-1, -1, -1)

    # Put this after the bail out to save some function calls.
    star = find_star(img, centerx, centery)
    adjust2 = (star[0] - center_kpno[0], center_kpno[1] - star[1])

    ractual = math.sqrt(adjust2[0] ** 2 + adjust2[1] ** 2)
    deltar = ractual - rexpected

    return (rexpected, ractual, deltar)

if __name__ == "__main__":
    t = timestring_to_obj("20190606", "c_ut041405.jpg")
    print(xy_to_radec(436, 592, t, "SW"))

    print(xy_to_radec(702, 324, t, "SW"))