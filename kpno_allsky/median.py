"""A module providing facilities for creating and saving median images.

A median image for a date is created by collating all the images taken that
night and finding the median greyscale value for every pixel position.
A method is provided to save the median image.
Two other methods are used to support the median creation: one implementing
the median of medains algorithm and one that converts a list of lists to a
list of tuples.
"""
import glob
import os
import numpy as np

import io_util
import image
from image import AllSkyImage


# This is necessary.
# Python works on tuples as required for color medians, which means we need to
# turn the numpy color array of length 3 into a tuple.
def ndarray_to_tuplelist(arr):
    """Convert an ndarray to a list of tuples.

    For an ndarray of shape (y, x) returns a list of y tuples where each tuple
    is of length x.

    Parameters
    ----------
    arr : ndarray
        A NumPy ndarray to be converted.

    Returns
    -------
    list
        A list of tuples.

    """
    templist = []

    # Runs over the second dimension (the longer one)
    for i in range(0, arr.shape[1]):
        tup = (arr[0, i], arr[1, i], arr[2, i], arr[3, i])
        templist.append(tup)

    return templist


# This works as wanted for tuples yay!
def median_of_medians(arr, i):
    """Find the ith smallest element of a list using the median of medians algorithm.

    Parameters
    ----------
    arr : array_like
        An array_like object of floats.
    i : int
        The positional rank.

    Returns
    -------
    float
        The ith smallest element of arr.

    Notes
    -----
    i = len(arr) // 2 corresponds to finding the median of arr. Details on the
    median of medians algorithm can be found at Wikipedia
    (https://en.wikipedia.org/wiki/Median_of_medians).
    """

    # Divide the array into sublists of length 5 and find the medians.
    sublists = []
    medians = []

    for j in range(0, len(arr), 5):
        temp = arr[j:j+5]
        sublists.append(temp)

    for sublist in sublists:
        medians.append(sorted(sublist)[len(sublist)//2])

    if len(medians) <= 5:
        pivot = sorted(medians)[len(medians)//2]
    else:
        # Find the median of the medians array using this method.
        pivot = median_of_medians(medians, len(medians)//2)

    low = [j for j in arr if j < pivot]
    high = [j for j in arr if j > pivot]
    identicals = [j for j in arr if j == pivot]

    lownum = len(low)
    # This edit is required to make sure this is valid for lists with dupes.
    identnum = len(identicals)

    if i < lownum:
        return median_of_medians(low, i)
    elif i < identnum + lownum:
        return pivot
    return median_of_medians(high, i - (lownum + identnum))


def median_all_date(date, camera="KPNO", color=False):
    """Find the median images for a given date.

    Parameters
    ----------
    date : str
        The date to find median images for.
    camera : {"KPNO", "SW"}
            The camera used to take the image. "KPNO" represents the all-sky
            camera at Kitt-Peak. "SW" represents the spacewatch all-sky camera.
    color : bool, optional
        If true, finds the median images in color, otherwise works in grayscale.
        Defaults to False.

    Returns
    -------
    dict of ndarrays
        A dictionary mapping exposure times to their median images.

    See Also
    --------
    io_util.load_all_date : Load images in color to find color medians.

    """
    directory = os.path.join(os.path.dirname(__file__), *["Images", "Original",
                                                          camera, date])

    # Gotta make sure those images exist.
    try:
        # This only finds images and ignores videos or .DS_Store
        if camera == "SW":
            files = sorted(glob.glob(os.path.join(directory, "*.jpg")))
        else:
            files = sorted(glob.glob(os.path.join(directory, "*.png")))
    except:
        print("Images directory not found for that date!")
        print("Are you sure you downloaded images?")
        exit()

    # These dictionaries hold the images and existence booleans.
    keys = ["All", 0.02, 0.3, 6]

    finalimg = {}
    superimg = {}
    exists = {}

    # By doing this with an array you can add more medians
    # just by adding them to the keys array.
    if not color:
        for key in keys:
            finalimg[key] = np.zeros((512, 512))
            superimg[key] = np.zeros((1, 1, 1))
            exists[key] = False
    else:
        for key in keys:
            finalimg[key] = np.zeros((512, 512, 3))
            superimg[key] = np.zeros((1, 1, 1, 1))

    # If not color load all the ones and seperate by exposure time
    # If color, then just load all of them ignoring exposure, for now.
    if not color:
        for f in files:
            f = f.split("/")[-1]
            # We have to reshape the images so that the lowest level
            # single value is a 1D array rather than just a number.
            # This is so when you concat the arrays it actually turns the
            # lowest value into a multivalue array.
            img = image.load_image(f, date, camera)
            temp = img.data.reshape(img.data.shape[0], img.data.shape[1], 1)

            # All Median
            # Make the super image have the correct
            # dimensions and starting values.
            # Concats if it already does.
            if exists["All"]:
                # Concatenates along the color axis
                superimg["All"] = np.concatenate((superimg["All"], temp), axis=2)
            else:
                # Since we run this only once this shortcut will save us
                # fractions of a second!
                superimg["All"] = temp
                exists["All"] = True

            # Exposure specific medians
            if camera == "KPNO":
                exposure = image.get_exposure(img)
                if exists[exposure]:
                    superimg[exposure] = np.concatenate((superimg[exposure], temp), axis=2)
                else:
                    superimg[exposure] = temp
                    exists[exposure] = True
    else:
        superimg["All"] = io_util.load_all_date(date, camera)

    print("Loaded images for", date)

    # Axis 2 is the color axis (In RGB space 3 is the color axis).
    # Axis 0 is y, axis 1 is x iirc.
    # Can you believe that this is basically the crux of this method?
    for key in keys:

        # If not color we can use magic np median techniques.
        if not color:
            finalimg[key] = np.median(superimg[key], axis=2)
        # In color we use the median of median because rgb tuples.
        else:
            # Let's run this loop as little as possible thanks.
            if not np.array_equal(superimg[key], np.zeros((1, 1, 1, 1))):
                supe = superimg[key]

                # Need to resize the final image to the same dimensions
                shape = superimg[key].shape
                finalimg[key] = np.zeros((shape[0], shape[1], 3))
                x = 0
                y = 0
                for row in supe:
                    for column in row:
                        tuples = ndarray_to_tuplelist(column)
                        median = median_of_medians(tuples, len(tuples) // 2)
                        finalimg[key][y][x] = [median[1], median[2], median[3]]
                        x += 1
                    y += 1
                    x = 0
    print("Median images complete for " + date)
    return finalimg


def save_medians(medians, date, color=False):
    """Save a dict of medians produced by median_all_date.

    Parameters
    ----------
    medians: dict
        Dictionary mapping exposure times to their median images.
    date : str
        The date of the median images.
    color : bool, optional
        If True, saves the median images in color, otherwise works in grayscale.
        Defaults to False.

    See Also
    --------
    image.save_image : Save an image.
    median_all_date : Generate median images for a given date.

    Notes
    -----
    Saves median images to Images/median/`date`/ if the medians are grayscale,
    and Images/median-color/`date`/ if the medians are in color.

    """
    if not color:
        loc = os.path.join(os.path.dirname(__file__), *["Images", "median", date])
        cmap = "gray"
    else:
        loc = os.path.join(os.path.dirname(__file__), *["Images", "median-color", date])
        cmap = None

    for key, median in medians.items():
        name = str(key).replace(".", "")

        # If blocks to only save the ones with actual data
        if not color and not np.array_equal(median, np.zeros((1, 1))):
            img = AllSkyImage(name, None, None, median)
            image.save_image(img, loc, cmap)

        elif color and not np.array_equal(median, np.zeros((512, 512, 3))):
            img = AllSkyImage(name, None, None, np.uint8(median))
            image.save_image(img, loc, cmap)


if __name__ == "__main__":
    dates = sorted(os.listdir(os.path.join("Images", "Original", "KPNO")))
    if ".DS_Store" in dates:
        print("Removing DS_Store")
        dates.remove(".DS_Store")
    for date in dates:
        print(date)
        medians = median_all_date(date, camera="KPNO")
        save_medians(medians, date)
