"""A module providing facilities to darken clouds in images.

Images taken with the camera at the Kitt Peak National Observatory are
taken with either 0.3-second or 6-second exposure times. This module provides
support for finding and darkening clouds for each of the two exposure
times.
"""
import math
import numpy as np
from scipy import ndimage
from PIL import Image

import mask
import io_util
from image import AllSkyImage
import image


center = (256, 252)


def cloud_contrast(img):
    """Darken cloud pixels in an image.

    This is a convenience method that will determine the appropriate method
    to use.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.
    Returns
    -------
    numpy.ndarray
        A higher contrast version of the original image.

    See Also
    --------
    zero_three_cloud_contrast: Darken cloud pixels in images with 0.3 second
     exposure times.
    six_cloud_contrast: Darken cloud pixels in images with 6 second
     exposure times.
    """
    exposure = image.get_exposure(img)
    print(exposure)

    if exposure == 0.3:
        return zero_three_cloud_contrast(img)
    elif exposure == 6:
        return six_cloud_contrast(img)

    return img


def zero_three_cloud_contrast(img):
    """Darken cloud pixels in an image taken with an exposure time of 0.3
    seconds.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    numpy.ndarray
        A higher contrast version of the original image.

    Notes
    -----
    In order to first determine which pixels should be considered clouds
    this method first finds the difference between the pixel at position (510,
    510) in the given image and the image taken at 05:29:36 on
    November 8, 2017. This difference is then subtracted from all the pixels,
    normalizing the image to have the same background pixel value. A greyscale
    closing is then performed, smudging out white pixel noise.

    The average value of all the pixels in this new normalized image is
    calculated and any pixel that is above this average value is considered a
    cloud pixel. This is because light reflected off the moon illuminates
    the clouds, raising them above the average pixel value.

    Once the cloud pixels are found, all non-cloud pixels are raised in value
    by 40, while the cloud pixels are reduced to 0.
    """
    # Temprary, I intend to change this slightly later.
    img2 = np.asarray(Image.open("Images/Original/KPNO/20171108/r_ut052936s31200.png").convert("L"))

    img3 = np.copy(img.data)
    img1 = np.int16(img.data)
    img2 = np.int16(img2)

    # Finds the difference from the "standard" .03s image.
    # Then subtracts that value from the entire image to normalize it to
    # standard image color.
    val = img1[510, 510] - img2[510, 510]
    img1 = img1 - val

    # Subtracts standard image from current image.
    # Performs closing to clean up some speckling in lower band of image.
    test = io_util.image_diff(img1, img2)
    test = ndimage.grey_closing(test, size=(2, 2))

    # Clouds are regions above the average value of the completed transform.
    avg = np.mean(test)
    cond = np.where(test > avg, 0, 1)

    # Increases black sky brightness in images where the moon is alone (thanks
    # to low dynamic range the sky is black because the moon is so bright)
    img3 = np.where(img3 < 150, img3 + 40, img3)
    final = np.multiply(img3, cond)

    # Find the mask and black out those pixels.
    masking = mask.generate_mask()

    final = AllSkyImage(img.name, img.date, img.camera, final)
    final = mask.apply_mask(masking, final)

    return final


def six_cloud_contrast(img):
    """Darken cloud pixels in an image taken with an exposure time of 6 seconds.

    Parameters
    ----------
    img : image.AllSkyImage
        The image.

    Returns
    -------
    numpy.ndarray
        A higher contrast version of the original image.

    Notes
    -----
    At the start of this method, the dead pixels and horizon objects are
    masked out. The image is inverted, and subtracted from itself four times.
    This highly increases the contrast between the clouds (which fall close to
    0 in the original pixel value) and the background,
    which will get reduced to 0. A copy of this image is used later in a
    separate calculation. Meanwhile, a greyscale closing is performed on this
    resulting image, which smooths out stars that were turned into small black
    dots in the inversion process.

    This result then gets thresholded, which creates a two tone, black and white
    version of the image. This is done by making each pixel with a value above
    10 as white and anything below as black.
    A binary closing is performed to remove any created
    singular white pixels. The horizon items are once again masked out, and
    a buffer circle of black pixels is created around the image content. As
    a result, the image is filled with white regions
    that correspond to the original clouds, with the rest of the image being
    black.

    In some images, however, the center of the Milky Way is bright enough to
    be recorded at a pixel value approximately equal to the value at which
    the clouds appear. To account for this, for each white region in the binary
    image, count the number of stars in the original image that would appear
    within that region. Removes any region where the density of stars
    is too high to be a cloud. This leaves a binary image with clouds in
    white, and everything else in black.

    From here, a scaling darkness fraction is determined by
    the original inversion image. Cloud pixels that are close to white
    in the inversion, from the darkest regions of the clouds, are scaled
    to 0, while the rest of the pixels are scaled less dark. This preserves
    the large scale structure of the clouds, but reduces them in brightness
    to nearly 0. The exact formula used to calculate this scaling darkness
    is 0.6 - (inverted pixel value) / 255.
    """

    # Find the mask and black out those pixels.
    masking = mask.generate_mask()
    img1 = mask.apply_mask(masking, img)

    # Inverts and subtracts 4 * the original image. This replicates previous
    # behaviour in one step.
    # Previous work flow: Invert, subtract, subtract, subtract.
    # If it goes negative I want it to be 0 rather than positive abs of num.
    invert = 255 - 4 * np.int16(img1.data)
    invert = np.where(invert < 0, 0, invert)

    # Smooth out the black holes left where stars were in the original.
    # We need them to be "not black" so we can tell if they"re in a region.
    closedimg = ndimage.grey_closing(invert, size=(2, 1))

    # Thresholds the image into black and white with a value of 10.
    # Pixels brighter than greyscale 10 are white, less than are 0.
    binimg = np.where(closedimg > 10, 1, 0)

    # Cleans up "floating" white pixels.
    binimg = ndimage.binary_opening(binimg)

    # Mask out the horizon objects so they don"t mess with cloud calculations.
    img1.data = binimg
    binimg = mask.apply_mask(masking, img1).data

    # Expand the white areas to make sure they cover the items they represent
    # from the inverted image.
    binimg = ndimage.binary_dilation(binimg)

    # Creates a buffer circle keeping the image isolated from the background.
    for row in range(0, binimg.shape[1]):
        for column in range(0, binimg.shape[0]):
            x = column - center[0]
            y = center[1] - row
            r = math.hypot(x, y)
            if (r < 246) and (r > 241):
                binimg[row, column] = 0

    # This structure makes it so that diagonally connected pixels are part of
    # the same region.
    struct = [[True, True, True], [True, True, True], [True, True, True]]
    labeled, num_features = ndimage.label(binimg, structure=struct)
    regionsize = [0] * (num_features + 1)
    starnums = [0] * (num_features + 1)

    for row in range(0, binimg.shape[1]):
        for column in range(0, binimg.shape[0]):
            regionsize[labeled[row, column]] += 1

            # This finds stars in "cloud" regions
            # Basically, if somewhat bright, and the region is marked "cloud."
            if img1.data[row, column] >= (95) and binimg[row, column] == 1:
                x = column - center[0]
                y = center[1] - row
                r = math.hypot(x, y)
                if r <= 240:
                    regionnum = labeled[row, column]
                    starnums[regionnum] += 1

    # The reason why I use density is mainly because of very small non-clouds.
    # They contain few stars, which rules out a strictly star count method.
    # This, however, is actually density^-1. I.e. it"s size/stars rather than
    # stars/size. This is because stars/size is very small sometimes.
    # I"m aware of a division by 0 warning here. If a region has no stars, then
    # this divides by 0. In fact this np.where exists to ignore that and set
    # zero star regions to a density of 0, since I ignore those later.
    # Hence I"m supressing the divide by 0 warning for these two lines.
    with np.errstate(divide="ignore"):
        density = np.divide(regionsize, starnums)
        density = np.where(np.asarray(starnums) < 1, 0, density)

    # Zeroes out densities < 12
    density = np.where(density < 12, 0, density)
    density[0] = 350

    # Creates a density "image".
    # This is an image where each feature has its value set to its density.
    for row in range(0, labeled.shape[1]):
        for column in range(0, labeled.shape[0]):
            value = labeled[row, column]
            labeled[row, column] = density[value]

    # If the value is less than the mean density, we want to mask it in the
    # "map" image. Hence set it to 0, everything else to 1, and multipy.
    # This keeps the non masks (x*1 = x) and ignores the others (x*0 = 0)
    m = np.mean(density[np.nonzero(density)])
    masked = np.where(labeled < m, 0, 1)
    invert2 = np.multiply(invert, masked)

    # The thinking here is that the whiter it is in the contrast++ image, the
    # darker it should be in the original. Thus increasing cloud contrast
    # without making it look like sketchy black blobs.
    multiple = .6 - invert2 / 255

    # Resets the img1 data since I used the img1 object to mask the binary.
    img1 = mask.apply_mask(masking, img)
    newimg = np.multiply(img1.data, multiple)

    # Creates a new AllSkyImage so that we don"t modify the original.
    new = AllSkyImage(img.name, img.date, img.camera, newimg)

    return new

