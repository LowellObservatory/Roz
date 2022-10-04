"""A module providing facilities for loading, saving, and downloading

Methods are provided for downloading and saving images taken at two different
all-sky cameras. These cameras are located at Kitt Peak, designated KPNO, and
at the Multiple Mirror Telescope Observatory, designated MMTO.
One class is provided to read the raw HTML provided by each camera"s website.
"""
import glob
import os
import time
from html.parser import HTMLParser
import numpy as np
import requests
from requests.exceptions import (TooManyRedirects, HTTPError, ConnectionError,
                                 Timeout, RequestException)
from PIL import Image


def download_url(link):
    """Read the data at a url.

    Parameters
    ----------
    link : str
        The link to access and download data from.

    Returns
    -------
    requests.Response or None
        A requests.Response object containing data on success,
        or None on failure.

    """
    tries = 0
    read = False

    while not read:
        try:
            # Tries to connect for 5 seconds.
            data = requests.get(link, timeout=5)

            # Raises the HTTP error if it occurs.
            data.raise_for_status()
            read = True

        # Too many redirects is when the link redirects you too much.
        except TooManyRedirects:
            print("Too many redirects.")
            return None
        # HTTPError is an error in the http code.
        except HTTPError:
            print("HTTP error with status code " + str(data.status_code))
            return None
        # This is a failure in the connection unrelated to a timeout.
        except ConnectionError:
            print("Failed to establish a connection to the link.")
            return None
        # Timeouts are either server side (too long to respond) or client side
        # (when requests doesn"t get a response before the timeout timer is up)
        # I have set the timeout to 5 seconds
        except Timeout:
            tries += 1

            if tries >= 3:
                print("Timed out after three attempts.")
                return None

            # Tries again after 5 seconds.
            time.sleep(5)

        # Covers every other possible exceptions.
        except RequestException as err:
            print("Unable to read link")
            print(err)
            return None
        else:
            print(link + " read with no errors.")
            return data


class DateHTMLParser(HTMLParser):
    """Parser for data passed from image websites.

    Attributes
    ----------
    data : list
        Extracted data from the image website HTML.
    """
    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []

    def handle_starttag(self, tag, attrs):
        """Extract image links from the HTML start tag.

        Parameters
        ----------
        tag : str
            The start tag
        attrs : list
            The attributes attached to the corresponding `tag`.

        """
        # All image names are held in tags of form <A HREF=imagename>
        if tag == "a":
            for attr in attrs:
                # If the first attribute is href we need to ignore it
                if attr[0] == "href":
                    self.data.append(attr[1])

    def clear_data(self):
        """Clear the data list of this parser instance.

        """
        self.data = []


def download_all_date(date, camera="kpno"):
    """Download all images for a given date and all-sky camera.

    Parameters
    ----------
    date : str
        Date to download images for, in yyyymmdd format.
    camera : str, optional
        Camera to download images from. Defaults to `kpno` (the all-sky camera
        at Kitt-Peak) but may be specified instead as `mmto` (the all-sky
        camera at the MMT Observatory) or `sw` (the all-sky camera at the
        Spacewatch collaboration).

    See Also
    --------
    download_image : Images are downloaded using download_image.

    Notes
    -----
    Over the course of the run time of this method various status updates will
    be printed. The method will exit early with a print out of what happened.

    Images will be saved to Images/Original/`camera`/`date`/.

    The Kitt-Peak National Observatory images are located at
    http://kpasca-archives.tuc.noao.edu/.

    The MMT Observatory images are located at
    http://skycam.mmto.arizona.edu/skycam/.

    The Spacewatch images are located at
    http://varuna.kpno.noao.edu/allsky-all/images/cropped/.
    """
    links = {"kpno": "http://kpasca-archives.tuc.noao.edu/",
             "mmto": "http://skycam.mmto.arizona.edu/skycam/",
             "sw": "http://varuna.kpno.noao.edu/allsky-all/images/cropped/"}

    # Creates the link
    if camera.lower() != "sw":
        link = links[camera] + date
    else:
        link = links[camera]

    # Gets the html for a date page,
    # then parses it to find the image names on that page.
    if camera.lower() == "kpno":
        htmllink = link + "/index.html"
    elif camera.lower() == "sw":
        htmllink = link + date[0:4] + "/" + date[4:6] + "/" + date[6:] + "/"
        print(htmllink)
    else:
        htmllink = link

    rdate = download_url(htmllink)

    if rdate is None:
        print("Failed to download dates.")
        return

    # Makes sure the date exists.
    if rdate.status_code == 404:
        print("Date not found.")
        return

    htmldate = rdate.text
    parser = DateHTMLParser()
    parser.feed(htmldate)
    parser.close()
    imagenames = parser.data

    # Strips everything that's not an image.
    ext = ".png"
    if camera.lower() == "mmto": ext = "fits"
    elif camera.lower() == "sw": ext = ".jpg"

    imagenames2 = []
    for item in imagenames:
        if item[-4:] == ext:
            imagenames2.append(item)
    imagenames = imagenames2

    # Runs through the array of image names and downloads them
    for image in imagenames:
        # We want to ignore the all image animations
        if image == "allblue.gif" or image == "allred.gif" or image[:1] == "b":
            continue
        # Otherwise request the html data of the page for that image
        # and save the image
        else:
            download_image(date, image, camera)

    print("All photos downloaded for " + date)


def download_image(date, image, camera="kpno", directory=None):
    """Download a single image.

    This method is of a similar form to download_all_date, where `date`
    provides the date and `camera` provides the camera. `image` is the name
    of the image to be downloaded.

    Parameters
    ----------
    date : str
        Date to download images for, in the form yyyymmdd.
    image : str
        Image name to download.
    camera : str, optional
        Camera to download images from. Defaults to `kpno` (the all-sky camera
        at Kitt-Peak) but may be specified instead as `mmto` (the all-sky
        camera at the MMT Observatory) or `sw` (the all-sky camera at the
        Spacewatch collaboration).
    directory : str, optional
        The directory to save the downloaded image to. Defaults to
        Images/Original/`camera`.upper()/`date`.

    Notes
    -----
    Over the course of the run time of this method various status updates will
    be printed. The method will exit early and fail to download the image
    with a failure print out.

    The Kitt-Peak National Observatory images are located at
    http://kpasca-archives.tuc.noao.edu/.

    The MMT Observatory images are located at
    http://skycam.mmto.arizona.edu/skycam/.

    The Spacewatch images are located at
    http://varuna.kpno.noao.edu/allsky-all/images/cropped/.
    """
    links = {"kpno": "http://kpasca-archives.tuc.noao.edu/",
             "mmto": "http://skycam.mmto.arizona.edu/skycam/",
             "sw": "http://varuna.kpno.noao.edu/allsky-all/images/cropped/"}

    # Creates the link
    if camera.lower() != "sw":
        link = links[camera] + date
    else:
        link = links[camera] + date[0:4] + "/" + date[4:6] + "/" + date[6:]

    # Collects originals in their own folder within Images
    if not directory:
        directory = "Images/Original/" + camera.upper() + "/" + date
    # Verifies that an Images folder exists, creates one if it does not.
    if not os.path.exists(directory):
        os.makedirs(directory)

    imageloc = link + "/" + image

    if camera.lower() != "sw":
        imagename = directory + "/" + image
    else:
        imagename = directory + "/c_ut" + image[-10:]

    rimage = download_url(imageloc)

    if rimage is None:
        print("Failed: " + imagename)
        return

    # Saves the image
    with open(imagename, "wb") as f:
        f.write(rimage.content)
    print("Downloaded: " + imagename)


def load_all_date(date, camera="KPNO"):
    """Load all images for a given date.

    Parameters
    ----------
    date : str
        The date in yyyymmdd format.
    camera : {"KPNO", "SW"}
            The camera used to take the image. "KPNO" represents the all-sky
            camera at Kitt-Peak. "SW" represents the spacewatch all-sky camera.

    Returns
    -------
    numpy.ndarray
        An ``ndarray`` that contains all images for that date. ``ndarray`` is
        of the shape (512, 512, 4, N) where N is the number of images for
        that day.

    See Also
    --------
    gray_and_color_image : Method used to load images.

    """
    directory = os.path.join("Images", *["Original", camera, date])

    # In theory this is only ever called from median_all_date.
    # Just in case though.
    try:
        if camera.lower() == "sw":
            files = sorted(glob.glob(os.path.join(directory, "*.jpg")))
        else:
            files = sorted(glob.glob(os.path.join(directory, "*.png")))
    except:
        print("Images directory not found for that date!")
        print("Are you sure you downloaded images?")
        exit()

    imgs = []

    # Up to 7 seconds quicker than the old method!
    # Has a bonus of being way way easier to read.
    for i, f in enumerate(files):
        temp = gray_and_color_image(f)
        imgs.append(temp)

    return np.concatenate(imgs, axis=3)


def gray_and_color_image(file):
    """Load an image in both grayscale and color.

    Load an image and return an image where each pixel is represented by a
    four item list, of the form [L, R, G, B] where L is the luma grayscale
    value.

    Parameters
    ----------
    file : str
        The location of the image to be read.

    Returns
    -------
    numpy.ndarray
        The ndarray representing the grayscale and color combination image.

    See Also
    --------
    PIL.Image.Image.convert : For more details on the ITU-R 601-2 luma
                              grayscale transform used by this method.

    Notes
    -----

    The Pillow documentation includes the following definition of the
    ITU-R 601-2 luma grayscale transform:

        L = R * 299/1000 + G * 587/1000 + B * 114/1000

    """

    img1 = np.asarray(Image.open(file).convert("RGB"))
    img2 = np.asarray(Image.open(file).convert("L"))

    # Reshape to concat
    img2 = img2.reshape(img2.shape[0], img2.shape[1], 1)
    img1 = np.concatenate((img2, img1), axis=2)

    # Return the reshaped image
    return img1.reshape(img1.shape[0], img1.shape[1], 4, 1)


def image_diff(img1, img2):
    """Find the mathematical difference between two grayscale images.

    Parameters
    ----------
    img1 : numpy.ndarray
        The first image.
    img2 : numpy.ndarray
        The second image.

    Returns
    -------
    numpy.ndarray
        The difference image.

    Notes
    -----
    The order of the parameters does not matter. In essence,
    image_diff(img1, img2) == image_diff(img2, img1).

    Greyscale values in the returned image represent the difference between
    the images. Black means the pixels were identical in both images, whereas
    white represents the maximum difference between the two,
    where in one image the pixel is white and in one it is black.
    """
    # I encountered a problem previously, in that
    # I assumed the type of the array would dynamically change.
    # This is python, so that"s not wrong per se.
    # Anyway turns out it"s wrong so I have to cast these to numpy ints.
    # I then have to cast back to uints because imshow
    # works differently on uint8 and int16.
    diffimg = np.uint8(abs(np.int16(img1) - np.int16(img2)))

    return diffimg


if __name__ == "__main__":
    download_all_date("20200316", "sw")
