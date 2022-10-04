#!/usr/bin/env python3
"""A script that downloads and manipulates spacewatch images.

This file exists as a standalone script that is run separate from the rest of
the kpno project. The script downloads images from
http://varuna.kpno.noao.edu/allsky.htm and archives them. Upon download the
script extracts the timestamp of the image from the image, and then blacks out
three of the four corners.
"""

import datetime
import logging
import os
import subprocess
import time
from html.parser import HTMLParser

import daemon
import ephem
import numpy as np
import pytesseract
import requests
from PIL import Image
from requests.exceptions import (TooManyRedirects, HTTPError, ConnectionError,
                                 Timeout, RequestException)


# Sets up a pyephem object for the camera.
# Using the lat/long of the other KPNO camera for now.
camera = ephem.Observer()
camera.lat = "31.959417"
camera.lon = "-111.598583"
camera.elevation = 2120


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
            logging.error("Too many redirects.")
            return None
        # HTTPError is an error in the http code.
        except HTTPError:
            logging.error("HTTP error with status code " + str(data.status_code))
            return None
        # This is a failure in the connection unrelated to a timeout.
        except ConnectionError:
            logging.error("Failed to establish a connection to the link.")
            return None
        # Timeouts are either server side (too long to respond) or client side
        # (when requests doesn"t get a response before the timeout timer is up)
        # I have set the timeout to 5 seconds
        except Timeout:
            tries += 1

            if tries >= 3:
                logging.error("Timed out after three attempts.")
                return None

            # Tries again after 5 seconds.
            time.sleep(5)

        # Covers every other possible exceptions.
        except RequestException as err:
            logging.error("Unable to read link")
            print(err)
            return None
        else:
            logging.info(link + " read with no errors.")
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
        # All image names are held in tags of form <img=imagename>
        if tag == "img":
            for attr in attrs:
                # If the first attribute is href we need to ignore it
                if attr[0] == "src":
                    self.data.append(attr[1])

    def clear_data(self):
        """Clear the data list of this parser instance.
        """
        self.data = []


def download_image(date):
    """Download a single image.

    This method is of a similar form to
    :func:`io_util.download_image <io_util.download_image>`, where `date`
    provides the date to save the image to.

    Parameters
    ----------
    date : str
        Date to download images for, in the form yyyymmdd.

    Notes
    -----
    Over the course of the run time of this method various status updates will
    be logged to `date`/download.log. The method will exit early and fail to
    download the image with a failure print out.

    The Spacewatch images are located at http://varuna.kpno.noao.edu/allsky.htm

    Spacewatch does not maintain an archive of the images, and the method
    downloads the most recent image and saves it with its timestamp into a
    folder determined by `date`.
    """
    # Creates the link
    link = "http://varuna.kpno.noao.edu/allsky/AllSkyCurrentImage.JPG"

    # Collects originals in their own folder within Images
    time = datetime.datetime.now()

    directory = os.path.join("Images", *["Original", "SW", date])
    # This directory is for use on the blackbox server.
    #directory = os.path.join("/media", *["data1", "spacewatch", date])

    # Verifies that an Images folder exists, creates one if it does not.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Label the images in a similar way to the kpno ones.
    imagename = os.path.join(directory, "temp.jpg")

    rimage = download_url(link)

    if rimage is None:
        logging.error("Failed: " + time.strftime("%Y%m%d %H:%M:%S"))
        return

    # Saves the image
    with open(imagename, "wb") as f:
        f.write(rimage.content)

    # This extracts the text relevant portion of the image and resizes it so
    # that py tesseract can extract the text better.
    temp_im = Image.open(imagename).crop((120, 0, 240, 30)).resize((480, 120))
    text = pytesseract.image_to_string(temp_im)
    text = text.replace(":", "")
    try:
        # This tests to see if the time extraction worked correctly.
        test = int(text)
        # Renames the image.
        new_name = "c_ut" + text + ".jpg"
    # This is entered if the number from image extraction is wrong, we assume
    # the image is two minutes after the previous one.
    except:
        if len(os.listdir(directory)) > 2:
            # -3 since -2 will be download.log and -1 is temp.jpg
            prev = sorted(os.listdir(directory))[-3]
            num = int(prev[4:-4]) + 200
            if "58" in prev:
                num = num + 4000
            new_name = "c_ut" + str(num) + ".jpg"
        # This is a contingency in case the very first image fails, in which
        # case we use the last even minute as the time for the image.
        else:
            now = datetime.datetime.utcnow()
            minutes = now.minute // 2 * 2
            now = now.replace(minute=minutes, second=5)
            new_name = "c_ut" + now.strftime('%H%M%S') + ".jpg"

    os.rename(imagename, os.path.join(directory, new_name))
    block_text(directory, new_name)
    logging.debug("Downloaded: " + new_name)


def make_video(directory):
    """Make a time-lapse of images.

    This method joins the images in `directory` into a timelapse image.

    Parameters
    ----------
    directory : str
        Directory where the images are saved in the form yyyymmdd.

    Notes
    -----
    The method uses ffmpeg to create the output video. The video will be saved
    as `directory`/vid.mp4. The framerate of the image will be 15 fps and
    the video is encoded using x264. The ffmpeg command used to generate
    the video is ``ffmpeg -framerate 15 -pattern_type glob -i
    `directory`/c_ut*.jpg -c:v libx264 -r 60 -pix_fmt yuv420p
    `directory`/vid.mp4``
    """
    ff_cmd = ["ffmpeg",
              "-framerate", "15",
              "-pattern_type", "glob",
              "-i", os.path.join(directory, "c_ut*.jpg"),
              "-c:v", "libx264",
              "-r", "60",
              "-pix_fmt", "yuv420p",
              os.path.join(directory, "vid.mp4")]
    logging.debug("Making video")
    logging.debug(" ".join(ff_cmd))
    proc = subprocess.Popen(ff_cmd)

    # This string of commands should kill the process when it's done.
    proc.communicate()
    proc.kill()
    proc.communicate()


def block_text(directory, name):
    """Blocks the text in the corners of the images.

    Parameters
    ----------
    directory : str
        Directory where the images are saved in the form yyyymmdd.
    name: str
        Name of the image.

    Notes
    -----
    The top right, lower right and lower left corner text will be blacked out.
    The top left corner, containing the date and time at which the image was
    taken will be left.
    """
    im = np.array(Image.open(os.path.join(directory, name)))
    # The top right, lower right, and loewr left corners of text.
    im[1000:, 860:] = 0
    im[1000:, :120] = 0
    im[:25, 860:] = 0
    Image.fromarray(im).save((os.path.join(directory, name)))


def run_and_download():
    """Runs the script and downloads the images.

    See Also
    --------
    download_image : Download a single image.
    block_text : Blocks the text in the corners of the images.
    make_video : Make a time-lapse of images.

    Notes
    -----
    This method exectures the code flow of the script. First the next sunset
    and next sunrise are determined. If it is daytime, the script sleeps until
    the next sunset. If it is currently nighttime, the script skips straight
    to start downloading. At that time it creates a folder to contain the
    images. The script downloads the image, and blacks out the corner text.
    At the end of the night the downloaded images are joined into a time-lapse
    video. It then sleeps until the next evening.
    """
    while True:
        sun = ephem.Sun()
        # Update the camera date to update next rising.
        camera.date = datetime.datetime.utcnow()
        setting = camera.next_setting(sun, use_center=True).datetime()
        rising = camera.next_rising(sun, use_center=True).datetime()
        now = datetime.datetime.utcnow()

        # If the next rising is before the next setting then we're in the night
        if not rising < setting:
            print("Current time:", now)
            print("Setting at:", setting)
            print("Rising at:", rising)
            delta = (setting - now).total_seconds()
            # Sleeps until the sun sets.
            time.sleep(delta)

        print("Sunset arrived, starting download.")
        day = datetime.datetime.utcnow().strftime("%Y%m%d")
        directory = os.path.join("Images", *["Original", "SW", day])

        # This directory is for use on the blackbox server.
        #directory = os.path.join("/media", *["data1", "spacewatch", day])

        # Verifies that an Images folder exists, creates one if it does not.
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_name = os.path.join(directory, "download.log")
        logging.basicConfig(filename=log_name, level=logging.DEBUG)
        logger = logging.getLogger()

        now = datetime.datetime.utcnow()
        while now < rising:
            try:
                download_image(day)

                # Sleeps until 6 seconds after two minutes from now.
                sleep_until = (datetime.datetime.now() + datetime.timedelta(seconds=120)).replace(second=6)
                sleep_for = (sleep_until - datetime.datetime.now()).total_seconds()
                time.sleep(sleep_for)

                now = datetime.datetime.utcnow()
            except Exception as e:
                logging.error(e)
                raise(e)

        # Makes the video at the end of the night
        make_video(directory)

        # This closes the old log.
        logger.handlers[0].stream.close()
        logger.removeHandler(logger.handlers[0])


if __name__ == "__main__":
    with daemon.DaemonContext():
        run_and_download()

