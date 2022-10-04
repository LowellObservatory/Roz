#!/usr/bin/env python3
"""A script that downloads TLE files.

This file exists as a standalone script that is run separate from the rest of
the kpno project. The script downloads every TLE from
http://www.celestrak.com/NORAD/elements/. Upon download the TLE files are joined
into one large TLE that is then compressed using gzip.
"""

import datetime
import gzip
import logging
import os
import shutil
import subprocess
import time
from html.parser import HTMLParser

#import daemon
import requests
from requests.exceptions import (TooManyRedirects, HTTPError, ConnectionError,
                                 Timeout, RequestException)

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


class TLEHTMLParser(HTMLParser):
    """Parser for finding TLE files.

    Attributes
    ----------
    data : list
        Extracted data from the TLE website HTML.
    """
    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []

    def handle_starttag(self, tag, attrs):
        """Extract TLE links from the HTML start tag.

        Parameters
        ----------
        tag : str
            The start tag
        attrs : list
            The attributes attached to the corresponding `tag`.

        """
        # TLEs are stored in files stored in tags <a href="name.txt">
        if tag == "a":
            if attrs[0][0] == "href" and attrs[0][1].endswith(".txt"):
                    self.data.append(attrs[0][1])

    def clear_data(self):
        """Clear the data list of this parser instance.
        """
        self.data = []


def download():
    """Download a day's TLE files.

    Notes
    -----
    On the website TLE files are separated by category. When downloaded, this
    method will combine them into one TLE file that is named with the date
    on which the TLE was downloaded. The TLE txt file is then compressed
    with gzip. The final file will be named `date`.txt.gz

    The TLE files are located at http://www.celestrak.com/NORAD/elements/.
    """
    # The parent directory. We run through this to find the TLEs to download.
    parent = "http://www.celestrak.com/NORAD/elements/"
    d = download_url(parent)
    parser = TLEHTMLParser()
    parser.feed(d.text)
    parser.close()
    locs = parser.data

    now = datetime.datetime.now()

    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%Y%m%d")

    # Make the file location if we need it
    store_loc = os.path.join("TLE", *[year, month])

    # Blackbox loc
    #store_loc = os.path.join("/media", *["data2", "tle", year, month])

    store_file = os.path.join(store_loc, day + ".txt")
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)

    # Loop through all the files
    for l in locs:
        #print("\"" + l + "\", ", end = "")
        # Link is the parent plus the name of the file at the end.
        link = parent + l
        data = download_url(link)

        # Writes the file into the folder.
        # Plus makes it so that if it doesn't exist it creates the file.
        with open(store_file, "a+") as f:
            f.write(data.text)

    # Compresses the text file
    with open(store_file, "rb") as f1:
        with gzip.open(store_file + ".gz", "wb") as f2:
            shutil.copyfileobj(f1, f2)

    # No longer need this.
    os.remove(store_file)


def run_and_download():
    """Runs the script and downloads the images.

    See Also
    --------
    download : Download the day's TLE file.

    Notes
    -----
    This method exectures the code flow of the script. Once a TLE file is downloaded
    the method will sleep until 12:30 the next day and repeat the process.
    """
    while True:
        try:
            download()

            print("Downloaded: ", datetime.datetime.now().strftime("%Y-%m-%d"))
            # Sleeps until 6 seconds after two minutes from now.
            sleep_until = (datetime.datetime.now() + datetime.timedelta(hours=24)).replace(hour=12, minute=30)
            sleep_for = (sleep_until - datetime.datetime.now()).total_seconds()
            time.sleep(sleep_for)

        except Exception as e:
            logging.error(e)
            raise(e)


if __name__ == "__main__":
    run_and_download()

