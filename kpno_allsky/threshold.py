"""A module providing analysis methods for determining cloudiness thresholds.

The predominant function of this module is to analyze saved cloudiness data
to try and determine the cloudiness value that corresponds to whether or not
the telescope dome is closed or open. The main method analyzes the cloudiness
of each image relative to the mean cloudiness for the phase of the moon on that
night. Each night images were taken when the dome was closed. Images whose
cloudiness is above a certain value are considered to have been taken when the
dome was closed. If the proportion of images above this value is
approximately equal to the known percentage of the night when the dome was
closed then this value is designated the threshold for each night.
Two helper methods are provided: one to find the number of days into a year a
given date is, and one to convert a date to a format accepted by pyephem.
"""

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ephem


def daynum(date):
    """Find the number of the days into a year a given date is.

    Parameters
    ----------
    date : str
        Date in yyyymmdd format.

    Returns
    -------
    num : int
        The number of days into the year the given date is.
    """
    # Strips out the information from the pass in string
    d1 = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8]))

    # The date for the start of the year.
    d2 = datetime.date(year=d1.year, month=1, day=1)

    # Gotta add one because of how subtraction works.
    days = (d1-d2).days + 1
    return days


def format_date(date, name):
    """Convert an image name into a format that pyephem accepts.

    Parameters
    ----------
    date : str
        Date on which the image was taken, in yyyymmdd format.
    name : str
        The image"s file name.

    Returns
    -------
    date : str
        A date and time in yyyy/mm/dd hh:mm:ss format.
    """
    formatdate = date[:4] + "/" + date[4:6] + "/" + date[6:8]
    time = name[4:6] + ":" + name[6:8] + ":" + name[8:10]
    return formatdate + " " + time


def find_threshold():
    """Find the mean cloudiness threshold for 2016 and 2017 combined.

    Returns
    -------
    total : float
        The threshold value if each day is considered individually.
    weektotal : float
        The threshold value if the days are grouped by week.

    Notes
    -----
    The method analyzes the cloudiness of each image relative to the mean
    cloudiness for the phase of the moon on that night.
    Each night images were taken when the dome was closed.
    Images whose cloudiness is above a certain value are considered to
    have been taken when the dome was closed. This method finds that value by
    using the percentage of each night when the dome is closed, which is
    already known. The proportion of images above this value will be
    approximately equal to the known percentage. This value is designated the
    threshold for each night. The method finds this threshold for every night
    and then returns the median of that dataset.

    This method additionally runs the same analysis where an entire week is
    considered at a time, rather than single nights. The returned value is
    the median of the 52 week thresholds.
    """
    # Sets up a pyephem object for the camera.
    camera = ephem.Observer()
    camera.lat = "31.959417"
    camera.lon = "-111.598583"
    camera.elevation = 2120
    camera.horizon = "-17"

    # Reads in the csv file using pandas.
    data_loc = directory = os.path.join(os.path.dirname(__file__), *["data", "daily-2007-2017.csv"])
    data = pd.read_csv(data_loc)

    # This array is for calculating the total average
    total = []
    for year in ["2016", "2017"]:
        # Gets the downloaded months
        directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed"])

        # If the data directory doesn"t exist we should exit here.
        # I was remarkably prescient writing this even though I had all the data
        # downloaded already.
        if not os.path.exists(directory):
            print("No data found.")

        months = sorted(os.listdir(directory))

        # Macs are dumb
        if ".DS_Store" in months:
            months.remove(".DS_Store")

        # We do this as a dict because 2017 is straight up missing some days of
        # images because I guess the camera was down?
        # Otherwise I"d just make a len 365 list.
        weekdict = [[] for i in range(0, 53)]
        daydict = {}
        for month in months:
            # If the month is not in this year, we skip it analyzing.
            if int(month) < int(year + "01") or int(month) > int(year + "12"):
                continue

            # Gets the days that were analyzed for that month
            directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed", month])
            days = sorted(os.listdir(directory))

            # Macs are still dumb.
            if ".DS_Store" in days:
                days.remove(".DS_Store")

            # Reads the data for each day.
            for day in days:
                # Skip the leap day for right now.
                if day == "20160229.txt":
                    continue

                # Get the number for that day to add it to the dict.
                i = daynum(day)

                weeknum = i // 7

                # Because we skip the leap day we need to bump the day num of
                # all days after that date down by one.
                # 60 because 31 + 28 = 59
                if year == "2016" and i >= 60:
                    i = i-1

                # Start with an empty list for that day.
                daydict[i] = []

                # This is the code that reads in the values and appends them.
                data_loc = os.path.join(directory, day)
                datafile = open(data_loc, "r")

                # Images is a list of images that were analyzed that night.
                images = []
                for line in datafile:
                    line = line.rstrip()
                    line = line.split(",")

                    # Appends the image name to images and the cloudiness
                    # relative to mean to the daydict.
                    images.append(line[0])
                    daydict[i].append(float(line[1]))

                    weekdict[weeknum].append(float(line[1]))

        # An ndarray of open fractions where index + 1 = day number
        opens = data.get("Y" + year).values
        thresh = []

        x = []
        x1 = []
        true = []
        # Runs over the dictionary, key is the day number. Val is the list of
        # cloudinesses

        openweeks = [[] for i in range(0, 53)]

        for key, val in daydict.items():
            # The fraction is the fraction of the night the dome was closed.
            # When we multiply to find the index we want the inverse frac though.
            frac = 1 - opens[key - 1]

            week = (key) // 7
            openweeks[week].append(opens[key - 1])

            # Finds the index at which the fraction of images above that index is
            # equal to the amount of the night that the dome was closed.
            working = sorted(val)

            # If we don"t have any images that night then just bail.
            if not working:
                continue

            # Multiply the frac by the length, to find the index above which
            # the correct fraction of the images is "dome closed." Rounds and
            # Subtracts one to convert it to the integer index.
            index = int(round(frac * len(working))) - 1

            # If the index is the final index then the "cloudiness relative to the
            # mean threshold" is slightly below that value so average down.
            # Otherwise take the average of that index and the one above since the
            # threshold actually falls inbetween.
            if index == len(working) - 1 and not frac == 1:
                num = np.mean([float(working[index]), float(working[index - 1])])
            # If the dome is closed the entire night, index will be given as -1
            # And we find the threshold as the average of the start and end
            # cloudiness. Instead we want the threshold to be the first
            # cloudiness as that way the dome is "closed" all night.
            elif frac == 0:
                num = float(working[0]) - 0.1
            elif frac == 1:
                num = float(working[-1])
            else:
                num = np.mean([float(working[index]), float(working[index + 1])])

            thresh.append(num)
            total.append(num)
            x.append(key)

            working = np.asarray(working)
            above = working[working > num]

            if not working.size == 0:
                frac = opens[key - 1]
                true.append(len(above)/len(working))
                x1.append(frac)

        weekthresh = []
        weektotal = []
        weektrue = []
        x2 = []
        x3 = []
        for i in range(0, len(weekdict)):
            frac = 1 - np.mean(openweeks[i])

            working = sorted(weekdict[i])

            # If we don"t have any images that night then just bail.
            if not working:
                continue

            # Multiply the frac by the length, to find the index above which
            # the correct fraction of the images is "dome closed." Rounds and
            # Subtracts one to convert it to the integer index.
            index = int(round(frac * len(working))) - 1

            # If the index is the final index then the "cloudiness relative to the
            # mean threshold" is slightly below that value so average down.
            # Otherwise take the average of that index and the one above since the
            # threshold actually falls inbetween.
            if index == len(working) - 1 and not frac == 1:
                num = np.mean([float(working[index]), float(working[index - 1])])
            # If the dome is closed the entire night, index will be given as -1
            # And we find the threshold as the average of the start and end
            # cloudiness. Instead we want the threshold to be the first
            # cloudiness as that way the dome is "closed" all night.
            elif frac == 0:
                num = float(working[0]) - 0.1
            elif frac == 1:
                num = float(working[-1])
            else:
                num = np.mean([float(working[index]), float(working[index + 1])])

            weekthresh.append(num)
            weektotal.append(num)
            x2.append(i)

            working = np.asarray(working)
            above = working[working > num]

            if not working.size == 0:
                frac = np.mean(openweeks[i])
                weektrue.append(len(above)/len(working))
                x3.append(frac)

        print(year + ": ")
        print("Min: " + str(np.amin(thresh)))
        print("25%: " + str(np.percentile(thresh, 25)))
        print("50%: " + str(np.median(thresh)))
        print("75%: " + str(np.percentile(thresh, 75)))
        print("Max: " + str(np.amax(thresh)))
        print()

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)

        above = np.ma.masked_where(thresh < np.median(thresh), thresh)
        below = np.ma.masked_where(thresh > np.median(thresh), thresh)
        ax.scatter(x, below, s=1)
        ax.scatter(x, above, s=1, c="r")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cloudiness Relative to Mean")
        plt.savefig("Images/Dome/Threshold-Day-" + year + ".png", dpi=256)
        plt.close()

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)
        ax.scatter(x1, true, s=1)
        ax.set_xlabel("True Fraction")
        ax.set_ylabel("Found Fraction")
        plt.savefig("Images/Dome/Verify-Day-" + year + ".png", dpi=256)
        plt.close()

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)

        print(year + " Week: " + str(np.median(weekthresh)))

        above = np.ma.masked_where(weekthresh < np.median(weekthresh), weekthresh)
        below = np.ma.masked_where(weekthresh > np.median(weekthresh), weekthresh)
        ax.scatter(x2, below, s=1)
        ax.scatter(x2, above, s=1, c="r")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cloudiness Relative to Mean")
        plt.savefig("Images/Dome/Threshold-Week-" + year + ".png", dpi=256)
        plt.close()

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)
        ax.scatter(x3, weektrue, s=1)
        ax.set_xlabel("True Fraction")
        ax.set_ylabel("Found Fraction")
        plt.savefig("Images/Dome/Verify-Week-" + year + ".png", dpi=256)
        plt.close()

    #years[year] = daydict

    return (np.median(total), np.median(weektotal))


def test_threshold():
    #This method uses find_threshold() to find the cloudiness thresholds. It
    #then uses the median for each individual night of images and checks the
    #proportion of images that are above the threshold against the fraction
    #of that night for which the dome is closed. It outputs four plots, which
    #are saved to Images/Dome/. Each plot plots the dome closed fraction on
    #the horizontal axis, and the fraction of images above the median threshold
    #for that year on the y axis. The four plots represent two plots with day
    #wise thresholding, and two with weekwise. One plot contains all 2016 data
    #and the other the 2017 data.

    for year in ["2016", "2017"]:

        # Reads in the csv file using pandas.
        data_loc = directory = os.path.join(os.path.dirname(__file__), *["data", "daily-2007-2017.csv"])
        data = pd.read_csv(data_loc)

        opens = data.get("Y" + year).values

        # Gets the downloaded months
        directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed"])

        # If the data directory doesn"t exist we should exit here.
        # I was remarkably prescient writing this even though I had all the data
        # downloaded already.
        if not os.path.exists(directory):
            print("No data found.")

        months = sorted(os.listdir(directory))

        # Macs are dumb
        if ".DS_Store" in months:
            months.remove(".DS_Store")

        test1, test2 = find_threshold()

        print(test1)
        print(test2)

        # We do this as a dict because 2017 is straight up missing some days of
        # images because I guess the camera was down?
        # Otherwise I"d just make a len 365 list.
        daydict = {}
        weekdict = [[] for i in range(0, 53)]
        for month in months:
            # If the month is not in this year, we skip it analyzing.
            if int(month) < int(year + "01") or int(month) > int(year + "12"):
                continue

            # Gets the days that were analyzed for that month
            directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed", month])
            days = sorted(os.listdir(directory))

            # Macs are still dumb.
            if ".DS_Store" in days:
                days.remove(".DS_Store")

            # Reads the data for each day.
            for day in days:
                # Skip the leap day for right now.
                if day == "20160229.txt":
                    continue

                # Get the number for that day to add it to the dict.
                i = daynum(day)

                weeknum = i // 7

                # Because we skip the leap day we need to bump the day num of
                # all days after that date down by one.
                # 60 because 31 + 28 = 59
                if year == "2016" and i >= 60:
                    i = i-1

                # Start with an empty list for that day.
                daydict[i] = []

                # This is the code that reads in the values and appends them.
                data_loc = os.path.join(directory, day)
                datafile = open(data_loc, "r")

                # Images is a list of images that were analyzed that night.
                images = []
                for line in datafile:
                    line = line.rstrip()
                    line = line.split(",")

                    # Appends the image name to images and the cloudiness
                    # relative to mean to the daydict.
                    images.append(line[0])
                    daydict[i].append(float(line[1]))

                    weekdict[weeknum].append(float(line[1]))

            x = []
            true = []
            weekfrac = [[] for i in range(0, 53)]
            # Runs over the dictionary, key is the day number. Val is the list
            # of cloudinesses
            for key, val in daydict.items():
                # The fraction is the fraction of the night the dome was closed.
                frac = opens[key - 1]

                # Appends the fract to the week average array.
                weekfrac[key // 7].append(frac)

                working = np.asarray(val)

                above = working[working > test1]
                if not working.size == 0:
                    true.append(len(above)/len(working))
                    x.append(frac)

            x1 = []
            weektrue = []
            # Runs over the week number
            for i, val in enumerate(weekdict):
                # The fraction is the average of the whole week.
                frac = np.mean(weekfrac[i])

                working = np.asarray(weekdict[i])

                above = working[working > test2]
                if not working.size == 0:
                    weektrue.append(len(above)/len(working))
                    x1.append(frac)

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)
        ax.scatter(x, true, s=1)
        ax.set_xlabel("True Fraction")
        ax.set_ylabel("Found Fraction")
        plt.savefig("Images/Dome/Differences-Day-" + year + ".png", dpi=256)
        plt.close()

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)
        ax.scatter(x1, weektrue, s=1)
        ax.set_xlabel("True Fraction")
        ax.set_ylabel("Found Fraction")
        plt.savefig("Images/Dome/Differences-Week-" + year + ".png", dpi=256)
        plt.close()


if __name__ == "__main__":
    test_threshold()
