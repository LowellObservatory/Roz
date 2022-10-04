"""A module containing various methods for analyzing the cloudiness of images.

This module is designed to be used in conjunction with downloaded images to
perform statistical analyses and plotting. Methods are provided for
analyzing months of data and saving the cloudiness values to a file. It is
then possible to take this data and plot it in various forms, across a variety
of different plots. The cloudiness can be plotted as a histogram, and two
different models can be fit to the histograms. These two models are a double
Gaussian and a Gaussian-Poisson hybrid.
"""
import os
import time
import ast
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ephem

import io_util
import moon
import histogram
import coordinates
import image


def get_latest_analyzed():
    """Find the latest analyzed date.

    Returns
    -------
    str
        The latest date analyzed in yyyymmdd format.

    Notes
    -----
    The latest date analyzed is found by searching through Data/ for the
    latest month that has been analyzed, and then searching through that month
    for the latest day that has been analyzed.
    """
    # Gets the downloaded months
    directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed"])
    months = sorted(os.listdir(directory))

    # If no dates have been analyzed yet return 0
    if not months or months[-1] == ".DS_Store":
        return 0

    # Gets the singular days that have been analyzed
    directory = os.path.join(directory, months[-1])
    days = sorted(os.listdir(directory))

    # If no days for this month have been analyzed yet return the first day.
    if not days:
        start = months[-1] + "01"
        return start

    # Return the final day analyzed. We will reanalyze this completely,
    # in case the analysis crashed mid day.
    # Second slice slices off the .txt
    start = days[-1][:-4]
    return start


def analyze():
    """Determine cloudiness of images for every day from January 1, 2017 onward.

    See Also
    --------
    histogram.cloudiness : Calculate the cloudiness of a histogram.

    Notes
    -----
    For each night, all images taken that day are downloaded. For every night
    analyzed, the cloudiness values for every image taken during that night
    are written to "Data/yyyymm/yyyymmdd.txt" where yyyymmdd corresponds to the
    date analyzed.
    """
    t1 = time.perf_counter()
    link = "http://kpasca-archives.tuc.noao.edu/"

    rlink = io_util.download_url(link)

    if rlink is None:
        print("Getting dates failed.")
        return

    html = rlink.text
    # This parser was designed to work with image names but it works the same
    # for dates so why make a new one?
    parser = io_util.DateHTMLParser()
    parser.feed(html)
    parser.close()
    datelinks = parser.data
    parser.clear_data()

    # 20080306 is the first date the camera is located correctly
    # 20080430 is the first date the images are printed correctly

    startdate = 20100101

    # Reads in the model coefficients.
    cloud_loc = os.path.join(os.path.dirname(__file__), *["data", "clouds.txt"])
    with open(cloud_loc, "r") as f:
        for line in f:
            line = line.rstrip()
            line = line.split(",")
            b = float(line[0])
            c = float(line[1])

    fails = []
    for date in datelinks:

        # Strips the /index from the date so we have just the date
        d = date[:8]

        # The month of the date for organization purposes.
        month = d[:6]

        # This makes it so we start with the date we left off with.
        if int(d) < startdate:
            continue

        if not 20140726 <= int(d) <= 20141231:
            continue

        # Print a new line to differentiate dates.
        print()
        print(d)

        rdate = io_util.download_url(link + date)

        # If we fail to get the data for the date, log it and move to the next.
        if rdate is None:
            fails.append(date)
            print("Failed to retrieve data for " + date)
            continue

        # Extracting the image names.
        htmldate = rdate.text
        parser.feed(htmldate)
        parser.close()
        imagenames = parser.data
        parser.clear_data()

        monthloc = os.path.join(os.path.dirname(__file__), *["data", "analyzed", month])
        if not os.path.exists(monthloc):
            os.makedirs(monthloc)

        datafile = os.path.join(monthloc, d + ".txt")

        with open(datafile, "w") as f:

            for name in imagenames:
                # This ignores all the blue filter images, since we don"t want
                # to process those.
                if name[:1] == "b":
                    continue
                # We want to ignore the all image animations
                if name == "allblue.gif" or name == "allred.gif":
                    continue

                # This creates an AllSkyImage we use for subsequent methods.
                # This is the path the image is saved to.
                # We need to check if it exists first, just in case.
                path = os.path.join(os.path.dirname(__file__), *["Images", "Original", "KPNO", d, name])

                # Download the image, if it hasn"t been downloaded
                if not os.path.isfile(path):
                    io_util.download_image(d, name)
                img = image.load_image(name, d, "KPNO")

                # Finds the moon and the sun in the image.
                # We only process images where the moon is visble (alt > 0)
                # And the sun is low enough to not wash out the image
                # (sun alt < -17)
                moonalt = moon.find_moon(img)[2]
                sunalt = moon.find_sun(img)[0]

                # Checks that the analysis conditions are met.
                if moonalt > 0 and sunalt < -17:

                    # Then we make a histogram and a "cloudiness fraction"
                    # Generates the moon mask.
                    mask = moon.moon_mask(img)
                    hist = histogram.generate_histogram(img, mask)

                    frac = histogram.cloudiness(hist)

                    # Correction for moon phase.
                    phase = moon.moon_phase(img)
                    val = b * phase * phase + c * phase
                    frac = frac / val

                    # Then we save the cloudiness fraction to the file for that
                    # date.
                    dataline = name + "," + str(frac) + "\n"
                    f.write(dataline)

                    print("Analyzed: " + img.formatdate)

                # Deletes an image if we didn"t analyze it.
                else:
                    os.remove(path)
    t2 = time.perf_counter()

    print(t2 - t1)

    print("The following dates failed to download: " + str(fails))


def month_plot():
    """Generate a plot of nightly cloudiness for each month.

    Notes
    -----
    The cloudiness values for every image saved using :func:`~analyze` are loaded. These
    values are plotted, with all images for a given month on a single plot.
    Plots are saved in "Images/Plots/" with the filename "scatter-`month`.png".
    """
    # Gets the downloaded months
    directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed"])

    # If the data directory doesn"t exist we should exit here.
    if not os.path.exists(directory):
        print("No data found.")
        return

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if ".DS_Store" in months:
        months.remove(".DS_Store")

    for month in months:
        # Gets the days that were analyzed for that month
        directory  = os.path.join(os.path.dirname(__file__), *["data", "analyzed", month])
        days = sorted(os.listdir(directory))

        data = []
        illum = []
        x = []
        # Reads the data for each day.
        for day in days:
            d1 = directory + day
            f1 = open(d1, "r")

            # Strips off the .txt so we can make a Time object.
            day = day[:-4]
            for line in f1:
                line = line.rstrip()
                line = line.split(",")

                # Gets the plot date for the timestring object.
                d = coordinates.timestring_to_obj(day, line[0]).plot_date

                data.append(float(line[1]))
                x.append(d)

                vis = moon.moon_visible(day, line[0])
                illum.append(vis)

            f1.close()

        # Sets up the plot
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 5)
        ax.plot_date(x, data, xdate=True, markersize=1)

        ax.plot_date(x, illum, xdate=True, markersize=1, color="red")

        # These are the x divisions which indicate the start of each day.
        xt = []
        start = int(days[0][:-4])
        end = int(days[-1][:-4]) + 1
        for i in range(start, end):
            # Finds the plot_date for the start of the day.
            x1 = coordinates.timestring_to_obj(str(i),
                                               "r_ut000000s00000").plot_date
            xt.append(x1)

        # Need the right end to be the start of next month.
        # Adding 100 increments the month number.
        # First if statement checks if this is december in which case we need
        # to roll over the year and reset the month.
        if str(start)[4:6] == "12":
            end = str(start + 8900)
        else:
            end = str(start + 100)
        d2 = coordinates.timestring_to_obj(end, "r_ut000000s00000").plot_date

        # Sets the limits and division ticks.
        ax.set_ylim(0, 1.0)
        ax.set_xlim(xt[0], d2)
        ax.set_xticks(xt)
        ax.xaxis.grid(True)
        ax.xaxis.set_ticklabels([])

        # Sets the axis labels and saves.
        ax.set_ylabel("Cloudiness Fraction")

        # Puts the date in a nice form for the plot axis.
        date = month[-2:] + "/01/" + month[:-2]
        ax.set_xlabel("Time After " + date)

        plotloc = os.path.join(os.path.dirname(__file__), *["Images", "Plots"])
        if not os.path.exists(plotloc):
            os.makedirs(plotloc)

        plt.savefig(plotloc + "scatter-" + month + ".png",
                    dpi=256, bbox_inches="tight")
        plt.close()


# Sets up a pyephem object for the camera.
camera = ephem.Observer()
camera.lat = "31.959417"
camera.lon = "-111.598583"
camera.elevation = 2120
camera.horizon = "-17"


def plot(years=["2015","2016","2017"], fit_histograms=False):
    """Generate various cloudiness plots and save them to Images/Plots.

    Parameters
    ----------
    years : list, optional
        List of years to include in the plots. Defaults to
        ["2015","2016","2017"].
    fit_histograms : bool, optional
        To fit the weekly histograms and plot the resulting fit variables
        or not. Defaults to False.

    Notes
    -----
    The cloudiness values for every image saved using :func:`~analyze` are loaded.
    These values are further analyzed to create new plots. The generated
    plots include plots of Cloudiness vs Moon Phase, Cloudiness vs Hours since
    sunset, Cloudiness vs Normalized time after sunset, Cloudiness vs Hours
    before sunrise, and Cloudiness vs Week Number. These plots are saved to
    Images/Plots. If `fit_histograms` is true, the weekly histograms will be
    fit by a double Gaussian or Gaussian-Poisson hybrid function, after which
    plots of week number versus the various fit parameters will be made and
    saved as well.
    """
    # Gets the downloaded months
    directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed"])

    # If the data directory doesn"t exist we should exit here.
    if not os.path.exists(directory):
        print("No data found.")
        return

    # Temporary arrays for moon phase plots
    phasenum = 50
    phasediv = 1 / phasenum  # Phase ranges from 0-1 so divisions is 1/num divs
    tphase = [[] for i in range(0, phasenum)]

    # Temporary arrays for sunset plots
    sunsetnum = 50
    sunsetdiv = 0.5 / sunsetnum  # Up to 12 hours after sunset = 0.5 day / divs
    tsunset = [[] for i in range(0, sunsetnum)]

    # Temporary arrays for sunrise plots
    sunrisenum = 50
    sunrisediv = 0.5 / sunrisenum  # 12 hours before sunrise = 0.5 day / divs
    tsunrise = [[] for i in range(0, sunrisenum)]

    # Temp other stuff
    normalnum = 50
    normaldiv = 1. / normalnum
    tnormal = [[] for i in range(0, normalnum)]

    # Temporary arrays for week plots
    tweek = [[] for i in range(0, 53)]
    tmoon = [[] for i in range(0, 53)]

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if ".DS_Store" in months:
        months.remove(".DS_Store")

    for month in months:
        # Gets the days that were analyzed for that month
        directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed", month])
        days = sorted(os.listdir(directory))

        # Strips out the year from the month
        year = month[:4]

        if not year in years:
            continue

        # Day 1 of the year, for week calculation.
        yearstart = year + "0101"
        day1 = coordinates.timestring_to_obj(yearstart, "r_ut000000s00000")

        # Reads the data for each day.
        for day in days:
            loc = os.path.join(directory, day)
            f1 = open(loc, "r")

            # Strips off the .txt so we can make a Time object.
            day = day[:-4]
            for line in f1:
                # Splits out the value and file.
                line = line.rstrip()
                line = line.split(",")
                val = float(line[1])
                name = line[0]

                # A phantom image for calculation that doesn"t actually have
                # any data in it.
                img = image.AllSkyImage(name, day, "KPNO", None)
                # Moon phase calculation.
                phase = moon.moon_phase(img)

                # Ignores cloudiness with moon phase less than 0.2
                if phase < 0.2:
                    continue

                b = int(phase // phasediv)
                tphase[b].append(val)
                formatdate = img.formatdate

                # Sets the date of calculation.
                camera.date = formatdate
                date = ephem.Date(formatdate)

                # Calculates the previous setting and next rising of the sun.
                sun = ephem.Sun()
                setting = camera.previous_setting(sun, use_center=True)
                rising = camera.next_rising(sun, use_center=True)

                # Finds the difference and bins it into the correct bin.
                diff = date - setting
                length = rising - setting
                b = int(diff // sunsetdiv)
                b2 = int((diff / length) // normaldiv)
                tsunset[b].append(val)
                tnormal[b2].append(val)

                # Finds the difference and bins it into the correct bin.
                diff = rising - date
                b = int(diff // sunrisediv)
                tsunrise[b].append(val)

                # Week of the year calculation.
                # Gets the date object for the image
                date = img.time

                # Finds the difference since the beginning of the year to find
                # The week number.
                diff = date - day1
                week = int(diff.value // 7)
                tweek[week].append(val)
                tmoon[week].append(phase)

        # Prints the progress if this is the final month of the year and we
        # just finished it.
        if month[4:] == "12":
            print("Collated " + year + " data")


    percents = ["25", "50", "75"]
    colors = [(1, 0, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]

    # Nan array for if there"s no data for that bin.
    nanarray = np.asarray([[float("nan"), float("nan"), float("nan")]])

    # This method sets up the plots. using the given x array and the data set
    # it finds the percentile values for each bin division.
    def setup_plot(x, dataset):
        # We only need this so the axis shape works out. We delete it later.
        data = np.asarray([[0, 0, 0]])

        for i, val in enumerate(dataset):
            temp = np.asarray(val)

            # Percentile returns an array of each of the three values.
            # We"re creating a data array where each column is all the
            # percentile values for each bin value.
            if temp.size == 0:
                data = np.append(data, nanarray, axis=0)
            else:
                d = np.reshape(np.percentile(temp, [25, 50, 75]), (1, 3))
                data = np.append(data, d, axis=0)

        # Deletes the first 0,0,0 array.
        data = np.delete(data, 0, 0)

        plt.ylim(0, 4.0)

        # This sets up the individual plots. You can just pass the data
        # multiararay but I need each line to be individually labeled.
        for i, val in enumerate(percents):
            plt.plot(x, data[0:data.shape[0], i], label=val + "%",
                     color=colors[i])

        # This fills between the lowest and top percentiles.
        plt.fill_between(x, data[0:data.shape[0], 0],
                         data[0:data.shape[0], -1], color=(1, 0.5, 0, 0.5))
        plt.legend()

    # Plotting and averaging code
    # Moon phase
    plt.ylabel("Cloudiness Relative to Mean")
    plt.xlabel("Moon Phase")

    x = np.asarray((range(0, phasenum)))
    x = x * phasediv + (phasediv / 2.)
    setup_plot(x, tphase)
    plt.savefig("Images/Plots/phase.png", dpi=256, bbox_inches="tight")
    plt.close()

    # Sunset
    plt.ylabel("Cloudiness Relative to Mean")
    plt.xlabel("Hours since sunset")

    x = np.asarray((range(0, sunsetnum)))
    x = x * sunsetdiv * 24 + (sunsetdiv * 12)
    setup_plot(x, tsunset)
    plt.savefig("Images/Plots/sunset.png", dpi=256, bbox_inches="tight")
    plt.close()

    # Normalized
    plt.ylabel("Cloudiness Relative to Mean")
    plt.xlabel("Normalized time after sunset")

    x = np.asarray((range(0, normalnum)))
    x = x * normaldiv + (normaldiv / 2.)
    setup_plot(x, tnormal)
    plt.savefig("Images/Plots/normalized-0.png", dpi=256, bbox_inches="tight")
    plt.close()

    # Sunrise
    x = np.asarray((range(0, sunrisenum)))
    x = x * sunrisediv * 24 + (sunrisediv * 12)

    # Sets up the plot before we plot the things
    plt.ylabel("Cloudiness Relative to Mean")
    plt.xlabel("Hours before sunrise")

    setup_plot(x, tsunrise)
    plt.savefig("Images/Plots/sunrise.png", dpi=256, bbox_inches="tight")
    plt.close()

    # Week
    x = np.asarray((range(1, 54)))
    #plt.ylabel("Cloudiness Relative to Mean")
    plt.xlabel("Week Number")
    plt.ylabel("Cloudiness Relative to Mean")

    # Moon phase averages.
    moon_avgs = []
    num_imgs = []
    for i, val in enumerate(tweek):
        moon_avgs.append(np.mean(tmoon[i]))
        num_imgs.append(len(val))

    setup_plot(x, tweek)

    #Reads in the csv file using pandas.
    domedata = pd.read_csv("daily-2007-2017.csv")

    closedict = dict.fromkeys(years)
    for y in years:
        name = "Y" + y
        closedict[y] = domedata.get(name).values

    tclose = [[] for i in range(0, 53)]
    closeav = []

    # i represents the day number of the year in 2016. If it is larger than 59
    # (31 + 28) we need to increase the day by 1 to account for the leap day.
    # This is because the domedata csv does not include February 29.
    # For example, March 1st is considered day 60, when in 2016 it
    # is actually day 61.
    for j, val in enumerate(closedict["2016"]):
        if j >= 60:
            i = j + 1
        else:
            i = j

        week1 = j // 7
        week2 = i // 7

        for key, val in closedict.items():
            # We need to append to a different week if it's a leap year, due
            # to the leap day changing the day number of the year after
            # February.
            week = week2 if int(key) % 4 == 0 else week1
            tclose[week].append(val[j])

    for i, val in enumerate(tclose):
        closeav.append(np.mean(val))

    # Alternative additional data.
    #plt.plot(x, moons, label="Moon Phase", color=(0, 1, 0, 1))

    #num_imgs = np.asarray(num_imgs)
    #num_imgs = num_imgs * 1 / (np.amax(num_imgs))
    #plt.plot(x, num_imgs, label="Normalized number of images",
             #color=(0, 1, 0, 1))

    plt.plot(x, closeav, label="Average Closed Fraction", color=(0,1,0,1))
    plt.legend()

    plt.savefig("Images/Plots/week.png", dpi=256, bbox_inches="tight")
    plt.close()

    if fit_histograms:
        data = np.asarray([[0, 0, 0, 0, 0]])

        split = 10
        w = 0.61 / split
        num = np.amax(tweek[2]) / w
        divs = np.asarray(range(0, int(num) + 1))
        divs = divs * w

        for i, val in enumerate(tweek):
            temp = np.asarray(val)

            # Percentile returns an array of each of the three values.
            # We"re creating a data array where each column is all the
            # percentile values for each bin value.
            if temp.size == 0:
                data = np.append(data, nanarray, axis=0)
            else:
                # Passes the data to the fitting method.
                temp = np.delete(temp, np.where(temp < 0.061))
                d, _ = find_fit(temp, divs)

                # Inserts a the sqaure root of the lambda for the mu-2 if the fit
                # Is a gaussian-poisson combo.
                if len(d) < 5:
                    d = np.asarray(d)
                    d = np.insert(d, 3, np.sqrt(d[2]))

                d = d.reshape(1,5)
                data = np.append(data, d, axis=0)

        # Deletes the first 0,0,0 array.
        data = np.delete(data, 0, 0)


        # Code to plot the fits, abstracted because it was getting a little
        # cumbersome to copy paste.
        def fit_plot(x, data, name):
            loc = "Images/Plots/week-" + name.lower() + ".png"
            plt.plot(x, data, label=name)
            plt.scatter(x, data, label=name, s=2, c="r")
            plt.xlabel("Week Number")
            plt.ylabel("Cloudiness Relative to Mean")
            plt.legend()
            plt.savefig(loc, dpi=256, bbox_inches="tight")
            plt.close()

        # This code plots the variables individually
        sigma1 = data[0:data.shape[0], 0]
        fit_plot(x, sigma1, "Sigma-1")

        mu1 = data[0:data.shape[0], 1]
        fit_plot(x, mu1, "Mu-1")

        cv1 = sigma1 / mu1
        fit_plot(x, cv1, "CV-1")

        sigma2 = data[0:data.shape[0], 2]
        fit_plot(x, sigma2, "Sigma-2")

        mu2 = data[0:data.shape[0], 3]
        print(np.argmin(mu2))
        mu2 = np.where(mu2 < -100, mu1, mu2)
        fit_plot(x, mu2, "Mu-2")

        cv2 = sigma2 / mu2
        fit_plot(x, cv2, "CV-2")

        frac = data[0:data.shape[0], 4]
        frac = (np.tanh(frac) + 1) / 2
        fit_plot(x, frac, "Frac")

        plt.plot(x, mu1 + mu2, label="Plus")
        plt.scatter(x, mu1 + mu2, label="Plus", s=2, c="r")
        plt.plot(x, np.abs(mu1 - mu2), label="Minus")
        plt.scatter(x, np.abs(mu1 - mu2), label="Minus", s=2, c="g")
        plt.xlabel("Week Number")
        plt.legend()
        plt.savefig("Images/Plots/week-mu-test.png", dpi=256, bbox_inches="tight")
        plt.close()

        plt.plot(x, sigma1 + sigma2, label="Plus")
        plt.scatter(x, sigma1 + sigma2, label="Plus", s=2, c="r")
        plt.plot(x, np.abs(sigma1 - sigma2), label="Minus")
        plt.scatter(x, np.abs(sigma1 - sigma2), label="Minus", s=2, c="g")
        plt.xlabel("Week Number")
        plt.legend()
        plt.savefig("Images/Plots/week-sigma-test.png", dpi=256, bbox_inches="tight")
        plt.close()

        plt.plot(x, cv1 + cv2, label="Plus")
        plt.scatter(x, cv1 + cv2, label="Plus", s=2, c="r")
        plt.plot(x, np.abs(cv1 - cv2), label="Minus")
        plt.scatter(x, np.abs(cv1 - cv2), label="Minus", s=2, c="g")
        plt.xlabel("Week Number")
        plt.legend()
        plt.savefig("Images/Plots/week-cv-test.png", dpi=256, bbox_inches="tight")
        plt.close()


def model():
    """Model the dependence of cloudiness on the moon phase.

    Notes
    -----
    Cloudiness is plotted on the y-axis against the moon phase on the x-axis,
    and then np.linalg is used to find a quadratic model that fits this data.
    This model is used to remove the dependence of moon phase on the
    cloudiness data.
    """
    loc = os.path.join(os.path.dirname(__file__), *["data", "phase.txt"])

    data = []

    # Reads in the phase and cloudiness information.
    # Literal eval makes it so that it reads the list as a list rather than
    # a string.
    with open(loc, "r") as f:
        for line in f:
            line = line.rstrip()
            b = ast.literal_eval(line)

            data.append(b)
    x = data[0]
    x.pop(0)
    x = np.asarray(x)

    # Just need to convert to floats including the nan.
    for i in range(1, len(data)):
        data[i] = [float(j) for j in data[i]]

    plt.ylim(0, 1.0)
    plt.ylabel("Average Cloudiness Fraction")
    plt.xlabel("Moon Phase")

    coeffs1 = []
    coeffs2 = []
    for i in range(1, len(data)):
        data[i].pop(0)

        # This line is to avoid hard coding in case I add more years of data.
        year = 2017 - len(data) + 1 + i

        # We do the fitting manually using the least squares algorithm.
        # This forces the fit to go through 0,0 since we do not pass a constant
        # coefficient, only the x^2 and x terms.
        A = np.vstack([x*x, x]).T
        b, c = np.linalg.lstsq(A, data[i])[0]

        coeffs1.append(b)
        coeffs2.append(c)

        plt.plot(x, data[i], label="Mean-" + str(year))
        plt.plot(x, b*x*x + c*x, label="Fit-" + str(year))

    # An average of the year wise fits.
    m = np.mean(coeffs1)
    n = np.mean(coeffs2)
    plt.plot(x, m*x*x + n*x, label="Mean Fit")

    # Writes the coefficients to a file for later use.
    cloud_loc = os.path.join(os.path.dirname(__file__), *["data", "clouds.txt"])
    with open(cloud_loc, "w") as f:
        f.write(str(m) + "," + str(n))

    plt.legend()
    plt.savefig("Images/temp.png", dpi=256, bbox_inches="tight")
    plt.close()


def histo():
    """Generate weekly histograms of cloudiness values for 2016 and 2017.

    Notes
    -----
    The cloudiness values for every image saved using :func:`~analyze` are loaded and
    collated into histograms containing one week each. Each week produces
    three histograms: one with only 2016 images, one with only 2017
    images, and one with both on the same histogram. The 159 histograms are
    saved to "Images/Plots/Weeks".
    """
    directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed"])

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if ".DS_Store" in months:
        months.remove(".DS_Store")

    tweek = {}
    tweek["all"] = [[] for i in range(0, 53)]

    for month in months:

        # Gets the days that were analyzed for that month
        directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed", month])
        days = sorted(os.listdir(directory))

        # Strips out the year from the month
        year = month[:4]

        # Initializes a year if it doesn"t exist.
        if not year in tweek:
            tweek[year] = [[] for i in range(0, 53)]

        # Day 1 of the year, for week calculation.
        yearstart = year + "0101"
        day1 = coordinates.timestring_to_obj(yearstart, "r_ut000000s00000")

        # Reads the data for each day.
        for day in days:
            loc = directory + day
            f1 = open(loc, "r")

            # Strips off the .txt so we can make a Time object.
            day = day[:-4]
            for line in f1:
                # Splits out the value and file.
                line = line.rstrip()
                line = line.split(",")
                val = float(line[1])
                name = line[0]

                # Moon phase calculation.
                phase = moon.moon_visible(day, name)

                # Ignores cloudiness with moon phase less than 0.2
                if phase < 0.2:
                    continue

                date = coordinates.timestring_to_obj(day, name)

                # Finds the difference since the beginning of the year to find
                # The week number.
                diff = date - day1
                week = int(diff.value // 7)
                tweek["all"][week].append(val)
                tweek[year][week].append(val)


    split = 10
    w = 0.61 / split

    # Starts by finding the divs because we want the width to be the same.
    num = np.amax(tweek["all"][2]) / w
    divs = np.asarray(range(0, int(num) + 1))
    divs = divs * w

    # Blocks out the non major tick labels. Only keeps the major ones
    # Pre bar split (hence the modulo of the split)
    labels = list(divs)
    for j, label in enumerate(divs):
        if j % split != 0:
            labels[j] = ""
        else:
            labels[j] = round(label, 3)

    binstop = -16 * split
    histstop = binstop + 1

    saveloc = "Images/Plots/Weeks"
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)

    # Don"t want to recreate this every time.
    x = np.arange(0, divs[binstop], 0.05)

    chosen = {1:0, 2:0, 3:0, 4:0}
    sames = []
    # Loops over each week (the i value)
    for i, val in enumerate(tweek["all"]):
        for year, value in tweek.items():

            temp = np.asarray(tweek[year][i])
            temp = np.delete(temp, np.where(temp < 0.061))

            hist, bins = np.histogram(temp, bins=divs)

            # Sets the size wider than the previous to fit all the bins.
            # I shave off a lot of 0 value bins later as well in plotting.
            fig = plt.figure()
            fig.set_size_inches(11.4, 8.4)

            print("Week " + str(i+1))
            plt.title("Week " + str(i+1))
            plt.ylim(0, 900 / (split / 2))
            plt.ylabel("Number of Occurrences")
            plt.xlabel("Cloudiness Relative to Mean")

            # Number of images used to make this histogram.
            num = len(temp)

            # Plots everything, histogram, and then the fitted data on top.
            if year == "all":
                # Changes the numbers for each year
                temp2 = np.asarray(tweek["2016"][i])
                temp2 = np.delete(temp2, np.where(temp2 < 0.061))
                num2 = len(temp2)
                num = num - num2

                # For all, we plot everything, then 2016 on top for separation.
                plot = plt.bar(bins[:binstop], hist[:histstop], width=w,
                               align="edge", tick_label=labels[:binstop],
                               label="2017 (" + str(num) + ")")

                hist2, bins2 = np.histogram(temp2, bins=divs)
                plot = plt.bar(bins2[:binstop], hist2[:histstop], width=w,
                               align="edge", tick_label=labels[:binstop],
                               label="2016 (" + str(num2) + ")")

            else:
                # Plots just the year histogram.
                plot = plt.bar(bins[:binstop], hist[:histstop], width=w,
                               align="edge", tick_label=labels[:binstop],
                               label=year + " (" + str(num) + ")")

            coeffs, best = find_fit(temp, divs)

            print("Fit " + str(best + 1) + " Chosen")

            # Increments the dictionary counter
            chosen[best+1] = chosen[best+1] + 1

            print(str(year) + ": " + str(coeffs))

            # Finds the y func, then scales so it has the same area as the hist
            if len(coeffs) == 4:
                y = function_gp(coeffs, x)
            else:
                y = function_gg(coeffs, x)
            #print("Area: " + str(np.trapz(y,x)))
            scale = np.sum(hist) * w / np.trapz(y,x)
            y = y * scale

            # Plots the fit.
            plt.plot(x, y, color=(0, 1, 0, 1), label="Fit-" + str(best + 1))

            plt.legend()
            plt.savefig("Images/Plots/Weeks/hist-" + str(i + 1) + "-" + year + ".png",
                        dpi=256, bbox_inches="tight")
            plt.close()

        print("Saved: Week " + str(i+1))
        print()

    print(chosen)

    nums = 0
    for i, val in enumerate(sames):
        if val:
            nums = nums + 1

    print("Trues: " + str(nums))
    print("Falses: " + str(len(sames) - nums))


# Divs is the divisions to use, used in finding the guess parameters
def find_fit(data, divs):
    """Find the best fit to a weekly histogram.

    Parameters
    ----------
    data : array_like
        The cloudiness data used in generating a histogram.
    divs : array_like
        The lower bounds on the divisions used in generating a histogram.

    Returns
    -------
    tuple
        Tuple, where the first item is a list of coefficients corresponding
        to the best fit, and the second item is an identifying number that
        corresponds to which fit (a double gaussian or a Gaussian and Poisson
        hybrid) is the best. 0 and 1 correspond to a double Gaussian, and 2 and
        3 correspond to a Gaussian and Poisson hybrid.
    """
    split = 10
    w = 0.61 / split

    # Assign a new array so we don"t edit the original
    temp = np.copy(data)
    hist, _ = np.histogram(temp, bins=divs)

    # The guess for the first mean is the maximum that occurs before the cutoff
    # at index 35 (about 2.44)
    # And the guess for the second mean is the maximum that occurs after.
    index = np.argmax(hist[:32])
    guess = (index + 0.5) * w
    guess2 = (np.argmax(hist[32:]) + 32.5) * w

    # This block of code finds the first time the histogram falls below
    # Half the max, which gives us the half width at half maximum.
    # (Aprroximately). Since the curve is not smooth this actually
    # Isn"t great. Typically the histogram will drop below the half
    # Max before jumping up above it again for a few bins.
    less = np.where(hist < (hist[index] / 2), True, False)
    hwhm = (np.argmax(less[index:])) * w
    guesssigma = hwhm / (np.sqrt(2 * np.log(2)))

    # Guesses the frac paramter by comparing the heights of the two maximums.
    maximum = np.amax(hist)
    guessfrac = maximum/(maximum + 1 * np.amax(hist[35:]))
    guessfrac = np.arctanh(2 * guessfrac - 1)

    # Sets guessfrac to 0 if it breaks for some reason.
    if np.isnan(guessfrac) or np.isinf(guessfrac):
        guessfrac = 0

    fitarr = []
    coeffsarr = []

    fit1 = fit_function(temp)
    fitarr.append(fit1.fun)
    coeffsarr.append(fit1.x)

    fit1 = fit_function(temp, init=[0.1, guess, 0.5, guess2, guessfrac])
    fitarr.append(fit1.fun)
    coeffsarr.append(fit1.x)

    fit1 = fit_function(temp, func="gp")
    fitarr.append(fit1.fun)
    coeffsarr.append(fit1.x)

    fit1 = fit_function(temp, func="gp", init=[0.1, guess, guess2, guessfrac])
    fitarr.append(fit1.fun)
    coeffsarr.append(fit1.x)

    fitarr = np.abs(np.asarray(fitarr))
    best = np.argmin(fitarr)

    orig = best
    coeffs = np.copy(coeffsarr)

    while np.abs(coeffsarr[best][0]) < 0.002:
        print("Delta Found")
        fitarr = np.delete(fitarr, best)
        coeffsarr.pop(best)

        if fitarr.size == 0:
            best = np.argmin(fitarr)
        else:
            best = orig
            coeffsarr = np.copy(coeffs)
            coeffsarr[best][0] = 0.05
            break

    return (coeffsarr[best], orig)


# Inverted the args for this, so they match those used by scipy"s minmize.
# Minimize changes the coefficients making those the variables here.
def function_gp(params, x):
    """A Gaussian and Poisson hybrid function.

    Parameters
    ----------
    params : array_like
        A list containing all of the defining parameters for the function.
        The first item is the standard deviation of the Gaussian, the second is
        the mean of the Gaussian, the third is the mean and variance
        of the Poisson, and the fourth is a fraction that defines how biased
        the function should be towards the Gaussian. For example, if the
        provided fraction is 1 then the function is only a Gaussian,
        whereas if the provided fraction is 0 the function is only the a
        Poisson.
    x : array_like
        The x coordinates to plug into the function.

    Returns
    -------
    numpy.ndarray
        The function values corresponding to the input x values.

    Notes
    -----
    The function is normalized so that the area under the curve from 0 to
    infinity is equal to 1.

    """
    sigma = params[0]
    mu = params[1]
    lamb = params[2]
    frac = params[3]

    # A is the normalization constant, here changed because 0 is a hard cutoff.
    # So we normalize 0 to infinity rather than -infinity to infinity.
    # Trapz is the integration method using the trapezoid rule.
    x1 = np.arange(0, 400, 0.1)
    A = 1 / np.trapz(np.exp(-((x1 - mu) ** 2) / (2 * sigma * sigma)), x1)

    # This constrains frac to be between 0 and 1 using the hyperbolic tangent
    frac = (np.tanh(frac) + 1) / 2

    p1 = frac * A * np.exp(-((x - mu) ** 2) / (2 * sigma * sigma))
    p2 = (1 - frac) * (lamb ** x / special.factorial(x)) * np.exp(-lamb)

    return p1 + p2


def function_gg(params, x):
    """A double Gaussian hybrid function.

    Parameters
    ----------
    params : array_like
        A list containing all of the defining parameters for the function.
        The first item is the standard deviation of the first (lower) Gaussian,
        the second is the mean of the first Gaussian, the third is the standard
        deviation of the second (higher) Gaussian, the fourth is the mean of
        the second Gaussian, and the fifth is a fraction that defines how
        biased the function should be towards the first Gaussian.
        For example, if the provided fraction is 1 then the function is only
        the first Gaussian, whereas if the provided fraction is 0 the function
        is only the second Gaussian.
    x : array_like
        The x coordinates to plug into the function.

    Returns
    -------
    numpy.ndarray
        The function values corresponding to the input x values.

    Notes
    -----
    The function is normalized so that the area under the curve from 0 to
    infinity is equal to 1.

    """
    sigma1 = params[0]
    mu1 = params[1]
    sigma2 = params[2]
    mu2 = params[3]
    frac = params[4]

    # A is the normalization constant, here changed because 0 is a hard cutoff.
    # So we normalize 0 to infinity rather than -infinity to infinity.
    # Trapz is the integration method using the trapezoid rule.
    x1 = np.arange(0, 400, 0.1)
    A = 1 / np.trapz(np.exp(-((x1 - mu1) ** 2) / (2 * sigma1 * sigma1)), x1)
    B = 1 / np.trapz(np.exp(-((x1 - mu2) ** 2) / (2 * sigma2 * sigma2)), x1)

    # This constrains frac to be between 0 and 1 using the hyperbolic tangent
    frac = (np.tanh(frac) + 1) / 2

    p1 = frac * A * np.exp(-((x - mu1) ** 2) / (2 * sigma1 * sigma1))
    p2 = (1 - frac) * B * np.exp(-((x - mu2) ** 2) / (2 * sigma2 * sigma2))

    return p1 + p2


def likelihood_gg(params, data):
    """The log likelihood value corresponding to the double Gaussian hybrid
    function.

    Parameters
    ----------
    params : array_like
        A list containing all of the defining parameters for the function.
        The first item is the standard deviation of the first (lower) Gaussian,
        the second is the mean of the first Gaussian, the third is the standard
        deviation of the second (higher) Gaussian, the fourth is the mean of
        the second Gaussian, and the fifth is a fraction that defines how
        biased the function should be towards the first Gaussian.
        For example, if frac = 1 then the function is only the first Gaussian,
        whereas if frac = 0 the function is only the second Gaussian.
    data : array_like
        The x coordinates to plug into the function.

    Returns
    -------
    float
        The log likelihood value.

    Notes
    -----
    The log likelihood of a function is calulated using a chi-squared analysis,
    and thus is the negative sum of the natural logarithm of the function
    values.
    """
    # In chi-squared there"s a 2 in front but since we"re minimizing I"ve
    # dropped it.
    chi = -np.sum(np.log(function_gg(params, data)))
    return chi


def likelihood_gp(params, data):
    """The log likelihood value corresponding to the Gaussian and Poisson hybrid
    function.

    params : array_like
        A list containing all of the defining parameters for the function.
        The first item is the standard deviation of the Gaussian, the second is
        the mean of the Gaussian, the third is the mean and standard deviation
        of the Poisson, and the fourth is a fraction that defines how biased
        the function should be towards the Gaussian. For example, if frac = 1
        then the function is only a Gaussian, whereas if frac = 0 the function
        is only a Poisson.
    data : array_like
        The x coordinates to plug into the function.

    Returns
    -------
    float
        The log likelihood value.

    Notes
    -----
    The log likelihood of a function is calulated using a chi-squared analysis,
    and thus is the negative sum of the natural logarithm of the function
    values.
    """
    # In chi-squared there"s a 2 in front but since we"re minimizing I"ve
    # dropped it.
    chi = -np.sum(np.log(function_gp(params, data)))
    return chi


# Fits the model to the data
# I abstracted this in case I need it somewhere else.
def fit_function(xdata, func="gg", init=None):
    """Fit a function to a histogram.

    Parameters
    ----------
    xdata : array_like
        The cloudiness data used to compute a histogram.
    func : str, optional
        Which function to fit. "gg" corresponds to a double Gaussian hybrid,
        and "gp" corresponds to a Gaussian and Poisson hybrid function.
        Defaults to "gg".
    init : list, optional
        A list of initial guesses for the function fitting. Defaults to
        None. If None, uses an initial guess of [0.1,0.5,0.5,3,0] for
        a double Gaussian hybrid function, and [0.1,0.5,3,0] for a Gaussian
        and Poisson hybrid function.

    Returns
    -------
    scipy.optimize.OptimizeResult
        An object corresponding to the fit function.
    """
    if func == "gg":
        likelihood = likelihood_gg

        if init is None:
            init = [0.1,0.5,0.5,3,0]
    else:
        likelihood = likelihood_gp

        if init is None:
            init = [0.1,0.5,3,0]

    # Default for maxiter is N * 200 but that"s not enough in this case so we
    # need to specify a higher value
    fit = optimize.minimize(likelihood, x0=init, args=xdata,
                            method="Nelder-Mead",
                            options={"disp":False, "maxiter":1200})
    return fit


# Saves the cloudiness data as a csv file.
def to_csv():
    """Convert the saved daily cloudiness data to a CSV file.

    Notes
    -----
    The cloudiness values for every image saved using :func:`~analyze` are loaded and
    converted to a single CSV file: data.csv.
    The CSV file has 5 headers: Date & Time, Year, Week Number, Normalized Time
    after Sunset and Cloudiness Relative to the Mean.
    """
    directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed"])

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if ".DS_Store" in months:
        months.remove(".DS_Store")

    # Temp other stuff
    normalnum = 50
    normaldiv = 1. / normalnum

    data = []

    for month in months:

        # Gets the days that were analyzed for that month
        directory = os.path.join(os.path.dirname(__file__), *["data", "analyzed", month])
        days = sorted(os.listdir(directory))

        # Strips out the year from the month
        year = month[:4]

        # Day 1 of the year, for week calculation.
        yearstart = year + "0101"
        day1 = coordinates.timestring_to_obj(yearstart, "r_ut000000s00000")

        # Reads the data for each day.
        for day in days:
            loc = os.path.join(directory, day)
            f1 = open(loc, "r")

            # Strips off the .txt so we can make a Time object.
            day = day[:-4]
            for line in f1:
                linedata = []

                # Splits out the value and file.
                line = line.rstrip()
                line = line.split(",")
                val = float(line[1])
                name = line[0]

                # Moon phase calculation.
                phase = moon.moon_visible(day, name)

                # Ignores cloudiness with moon phase less than 0.2
                if phase < 0.2:
                    continue

                date = coordinates.timestring_to_obj(day, name)

                # Finds the difference since the beginning of the year to find
                # The week number.
                diff = date - day1
                week = int(diff.value // 7)

                # Sunset time calculation.
                # 12 hours after sunset for 50 bins = 0.01 of a day per bin.
                formatdate = day[:4] + "/" + day[4:6] + "/" + day[6:]
                time = name[4:6] + ":" + name[6:8] + ":" + name[8:10]
                formatdate = formatdate + " " + time

                linedata.append(formatdate)
                linedata.append(year)
                linedata.append(week)

                # Sets the date of calculation.
                camera.date = formatdate
                date = ephem.Date(formatdate)

                # Calculates the previous setting and next rising of the sun.
                sun = ephem.Sun()
                setting = camera.previous_setting(sun, use_center=True)
                rising = camera.next_rising(sun, use_center=True)

                # Finds the difference and bins it into the correct bin.
                diff = date - setting
                length = rising - setting
                normalize = int((diff / length) // normaldiv)

                linedata.append(normalize)
                linedata.append(val)

                data.append(np.asarray(linedata))

    data = np.asarray(data)

    d2 = pd.DataFrame(data, columns=["Date & Time", "Year", "Week Number",
                                     "Normalized Time after Sunset",
                                     "Cloudiness Relative to the Mean"])

    d2.to_csv("data.csv")


if __name__ == "__main__":
    #analyze()
    plot(["2014", "2015", "2016", "2017"])
