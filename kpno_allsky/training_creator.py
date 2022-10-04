#!/usr/bin/env python3
import os
import random
from shutil import copyfile
import time

from PIL import Image
from matplotlib.widgets import RectangleSelector
import numpy as np

# The following two lines force matplotlib to use TkAgg as the backend.
# Blitting seems to be broken for me on Qt5Agg which is why I've switched.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

import coordinates
from image import AllSkyImage
import image
import mask
import moon
from io_util import DateHTMLParser
import io_util


class TaggableImage:
    def __init__(self, name, update=False, camera="kpno"):
        if update and camera != "kpno":
            raise ValueError("Cannot run update and Spacewatch mode simultaneously")

        self.press = False
        self.name = name
        self.camera = camera

        # Artists for blitting.
        self.artists = []

        # Loads the image and reshapes to 512 512 in greyscale.
        if not update:
            loc = os.path.join(os.path.dirname(__file__),
                               *["Images", "data", "to_label", name])
        else:
            loc = os.path.join(os.path.dirname(__file__),
                               *["Images", "data", "train", "6", name])
        with open(loc, 'rb') as f:
            if camera.lower() == "kpno":
                img = Image.open(f).convert('L')
                self.img = np.asarray(img).reshape((512, 512))
            else:
                img = Image.open(f).convert("RGB")
                self.img = np.asarray(img).reshape((1024, 1024, 3))

        # The grid division
        self.div = 16
        self.good = True

        self.update = update

        # The masking image that we're creating.
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype="uint8")

        if update:
            loc = os.path.join(os.path.dirname(__file__),
                               *["Images", "data", "labels", "6", name])
            with open(loc, 'rb') as f:
                self.mask = np.array(Image.open(f).convert('L'))

        # This is gray, but we change it to 255 if we're labeling ghosts.
        self.val = 128


    def set_up_plot(self):
        # Sets up the figure and axis.
        fig, ax = plt.subplots()
        fig.set_size_inches(15,15)
        self.artists.append(ax.imshow(self.img, cmap='gray'))

        # Center and radii for the circles of 30 degrees altitude and the
        # circle of minimum safe distance (green)
        center = coordinates.center_kpno if self.camera.lower() == "kpno" else coordinates.center_sw
        r = 167 if self.camera.lower() == "kpno" else 330
        # Circle at 30 degrees altitude, where the training patches end.
        circ1 = Circle(center, radius=r, fill=False, edgecolor="cyan")
        self.artists.append(ax.add_patch(circ1))

        # Extra ten pixels in the radius so we are sure to get any pixels that
        # would be caught in the training patches.
        circ2 = Circle(center, radius=r + 10, fill=False, edgecolor="green")
        self.artists.append(ax.add_patch(circ2))

        # This is a little hacky but it recreates the grid shape with individual
        # rectangular patches. This way the grid can be updated with the rest
        # of the image upon clicking.
        upper = 64 if self.camera.lower() == "kpno" else 160
        lower = 448 if self.camera.lower() == "kpno" else 864
        range_top = 24 if self.camera.lower() == "kpno" else 45
        for i in range(1, range_top):
            height = i * self.div
            r = Rectangle((upper, upper), height, height, edgecolor="m", fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((upper, lower-height), height, height, edgecolor="m", fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((lower-height, upper), height, height, edgecolor="m", fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((lower-height, lower-height), height, height, edgecolor="m", fill=False)
            self.artists.append(ax.add_patch(r))

        # Shows the divisions on the x and y axis.
        if self.camera.lower() == "kpno":
            grid = np.arange(0, 513, self.div)
        else:
            grid = np.arange(0, 513 * 2, self.div * 2)
        plt.xticks(grid)
        plt.yticks(grid)

        viridis_mod = cm.viridis
        viridis_mod.set_under("k", alpha=0)
        self.artists.append(ax.imshow(self.mask, cmap=viridis_mod, alpha=0.25,
                                      vmin=0, vmax=255, animated=True,
                                      clim=[1, 255]))

        self.ax = ax
        self.fig = fig


    def on_click(self, event):
        self.press = True
        # Finds the box that the click was in.
        x = event.xdata
        x = int(x//self.div * self.div)

        y = event.ydata
        y = int(y//self.div * self.div)

        # Sets the box to white, and then updates the plot with the new
        # masking data.
        if not self.update:
            self.mask[y:y + self.div, x:x + self.div] = self.val
        else:
            self.mask[y:y + self.div, x:x + self.div] = 255
        self.artists[-1].set_data(self.mask)

        # Does the blitting update
        for a in self.artists:
            self.ax.draw_artist(a)
        self.fig.canvas.blit(self.ax.bbox)


    def on_motion(self, event):
        # If we're not currently clicked while moving don't make the box white.
        if not self.press:
            return

        # Don't want to run all this code if the box we're in is already white.
        if self.mask[int(event.ydata), int(event.xdata)] == self.val:
            return

        # Finds the box that we're currently in.
        x = event.xdata
        x = int(x//self.div * self.div)

        y = event.ydata
        y = int(y//self.div * self.div)

        # Sets the box to white, and then updates the plot with the new
        # masking data.
        if not self.update:
            self.mask[y:y + self.div, x:x + self.div] = self.val
        else:
            self.mask[y:y + self.div, x:x + self.div] = 255
        self.artists[-1].set_data(self.mask)

        # Does the blitting update
        for a in self.artists:
            self.ax.draw_artist(a)
        self.fig.canvas.blit(self.ax.bbox)


    def on_release(self, event):
        self.press = False


    def connect(self):
        cidpress = self.fig.canvas.mpl_connect('button_press_event',
                                               self.on_click)
        cidmove = self.fig.canvas.mpl_connect('motion_notify_event',
                                               self.on_motion)
        cidrelease = self.fig.canvas.mpl_connect('button_release_event',
                                                 self.on_release)


    def save(self):
        # When the plot is closed we save the newly created label mask.
        save_im = image.AllSkyImage(self.name, None, None, self.mask)

        if self.camera.lower() == "kpno":
            # Gets the exposure for the saving location.
            exp_im = image.AllSkyImage(self.name, None, None, self.img)
            exp = image.get_exposure(exp_im)
            loc = os.path.join(os.path.dirname(__file__),
                               *["Images", "data", "labels", str(exp)])

            # Maks the antenna
            m = mask.generate_mask()
            save_im = mask.apply_mask(m, save_im)
        else:
            loc = os.path.join(os.path.dirname(__file__),
                               *["Images", "data", "labels-sw"])

        # Saves the image.
        image.save_image(save_im, loc)

        if not self.update:
            # Moves the downloaded image into the training folder.
            loc = os.path.join(os.path.dirname(__file__),
                               *["Images", "data", "to_label", self.name])
            if self.camera.lower() == "kpno":
                dest = os.path.join(os.path.dirname(__file__),
                                    *["Images", "data", "train", str(exp), self.name])
            else:
                dest = os.path.join(os.path.dirname(__file__),
                                    *["Images", "data", "train", "sw", self.name])
            os.rename(loc, dest)
            print("Moved: " + loc)

    def cleanup(self, event):
        # Deletes the downloaded image so that we don't have it clogging
        # everything up.
        loc = os.path.join(os.path.dirname(__file__),
                           *["Images", "data", "to_label", self.name])
        os.remove(loc)
        self.good = False
        print("Deleted: " + loc)


    # Swaps the tag value from cloud to ghost.
    def swap(self, event):
        if self.val == 255:
            self.val = 128
        else:
            self.val = 255



def get_image_kpno(update, i=0):
    if not update:
        # Verifying that the label location exists
        label_loc = os.path.join(os.path.dirname(__file__),
                                 *["Images", "data", "to_label"])
        if not os.path.exists(label_loc):
            os.makedirs(label_loc)

        # The link to the camera.
        link = "http://kpasca-archives.tuc.noao.edu/"

        # This extracts the dates listed and then picks one at random.
        data = io_util.download_url(link)
        htmldata = data.text
        parser = DateHTMLParser()
        parser.feed(htmldata)
        parser.close()
        date = random.choice(parser.data)
        parser.clear_data()

        link = link + date

        # This extracts the images from the given date and then picks at random.
        data = io_util.download_url(link)
        htmldata = data.text
        parser.feed(htmldata)
        parser.close()
        image = random.choice(parser.data)

        # Need to verify that we're actually in twilight
        all_sky = AllSkyImage(image, date, "KPNO", None)
        sunalt = moon.find_sun(all_sky)[0]
        bad_name = image == "allblue.gif" or image == "allred.gif" or image[:1] == 'b'

        # This loop ensures that we don't accidentally download the all night
        # gifs or an image in a blue filter or an image that isn't twilight
        while bad_name or sunalt > -17:
            image = random.choice(parser.data)

            # Regenerates the conditions
            bad_name = image == "allblue.gif" or image == "allred.gif" or image[:1] == 'b'
            all_sky = AllSkyImage(image, date, "KPNO", None)
            sunalt = moon.find_sun(all_sky)[0]

        # Downloads the image
        io_util.download_image(date[:8], image, directory=label_loc)

        # Returns the image name.
        return image
    else:
        images = os.listdir(os.path.join(os.path.dirname(__file__),
                                         *["Images", "data", "train", "6"]))
        images = sorted(images)
        if images[0] == ".DS_Store":
            return images[i+1]
        return images[i]


def get_image_sw(i=0):
    base_loc = os.path.join(os.path.dirname(__file__),
                            *["Images", "Original", "SW"])
    all_dates = os.listdir(base_loc)
    if ".DS_Store" in all_dates:
        all_dates.remove(".DS_Store")
    date = random.choice(all_dates)

    all_images = os.listdir(os.path.join(base_loc, date))
    if ".DS_Store" in all_images:
        all_images.remove(".DS_Store")
    image = random.choice(all_images)

    # Once we have an image name we copy it to Images/data/to_label
    # First we need to make sure it exists.
    label_loc = os.path.join(os.path.dirname(__file__),
                             *["Images", "data", "to_label"])
    if not os.path.exists(label_loc):
        os.makedirs(label_loc)

    # Downloads the image
    copyfile(os.path.join(base_loc, *[date, image]), os.path.join(label_loc, image))

    # Returns the image name.
    return image



if __name__ == "__main__":
    update = False
    camera = "kpno"

    if camera.lower() == "kpno":
        done = {}
        # The list of all the pictures that have already been finished.
        finished_loc = os.path.join(os.path.dirname(__file__),
                                    *["Images", "data", "labels", "0.3"])
        if not os.path.exists(finished_loc):
            os.makedirs(finished_loc)
        done["0.3"] = os.listdir(finished_loc)

        # Separate out the 0.3s and 6s images.
        finished_loc = os.path.join(os.path.dirname(__file__),
                                    *["Images", "data", "labels", "6"])
        if not os.path.exists(finished_loc):
            os.makedirs(finished_loc)
        done["6"] = os.listdir(finished_loc)
    else:
        # The list of all the pictures that have already been finished.
        finished_loc = os.path.join(os.path.dirname(__file__),
                                    *["Images", "data", "labels-sw"])
        if not os.path.exists(finished_loc):
            os.makedirs(finished_loc)
        done = os.listdir(finished_loc)

    i = 0
    # We run this loop until the user kills the program.
    while True:
        # Loads the image into the frame to label.
        if camera.lower() == "kpno":
            name = get_image_kpno(update, i)
            good = not update and ((not name in done["0.3"]) or (not name in done["6"]))
        else:
            name = get_image_sw(i)
            good = not update and not name in done
        if good:
            im = TaggableImage(name, camera=camera)
            im.set_up_plot()
            im.connect()

            # Adds the swap button.
            b_ax = plt.axes([0.5, 0.1, 0.1, 0.05])
            button1 = Button(b_ax, "Swap Label")
            button1.on_clicked(im.swap)

            # Adds the bad image button
            b_ax = plt.axes([0.6, 0.1, 0.1, 0.05])
            button2 = Button(b_ax, "Bad Image")
            button2.on_clicked(im.cleanup)

            plt.show()
            print(im.good)
            if im.good:
                im.save()
                i += 1

            print("Num images:" + str(i))

        elif update:
            im = TaggableImage(name, update, camera=camera)
            im.set_up_plot()
            im.connect()

            # Adds the bad image button
            b_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
            button = Button(b_ax, "Bad Image")
            button.on_clicked(im.cleanup)

            plt.show()

            if im.good:
                im.save()

            i += 1
