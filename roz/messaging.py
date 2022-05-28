# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 22-Oct-2021
#
#  @author: tbowers

"""Messaging system for Roz.

This module is part of the Roz package, written at Lowell Observatory.

This module is based heavily upon `pypeit.pypmsgs`, with gratitude to
those authors.

If problems are found at any point in the code base, alerts are issued via
email/slack for humans to check on.

This module primarily trades in... not quite sure yet.
"""

# Built-In Libraries
import inspect
import os
import sys

# 3rd Party Libraries
import astropy
from astropy.table import Table
import numpy
import scipy

# Lowell Libraries
from johnnyfive import slack as j5s

# Internal Imports
from roz import graphics_maker
from roz import utils
import roz

# Constants
MACHINE = os.uname()[1].split(".")[0]


class RozError(Exception):
    """Custom Error class"""


class Messages:
    """
    Create colored text for messages printed to screen.

    For further details on colors see the following example:
    http://ascii-table.com/ansi-escape-sequences.php

    Parameters
    ----------
    log : str or None
      Name of saved log file (no log will be saved if log=="")
    verbosity : int (0,1,2)
      Level of verbosity:
        0 = No output
        1 = Minimal output
        2 = All output (default)
    colors : bool
      If true, the screen output will have colors, otherwise
      normal screen output will be displayed
    """

    def __init__(self, log=None, verbosity=None, colors=True):

        self._defverb = 1
        self._verbosity = self._defverb if verbosity is None else verbosity

        # Initialize the log
        self._log = None
        self._initialize_log_file(log=log)

        # Initialize the color attributes
        self._start = None
        self._end = None
        self._black_clear = None
        self._yellow_clear = None
        self._blue_clear = None
        self._green_clear = None
        self._red_clear = None
        self._white_red = None
        self._white_green = None
        self._white_black = None
        self._white_blue = None
        self._black_yellow = None
        self._yellow_black = None

        self._disablecolors()
        if colors:
            self._enablecolors()

    # Internal Methods =============================================#
    def _devmsg(self):
        """
        Prints the module, line, and function at the current call
        if enough verbosity is set
        """
        if self._verbosity == 2:
            info = inspect.getouterframes(inspect.currentframe())[3]
            return (
                f"{self._start}{self._blue_clear}{info[1].split('/')[-1]} "
                f"{str(info[2])} {info[3]}(){self._end} - "
            )
        return ""

    def _print(self, premsg, msg, last=True):
        """
        Print to standard error and the log file
        """
        devmsg = self._devmsg()
        _msg = premsg + devmsg + msg
        if self._verbosity != 0:
            print(_msg, file=sys.stderr)
        if self._log:
            clean_msg = self._cleancolors(_msg)
            self._log.write(clean_msg + "\n" if last else clean_msg)

    def _initialize_log_file(self, log=None):
        """
        Expects self._log is already None.
        """
        if log is None:
            return

        # Initialize the log
        self._log = open(log, "w", encoding="utf-8")

        self._log.write("------------------------------------------------------\n\n")
        self._log.write(
            f"This log was generated with version {roz.__version__} of Roz\n\n"
        )
        self._log.write(f"You are using scipy version={scipy.__version__}\n")
        self._log.write(f"You are using numpy version={numpy.__version__}\n")
        self._log.write(f"You are using astropy version={astropy.__version__}\n\n")
        self._log.write("------------------------------------------------------\n\n")

    def _reset_log_file(self, log):
        if self._log:
            self._log.close()
            self._log = None
        self._initialize_log_file(log=log)

    # Set the colors ===============================================#
    def _enablecolors(self):
        """
        Enable colored output text
        """

        # Start and end colored text
        self._start = "\x1B["
        self._end = "\x1B[" + "0m"

        # Clear Backgrounds
        self._black_clear = "1;30m"
        self._yellow_clear = "1;33m"
        self._blue_clear = "1;34m"
        self._green_clear = "1;32m"
        self._red_clear = "1;31m"

        # Colored Backgrounds
        self._white_red = "1;37;41m"
        self._white_green = "1;37;42m"
        self._white_black = "1;37;40m"
        self._white_blue = "1;37;44m"
        self._black_yellow = "1;37;43m"
        self._yellow_black = "1;33;40m"

    def _disablecolors(self):
        """
        Disable colored output text
        """

        # Start and end colored text
        self._start = ""
        self._end = ""

        # Clear Backgrounds
        self._black_clear = ""
        self._yellow_clear = ""
        self._blue_clear = ""
        self._green_clear = ""
        self._red_clear = ""

        # Coloured Backgrounds
        self._white_red = ""
        self._white_green = ""
        self._white_black = ""
        self._white_blue = ""
        self._black_yellow = ""
        self._yellow_black = ""

    def _cleancolors(self, msg):
        cols = [
            self._end,
            self._start,
            self._black_clear,
            self._yellow_clear,
            self._blue_clear,
            self._green_clear,
            self._red_clear,
            self._white_red,
            self._white_green,
            self._white_black,
            self._white_blue,
            self._black_yellow,
            self._yellow_black,
        ]
        for i in cols:
            msg = msg.replace(i, "")
        return msg

    # Public-facing methods ========================================#
    def reset(self, log=None, verbosity=None, colors=True):
        """
        Reinitialize the object.

        Needed so that there can be a default object for all modules,
        but also a dynamically defined log file.
        """
        # Initialize other variables
        self._verbosity = self._defverb if verbosity is None else verbosity
        self._reset_log_file(log)
        self._initialize_log_file(log=log)
        self._disablecolors()
        if colors:
            self._enablecolors()

    def close(self):
        """
        Close the log file before the code exits
        """
        return self._reset_log_file(None)

    def error(self, msg):
        """
        Print an error message
        """
        premsg = "\n" + self._start + self._white_red + "[ERROR]   ::" + self._end + " "
        self._print(premsg, msg)

        # Close log file and raise an exception
        self.close()
        raise RozError(msg)

    def info(self, msg):
        """
        Print an information message
        """
        premsg = f"{self._start}{self._green_clear}[INFO]    ::{self._end} "
        self._print(premsg, msg)

    def table(self, msg):
        """
        Print a test message
        """
        premsg = f"{self._start}{self._blue_clear}[SUMMARY] ::{self._end} "
        self._print(premsg, msg)

    def validate(self, msg):
        """
        Print a test message
        """
        premsg = f"{self._start}{self._yellow_clear}[VALIDATE]::{self._end} "
        self._print(premsg, msg)

    def test(self, msg):
        """
        Print a test message
        """
        if self._verbosity == 2:
            premsg = f"{self._start}{self._white_blue}[TEST]    ::{self._end} "
            self._print(premsg, msg)

    def warn(self, msg):
        """
        Print a warning message
        """
        premsg = f"{self._start}{self._red_clear}[WARNING] ::{self._end} "
        self._print(premsg, msg)

    def bug(self, msg):
        """
        Print a bug message
        """
        premsg = f"{self._start}{self._white_black}[BUG]     ::{self._end} "
        self._print(premsg, msg)

    def work(self, msg):
        """
        Print a work in progress message
        """
        if self._verbosity == 2:
            premsg = (
                f"{self._start}{self._black_clear}[WORK IN ]::{self._end}\n"
                f"{self._start}{self._yellow_clear}[PROGRESS]::{self._end} "
            )
            self._print(premsg, msg)

    @staticmethod
    def newline():
        """
        Return a text string containing a newline to be used with messages
        """
        return "\n             "


def send_alert(alert_type, caller=None, no_slack=False):
    """send_alert Send out an alert

    The medium for alerts needs to be decided -- should it be via email from
    lig.nanni@lowell.edu, or over Slack, or both, or something different?

    There are various types of alerts that can be sent... maybe choose the
    medium based on the input `alertclass`?
    """
    from roz import msgs

    msgs.warn(f"{alert_type.replace('`','')}: {caller}")

    # TODO: Gotta find a way to not re-init the Slack instance with each call.
    #       Otherwise, with each alert (could be many in short sequence) the
    #       code goes through the whole initialization mess (disk reads,
    #       credential handshakes, etc.).

    if not no_slack:
        slack_alert = j5s.SlackChannel("bot_test")
        slack_alert.send_message(
            f"Alert from Roz on `{MACHINE}`:\n{alert_type}: `{caller}`"
        )


def post_report(report):
    """post_report Post the Problem Report to Slack

    _extended_summary_

    Parameters
    ----------
    report : `str`
        The problem report, as a string with occasional newlines
    """
    slack_report = j5s.SlackChannel("bot_test")
    slack_report.send_message(f"Problem report from Roz on `{MACHINE}`:")

    # Split up the report into frame sections to meet the message size limit
    rlist = report.split("*.*.")
    for subreport in rlist:
        if subreport.strip() == "":
            continue
        slack_report.send_message(f"```\n{subreport}```")


def post_pngs(metadata_tables, directory, inst_flags):
    """post_pngs Post the problematic PNGs to Slack

    _extended_summary_

    Parameters
    ----------
    metadata_tables : `dict`
        _description_
    directory : `pathlib.Path`
        Path to the processing directory
    """
    slack_report = j5s.SlackChannel("bot_test")

    # Loop through Frame Types
    for ftype in metadata_tables:
        if metadata_tables[ftype]:

            # Loop through Filters
            for filt in metadata_tables[ftype]:
                if isinstance(metadata_tables[ftype][filt], Table):

                    # Grab frames marked as "PROBLEM"
                    mask = metadata_tables[ftype][filt]["problem"] == 1
                    pngs = list(metadata_tables[ftype][filt]["filename"][mask])

                    # Go through the files one by one, make PNGs, and upload
                    for png in pngs:
                        png_fn = graphics_maker.make_png_thumbnail(
                            directory.joinpath(png),
                            inst_flags,
                            problem=True,
                            debug=False,
                        )

                        # NOTE: SlackChannel.upload_file() requires `str` filename
                        slack_report.upload_file(
                            str(utils.Paths.thumbnail.joinpath(png_fn)),
                            title=png_fn,
                        )
