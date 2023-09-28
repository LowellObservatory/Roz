# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 31-May-2022
#
#  @author: tbowers

"""Alerting system for Roz.

This module is part of the Roz package, written at Lowell Observatory.

If problems are found at any point in the code base, alerts are issued via
email/slack for humans to check on.

This module primarily trades in... not quite sure yet.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

# Built-In Libraries
import inspect
import os
import pathlib

# 3rd Party Libraries
import astropy.table

# Lowell Libraries

# Internal Imports
from roz import graphics_maker
from roz import msgs
from roz import slack
from roz import utils

# Constants
MACHINE = os.uname()[1].split(".")[0]


def send_alert(alert_type, no_slack=False, **kwargs):
    """Send out an alert

    The medium for alerts needs to be decided -- should it be via email from
    lig`dot`nanni`at`lowell`dot`edu, or over Slack, or both, or something
    different?

    There are various types of alerts that can be sent... maybe choose the
    medium based on the input ``alertclass``?

    Parameters
    ----------
    alert_type : str
        One of several recognized alert types
    no_slack : bool, optional
       Do not post to Slack (Default: False)
    """

    # Parse out kwargs
    caller = kwargs.get("caller", construct_caller(inspect.stack()[1]))

    # Case out `alert_type`
    if alert_type == "empty_dir":
        dumbwaiter = kwargs.get("dumbwaiter")
        frameclass = kwargs.get("frameclass")
        alert_msg = (
            f"Empty Directory: `{utils.subpath(dumbwaiter.dirs['data'])}` "
            f"does not contain any sequential {frameclass} FITS files"
        )

    elif alert_type == "not_implemented":
        dumbwaiter = kwargs.get("dumbwaiter")
        alert_msg = f"Function not yet implemented for data in {dumbwaiter.nightname}"

    elif alert_type == "dir_not_found":
        dirname = kwargs.get("dirname")
        alert_msg = f"Directory not found at `{dirname}`"

    elif alert_type == "no_inst_found":
        dirname = kwargs.get("dirname")
        alert_msg = f"No instrument found in `{dirname}`"

    elif alert_type == "inst_not_support":
        inst = kwargs.get("inst")
        alert_msg = (
            f"Instrument `{inst}` not yet supported; update instrument_flags.ecsv"
        )

    elif alert_type == "file_not_open":
        filename = kwargs.get("filename")
        exception = kwargs.get("exception")
        alert_msg = f"Could not open {filename} because of {exception}."

    elif alert_type == "text":
        alert_msg = kwargs.get("text")

    else:
        alert_msg = ""

    # Emit the alert message to screen and slack
    msgs.warn(f"{alert_msg.replace('`','')}: {caller}")
    if not no_slack:
        slack.send(f"Alert from Roz on `{MACHINE}`:\n{alert_msg}: `{caller}`")


def post_report(report):
    """Post the Problem Report to Slack

    _extended_summary_

    Parameters
    ----------
    report : str
        The problem report, as a string with occasional newlines
    """
    slack.send(f"Problem report from Roz on `{MACHINE}`:")

    # Split up the report into frame sections to meet the message size limit
    rlist = report.split("*.*.")
    for subreport in rlist:
        if subreport.strip() == "":
            continue
        slack.send(f"```\n{subreport}```")


def post_pngs(metadata_tables, directory, inst_flags):
    """Post the problematic PNGs to Slack

    _extended_summary_

    Parameters
    ----------
    metadata_tables : dict
        _description_
    directory : :obj:`pathlib.Path`
        Path to the processing directory
    """
    # Loop through Frame Types
    for ftype in metadata_tables:
        if metadata_tables[ftype]:
            # Loop through Filters
            for filt in metadata_tables[ftype]:
                if isinstance(metadata_tables[ftype][filt], astropy.table.Table):
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
                        slack.file(
                            str(utils.Paths.thumbnail.joinpath(png_fn)),
                            title=png_fn,
                        )


def construct_caller(stack):
    """Construct the calling function name

    Form: ``module.function():lineno``

    Parameters
    ----------
    stack : :obj:`inspect.stack`
        The current stack, position [1], which is the caller.

    Returns
    -------
    str
        The calling function in module context
    """
    module = pathlib.Path(stack.filename).stem
    return f"{module}.{stack.function}():{stack.lineno}"
