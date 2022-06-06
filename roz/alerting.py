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
"""

# Built-In Libraries
import os

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


def send_alert(alert_type, caller=None, no_slack=False):
    """send_alert Send out an alert

    The medium for alerts needs to be decided -- should it be via email from
    lig.nanni@lowell.edu, or over Slack, or both, or something different?

    There are various types of alerts that can be sent... maybe choose the
    medium based on the input `alertclass`?
    """

    msgs.warn(f"{alert_type.replace('`','')}: {caller}")

    if not no_slack:
        slack.send(f"Alert from Roz on `{MACHINE}`:\n{alert_type}: `{caller}`")


def post_report(report):
    """post_report Post the Problem Report to Slack

    _extended_summary_

    Parameters
    ----------
    report : `str`
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
    """post_pngs Post the problematic PNGs to Slack

    _extended_summary_

    Parameters
    ----------
    metadata_tables : `dict`
        _description_
    directory : `pathlib.Path`
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
