# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 22-Oct-2021
#
#  @author: tbowers

"""Alert for various issues.

This module is part of the Roz package, written at Lowell Observatory.

If problems are found at any point in the code base, alerts are issued via
email/slack for humans to check on.

This module primarily trades in Exception-like `Alert` objects.
"""

# Built-In Libraries
import os
import warnings

# 3rd Party Libraries

# Lowell Libraries
from johnnyfive import gmail as j5g
from johnnyfive import slack as j5s

# Internal Imports


# Constants
MACHINE = os.uname()[1].split(".")[0]


def send_alert(alert_type, caller=None):
    """send_alert Send out an alert

    The medium for alerts needs to be decided -- should it be via email from
    lig.nanni@lowell.edu, or over Slack, or both, or something different?

    There are various types of alerts that can be sent... maybe choose the
    medium based on the input `alertclass`?
    """
    print(f"***** Alert: {alert_type.replace('`','')}: {caller}")

    # TODO: Gotta find a way to not re-init the Slack instance with each call.
    #       Otherwise, with each alert (could be many in short sequence) the
    #       code goes through the whole initialization mess (disk reads,
    #       credential handshakes, etc.).

    slack_alert = j5s.SlackChannel("bot_test")
    slack_alert.send_message(f"From Roz on `{MACHINE}`:: {alert_type}: `{caller}`")


def build_problem_report(validation_dict):
    """build_problem_report _summary_

    _extended_summary_

    Parameters
    ----------
    validation_dict : _type_
        _description_
    """
