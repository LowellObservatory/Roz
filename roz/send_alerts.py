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
import warnings

# 3rd Party Libraries
from astropy.utils.exceptions import AstropyWarning

# Lowell Libraries
from johnnyfive import gmail as j5g
from johnnyfive import slack as j5s

# Internal Imports
from roz.utils import InputError

# Silence Superflous AstroPy Warnings
warnings.simplefilter("ignore", AstropyWarning)


def send_alert(alert_type):
    """send_alert Send out an alert

    The medium for alerts needs to be decided -- should it be via email from
    lig.nanni@lowell.edu, or over Slack, or both, or something different?

    There are various types of alerts that can be sent... maybe choose the
    medium based on the input `alertclass`?
    """
    print(f"***** Alert Alert Alert: {alert_type}")
