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

# Internal Imports


# Silence Superflous AstroPy Warnings
warnings.simplefilter('ignore', AstropyWarning)


# Create various error classes to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


# Create various alert classes to use
class _BaseAlert():
    """_Baselert Base alert class, to contain useful common things
    """
    def __init__(self):
        pass


class BadDirectoryAlert(_BaseAlert):
    """ConfluenceAlert Caused by a bad directory used in the call
    """
    def __init__(self):
        _BaseAlert.__init__(self)
        self.type = 'Bad Directory'


class CantRunAlert(_BaseAlert):
    """ConfluenceAlert Caused by the code not being able to run
    """
    def __init__(self):
        _BaseAlert.__init__(self)
        self.type = 'Cannot Run on Anything'


class ConfluenceAlert(_BaseAlert):
    """ConfluenceAlert Caused by an issue connecting with Confluence
    """
    def __init__(self):
        _BaseAlert.__init__(self)
        self.type = 'Confluence Issue'


def send_alert(alertclass):
    """send_alert Send out an alert

    The medium for alerts needs to be decided -- should it be via email from
    lig.nanni@lowell.edu, or over Slack, or both, or something different?

    There are various types of alerts that can be send... maybe choose the
    medium based on the input `alertclass`?
    """
    print(f"Alert Alert Alert: {alertclass.type}")
