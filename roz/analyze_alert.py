# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 22-Oct-2021
#
#  @author: tbowers

"""Analyze the metadata tables, and issue alerts if there are "problems"

This module is part of the Roz package, written at Lowell Observatory.

This module analyzes the metadata from the frames in question for quality
assurance.  If problems are found, alerts are issued via email/slack for
humans to check on.

This module primarily trades in AstroPy Table objects (`astropy.table.Table`).
"""

# Built-In Libraries
import warnings

# 3rd Party Libraries
from astropy.utils.exceptions import AstropyWarning
import numpy as np

# Internal Imports


# Silence Superflous AstroPy Warnings
warnings.simplefilter('ignore', AstropyWarning)


# Create various error classes to use
class InputError(ValueError):
    """InputError Locally defined error that inherits ValueError
    """


# Create various alert classes to use
class ConfluenceAlert():
    """ConfluenceAlert Caused by an issue connecting with Confluence
    """


def validate_bias_table(bias_meta):
    """validate_bias_table Analyze and validate the bias frame metadata table

    [extended_summary]

    Parameters
    ----------
    bias_meta : `astropy.table.Table`
        A table containing information about the bias frames for analysis

    Returns
    -------
    `astropy.table.Table`
        The, um, validated table?  This may change.
    """
    # For now, just print some stats and return the table.
    print("\nIn validate_bias_table():")
    print(np.mean(bias_meta['crop_avg']), np.median(bias_meta['crop_med']),
          np.mean(bias_meta['crop_std']))

    # Add logic checks for header datatypes (edge cases)

    return bias_meta


def validate_flat_table(flat_meta, lmi_filt):
    """validate_flat_table Analyze and validate the flat frame metadata table

    Separates the wheat from the chaff -- returning a subtable for the
    specified filter, or None.

    Parameters
    ----------
    flat_meta : `astropy.table.Table`
        Table containing the flat frame metadata
    lmi_filt : `str`
        LMI filter to validate

    Returns
    -------
    `astropy.table.Table` or `None`
        If the `lmi_filt` was used in this set, return the subtable of
        `flat_meta` containing that filter.  Otherwise, return `None`.
    """
    # Find the rows of the table corresponding to this filter, return if 0
    idx = np.where(flat_meta['filter'] == lmi_filt)
    if len(idx[0]) == 0:
        return None

    # For ease, pull these rows into a subtable
    subtable = flat_meta[idx]

    # Make sure 'flats' have a reasonable flat countrate, or total counts
    #  in the range 1,500 - 52,000 ADU above bias.  (Can get from countrate *
    #  exptime).

    # Do something...
    print("\nIn validate_flat_table():")
    print(lmi_filt)
    subtable.pprint()
    print(np.mean(subtable['frame_avg']), np.median(subtable['frame_med']))

    # Find the mean quadric surface for this set of flats
    # quadsurf = np.mean(np.asarray(subtable['quadsurf']), axis=0)
    # print(quadsurf)

    return subtable


def send_alert(alertclass):
    """send_alert Send out an alert for "funny" frames

    The medium for alerts needs to be decided -- should it be via email from
    lig.nanni@lowell.edu, or over Slack, or both, or something different?
    """
