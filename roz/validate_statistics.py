# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 22-Mar-2022
#
#  @author: tbowers

"""Validate the statistics for 1 night against accumulated Historical Data

This module is part of the Roz package, written at Lowell Observatory.

This module takes the computed statistics from a collection of grames and
validates them against the historical data pulled from the InfluxDB.  This
module has the dual purpose of comparing the new frames with the historical
corpus and adding that information to both the database object and outgoing
alerts.

This module primarily trades in AstroPy Table objects (`astropy.table.Table`)
and the internal database objects (`roz.database_manager.HistoricalData` and
`roz.database_manager.CalibrationDatabase`).
"""

# Built-In Libraries
import os
import warnings

# 3rd Party Libraries
from astropy.stats import mad_std
from astropy.table import Table
from astropy.wcs import FITSFixedWarning
import ccdproc as ccdp
from ccdproc.utils.slices import slice_from_string as get_slice
import numpy as np
from tqdm import tqdm

# Internal Imports
from roz import database_manager as dm
from roz import send_alerts as sa
from roz import utils


def validate_bias_table(bias_meta, junk=None):
    return validate_metadata_table(bias_meta, "bias")


def validate_metadata_table(meta_table, frametype=None):
    """validate_bias_table Analyze and validate the bias frame metadata table

    [extended_summary]

    Parameters
    ----------
    meta_table : `astropy.table.Table`
        A table containing information about the frames for analysis

    Returns
    -------
    `astropy.table.Table`
        The, um, validated table?  This may change.
    """
    # If there were no biases at all (blank Table), return None
    if not meta_table:
        return None

    # These are the metrics we will validate (remove problematic metrics)
    metrics = [
        metric
        for metric in meta_table.colnames
        if any(s in metric for s in ["qs_", "crop_", "frame_", "_flat"])
    ]
    for removal in ["qs_maj", "qs_bma", "qs_open"]:
        metrics.remove(removal)

    # Print the banner
    print(f"==> Validating {frametype.upper()} in validate_metadata_table():")

    # Pull the Historical Data matching this set
    hist = dm.HistoricalData(
        sorted(list(set(meta_table["instrument"])))[0].lower(),
        frametype,
        binning=sorted(list(set(meta_table["binning"])))[0],
        numamp=sorted(list(set(meta_table["numamp"])))[0],
        ampid=sorted(list(set(meta_table["ampid"])))[0],
        debug=False,
    )
    hist.perform_query()

    # Build some quick dictionaries containing the Gaussian statistics
    mu = {}
    sig = {}
    for check in metrics:
        mu[check] = hist.metric_mean(check)
        sig[check] = hist.metric_stddev(check)

    # Loop through the frames one by one
    for row in meta_table:
        # Then, loop over the list of metrics in bias_meta that should be compared
        for check in metrics:
            # Greater than 3 sigma deviation, alert
            if (deviation := np.abs(row[check] - mu[check]) / sig[check]) > 3.0:
                sa.send_alert(
                    f"{frametype.upper()} frame {row['obserno']} with timestamp "
                    f"`{row['dateobs'][:19]}` has a `{check}` that is "
                    f"{deviation:.2f} sigma from the metric mean",
                    "validate_metadata_table()",
                )
                # TODO: Figure out how to send the image of the frame and a
                #       graph showing the tend over time of this metric along
                #       with the discrepant value.
                # TODO: Also, figure out how to add an extra column to the
                #       meta_table containing a flag for discrepant frame.
                #       It's possible that we could loop through the table
                #       somewhere else in the code, and upload the above
                #       graphs / PNGs at that time.

    # For now, just print some stats and return the table.
    print(
        f"  Mean: {np.mean(meta_table['crop_avg']):.2f}  "
        f"  Median: {np.median(meta_table['crop_med']):.2f}  "
        f"  Stddev: {np.mean(meta_table['crop_std']):.2f}"
    )

    # Add logic checks for header datatypes (edge cases)

    return meta_table


def validate_dark_table(dark_meta, junk=None):
    """validate_dark_table Analyze and validate the dark frame metadata table

    NOTE: Not yet implemented
    """
    return dark_meta


def validate_flat_table(flat_meta, flat_filter, junk=None):
    """validate_flat_table Analyze and validate the flat frame metadata table

    Separates the wheat from the chaff -- returning a subtable for the
    specified filter, or None.

    Parameters
    ----------
    flat_meta : `astropy.table.Table`
        Table containing the flat frame metadata
    flat_filter : `str`
        Flatfield filter to validate

    Returns
    -------
    `astropy.table.Table` or `None`
        If the `flat_filter` was used in this set, return the subtable of
        `flat_meta` containing that filter.  Otherwise, return `None`.
    """
    # If there were no flats at all (blank Table), return None
    if not flat_meta:
        return None

    # Find the rows of the table corresponding to this filter, return if none
    subtable = flat_meta[flat_meta["filter"] == flat_filter]
    if not subtable:
        return None

    # Make sure 'flats' have a reasonable flat countrate, or total counts
    #  in the range 1,500 - 52,000 ADU above bias.  (Can get from countrate *
    #  exptime).

    # Do something...
    print(f"==> Validating {flat_filter} in validate_flat_table():")
    print(
        f"  Mean: {np.mean(subtable['frame_avg']):.2f}  "
        f"  Median:  {np.median(subtable['frame_med']):.2f}  "
        f"  Stddev:  {np.mean(subtable['frame_std']):.2f}"
    )

    # Find the mean quadric surface for this set of flats
    # quadsurf = np.mean(np.asarray(subtable['quadsurf']), axis=0)
    # print(quadsurf)

    return subtable


def validate_skyf_table(skyf_meta, junk=None):
    """validate_dark_table Analyze and validate the sky flat frame metadata table

    NOTE: Not yet implemented
    """
    return skyf_meta
