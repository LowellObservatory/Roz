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

# 3rd Party Libraries
import numpy as np

# Internal Imports
from roz import database_manager as dm
from roz import send_alerts as sa


def validate_calibration_metadata(table_dict, filt_list=None):
    """validate_metadata_tables Analyze and validate calibration metadata tables

    _extended_summary_

    Parameters
    ----------
    table_dict : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Blank output dictionary; frametype conversion dictionary
    validated_metadata = {}
    frametypes = {
        "bias": "bias",
        "dark": "dark",
        "flat": "dome flat",
        "skyf": "sky flat",
    }

    # Loop through tables included in the dictionary (bias, flat, etc.)
    for tabname, meta_table in table_dict.items():
        out_key = tabname.replace("_meta", "")

        # If there is no information (blank Table), return None
        if not meta_table:
            validated_metadata[out_key] = None
            continue

        # For DOME FLAT frames, consider individual filters separately
        if out_key == "flat":
            flat_dict = {"filters": filt_list}
            # Put data from each filter into a subdictionary
            flat_dict.update(
                {
                    filt: perform_validation(
                        meta_table[meta_table["filter"] == filt],
                        frametypes[out_key],
                        filt=filt,
                    )
                    for filt in filt_list
                }
            )
            validated_metadata[out_key] = flat_dict

        # For everything else, validate en masse
        else:
            validated_metadata[out_key] = perform_validation(
                meta_table, frametypes[out_key]
            )

    return validated_metadata


def perform_validation(meta_table, frametype, filt=None):
    """perform_validation _summary_

    _extended_summary_

    Parameters
    ----------
    meta_table : `astropy.table.Table`
        The metadata table to validate
    frametype : `str`
        Frame type (e.g., `bias`, `dome flat`, etc.)
    filt : `str`, optional
        Filter used for flats [Default: None]

    Returns
    -------
    `astropy.table.Table`
        The validated metadata table
    """
    # If passed a blank table, return None here.
    if not meta_table:
        return None

    # These are the metrics we will validate for this frametype, namely
    #   Quadric Surface, frame and crop stats, flatness statistics,
    #   and thing positions
    metrics = [
        metric
        for metric in meta_table.colnames
        if any(s in metric for s in ["qs_", "crop_", "frame_", "_flat", "pos"])
    ]
    # Remove less helpful quadric surface metrics from the validation
    for removal in ["qs_maj", "qs_bma", "qs_open", "qs_rot"]:
        metrics.remove(removal)

    # Print the banner; pull the Historical Data matching this set
    fstr = f" : {filt}" if filt else ""
    print(
        f"==> Validating {frametype.upper()}{fstr} in validate_calibration_metadata_():"
    )
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
    mu = {check: hist.metric_mean(check) for check in metrics}
    sig = {check: hist.metric_stddev(check) for check in metrics}

    # Loop through the frames one by one
    for row in meta_table:

        # TODO: Gotta find a way to consolidate the alerts to have only 1 per
        #       frame.  It gets a little rediculous, especially when there are
        #       are a whole bunch that have gone awry.

        # Then, loop over the list of metrics in bias_meta that should be compared
        for check in metrics:
            # Greater than 3 sigma deviation, alert  [also avoid divide by zero]
            deviation = np.abs(row[check] - mu[check]) / np.max([sig[check], 1e-3])
            if deviation > 3.0:
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


def validate_science_metadata(table_dict):
    """validate_science_metadata  Analyze and validate science metadata tables

    NOTE: Not yet implemented
    """
