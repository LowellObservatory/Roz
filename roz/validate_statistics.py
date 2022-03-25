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


def validate_calibration_metadata(table_dict, filt_list=None, sigma_thresh=3.0):
    """validate_metadata_tables Analyze and validate calibration metadata tables

    _extended_summary_

    Parameters
    ----------
    table_dict : _type_
        _description_
    sigma_thresh : `float`, optional
        The sigma discrepancy threshold for flagging a frame as being
        'problematic'  [Default: 3.0]


    Returns
    -------
    _type_
        _description_
    """
    # Blank output dict; blank report dict; frametype conversion dict
    validated_metadata = {}
    validation_report = {}
    frametypes = {
        "bias": "bias",
        "dark": "dark",
        "flat": "dome flat",
        "skyf": "sky flat",
    }

    # Loop through tables included in the dictionary (bias, flat, etc.)
    for tabname, meta_table in table_dict.items():
        out_key = tabname.replace("_meta", "")

        # If there is no information (blank Table), insert None into output dicts
        if not meta_table:
            validated_metadata[out_key] = None
            validation_report[out_key] = None
            continue

        # For all Tables that contain information, build the report dictionary
        #  in the same fashion: [out_key][filter][framecollection][frameinfo]
        #  The metadata tables should also be [out_key][filter][augmented_meta]

        # Build the basic [filter] steps of the dictionaries:
        frame_dict = {"filters": filt_list if out_key == "flat" else ["DARK"]}
        frame_report = {}

        # Put data from each filter into a subdictionary
        for filt in frame_dict["filters"]:
            frame_dict[filt], frame_report[filt] = perform_validation(
                meta_table[meta_table["filter"] == filt],
                frametypes[out_key],
                filt=filt,
                sigma_thresh=sigma_thresh,
            )
        validated_metadata[out_key] = frame_dict
        validation_report[out_key] = frame_report

    return validated_metadata, validation_report


def perform_validation(meta_table, frametype, filt=None, sigma_thresh=3):
    """perform_validation Perform the validation on this frametype

    This function is the heart of the validation scheme, doing the actual
    comparison with historical data and

    Parameters
    ----------
    meta_table : `astropy.table.Table`
        The metadata table to validate
    frametype : `str`
        Frame type (e.g., `bias`, `dome flat`, etc.)
    filt : `str`, optional
        Filter used for flats [Default: None]
    sigma_thresh : `float`, optional
        The sigma discrepancy threshold for flagging a frame as being
        'problematic'  [Default: 3.0]

    Returns
    -------
    meta_table : `astropy.table.Table`
        The validated metadata table
    report : `dict`
        Problem report dictionary
    """
    # Start with a basic report dictionary
    report = {"frametype": frametype, "filter": filt}

    # If passed a blank table, return None here.
    if not meta_table:
        report["status"] = "EMPTY"
        return None, report

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
    fstr = f" : {filt}" if filt != "DARK" else ""
    print(
        f"==> Validating {frametype.upper()}{fstr} in validate_calibration_metadata_():"
    )
    hist = dm.HistoricalData(
        sorted(list(set(meta_table["instrument"])))[0].lower(),
        frametype,
        filter=sorted(list(set(meta_table["filter"])))[0],
        binning=sorted(list(set(meta_table["binning"])))[0],
        numamp=sorted(list(set(meta_table["numamp"])))[0],
        ampid=sorted(list(set(meta_table["ampid"])))[0],
        debug=False,
    )
    hist.perform_query()

    # Build some quick dictionaries containing the Gaussian statistics
    n_vals = {check: hist.metric_n(check) for check in metrics}
    mu = {check: hist.metric_mean(check) for check in metrics}
    sig = {check: hist.metric_stddev(check) for check in metrics}

    # Make empty arrays to hold 'problem' and 'obstruction' flags
    p_flag = np.zeros(len(meta_table), dtype=np.int8)
    o_flag = np.zeros(len(meta_table), dtype=np.int8)

    # Loop through the frames one by one
    frame_status = "GOOD"
    for i, row in enumerate(meta_table):
        report[(tag := f"FRAME_{row['obserno']:03d}")] = {}

        # Then, loop over the list of metrics in bias_meta that should be compared
        for check in metrics:
            # If fewer than 30 comparison frames in the DB, skip
            if n_vals[check] < 30:
                continue

            # Check for RC/IC positions -- if nonsensical, then set to NaN
            #   If the mean value is nonsensical, just move on
            if "pos" in check:
                if np.abs(row[check]) > 500:
                    meta_table[check][i] = np.nan
                    continue
                if np.abs(mu[check]) > 500:
                    continue

            # Greater than 3 sigma deviation, alert  [also avoid divide by zero]
            deviation = np.abs(row[check] - mu[check]) / np.max([sig[check], 1e-3])
            if deviation > sigma_thresh:
                report[tag].update(
                    {
                        "timestamp": row["dateobs"],
                        "obserno": row["obserno"],
                        check: deviation,
                    }
                )
                # Update the frame_status for the report
                frame_status = "PROBLEM"
                # Add `p_flag` for this frame
                p_flag[i] = 1
                if "pos" in check:
                    o_flag[i] = 1

    meta_table["problem"] = p_flag
    meta_table["obstruction"] = o_flag

    # For now, just print some stats and return the table.
    print(
        f"  Mean: {np.mean(meta_table['crop_avg']):.2f}  "
        f"  Median: {np.median(meta_table['crop_med']):.2f}  "
        f"  Stddev: {np.mean(meta_table['crop_std']):.2f}"
    )

    # Add logic checks for header datatypes (edge cases)
    report["status"] = frame_status
    return meta_table, report


def validate_science_metadata(table_dict):
    """validate_science_metadata  Analyze and validate science metadata tables

    NOTE: Not yet implemented
    """


def build_problem_report(report_dict, sigma_thresh=3.0):
    """build_problem_report Construct the Problem Report

    Parse through the report dictionary to build a string

    This is the format of the report dictionary, from `vs.perform_validation()`:
        Top-level keys: [nightname, flags, binning, frame_reports]
        Under frame_reports: [bias, flat, (etc.)]
        Under frame_type: [FILTER, ...]
        Under filter: [frametype, filter, status, [FRAME_NNN, ...]]

    Parameters
    ----------
    report_dict : `dict`
        The validation report dictionary from `vs.perform_validation()`
    sigma_thresh : `float`, optional
        The sigma discrepancy threshold for flagging a frame as being
        'problematic'  [Default: 3.0]

    Returns
    -------
    `str` or `None`
        If problems were found in the validation, the Problem Report is returned,
        otherwise `None`.
    """
    # First, gather info on status:
    status_list = []
    for ftype in report_dict["frame_reports"]:
        if not report_dict["frame_reports"][ftype]:
            continue
        for filt in report_dict["frame_reports"][ftype]:
            status_list.append(report_dict["frame_reports"][ftype][filt]["status"])

    # If everything is happy, return None
    if "PROBLEM" not in status_list:
        return None

    # Okay, let's build the problem report!
    report = (
        f"Problem Report for directory {report_dict['nightname']}, "
        f"binning {report_dict['binning']}\n"
        f"Site: {report_dict['flags']['site'].upper()}, "
        f"Instrument: {report_dict['flags']['instrument'].upper()}\n"
        f"Statistical deviation threshold: {sigma_thresh}σ from historical values*.*."
    )
    # Loop through frame types first:
    for ftype in report_dict["frame_reports"]:
        if not report_dict["frame_reports"][ftype]:
            continue

        # Then loop through filters:
        for filt in report_dict["frame_reports"][ftype]:
            if report_dict["frame_reports"][ftype][filt]["status"] != "PROBLEM":
                continue
            # Add information about problems:
            ff_str = f"{report_dict['frame_reports'][ftype][filt]['frametype']}"
            if ftype not in ["bias", "dark"]:
                ff_str += f" : {report_dict['frame_reports'][ftype][filt]['filter']}"
            report += f"For {ff_str.upper()} the following frames had problems:\n"

            # Find individual frame info
            for key, fdict in report_dict["frame_reports"][ftype][filt].items():
                if "FRAME_" not in key or not fdict:
                    continue
                time = fdict.pop("timestamp")
                obsn = fdict.pop("obserno")
                report += (
                    f"  File {report_dict['flags']['prefix']}.{obsn:04d}.fits, "
                    f"DATE-OBS: {time}\n    "
                )

                # Loop through discrepant items
                for i, (key2, val) in enumerate(fdict.items()):
                    if "pos" in key2:
                        report += f"Possible Obstruction: {key2}"
                    else:
                        report += f"{key2}: {val:.2f}σ  "
                    # Keep the report readable
                    if (i + 1) % 4 == 0:
                        report += "\n    "
                report += "\n"

            # Put a break between each filter
            report += "*.*."

    return report
