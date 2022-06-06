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
from roz import database_manager
from roz import msgs


# Calibration Validation Functions ===========================================#
def validate_calibration_metadata(
    table_dict, filt_list=None, sigma_thresh=3.0, scheme="simple", **kwargs
):
    """validate_metadata_tables Analyze and validate calibration metadata tables

    For all Tables that contain information, the metadata tables and report
    dictionaries are constructed the same regardless of frametype.

    The validated metadata tables are constructed thuswise:
        [frame_type][filter][augmented_meta]

    The report dictionaries are constructed thuswise:
        [frame_type][filter][frame_collection][frame_info]

    This function forms the narrative structure of the validation with the
    heavy lifting reserved for a helper function
    `perform_calibration_validation()`.

    Parameters
    ----------
    table_dict : `dict`
        Dictionary containing the metadata tables, with keys being the
        different types of frames to consider (e.g., 'bias', 'flat', etc.)
    sigma_thresh : `float`, optional
        The sigma discrepancy threshold for flagging a frame as being
        'problematic'  [Default: 3.0]
    scheme : `str`, optional
        The validation scheme to be used  [Default: simple]

    ---- Various debugging keyword arguments (to be removed later)
    no_prob : `bool`, optional
        Only use metrics not marked as "problem" by previous validation
        [Default: True]
    all_time : `bool`, optional
        For validation of current frames, compare against all matches,
        regardless of the timestamp [Default: False]

    Returns
    -------
    validated_metadata : `dict`
        Validated metadata tables, as desciebed above
    validation_report : `dict`
        Validation report dictionary, as described above
    scheme_str : `str`
        The string to be printed in the Problem Report about the
        validation scheme
    """
    # Build the `scheme_str` to return (to be printed in the Problem Report)
    if scheme == "simple":
        scheme_str = (
            f"Statistical deviation threshold: {sigma_thresh}σ from historical values"
        )
    elif scheme == "none":
        scheme_str = "No data validation performed"
    else:
        scheme_str = "Scheme not implemented"

    # Create the blank output dict; blank report dict; frametype conversion dict
    validated_metadata = {}
    validation_report = {}
    frame_translation = {
        "bias": "bias",
        "dark": "dark",
        "flat": "dome flat",
        "skyf": "sky flat",
    }

    # Loop through tables included in the dictionary (bias, flat, etc.)
    for tabname, meta_table in table_dict.items():
        frame_type = tabname.replace("_meta", "")

        # If there is no information (blank Table), insert None into output dicts
        if not meta_table:
            validated_metadata[frame_type] = None
            validation_report[frame_type] = None
            continue

        # Build the basic [filter] steps of the dictionaries:
        # TODO: Not strictly correct, if we start to consider SKY FLATS
        frame_dict = {"filters": filt_list if frame_type == "flat" else ["DARK"]}
        frame_report = {}

        # Put data from each filter into a subdictionary
        for filt in frame_dict["filters"]:
            frame_dict[filt], frame_report[filt] = perform_calibration_validation(
                meta_table[meta_table["filter"] == filt],
                frame_translation[frame_type],
                scheme,
                filt=filt,
                sigma_thresh=sigma_thresh,
                **kwargs,
            )
        validated_metadata[frame_type] = frame_dict
        validation_report[frame_type] = frame_report

    return validated_metadata, validation_report, scheme_str


def perform_calibration_validation(
    meta_table,
    frametype,
    scheme,
    filt=None,
    sigma_thresh=3.0,
    **kwargs,
):
    """perform_validation Perform the validation on this frametype

    This function is the heart of the validation scheme, doing the actual
    comparison with historical data.  This function was pulled out separate
    to keep the calling function, validate_calibration_metadata(), cleaner
    and easier to read.

    Currently, the scheme='simple' validation consists of finding whether all
    the measured statistics fall within `sigma_thresh` sigma of the population
    mean, as pulled from the InfluxDB database.

    In the future, other validation schemes could be implemented, as desired,
    based on historical trends or multivariate relationships (e.g., the
    variation in bias level with mount temperature) or something else entirely.

    Parameters
    ----------
    meta_table : `astropy.table.Table`
        The metadata table to validate
    frametype : `str`
        Frame type (e.g., `bias`, `dome flat`, etc.)
    scheme : `str`
        The validation scheme to be used
        [NOTE: only 'simple' and 'none' are supported]
    filt : `str`, optional
        Filter used for flats [Default: None]
    sigma_thresh : `float`, optional
        The sigma discrepancy threshold for flagging a frame as being
        'problematic'  [Default: 3.0]

    ---- Various debugging keyword arguments (to be removed later)
    no_prob : `bool`, optional
        Only use metrics not marked as "problem" by previous validation
        [Default: True]
    all_time : `bool`, optional
        Get all matches, regardless of the timestamp [Default: False]

    Returns
    -------
    meta_table : `astropy.table.Table`
        The validated metadata table
    report : `dict`
        Problem report dictionary
    """
    # Parse KWARGS -- Debugging options that can be removed when in production
    no_prob = kwargs.get("no_prob", True)
    all_time = kwargs.get("all_time", False)

    # Check `scheme`
    # TODO: As additional validation schemes are developed, change this
    if scheme not in ["simple", "none"]:
        msgs.warn(
            "Only 'simple' and 'none' validation of calibration frames are "
            f"available this time.  `{scheme}` not supported.  "
            "(Using 'simple'...)"
        )
        scheme = "simple"

    # Start with a basic report dictionary
    report = {"frametype": frametype, "filter": filt}

    # If passed a blank table, return None here.
    if not meta_table:
        report["status"] = "EMPTY"
        return None, report

    # Print the banner
    msgs.validate(
        f"Doing `{scheme.upper()}` validation of {frametype.upper()}"
        f"{f' : {filt}' if filt != 'DARK' else ''} frames:"
    )

    ###################
    # NONE VALIDATION #
    ###################

    if scheme == "none":

        # Check that values for mechanical positions are sensical, if not -> NaN
        for col in [colname for colname in meta_table.colnames if "pos" in colname]:
            meta_table[col][np.where(np.abs(meta_table[col]) > 500.0)] = np.nan

        # Check that the temperature values are sensible, if not -> NaN
        for col in [colname for colname in meta_table.colnames if "temp" in colname]:
            meta_table[col][np.where(np.abs(meta_table[col]) > 40)] = np.nan

        # Add "ALL GOOD" flag columns
        meta_table["problem"] = np.zeros(len(meta_table), dtype=np.int8)
        meta_table["obstruction"] = np.zeros(len(meta_table), dtype=np.int8)

        # Package and return
        report["status"] = "GOOD"
        return meta_table, report

    #####################
    # SIMPLE VALIDATION #
    #####################

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

    # Pull the Historical Data matching this set
    hist = database_manager.HistoricalData(
        sorted(list(set(meta_table["instrument"])))[0].lower(),
        frametype,
        filter=sorted(list(set(meta_table["filter"])))[0],
        binning=sorted(list(set(meta_table["binning"])))[0],
        numamp=sorted(list(set(meta_table["numamp"])))[0],
        ampid=sorted(list(set(meta_table["ampid"])))[0],
        debug=False,
    )
    hist.perform_query(all_time=all_time)

    # Build some quick dictionaries containing the Gaussian statistics
    n_vals = {check: hist.metric_n(check, no_prob=no_prob) for check in metrics}
    mu = {check: hist.metric_mean(check, no_prob=no_prob) for check in metrics}
    sig = {check: hist.metric_stddev(check, no_prob=no_prob) for check in metrics}

    # Make empty arrays to hold 'problem' and 'obstruction' flags
    p_flag = np.zeros(len(meta_table), dtype=np.int8)
    o_flag = np.zeros(len(meta_table), dtype=np.int8)

    # Check that values for mechanical positions are sensical, if not -> NaN
    for col in [colname for colname in meta_table.colnames if "pos" in colname]:
        meta_table[col][np.where(np.abs(meta_table[col]) > 500.0)] = np.nan

    # Check that the temperature values are sensible, if not -> NaN
    for col in [colname for colname in meta_table.colnames if "temp" in colname]:
        meta_table[col][np.where(np.abs(meta_table[col]) > 40)] = np.nan

    # Loop through the frames one by one
    ftype_status = "GOOD"
    for i, row in enumerate(meta_table):
        report[(tag := f"FRAME_{row['obserno']:03d}")] = {}

        # Then, loop over the list of metrics in bias_meta that should be compared
        for check in metrics:
            # If fewer than 30 comparison frames in the DB, skip
            if n_vals[check] < 30:
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
                ftype_status = "PROBLEM"
                # Add `p_flag` for this frame
                p_flag[i] = 1
                if "pos" in check:
                    o_flag[i] = 1

    meta_table["problem"] = p_flag
    meta_table["obstruction"] = o_flag

    # Print some statistics about this frame collection
    msgs.validate(
        f"  Mean: {np.mean(meta_table['crop_avg']):.2f}  "
        f"  Median: {np.median(meta_table['crop_med']):.2f}  "
        f"  Stddev: {np.mean(meta_table['crop_std']):.2f}"
    )

    # Package and return
    report["status"] = ftype_status
    return meta_table, report


# Science Validation Functions ===============================================#
def validate_science_metadata(table_dict):
    """validate_science_metadata  Analyze and validate science metadata tables

    NOTE: Not yet implemented
          Returns whatever it's passed
    """
    return perform_science_validation(table_dict)


def perform_science_validation(meta_table):
    """perform_validation Perform the validation on this frametype

    NOTE: Not yet implemented
          Returns whatever it's passed
    """
    return meta_table


# Other Functions ============================================================#
def build_problem_report(report_dict):
    """build_problem_report Construct the Problem Report

    Parse through the report dictionary to build a string

    This is the format of the report dictionary, from `vs.perform_validation()`:
        Top-level keys: [nightname, flags, binning, frame_reports]
        Under frame_reports: [bias, flat, (etc.)]
        Under frame_type: [filters, FILTER1, ...]
        Under filter: [frametype, filter, status, [FRAME_NNN, ...]]

    Parameters
    ----------
    report_dict : `dict`
        The validation report dictionary from `vs.perform_validation()`

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

    # If no problems, return None
    if "PROBLEM" not in status_list:
        return None

    # Okay, let's build the problem report!
    report = (
        f"Problem Report for directory {report_dict['nightname']}, "
        f"binning {report_dict['binning']}\n"
        f"Site: {report_dict['flags']['site'].upper()}, "
        f"Instrument: {report_dict['flags']['instrument'].upper()}\n"
        f"{report_dict['valid_scheme']}*.*."
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
