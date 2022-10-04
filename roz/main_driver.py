# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 22-Oct-2021
#
#  @author: tbowers

"""The imaginatively named main driver module

This module is part of the Roz package, written at Lowell Observatory.

This module sits in the driver's seat... literally.  It will be here that the
manual and automatic interfaces to the Roz code attach.

Furthermore, as the main driver module, it calls all of the various other
modules, as needed.  There should, therefore, be little need for cross-calling
between the other non-utility modules in this package.

This module primarily trades in... driving?

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

# Built-In Libraries
import argparse
import sys

# 3rd Party Libraries

# Internal Imports
from roz import alerting
from roz import lmi_confluence_table
from roz import database_manager
from roz import gather_frames
from roz import msgs
from roz import process_frames


# The MAIN Attraction ========================================================#
def main(
    directories=None,
    do_science=False,
    sigma_thresh=3.0,
    validation_scheme="simple",
    mem_limit=8.192e9,
    **kwargs,
):
    """This is the main function.

    This function takes the directory input, determines the instrument
    in question, and calls the appropriate ``run_*()`` function.

    In the future, if Roz is employed to analyze more than calibration frames,
    other argmuments to this function will be needed, and other driving
    routines will need to be added above.

    Parameters
    ----------
    directory : :obj:`str` or :obj:`pathlib.Path` or :obj:`list`, optional
        The directory or directories upon which to operate (Default: None)
    do_science : bool, optional
        Also do QA on science frames?  (Default: False)
    sigma_thresh : float, optional
        The sigma discrepancy threshold for flagging a frame as being
        "problematic"  (Default: 3.0)
    validation_scheme : str, optional
        The frame validation scheme to use  (Default: "simple")
    mem_limit : float, optional
        Memory limit for the image combination routine (Default: 8.192e9 bytes)
    skip_cals : bool, optional
        DEBUG KWARG OPTION.  Do not process the calibration frames. (Default:
        False)
    no_cold : bool, optional
        DEBUG KWARG OPTION.  Pass to ``skip_cold`` in
        :func:`~roz.gather_frames.Dumbwaiter.cold_storage`  (Default: False)
    no_prob : bool, optional
        DEBUG KWARG OPTION.  Only use metrics not marked as "problem" by
        previous validation  (Default: True)
    all_time : bool, optional
        DEBUG KWARG OPTION.  For validation of current frames, compare against
        all matches, regardless of the timestamp  (Default: False)
    """
    # Parse input processing arguments into a dict for Dumbwaiter
    proc_args = {
        "calibration": not kwargs.get("skip_cals", False),
        "science": do_science,
        "allsky": True,
    }

    # Check if the input `directories` is just a string; --> list
    if not isinstance(directories, list):
        directories = [directories]

    # Loop through the directories provided
    for directory in directories:

        # Give some visual space between directories being processed
        print(f"\n{'-'*30}\n")

        # Load this directory into the dumbwaiter
        dumbwaiter = gather_frames.Dumbwaiter(directory, proc_args)

        # Loop over the valid frameclasses
        for frameclass in gather_frames.FRAMECLASSES:

            # Check if this frameclass is to be processed
            if frameclass not in dumbwaiter.process_frameclass:
                continue

            # If empty, send notification and move along
            if dumbwaiter.empty(frameclass):
                alerting.send_alert(
                    "empty_dir", dumbwaiter=dumbwaiter, frameclass=frameclass, **kwargs
                )
                continue

            # Copy over the sorted frames to processing; package for cold storage
            dumbwaiter.serve_frames(frameclass)
            dumbwaiter.cold_storage(frameclass, **kwargs)

            # Call the appropriate `run_*` function for this frameclass
            globals()[f"run_{frameclass}"](
                dumbwaiter,
                sigma_thresh=sigma_thresh,
                validation_scheme=validation_scheme,
                mem_limit=mem_limit,
                **kwargs,
            )


# Run Functions ==========================================================#
def run_calibration(
    dumbwaiter: gather_frames.Dumbwaiter,
    sigma_thresh=3.0,
    validation_scheme="simple",
    mem_limit=None,
    **kwargs,
):
    """Run Roz on the Instrument Calibration frames

    Collect the calibration frames for the instrument represented by this
    :class:`~roz.gather_frames.Dumbwaiter`, process them, and collect
    statistics into a :class:`~roz.database_manager.CalibrationDatabase`
    object.  Upload the data to an InfluxDB database, analyze the frames
    for irregularities compared to historical data, and send alerts, if
    necessary.  If desired, also update the appropriate Confluence page(s)
    for user support.

    Parameters
    ----------
    waiter : :obj:`~roz.gather_frames.Dumbwaiter`
        The dumbwaiter holding the incoming files for processing
    sigma_thresh : float, optional
        The sigma discrepancy threshold for flagging a frame as being
        "problematic"  (Default: 3.0)
    validation_scheme : str, optional
        The frame validation scheme to use  (Default: "simple")
    mem_limit : float, optional
        Memory limit for the image combination routine.  (Default: None)
    no_prob : bool, optional
        DEBUG KWARG OPTION.  Only use metrics not marked as "problem" by
        previous validation  (Default: True)
    all_time : bool, optional
        DEBUG KWARG OPTION.  For validation of current frames, compare against
        all matches, regardless of the timestamp  (Default: False)
    """
    # Collect the calibration frames within the processing directory
    calibs = process_frames.CalibContainer(
        dumbwaiter.dirs["proc"],
        dumbwaiter.flags,
        mem_limit=mem_limit,
    )

    # Loop through the CCD configuration schemes used
    for config in calibs.unique_detector_configs:
        ccd_bin, amp_id = config

        # Print out a nice status message for those interested
        print("")
        msgs.info(
            f"Processing {dumbwaiter.nightname} for {ccd_bin.replace(' ', 'x')} "
            f"binning, amplifier{'s' if len(amp_id)>1 else ''} {amp_id}..."
        )

        # Reset the metadata tables and calib frames for this configuration
        calibs.reset_config()

        # Process the BIAS frames to produce a reduced frame and statistics
        if dumbwaiter.flags["get_bias"]:
            calibs.process_bias(config)

        # Process the DARK frames to produce a reduced frame and statistics
        if dumbwaiter.flags["get_dark"]:
            calibs.process_dark(config)

        # Process the DOME (& SKY?) FLAT frames to produce statistics
        if dumbwaiter.flags["get_flat"]:
            calibs.process_domeflat(config)
            calibs.process_skyflat(config)

        # Take the metadata from the calibration frames and produce DATABASE
        database = database_manager.CalibrationDatabase(
            dumbwaiter.flags,
            dumbwaiter.dirs["proc"],
            dumbwaiter.nightname,
            config,
            calib_container=calibs,
        )
        # Validate the metadata tables
        database.validate(
            sigma_thresh=sigma_thresh,
            scheme=validation_scheme,
            **kwargs,
        )
        # Write the contents to InfluxDB
        database.write_to_influxdb(testing=kwargs.get("skip_db_write", False))

        # Update the LMI Filter Information page on Confluence if single-amplifier
        if (
            dumbwaiter.flags["instrument"].lower() == "lmi"
            and len(amp_id) == 1
            and not kwargs.get("no_confluence", False)
        ):
            # Images for all binnings, values only for 2x2 binning
            lmi_confluence_table.update_filter_characterization(
                database, png_only=(ccd_bin != "2 2"), delete_existing=False
            )


def run_science(
    dumbwaiter: gather_frames.Dumbwaiter,
    sigma_thresh=3.0,
    validation_scheme="simple",
    mem_limit=None,
    **kwargs,
):

    """Run Roz on the Instrument Science frames

    Parameters
    ----------
    waiter : :obj:`~roz.gather_frames.Dumbwaiter`
        The dumbwaiter holding the incoming files for processing
    sigma_thresh : float, optional
        The sigma discrepancy threshold for flagging a frame as being
        "problematic"  (Default: 3.0)
    validation_scheme : str, optional
        The frame validation scheme to use  (Default: "simple")
    mem_limit : float, optional
        Memory limit for the image combination routine.  (Default: None)
    no_prob : bool, optional
        DEBUG KWARG OPTION.  Only use metrics not marked as "problem" by
        previous validation  (Default: True)
    all_time : bool, optional
        DEBUG KWARG OPTION.  For validation of current frames, compare against
        all matches, regardless of the timestamp  (Default: False)
    """
    alerting.send_alert("not_implemented", dumbwaiter=dumbwaiter, **kwargs)


def run_allsky(
    dumbwaiter: gather_frames.Dumbwaiter,
    sigma_thresh=3.0,
    validation_scheme="simple",
    mem_limit=None,
    **kwargs,
):
    """Run Roz on the All-Sky Camera frames

    Parameters
    ----------
    waiter : :obj:`~roz.gather_frames.Dumbwaiter`
        The dumbwaiter holding the incoming files for processing
    sigma_thresh : float, optional
        The sigma discrepancy threshold for flagging a frame as being
        "problematic"  (Default: 3.0)
    validation_scheme : str, optional
        The frame validation scheme to use  (Default: "simple")
    mem_limit : float, optional
        Memory limit for the image combination routine.  (Default: None)
    no_prob : bool, optional
        DEBUG KWARG OPTION.  Only use metrics not marked as "problem" by
        previous validation  (Default: True)
    all_time : bool, optional
        DEBUG KWARG OPTION.  For validation of current frames, compare against
        all matches, regardless of the timestamp  (Default: False)
    """
    # Collect the calibration frames within the processing directory
    allskies = process_frames.AllSkyContainer(
        dumbwaiter.dirs["proc"],
        dumbwaiter.flags,
        mem_limit=mem_limit,
    )

    msgs.bug("Acquired AllSkyContainer!")

    msgs.bug(
        f"These are the unique_detector_configs: {allskies.unique_detector_configs}"
    )

    # Loop through the CCD configuration schemes used
    for config in allskies.unique_detector_configs:
        ccd_bin, amp_id = config

        msgs.bug("This should print.  Yes?  Yes?")

        # Print out a nice status message for those interested
        print("")
        msgs.info(
            f"Processing {dumbwaiter.nightname} for {ccd_bin.replace(' ', 'x')} "
            f"binning, amplifier{'s' if len(amp_id)>1 else ''} {amp_id}..."
        )

        # Reset the metadata tables for this config
        allskies.reset_config()

        # Process the ALLSKY frames!
        allskies.process_allsky(config)

        # Take the metadata from the calibration frames and produce DATABASE
        database = database_manager.AllSkyDatabase(
            dumbwaiter.flags,
            dumbwaiter.dirs["proc"],
            dumbwaiter.nightname,
            config,
            allsky_container=allskies,
        )
        # Validate the metadata tables
        database.validate(
            sigma_thresh=sigma_thresh,
            scheme=validation_scheme,
            **kwargs,
        )
        # Write the contents to InfluxDB
        database.write_to_influxdb(testing=kwargs.get("skip_db_write", False))


# Console Script Entry Point =================================================#
def entry_point():
    """Command-line script entry point

    Parameters
    ----------
    args : Any, optional
        Command-line arguments passed in (Default: None)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="roz",
        description="Lowell Observatory quality assurance of instrument data",
    )
    parser.add_argument(
        "directory",
        metavar="dir",
        type=str,
        nargs="+",
        help="The directory or directories on which to run Roz",
    )
    parser.add_argument(
        "-a",
        "--all_time",
        action="store_true",
        help="Use all historical data, regardless of timestamp (disregard conf file)",
    )
    parser.add_argument(
        "-r",
        "--ram",
        type=int,
        default=8,
        help="Gigabytes of RAM to use for image combining (default: 8)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="SIMPLE",
        help="Validation scheme to use [*SIMPLE*, NONE]",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help=(
            "Sigma threshold for reporting problematic frames for SIMPLE "
            "validation (default: 3.0)"
        ),
    )
    parser.add_argument(
        "--no_cold", action="store_true", help="Do not copy to cold storage"
    )
    parser.add_argument(
        "--use_problems",
        action="store_true",
        help="Use historical data marked as problem in the analysis",
    )
    parser.add_argument(
        "--skip_db",
        action="store_true",
        help="Skip writing to the InfluxDB",
    )
    parser.add_argument(
        "--no_confluence", action="store_true", help="Do not update Confluence"
    )
    parser.add_argument(
        "--nocal",
        action="store_true",
        help="Do not process the calibration frames",
    )
    parser.add_argument(
        "--silence_empty_alerts",
        action="store_true",
        help="Do not send Slack alerts on empty directories",
    )
    parser.add_argument(
        "--sci",
        action="store_true",
        help="Process the science frames, too?  (Not yet implemented)",
    )
    pargs = parser.parse_args()

    # Giddy Up!
    sys.exit(
        main(
            pargs.directory,
            do_science=pargs.sci,
            skip_cals=pargs.nocal,
            validation_scheme=pargs.scheme.lower(),
            sigma_thresh=pargs.sigma,
            no_prob=not pargs.use_problems,
            all_time=pargs.all_time,
            no_cold=pargs.no_cold,
            no_confluence=pargs.no_confluence,
            skip_db_write=pargs.skip_db,
            no_slack_empty=pargs.silence_empty_alerts,
            mem_limit=1.024e9 * pargs.ram,
        )
    )
