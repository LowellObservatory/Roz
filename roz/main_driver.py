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
"""

# Built-In Libraries
import argparse

# 3rd Party Libraries

# Internal Imports
from roz import confluence_updater as cu
from roz import database_manager as dm
from roz import gather_frames as gf
from roz import process_calibrations as pc
from roz import send_alerts as sa
from roz import utils


# The MAIN Attraction ========================================================#
def main(
    directories=None,
    do_science=False,
    sigma_thresh=3.0,
    validation_scheme="simple",
    mem_limit=8.192e9,
    **kwargs,
):
    """main This is the main function.

    This function takes the directory input, determines the instrument
    in question, and calls the appropriate run_*() function.

    In the future, if Roz is employed to analyze more than calibration frames,
    other argmuments to this function will be needed, and other driving
    routines will need to be added above.

    Parameters
    ----------
    directory : `str` or `pathlib.Path` or `list` of either, optional
        The directory or directories upon which to operate [Default: None]
    do_science : `bool`, optional
        Also do QA on science frames?  [Default: False]
    sigma_thresh : `float`, optional
        The sigma discrepancy threshold for flagging a frame as being
        'problematic'  [Default: 3.0]
    validation_scheme : `str`, optional
        The frame validation scheme to use  [Default: simple]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]

    ---- Various debugging keyword arguments (to be removed later)
    skip_cals : `bool`, optional
        Do not process the calibration frames.  [Default: False]
    no_cold : `bool`, optional
        Pass to `skip_cold` in gf.Dumbwaiter.cold_storage()  [Default: False]
    no_prob : `bool`, optional
        Only use metrics not marked as "problem" by previous validation
        [Default: True]
    all_time : `bool`, optional
        For validation of current frames, compare against all matches,
        regardless of the timestamp [Default: False]
    """
    # Check if the input `directories` is just a string; --> list
    if isinstance(directories, str):
        directories = [directories]

    # Parse KWARGS -- Debugging options that can be removed when in production
    skip_cals = kwargs.get("skip_cals", False)
    no_cold = kwargs.get("no_cold", False)
    no_prob = kwargs.get("no_prob", True)
    all_time = kwargs.get("all_time", False)

    # Loop through the directories provided
    for directory in directories:

        # Call the appropriate Dumbwaiter(s) to sort files and copy them for processing
        waiters = []
        if not skip_cals:
            waiters.append(gf.Dumbwaiter(directory, frameclass="calibration"))
        if do_science:
            waiters.append(gf.Dumbwaiter(directory, frameclass="science"))

        # Loop over appropriate Dumbwaiter(s):
        for dumbwaiter in waiters:

            # If empty, send notification and move along
            if dumbwaiter.empty:
                sa.send_alert(
                    f"Empty Directory: `{utils.subpath(dumbwaiter.dirs['data'])}` "
                    f"does not contain any sequential {dumbwaiter.frameclass} "
                    "FITS files",
                    "main_driver.main()",
                )
                continue

            # Copy over the sorted frames to processing, and package for cold storage
            dumbwaiter.copy_frames_to_processing()
            dumbwaiter.cold_storage(skip_cold=no_cold)

            # Giddy up!
            run = Run(
                dumbwaiter,
                sigma_thresh=sigma_thresh,
                validation_scheme=validation_scheme,
                mem_limit=mem_limit,
                no_prob=no_prob,
                all_time=all_time,
            )
            run.proc()


# Run Functions ==============================================================#
class Run:
    """Run Class for Running the Processing

    _extended_summary_

    Parameters
    ----------
    waiter : `roz.gather_frames.Dumbwaiter`
        The dumbwaiter holding the incoming files for processing
    sigma_thresh : `float`, optional
        The sigma discrepancy threshold for flagging a frame as being
        'problematic'  [Default: 3.0]
    validation_scheme : `str`, optional
        The frame validation scheme to use  [Default: simple]
    mem_limit : `float`, optional
        Memory limit for the image combination routine.  [Default: None]

    ---- Various debugging keyword arguments (to be removed later)
    no_prob : `bool`, optional
        Only use metrics not marked as "problem" by previous validation
        [Default: True]
    all_time : `bool`, optional
        For validation of current frames, compare against all matches,
        regardless of the timestamp [Default: False]

    """

    def __init__(
        self,
        waiter,
        sigma_thresh=3.0,
        validation_scheme="simple",
        mem_limit=None,
        **kwargs,
    ):
        # Set instance attributes
        self.waiter = waiter
        self.mem_limit = mem_limit
        self.flags = self.waiter.inst_flags
        self.sigma_thresh = sigma_thresh
        self.scheme = validation_scheme

        # Parse KWARGS -- Debugging options that can be removed when in production
        self.no_prob = kwargs.get("no_prob", True)
        self.all_time = kwargs.get("all_time", False)

    def proc(self):
        """proc Process the files specified in the Dumbwaiter

        Chooses which run_* method to call based on the `frameclass`
        """
        method = f"run_{self.waiter.frameclass[:3]}"
        if hasattr(self, method) and callable(func := getattr(self, method)):
            func()

    def run_cal(self, bin_list="1x1"):
        """run_cal Run Roz on the Instrument Calibration frames

        Collect the calibration frames for the instrument represented by this
        Dumbwaiter, process them, and collect statistics into a
        `CalibrationDatabase` object.  Upload the data to an InfluxDB database,
        analyze the frames for irregularities compared to historical data, and
        send alerts, if necessary.  If desired, also update the appropriate
        Confluence page(s) for user support.

        Keyword argument `bin_list` is a default in case `check_bin` is not
        specified in the instrument flags.
        """
        # Collect the calibration frames for the processing directory
        cframes = gf.gather_cal_frames(self.waiter.dirs["proc"], self.flags)

        # Copy over `bin_list`, if returned from the above routine
        if "bin_list" in cframes:
            bin_list = cframes["bin_list"]

        # Loop through the binning schemes used
        for binning in bin_list:

            # Print out a nice status message for those interested
            human_bin = binning.replace(" ", "x")
            print(f"\nProcessing the database for {human_bin} binning...")

            # Set default meta and combined frames to `NoneType`
            bias_meta = dark_meta = flat_meta = skyf_meta = None
            bias_frame = dark_frame = None

            # Process the BIAS frames to produce a reduced frame and statistics
            if self.flags["get_bias"]:
                bias_meta, bias_frame = pc.process_bias(
                    cframes["bias_cl"],
                    binning=binning,
                    mem_limit=self.mem_limit,
                    produce_combined=self.flags["get_flat"],
                )

            # Process the DARK frames to produce a reduced frame and statistics
            if self.flags["get_dark"]:
                dark_meta, dark_frame = pc.process_dark(
                    cframes["dark_cl"],
                    binning=binning,
                    mem_limit=self.mem_limit,
                    produce_combined=self.flags["get_flat"],
                )

            # Process the DOME (&SKY?) FLAT frames to produce statistics
            if self.flags["get_flat"]:
                flat_meta = pc.process_domeflat(
                    cframes["domeflat_cl"],
                    bias_frame=bias_frame,
                    dark_frame=dark_frame,
                    binning=binning,
                    instrument=self.flags["instrument"],
                )
                skyf_meta = pc.process_skyflat(
                    cframes["skyflat_cl"],
                    bias_frame=bias_frame,
                    dark_frame=dark_frame,
                    binning=binning,
                    instrument=self.flags["instrument"],
                )

            # Take the metadata from the calibration frames and produce DATABASE
            database = dm.CalibrationDatabase(
                self.flags,
                self.waiter.dirs["proc"],
                self.waiter.nightname,
                binning,
                bias_meta=bias_meta,
                dark_meta=dark_meta,
                flat_meta=flat_meta,
                skyf_meta=skyf_meta,
            )
            # Validate the metadata tables, and write contents to InfluxDB
            database.validate(
                sigma_thresh=self.sigma_thresh,
                no_prob=self.no_prob,
                all_time=self.all_time,
                scheme=self.scheme,
            )
            database.write_to_influxdb(testing=False)

            if self.flags["instrument"].lower() == "lmi":
                # Update the LMI Filter Information page on Confluence
                #  Images for all binnings, values only for 2x2 binning
                cu.update_filter_characterization(
                    database, png_only=(human_bin != "2x2")
                )

    def run_sci(self):
        """run_cal Run Roz on the Instrument Science frames

        _extended_summary_
        """
        sa.send_alert(
            f"Warning: `run_sci` is not yet implemented; `{self.waiter.nightname}`",
            "main_driver.Run.run_sci()",
        )


# Console Script Entry Point =================================================#
def entry_point():
    """entry_point Command-line script entry point

    Parameters
    ----------
    args : `Any`, optional
        Command-line arguments passed in [Default: None]
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
        "--nocal",
        action="store_true",
        help="Do not process the calibration frames",
    )
    parser.add_argument(
        "--sci",
        action="store_true",
        help="Process the science frames, too?  (Not yet implemented)",
    )
    pargs = parser.parse_args()

    # Giddy Up!
    main(
        pargs.directory,
        do_science=pargs.sci,
        skip_cals=pargs.nocal,
        validation_scheme=pargs.scheme.lower(),
        sigma_thresh=pargs.sigma,
        no_prob=not pargs.use_problems,
        all_time=pargs.all_time,
        no_cold=pargs.no_cold,
        mem_limit=1.024e9 * pargs.ram,
    )
