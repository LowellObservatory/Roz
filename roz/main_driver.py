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

# 3rd Party Libraries

# Internal Imports
from roz import confluence_updater as cu
from roz import database_manager as dm
from roz import gather_frames as gf
from roz import process_calibrations as pc


# The MAIN Attraction ========================================================#
def main(
    directories=None,
    do_science=False,
    skip_cals=False,
    no_cold=False,
    mem_limit=8.192e9,
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
    skip_cals : `bool`, optional
        Do not process the calibration frames.  [Default: False]
    no_cold : `bool`, optional
        Pass to `testing` in gf.Dumbwaiter.cold_storage()  [Default: False]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]
    """
    # Check if the input `directories` is just a string; --> list
    if isinstance(directories, str):
        directories = [directories]

    # Loop through the directories prvided
    for directory in directories:

        # If this directory is not extant and full of FITS files, move along
        if not gf.check_directory_okay(directory, "main()"):
            continue

        # Call the appropriate Dumbwaiter(s) to sort files and copy them for processing
        waiters = []
        if not skip_cals:
            waiters.append(gf.Dumbwaiter(directory, frameclass="calibration"))
        if do_science:
            waiters.append(gf.Dumbwaiter(directory, frameclass="science"))

        # Loop over appropriate Dumbwaiter(s):
        for dumbwaiter in waiters:

            # If empty, move along
            if dumbwaiter.empty:
                continue

            # Copy over the sorted frames to processing, and package for cold storage
            dumbwaiter.copy_frames_to_processing()
            dumbwaiter.cold_storage(testing=no_cold)

            # Giddy up!
            run = Run(dumbwaiter, mem_limit=mem_limit)
            run.proc()


# Run Functions ==============================================================#
class Run:
    """Run Class for Running the Processing

    _extended_summary_

    Parameters
    ----------
    waiter : `roz.gather_frames.Dumbwaiter`
        The dumbwaiter holding the incoming files for processing
    mem_limit : `float`, optional
        Memory limit for the image combination routine.  [Default: None]
    """

    def __init__(self, waiter, mem_limit=None):
        # Set instance attributes
        self.waiter = waiter
        self.mem_limit = mem_limit
        self.flags = self.waiter.inst_flags

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
        cframes = gf.gather_cal_frames(self.waiter.proc_dir, self.flags)

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
                self.waiter.proc_dir,
                self.waiter.nightname,
                bias_meta=bias_meta,
                dark_meta=dark_meta,
                flat_meta=flat_meta,
                skyf_meta=skyf_meta,
            )
            # Validate the metadata tables, and write contents to InfluxDB
            database.validate()
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


# ============================================================================#
if __name__ == "__main__":
    # Set up the environment to import the program
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog="main_driver", description="Roz main driver")
    parser.add_argument(
        "directory",
        metavar="dir",
        type=str,
        nargs="+",
        help="The directory or directories on which to run Roz",
    )
    parser.add_argument(
        "--science",
        action="store_true",
        help="Process the science frames, too?",
    )
    parser.add_argument(
        "--nocal",
        action="store_true",
        help="Do not process the calibration frames",
    )
    parser.add_argument(
        "--gb16", action="store_true", help="Allow 16GB RAM for image combining"
    )
    args = parser.parse_args()

    # Giddy Up!
    main(
        args.directory,
        do_science=args.science,
        skip_cals=args.nocal,
        mem_limit=16.384e9 if args.gb16 else 8.192e9,
    )
