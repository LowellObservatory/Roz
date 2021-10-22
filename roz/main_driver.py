# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""The imaginatively named main driver module

Further description.
"""

# Built-In Libraries
import os

# 3rd Party Libraries

# Internal Imports
from .gather_frames import gather_cal_frames
from .process_calibrations import process_bias, process_flats, \
    produce_database_object
from .utils import set_instrument_flags


#=============================================================================#
def main(args=None, directory=None, mem_limit=8.192e9):
    """main This is the main body function

    Collect the LMI calibration frames, produce statistics, and return a
    `CalibrationDatabase` object for this one directory.

    Parameters
    ----------
    args : `list`, optional
        If this file is called from the command line, these are the command
        line arguments.  [Default: None]
    directory : `str` or `pathlib.Path`, optional
        The directory to operate on [Default: None]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]

    Returns
    -------
    `database_manager.CalibrationDatabase`
        Database object to be fed into... something?
    """
    # Parse command-line arguments, if called that way
    if args is not None:
        if len(args) == 1:
            print("ERROR: Must specify a directory to process.")
            return None

        # If not passed a directory, exit
        if not os.path.isdir(args[1]):
            print("ERROR: Must specify a directory to process.")
            return None
        directory = args[1]

    inst_flags = set_instrument_flags('lmi')

    # Collect the BIAS & FLAT frames for this directory
    bias_cl, flat_cl, bin_list = gather_cal_frames(directory, inst_flags)

    # Process the BIAS frames to produce a reduced frame and statistics
    bias_meta, bias_frame = process_bias(bias_cl, binning=bin_list[0],
                                                mem_limit=mem_limit)

    # Process the FLAT frames to produce statistics
    flat_meta = process_flats(flat_cl, bias_frame, binning=bin_list[0])

    # Take the metadata from the BAIS and FLAT frames and produce something
    database = produce_database_object(bias_meta, flat_meta, inst_flags)

    # Write to InfluxDB
    database.write_to_influxdb()

    # OR --- Could return the database to calling routine and have that call
    #  the .write_to_influxdb() method.

    return database


if __name__ == "__main__":
    import sys
    main(sys.argv)
