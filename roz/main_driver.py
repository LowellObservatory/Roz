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

This module primarily trades in... driving?
"""

# Built-In Libraries
import os

# 3rd Party Libraries

# Internal Imports
from .analyze_alert import send_alert, CantRunAlert
from .gather_frames import divine_instrument, gather_cal_frames
from .process_calibrations import (
    process_bias,
    process_flats,
    produce_database_object
)
from .utils import set_instrument_flags


def run_lmi_cals(directory, mem_limit=None):
    """run_lmi_cals Run Roz on the LMI Calibration frames

    Collect the LMI calibration frames, produce statistics, and return a
    `CalibrationDatabase` object for this one directory.

    Parameters
    ----------
    directory : `str` or `pathlib.Path`, optional
        The directory to operate on [Default: None]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]

    Returns
    -------
    [type]
        [description]
    """
    inst_flags = set_instrument_flags('lmi')

    # Collect the BIAS & FLAT frames for this directory
    bias_cl, flat_cl, bin_list = gather_cal_frames(directory, inst_flags)

    db_list = {}
    # Loop through the binning schemes used
    for binning in bin_list:

        # Print out a nice status message for those interested
        human_bin = binning.replace(' ','x')
        print(f"Processing the database for {human_bin} LMI binning.")

        # Process the BIAS frames to produce a reduced frame and statistics
        bias_meta, bias_frame = process_bias(bias_cl, binning=binning,
                                                    mem_limit=mem_limit)

        # Process the FLAT frames to produce statistics
        flat_meta = process_flats(flat_cl, bias_frame, binning=binning)

        # Take the metadata from the BAIS and FLAT frames and produce DATABASE
        database = produce_database_object(bias_meta, flat_meta, inst_flags)

        # Write the contents of the database to InfluxDB
        database.write_to_influxdb()

        # Update the LMI Filter Information page on Confluence for 2x2 binning
        if binning == '2 2':
            database.update_filter_table()

        # Add the database to a dictionary containing the different binnings
        db_list[human_bin] = database

    # Return the database object to the calling function
    return db_list


def run_deveny_cals(directory, mem_limit=None):
    """run_lmi_cals Run Roz on the LMI Calibration frames

    Collect the LMI calibration frames, produce statistics, and return a
    `CalibrationDatabase` object for this one directory.

    Parameters
    ----------
    directory : `str` or `pathlib.Path`, optional
        The directory to operate on [Default: None]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]

    Returns
    -------
    [type]
        [description]
    """
    # TODO: Deal with multiple binning schemes for DeVeny

    inst_flags = set_instrument_flags('deveny')

    # Collect the BIAS frames for this directory
    bias_cl, bin_list = gather_cal_frames(directory, inst_flags)

    # Process the BIAS frames to produce a reduced frame and statistics
    bias_meta, bias_frame = process_bias(bias_cl, binning=bin_list[0],
                                                mem_limit=mem_limit)

    return produce_database_object(bias_meta, bias_meta, inst_flags)


#=============================================================================#
def main(args=None, directory=None, mem_limit=8.192e9):
    """main This is the main body function

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
    if args:
        if len(args) == 1:
            print("ERROR: Must specify a directory to process.")
            return None

        # If not passed a directory, exit
        if not os.path.isdir(args[1]):
            print("ERROR: Must specify a directory to process.")
            return None
        directory = args[1]

    # Determine which instrument these data are for, based on FITS headers
    instrument = divine_instrument(directory)

    # Giddy up!
    if instrument == 'lmi':
        db_list = run_lmi_cals(directory, mem_limit=mem_limit)
    elif instrument == 'deveny':
        db_list = run_deveny_cals(directory, mem_limit=mem_limit)
    else:
        send_alert(CantRunAlert)
        db_list = {}

    # Return the Database
    return db_list

if __name__ == "__main__":
    import sys
    main(sys.argv)
