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
from .confluence_updater import update_filter_characterization
from .gather_frames import gather_cal_frames, Dumbwaiter
from .process_calibrations import (
    process_bias,
    process_flats,
    produce_database_object
)
from .send_alerts import send_alert
from .utils import set_instrument_flags


def run_lmi_cals(directory, mem_limit=None):
    """run_lmi_cals Run Roz on the LMI Calibration frames

    Collect the LMI calibration frames, produce statistics, and return a list
    of `CalibrationDatabase` objects for each binning scheme in this one
    directory.

    This is the main driver routine for LMI frames, and will call all of the
    various other modules, as needed.  As such, there should be little need
    for cross-calling between the other non-utility modules in this package.

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        The directory containing LMI frames to analyze.
    mem_limit : `float`, optional
        Memory limit for the image combination routine.  [Default: None]

    Returns
    -------
    `list` of `roz.database_manager.CalibrationDatabase`
        A list of the Calibration Database objects for each binning scheme
    """
    inst_flags = set_instrument_flags('lmi')

    # Collect the BIAS & FLAT frames for this directory
    bias_cl, flat_cl, bin_list = gather_cal_frames(directory, inst_flags)

    db_list = {}
    # Loop through the binning schemes used
    for binning in bin_list:

        # Print out a nice status message for those interested
        human_bin = binning.replace(' ','x')
        print(f"Processing the database for {human_bin} LMI binning...")

        # Process the BIAS frames to produce a reduced frame and statistics
        bias_meta, bias_frame = process_bias(bias_cl, binning=binning,
                                             mem_limit=mem_limit)

        # Process the FLAT frames to produce statistics
        flat_meta = process_flats(flat_cl, bias_frame, binning=binning)

        # Take the metadata from the BAIS and FLAT frames and produce DATABASE
        database = produce_database_object(bias_meta, flat_meta, inst_flags)
        # TODO: Find a better way to do this
        database.proc_dir = directory

        # Write the contents of the database to InfluxDB
        database.write_to_influxdb()

        # Update the LMI Filter Information page on Confluence
        #  Images for all binnings, values only for 2x2 binning
        update_filter_characterization(database, png_only=(human_bin != '2x2'))

        # Add the database to a dictionary containing the different binnings
        db_list[human_bin] = database

    # Return the list of database objects to the calling function
    return db_list


def run_deveny_cals(directory, mem_limit=None):
    """run_lmi_cals Run Roz on the DeVeny Calibration frames

    Collect the DeVeny calibration frames, produce statistics, and return a
    list of `CalibrationDatabase` objects for each binning scheme in this one
    directory.

    This is the main driver routine for DeVeny frames, and will call all of
    the various other modules, as needed.  As such, there should be little
    need for cross-calling between the other modules in this package.

    Parameters
    ----------
    directory : `str` or `pathlib.Path`
        The directory containing DeVeny frames to analyze.
    mem_limit : `float`, optional
        Memory limit for the image combination routine.  [Default: None]

    Returns
    -------
    `list` of `roz.database_manager.CalibrationDatabase`
        A list of the Calibration Database objects for each binning scheme
    """
    inst_flags = set_instrument_flags('deveny')

    # Collect the BIAS frames for this directory
    bias_cl, bin_list = gather_cal_frames(directory, inst_flags)

    db_list = {}
    # Loop through the binning schemes used
    for binning in bin_list:

        # Print out a nice status message for those interested
        human_bin = binning.replace(' ','x')
        print(f"Processing the database for {human_bin} DeVeny binning...")

        # Process the BIAS frames to produce a reduced frame and statistics
        bias_meta = process_bias(bias_cl, binning=bin_list[0],
                                 mem_limit=mem_limit, produce_combined=False)

        # Take the metadata from the BAIS frames and produce DATABASE
        database = produce_database_object(bias_meta, bias_meta, inst_flags)
        # TODO: Find a better way to do this
        database.proc_dir = directory

        # Add the database to a dictionary containing the different binnings
        db_list[human_bin] = database

    # Return the list of database objects to the calling function
    return db_list


#=============================================================================#
def main(args=None, directory=None, mem_limit=8.192e9):
    """main This is the main function.

    This function takes the directory input, determines which instrument is
    in questions, and calls the appropriate run_*_cals() function.

    In the future, if Roz is employed to analyze more than calibration frames,
    other argmuments to this function will be needed, and other driving
    routines will need to be added above.

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
    `list` of `roz.database_manager.CalibrationDatabase`
        A list of the Calibration Database objects for each binning scheme
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

    #=================================#
    # Given the directory (which will be on a remote file server in
    #  production), call the dumbwaiter to determine which files need to be
    #  copied and then carry out that operation.
    dumbwaiter = Dumbwaiter(directory)
    if dumbwaiter.empty:
        return None
    dumbwaiter.copy_frames_to_processing()
    # This really could happen at any time... putting it here for now.
    dumbwaiter.cold_storage()

    # Giddy up!
    if dumbwaiter.instrument == 'lmi':
        return run_lmi_cals(dumbwaiter.proc_dir, mem_limit=mem_limit)
    if dumbwaiter.instrument == 'deveny':
        return run_deveny_cals(dumbwaiter.proc_dir, mem_limit=mem_limit)

    send_alert('BadInstrumentAlert : main()')
    return None


if __name__ == "__main__":
    import sys
    main(sys.argv)
