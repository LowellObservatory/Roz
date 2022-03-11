# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 07-Oct-2021
#
#  @author: tbowers

"""Manage the Database for LDT Instrument Calibration Frame Information

This module is part of the Roz package, written at Lowell Observatory.

This module houses the database class CalibrationDatabase().  Among its methods
are those for committing data to an InfluxDB database for posterity.  Functions
from the LIGMOS library are used for the InfluxDB pieces.

This module primarily trades in its own class.
"""

# Built-In Libraries
import datetime as dt

# 3rd Party Libraries
import numpy as np

# Lowell Libraries
from ligmos import utils as lig_utils

# Internal Imports
from roz import process_calibrations as pc
from roz import utils

# Set API Components
__all__ = ["CalibrationDatabase", "ScienceDatabase", "HistoricalData"]


class CalibrationDatabase:
    """CalibrationDatabase

    Database class for calibration frames

    Provides a container for the metadata from a night plus the methods needed
    to insert them into the InfluxDB database
    """

    def __init__(self, inst_flags, proc_dir):
        """__init__ Class initialization

        [extended_summary]

        Parameters
        ----------
        inst_flags : `dict`
            Dictionary of instrument flags.
        proc_dir : `str` or `pathlib.Path`
            Path to the processing directory
        """
        # Set internal variables
        self.flags = inst_flags
        self.proc_dir = proc_dir

        # Set up the internal dictionaries to hold BIAS and FLAT metadata
        self.bias = None
        self.flat = {} if self.flags["get_flat"] else None

        # Read in the InfluxDB config file
        self.db_set = utils.read_ligmos_conffiles("databaseSetup")

        # The InfluxDB object is thuswise constructed:
        self.idb = lig_utils.database.influxobj(
            tablename=self.db_set.tablename,
            host=self.db_set.host,
            port=self.db_set.port,
            user=self.db_set.user,
            pw=self.db_set.password,
            connect=True,
        )

    @property
    def bias_temp(self):
        """bias_temp Bias Level and Temperature

        Return the bias levels and temperatures as a @property of the instance

        Returns
        -------
        bias_crop_avg : `numpy.ndarray`
           Array of the mean bias levels in the CROP region of the CCD
        mnttemp : `numpy.ndarray`
            Array of the corresponding mount temperatures
        """
        # If the bias table is empty, return zeros
        if self.bias is None:
            return np.asarray([0]), np.asarray([0])
        return np.asarray(self.bias["crop_avg"]), np.asarray(self.bias["mnttemp"])

    def write_to_influxdb(self, testing=True):
        """write_to_influxdb Write the contents to the InfluxDB

        Following the example of Ryan's Docker_Pi/MesaTools/onewireTemps/,
        this method packetizes the bias and flat metadata dictionaries and
        commits them to the InfluxDB database, whose location and credentials
        are in config/roz.conf

        Parameters
        ----------
        testing : `bool`, optional
            If testing, don't commit to InfluxDB  [Default: True]
        """
        # If bias table is extant, loop over frames in self.bias to commit
        #  each one individually
        if self.bias is not None:
            for entry in self.bias:
                packet = neatly_package(entry, measure=self.db_set.metricname)
                # Commit
                if not testing:
                    self.idb.singleCommit(packet, table=self.db_set.tablename)

        # If not LMI, then bomb out now
        if self.flags["instrument"] != "LMI":
            return

        # Loop through the filters, making FLAT packets and commit them
        for filt in utils.LMI_FILTERS:
            # Skip filters not used in this data set
            # print(f"Committing LMI filter {filt}...")
            if self.flat[filt] is None:
                continue

            # Loop
            for entry in self.flat[filt]:
                packet = neatly_package(entry, measure=self.db_set.metricname)
                # Commit
                if not testing:
                    self.idb.singleCommit(packet, table=self.db_set.tablename)


class ScienceDatabase:
    """ScienceDatabase

    Database class for science frames

    Provides a container for the metadata from a night plus the methods needed
    to insert them into the InfluxDB database
    """

    def __init__(self):
        pass

    def write_to_influxdb(self, testing=True):
        """write_to_influxdb _summary_

        _extended_summary_

        Parameters
        ----------
        testing : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """


class HistoricalData:
    """HistoricalData _summary_

    This class pulls historical data from the InfluxDB for comparison with the
    present frames to alert for changes.

        Parameters
        ----------
        instrument : `str`
            Instrument name for which to pull (REQUIRED)
        frametype : `str`
            Frame type for which to pull (REQUIRED)
        filter : `str, optional
            Filter for which to pull [Default: None]
        binning : `str` (of form 'cxr'), optional
            Binning for which to pull [Default: None]
        numamp : `int`, optional
            Number of amplifiers for which to pull [Default: None]
        ampid : `str`, optional
            Amplifier ID for which to pull [Default: None]
        cropborder : `int`, optional
            Crop border size for which to pull [Default: None]
    """

    def __init__(
        self,
        instrument,
        frametype,
        filter=None,
        binning=None,
        numamp=None,
        ampid=None,
        cropborder=None,
    ):
        pass





# Non-Class Functions ========================================================#
def build_calibration_database(bias_meta, flat_meta, inst_flags, proc_dir):
    """produce_database_object Stuff the metadata tables into a database object

    [extended_summary]

    Parameters
    ----------
    bias_meta : `astropy.table.Table`
        Table containing the metadata and statistics for BIAS frames
    flat_meta : `astropy.table.Table`
        Table containing the metadata and statistics for FLAT frames
    inst_flags : `dict`
        Dictionary of instrument flags from .utils.set_instrument_flags()
    proc_dir : `str` or `pathlib.Path`
        Path to the processing directory

    Returns
    -------
    `roz.database_manager.CalibrationDatabase`
        Database object for use with... something?
    """
    # Instantiate the database
    database = CalibrationDatabase(inst_flags, proc_dir)

    if inst_flags["get_bias"]:
        # Analyze the bias_meta table, and insert it into the database
        database.bias = pc.validate_bias_table(bias_meta)

    if inst_flags["get_flat"]:
        # Analyze the flat_meta table, sorted by LMI_FILTERS, and insert
        for lmi_filt in utils.LMI_FILTERS:
            database.flat[lmi_filt] = pc.validate_flat_table(flat_meta, lmi_filt)

    # Return the filled database
    return database


def neatly_package(table_row, measure="Instrument_Data"):
    """neatly_package Carefully curate and package the InfluxDB packet

    This function translates the internal database into an InfluxDB object.

    Makes an InfluxDB styled packet given the measurement name, metadata tags,
    and actual fields/values to put into the database

    Parameters
    ----------
    table_row : `astropy.table.Row`
        The row of data to commit to InfluxDB
    measure : `str`, optional
        The database MEASUREMENT into which to place this row
        [Default: "Instrument_Data"]

    Returns
    -------
    `list` of `dict`
        Single-element list of packet `dict` containing the information to be
        inserted into the InfluxDB database.
    """
    # Convert the AstroPy Table Row into a dict by adding colnames
    row_as_dict = dict(zip(table_row.colnames, table_row))

    # We want the database timestamp to be that of the image DATEOBS,
    #  not the current time.  Therefore, we need to create a datetime()
    #  object from the field `dateobs`.  (NOTE: .fromisoformat() not working)
    timestamp = dt.datetime.strptime(
        f"{row_as_dict.pop('dateobs')}", "%Y-%m-%dT%H:%M:%S.%f"
    )

    # Build the tags from information in the table Row
    tags = {
        "instrument": row_as_dict.pop("instrument").lower(),
        "frametype": row_as_dict.pop("frametyp").lower(),
        "filter": row_as_dict.pop("filter"),
        "binning": row_as_dict.pop("binning"),
        "numamp": row_as_dict.pop("numamp"),
        "ampid": row_as_dict.pop("ampid"),
        "cropborder": row_as_dict.pop("cropsize"),
    }

    # Strip off the filename, as it can be reconstructed from obserno
    row_as_dict.pop("filename")

    # Build the packet as a dictionary with the proper InfluxDB keys
    packet = {
        "measurement": measure,
        "tags": tags,
        "time": timestamp,
        "fields": row_as_dict,
    }

    # InfluxDB expects a list of dicts, so return such
    return [packet]
