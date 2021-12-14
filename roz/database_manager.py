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
from .utils import read_ligmos_conffiles, LMI_FILTERS


class CalibrationDatabase():
    """CalibrationDatabase

    Database class for calibration frames

    Provides a container for the metadata from a night plus the methods needed
    to insert them into the InfluxDB database
    """

    def __init__(self, inst_flags):
        """__init__ Class initialization

        [extended_summary]

        Parameters
        ----------
        inst_flags : `dict`
            Dictionary of instrument flags.
        """
        # Set internal variables
        self.flags = inst_flags
        self.proc_dir = None

        # Set up the internal dictionaries to hold BIAS and FLAT metadata
        self.bias = None
        self.flat = {} if self.flags['get_flats'] else None

        # Read in the InfluxDB config file
        self.db_set = read_ligmos_conffiles('databaseSetup')

        # The InfluxDB object is thuswise constructed:
        self.idb = lig_utils.database.influxobj(
                    tablename=self.db_set.tablename, host=self.db_set.host,
                    port=self.db_set.port, user=self.db_set.user,
                    pw=self.db_set.password, connect=True)

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
        return np.asarray(self.bias['crop_avg']), \
               np.asarray(self.bias['mnttemp'])

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
        # Loop over frames in self.bias to commit each one individually
        for entry in self.bias:

            # TODO: `meas` should reflect the instrument (LMI) OR the entire
            #  self.idb object should point to a `tablename` reflecing LMI
            packet = neatly_package(entry, self.bias.colnames)
            # Commit
            if not testing:
                self.idb.singleCommit(packet, table=self.db_set.tablename)

        # If not LMI, then bomb out now
        if self.flags['instrument'] != 'LMI':
            return

        # Loop through the filters, making FLAT packets and commit them
        for filt in LMI_FILTERS:
            # Skip filters not used in this data set
            #print(f"Committing LMI filter {filt}...")
            if self.flat[filt] is None:
                continue

            # Loop
            for entry in self.flat[filt]:
                packet = neatly_package(entry, self.flat[filt].colnames)
                # Commit
                if not testing:
                    self.idb.singleCommit(packet, table=self.db_set.tablename)


# Non-Class Functions ========================================================#
def neatly_package(table_row, colnames, measure='Instrument_Data'):
    """neatly_package Carefully curate and package the InfluxDB packet

    This function translates the internal database into an InfluxDB object.

    Parameters
    ----------
    table_row : `astropy.table.Row`
        The row of data to commit to InfluxDB
    colnames : `list`
        List of the column names corresponding to this row
    measure : `str`, optional
        The database MEASUREMENT into which to place this row
        [Default: "Instrument_Data"]
    tags : `dict`, optional
        Tags with which to mark these fields within the measurement
        [Default: None] -- If `None`, build standard tags
    """
    # Convert the AstroPy Table Row into a dict by adding colnames
    row_as_dict = dict(zip(colnames, table_row))

    # We want the database timestamp to be that of the image DATEOBS,
    #  not the current time.  Therefore, we need to create a datetime()
    #  object from the field `dateobs`.
    timestamp = dt.datetime.strptime(f"{row_as_dict.pop('dateobs')}",
                                        '%Y-%m-%dT%H:%M:%S.%f')

    # Build the tags from information in the table Row
    tags = {'instrument': row_as_dict.pop('instrument').lower(),
            'frametype': row_as_dict.pop('frametyp').lower(),
            'filter': row_as_dict.pop('filter'),
            'binning': row_as_dict.pop('binning'),
            'numamp': row_as_dict.pop('numamp'),
            'ampid': row_as_dict.pop('ampid'),
            'cropborder': row_as_dict.pop('cropsize')}

    # Strip off the filename, as it can be reconstructed from obserno
    row_as_dict.pop('filename')

    # Create the packet for upload to the InfluxDB
    packet = lig_utils.packetizer.makeInfluxPacket(
                meas=[measure], ts=timestamp, fields=row_as_dict,
                tags=tags, debug=False)

    return packet
