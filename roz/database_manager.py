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

Further description.
"""

# Built-In Libraries
import datetime as dt

# 3rd Party Libraries
import numpy as np

# Lowell Libraries
from ligmos import utils, workers

# Internal Imports
from .utils import LMI_FILTERS


class CalibrationDatabase():
    """CalibrationDatabase

    Database class for calibration frames

    Provides a container for the metadata from a night
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

        # Set up the internal dictionaries to hold BIAS and FLAT metadata
        self.bias = None
        self.flat = {} if self.flags['get_flats'] else None

        # Read in the InfluxDB config file
        # TODO: This needs to be accessed relative to wherever the full code
        #  is called from.
        conf_file = '/Users/tbowers/d1/codes/Roz/config/dbconfig.conf'

        # By doing it this way we ignore the 'enabled' key
        #    but we avoid contortions needed if using
        #    utils.confparsers.parseConfig, so it's worth it
        self.db_set = utils.confparsers.rawParser(conf_file)
        print(self.db_set.keys())
        self.db_set = workers.confUtils.assignConf(self.db_set['databaseSetup'],
                                                   utils.classes.baseTarget,
                                                   backfill=True)

        self.idb = utils.database.influxobj(tablename=self.db_set.tablename,
                                            host=self.db_set.host,
                                            port=self.db_set.port,
                                            user=self.db_set.user,
                                            pw=self.db_set.password,
                                            connect=True)

    @property
    def bias_temp(self):
        """bias_temp Bias level and Temperature

        [extended_summary]

        Returns
        -------
        `numpy.ndarray`, `numpy.ndarray`
            A tuple of an array of the mean bias level in the
            [100:-100,100:-100] region of the CCD along with an array of the
            corresponding mount temperature.
        """
        return np.asarray(self.bias['cen_avg']), \
               np.asarray(self.bias['mnttemp'])

    def write_to_influxdb(self):
        """write_to_influxdb Write the contents to the InfluxDB

        Following the example of Ryan's Docker_Pi/MesaTools/onewireTemps/,
        this method packetizes the bias and flat metadata dictionaries and
        commits them to the InfluxDB database, whose location and credentials
        are in ../config/dbconfig.conf
        """
        # Loop over frames in self.bias to commit each one individually
        for entry in self.bias:

            # TODO: `meas` should reflect the instrument (LMI) OR the entire
            #  self.idb object should point to a `tablename` reflecing LMI
            self._packetize_commit_influxdb(entry, self.bias.colnames, 'bias')

        # If not LMI, then bomb out now
        if self.flags['instrument'] != 'LMI':
            return

        print(f"Key names in self.flat: {self.flat.keys()}")
        # Loop through the filters, making FLAT packets and commit them
        for filt in LMI_FILTERS:
            # Skip filters not used in this data set
            print(f"Committing LMI filter {filt}...")
            if self.flat[filt] is None:
                continue

            # Loop
            for entry in self.flat[filt]:
                self._packetize_commit_influxdb(
                    entry, self.flat[filt].colnames, f"flat_{filt}")

    def _packetize_commit_influxdb(self, table_row, colnames, measure):
        """_packetize_commit_influxdb Make and commit InfluxDB Packet

        [extended_summary]

        Parameters
        ----------
        table_row : `astropy.table.Row`
            The row of data to commit to InfluxDB
        colnames : `list`
            List of the column names corresponding to this row
        measure : `str`
            The database field into which to place this row (frame)
        """
        # Convert the AstroPy Table Row into a dict by adding colnames
        row_as_dict = dict( zip(colnames, table_row) )

        # We want the database timestamp to be that of the image DATEOBS,
        #  not the current time.  Therefore, we need to create a datetime()
        #  object from the fields `utdate` and `utcstart`.
        timestamp = dt.datetime.strptime(
                        f"{table_row['utdate']}T{table_row['utcstart']}",
                        '%Y-%m-%dT%H:%M:%S.%f')

        # Create the packet for upload to the InfluxDB
        packet = utils.packetizer.makeInfluxPacket(
                 meas=[measure], ts=timestamp, fields=row_as_dict, debug=True)

        # Commit
        self.idb.singleCommit(packet, table=self.db_set.tablename)


#=============================================================================#
def main():
    """
    This is the main body function.
    """


if __name__ == "__main__":
    main()
