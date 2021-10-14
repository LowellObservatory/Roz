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

# 3rd Party Libraries
import numpy as np

# Lowell Libraries
from ligmos import utils, workers

# Internal Imports
from .utils import LMI_FILTERS


class CalibrationDatabase():
    """CalibrationDatabase

    Database class for calibration frames
    """

    def __init__(self):

        self.bias = None

        self.flat = {}
        for lmi_filt in LMI_FILTERS:
            self.flat[lmi_filt] = None

        # Read in our config file
        conf_file = './config/dbconfig.conf'

        # By doing it this way we ignore the 'enabled' key
        #    but we avoid contortions needed if using
        #    utils.confparsers.parseConfig, so it's worth it
        self.db_set = utils.confparsers.rawParser(conf_file)
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
        commits them to the InfluxDB database, whose locations and credentials
        are in ../config/dbconfig.conf
        """
        # Make BIAS packet and commit
        bias_pkt = utils.packetizer.makeInfluxPacket(meas=['bias'],
                                                     fields=self.bias)
        self.idb.singleCommit(bias_pkt, table=self.db_set.tablename)

        # Loop through the filters, making FLAT packets and commit them
        for filt in LMI_FILTERS:
            # Skip filters not used in this data set
            if self.flat[filt] is None:
                continue
            flat_pkt = utils.packetizer.makeInfluxPacket(
                meas=[f"flat_{filt}"], fields=self.flat[filt])
            self.idb.singleCommit(flat_pkt, table=self.db_set.tablename)


#=============================================================================#
def main():
    """
    This is the main body function.
    """


if __name__ == "__main__":
    main()
