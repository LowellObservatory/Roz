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

    @property
    def bias_temp(self):
        """bias_temp Bias level and Temperature

        [extended_summary]

        Returns
        -------
        `numpy.ndarray`, `numpy.ndarray`
            A tuple of an array of the mean bias level in the [100:-100,100:-100]
            region of the CCD along with an array of the corresponding mount
            temperature.
        """
        return np.asarray(self.bias['cen_avg']), np.asarray(self.bias['mnttemp'])


#=============================================================================#
def main():
    """
    This is the main body function.
    """


if __name__ == "__main__":
    main()
