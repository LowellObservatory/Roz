# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Manage the Database for LDT Instrument Calibration Frame Information

Further description.
"""

# Built-In Libraries

# 3rd Party Libraries

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


#=============================================================================#
def main():
    """
    This is the main body function.
    """


if __name__ == "__main__":
    main()
