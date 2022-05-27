# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 27-May-2022
#
#  @author: tbowers

"""Process the Science Frames for 1 Night for specified instrument

This module is part of the Roz package, written at Lowell Observatory.

This module takes the gathered science frames from a night (as collected by
roz.gather_frames) and performs basic data processing (bias & overscan
subtraction) before gathering statistics.  The statistics are then stuffed into
a database object (from roz.database_manager) for later use.

This module primarily trades in AstroPy Table objects (`astropy.table.Table`)
and CCDPROC Image File Collections (`ccdproc.ImageFileCollection`), along with
the odd AstroPy CCDData object (`astropy.nddata.CCDData`) and basic python
dictionaries (`dict`).
"""

# Built-In Libraries
# import os
# import warnings

# 3rd Party Libraries
# from astropy.stats import mad_std
# from astropy.table import Table
# from astropy.wcs import FITSFixedWarning
# import ccdproc as ccdp
# from ccdproc.utils.slices import slice_from_string as get_slice
# import numpy as np
# from tqdm import tqdm

# Internal Imports
from roz import gather_frames

# from roz import utils

# # Silence Superflous AstroPy FITS Header Warnings
# warnings.simplefilter("ignore", FITSFixedWarning)


class ScienceContainer:
    """Class for containing and processing science frames

    This container holds the gathered science frames in the processing
    directory, as well as the processing routines needed for the various types
    of frames.  The class holds the general information needed by all
    processing methods.

    Parameters
    ----------
    directory : `pathlib.Path`
        Processing directory
    inst_flags : `dict`
        Dictionary of instrument flags from utils.set_instrument_flags()
    debug : `bool`, optional
        Print debugging statements? [Default: True]
    mem_limit : `float`, optional
        Memory limit for the image combination routine [Default: 8.192e9 bytes]
    """

    def __init__(
        self,
        directory,
        inst_flag,
        debug=True,
        mem_limit=8.192e9,
    ):
        # Parse in arguments
        self.directory = directory
        self.flags = inst_flag
        self.debug = debug
        self.mem_limit = mem_limit

        # Get the frame dictionary to be used
        self.frame_dict = gather_frames.gather_other_frames(
            thing1=self.directory, thing2=self.flags
        )

    def process_science(self, ccd_bin):
        """process_science _summary_

        _extended_summary_

        Parameters
        ----------
        ccd_bin : _type_
            _description_
        """
        if self.debug:
            print(ccd_bin)

    def process_standard(self, ccd_bin):
        """process_standard _summary_

        _extended_summary_

        Parameters
        ----------
        ccd_bin : _type_
            _description_
        """
        if self.debug:
            print(ccd_bin)
