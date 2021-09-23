# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Analyze LMI Flat Field Frames for 1 Night

Further description.
"""

# Built-In Libraries

# 3rd Party Libraries
import ccdproc as ccdp

# Internal Imports


# List of LMI Filters
LMI_FILTERS = ['U', 'B', 'V', 'R', 'I',
               'SDSS-U', 'SDSS-G', 'SDSS-R', 'SDSS-I', 'SDSS-Z',
               'VR', 'YISH', 'OIII', 'HALPHAON', 'HALPHAOFF',
               'WR-WC', 'WR-WN', 'WR-CT',
               'UC','BC','GC','RC','C2','C3','CN','CO+','H2O+','OH','NH']


def collect_flats(dir):

    # Create an ImageFileCollection
    icl = ccdp.ImageFileCollection(dir)

    for flat_type in ['DOME FLAT', 'SKY FLAT']:
        # Grab all of the flats of this type
        flat_cl = icl.filter(obstype=flat_type)

        # If no files, move along, move along
        if not flat_cl.files:
            continue

        # Create the list of unique filters for this set
        flat_filt = sorted(list(set(list(flat_cl.summary['filters']))))




#=============================================================================#
def main():
    """
    This is the main body function.
    """
    pass


if __name__ == "__main__":
    main()
