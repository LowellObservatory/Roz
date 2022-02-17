# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Init File
"""

# Boilerplate variables
# TODO: Put this into someplace better
__author__ = 'Timothy P. Ellsworth Bowers'
__copyright__ = 'Copyright 2021'
__credits__ = ['Lowell Observatory']
__license__ = 'MPL-2.0'
__version__ = '0.1.0'
__email__ = 'tbowers@lowell.edu'
__status__ = 'Development Status :: 3 - Alpha'

# Imports for signal and log handling
import os
import sys
import signal
import warnings

from .version import version

def short_warning(message, category, filename, lineno, file=None, line=None):
    """
    Return the format for a short warning message.
    """
    return ' %s: %s (%s:%s)\n' % (category.__name__, message, os.path.split(filename)[1], lineno)

warnings.formatwarning = short_warning


# Set version
__version__ = version
