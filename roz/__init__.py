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


# Imports for signal and log handling
import os
import warnings

from .version import version


def short_warning(message, category, filename, lineno, file=None, line=None):
    """
    Return the format for a short warning message.
    """
    return f" {category.__name__}: {message} ({os.path.split(filename)[1]}:{lineno})\n"


warnings.formatwarning = short_warning


# Set version
__version__ = version
