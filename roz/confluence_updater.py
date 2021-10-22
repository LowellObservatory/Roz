# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 23-Sep-2021
#
#  @author: tbowers

"""Update the Confluence (or other) Webpage with Latest Stats

This module is part of the Roz package, written at Lowell Observatory.

This module takes the database objects produces elsewhere and prepares updated
content tables for upload to Confluence.

Confluence API Documentation:
        https://atlassian-python-api.readthedocs.io/index.html

This module primarily trades in internal databse objects
(`roz.database_manager.CalibrationDatabase`).
"""

# Built-In Libraries

# 3rd Party Libraries
from astropy.table import Table
from atlassian import Confluence
import keyring

# Internal Imports


def test_confluence():
    """test_confluence Dummy test routine

    [extended_summary]
    """
    # Instantiate the class
    confluence = setup_confluence()

    # This the page we want to muck with
    space = 'LDTOI'
    title = 'LMI Filter Characterization 2'

    # If it doesn't already exist, create it, then get the page_id
    if not confluence.page_exists(space, title):
        body = "Junk content added by python script."
        confluence.create_page(space, title, body, parent_id=55214493)
    page_id = confluence.get_page_id(space, title)

    # Add a comment!
    text = 'This comment was added via the atlassian-python-api'
    confluence.add_comment(page_id, text)

    # Make a CSV file to add to this page
    t = Table.read('/Users/tbowers/d1/codes/chicken-pi/data/coop_20211003.fits')
    t.write('coop_20211003.csv')

    # Attach the CSV file to the page
    confluence.attach_file('coop_20211003.csv', name='Coop Data',
                           content_type='application/fits', page_id=page_id,
                           comment='Test Upload')

    # Get some information
    info = confluence.get_attachments_from_content(page_id, start=0, limit=50)
    print(info)


def update_filter_characterization():
    """update_filter_characterization Update the Confluence Page

    This routine updates the Confluence page for LMI Filter Characterization
    """
    # Instantiate the class
    confluence = setup_confluence()

    # This the page we want to update -- we'll request the page_id for ease
    space = 'LDTOI'
    title = 'LMI Filter Characterization 2'

    # If it doesn't already exist, create it, then get the page_id
    if not confluence.page_exists(space, title):
        body = "New Page!!!  You must add the appropriate macros to read the attached file."
        confluence.create_page(space, title, body, parent_id=55214493)
    page_id = confluence.get_page_id(space, title)

    # Make sure the CSV file(s) we want to update are extant
    filename = 'csvtable.csv'

    # Remove the attachment on the Confluence page before uploading the new one
    confluence.delete_attachment(page_id, filename)

    # Attach the CSV file(s) to the Confluence page
    confluence.attach_file(filename, name='Something', content_type='text/csv',
                           page_id=page_id, comment='Something')


def setup_confluence():
    """setup_confluence Set up the Confluence class instance

    Uses `keyring` to hold API credentials for Confluence, including the URL.

    As such, the appropriate values for `url`, `uname`, and `passwd` need to
    be loaded into the keyring `service` = `confluence` on the machine running
    this code.

    The python package `keyring` includes a CLI for adding this information:
        `keyring set confluence [url,uname,passwd]`

    Returns
    -------
    `atlassian.Confluence`
        Confluence class, initialized with credentials
    """
    return Confluence( url=keyring.get_password('confluence', 'url'),
                       username=keyring.get_password('confluence', 'uname'),
                       password=keyring.get_password('confluence', 'passwd') )


#=============================================================================#
def main():
    """
    This is the main body function.
    """
    test_confluence()


if __name__ == "__main__":
    main()
