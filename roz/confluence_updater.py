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
import datetime as dt

# 3rd Party Libraries
from astropy.io.votable import parse
from astropy.table import Column
from atlassian import Confluence
from bs4 import BeautifulSoup
import keyring

# Internal Imports
from utils import LMI_FILTERS


def update_filter_characterization(delete_existing=True):
    """update_filter_characterization Update the Confluence Page

    This routine updates the Confluence page for LMI Filter Characterization

    Parameters
    ----------
    delete_existing : `bool`, optional
        Delete the existing table on Confluence before upoading the new one?
        [Defualt: True]  NOTE: Once in production, maybe turn this to False?
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

    # Create the updated version of the CSV table
    filename = 'lmi_filter_table.html'

    success = create_lmi_filter_table(filename)

    # Remove the attachment on the Confluence page before uploading the new one
    # TODO: Need to decide if this step is necessary IN PRODUCTION -- maybe no?
    if delete_existing:
        confluence.delete_attachment(page_id, filename)

    # Attach the HTML file to the Confluence page
    confluence.attach_file(filename, name=filename, content_type='text/html',
                           page_id=page_id, comment='HTML Table')


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


def create_lmi_filter_table(filename):
    """create_lmi_filter_table Create the LMI Filter Information Table

    Creates the HTML table of LMI filter information for upload to Confluence.
    This table is partially static (basic information about the filters, etc.),
    and partially dynamic, listing the UT date of the last flatfield, and the
    most recent estimation of countrate for that filter/lamp combination.

    This table also holds (links to) PNG images of 1) a carefully curated
    "best of" flatfield, and 2) the most recent flatfield in this filter.

    Parameters
    ----------
    filename : `string`
        Local filename of the HTML table to create.

    Returns
    -------
    `int`
        Success/failure bit
    """
    # Get the base (static) table
    lmi_filt, section_head = lmi_filter_table()

    # Add the fun stuff!  NOTE: It's just filler for the moment...
    lastflat = []
    countrate = []
    timeto20k = []
    nominallink = []
    lastlink = []
    for i in range(1,len(LMI_FILTERS)+7):
        lastflat.append('2022-01-01')
        countrate.append(i*i)
        timeto20k.append(20000/(i*i))
        nominallink.append('Click Here')
        lastlink.append('Click Here')
    lmi_filt['Nominal Image'] = nominallink
    lmi_filt['Latest Image'] = lastlink
    lmi_filt['UT Date of Last Flat'] = Column(lastflat)
    lmi_filt['Count Rate (ADU/s)'] = Column(countrate, format='.0f')
    lmi_filt['Exptime for 20k cts (s)'] = Column(timeto20k, format='.0f')

    # Print to screen for the time being... will remove later.
    lmi_filt.pprint()

    # CSS stuff to make the HTML table pretty
    cssdict = {'css': 'table, td, th {\n      border: 1px solid black;\n   }\n'
                      '   table {\n      width: 100%;\n'#      table-layout: fixed;\n'
                      '      border-collapse: collapse;\n   }\n   '
                      'td {\n      padding: 10px;\n   }\n   th {\n      color: white;\n'
                      '      background: #6D6E70;\n   }'}
    lmi_filt.write(filename, overwrite=True, htmldict=cssdict)

    # Get the number of columns for use with the table stuff below
    ncols = len(lmi_filt.colnames)

    # Now that AstroPy has done the hard work writing this table to HTML,
    #  we need to modify it a bit for visual clarity.  Use BeautifulSoup!
    with open(filename) as html:
        soup = BeautifulSoup(html, 'html.parser')

    # Add the `creation date` line to the body of the HTML above the table
    timestr = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    table_tag = soup.find('table')
    itdate = soup.new_tag('i')                # Italics, for the fun of it
    itdate.string = f"Table Auto-Generated {timestr} UTC by Roz."
    table_tag.insert_before(itdate)

    # Add the section headings for the different filter groups:
    for i,row in enumerate(soup.find_all('tr')):
        # At each row, search through the `section_head` table to correctly
        #  insert the appropriate section header
        for sechead in section_head:
            if i == sechead['insert_after']:
                row.insert_after(add_cutin_head(soup, ncols, sechead['section'],
                                                sechead['extra']))

    # Now that we've mucked with the HTML document, rewerite it to disk
    with open(filename, "wb") as f_output:
        f_output.write(soup.prettify("utf-8"))

    # Return value -- 0 is success!
    return 0


def add_cutin_head(soup, ncols, text, extra=''):
    """add_cutin_head Put together the "Cut-in Headings" for the HTML Table

    This is a bunch of BeautifulSoup tag stuff needed to make the section
    headings in the HTML table.  This function is purely a DRY block.

    Parameters
    ----------
    soup : `bs4.BeautifulSoup`
        The BeautifulSoup parsed-HTML object
    ncols : `int`
        Number of columns in the HTML table, needed for spanning
    text : `str`
        The bold/underlined text for the header
    extra : `str`, optional
        Regular text to appear after the bold/underlined text [Default: '']

    Returns
    -------
    `bs4.element.Tag`
        The newly tagged row for insertion into the HTML table
    """
    # Create the new row tag, and everything that goes inside it
    newrow = soup.new_tag('tr')
    # One column spanning the whole row, with Lowell-gray for background
    newcol = soup.new_tag('td', attrs={'colspan':ncols, 'bgcolor':'#DBDCDC'})
    # Bold/Underline the main `text` for the header; append to newcol
    bold = soup.new_tag('b')
    uline = soup.new_tag('u')
    uline.string = text
    bold.append(uline)
    newcol.append(bold)
    # Add any `extra` text in standard font after the bold/underline portion
    newcol.append(extra)
    # Put the column tag inside the row tag
    newrow.append(newcol)
    # All done
    return newrow


def lmi_filter_table(xml_table='lmi_filter_table.xml'):
    """lmi_filter_table Create the static portions of the LMI Filter Table

    This function reads in the XML information for the static portion of the
    LMI Filter Table, including the section headers (and locations).

    The previous version of this function had the information hard-coded,
    which would have been a pain to add new filters to.

    Parameters
    ----------
    xml_table : `str`, optional
        Filename of the XML file containing the LMI Filter Information

    Returns
    -------
    filter_table : `astropy.table.Table`
        The basic portions of the AstroPy table for LMI Filter Information
    section_head : `astropy.table.Table`
        The section headings for the HTML table
    """
    # Read in the XML table.
    # TODO: We need to deal with file locations once we have that structure.
    votable = parse(xml_table)

    # The VOTable has both the LMI Filter Info and the section heads for the HTML
    filter_table = votable.get_table_by_index(0).to_table(use_names_over_ids=True)
    section_head = votable.get_table_by_index(1).to_table()

    return filter_table, section_head


#=============================================================================#
def main(args):
    """
    This is the main body function.
    """
    # Main use for testing
    if len(args) == 1:
        update_filter_characterization()


if __name__ == "__main__":
    import sys
    main(sys.argv)
