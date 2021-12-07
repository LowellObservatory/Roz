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
content tables for upload to Confluence.  The only function herein that should
be called directly is update_filter_characterization().

Confluence API Documentation:
        https://atlassian-python-api.readthedocs.io/index.html

This module primarily trades in internal databse objects
(`roz.database_manager.CalibrationDatabase`).
"""

# Built-In Libraries
import datetime as dt

# 3rd Party Libraries
from astropy.io.votable import parse as vo_parse
from astropy.table import Column
from atlassian import Confluence
from bs4 import BeautifulSoup

# Lowell Libraries
from ligmos import utils as lig_utils, workers as lig_workers

# Internal Imports
from .send_alerts import send_alert, ConfluenceAlert
from .utils import (
    HTML_TABLE_FN,
    LMI_FILTERS,
    LMI_DYNTABLE,
    ROZ_CONFIG,
    ROZ_DATA,
    XML_TABLE
)


def update_filter_characterization(database, delete_existing=False):
    """update_filter_characterization Update the Confluence Page

    This routine is the main function in this module, and should be the only
    one called directly.  It updates the Confluence page for LMI Filter
    Characterization.

    Parameters
    ----------
    database : `roz.database_manager.CalibrationDatabase`
        The database of calibration frames
    delete_existing : `bool`, optional
        Delete the existing table on Confluence before upoading the new one?
        [Defualt: True]  NOTE: Once in production, maybe turn this to False?
    """
    # Instantiate the Confluence communication class and read in SPACE and TITLE
    confluence, space, title = setup_confluence()

    # If the page doesn't already exist, send alert and return
    if not confluence.page_exists(space, title):
        send_alert(ConfluenceAlert)
        return

    # Update the HTML table attached to the Confluence page
    local_filename = ROZ_DATA.joinpath(HTML_TABLE_FN)
    success = update_lmi_filter_table(local_filename, database)

    # Get the `page_id` needed for intracting with the page we want to update
    page_id = confluence.get_page_id(space, title)

    # Remove the attachment on the Confluence page before uploading the new one
    # TODO: Need to decide if this step is necessary IN PRODUCTION -- maybe no?
    if delete_existing:
        confluence.delete_attachment(page_id, HTML_TABLE_FN)

    # Attach the HTML file to the Confluence page
    confluence.attach_file(local_filename, name=HTML_TABLE_FN, page_id=page_id,
                           content_type='text/html',
                           comment='LMI Filter Information Table')


def update_lmi_filter_table(filename, database, debug=True):
    """update_lmi_filter_table Update the LMI Filter Information Table

    Updates the HTML table of LMI filter information for upload to Confluence.
    This table is partially static (basic information about the filters, etc.),
    and partially dynamic, listing the UT date of the last flatfield, and the
    most recent estimation of countrate for that filter/lamp combination.

    This table also holds (links to) PNG images of 1) a carefully curated
    nominal flatfield, and 2) the most recent flatfield in this filter.

    Parameters
    ----------
    filename : `string`
        Local filename of the HTML table to create.
    database : `roz.database_manager.CalibrationDatabase`
        The database of calibration frames
    debug : `bool`, optional
        Print debugging statements? [Default: True]

    Returns
    -------
    `int`
        Success/failure bit
    """
    # Get the base (static) table
    lmi_filt, section_head = read_lmi_static_table()

    # Use the `database` to create the dynamic portions of the LMI table
    lmi_filt = construct_lmi_dynamic_table(lmi_filt, database)

    if debug:
        # Print the table to screen for debugging
        lmi_filt.pprint()

    # Use the AstroPy Table `lmi_filt` to construct the HTML table
    construct_lmi_html_table(lmi_filt, section_head, filename, debug=debug)

    # Return value -- 0 is success!
    return 0


def construct_lmi_dynamic_table(lmi_filt, database):
    """construct_lmi_dynamic_table Construct the dynamic portions of the table

    This function augments the static table (from the XML file) with dynamic
    information contained in the `database`.

    Parameters
    ----------
    lmi_filt : `astropy.table.Table`
        The AstroPy Table representation of the static portions of the LMI
        Filter Information table
    database : `roz.database_manager.CalibrationDatabase`
        The database of calibration frames

    Returns
    -------
    `astropy.table.Table`
        The dynamically augmented LMI Filter Information Table
    """
    # Add the fun stuff!  NOTE: It's just filler for the moment...
    lastflat = []
    countrate = []
    timeto20k = []
    nominallink = []
    lastlink = []
    for i in range(len(LMI_FILTERS)):
        j = i+1
        lastflat.append('2022-01-01')
        countrate.append(j*j)
        timeto20k.append(20000/(j*j))
        nominallink.append('Click Here')
        lastlink.append('Click Here')
    lmi_filt['Nominal Image'] = nominallink
    lmi_filt['Latest Image'] = lastlink
    lmi_filt['UT Date of Last Flat'] = Column(lastflat)
    lmi_filt['Count Rate (ADU/s)'] = Column(countrate, format='.0f')
    lmi_filt['Exptime for 20k cts (s)'] = Column(timeto20k, format='.0f')

    return lmi_filt


# Utility Functions ==========================================================#
def setup_confluence():
    """setup_confluence Set up the Confluence class instance

    Reads in the confluence.conf configuration file, which contains the URL,
    username, and password.  Also contained in the configuration file are
    the Confluence space and page title into which the updated table will be
    placed.

    Returns
    -------
    confluence : `atlassian.Confluence`
        Confluence class, initialized with credentials
    space : `str`
        The Confluence space containing the LMI Filter Information page
    title : `str`
        The page title for the LMI Filter Information
    """
    # Read in and parse the configuration file
    setup = lig_utils.confparsers.rawParser(
                                  ROZ_CONFIG.joinpath('confluence.conf'))
    setup = lig_workers.confUtils.assignConf(
                                  setup['confluenceSetup'],
                                  lig_utils.classes.baseTarget,
                                  backfill=True)
    # Return
    return Confluence( url=setup.host,
                       username=setup.user,
                       password=setup.password ), \
           setup.space, setup.lmi_filter_title


def read_lmi_static_table(xml_table=XML_TABLE):
    """read_lmi_static_table Create the static portions of the LMI Filter Table

    This function reads in the XML information for the static portion of the
    LMI Filter Table, including the section headers (and locations).

    The previous version of this function had the information hard-coded,
    which would have been a pain to add new filters to.

    Parameters
    ----------
    xml_table : `str`, optional
        Path + filename of the XML file containing the LMI Filter Information
        [Default: utils.XML_TABLE]

    Returns
    -------
    filter_table : `astropy.table.Table`
        The basic portions of the AstroPy table for LMI Filter Information
    section_head : `astropy.table.Table`
        The section headings for the HTML table
    """
    # Read in the XML table.
    votable = vo_parse(xml_table)

    # The VOTable has both the LMI Filter Info and the section heads for the HTML
    filter_table = votable.get_table_by_index(0).to_table(use_names_over_ids=True)
    section_head = votable.get_table_by_index(1).to_table()

    return filter_table, section_head


def construct_lmi_html_table(lmi_filt, section_head, filename, debug=False):
    """construct_lmi_html_table Construct the HTML table

    Use the AstroPy table to construct and beautify the HTML table for the
    LMI Filter Information page.  This function takes the output of the
    dynamically created table and does fixed operations to it to make it
    nicely human-readable.

    Parameters
    ----------
    lmi_filt : `astropy.table.Table`
        The LMI Filter Information table
    section_head : `astropy.table.Table`
        The section headings for the HTML table
    filename : `str`
        The filename for the HTML table
    debug : `bool`, optional
        Print debugging statements? [Default: False]
    """
    # CSS stuff to make the HTML table pretty -- yeah, keep this hard-coded
    cssdict = {'css': 'table, td, th {\n      border: 1px solid black;\n   }\n'
                      '   table {\n      width: 100%;\n'
                      '      border-collapse: collapse;\n   }\n   '
                      'td {\n      padding: 10px;\n   }\n   th {\n'
                      '      color: white;\n      background: #6D6E70;\n   }'}
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
    if debug:
        print(f"HTML table timestamp: {timestr}")

    # Add the section headings for the different filter groups:
    for i,row in enumerate(soup.find_all('tr')):
        # At each row, search through the `section_head` table to correctly
        #  insert the appropriate section header
        for sechead in section_head:
            if i == sechead['insert_after']:
                row.insert_after(add_section_header(soup, ncols,
                                                    sechead['section'],
                                                    sechead['extra']))

    # Now that we've mucked with the HTML document, rewerite it to disk
    with open(filename, "wb") as f_output:
        f_output.write(soup.prettify("utf-8"))


def add_section_header(soup, ncols, text, extra=''):
    """add_section_header Put together the Section Headings for the HTML Table

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
