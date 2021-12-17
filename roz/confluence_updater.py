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
import os
from time import sleep

# 3rd Party Libraries
from astropy.io.votable import parse as vo_parse
from astropy.table import join, Column, Table
from atlassian import Confluence
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import numpy as np
from numpy.ma.core import MaskedConstant

# Internal Imports
from .graphics_maker import make_png_thumbnail
from .send_alerts import send_alert
from .utils import (
    read_ligmos_conffiles,
    table_sort_on_list,
    two_sigfig,
    HTML_TABLE_FN,
    ECSV_FILTERS,
    ECSV_SECHEAD,
    LMI_FILTERS,
    LMI_DYNTABLE,
    ROZ_DATA,
    ROZ_THUMB,
    XML_TABLE
)


# Outward-facing function ====================================================#
def update_filter_characterization(database, png_only=False,
                                   delete_existing=False):
    """update_filter_characterization Update the Confluence Page

    This routine is the main function in this module, and should be the only
    one called directly.  It updates the Confluence page for LMI Filter
    Characterization.

    Parameters
    ----------
    database : `roz.database_manager.CalibrationDatabase`
        The database of calibration frames
    png_only : `bool`, optional
        Only update the PNG image and not the countrate/exptime columns
        [Default: False]
    delete_existing : `bool`, optional
        Delete the existing table on Confluence before upoading the new one?
        [Defualt: True]  NOTE: Once in production, maybe turn this to False?
    """
    # Instantiate the Confluence communication class; read in SPACE and TITLE
    confluence, space, title = setup_confluence()

    # If the page doesn't already exist, send alert and return
    if not safe_confluence_connect(confluence.page_exists, space, title):
        send_alert('ConfluenceAlert : update_filter_characterization()')
        return

    # Get the `page_id` needed for intracting with the page we want to update
    page_id = safe_confluence_connect(confluence.get_page_id, space, title)
    print(f"This is the page_id: {page_id}")

    # Update the HTML table attached to the Confluence page
    local_filename = ROZ_DATA.joinpath(HTML_TABLE_FN)
    attachment_url = f"{confluence.url}download/attachments/{page_id}/"
    png_fn = update_lmi_filter_table(local_filename, database, attachment_url,
                                     png_only=png_only)

    # Remove the attachment on the Confluence page before uploading the new one
    # TODO: Need to decide if this step is necessary IN PRODUCTION -- maybe no?
    if delete_existing:
        safe_confluence_connect(confluence.delete_attachment,
                                page_id, HTML_TABLE_FN)

    # Attach the HTML file to the Confluence page
    safe_confluence_connect(confluence.attach_file, local_filename,
                            name=HTML_TABLE_FN, page_id=page_id,
                            content_type='text/html',
                            comment='LMI Filter Information Table')

    # Attach any PNGs created
    for png in png_fn:
        safe_confluence_connect(confluence.attach_file,
                                ROZ_THUMB.joinpath(png), name=png,
                                page_id=page_id, content_type='image/png',
                                comment='Flat Field Image')


# Descriptive, high-level functions ==========================================#
def update_lmi_filter_table(filename, database, attachment_url,
                            png_only=False, debug=False):
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
    attachment_url : `str`
        The URL for attachments for this page in Confluence.
    png_only : `bool`, optional
        Only update the PNG image and not the countrate/exptime columns
        [Default: False]
    debug : `bool`, optional
        Print debugging statements? [Default: True]

    Returns
    -------
    `list`
        List of PNG filenames created for this run.
    """
    # Get the base (static) table
    lmi_filt, section_head = read_lmi_static_table()

    # Use the `database` to modify the dynamic portions of the LMI table
    lmi_filt, png_fn = modify_lmi_dynamic_table(lmi_filt, database,
                                                attachment_url,
                                                png_only=png_only)
    if debug:
        lmi_filt.pprint()

    # Use the AstroPy Table `lmi_filt` to construct the HTML table
    construct_lmi_html_table(lmi_filt, section_head, filename,
                             link_text="Image Link", debug=True)

    # Return list of PNG filenames
    return png_fn


def modify_lmi_dynamic_table(lmi_filt, database, attachment_url,
                             png_only=False, debug=False):
    """modify_lmi_dynamic_table Modify the dynamic portions of the table

    This function augments the static table (from the XML file) with dynamic
    information contained in the `database`.

    It should be noted that the Count Rate and Exptime columns are generated
    solely from the last night's flats.  So, if something funny was going on
    for those frames, the quoted values for these columns will be incorrect
    until closer-to-nominal flats are collected again.

    The use of `join(join_type=left)` means that the HTML table produced and
    the saved `dyntable` will always have the same rows at the current
    lmi_filter_table.[xml,ecsv] file.

    Parameters
    ----------
    lmi_filt : `astropy.table.Table`
        The AstroPy Table representation of the static portions of the LMI
        Filter Information table
    database : `roz.database_manager.CalibrationDatabase`
        The database of calibration frames
    attachment_url : `str`
        The URL for attachments for this page in Confluence.
    debug : `bool`, optional
        Print debugging statements? [Default: True]

    Returns
    -------
    lmi_filt : `astropy.table.Table`
        The dynamically augmented LMI Filter Information Table
    png_fn : `list`
        List of the PNG filenames created during this run
    """
    # Check if the dynamic FITS table is extant
    if os.path.isfile(LMI_DYNTABLE):
        # Read it in!
        dyntable = Table.read(LMI_DYNTABLE)
    else:
        # Make a blank table, including the LMI_FILTERS for correspondence
        nrow = len(LMI_FILTERS)
        col0 = Column(LMI_FILTERS, name='Filter')
        col1 = Column(name='Latest Image', length=nrow, dtype='U256')
        col2 = Column(name='UT Date of Latest Flat', length=nrow, dtype='U128')
        col3 = Column(name='Count Rate (ADU/s)', length=nrow, dtype=float)
        col4 = Column(name='Exptime for 20k cts (s)', length=nrow, dtype=float)
        dyntable = Table([col0, col1, col2, col3, col4])

    # The astropy.table function join() combines tables based on common keys,
    #  however, it also sorts the table...
    lmi_filt = join(lmi_filt, dyntable, join_type='left', keys='Filter')
    # Undo the alpha sorting done by .join()
    lmi_filt = table_sort_on_list(lmi_filt, 'FITS Header Value', LMI_FILTERS)
    # Make sure the `Latest Image` column has enough space for long URLs
    lmi_filt['Latest Image'] = lmi_filt['Latest Image'].astype('U256')

    if debug:
        lmi_filt.pprint()

    # Loop through the filters, updating the relevant columns of the table
    png_fn = []
    for i,filt in enumerate(LMI_FILTERS):
        # Skip filters not used in this data set
        if database.flat[filt] is None:
            continue

        # But, only update if the DATOBS of this flat is LATER than what's
        #  already in the table.  If OLD > NEW, skip.
        new_date = database.flat[filt]['dateobs'][-1].split('T')[0]
        if not isinstance(lmi_filt['UT Date of Latest Flat'][i], MaskedConstant):
            if (existing_date := lmi_filt['UT Date of Latest Flat'][i].strip()) :
                if dt.datetime.strptime(existing_date, "%Y-%m-%d") >= \
                                    dt.datetime.strptime(new_date, "%Y-%m-%d") :
                    continue

        # TODO: Add a check here for whether the correct lamps were used.  This
        #       can be determined by checking that the count rate is within
        #       some nominal range.

        # Call the PNG-maker to PNG-ify the latest image; record PNG's filename
        fname = database.proc_dir.joinpath(database.flat[filt]['filename'][-1])
        png_fn.append( make_png_thumbnail(fname, database.flags) )

        # Update the dynamic columns
        lmi_filt['Latest Image'][i] = f"{attachment_url}{png_fn[-1]}?api=v2"
        lmi_filt['UT Date of Latest Flat'][i] = new_date
        if not png_only:
            lmi_filt['Count Rate (ADU/s)'][i] = \
                        (count_rate := np.mean(database.flat[filt]['crop_med']))
            lmi_filt['Exptime for 20k cts (s)'][i] = 20000. / count_rate

    # Split off the dyntable portion again, and write it back to disk for later
    dyntable = Table( [ lmi_filt['Filter'], lmi_filt['Latest Image'],
                        lmi_filt['UT Date of Latest Flat'],
                        lmi_filt['Count Rate (ADU/s)'],
                        lmi_filt['Exptime for 20k cts (s)'] ],
                        names=['Filter','Latest Image','UT Date of Latest Flat',
                               'Count Rate (ADU/s)', 'Exptime for 20k cts (s)'] )
    dyntable.write(LMI_DYNTABLE, overwrite=True)

    # Add formatting constraints to the `lmi_filt` table columns
    lmi_filt['Count Rate (ADU/s)'] = \
                Column(lmi_filt['Count Rate (ADU/s)'].filled(0),
                       format=two_sigfig)
    lmi_filt['Exptime for 20k cts (s)'] = \
                Column(lmi_filt['Exptime for 20k cts (s)'].filled(0),
                       format=two_sigfig)

    return lmi_filt, png_fn


# Utility Functions (Alphabetical) ===========================================#
def add_html_section_header(soup, ncols, text, extra=''):
    """add_html_section_header Construct Section Headings for the HTML Table

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
    newcol.append('' if isinstance(extra, MaskedConstant) else extra)
    # Put the column tag inside the row tag
    newrow.append(newcol)
    # All done
    return newrow


def construct_lmi_html_table(lmi_filt, section_head, filename,
                             link_text='Click Here', debug=False):
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
    link_text : `str`, optional
        What the link text should say in the HTML document for PNG URLs.
        [Default: 'Click Here']
    debug : `bool`, optional
        Print debugging statements? [Default: False]
    """
    # Count the number of columns for use with the HTML table stuff below
    ncols = len(lmi_filt.colnames)

    # CSS stuff to make the HTML table pretty -- yeah, keep this hard-coded
    cssdict = {'css': 'table, td, th {\n      border: 1px solid black;\n   }\n'
                      '   table {\n      width: 100%;\n'
                      '      border-collapse: collapse;\n   }\n   '
                      'td {\n      padding: 10px;\n   }\n   th {\n'
                      '      color: white;\n      background: #6D6E70;\n   }'}

    # Use the AstroPy HTML functionality to get us most of the way there
    lmi_filt.write(filename, overwrite=True, htmldict=cssdict)

    # Now that AstroPy has done the hard work writing this table to HTML,
    #  we need to modify it a bit for visual clarity.  Use BeautifulSoup!
    with open(filename) as html:
        soup = BeautifulSoup(html, 'html.parser')

    # Add the `creation date` line to the body of the HTML above the table
    timestr = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    itdate = soup.new_tag('i')                # Italics, for the fun of it
    itdate.string = f"Table Auto-Generated {timestr} UTC by Roz."
    # Place the italicized date string ahead of the table
    soup.find('table').insert_before(itdate)
    if debug:
        print(f"HTML table timestamp: {timestr}")

    # Add the section headings for the different filter groups:
    for i,row in enumerate(soup.find_all('tr')):
        # At each row, search through the `section_head` table to correctly
        #  insert the appropriate section header
        for sechead in section_head:
            if i == sechead['insert_after']:
                row.insert_after(add_html_section_header(soup, ncols,
                                                         sechead['section'],
                                                         sechead['extra']))

    # Convert the bare URLs of all PNG thumbnails into hyperlinks.  Use a
    #  recursive function to navigate the HTML BeautifulSoup tree.
    wrap_plaintext_links(soup, soup, link_text=link_text)

    # Now that we've mucked with the HTML document, rewerite it to disk
    with open(filename, "wb") as f_output:
        f_output.write(soup.prettify("utf-8"))


def read_lmi_static_table(table_type='ecsv'):
    """read_lmi_static_table Create the static portions of the LMI Filter Table

    This function reads in the information for the static portion of the
    LMI Filter Table, including the section headers (and locations).

    At present, there are two representations of these data: XML VOTABLE format
    and YAML-based ECSV (Astropy Table).

    Parameters
    ----------
    table_type : `str`, optional
        Type of table containing the data to be read in.  Choices are `ecsv`
        for the ECSV YAML-based AstroPy Table version, or `xml` for the
        XML-based VOTABLE protocol.  [Default: ecsv]

    Returns
    -------
    filter_table : `astropy.table.Table`
        The basic portions of the AstroPy table for LMI Filter Information
    section_head : `astropy.table.Table`
        The section headings for the HTML table

    Raises
    ------
    ValueError
        Raised if improper `table_type` passed to the function.
    """
    if table_type == 'xml':
        # Read in the XML table.
        votable = vo_parse(XML_TABLE)

        # The VOTable has both the LMI Filter Info and the section heads for the HTML
        filter_table = votable.get_table_by_index(0).to_table(use_names_over_ids=True)
        section_head = votable.get_table_by_index(1).to_table()

    elif table_type == 'ecsv':
        # Read in the ECSV tables (LMI Filter Info and the HTML section headings)
        filter_table = Table.read(ECSV_FILTERS)
        section_head = Table.read(ECSV_SECHEAD)

    else:
        raise ValueError(f"Table type {table_type} not recognized!")

    return filter_table, section_head


def safe_confluence_connect(func, *args, **kwargs):
    """safe_confluence_connect Safely connect to Confluence (error-catching)

    Wrapper for confluence-connection functions to catch errors that might be
    kicked (ConnectionTimeout, for instance).

    This function performs a semi-infinite loop, pausing for 5 seconds after
    each failed function call, up to a maximum of 5 minutes.

    Parameters
    ----------
    func : `method`
        The Confluence class method to be wrapped

    Returns
    -------
    `Any`
        The return value of `func` -- or None if unable to run `func`
    """
    # Starting value, pause (in seconds), and total timeout (in minutes)
    i, pause, timeout = 1, 5, 5

    while True:
        try:
            # Nominal function return
            return func(*args, **kwargs)
        except Exception as exception:
            # If any fail, notify, pause, and retry
            # TODO: Maybe limit the scope of `Exception` to urllib3/request?
            print(f"\nExecution of `{func.__name__}` failed because of "
                  f"{exception.__context__}\nWaiting {pause} seconds "
                  f"before starting attempt #{(i := i+1)}")
            sleep(pause)
        # Give up after `timeout` minutes...
        if i >= int(timeout*60/pause):
            break
    return None


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
    setup = read_ligmos_conffiles('confluenceSetup')

    # Return
    return Confluence( url=setup.host,
                       username=setup.user,
                       password=setup.password ), \
           setup.space, setup.lmi_filter_title


def wrap_plaintext_links(bs_tag, soup, link_text='Click Here'):
    """wrap_plaintext_links Wrap bare URLs into hyperlinks

    Finds all elements in the parsed HTML file that are bare URLs, and
    converts them into hyperlinks with some bland link text like "Clik Here".

    Shamelessly stolen from https://stackoverflow.com/questions/33364955/

    Parameters
    ----------
    bs_tag : `bs4.BeautifulSoup` or `bs4.element.Tag`
        A BeautifulSoup object.
    soup : `bs4.BeautifulSoup`
        The top-level BeautifulSoup object, needed for the recursive execution.
    link_text : `str`, optional
        What the link text should say in the HTML document.
        [Default: 'Click Here']
    """
    # The try/except catches bs_tag items that don't have children
    try:
        for element in bs_tag.children:
            if isinstance(element, NavigableString) and \
                element.string[:4] == 'http':
                # If this is a string that starts with 'http', linkify!
                link = soup.new_tag('a', href=element.string)
                link.string = link_text
                element.replace_with(link)
            elif element.name != "a":
                # Recursive execution to traverse the tree
                wrap_plaintext_links(element, soup, link_text=link_text)
    except AttributeError:
        pass
