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

This module primarily trades in internal databse objects
(`roz.database_manager.CalibrationDatabase`).
"""

# Built-In Libraries
import datetime
import warnings

# 3rd Party Libraries
import astropy.io.votable
import astropy.table
import bs4
import numpy as np

# Lowell Libraries
import johnnyfive

# Internal Imports
from roz import alerting
from roz import graphics_maker
from roz import msgs
from roz import utils


# Set API Components
__all__ = ["update_filter_characterization"]


# Outward-facing function ====================================================#
def update_filter_characterization(
    database, png_only=False, delete_existing=False, debug=False
):
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
        [Defualt: False]  NOTE: Once in production, maybe turn this to True?
    debug : `bool`, optional
        Pring debugging statements?  [Default: False]
    """
    # Silence JohnnyFive's PermissionWarning -- we know, and we don't care
    warnings.simplefilter("ignore", johnnyfive.PermissionWarning)

    # Instantiate a ConfluencePage object for the LMI Filter Page
    page_info = utils.read_ligmos_conffiles("lmifilterSetup")
    lmi_filter_info = johnnyfive.ConfluencePage(page_info.space, page_info.page_title)

    # If the page doesn't already exist (or Confluence times out),
    #   send alert and return
    if not lmi_filter_info.exists:
        alerting.send_alert(
            "LMI Filter Information does not exist in the expected location",
            "confluence_updater.update_filter_characterization()",
        )
        return

    if debug:
        # Get the `page_id` needed for intracting with the page we want to update
        msgs.test(f"This is the page_id: {lmi_filter_info.page_id}")

    # Update the HTML table attached to the Confluence page
    png_fn = update_lmi_filter_table(
        utils.Paths.local_html_table_fn,
        database,
        lmi_filter_info.attachment_url,
        png_only=png_only,
        debug=debug,
    )

    # If nothing changed, return now
    if not png_fn:
        return

    # Remove the attachment on the Confluence page before uploading the new one
    if delete_existing:
        lmi_filter_info.delete_attachment(utils.Paths.html_table_fn)

    # Attach the HTML file to the Confluence page
    lmi_filter_info.attach_file(
        utils.Paths.local_html_table_fn,
        name=utils.Paths.html_table_fn,
        content_type="text/html",
        comment="LMI Filter Information Table",
    )

    # Attach any PNGs created, but only after removing the existing one for that filter
    attch_fnames = [
        result["title"] for result in lmi_filter_info.get_page_attachments()["results"]
    ]
    for png in png_fn:
        # This is the instrument plus filter, like "lmi.SDSS-Z." or "lmi.R."
        prefix = f"{'.'.join(png.split('.')[:2])}."
        # List of existing filenames to remove from Confluence prior to upload
        existing_png = [attch for attch in attch_fnames if attch.startswith(prefix)]
        for existing in existing_png:
            msgs.warn(f"Deleting {existing} from Confluence...")
            lmi_filter_info.delete_attachment(existing)
        # Now, attach this new file!
        lmi_filter_info.attach_file(
            utils.Paths.thumbnail.joinpath(png),
            name=png,
            content_type="image/png",
            comment="Flat Field Image",
        )
        msgs.info(f"Uploaded: {png}")

    # Print a happy little message if we actually did anything
    msgs.info(f"Successfully updated the Confluence page `{page_info.page_title}`")


# Descriptive, high-level functions ==========================================#
def update_lmi_filter_table(
    filename, database, attachment_url, png_only=False, debug=False
):
    """update_lmi_filter_table Update the LMI Filter Information Table

    Updates (on disk) the HTML table of LMI filter information for upload to
    Confluence.  This table is partially static (basic information about the
    filters, etc.), and partially dynamic, listing the UT date of the last
    flatfield, and the most recent estimation of countrate for that filter/lamp
    combination.

    This table also holds (links to) PNG images of 1) a carefully curated
    nominal flatfield, and 2) the most recent flatfield in this filter.

    Parameters
    ----------
    filename : `string`
        Local filename of the HTML table to create or update
    database : `roz.database_manager.CalibrationDatabase`
        The database of calibration frames
    attachment_url : `str`
        The URL for attachments for this page in Confluence.  (Needed for
        creating the proper links within the HTML table.)
    png_only : `bool`, optional
        Only update the PNG image and not the countrate/exptime columns
        [Default: False]
    debug : `bool`, optional
        Print debugging statements? [Default: False]

    Returns
    -------
    `list`
        List of PNG filenames created for this run.
    - on disk -
        Updates the HTML table stored in the data/ directory
    """
    # Get the base (static) table
    lmi_filt, section_head = load_lmi_static_table()

    # Use the `database` to modify the dynamic portions of the LMI table
    lmi_filt, png_fn = modify_lmi_dynamic_table(
        lmi_filt, database, attachment_url, png_only=png_only, debug=debug
    )
    if debug:
        lmi_filt.pprint()

    # If anything was updated, there will be something in `png_fn`:
    if png_fn:
        # Use the AstroPy Table `lmi_filt` to construct the HTML table
        #   and write it to disk
        construct_lmi_html_table(
            lmi_filt, section_head, filename, link_text="Image Link", debug=debug
        )

    # Return list of PNG filenames
    return png_fn


def modify_lmi_dynamic_table(
    lmi_filt, database, attachment_url, png_only=False, debug=False
):
    """modify_lmi_dynamic_table Modify the dynamic portions of the table

    Read in, modify, then write out the dynamic portion of the HTML table, and
    finally join with the static postion of the table (from the XML file)
    before returning.  This function modifies the dynamic portion of the table
    with information contained in the `database`.

    This function checks whether the countrates are within a reasonable level
    before adjusting the countrate or exptime columns.

    Before returning, the function joins the dynamic table with the static
    portion though use of the `astropy.table.join(join_type="left")` function.
    This ensures that the HTML table produced and the saved `dyntable` will
    always habve the same rows as the current lmi_filter_table.[xml,ecsv] file.

    NOTE: If there are changes to the LMI filter complement that need to be
          reflected in the HTML table, those changes must be made to BOTH the
          ECSV and XML tables.

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
        Print debugging statements? [Default: False]

    Returns
    -------
    lmi_filt : `astropy.table.Table`
        The dynamically augmented LMI Filter Information Table
    png_fn : `list`
        List of the PNG filenames created during this run
    """
    # Read in the dynamic-portion FITS table, if extant
    if utils.Paths.lmi_dyntable.is_file():
        dyntable = astropy.table.Table.read(utils.Paths.lmi_dyntable)

    else:
        # Make a blank table, including the lmi_filters for correspondence
        dyntable = astropy.table.Table(
            data=np.zeros(
                len(utils.FILTER_LIST["LMI"]),
                dtype=[
                    ("Filter", "U128"),
                    ("Latest Image", "U256"),
                    ("UT Date of Latest Flat", "U128"),
                    ("Count Rate (ADU/s)", float),
                    ("Exptime for 20k cts (s)", float),
                ],
            )
        )
        dyntable["Filter"] = utils.FILTER_LIST["LMI"]

    # Merge the static and dynamic portions together to ensure conformity
    #   The astropy.table function join() combines tables based on common keys,
    #   however, it also sorts the table...
    lmi_filt = astropy.table.join(lmi_filt, dyntable, join_type="left", keys="Filter")
    #   Undo the alpha sorting done by .join()
    lmi_filt = utils.table_sort_on_list(
        lmi_filt, "FITS Header Value", utils.FILTER_LIST["LMI"]
    )
    #   Make sure the `Latest Image` and `UT Date` columns are long enough
    lmi_filt["Latest Image"] = lmi_filt["Latest Image"].astype("U256")
    lmi_filt["UT Date of Latest Flat"] = lmi_filt["UT Date of Latest Flat"].astype(
        "U128"
    )

    if debug:
        lmi_filt.pprint()

    # Loop through the filters, updating the relevant columns of the table
    png_fn = []
    for i, filt in enumerate(utils.FILTER_LIST["LMI"]):

        # Skip filters not used in this data set (also check for `database.flat`)
        if not database.v_tables["flat"] or not database.v_tables["flat"][filt]:
            continue

        # Only update if the DATOBS of this flat is LATER than extant value
        # NOTE: Add several hours to the header DATEOBS to ensure it's past 5pm MST
        incoming_date = utils.scrub_isot_dateobs(
            database.v_tables["flat"][filt]["dateobs"][-1], add_hours=3
        )
        # There are 3 possible scenarios here:
        #   1. There is no existing date, so insert
        #   2. The existing date is older than the incoming date, so replace
        #   3. The existing date is newer than the incoming date, so skip

        # Case 1
        if isinstance(lmi_filt["UT Date of Latest Flat"][i], np.ma.core.MaskedConstant):
            # The incoming date stands
            if debug:
                msgs.test(f"Filter {filt}: Case 1")

        # Case 2
        elif (
            utils.scrub_isot_dateobs(lmi_filt["UT Date of Latest Flat"][i])
            < incoming_date
        ):
            # The incoming date replaces the extant date
            if debug:
                msgs.test(f"Filter {filt}: Case 2")

        # Case 3
        else:
            # The existing date is newer; skip this filter
            if debug:
                msgs.test(f"Filter {filt}: Case 3")
            continue

        # Check whether the correct flat lamps were used (as judged by count rate)
        good_lamps = check_lamp_countrates(database.v_tables["flat"][filt])
        if debug:
            msgs.test(f"These are good_lamps: {good_lamps}")

        # Call the PNG-maker to PNG-ify the latest image (if good); record PNG's filename
        if any(good_lamps):
            png_fn.append(
                graphics_maker.make_png_thumbnail(
                    database.proc_dir.joinpath(
                        database.v_tables["flat"][filt]["filename"][good_lamps][-1]
                    ),
                    database.v_report["flags"],
                )
            )

            # Update the dynamic columns
            lmi_filt["Latest Image"][i] = f"{attachment_url}{png_fn[-1]}?api=v2"
            lmi_filt["UT Date of Latest Flat"][i] = incoming_date.strftime("%Y-%m-%d")

            # Compute the expected count rates, if specified for GOOD LAMPS
            if not png_only:
                lmi_filt["Count Rate (ADU/s)"][i] = (
                    count_rate := np.mean(
                        database.v_tables["flat"][filt]["crop_med"][good_lamps]
                    )
                )
                lmi_filt["Exptime for 20k cts (s)"][i] = 20000.0 / count_rate

    # Split off the dyntable portion again, and write it back to disk for later
    dyntable = astropy.table.Table(
        [
            lmi_filt["Filter"],
            lmi_filt["Latest Image"],
            lmi_filt["UT Date of Latest Flat"],
            lmi_filt["Count Rate (ADU/s)"],
            lmi_filt["Exptime for 20k cts (s)"],
        ],
        names=[
            "Filter",
            "Latest Image",
            "UT Date of Latest Flat",
            "Count Rate (ADU/s)",
            "Exptime for 20k cts (s)",
        ],
    )
    dyntable.write(utils.Paths.lmi_dyntable, overwrite=True)

    # Add formatting constraints to the `lmi_filt` table columns
    lmi_filt["Count Rate (ADU/s)"] = astropy.table.Column(
        lmi_filt["Count Rate (ADU/s)"].filled(0), format=utils.two_sigfig
    )
    lmi_filt["Exptime for 20k cts (s)"] = astropy.table.Column(
        lmi_filt["Exptime for 20k cts (s)"].filled(0), format=utils.two_sigfig
    )

    return lmi_filt, png_fn


# Utility Functions (Alphabetical) ===========================================#
def add_html_section_header(soup, ncols, text, extra=""):
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
    newrow = soup.new_tag("tr")
    # One column spanning the whole row, with Lowell-gray for background
    newcol = soup.new_tag("td", attrs={"colspan": ncols, "bgcolor": "#DBDCDC"})
    # Bold/Underline the main `text` for the header; append to newcol
    bold = soup.new_tag("b")
    uline = soup.new_tag("u")
    uline.string = text
    bold.append(uline)
    newcol.append(bold)
    # Add any `extra` text in standard font after the bold/underline portion
    newcol.append("" if isinstance(extra, np.ma.core.MaskedConstant) else extra)
    # Put the column tag inside the row tag
    newrow.append(newcol)
    # All done
    return newrow


def check_lamp_countrates(table):
    """Check LMI Flat Field Count Rates against lamp type

    Only receives the table associated with a single filter.

    Define Minimum Count Rates for each filter under the correct lamps:

    Parameters
    ----------
    table : `astropy.table.Table`
        Validated table of frames for this filter

    Returns
    -------
    `np.ndarray`
        Boolean array indicating whether the count rate for this filter is
        appropriate (i.e., was the correct lamp used?).
    """
    # Johnson-Cousins Filters:
    if (filt := table["filter"][0]) in ["U", "B", "V", "R", "I"]:
        min_count = 1500
    # Sloan Filters:
    elif filt in ["SDSS u'", "SDSS g'", "SDSS r'", "SDSS i'", "SDSS z'"]:
        min_count = 2000
    # Other Broadband Filters:
    elif filt == "V+R":
        min_count = 5000
    elif filt == "Yish":
        min_count = 500
    # Comet Filters:
    elif filt in [
        "Ultraviolet Continuum",
        "Blue Continuum",
        "Green Continuum",
        "Red Continuum",
        "C2",
        "C3",
        "CN",
        "CO+",
        "H2O+",
        "OH",
        "NH",
    ]:
        min_count = 200
    # Everything else, don't check
    else:
        min_count = 0

    # Return a boolean array-like of whether we pass muster
    return table["crop_med"] > min_count


def construct_lmi_html_table(
    lmi_filt, section_head, filename, link_text="Click Here", debug=False
):
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

    # CSS stuff to make the HTML table pretty -- read it in from file
    with open(utils.Paths.css_table, "r", encoding="utf8") as fileobj:
        css_style = fileobj.readlines()

    # Use the AstroPy HTML functionality to get us most of the way there
    lmi_filt.write(filename, overwrite=True, htmldict={"css": "".join(css_style)})

    # Now that AstroPy has done the hard work writing this table to HTML,
    #  we need to modify it a bit for visual clarity.  Use BeautifulSoup!
    with open(filename, encoding="utf8") as fileobj:
        soup = bs4.BeautifulSoup(fileobj, "html.parser")

    # Add the `creation date` line to the body of the HTML above the table
    timestr = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    itdate = soup.new_tag("i")  # Italics, for the fun of it
    itdate.string = f"Table Auto-Generated {timestr} UTC by Roz."
    # Place the italicized date string ahead of the table
    soup.find("table").insert_before(itdate)
    if debug:
        msgs.test(f"HTML table timestamp: {timestr}")

    # Add the section headings for the different filter groups:
    for i, row in enumerate(soup.find_all("tr")):
        # At each row, search through the `section_head` table to correctly
        #  insert the appropriate section header
        for sechead in section_head:
            if i == sechead["insert_after"]:
                row.insert_after(
                    add_html_section_header(
                        soup, ncols, sechead["section"], sechead["extra"]
                    )
                )

    # Convert the bare URLs of all PNG thumbnails into hyperlinks.  Use a
    #  recursive function to navigate the HTML BeautifulSoup tree.
    wrap_plaintext_links(soup, soup, link_text=link_text)

    # Now that we've mucked with the HTML document, rewerite it to disk
    with open(filename, "wb") as fileobj:
        fileobj.write(soup.prettify("utf-8"))


def load_lmi_static_table(table_type="ecsv"):
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
    """
    if table_type == "xml":
        # Read in the XML table.
        votable = astropy.io.votable.parse(utils.Paths.xml_table)

        # The VOTable has both the LMI Filter Info and the section heads for the HTML
        filter_table = votable.get_table_by_index(0).to_table(use_names_over_ids=True)
        section_head = votable.get_table_by_index(1).to_table()

    elif table_type == "ecsv":
        # Read in the ECSV tables (LMI Filter Info and the HTML section headings)
        filter_table = astropy.table.Table.read(utils.Paths.ecsv_filters)
        section_head = astropy.table.Table.read(utils.Paths.ecsv_sechead)

    else:
        msgs.error(f"Table type {table_type} not recognized!")

    return filter_table, section_head


def wrap_plaintext_links(bs_tag, soup, link_text="Click Here"):
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
            if (
                isinstance(element, bs4.element.NavigableString)
                and element.string[:4] == "http"
            ):
                # If this is a string that starts with 'http', linkify!
                link = soup.new_tag("a", href=element.string)
                link.string = link_text
                element.replace_with(link)
            elif element.name != "a":
                # Recursive execution to traverse the tree
                wrap_plaintext_links(element, soup, link_text=link_text)
    except AttributeError:
        pass


# def purge_page_attachments(args=None):
#     """purge_page_attachments Quick script for clearing accumulated attachments

#     Console Script for the quick cleaning of accumulated attachment cruft
#     """
#     # Instantiate a ConfluencePage object for the LMI Filter Page
#     page_info = utils.read_ligmos_conffiles("lmifilterSetup")
#     lmi_filter_info = johnnyfive.ConfluencePage(page_info.space, page_info.page_title)

#     # Get the attachments, and parse out the filename "titles" from the returned object
#     attachments = lmi_filter_info.get_page_attachments(limit=200)
#     titles = [result["title"] for result in attachments["results"]]

#     # Say something about what we're up to, then get to it
#     print(f"Puring {len(titles)} attachments from {page_info.page_title}...")
#     for title in titles:
#         print(f"Deleting {title}...")
#         lmi_filter_info.delete_attachment(title)
