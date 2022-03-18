# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 07-Oct-2021
#
#  @author: tbowers

"""Manage the Database for LDT Instrument Calibration Frame Information

This module is part of the Roz package, written at Lowell Observatory.

This module houses the database class CalibrationDatabase().  Among its methods
are those for committing data to an InfluxDB database for posterity.  Functions
from the LIGMOS library are used for the InfluxDB pieces.

This module primarily trades in its own class.
"""

# Built-In Libraries
import datetime as dt

# 3rd Party Libraries
from astropy.table import Table
from influxdb import DataFrameClient
import numpy as np

# Lowell Libraries
from ligmos import utils as lig_utils
from ligmos import workers as lig_workers
from johnnyfive import utils as j5u

# Internal Imports
from roz import process_calibrations as pc
from roz import utils

# Set API Components
__all__ = ["CalibrationDatabase", "ScienceDatabase", "HistoricalData"]


class CalibrationDatabase:
    """CalibrationDatabase

    Database class for calibration frames

    Provides a container for the metadata from a night plus the methods needed
    to insert them into the InfluxDB database

    Parameters
    ----------
    inst_flags : `dict`
        Dictionary of instrument flags from .utils.set_instrument_flags()
    proc_dir : `str` or `pathlib.Path`
        Path to the processing directory
    bias_meta : `astropy.table.Table`, optional
        Table containing the metadata and statistics for BIAS frames [Default: None]
    dark_meta : `astropy.table.Table`, optional
        Table containing the metadata and statistics for DARK frames [Default: None]
    flat_meta : `astropy.table.Table`, optional
        Table containing the metadata and statistics for FLAT frames [Default: None]
    """

    def __init__(
        self, inst_flags, proc_dir, bias_meta=None, dark_meta=None, flat_meta=None
    ):
        # Set internal variables
        self.flags = inst_flags
        self.proc_dir = proc_dir

        # Set up the internal dictionaries to hold calibration metadata
        self.bias = (
            pc.validate_bias_table(bias_meta) if inst_flags["get_bias"] else None
        )
        self.dark = (
            pc.validate_dark_table(dark_meta) if inst_flags["get_dark"] else None
        )
        if inst_flags["get_flat"]:
            self.flat = {}
            for lmi_filt in utils.LMI_FILTERS:
                self.flat[lmi_filt] = pc.validate_flat_table(flat_meta, lmi_filt)
        else:
            self.flat = None

        # Read in the InfluxDB config file
        self.db_set = utils.read_ligmos_conffiles("databaseSetup")

        # The InfluxDB object is thuswise constructed:
        self.idb = lig_utils.database.influxobj(
            tablename=self.db_set.tablename,
            host=self.db_set.host,
            port=self.db_set.port,
            user=self.db_set.user,
            pw=self.db_set.password,
            connect=True,
        )

    @property
    def bias_temp(self):
        """bias_temp Bias Level and Temperature

        Return the bias levels and temperatures as a @property of the instance

        Returns
        -------
        bias_crop_avg : `numpy.ndarray`
           Array of the mean bias levels in the CROP region of the CCD
        mnttemp : `numpy.ndarray`
            Array of the corresponding mount temperatures
        """
        # If the bias table is empty, return zeros
        if self.bias is None:
            return np.asarray([0]), np.asarray([0])
        return np.asarray(self.bias["crop_avg"]), np.asarray(self.bias["mnttemp"])

    def write_to_influxdb(self, testing=True):
        """write_to_influxdb Write the contents to the InfluxDB

        Following the example of Ryan's Docker_Pi/MesaTools/onewireTemps/,
        this method packetizes the bias and flat metadata dictionaries and
        commits them to the InfluxDB database, whose location and credentials
        are in config/roz.conf

        Parameters
        ----------
        testing : `bool`, optional
            If testing, don't commit to InfluxDB  [Default: True]
        """
        # If bias table is extant, loop over frames in self.bias to commit
        #  each one individually
        if self.bias is not None:
            for entry in self.bias:
                packet = neatly_package(entry, measure=self.db_set.metricname)
                # Commit in a safe way
                if not testing:
                    j5u.safe_service_connect(
                        self.idb.singleCommit, packet, table=self.db_set.tablename
                    )

        # If not LMI, then bomb out now
        if self.flags["instrument"] != "LMI":
            return

        # Loop through the filters, making FLAT packets and commit them
        for filt in utils.LMI_FILTERS:
            # Skip filters not used in this data set
            # print(f"Committing LMI filter {filt}...")
            if self.flat[filt] is None:
                continue

            # Loop
            for entry in self.flat[filt]:
                packet = neatly_package(entry, measure=self.db_set.metricname)
                # Commit
                if not testing:
                    self.idb.singleCommit(packet, table=self.db_set.tablename)


class ScienceDatabase:
    """ScienceDatabase

    Database class for science frames

    Provides a container for the metadata from a night plus the methods needed
    to insert them into the InfluxDB database
    """

    def __init__(self):
        pass

    def write_to_influxdb(self, testing=True):
        """write_to_influxdb _summary_

        _extended_summary_

        Parameters
        ----------
        testing : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """


class HistoricalData:
    """HistoricalData _summary_

    This class pulls historical data from the InfluxDB for comparison with the
    present frames to alert for changes.

        Parameters
        ----------
        instrument : `str`
            Instrument name for which to pull (REQUIRED)
        frametype : `str`
            Frame type for which to pull (REQUIRED)
        filt : `str, optional
            Filter for which to pull [Default: None]
        binning : `str` (of form 'cxr'), optional
            Binning for which to pull [Default: None]
        numamp : `int`, optional
            Number of amplifiers for which to pull [Default: None]
        ampid : `str`, optional
            Amplifier ID for which to pull [Default: None]
        cropborder : `int`, optional
            Crop border size for which to pull [Default: None]
    """

    def __init__(
        self,
        instrument,
        frametype,
        filt=None,
        binning=None,
        numamp=None,
        ampid=None,
        cropborder=None,
    ):
        # Init various attributes
        self.results = None

        # Create the tag dictionary from the class init inputs
        self.tagdict = {
            "instrument": instrument,
            "frametype": frametype,
            "filter": filt,
            "binning": binning,
            "numamp": numamp,
            "ampid": ampid,
            "cropborder": cropborder,
        }

        # Parse the configuration file
        db_query, db_info = lig_utils.confparsers.parseConfig(
            utils.Paths.dbqueries,
            lig_utils.classes.databaseQuery,
            passfile=None,
            searchCommon=True,
            enableCheck=False,
        )

        # Formally create the query from the parsed configuration file
        self.query = lig_workers.confUtils.assignComm(
            db_query, db_info, commkey="database"
        )

    def perform_query(self):
        """perform_query Perform the InfluxDB query

        _extended_summary_

        Returns
        -------
        `astropy.table.Table`
            The AstroPy table containing the query results
        """
        # There should only be one query key
        key = list(self.query.keys())[0]

        self.results = get_results_table(
            self.query[key], tags=self.tagdict, debug=False
        )

    @property
    def instruments(self):
        """
        Returns
        -------
        `list`
            Sorted list of the unique instruments found in this query result
        """
        return self._sorted_list_set("instrument")

    @property
    def frametypes(self):
        """
        Returns
        -------
        `list`
            Sorted list of the unique frametypes found in this query result
        """
        return self._sorted_list_set("frametype")

    @property
    def filters(self):
        """
        Returns
        -------
        `list`
            Sorted list of the unique filters found in this query result
        """
        return self._sorted_list_set("filter")

    @property
    def binnings(self):
        """
        Returns
        -------
        `list`
            Sorted list of the unique binnings found in this query result
        """
        return self._sorted_list_set("binning")

    @property
    def numamps(self):
        """
        Returns
        -------
        `list`
            Sorted list of the unique numamps found in this query result
        """
        return self._sorted_list_set("numamp")

    @property
    def ampids(self):
        """
        Returns
        -------
        `list`
            Sorted list of the unique ampids found in this query result
        """
        return self._sorted_list_set("ampid")

    @property
    def cropborders(self):
        """
        Returns
        -------
        `list`
            Sorted list of the unique cropborders found in this query result
        """
        return self._sorted_list_set("cropborder")

    def _sorted_list_set(self, tagname):
        """_sorted_list_set Return a Sorted Unique List of result tagname

        Checks to see if the tagname exists in the results table; if not,
        return an empty list.

        Parameters
        ----------
        tagname : `str`
            The InfluxDB tagname to be returned

        Returns
        -------
        `list`
            The sorted list of unique entries
        """
        if tagname in self.results.colnames:
            return sorted(list(set(self.results[tagname])))
        return []


# Non-Class Helper Functions =================================================#
def build_influxdb_query(dbq, tags=None, debug=False):
    """build_influxdb_query Build the query string for InfluxDB

    This function builds the (long) query string to be posted to InfluxDB

    NOTE: dtime = int(dbq.rangehours) is the time from present (in hours)
          to query back

    Parameters
    ----------
    dbq : `ligmos.utils.classes.databaseQuery`
        The database query object, as read from the configuration file
    tags : `dict`, optional
        The tags to which to limit the database search [Default: None]
    debug : `bool`, optional
       Print debugging statements? [Default: False]

    Returns
    -------
    `str`
        The InfluxDB-compliant query string
    """
    try:
        dtime = int(dbq.rangehours)
    except ValueError:
        print(f"Can't convert {dbq.rangehours} to int... using ~1.5yrs")
        dtime = 13000

    if dbq.database.type.lower() != "influxdb":
        print("Error: Database must be of type `influxdb`!")
        return None

    if debug is True:
        print(
            f"Searching for {dbq.fields} in {dbq.tablename}.{dbq.metricname} "
            f"on {dbq.database.host}:{dbq.database.port}"
        )

    # Begin by specifying we want ALL THE FIELDS from the Metric Name
    query = f'SELECT * FROM "{dbq.metricname}"'

    # Add the Time Range: Namely the most recent `dtime` hours
    query += f" WHERE time > now() - {dtime}h"

    # Finally, add the tags as the primary constraints on the query
    if tags:
        query += " AND "
        for tagname, tagval in tags.items():
            # Check that tagval is not None:
            if tagval:
                query += f"\"{tagname}\"='{tagval}' AND "

        # Strip the trailing ' AND ':
        query = query.rstrip(" AND ")

    # Return the mess
    return query


def get_results_table(query, tags=None, debug=False):
    """get_results_table Get the query results as a Table

    This function is a simplified version of
    `ligmos.utils.database.getResultsDataFrame()`, in that it doesn't try to
    catch all eventualities.  It also returns an AstroPy Table instead of a
    pandas dataframe, for simplicity of use with the rest of Roz.

    Parameters
    ----------
    query : `ligmos.utils.classes.databaseQuery`
        The database query object, as read from the configuration file
    tags : `dict`, optional
        The tags to which to limit the database search [Default: None]
    debug : `bool`, optional
       Print debugging statements? [Default: False]

    Returns
    -------
    `astropy.table.Table`
        The Table containing the database query results
    """
    # Build the query string
    query_str = build_influxdb_query(query, tags=tags, debug=debug)
    print(f"This is the query string:\n{query_str}")

    # InfluxDB Data Frame Client:
    idfc = DataFrameClient(
        host=query.database.host,
        port=query.database.port,
        username=query.database.user,
        password=query.database.password,
        database=query.tablename,
    )

    # Get the results of the query in a safe way:
    results = j5u.safe_service_connect(idfc.query, query_str)

    # If `results` is empty, return a (nearly) empty table
    if results == {}:
        # TODO: Convert this to a warning or send_alert() thing
        print("Query returned no results!")
        return Table(
            names=("timestamp", "instrument", "frametype"), dtype=("O", "U12", "U12")
        )

    # `results` is a dict of pandas dataframes; but in our case there is only
    #   one key in the dict, namely `query.metricname`.

    # First, extract the "index", which is the timestamp for the measurement
    timestamp = results[query.metricname].index.to_pydatetime()

    # Convert to a single AstroPy Table
    table = Table.from_pandas(results[query.metricname])

    # Add the timestamps as an additional column
    table["timestamp"] = timestamp

    return table


def neatly_package(table_row, measure):
    """neatly_package Carefully curate and package the InfluxDB packet

    This function translates the internal database into an InfluxDB object.

    Makes an InfluxDB styled packet given the measurement name, metadata tags,
    and actual fields/values to put into the database

    Parameters
    ----------
    table_row : `astropy.table.Row`
        The row of data to commit to InfluxDB
    measure : `str`
        The database MEASUREMENT into which to place this row

    Returns
    -------
    `list` of `dict`
        Single-element list of packet `dict` containing the information to be
        inserted into the InfluxDB database.
    """
    # Convert the AstroPy Table Row into a dict by adding colnames
    row_as_dict = dict(zip(table_row.colnames, table_row))

    # We want the database timestamp to be that of the image DATEOBS,
    #  not the current time.  Therefore, we need to create a datetime()
    #  object from the field `dateobs`.  (NOTE: .fromisoformat() not working)
    timestamp = dt.datetime.strptime(
        f"{row_as_dict.pop('dateobs')}", "%Y-%m-%dT%H:%M:%S.%f"
    )

    # Build the tags from information in the table Row
    tags = {
        "instrument": row_as_dict.pop("instrument").lower(),
        "frametype": row_as_dict.pop("frametyp").lower(),
        "filter": row_as_dict.pop("filter"),
        "binning": row_as_dict.pop("binning"),
        "numamp": row_as_dict.pop("numamp"),
        "ampid": row_as_dict.pop("ampid"),
        "cropborder": row_as_dict.pop("cropsize"),
    }

    # Strip off the filename, as it can be reconstructed from obserno
    row_as_dict.pop("filename")

    # Build the packet as a dictionary with the proper InfluxDB keys
    packet = {
        "measurement": measure,
        "tags": tags,
        "time": timestamp,
        "fields": row_as_dict,
    }

    # InfluxDB expects a list of dicts, so return such
    return [packet]


# Testing ====================================================================#
if __name__ == "__main__":
    hist = HistoricalData("lmi", "dome flat")
    hist.perform_query()
    hist.results.pprint()
    print(hist.results.colnames)

    print(hist.instruments)
    print(hist.frametypes)
    print(hist.filters)
    print(hist.binnings)
    print(hist.numamps)
    print(hist.ampids)
    print(hist.cropborders)
