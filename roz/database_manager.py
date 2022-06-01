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

This module houses the database classes `CalibrationDatabase()`,
`ScienceDatabase()`, and `HistoricalData()`.  Among their methods are those
for committing data to an InfluxDB database for posterity, as well as reading
back in committed data for comparison with new frames.  Functions from the
LIGMOS library are used for most of the InfluxDB pieces, and custom
distillations of extant LIGMOS functions are also included here when Roz's
needs differ from those of other LIGMOS consumers.

This module primarily trades in its own classes.
"""

# Built-In Libraries
import warnings

# 3rd Party Libraries
from astropy.table import Table
from influxdb import DataFrameClient
import numpy as np

# Lowell Libraries
import johnnyfive
import ligmos

# Internal Imports
from roz import alerting
from roz import utils
from roz import validate_statistics

# Set API Components
__all__ = ["CalibrationDatabase", "ScienceDatabase", "HistoricalData"]


class CalibrationDatabase:
    """CalibrationDatabase

    Database class for calibration frames

    Provides a container for the metadata from a night plus the methods needed
    to insert them into the InfluxDB database.

    Parameters
    ----------
    inst_flags : `dict`
        Dictionary of instrument flags from utils.set_instrument_flags()
    proc_dir : `str` or `pathlib.Path`
        Path to the processing directory
    nightname: `str`
        Name of the night (e.g., `lmi/20210106b` or `deveny/20220221a`)
    calib_container: `process_calibrations.CalibContainer`
        Calibration Container Class from adjoining module; contains the
        various calibration metadata tables
    """

    def __init__(self, inst_flags, proc_dir, nightname, binning, calib_container):
        # Set instance attributes
        self.proc_dir = proc_dir

        # Construct the basic validation report dictionary:
        self.v_report = {
            "nightname": nightname,
            "flags": inst_flags,
            "binning": binning.replace(" ", "x"),
        }

        # Place the metadata tables into a dictionary; init empty validated dict
        self.meta_tabls = {
            attr: getattr(calib_container, attr, None)
            for attr in dir(calib_container)
            if "_meta" in attr
        }
        self.v_tables = {}

        # Read in the InfluxDB config file
        self.db_setup = utils.read_ligmos_conffiles("databaseSetup")

        # The InfluxDB object is thuswise constructed:
        self.influxdb = ligmos.utils.database.influxobj(
            tablename=self.db_setup.tablename,
            host=self.db_setup.host,
            port=self.db_setup.port,
            user=self.db_setup.user,
            pw=self.db_setup.password,
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
        if "bias" not in self.v_tables or not self.v_tables["bias"]:
            return np.asarray([0]), np.asarray([0])
        return (
            np.asarray(self.v_tables["bias"]["crop_avg"]),
            np.asarray(self.v_tables["bias"]["mnttemp"]),
        )

    def validate(self, sigma_thresh=3.0, scheme="simple", **kwargs):
        """validate Run the validation routines on the tables

        The primary validation is the 'simple' scheme, whereby frames are
        checked against the historical statistics, looking for things that
        deviate by more than `sigma_thresh` sigma.

        Other validation `schemes` that could happen are:
          * ???

        Parameters
        ----------
        sigma_thresh : `float`, optional
            The sigma discrepancy threshold for flagging a frame as being
            'problematic'  [Default: 3.0]
        scheme : `str`, optional
            The validation scheme to be used  [Default: simple]

        ---- Various debugging keyword arguments (to be removed later)
            no_prob : `bool`, optional
            Only use metrics not marked as "problem" by previous validation
            [Default: True]
        all_time : `bool`, optional
            For validation of current frames, compare against all matches,
            regardless of the timestamp [Default: False]

        """
        # Load in the filter list for this instrument
        instrument = self.v_report["flags"]["instrument"]
        try:
            filter_list = utils.FILTER_LIST[instrument]
        except KeyError:
            warnings.warn(
                f"No filter list set for instrument {instrument} "
                "in utils.py!  Using ['OPEN'].",
                utils.DeveloperWarning,
            )
            filter_list = ["OPEN"]

        # Load up the internal dictionaries with validated data and reports
        (
            self.v_tables,
            frame_reports,
            s_str,
        ) = validate_statistics.validate_calibration_metadata(
            self.meta_tabls,
            filt_list=filter_list,
            sigma_thresh=sigma_thresh,
            scheme=scheme,
            **kwargs,
        )

        # Add the `frame_reports` and `scheme_string` to the full validation report
        self.v_report.update({"frame_reports": frame_reports, "valid_scheme": s_str})

        # Convert the validation report into a problem report; post
        if problem_report := validate_statistics.build_problem_report(self.v_report):
            print("++++> Posting Problem Report to Slack...")
            alerting.post_report(problem_report)
            alerting.post_pngs(self.v_tables, self.proc_dir, self.v_report["flags"])

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
        # Loop through the frame types, pulling the dictionary of tables
        for frametype, frametype_ftables in self.v_tables.items():

            # Check that the table dictionary is extant
            if not frametype_ftables:
                continue

            # List of packets to be committed
            pkt_list = []
            # Loop through the filters, making packets and accumulating them
            for filt in frametype_ftables["filters"]:
                # Skip filters not used in this data set (i.e., empty table)
                if not frametype_ftables[filt]:
                    continue
                # Loop over frames (i.e. rows in table)
                for row in frametype_ftables[filt]:
                    pkt_list.append(
                        neatly_package(row, measure=self.db_setup.metricname)
                    )

            # Commit the packet list and print a message
            if not testing:
                print(
                    f"Writing {len(pkt_list)} {frametype.upper()} frames to InfluxDB...",
                    end=" ",
                )
                johnnyfive.safe_service_connect(
                    self.influxdb.singleCommit, pkt_list, table=self.db_setup.tablename
                )


class ScienceDatabase:
    """ScienceDatabase

    Database class for science frames

    Provides a container for the metadata from a night plus the methods needed
    to insert them into the InfluxDB database
    """

    def __init__(self):
        pass

    def bogus_public_method(self):
        """bogus_public_method Making the linter happy!"""

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
    filter : `str, optional
        Filter for which to pull [Default: None]
    binning : `str` (of form 'cxr'), optional
        Binning for which to pull [Default: None]
    numamp : `int`, optional
        Number of amplifiers for which to pull [Default: None]
    ampid : `str`, optional
        Amplifier ID for which to pull [Default: None]
    cropborder : `int`, optional
        Crop border size for which to pull [Default: None]
    debug : `bool`, optional
        Print debugging statements?  [Default: False]
    """

    def __init__(self, instrument, frametype, debug=False, **kwargs):
        # Init various attributes
        self.results = None

        # Create the tag dictionary from the class init inputs
        self.tagdict = {"instrument": instrument, "frametype": frametype}
        self.tagdict.update(kwargs)
        if debug:
            print(self.tagdict)

        # Parse the database query configuration file
        db_query, db_info = ligmos.utils.confparsers.parseConfig(
            utils.Paths.dbqueries,
            ligmos.utils.classes.databaseQuery,
            passfile=None,
            searchCommon=True,
            enableCheck=False,
            debug=debug,
        )

        # Formally create the query from the parsed configuration file
        query = ligmos.workers.confUtils.assignComm(
            db_query, db_info, commkey="database"
        )
        # There should only be one query key; extract into self.query
        self.query = query[list(query.keys())[0]]

        # Build the InfluxDB Data Frame Client:
        self.idfc = DataFrameClient(
            host=self.query.database.host,
            port=self.query.database.port,
            username=self.query.database.user,
            password=self.query.database.password,
            database=self.query.tablename,
        )

    def perform_query(self, all_time=False, debug=False):
        """perform_query Perform the InfluxDB query, saving as a Table

        This method is a simplified version of
        `ligmos.utils.database.getResultsDataFrame()`, in that it doesn't try
        to catch all eventualities.  It also saves the result in an AstroPy
        Table instead of a pandas dataframe, for simplicity of use with the
        rest of Roz.

        Parameters
        ----------
        all_time : `bool`, optional
            Get all matches, regardless of the timestamp (i.e., disegard
            the value in seldf.query.rangehours)  [Default: False]
        """
        # Build the InfluxDB query string
        query_str = build_influxdb_query(
            self.query, tags=self.tagdict, all_time=all_time, debug=debug
        )
        if debug:
            print(f"This is the query string:\n{query_str}")

        # Get the results of the query in a safe way:
        results = johnnyfive.safe_service_connect(self.idfc.query, query_str)

        # If `results` is empty, assign a (nearly) empty table
        if results == {}:
            warnings.warn("The InfluxDB query returned no results!")
            self.results = Table(
                names=("timestamp", "instrument", "frametype"),
                dtype=("O", "U12", "U12"),
            )
            return

        # `results` is a dict of pandas dataframes; but in our case there is only
        #   one key in the dict, namely `self.query.metricname`.

        # First, extract the "index", which is the timestamp for the measurement
        timestamps = results[self.query.metricname].index.to_pydatetime()

        # Convert to a single AstroPy Table; add timestamps as additional column
        self.results = Table.from_pandas(results[self.query.metricname])
        self.results["timestamp"] = timestamps

    def metric_n(self, metric, no_prob=True, **kwargs):
        """metric_n Find the length of the returned metric

        Parameters
        ----------
        metric : `str`
            The InfluxDB field name for which to compute the mean.
        no_prob : `bool`, optional
            Only return metrics not marked as "problem" by previous validation
            [Default: True]
        **kwargs : `str`, optional
            The method also accepts key/value pairs of InfluxDB TAGS to further
            narrow the result table to a specific, say, `filter` or `binning`.

        Returns
        -------
        `int`
            The number of frames in the database matching this search
        """
        # Return the number of elements in the table metric
        return len(self._check_metric_kwargs(metric, no_prob=no_prob, **kwargs))

    def metric_mean(self, metric, no_prob=True, **kwargs):
        """metric_mean Compute the Mean of the returned Metric

        Uses np.nanmean() to produce a NaN-resistant mean of the specified
        metric in the InfluxDB result table.

        Parameters
        ----------
        metric : `str`
            The InfluxDB field name for which to compute the mean.
        no_prob : `bool`, optional
            Only return metrics not marked as "problem" by previous validation
            [Default: True]
        **kwargs : `str`, optional
            The method also accepts key/value pairs of InfluxDB TAGS to further
            narrow the result table to a specific, say, `filter` or `binning`.

        Returns
        -------
        `float`
            The mean value of the specified metric.  If the metric does not
            exist, or is full of undefined values, the method will return
            `np.nan`.
        """
        # Return the NaN-resistant mean of the table metric
        return np.nanmean(self._check_metric_kwargs(metric, no_prob=no_prob, **kwargs))

    def metric_stddev(self, metric, no_prob=True, **kwargs):
        """metric_mean Compute the Standard Deviation of the returned Metric

        Uses np.nanstd() to produce a NaN-resistant standard deviation of the
        specified metric in the InfluxDB result table.

        Parameters
        ----------
        metric : `str`
            The InfluxDB field name for which to compute the standard
            deviation.
        no_prob : `bool`, optional
            Only return metrics not marked as "problem" by previous validation
            [Default: True]
        **kwargs : `str`, optional
            The method also accepts key/value pairs of InfluxDB TAGS to further
            narrow the result table to a specific, say, `filter` or `binning`.

        Returns
        -------
        `float`
            The standard deviation value of the specified metric.  If the
            metric does not exist, or is full of undefined values, the method
            will return `np.nan`.
        """
        # Return the NaN-resistant standard deviation of the table metric
        return np.nanstd(self._check_metric_kwargs(metric, no_prob=no_prob, **kwargs))

    # The following methods are @property methods of the class =====#
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

    # Internal-use class methods ===================================#
    def _check_metric_kwargs(self, metric, no_prob=True, **kwargs):
        """_check_metric_kwargs Do QA testing on the input metric & kwargs

        _extended_summary_

        Parameters
        ----------
        metric : `str`
            The InfluxDB field name for which to compute something.
        no_prob : `bool`, optional
            Only return metrics not marked as "problem" by previous validation
            [Default: True]
        **kwargs : `str`, optional
            The method also accepts key/value pairs of InfluxDB TAGS to further
            narrow the result table to a specific, say, `filter` or `binning`.

        Returns
        -------
        `astropy.table.Column` or `np.nan`
            The specified column of the InfluxDB result table -- or NaN, if
            the metric is empty or does not exist.
        """
        # Check that all kwarg keys are in the results table
        if absent_keys := [key for key in kwargs if key not in self.results.colnames]:
            warnings.warn(
                f"The tag{'s' if (plural := len(absent_keys) > 1) else ''} "
                f"{absent_keys} {'are' if plural else 'is'} not in the results table!"
            )

        # Use any passed **kwargs to further narrow the self.results table
        results = (
            {key: val for key, val in kwargs.items() if key in self.results.colnames}
            if kwargs
            else self.results
        )

        # If the specifid metric is not in the table, return NaN
        if metric not in results.colnames:
            warnings.warn(f"The metric `{metric}` is not in the results table!")
            return np.nan

        # Trim out rows marked as "PROBLEM"
        if no_prob:
            results = results[results["problem"] != 1]

        # Return the desired metric
        return results[metric]

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
def build_influxdb_query(dbq, tags=None, all_time=False, debug=False):
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
    all_time : `bool`, optional
        Get all matches, regardless of the timestamp [Default: False]
    debug : `bool`, optional
       Print debugging statements? [Default: False]

    Returns
    -------
    `str`
        The InfluxDB-compliant query string
    """
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
    if not all_time:
        try:
            dtime = int(dbq.rangehours)
        except ValueError:
            print(f"Can't convert {dbq.rangehours} to int... using ~1.5yrs")
            dtime = 13000
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

    # Return the completed string
    return query


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
    `dict`
        Packet dictionary containing the information to be inserted into the
        InfluxDB database.
    """
    # Convert the AstroPy Table Row into a dict by adding colnames
    row_as_dict = dict(zip(table_row.colnames, table_row))

    # We want the database timestamp to be that of the image DATEOBS,
    #  not the current time.  Create a datetime() object from `dateobs`.
    timestamp = utils.scrub_isot_dateobs(row_as_dict.pop("dateobs"))

    # Build the tags from information in the table Row
    tags = {
        "instrument": row_as_dict.pop("instrument").lower(),
        "frametype": row_as_dict.pop("frametype").lower(),
        "filter": row_as_dict.pop("filter"),
        "binning": row_as_dict.pop("binning"),
        "numamp": row_as_dict.pop("numamp"),
        "ampid": row_as_dict.pop("ampid"),
        "cropborder": row_as_dict.pop("cropsize"),
    }

    # Strip off the filename, as it can be reconstructed from obserno
    row_as_dict.pop("filename")

    # Remove any NaN / inf fields
    row_as_dict = {k: v for k, v in row_as_dict.items() if np.isfinite(v)}

    # Build & return the packet as a dictionary with the proper InfluxDB keys
    return {
        "measurement": measure,
        "tags": tags,
        "time": timestamp,
        "fields": row_as_dict,
    }


# Testing ====================================================================#
if __name__ == "__main__":
    from roz import graphics_maker as gm

    hist = HistoricalData("lmi", "bias", binning="3x3", debug=True)
    hist.perform_query()
    # hist.results.pprint()
    print(f"Table column names:\n{hist.results.colnames}")

    print("")
    print(f"Instruments: {hist.instruments}")
    print(f"Frametypes: {hist.frametypes}")
    print(f"Filters: {hist.filters}")
    print(f"Binnings: {hist.binnings}")
    # print(hist.numamps)
    # print(hist.ampids)
    # print(hist.cropborders)
    print("")
    for metrc in hist.results.colnames:
        notjunk = np.issubdtype(hist.results[metrc].dtype, np.floating)
        if notjunk:
            mu = hist.metric_mean(metrc)
            sig = hist.metric_stddev(metrc)
            print(
                f"For {len(hist.results)} LMI bias frames, the {metrc} is: {mu:.2f} ± {sig:.2f}"
            )
        # print([d.isoformat(timespec="minutes") for d in hist.results["timestamp"].tolist()])

    gm.plot_lmi_bias_temp(
        hist.results["crop_avg"], hist.results["mnttemp"], binning=hist.binnings
    )
