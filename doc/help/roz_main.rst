.. code-block:: console

    $ roz -h
    usage: roz [-h] [-a] [-r RAM] [--scheme SCHEME] [--sigma SIGMA] [--no_cold] [--use_problems] [--skip_db] [--no_confluence] [--nocal]
            [--silence_empty_alerts] [--sci]
            dir [dir ...]

    Lowell Observatory quality assurance of instrument data

    positional arguments:
    dir                   The directory or directories on which to run Roz

    optional arguments:
    -h, --help            show this help message and exit
    -a, --all_time        Use all historical data, regardless of timestamp (disregard conf file)
    -r RAM, --ram RAM     Gigabytes of RAM to use for image combining (default: 8)
    --scheme SCHEME       Validation scheme to use [*SIMPLE*, NONE]
    --sigma SIGMA         Sigma threshold for reporting problematic frames for SIMPLE validation (default: 3.0)
    --no_cold             Do not copy to cold storage
    --use_problems        Use historical data marked as problem in the analysis
    --skip_db             Skip writing to the InfluxDB
    --no_confluence       Do not update Confluence
    --nocal               Do not process the calibration frames
    --silence_empty_alerts
                            Do not send Slack alerts on empty directories
    --sci                 Process the science frames, too? (Not yet implemented)