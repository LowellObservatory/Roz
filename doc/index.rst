.. Roz documentation master file, created by
   sphinx-quickstart on Fri May 27 15:35:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org/

.. |stars| image:: https://img.shields.io/github/stars/fpavogt/fcmaker.svg?style=social&label=Stars
   :target: https://github.com/LowellObservatory/Roz

.. |watch| image:: https://img.shields.io/github/watchers/fpavogt/fcmaker.svg?style=social&label=Watch
   :target: https://github.com/LowellObservatory/Roz

Roz |stars| |watch|
===================
|astropy|

**Version**: |version|

.. toctree::
   :maxdepth: 2
   :caption: Contents:

----

Roz is designed to analyze long-term trends in astronomical data frames.  It
grew out of the need to measure consistency in flat field images taken with the
Lowell Discovery Telescope (LDT) in Happy Jack, AZ.  It has since expanded its
pervue.  The package is named after the meticulous records keeper in the Pixar movie
`Monsters, Inc.`, and we may yet discover that it has a secret identity, too.

The exact structure of Roz is still undergoing evolution as new data types are
added to the package's abilities.


----

To run, the main driver may be accessed with the script ``roz``:

.. include:: help/roz_main.rst




.. toctree::
   :caption: For developers
   :maxdepth: 1

   Roz API <api/modules>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
