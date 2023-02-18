========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/extq/badge/?style=flat
    :target: https://readthedocs.org/projects/extq
    :alt: Documentation Status

.. |version| image:: https://img.shields.io/pypi/v/extq.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/extq

.. |wheel| image:: https://img.shields.io/pypi/wheel/extq.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/extq

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/extq.svg
    :alt: Supported versions
    :target: https://pypi.org/project/extq

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/extq.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/extq

.. |commits-since| image:: https://img.shields.io/github/commits-since/chatipat/extq/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/chatipat/extq/compare/v0.0.0...master



.. end-badges

Dynamical analysis of trajectory data.

* Free software: MIT license

Installation
============

::

    pip install extq

You can also install the in-development version with::

    pip install https://github.com/chatipat/extq/archive/master.zip


Documentation
=============


https://extq.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
