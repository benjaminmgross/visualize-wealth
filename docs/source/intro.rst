.. _intro:

*****************
Modules Explained
*****************

**Optimization of Functions:**

In all cases, functions have been optimized to ensure the most
expeditious calculation times possible, so most of the lag time the
user experiences will be due to the ``Yahoo!`` API Calls. 


**Examples:**

Fairly comprehensive examples can be found in the ``README`` section
of this documentation, which also resides on the `Splash Page
<http://www.github.com/benjaminmgross/wealth-viz>`_ of this
project's GitHub Repository.


**Testing**:

The folder under ``visualize_wealth/tests/`` has fairly significant
tests that illustrate the calculations of:

1. Portoflio construction methods

2. The portfolio statistic calculations use in the :mod:`analyze` module

All the files are Microsoft Excel files (extensions `.xslx`), so
seeing cell calculations and the steps to arrive at a final value
should be pretty straightforward.

.. _constructing-portfolios:

Constructing portfolios
=======================

In general I've provided for three ways to construct a portfolio using 
this package:

1. **The Blotter Method:** Provide trades given tickers, buy / sell
    dates, and prices.  In general, this method most closely approximates
    how you would reconstruct your own portfolio performance (if you had
    to).  You would essentially take trade confirmations of buys and
    sells, calcluate dividends and splits, and then arrive at an
    aggregate  portfolio path.  Specific examples can be found under 
    `blotter method examples <./readme.html#the-blotter-method>`_.

2. **The Weight Allocation Method:** Provide a weighting scheme with
    dates along the first column with column titles as tickers and
    percentage allocations to each ticker for a given date for values.
    Specific examples can be found under
    `weight allocation examples
    <./readme.html#the-weight-allocation-method>`_.

3. **The Initial Allocation Method:** with specific Rebalancing
    Periods provide an initial allocation scheme representing the
    static weights to rebalance to at some given interval, and then
    define the rebalancing interval as  'weekly',   'monthly',
    'quarterly', and 'yearly.' Specific examples can found under
    `initial allocation examples
    <./readme.html#the-initial-allocation-rebalancing-method>`_.

Much more detailed examples of these three methods, as well as
specific examples of code that would leverage this package can be
found in the `Portfolio Construction Examples <./readme.html#portfolio-construction-examples>`_.


In the cases where prices are not available to the investor, helper
functions for all of the construction methods are available that use
`Yahoo!'s API <http://www.finance.yahoo.com>`_ to pull down relevant
price series to be incorporated into the portfolio series calculation. 

.. _analyzing-portfolio-performance:

Analyzing portfolio performance
===============================

In general, there's a myriad of statistics and analysis that 
can be done to analyze portfolio performance.  I don't go really
deeply into the uses of any of these statistics, but any number of
them can be "Google-ed", if there aren't live links already provided.

Fairly extensive formulas are provided for each of the performance statistics
inside of the :mod:`analyze`.  Formulas were inserted using the
`MathJax <http://www.mathjax.org/>`_ rendering capability that `Sphinx
<http://sphinx-doc.org/>`_ provides.  


.. _construct-portfolio-documentation:

``construct_portfolio.py`` documentation
========================================    

Full documentation for each of the functions of :mod:`construct-portfolio`

.. automodule:: visualize_wealth.construct_portfolio
   :members:

.. _analyze-documentation:

``analyze.py`` documentation
============================

Full documentation for each of the functions of :mod:`analyze`

.. automodule:: visualize_wealth.analyze
   :members:


Sphinx Customizations
=====================

The documentation for this module was created using `Sphinx
<http://sphinx-doc.org/>`_. I keep a couple of commands here that I
use when re-committing to `GitHub <http://www.github.com>`_, or
regenerating the documentation.  It serves
as reference for myself, but in case other people might find it useful
I've posted that as well

.. _convert-markdown-to-rst:

Convert `README.md` to `.rst` format
------------------------------------

I use `Pandoc <http://johnmacfarlane.net/pandoc/>`_ to convert my
`README <./readme.html>`_ (that's in Markdown format) into ``.rst`` format (that
can be interepreted by Sphinx).

.. code:: bash
   
   $ cd visualize_wealth
   $ pandoc README.md -f markdown -t rst -o docs/source/readme.rst


.. _build-sphinx-documentation:

Build Sphinx Documentation
--------------------------

.. code:: bash
    
    #rebuild the package
    $ cd visualize_wealth
    $ python setup.py install

    #rebuild the documentation
    $ sphinx-build -b html docs/source/ docs/build/
 
.. _sphinx-customizations:

Sphinx Customizations in ``conf.py``
------------------------------------

Current customizations I use in order to make the docs look like they
currently do

.. code:: python

    #use the sphinxdoc theme
    html_theme = 'sphinxdoc'

    #enable todos
    todo_include_todos = True
    extensions = ['sphinx.ext.autodoc',..., 'sphinx.ext.todo']
    
