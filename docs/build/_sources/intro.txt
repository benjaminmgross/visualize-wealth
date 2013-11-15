Visualize Wealth
****************

There are few tools available to the public that allow them to
effectively evaluate portfolio performance, analyze historical
results, and isolate decisions made within the portfolio that
contributed to the overall benefit / detriment of the investor.  That
is the aim of this library.


**Optimization of Functions:**

In all cases, functions have been optimized to ensure the most
expeditious calculation times possible, so most of the lag time the
user experiences will be due to the ``Yahoo!`` API Calls. 


**Examples:**

Fairly comprehensive examples can be found in the ``README`` section
of this documentation, which also resides on the `Splash Page
<http://www.github.com/benjaminmgross/grind-my-ass-ets>`_ of this
project's GitHub Repository.


**Testing**:

The folder under ``visualize_wealth/tests/`` has fairly significant
tests that illustrate the calculations of:

1. Portoflio Construction (all methods)

2. The portfolio statistic calculations use in the ``analyze.py``
module

Constructing Portfolios
========================

In general there are three ways to construct a portfolio using 
this module:

1. Provide trades given tickers, buy / sell dates, and prices

2. Provide a weighting scheme with dates along the first column with  
   column titles as tickers and percentage allocations to each ticker
   for a given date for values.  For example:

3. Provide an inial allocation scheme representing the static weights
   to rebalance to at some given interval, and then define the
   rebalancing interval as  'weekly',   'monthly', 'quarterly', and 'yearly.'

In the cases where prices are not available to the investor, helper
functions for all of the construction methods are available that use
`Yahoo!'s API <http://www.finance.yahoo.com>`_ to pull down relevant
price series to be incorprated into the portfolio series calculation. 

    
.. automodule:: visualize_wealth.construct_portfolio
   :members:

Analyzing Portfolio Peformance:
===============================

In general, there's a myriad of statistics and analysis that 
can be done to analyze portfolio performance, but in general, they
can be grouped into:

1. **Absolute Statistics:** Statistics that can be calculated without
   another portfolio to compare against

2. **Relative** -or- **Benchmark Statistics:** Those statistics that
    can only be calculated using some comparative series or benchmark

.. automodule:: visualize_wealth.analyze
   :members:


Add this to your ``conf.py`` file::

	s = os.path.abspath('../')
	sys.path.append(s)
