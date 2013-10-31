Visualize Wealth
****************

The goal of these modules is to be able to construct 
portfolio performance and perform comprehensive
analytics


Constructing Portfolios
========================

In general there are three ways to construct a portfolio using 
this module:

1. Provide trades given tickers, buy / sell dates, and prices

2. Provide a weighting scheme with dates along the first column with  
   column titles as tickers and percentage allocations to each ticker
   for a given date for values.  For example:

   +---------+---------+---------+---------+
   | Date    | ticker 1|ticker 2 |ticker 3 |
   +=========+=========+=========+=========+
   |1/1/2000 | .25     | .30     | .45     |
   +---------+---------+---------+---------+
   |6/5/2003 |  .10    | .65     | .25     |
   +---------+---------+---------+---------+


3. Provide an inial allocation scheme, similar to the one provided
   above, and then set some rebalancing frequency, say, 'weekly', 
   'monthly', 'quarterly', and 'yearly.'

The main aggregation functions that perform portfolio construction
begin with ``portfolio_from`` and are then succeeded by either:

a. ``from_blotter``: corresponding to bullet point 1

b. ``from_weight_file``: corresponding to bullet point 2

c. ``from_initial_weights``: corresponding to bullet point 3
  
    
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
