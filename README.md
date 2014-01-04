#README.md

A library built in Python to construct, backtest, analyze, and evaluate portfolios and their benchmarks, with comprehensive documentation and manual calculations to illustrate all underlying methodologies and statistics.

##License

This program is free software and is distrubuted under the
[GNU General Public License version 3](http://www.gnu.org/licenses/quick-guide-gplv3.html) ("GNU GPL v3")

&copy; Benjamin M. Gross 2013


##Dependencies

`pandas`: extensively used (`numpy` and `scipy` obviously, but
`pandas` depends on those)

`urllib2`: for Yahoo! API calls to append price `DataFrame`s with
Dividends


##Installation

To install the `visualize_wealth` modules onto your computer, go into
your desired folder of choice (say `Downloads`), and:

1. Clone the repository

	    $ cd ~/Downloads
	    $ git clone https://github.com/benjaminmgross/grind-my-ass-ets

2. `cd` into the `grind-my-ass-ets` directory

        $ cd grind-my-ass-ets

3. Install the package

        $python setup.py install

4. Check your install.  From anywhere on your machine, be able to open
   `iPython` and import the library, for example:

	    $ cd ~/
	    $ ipython

        IPython 1.1.0 -- An enhanced Interactive Python.
        ?         -> Introduction and overview of IPython's features.
        %quickref -> Quick reference.
        help      -> Python's own help system.
        object?   -> Details about 'object', use 'object??' for extra details.
	
        In [1]: import visualize_wealth

**"Ligget Se!"**

##Documentation

The `README.md` file has fairly good examples, but I've gone to great lengths to autogenerate documentation for the code using [Sphinx](http://sphinx-doc.org/).  Therefore, aside from the docstrings, when you `git clone` the repository, you can `cd` into:

	$ cd visualize_wealth/docs/build/

and find full `.html` browseable code documentation (that's pretty f*cking beautiful... if I do say so my damn self) with live links, function explanations (that also have live links to their respective definition on the web), etc.

Also I've created an Excel spreadsheet that illustrates almost all of the `analyze.py` portfolio statistic calculations.  That spreadsheet can found in:

	visualize_wealth > tests > test_analyze.xlsx

In fact, the unit testing for the `analyze.py` portfolio statistics tests the python calculations against this same excel spreadsheet, so you can really get into the guts of how these things are calculated.


##[Portfolio Construction Examples](portfolio-construction-examples)

Portfolios can (generally) be constructed in one of three ways:

1. The Blotter Method
2. Weight Allocation Method
3. Initial Allocation with specific Rebalancing Period Method

### 1. [The Blotter Method](blotter-method-examples)

**The blotter method:** In finance, a spreadsheet of "buys/sells", "Prices", "Dates" etc. is called a "trade blotter."  This also would be the easiest way for an investor to actually analyze the past performance of her portfolio, because trade confirmations provide this exact data.
   
This method is most effectively achieved by providing an Excel / `.csv` file with the following format:

| Date   |Buy / Sell| Price |Ticker|
|:-------|:---------|:------|:-----|
|9/4/2001| 50       | 123.45| EFA  |
|5/5/2003| 65       | 107.71| EEM	|
|6/6/2003|-15       | 118.85| EEM 	|

where "Buys" can be distinguished from "Sells" because buys are positive (+) and sells are negative (-).

For example, let's say I wanted to generate a random portfolio containing the following tickers and respective asset classes, using the `generate_random_portfolio_blotter` method

|Ticker  | Description              | Asset Class        | Price Start|
|:-------|:-------------------------|:-------------------|:-----------|
| IWB    | iShares Russell 1000     | US Equity          | 5/19/2000  |
| IWR    | iShares Russell Midcap   | US Equity          | 8/27/2001  |
| IWM    | iShares Russell 2000     | US Equity          | 5/26/2000  |
| EFA    | iShares EAFE             | Foreign Dev Equity | 8/27/2001  |
| EEM    | iShares EAFE EM          | Foreign EM Equity  | 4/15/2003  |
| TIP    | iShares TIPS             | Fixed Income       | 12/5/2003  |
| TLT    | iShares LT Treasuries    | Fixed Income       | 7/31/2002  |
| IEF    | iShares MT Treasuries    | Fixed Income       | 7/31/2002  |
| SHY    | iShares ST Treasuries    | Fixed Income       | 7/31/2002  |
| LQD    | iShares Inv Grade        | Fixed Income       | 7/31/2002  |
| IYR    | iShares Real Estate      | Alternative        | 6/19/2000  |
| GLD    | iShares Gold Index       | Alternative        | 11/18/2004 |
| GSG    | iShares Commodities      | Alternative        | 7/21/2006  |

I could construct a portfolio of random trades (i.e. the "blotter method"), say 20 trades for each asset, by executing the following:
	
	        #import the modules
	In [5]: import vizualize_wealth.construct_portfolio as vwcp

	In [6]: ticks = ['IWB','IWR','IWM','EFA','EEM','TIP','TLT','IEF',
	                 'SHY','LQD','IYR','GLD','GSG']		
	In [7]: num_trades = 20
	
	        #construct the random trade blotter
	In [8]: blotter = vwcp.generate_random_portfolio_blotter(ticks, num_trades)
	
	        #construct the portfolio panel
	In [9]: port_panel = vwcp.panel_from_blotter(blotter)
	
Now I have a `pandas.Panel`. Before we constuct the cumulative portfolio values, let's examine the dimensions of the panel (which are generally the same for all construction methods, although the columns of the `minor_axis` are different because the methods call for different optimized calculations) with the following dimensions:

	#tickers are `panel.items`
	In [10]: port_panel.items
	Out[10]: Index([u'EEM', u'EFA', u'GLD', u'GSG', u'IEF', u'IWB', u'IWM', u'IWR', 
				u'IYR', u'LQD', u'SHY', u'TIP', u'TLT'], dtype=object)

	#dates are along the `panel.major_axis`
	In [12]: port_panel.major_axis
	Out[12]: 
	<class 'pandas.tseries.index.DatetimeIndex'>
	[2000-07-06 00:00:00, ..., 2013-10-30 00:00:00]
	Length: 3351, Freq: None, Timezone: None

	#price data, cumulative investment, dividends, and split ratios are `panel.minor_axis`
	In [13]: port_panel.minor_axis
	Out[13]: Index([u'Open', u'High', u'Low', u'Close', u'Volume', u'Adj Close',
		u'Dividends',u'Splits', u'contr_withdrawal', u'cum_investment', 
		u'cum_shares'], dtype=object)

There is a lot of information to be gleaned from this data object, but the most common goal would be to convert this `pandas.Panel` to a Portfolio `pandas.DataFrame` with columns `['Open', 'Close']`, so it can be compared against other assets or combination of assets.  In this case, use `pfp_from_blotter`(which stands for "portfolio_from_panel" + portfolio construction method [i.e. blotter, weights, or initial allocaiton] which in this case was "the blotter method").
	
		#construct_the portfolio series
		In [14]: port_df = vwcp.pfp_from_blotter(panel, 1000.)
	
		In [117]: port_df.head()
		Out[117]: 
        	          Close         Open
		Date                                
		2000-07-06  1000.000000   988.744754
		2000-07-07  1006.295307  1000.190767
		2000-07-10  1012.876765  1005.723006
		2000-07-11  1011.636780  1011.064479
		2000-07-12  1031.953453  1016.978253

###2. [The Weight Allocation Method](weight-allocation-method-examples)

A commonplace way to test portoflio management strategies using a
group of underlying assets is to construct aggregate portofolio
performance, given a specified weighting allocation to specific assets
on specified dates.  Specifically, those (often times) percentage
allocations represent a recommended allocation at some point in time,
based on some "view" derived from either the output of a model or some qualitative
analysis.  Therefore, having an engine that is capable of taking in a weighting file (say, a `.csv`) with the following format:

|Date    | Ticker 1  | Ticker 2  | Ticker 3 | Ticker 4 |
|:-------|:---------:|:---------:|:--------:|:--------:|
|1/1/2002| 5%        | 20%       | 30%      | 45%      |
|6/3/2003| 40%       | 10%       | 40%      | 10%      |
|7/8/2003| 25%       | 25%       | 25%      | 25%      |

and turning the above allocation file into a cumulative portfolio
value that can then be analyzed and compared (both in isolation and
relative to specified benchmarks) is highly valuable in the process of
portfolio strategy creation.

A quick example of a weighting allocation file can be found in the
Excel File `visualize_wealth/tests/panel from weight file test.xlsx`,
where the tab `rebal_weights` represents one of these specific
weighting files.

To construct a portfolio of using the **Weighting Allocation Method**,
a process such as the following would be carried out.

	#import the library
	import visualize_wealth.construct_portfolio as vwcp

If we didn't have the prices already, there's a function for that

	#fetch the prices and put them into a pandas.Panel
    price_panel = vwcp.fetch_data_for_weight_allocation_method(weight_df)

	#construct the panel that will go into the portfolio constructor

	 port_panel = vwcp.panel_from_weight_file(weight_df, price_panel,
	     start_value = 1000.)

Construct the `pandas.DataFrame` for the portfolio, starting at
`start_value` of 1000 with columns `['Open', Close']`

	portfolio = vwcp.pfp_from_weight_file(port_panel)

Now a portfolio with `index` of daily values and columns
`['Open', 'Close']` has been created upon which analytics and
performance analysis can be done.

### 3. [The Initial Allocation & Rebalancing Method](initial-allocation-method-examples)

The standard method of portoflio construction that pervades in many
circles to this day is static allocation with a given interval of
rebalancing. For instance, if I wanted to implement Oppenheimers'
[The New 60/40](https://www.oppenheimerfunds.com/digitalAssets/Discover-the-New-60-40-43f7f642-e0aa-40d9-a3fc-00f31be5a4fa.pdf)
static portfolio, rebalancing on a yearly interval, my weighting
scheme would be as follows:

| Ticker | Name                     | Asset Class        | Allocation |
|:-------|:-------------------------|:-------------------|:-----------|
| IWB    | iShares Russell 1000     | US Equity          |        15% |
| IWR    | iShares Russell Midcap   | US Equity          |       7.5% |
| IWM    | iShares Russell 2000     | US Equity          |       7.5% |
| SCZ    | iShares EAFE Small Cap   | Foreign Dev Equity |       7.5% |
| EFA    | iShares EAFE             | Foreign Dev Equity |      12.5% |
| EEM    | iShares EAFE EM          | Foreign EM Equity  |        10% |
| TIP    | iShares TIPS             | Fixed Income       |         5% |
| TLT    | iShares LT Treasuries    | Fixed Income       |       2.5% |
| IEF    | iShares MT Treasuries    | Fixed Income       |       2.5% |
| SHY    | iShares ST Treasuries    | Fixed Income       |         5% |
| HYG    | iShares High Yield       | Fixed Income       |       2.5% |
| LQD    | iShares Inv Grade        | Fixed Income       |       2.5% |
| PCY    | PowerShares EM Sovereign | Fixed Income       |         2% |
| BWX    | SPDR intl Treasuries     | Fixed Income       |         2% |
| MBB    | iShares MBS              | Fixed Income       |         1% |
| PFF    | iShares Preferred Equity | Alternative        |       2.5% |
| IYR    | iShares Real Estate      | Alternative        |         5% |
| GLD    | iShares Gold Index       | Alternative        |       2.5% |
| GSG    | iShares Commodities      | Alternative        |         5% |

To implement such a weighting scheme, we can use the same worksheet
`visualize_wealth/tests/panel from weight file test.xlsx`, and the
tab.  `static_allocation`.  Note there is only a single row of
weights, as this will be the "static allocation" to be rebalanced to
at some given interval.

    #import the construct_portfolio library
	import visualize_wealth.construct_portfolio as vwcp

Let's use the `static_allocation` provided in the `panel from weight
file.xlsx` workbook

    f = pandas.ExcelFile('tests/panel from weight file test.xlsx')
	static_alloc = f.parse('static_allocation', index_col = 0,
	    header_col = 0)

Again, assume we don't have the prices and need to donwload them, use
the `fetch_data_for_initial_allocation_method`

    price-panel = vwcp.fetch_data_for_initial_allocation_method(static_alloc)

Construct the `panel` for the portoflio while determining the desired
rebalance frequency

    panel =	vwcp.panel_from_initial_weights(weight_series = static_alloc,
		static_alloc, price_panel = price_panel, rebal_frequency = 'quarterly')


Construct the final portfolio with columns `['Open', 'Close']`

    portfolio = vwcp.pfp_from_weight_file(panel)

Take a look at the portfolio series:

    In [10:] portfolio.head()
	Out[11:]

	            Close        Open
	Date
	2007-12-12  1000.000000  1007.885932
	2007-12-13   991.329125   990.717915
	2007-12-14   978.157960   983.057829
	2007-12-17   961.705069   969.797167
	2007-12-18   969.794966   972.365687


##ToDo List:

* occassionally `generate_random_asset_path` will return with an Assertion Error

* Add the following statistics to the `analyze.py` library:
   - ~~Absolute Alpha: $$R_p - R_b$$~~
   - Treynor ratio: $$\\textrm{T.R.}\\triangleq \\frac{r_i - r_f}{\\beta{i}}$$
   - Information Ratio or Appraisal Ratio: $$\\textrm{I.R.} \\triangleq \\frac{\\alpha}{\\omega}$$, or absolute alpha / tracking error.  Other formulations include Jensens's Alpha / Idiosyncratic Vol
   - Up / Down beta (or [Dual-Beta](http://en.wikipedia.org/wiki/Dual-beta))

* Best broad asset classes to determine "best fit portfolio"

| Ticker | Name                         | Price Data Begins | 
|:-------|:-----------------------------|:------------------|
|VTSMX   | Vanguard Total Stock Market  | 6/20/1996         |
|VBMFX   |Vanguard Total Bond Market    | 6/4/1990          |
|VGTSX   |Vanguard Total Intl Stock     | 6/28/1996         |

* Rebuild Process:

  1. If the `README.md` file is altered, run:
  
         $ pandoc -f markdown -t rst README.md -o docs/source/readme.rst 
	  
  2. Then rebuild the Sphinx documentation
  
	     $ sphinx-build -b html docs/source/ docs/build/

