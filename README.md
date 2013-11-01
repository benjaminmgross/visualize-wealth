#README.md

##License

This program is free software and is distrubuted under the
[GNU General Public License version 3](http://www.gnu.org/licenses/quick-guide-gplv3.html)("GNU
GPL v3")

&copy; Benjamin M. Gross 2013


##Dependencies

`pandas`: extensively used (`numpy` and `scipy` obviously, but `pandas` depends on those)
`urllib2`: for Yahoo! API calls to append price `DataFrame`s with
Dividends


##Installation
Working on it...


##General Summary & Examples

Portfolios can (generally) be constructed in one of three ways:

1. **The blotter method:** In finance, a spreadsheet of "buys/sells", "Prices", "Dates" etc. is called a "trade blotter."  This also would be the easiest way for an investor to actually analyze the past performance of her portfolio, because trade confirmations provide this exact data.
   
   This method is most effectively achieved by providing an Excel / `.csv` file with the following format:
   
   | Date | Buy / Sell | Price |Ticker	|
   |:-----|:-----------|:------|:-----	|
   |9/4/01| 50 		    | 123.45| EFA  	|
   |5/5/03| 65         | 107.71| EEM	|
   |6/6/03|-15         | 118.85| EEM 	|
   
  
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
		In [9]: port_panel = vwcp.portfolio_from_blotter(blotter)
	
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
		 u'Dividends',u'Splits', u'contr_withdrawal', u'cum_investment', u'cum_shares'], 
		 dtype=object)

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

2. The Weight Allocation Method

	
	

##ToDo List:

* Best broad asset classes to determine "best fit portfolio"

| Ticker | Name                         | Price Data Begins | 
|:-------|:-----------------------------|:------------------|
|VTSMX   | Vanguard Total Stock Market  | 6/20/1996         |
|VBMFX   |Vanguard Total Bond Market    | 6/4/1990          |
|VGTSX   |Vanguard Total Intl Stock     | 6/28/1996         |


* Put together `General Summary & Examples`

