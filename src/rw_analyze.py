#!/usr/bin/env python
# encoding: utf-8
"""
analyze.py

Created by Benjamin Gross in 2013
"""
import argparse
import pandas
import numpy
import pandas.io.data
import datetime
import collections
import scipy.optimize as sopt
import urllib2

PIVOT_COL = 'Close'

def format_blotter(blotter_file):
    """
    INPUTS:
    ------
    blotter_file: assumes the columns of the blotter file are 'Buy/Sell'
    """
    blot = pandas.DataFrame.from_csv(blotter_file)
    blot['Buy/Sell'] = map(str.strip, blot['Buy/Sell'])
    blot.loc[blot['Buy/Sell'] == 'Sell', 'Shares'] = -blot.loc[blot['Buy/Sell'] == 'Sell', 'Shares']
    
    return blot

def construct_price_frame(ticker, start_date, end_date=None):
    """
    ticker: The ticker
    start_date: a datetime.datetime object to begin the price series
    end_date: a datetime.datetime object to end the price series
    """
    reader = pandas.io.data.DataReader

    if end_date == None:
        end = datetime.datetime.today()
    else:
        end = end_date

    #construct the dividend data series
    b_str = 'http://ichart.finance.yahoo.com/table.csv?s='

    if end_date == None:
        end_date = datetime.datetime.today()

    a = '&a=' + str(start_date.month)
    b = '&b=' + str(start_date.day)
    c = '&c=' + str(start_date.year)
    d = '&d=' + str(end.month)
    e = '&e=' + str(end.day)
    f = '&f=' + str(end.year)
    tail = '&g=v&ignore=.csv'
    url = b_str + ticker + a + b + c + d + e + f + tail
    socket = urllib2.urlopen(url)
    div_df = pandas.io.parsers.read_csv(socket, index_col = 0)
    
    price_df = reader(ticker, data_source = 'yahoo', 
                      start = start_date, end = end_date)

    return price_df.join(div_df).fillna(0.0)


def blotter_to_price_series(blotter_df, price_df):
    """
    INPUTS:
    -------
    blotter_df: DataFrame of a single ticker with index of dates, column of 'Shares'
                Indicating the purchase / sale (pos / neg) on a given day

    price_df: DataFrame containing columns of 'Close' and 'Adj Close'
    """
    blotter_df.sort(inplace = True)
    #make sure all dates in the blotter file are also in the price file
    #consider, if those dates aren't in price frame, assign the "closest date" value
    assert numpy.all(map(lambda x: numpy.any(price_df.index == x), 
                         blotter_df.index)), "Buy/Sell Dates not in Price File"

    #find the first date
    d_0 = numpy.where(price_df.index == blotter_df.index[0])[0][0]
    ind = price_df.index[d_0:]
    cols = ['bs','adj_fac', 'cum_shares', 'adj_qty', 'contr_with', 'cum_inv',
            'asset_value']

    #preallocate the asset value frame
    ret_df = pandas.DataFrame(numpy.zeros([len(ind), len(cols)]), index = ind,
                              columns = cols)
    
    ret_df['adj_fac'] = price_df['Close'].div(price_df['Adj Close'])[d_0:]

    #The formulation only uses cumulative shares *after* the first purchase
    tmp = blotter_df[1:]
    ret_df.loc[tmp.index, 'bs'] = tmp['Shares']
    ret_df['cum_shares'] = ret_df['bs'].cumsum()
    k = blotter_df.ix[0, 'Shares']*ret_df.ix[0, 'adj_fac']
    ret_df['adj_qty'] = k/ret_df['adj_fac'] + ret_df['cum_shares']
    #import pdb
    #pdb.set_trace()
    ret_df.ix[0, 'contr_with'] = ret_df.ix[0, 'adj_qty']*price_df.ix[d_0, 'Close']
    ret_df.ix[1:,'contr_with'] = ret_df.ix[1:,'bs'].mul(price_df.ix[d_0+1:,'Close'])
    ret_df['cum_inv'] = ret_df['contr_with'].cumsum()
    ret_df['asset_value'] = ret_df['adj_qty'].mul(price_df.ix[d_0:, 'Close'])

    return ret_df
	
def make_prices_and_shares(blotter_file, pivot_col = PIVOT_COL):
	"""
	Run the analysis given a blotter file	
	"""
	
	blotter = pandas.DataFrame.from_csv(blotter_file + '.csv')
	buys = blotter.ix[blotter['Buy/Sell'] == 'Buy', :]
	today = datetime.datetime.today()
	index = pandas.DatetimeIndex(pandas.bdate_range(start = min(buys.index), end = today))
	#index = pandas.DatetimeIndex(start = min(buys.index), end = today, freq = 'd')
	d = collections.OrderedDict().fromkeys(buys.Ticker, [])
	
	for i, ticker in enumerate(buys.Ticker):
		date = buys.index[i]
		print ticker
		temp = pandas.io.data.DataReader(name = ticker, data_source = 'yahoo',
                                                 start = min(buys.index))
		d[ticker] = temp[PIVOT_COL]

	prices_frame = pandas.DataFrame(d, index = index)
	#first_full_row = find_first_full_row(prices_frame)
	prices_frame = prices_frame.fillna(method = 'pad')
	
	share_frame = pandas.DataFrame(numpy.zeros(prices_frame.shape), 
                                       columns = prices_frame.columns, 
                                       index = prices_frame.index)
	for i, ticker in enumerate(blotter.Ticker):
		if blotter['Buy/Sell'][i] == 'Buy':
			#print "bought " + datetime.datetime.strftime(blotter.index[i], format = '%m-%d-%Y')
			share_frame[ticker][blotter.index[i]] = blotter.Shares[i]
		elif blotter['Buy/Sell'][i] == 'Sell':
			#print "sold " + datetime.datetime.strftime(blotter.index[i], format = '%m-%d-%Y')
			share_frame[ticker][blotter.index[i]] = -1.*blotter.Shares[i]
		else:
			print "share insert failed"
	
	share_frame = share_frame.cumsum(axis = 0)
	prices_frame = prices_frame.dropna(how = 'all')
	share_frame = share_frame.ix[prices_frame.index, :]
	return prices_frame, share_frame
	
def make_portfolio(blotter_file, pivot_col = PIVOT_COL):
	shares, prices = make_prices_and_shares(blotter_file = blotter_file , pivot_col = PIVOT_COL)
	portfolio_price = numpy.multiply(prices, shares).sum(axis = 1)
	
	#benchmark_assets = 
	#bfb_frame = get_bfb((prices_frame*share_frame).sum(axis = 1), list_of_ticks)
	
	return portfolio_price

def best_fitting_benchmark_weights(portfolio_returns, asset_returns):
	"""
	INPUTS:
	--------
	* portfolio_returns: m x 1 pandas.TimeSeries of Linear Portfolio Returns
	* asset_returns: m x n pandas.DataFrame of Linear Asset Returns

	OUTPUTS:
	--------
	* a pandas.TimeSeries of nonnegative weights for each asset
	  such that the r_squared from the regression of Y ~ Xw + e is maximized

	"""
	def _r_squared(weights):
		"""
		Potentially incorporate the degrees of freedom in this function to improve estimate
		"""
		estimate = asset_returns.dot(weights)
		sse = ((estimate - portfolio_returns)**2).sum()
		sst = ((portfolio_returns - portfolio_returns.mean())**2).sum()
		return 1 - sse/sst

	def _obj_fun(weights):
		"""
		To maximize the r_squared, minimize the negative of r_squared
		"""  
		return - _r_squared(weights)


	assert portfolio_returns.shape[0] == asset_returns.shape[0], "inputs must be same shape"

	num_assets = asset_returns.shape[1]
	guess = numpy.zeros(num_assets,)
	#sum_to_one = lambda x: numpy.dot(numpy.tile(x, num_assets,), numpy.ones(num_assets,)) - 1

	#ensure the boundaries of the function are (0, 1)
	ge_zero = [(0,1) for i in numpy.arange(num_assets)]

	#optimize to maximize r-squared, using the 'TNC' method (that uses the boundary functionality)
	opt = sopt.minimize(_obj_fun, x0 = guess, method = 'TNC', bounds = ge_zero)

	normed = opt.x*(1./numpy.sum(opt.x))

	return pandas.TimeSeries(normed, index = asset_returns.columns)

def generate_data(blotter_file, pivot_col = PIVOT_COL):
	price_frame = make_price_frame(blotter_file, pivot_col)
	share_frame = make_shares_frame(price_frame, price_index)
	

if __name__ == '__main__':
	
	usage = sys.argv[0] + "file_loc"
	description = "Creates a portfolio when pointed to a blotter file in format <Date> <Buy/Sell> <Ticker> <Shares>"
	parser = argparse.ArgumentParser(description = description, usage = usage)
	parser.add_argument('file_loc', nargs = 1, type = str, help = 'describe input 1')
	#parser.add_argument('name_2', nargs = '+', type = int, help = "describe input 2")

	args = parser.parse_args()
	
	pull_prices(input_1 = args.file_loc[0])
