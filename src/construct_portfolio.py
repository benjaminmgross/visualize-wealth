#!/usr/bin/env python
# encoding: utf-8
"""
construct_portfolio.py
Created by Benjamin Gross in 2013
"""
import argparse
import pandas
import numpy
import pandas.io.data
import datetime
import collections
import urllib2

def format_blotter(blotter_file):
    """
    A blotter file could have positive values for both the buy and sell, 
    this function transforms Sell values that are positive to negative values
    (to be used in the portfolio constructors)
    
    INPUTS:
    ------
    blotter_file: pandas.DataFrame with at least index (dates of Buy / Sell) columns 
    = ['Buy/Sell', 'Shares'] or a string of the file location to such a formatted 
    file

    RETURNS:
    --------
    pandas.DataFrame
    """
    if isinstance(blotter_file, str):
        blot = pandas.DataFrame.from_csv(blotter_file)
    else:
        blot = blotter_file.copy()

    blot['Buy/Sell'] = map(str.strip, blot['Buy/Sell'])
    blot.loc[blot['Buy/Sell'] == 'Sell', 'Shares'] = (-blot.loc[blot['Buy/Sell'] 
                                                       == 'Sell', 'Shares'])
    
    return blot

def append_price_frame_with_dividends(ticker, start_date, end_date=None):
    """
    Given a ticker, start_date, & end_date, return a pandas.DataFrame with 
    a Dividend Columns

    INPUTS:
    -------
    ticker: string of ticker
    start_date: a datetime.datetime object to begin the price series
    end_date: a datetime.datetime object to end the price series

    RETURNS:
    -------
    price_df: DataFrame with columns ['Close', 'Adj Close', 'Dividends']
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

def calculate_splits(price_df, tol = .1):
    """
    Given a price_df of the format append_price_frame_with_dividends, return a 
    DataFrame with a split factor columns named 'Splits'
    
    INPUTS:
    -------
    price_df: DataFrame with columns ['Close', 'Adj Close', 'Dividends']
    tol: float tolerance to determine whether a split has occurred

    RETURNS:
    --------
    DataFrame: columns ['Close', 'Adj Close', 'Dividends', Splits']
    """
    div_mul = 1 - price_df['Dividends'].shift(-1).div(price_df['Close'])
    rev_cp = div_mul[::-1].cumprod()[::-1]
    rev_cp[-1] = 1.0
    est_adj = price_df['Adj Close'].div(rev_cp)
    eps = est_adj.div(price_df['Close'])
    spl_mul = eps.div(eps.shift(1))
    did_split = numpy.abs(spl_mul - 1) > tol
    splits = spl_mul[did_split]
    for date in splits.index:
        if splits[date] > 1.0:
            splits[date] = numpy.round(splits[date], 0)
        elif splits[date] < 1.0:
            splits[date] = 1./numpy.round(1./splits[date], 0)
    splits.name = 'Splits'
    return price_df.join(splits)

def blotter_to_split_adjusted_shares(blotter_series, price_df):
    """
    Given a series of dates and purchases (+) / sales (-) and a DataFrame with Close
    Adj Close, Dividends, & Splits, calculate the cumulative share balance for the
    position
    
    INPUTS:
    -------
    blotter_series: Series where index is buy/sell dates
    price_df: DataFrame with columns ['Close', 'Adj Close', 'Dividends', 'Splits']

    RETURNS:
    -------
    DataFrame containing contributions, withdrawals, price values
    """

    blotter_series = blotter_series.sort_index()
    #make sure all dates in the blotter file are also in the price file
    #consider, if those dates aren't in price frame, assign the "closest date" value
    assert numpy.all(map(lambda x: numpy.any(price_df.index == x), 
                         blotter_series.index)), "Buy/Sell Dates not in Price File"

    #now cumsum the buy/sell chunks and multiply by splits to get total shares
    bs_series = pandas.Series()
    start_dts = blotter_series.index
    end_dts = pandas.to_datetime(numpy.append(blotter_series.index[1:], 
                                              price_df.index[-1]))

    dt_chunks = zip(start_dts, end_dts)
    end = 0.

    for i, chunk in enumerate(dt_chunks):
        #print str(i) + ' of ' + str(len(dt_chunks)) + ' total'
        tmp = price_df[chunk[0]:chunk[1]][:-1]
        if chunk[1] == price_df.index[-1]:
            tmp = price_df[chunk[0]:chunk[1]]
        splits = tmp[pandas.notnull(tmp['Splits'])]
        vals = numpy.append(blotter_series['Buy/Sell'][chunk[0]] + end,
                            splits['Splits'].values)
        dts = pandas.to_datetime(numpy.append(chunk[0], splits['Splits'].index))
        tmp_series = pandas.Series(vals, index = dts)
        tmp_series = tmp_series.cumprod()
        tmp_series = tmp_series[tmp.index].ffill()
        bs_series = bs_series.append(tmp_series)
        end = bs_series[-1]

    bs_series.name = 'cum_shares'

    #construct the contributions, withdrawals, & cumulative investment

    #if a trade is missing a price, it gets assigned the closing price of that day

    no_price = blotter_series['Price'][pandas.isnull(blotter_series['Price'])]
    blotter_series.ix[no_price.index, 'Price'] = price_df.ix[no_price.index, 'Close']

    contr = blotter_series['Buy/Sell'].mul(blotter_series['Price'])
    cum_inv = contr.cumsum()
    contr = contr[price_df.index].fillna(0.0)
    cum_inv = cum_inv[price_df.index].ffill()
    res = pandas.DataFrame({'cum_shares':bs_series, 'contr_withdrawal':contr, 
                            'cum_investment':cum_inv})
    
    return price_df.join(res)
        
def construct_random_trades(split_df, num_trades):
    """
    Create random trades on random trade dates, but never allow shares to go negative
    
    INPUTS:
    -------
    split_df: DataFrame that has 'Close', 'Dividends', 'Splits'

    RETURNS:
    --------
    blotter_series: a blotter with random trades, num_trades
    """
    ind = numpy.sort(numpy.random.randint(0, len(split_df), size = num_trades))
    #This unique makes sure there aren't double trade day entries which breaks 
    #the blotter_to_split_adjusted_shares
    ind = numpy.unique(ind)
    dates = split_df.index[ind]
    trades = numpy.random.randint(-100, 100, size = len(ind))
    trades = numpy.round(trades, -1)

    while numpy.any(trades.cumsum() < 0):
        trades[numpy.argmin(trades)] *= -1.    

    return pandas.Series( trades, index = dates, name = 'Buy/Sell')

def blotter_to_cum_shares(blotter_series, ticker, start_date, end_date, tol):
    price_df = append_price_frame_with_dividends(ticker, start_date, end_date)
    split_df = calculate_splits(price_df)
    return blotter_to_split_adjusted_shares(blotter_series, split_df)

def generate_random_asset_path(ticker, start_date, num_trades):
    end_date = datetime.datetime.today()
    prices = append_price_frame_with_dividends(ticker, start_date)
    blotter = construct_random_trades(prices, num_trades)
    blotter.to_csv('../tests/' + ticker + '.csv')
    return blotter_to_cum_shares(blotter_series = blotter, ticker = ticker,
                                 start_date = start_date, end_date = end_date, 
                                 tol = .1)

def aggregate_blotter_to_portfolio(agg_blotter):
    #import pdb
    #pdb.set_trace()
    tickers = pandas.unique(agg_blotter['Ticker'])
    start_date = agg_blotter.sort_index().index[0]
    end_date = datetime.datetime.today()
    val_d = {}
    for ticker in tickers:
        blotter_series = agg_blotter[agg_blotter['Ticker'] == ticker].sort_index()
        val_d[ticker] = blotter_to_cum_shares(blotter_series, ticker,
                                              start_date, end_date, tol = .1)

    return pandas.Panel(val_d)

def generate_random_portfolio_blotter(tickers, num_trades):
    blot_d = {}
    price_d = {}
    for ticker in tickers:
        tmp = append_price_frame_with_dividends(ticker, start_date =
                                                datetime.datetime(1990, 1, 1))
        price_d[ticker] = calculate_splits(tmp)
        blot_d[ticker] = construct_random_trades(price_d[ticker], num_trades)
    ind = []
    agg_d = {'Ticker':[],  'Buy/Sell':[], 'Price':[]}
    for ticker in tickers:
        for i, trade in enumerate(blot_d[ticker]):
            ind.append(blot_d[ticker].index[i])
            agg_d['Ticker'].append(ticker)
            agg_d['Buy/Sell'].append(trade)
            price = price_d[ticker].ix[blot_d[ticker].index[i], 'Close']
            agg_d['Price'].append(price)

    return pandas.DataFrame(agg_d, index = ind)

def test_funs():
    """
    >>> import pandas.util.testing as put
    >>> xl_file = pandas.ExcelFile('../tests/test_splits.xlsx')
    >>> blotter = xl_file.parse('blotter', index_col = 0)
    >>> cols = ['Close', 'Adj Close', 'Dividends']
    >>> price_df = xl_file.parse('calc_sheet', index_col = 0)
    >>> price_df = price_df[cols]
    >>> split_frame = calculate_splits(price_df)
    >>> shares_owned = blotter_to_split_adjusted_shares(blotter, split_frame)
    >>> test_vals = xl_file.parse('share_balance', index_col = 0)['cum_shares']
    >>> put.assert_series_equal(shares_owned['cum_shares'].dropna(), test_vals)
    """

if __name__ == '__main__':

    usage = sys.argv[0] + "file_loc"
    description = "description"
    parser = argparse.ArgumentParser(description = description, usage = usage)
    parser.add_argument('arg_1', nargs = 1, type = str, help = 'help_1')
    parser.add_argument('arg_2', nargs = 1, type = int, help = 'help_2')
    args = parser.parse_args()
