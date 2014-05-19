#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: utils.py
   :synopsis: Helper fuctions and knick knacks for portfolio analysis

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>
"""

import argparse
import pandas
import numpy

def first_valid_date(prices):
    """
    Helper function to determine the first valid date from a set of 
    different prices Can take either a :class:`dict` of 
    :class:`pandas.DataFrame`s where each key is a ticker's 'Open', 
    'High', 'Low', 'Close', 'Adj Close' or a single 
    :class:`pandas.DataFrame` where each column is a different ticker

    :ARGS:

        prices: either :class:`dictionary` or :class:`pandas.DataFrame`

    :RETURNS:

        :class:`pandas.Timestamp` 
   """
    iter_dict = { pandas.DataFrame: lambda x: x.columns,
                  dict: lambda x: x.keys() } 

    try:
        each_first = map(lambda x: prices[x].dropna().index.min(),
                         iter_dict[ type(prices) ](prices) )
        return max(each_first)
    except KeyError:
        print "prices must be a DataFrame or dictionary"
        return

def tickers_to_dict(ticker_list, api = 'yahoo', start = '01/01/1990'):
    """
    Utility function to return ticker data where the input is either a 
    ticker, or a list of tickers.

    :ARGS:

        ticker_list: :class:`list` in the case of multiple tickers or 
        :class:`str` in the case of one ticker

        api: :class:`string` identifying which api to call the data 
        from.  Either 'yahoo' or 'google'

        start: :class:`string` of the desired start date
                
    :RETURNS:

        :class:`dictionary` of (ticker, price_df) mappings or a
        :class:`pandas.DataFrame` when the ``ticker_list`` is 
        :class:`str`
    """
    if isinstance(ticker_list, (str, unicode)):
        return __get_data(ticker_list, api = api, start = start)
    else:
        d = {}
        for ticker in ticker_list:
            d[ticker] = __get_data(ticker, api = api, start = start)
    return d

def tickers_to_frame(ticker_list, api = 'yahoo', start = '01/01/1990', 
                     join_col = 'Adj Close'):
    """
    Utility function to return ticker data where the input is either a 
    ticker, or a list of tickers.

    :ARGS:

        ticker_list: :class:`list` in the case of multiple tickers or 
        :class:`str` in the case of one ticker

        api: :class:`string` identifying which api to call the data 
        from.  Either 'yahoo' or 'google'

        start: :class:`string` of the desired start date

        join_col: :class:`string` to aggregate the 
        :class:`pandas.DataFrame`
                
    :RETURNS:

        :class:`pandas.DataFrame` of (ticker, price_df) mappings or a
        :class:`pandas.DataFrame` when the ``ticker_list`` is 
        :class:`str`
    """
    def __get_data(ticker, api, start):
        reader = pandas.io.data.DataReader
        try:
            data = reader(ticker, api, start = start)
            print "worked for " + ticker
            return data
        except:
            print "failed for " + ticker
            return
    if isinstance(ticker_list, (str, unicode)):
        return __get_data(ticker_list, api = api, start = start)[join_col]
    else:
        d = {}
        for ticker in ticker_list:
            d[ticker] = __get_data(ticker, api = api, 
                                   start = start)[join_col]
    return pandas.DataFrame(d)


def zipped_time_chunks(index, interval):
    """
    Given different period intervals, return a zipped list of tuples
    of length 'period_interval', containing only full periods

    .. note:: 

        The function assumes indexes are of 'daily_frequency'
    
    :ARGS:
    
        index: :class:`pandas.DatetimeIndex`

        per_interval: :class:`string` either 'monthly', 'quarterly',
        or 'yearly'
    """
    #create the time chunks
    #calculate raer for each of the time chunks
    time_d = {'monthly': lambda x: x.month, 
              'quarterly':lambda x:x.quarter,
              'yearly':lambda x: x.year}

    ind = time_d[interval](index[:-1]) != time_d[interval](index[1:])
    
    
    if ind[0] == True: #The series started on the last day of period
        index = index.copy()[1:] #So we can't get a Period
        ind = time_d[interval](index[:-1]) != time_d[interval](index[1:])

    ldop = index[ind]
    fdop = index[numpy.append(True, ind[:-1])]
    return zip(fdop, ldop)

def normalized_price(price_df):
    """
    Return the normalized price of a series

    :ARGS:

        price_df: :class:`pandas.Series` or :class:`pandas.DataFrame`

    :RETURNS:
        
        same as the input
    """
    if isinstance(price_df, pandas.Series):

        if pandas.isnull(price_df).any():
            print "This series contains null values"
            return
        else:
            return price_df.div(price_df[0])
    
    elif isinstance(price_df, pandas.DataFrame):
        if pandas.isnull(price_df).any().any():
            print "This DataFrame contains null values"
            return
        else:
            return price_df.div(price_df.iloc[0, :] )
    else:
        print "Input must be pandas.Series or pandas.DataFrame"
        return
        
def __get_data(ticker, api, start):
    """
    Helper function to get Yahoo! Data with exceptions built in and 
    messages that confirm success for given tickers

    ARGS:
        
        ticker: either a :class:`string` of a ticker or a :class:`list`
        of tickers

        api: :class:`string` the api from which to get the data, 
        'yahoo'or 'google'

        start: :class:`string` the start date to start the data 
        series

    """
    reader = pandas.io.data.DataReader
    try:
        data = reader(ticker, api, start = start)
        print "worked for " + ticker
        return data
    except:
        print "failed for " + ticker
        return

def scipt_function(arg_1, arg_2):
	return None

if __name__ == '__main__':
    usage = sys.argv[0] + "usage instructions"
    description = "describe the function"
    parser = argparse.ArgumentParser(description = description, 
                                     usage = usage)
    parser.add_argument('name_1', nargs = 1, type = str, 
                        help = 'describe input 1')
    parser.add_argument('name_2', nargs = '+', type = int, 
                        help = "describe input 2")
    args = parser.parse_args()
    script_function(input_1 = args.name_1[0], input_2 = args.name_2)
