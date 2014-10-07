#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: visualize_wealth.utils.py
   :synopsis: Helper fuctions and knick knacks for portfolio analysis

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>
"""

import argparse
import pandas
import numpy
import datetime

def append_store_prices(ticker_list, store_path, start = '01/01/1990'):
    """
    Given an existing store located at ``path``, check to make sure
    the tickers in ``ticker_list`` are not already in the data
    set, and then insert the tickers into the store.

    :ARGS:

        ticker_list: :class:`list` of tickers to add to the
        :class:`pandas.HDStore`

        store_path: :class:`string` of the path to the     
        :class:`pandas.HDStore`

        start: :class:`string` of the date to begin the price data

    :RETURNS:

        :class:`NoneType` but appends the store and comments the
         successes ands failures
    """
    try:
        store = pandas.HDFStore(path = store_path,  mode = 'a')
    except IOError:
        print  path + " is not a valid path to an HDFStore Object"
        return
    store_keys = map(lambda x: x.strip('/'), store.keys())
    not_in_store = numpy.setdiff1d(ticker_list, store_keys )
    new_prices = tickers_to_dict(not_in_store, start = start)

    #attempt to add the new values to the store
    for val in new_prices.keys():
        try:
            store.put(val, new_prices[val])
            print val + " has been stored"
        except:
            print val + " couldn't store"
    store.close()
    return None

def check_store_for_tickers(ticker_list, store):
    """
    Determine which, if any of the :class:`list` `ticker_list` are
    inside of the HDFStore.  If all tickers are located in the store
    returns 1, otherwise returns 0 (provides a "check" to see if
    other functions can be run)

    :ARGS:

        ticker_list: iterable of tickers to be found in the store located
        at :class:`string` store_path

        store: :class:`HDFStore` of the location to the HDFStore

    :RETURNS:

        :class:`bool` True if all tickers are found in the store and
        False if not all the tickers are found in the HDFStore

    """
    if isinstance(ticker_list, pandas.Index):
        #pandas.Index is not sortable, so much tolist() it
        ticker_list = ticker_list.tolist()

    store_keys = map(lambda x: x.strip('/'), store.keys())
    not_in_store = numpy.setdiff1d(ticker_list, store_keys)

    #if len(not_in_store) == 0, all tickers are present
    if not len(not_in_store):
        print "All tickers in store"
        ret_val = True
    else:
        for ticker in not_in_store:
            print "store does not contain " + ticker
        ret_val = False
    return ret_val

def check_store_path_for_tickers(ticker_list, store_path):
    """
    Determine which, if any of the :class:`list` `ticker_list` are
    inside of the HDFStore.  If all tickers are located in the store
    returns 1, otherwise returns 0 (provides a "check" to see if
    other functions can be run)

    :ARGS:

        ticker_list: iterable of tickers to be found in the store located
        at :class:`string` store_path

        store_path: :class:`string` of the location to the HDFStore

    :RETURNS:

        :class:`bool` True if all tickers are found in the store and
        False if not all the tickers are found in the HDFStore
    """
    try:
        store = pandas.HDFStore(path = store_path, mode = 'r+')
    except IOError:
        print  store_path + " is not a valid path to an HDFStore Object"
        return

    if isinstance(ticker_list, pandas.Index):
        #pandas.Index is not sortable, so much tolist() it
        ticker_list = ticker_list.tolist()

    store_keys = map(lambda x: x.strip('/'), store.keys())
    not_in_store = numpy.setdiff1d(ticker_list, store_keys)
    store.close()

    #if len(not_in_store) == 0, all tickers are present
    if not len(not_in_store):
        print "All tickers in store"
        ret_val = True
    else:
        for ticker in not_in_store:
            print "store does not contain " + ticker
        ret_val = False
    return ret_val

def check_trade_price_start(weight_df, price_df):
    """
    Check to ensure that initial weights / trade dates are after
    the first available price for the same ticker

    :ARGS:

        weight_df: :class:`pandas.DataFrame` of the weights to 
        rebalance the portfolio

        price_df: :class:`pandas.DataFrame` of the prices for each
        of the tickers

    :RETURNS:

        :class:`pandas.Series` of boolean values for each ticker
        where True indicates the first allocation takes place 
        after the first price (as desired) and False the converse
    """
    msg = "tickers in the weight_df and price_df must be the same"
    assert set(weight_df.columns) == set(price_df.columns), msg

    ret_d = {}
    for ticker in weight_df.columns:
        first_alloc = (weight_df[ticker] > 0).argmin()
        first_price = price_df[ticker].notnull().argmin()
        ret_d[ticker] = first_alloc > first_price

    return pandas.Series(ret_d)

def create_data_store(ticker_list, store_path):
    """
    Creates the ETF store to run the training of the logistic 
    classificaiton tree

    :ARGS:
    
        ticker_list: iterable of tickers

        store_path: :class:`str` of path to ``HDFStore``
    """
    #check to make sure the store doesn't already exist
    if os.path.isfile(store_path):
        print "File " + store_path + " already exists"
        return
    
    store = pandas.HDFStore(store_path, 'w')
    success = 0
    for ticker in ticker_list:
        try:
            tmp = tickers_to_dict(ticker, 'yahoo', start = '01/01/2000')
            store.put(ticker, tmp)
            print ticker + " added to store"
            success += 1
        except:
            print "unable to add " + ticker + " to store"
    store.close()

    if success == 0: #none of it worked, delete the store
        print "Creation Failed"
        os.remove(path)
    print 
    return None

def index_intersect(arr_a, arr_b):
    """
    Return the intersection of two :class:`pandas` objects, either a
    :class:`pandas.Series` or a :class:`pandas.DataFrame`

    :ARGS:

        arr_a: :class:`pandas.DataFrame` or :class:`pandas.Series`
        arr_b: :class:`pandas.DataFrame` or :class:`pandas.Series`

    :RETURNS:

        :class:`pandas.DatetimeIndex` of the intersection of the two 
        :class:`pandas` objects
    """
    arr_a = arr_a.sort_index()
    arr_a = arr_a.dropna()
    arr_b = arr_b.sort_index()
    arr_b = arr_b.dropna()
    if arr_a.index.equals(arr_b.index) == False:
        return arr_a.index & arr_b.index
    else:
        return arr_a.index

def index_multi_intersect(frame_list):
    """
    Returns the index intersection of multiple 
    :class:`pandas.DataFrame`'s or :class:`pandas.Series`

    :ARGS:

        frame_list: :class:`list` containing either ``DataFrame``'s or
        ``Series``
    
    :RETURNS:

        :class:`pandas.DatetimeIndex` of the objects' intersection
    """
    #check to make sure all objects are Series or DataFrames
    if not all(map(lambda x: isinstance(x, (
            pandas.Series, pandas.DataFrame) ), frame_list)):
        print "All objects must be Series or DataFrame's"
        return
        
    return reduce(lambda x, y: x & y, 
           map(lambda x: x.dropna().index, frame_list) )

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

def setup_trained_hdfstore(trained_data, store_path):
    """
    The ``HDFStore`` doesn't work properly when it's compiled by different
    versions, so the appropriate thing to do is to setup the trained data
    locally (and not store the ``.h5`` file on GitHub).

    :ARGS:

        trained_data: :class:`pandas.Series` with tickers in the index and
        asset  classes for values 

        store_path: :class:`str` of where to create the ``HDFStore``
    """
    
    create_data_store(trained_data.index, store_path)
    return None

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
    if not isinstance(join_col, str):
        print "join_col must be a string"
        return

    if isinstance(ticker_list, (str, unicode)):
        return __get_data(ticker_list, api = api, start = start)[join_col]
    else:
        d = {}
        for ticker in ticker_list:
            d[ticker] = __get_data(ticker, api = api,
                                   start = start)[join_col]
    return pandas.DataFrame(d)

def update_store_prices(store_path):
    """
    Update to the most recent prices for all keys of an existing store, 
    located at path ``path``.

    :ARGS:

        store_path: :class:`string` the location of the ``HDFStore`` file

    :RETURNS:

        :class:`NoneType` but updates the ``HDF5`` file, and prints to 
        screen which values would not update

    """
    reader = pandas.io.data.DataReader
    strftime = datetime.datetime.strftime
    today_str = strftime(datetime.datetime.today(), format = '%m/%d/%Y')
    try:
        store = pandas.HDFStore(path = store_path, mode = 'r+')
    except IOError:
        print  store_path + " is not a valid path to an HDFStore Object"
        return

    for key in store.keys():
        stored_data = store.get(key)
        last_stored_date = stored_data.dropna().index.max()
        today = datetime.datetime.date(datetime.datetime.today())
        if last_stored_date < pandas.Timestamp(today):
            try:
                tmp = reader(key.strip('/'), 'yahoo', start = strftime(
                    last_stored_date, format = '%m/%d/%Y'))

                #need to drop duplicates because there's 1 row of 
                #overlap
                tmp = stored_data.append(tmp)
                tmp["index"] = tmp.index
                tmp.drop_duplicates(cols = "index", inplace = True)
                tmp = tmp[tmp.columns[tmp.columns != "index"]]
                store.put(key, tmp)
            except IOError:
                print "could not update " + key

    store.close()
    return None


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
    
    
    if ind[0]: #The series started on the last day of period
        index = index.copy()[1:] #So we can't get a Period
        ind = time_d[interval](index[:-1]) != time_d[interval](index[1:])

    ldop = index[ind]
    fdop = index[numpy.append(True, ind[:-1])]
    return zip(fdop, ldop)

        
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
