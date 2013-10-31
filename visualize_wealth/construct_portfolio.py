#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: construct_portfolio.py
   :synopsis: Engine to construct portfolios using three general methodologies:
   1. By providing purchase quantities and tickers at given dates
   2. By providing a weight file allocation 
   3. By providing an initial allocation and a rebalancing frequency

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>
"""
import argparse
import pandas
import numpy
import pandas.io.data
import datetime
import urllib2

def format_blotter(blotter_file):
    """  
    Pass in either a location of a blotter file (in ``.csv`` format) or blotter 
    ``pandas.DataFrame`` with all positive values and return a ``pandas.DataFrame``
    where Sell values are then negative values
    
    **ARGS:**
    
        **blotter_file:** ``pandas.DataFrame`` with at least index (dates of 
        Buy / Sell) columns = ['Buy/Sell', 'Shares'] or a string of the file
        location to such a formatted file

    **RETURNS:**

        **blotter:** of type ``pandas.DataFrame`` where sell values have been made 
        negative
        
    """
    if isinstance(blotter_file, str):
        blot = pandas.DataFrame.from_csv(blotter_file)

    #remove whitespaces
    blot['Buy/Sell'] = map(str.strip, blot['Buy/Sell'])

    #if the Sell values are not negative, make them negative
    blot.loc[blot['Buy/Sell'] == 'Sell', 'Shares'] = (-blot.loc[blot['Buy/Sell'] 
                                                       == 'Sell', 'Shares'])
    
    return blot

def append_price_frame_with_dividends(ticker, start_date, end_date=None):
    """
    Given a ticker, start_date, & end_date, return a pandas.DataFrame with 
    a Dividend Columns

    **ARGS:**

        **ticker:** ``str`` of ticker

        **start_date:** ``datetime.datetime`` or string of format "mm/dd/yyyy"

        **end_date:** a ``dtetime.datetime`` or string of format "mm/dd/yyyy"

    **RETURNS:**
    
        **price_df:** a ``pandas.DataFrame`` with columns ['Close', 'Adj Close',
         'Dividends']

    **USAGE:**::
    
         frame_with_divs = construct_portfolio.append_price_frame_with_dividends(
             'EEM', '01/01/2000', '01/01/2013')
                     
    """
    reader = pandas.io.data.DataReader

    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%m/%d/%Y")

    if end_date == None:
        end = datetime.datetime.today()
    elif isinstance(end_date, str):
        end = datetime.datetime.strptime(end_date, "%m/%d/%Y")
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
    
    **ARGS:**
    
        **price_df:** a ``pandas.DataFrame`` with columns ['Close', 'Adj Close',
         'Dividends']

         **tol:** ``flt`` of the tolerance to determine whether a split has occurred

    **RETURNS:**
    
         **price:** ``pandas.DataFrame`` with columns ['Close', 'Adj Close',
          'Dividends', Splits']

    **USAGE:**::
    
        price_df_with_divs_and_split_ratios = construct_portfolio.calculate_splits(
            price_df_with_divs, tol = 0.1)
    
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

def blotter_and_price_df_to_cum_shares(blotter_df, price_df):
    """
    Given a ``blotter DataFrame`` of dates, purchases (+/-),  and a ``price 
    DataFrame`` with Close Adj Close, Dividends, & Splits, calculate the cumulative
    share balance for the position
    
    **ARGS:**
    
        **blotter_df:** a  ``pandas.DataFrame`` where index is buy/sell dates

        **price_df:** a ``pandas.DataFrame`` with columns ['Close', 'Adj Close', 'Dividends', 'Splits']

    **RETURNS:**                          

         ``pandas.DataFrame`` containing contributions, withdrawals, price values

    **USAGE:**::

        agg_stats_for_single_asset = construct_portfolio.blotter_to_split_adj_shares(
            single_asset_blotter, split_adj_price_frame)
        
    """

    blotter_df = blotter_df.sort_index()
    #make sure all dates in the blotter file are also in the price file
    #consider, if those dates aren't in price frame, assign the "closest date" value
    assert numpy.all(map(lambda x: numpy.any(price_df.index == x), 
                         blotter_df.index)), "Buy/Sell Dates not in Price File"

    #now cumsum the buy/sell chunks and multiply by splits to get total shares
    bs_series = pandas.Series()
    start_dts = blotter_df.index
    end_dts = pandas.to_datetime(numpy.append(blotter_df.index[1:],
                                              price_df.index[-1]))

    dt_chunks = zip(start_dts, end_dts)
    end = 0.

    for i, chunk in enumerate(dt_chunks):
        #print str(i) + ' of ' + str(len(dt_chunks)) + ' total'
        tmp = price_df[chunk[0]:chunk[1]][:-1]
        if chunk[1] == price_df.index[-1]:
            tmp = price_df[chunk[0]:chunk[1]]
        splits = tmp[pandas.notnull(tmp['Splits'])]
        vals = numpy.append(blotter_df['Buy/Sell'][chunk[0]] + end,
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
    no_price = blotter_df['Price'][pandas.isnull(blotter_df['Price'])]
    blotter_df.ix[no_price.index, 'Price'] = price_df.ix[no_price.index, 'Close']

    contr = blotter_df['Buy/Sell'].mul(blotter_df['Price'])
    cum_inv = contr.cumsum()
    contr = contr[price_df.index].fillna(0.0)
    cum_inv = cum_inv[price_df.index].ffill()
    res = pandas.DataFrame({'cum_shares':bs_series, 'contr_withdrawal':contr, 
                            'cum_investment':cum_inv})
    
    return price_df.join(res)
        
def construct_random_trades(split_df, num_trades):
    """
    Create random trades on random trade dates, but never allow shares to go negative
    
    **ARGS:**
    
        **split_df:** ``pandas.DataFrame`` that has 'Close', 'Dividends', 'Splits'

    **RETURNS:**

         **blotter_frame:** ``pandas.DataFrame`` a blotter with random trades,
          num_trades
    """
    ind = numpy.sort(numpy.random.randint(0, len(split_df), size = num_trades))
    #This unique makes sure there aren't double trade day entries which breaks 
    #the function blotter_and_price_df_to_cum_shares
    ind = numpy.unique(ind)
    dates = split_df.index[ind]

    #construct random execution prices
    prices = []
    for date in dates:
        u_lim = split_df.loc[date, 'High']
        l_lim = split_df.loc[date, 'Low']
        prices.append(numpy.random.rand()*(u_lim - l_lim + 1) + l_lim)
        
    trades = numpy.random.randint(-100, 100, size = len(ind))
    trades = numpy.round(trades, -1)

    while numpy.any(trades.cumsum() < 0):
        trades[numpy.argmin(trades)] *= -1.    

    return pandas.DataFrame({'Buy/Sell':trades, 'Price':prices}, index = dates)

def blotter_to_cum_shares(blotter_series, ticker, start_date, end_date, tol):
    """
    Similar to :fun:blotter_and_price_df_to_cum_shares except that a split adjusted 
    ``price_df`` is not necessary.  Only a blotter,  ticker, start_date, & end_date
    are needed.  Returns 

    **ARGS:**

        **blotter_series:** a  ``pandas.Series`` with index of dates and values of
         quantity

         **ticker:** ``str`` the ticker for which the buys and sells occurs

         **start_date:** type ``str`` or ``datetime.datetime``

         **end_date:** ``str`` or ``datetime.datetime``

         **tol:** ``flt``  the tolerance to find the split dates (.1 recommended)
    
    **RETURNS:**

         ``pandas.DataFrame`` containing contributions, withdrawals, price values
    """

    price_df = append_price_frame_with_dividends(ticker, start_date, end_date)
    split_df = calculate_splits(price_df)
    return blotter_and_price_df_to_cum_shares(blotter_series, split_df)

def generate_random_asset_path(ticker, start_date, num_trades):
    import pdb
    pdb.set_trace()
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%m/%d/%Y")
    end_date = datetime.datetime.today()
    prices = append_price_frame_with_dividends(ticker, start_date)
    blotter = construct_random_trades(prices, num_trades)
    #blotter.to_csv('../tests/' + ticker + '.csv')
    return blotter_to_cum_shares(blotter_series = blotter, ticker = ticker,
                                 start_date = start_date, end_date = end_date, 
                                 tol = .1)

def generate_random_portfolio_blotter(tickers, num_trades):
    """
    Construct a random trade blotter, given a list of tickers and a number of trades 
    (to be used for all tickers), prices will be  the 'Close' of that ticker in the 
    price DataFrame that is collected

    **INPUTS:**
    
        **tickers:** a ``list`` with the tickers to be used

        **num_trades:** ``int``, the number of trades to randomly generate for each
         ticker

    **RETURNS:**

        ``pandas.DataFrame`` with columns 'Ticker', 'Buy/Sell' (+ for buys, - for
         sells) and 'Price'
    
    """
    blot_d = {}
    price_d = {}
    for ticker in tickers:
        tmp = append_price_frame_with_dividends(ticker, start_date =
                                                datetime.datetime(1990, 1, 1))
        price_d[ticker] = calculate_splits(tmp)
        blot_d[ticker] = construct_random_trades(price_d[ticker], num_trades)
    ind = []
    agg_d = {'Ticker':[],  'Buy/Sell':[], 'Price':[]}
    #import pdb
    #pdb.set_trace()
    for ticker in tickers:
        for date in blot_d[ticker].index:
            ind.append(date)
            agg_d['Ticker'].append(ticker)
            agg_d['Buy/Sell'].append(blot_d[ticker].loc[date, 'Buy/Sell'])
            agg_d['Price'].append(blot_d[ticker].loc[date, 'Price'])

    return pandas.DataFrame(agg_d, index = ind)

def portfolio_from_blotter(blotter_df):
    """
    The aggregation function to construct a portfolio given a blotter of tickers,
    trades, and number of shares.  

    **ARGS:**

        **agg_blotter_df:** a ``pandas.DataFrame`` with columns ['Ticker',
         'Buy/Sell', 'Price'],  where the 'Buy/Sell' column is the quantity of
          shares, (+) for buy, (-) for sell

    **RETURNS:**
    
        ** ``pandas.Panel``** with dimensions [tickers, dates, price data]
    """
    tickers = pandas.unique(blotter_df['Ticker'])
    start_date = blotter_df.sort_index().index[0]
    end_date = datetime.datetime.today()
    val_d = {}
    for ticker in tickers:
        blotter_series = blotter_df[blotter_df['Ticker'] == ticker].sort_index()
        val_d[ticker] = blotter_to_cum_shares(blotter_series, ticker,
                                              start_date, end_date, tol = .1)

    return pandas.Panel(val_d)

def fetch_data_for_portfolio_construction(weight_df):
    """
    Given a weight frame with index of allocation dates and columns of percentage 
    allocations, fetch the data using Yahoo!'s API and return a panel of [tickers, 
    dates, price data

    **ARGS:**
    
        **weight_df:** a ``pandas.DataFrame`` with dates as index and tickers as
         columns

    **RETURNS:**
    
        ``pandas.Panel`` where:
        * ``panel.items:`` are tickers
        * ``panel.major_axis:`` dates
        * ``panel.minor_axis:`` price information, specifically: 
        ['ac_c', 'c0_ac0', 'n0', 'Adj_Q', 'Asset Value', 'Open', 'Close', 
        'Adj Close']
    """
    reader = pandas.io.data.DataReader
    d_0 = weight_df.index.min()
    tickers = weight_df.columns
    opens = {}
    closes = {}
    acs = {}
    
    for ticker in tickers:
        tmp = reader(ticker, 'yahoo', start = d_0)
        opens[ticker]  = tmp['Open']
        closes[ticker] = tmp['Close']
        acs[ticker] = tmp['Adj Close']
        
    
    close_df = pandas.DataFrame(closes, columns = weight_df.columns)
    adj_df = pandas.DataFrame(acs, columns = weight_df.columns)
    open_df = pandas.DataFrame(opens, columns = weight_df.columns)
    
    assert numpy.all(close_df.index == adj_df.index), (
        "Close and Adj Close Indexes are not the same")

    index = close_df.index
    
    #determine the dates chunks
    import pdb
    #pdb.set_trace()
    
    #the columns correspond to the calculation sheet for interpretability
    columns = ['ac_c', 'c0_ac0', 'n0', 'Adj_Q', 'Asset Value', 'Open', 
               'Close', 'Adj Close']

    #preallocate for the calculation
    panel = pandas.Panel(numpy.zeros([len(tickers), len(index), len(columns)]),
                         major_axis = index, items = tickers, minor_axis = columns)
    
    #insert the Close into the panel
    panel.loc[:, :, 'Close'] = close_df
    panel.loc[:, :, 'Open'] = open_df
    panel.loc[:, :, 'Adj Close'] = adj_df

    return panel

def portfolio_from_weight_file(weight_df, price_panel, start_value):
    """
    Returns a pandas.DataFrame with columns ['Close', 'Open'] when provided
    a pandas.DataFrame of weight allocations and a starting  value of the index

    **ARGS:**
    
        **weight_df:** ``pandas.DataFrame`` of a weight allocation with tickers for
        columns, index of dates and weight allocations to each of the tickers

        **price_panel:** ``pandas.Panel`` with dimensions [tickers, index, price
         data]

    **RETURNS:**
    
        ``pandas.Panel`` with dimensions (tickers, dates, price date)
    
    """

    #these columns correspond to the columns in sheet 'value_calcs!' in "panel from
    #weight file test.xlsx"
    
    columns = ['ac_c', 'c0_ac0', 'n0', 'Adj_Q', 'Asset Value', 'Open', 
               'Close', 'Adj Close']
    panel = price_panel.reindex(minor_axis = columns)
    port_cols = ['Close', 'Open']
    index = panel.major_axis
    port_df = pandas.DataFrame(numpy.zeros([ len(index), len(port_cols)]), 
                               index = index, columns = port_cols)

    a = weight_df.index
    b = pandas.to_datetime(numpy.append(weight_df.index[1:], index[-1]))
    dt_chunks = zip(a, b)
    
    #fill in the Adjusted Quantity values and the aggregate position values
    p_val = start_value
    for chunk in dt_chunks:
        n = len(panel.loc[:, chunk[0]:chunk[1], 'Close'])
        c0_ac0 = panel.loc[:, chunk[0], 'Close'].div(
            panel.loc[:, chunk[0], 'Adj Close'])
        n0 = p_val*weight_df.loc[chunk[0], :].div(panel.loc[:, chunk[0], 'Close'])
        panel.loc[:, chunk[0]:chunk[1], 'ac_c']  = (
            panel.loc[:, chunk[0]:chunk[1],'Adj Close'].div(
            panel.loc[:, chunk[0]:chunk[1], 'Close']))
        panel.loc[:, chunk[0]:chunk[1], 'c0_ac0'] = (
            numpy.tile(c0_ac0.values, [n, 1]).transpose())
        panel.loc[:, chunk[0]:chunk[1], 'n0'] = numpy.tile(
            n0.values, [n, 1]).transpose()
        panel.loc[:, chunk[0]:chunk[1], 'Adj_Q'] = (panel.loc[:,
            chunk[0]:chunk[1], ['c0_ac0', 'ac_c', 'n0']].apply(numpy.product, 
            axis = 2))

        #assign the portfolio values
        port_df.loc[chunk[0]:chunk[1], 'Close'] = (
            panel.loc[:, chunk[0]:chunk[1],'Adj_Q'].mul(
            panel.loc[:, chunk[0]:chunk[1], 'Close']).sum(axis = 1))
        port_df.loc[chunk[0]:chunk[1], 'Open'] = (
            panel.loc[:, chunk[0]:chunk[1],'Adj_Q'].mul(
            panel.loc[:, chunk[0]:chunk[1], 'Open']).sum(axis = 1))

        p_val = port_df.loc[chunk[1], 'Close']

    #The portfolio should start at the first trade, a[0]
    return port_df[a[0]:]

def portfolio_from_initial_weights(weight_series, price_panel, start_value,
                                   rebal_frequency):
    """
    Returns a pandas.DataFrame with columns ['Close', 'Open'] when provided
    a pandas.Series of intial weight allocations, the date of those initial weight 
    allocations (series.name), a starting value of the index, and a rebalance  
    frequency (this is the classical "static" construction" methodology, rebalancing
    at somspecified interval)

    **ARGS:**
    
        **weight_series:** ``pandas.Series`` of a weight allocation with an index of
         tickers, and a name of the initial allocation

        **price_panel:** ``pandas.Panel`` with dimensions [tickers, index, price
          data]

        **start_value:** ``flt`` of the value to start the index

        **rebal_frequency:** ``str`` of 'weekly', 'monthly', 'quarterly', 'yearly'

    **RETURNS:**
    
         **price:** of type ``pandas.DataFrame`` with portfolio 'Close' and 'Open'
    """
    
    d_0 = numpy.max(price_panel.loc[:, :, 'Close'].apply(
        pandas.Series.first_valid_index))
    index = price_panel.loc[:, d_0:, :].major_axis
    
    assert numpy.any(index == weight_series.name), (
        "The first trade date is not part of the prices panel")
    
    interval_dict = {'weekly':lambda x: x[:-1].week != x[1:].week, 
                     'monthly': lambda x: x[:-1].month != x[1:].month,
                     'quarterly':lambda x: x[:-1].quarter != x[1:].quarter,
                     'yearly':lambda x: x[:-1].year != x[1:].year}

    #create a boolean array of rebalancing dates
    ind = numpy.append(True, interval_dict[rebal_frequency](index))
    weight_df = pandas.DataFrame(numpy.tile(weight_series.values, 
        [len(index[ind]), 1]), index = index[ind], columns = weight_series.index)
                    
    return portfolio_from_weight_file(weight_df, price_panel, start_value)


def test_funs():
    """
    >>> import pandas.util.testing as put
    >>> xl_file = pandas.ExcelFile('../tests/test_splits.xlsx')
    >>> blotter = xl_file.parse('blotter', index_col = 0)
    >>> cols = ['Close', 'Adj Close', 'Dividends']
    >>> price_df = xl_file.parse('calc_sheet', index_col = 0)
    >>> price_df = price_df[cols]
    >>> split_frame = calculate_splits(price_df)

    >>> shares_owned = blotter_and_price_df_to_cum_shares(blotter, split_frame)
    >>> test_vals = xl_file.parse('share_balance', index_col = 0)['cum_shares']
    >>> put.assert_series_equal(shares_owned['cum_shares'].dropna(), test_vals)

    >>> xl_file = pandas.ExcelFile('../tests/panel from weight file test.xlsx')
    >>> weight_df = xl_file.parse('rebal_weights', index_col = 0)
    >>> tickers = ['EEM', 'EFA', 'IYR', 'IWV', 'IEF', 'IYR', 'SHY']
    >>> d = {}
    >>> for ticker in tickers:
    ...     d[ticker] = xl_file.parse(ticker, index_col = 0)
    >>> portfolio = portfolio_from_weight_file(weight_df, pandas.Panel(d), 1000)
    >>> manual_calcs = xl_file.parse('index_result', index_col = 0)
    >>> put.assert_series_equal(manual_calcs['Close'], portfolio['Close'])
    """
    return None

if __name__ == '__main__':

    usage = sys.argv[0] + "file_loc"
    description = "description"
    parser = argparse.ArgumentParser(description = description, usage = usage)
    parser.add_argument('arg_1', nargs = 1, type = str, help = 'help_1')
    parser.add_argument('arg_2', nargs = 1, type = int, help = 'help_2')
    args = parser.parse_args()
