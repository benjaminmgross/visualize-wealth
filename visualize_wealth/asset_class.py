#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: asset_class.py
   :synopsis: Asset Class Attribution Analysis Made Easy

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>
"""

import pandas
import numpy
import scipy.optimize as sopt
import visualize_wealth.utils as utils
import visualize_wealth.analyze as vwa


AC_DICT = {'VTSMX':'US Equity', 'VBMFX':'Fixed Income', 
           'VGTSX':'Foreign Equity', 'IYR':'Alternative', 
           'GLD':'Alternative', 'GSG':'Alternative',
           'WPS':'Alternative'}

def multicol_subclass(frame, weights = None):
    if not isinstance(weights, pandas.Series):
        n = len(frame.columns)
        weights = pandas.Series([1./n for col in numpy.arange(n)],
            index = frame.columns)
    d = {}
    for col in frame.columns:
        d[col] = subclass(frame[col])
    
    #create a union index, b/c pandas takes interse
    ind = reduce(lambda x, y: x | y, 
                 map(lambda x: x.index, d.values()) )
    
    sc = pandas.DataFrame(d, index = ind)
    return sc.loc[sc.index != 'Asset Class'].mul(
        weights).sum(axis = 1)

def multi_asset_class_by_interval(frame, interval, weights):
    return None

def mulit_subclass_by_interval(frame, interval, weights):
    return None

def asset_class_helper_fn(series, benchmarks):
    """
    Given as series of prices, find the most likely asset class of the 
    asset, based on r-squared attribution of return variance (i.e. 
    maximizing r-squared).

    :ARGS:

        series: :class:`pandas.Series` of asset prices
    
    

    .. note:: Functionality for Asset Allocation Funds

        Current functionality only allows for a single asset class to 
        be chosen in an effort not to overfit the attribution of asset 
        returns.  This logic works well for "single asset class ETFs 
        and Mutual Funds" but not for multi-asset class strategies

    """
    rsq_d = {}
    for ticker in benchmarks.columns:
        ind = utils.dtindex_clean_intersect(
            series, benchmarks[ticker])
        rsq_d[ticker] = vwa.r_squared(
            series[ind], benchmarks[ticker][ind])
    rsq = pandas.Series(rsq_d)
    return AC_DICT[rsq.argmax()]

def multicol_asset_class(frame, weights = None):
    """
    Returns the asset class weightings given a :class:`pandas.DataFrame`
    of asset prices and the weights of each of the assets
    
    :ARGS:

        frame: :class:`pandas.DataFrame` of asset prices
    
        weights: :class:`pandas.Series` of asset weights

    RETURNS:

        :class:`pandas.Series` with values of the percentage of each
        asset class and index of asset class weights                                                                                         
    """
    if not isinstance(weights, pandas.Series):
        n = len(frame.columns)
        weights = pandas.Series([1./n for col in numpy.arange(n)],
            index = frame.columns)

    benchmarks = utils.tickers_to_frame(AC_DICT.keys(),
        join_col = 'Adj Close')
    d = {}
    for col in frame.columns:
        d[col] = [asset_class_helper_fn(frame[col], benchmarks),
                  weights[col]]
    return pandas.DataFrame(d, index = ['asset class', 'weight'])

def asset_class(series):
    """
    Returns the asset class for a given price series, where asset class
    is either 'US Equity', 'Foreign Equity' or 'Fixed Income' or '
    Alternative'

    :ARGS:

        series: :class:`pandas.Series` of asset prices

    :RETURNS:

        :class:`string` of the broad asset class, either ['US Equity',
        'Foreign Equity', 'Fixed Income', 'Alternative']


    """
    return asset_class_helper_fn(series, 
        utils.tickers_to_frame(AC_DICT.keys(), join_col = 'Adj Close'))

def asset_class_dict(asset_class):
    """
    All of the ticker and asset class information stored in dictionary 
    form for use by other functions

    :ARGS:

        asset_class: :class:`string` of 'US Equity', 'Foreign Equity',
        'Alternative' or 'Fixed Income'.

    :RETURNS:

        :class:dict of the asset subclasses and respective tickers
    """
    fi_dict = {'US Inflation Protected':'TIP', 'Foreign Treasuries':'BWX',
               'Foreign High Yield':'PCY','US Investment Grade':'LQD',
               'US High Yield':'HYG', 'US Treasuries ST':'SHY',
               'US Treasuries LT':'TLT', 'US Treasuries MT':'IEF'}

    us_eq_dict = {'U.S. Large Cap Growth':'JKE', 'U.S. Large Cap Value':'JKF',
                  'U.S. Mid Cap Growth':'JKH','U.S. Mid Cap Value':'JKI',
                  'U.S. Small Cap Growth':'JKK', 'U.S. Small Cap Value':'JKL'}

    for_eq_dict = {'Foreign Developed Small Cap':'SCZ',
                   'Foreign Developed Large Growth':'EFG',
                   'Foreign Developed Large Value':'EFV',
                   'Foreign Emerging Market':'EEM'}


    alt_dict = {'Commodities':'GSG', 'U.S. Real Estate':'IYR',
                'Foreign Real Estate':'WPS', 'U.S. Preferred Stock':'PFF'}

    class_dict = {'US Equity': us_eq_dict, 'Foreign Equity': for_eq_dict,
                  'Alternative': alt_dict, 'Fixed Income': fi_dict}

    return class_dict[asset_class]

def best_fitting_weights(series, asset_class_prices):
    """
    Return the best fitting weights given a :class:`pandas.Series` of 
    asset prices and a :class:`pandas.DataFrame` of asset class prices.  
    Can be used with the :func:`clean_dates` function to ensure an 
    intersection of the two indexes is being passed to the function
    
    :ARGS:

        asset_prices: m x 1 :class:`pandas.TimeSeries` of asset_prices

        ac_prices: m x n :class:`pandas.DataFrame` asset class ("ac") 
        prices

    :RETURNS:

        :class:`pandas.TimeSeries` of nonnegative weights for each 
        asset such that the r_squared from the regression of 
        :math:`Y ~ Xw + e` is maximized

    """

    def _r_squared_adj(weights):
        """
        The Adjusted R-Squared that incorporates the number of 
        independent variates using the `Formula Found of Wikipedia
        <http://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2>_`
        """

        estimate = numpy.dot(ac_rets, weights)
        sse = ((estimate - series_rets)**2).sum()
        sst = ((series_rets - series_rets.mean())**2).sum()
        rsq = 1 - sse/sst
        p, n = weights.shape[0], ac_rets.shape[0]
        return rsq - (1 - rsq)*(float(p)/(n - p - 1))

    def _obj_fun(weights):
        """
        To maximize the r_squared_adj minimize the negative of r_squared
        """
        return -_r_squared_adj(weights)

    #linear price changes to create a weighted return
    ac_rets = asset_class_prices.pct_change()
    series_rets = series.pct_change()

    #de-mean the sample
    ac_rets = ac_rets.sub(ac_rets.mean() )
    series_rets = series_rets.sub( series_rets.mean() )

    num_assets = ac_rets.shape[1]
    guess = numpy.zeros(num_assets,)

    #ensure the boundaries of the function are (0, 1)
    ge_zero = [(0,1) for i in numpy.arange(num_assets)]

    #optimize to maximize r-squared, using the 'TNC' method 
    #(that uses the boundary functionality)
    opt = sopt.minimize(_obj_fun, x0 = guess, method = 'TNC', 
                        bounds = ge_zero)
    normed = opt.x*(1./numpy.sum(opt.x))

    return pandas.TimeSeries(normed, index = ac_rets.columns)

def subclass(series):
    """
    Aggregator function that returns the overall asset class, and 
    proportion of subclasses attributed to the returns of ``series.``

    :ARGS:

        series: :class:`pandas.Series` of asset prices

    :RETURNS:

        :class:`pandas.Series` of the subclasses and asset class for 
        the entire time period
    """
    a_class = asset_class(series)
    pairs = asset_class_dict(a_class)
    d_o = utils.first_valid_date(series)
    benchmarks = utils.tickers_to_frame(pairs.values(), start = d_o,
                                        join_col = 'Adj Close')
    
    sub_classes = subclass_helper_fn(series,benchmarks,  a_class)
    return sub_classes.append(
        pandas.Series([a_class], ['Asset Class']))

def subclass_by_interval(series, interval):
    """
    Aggregator function to determine the asset class for the entire 
    period, followed by asset subclass over intervals of ``interval.``

    :ARGS:

        series: :class:`pandas.Series` of asset prices

        interval: :class:string of the interval, currently only accepts 
        ``quarterly`` or ``annual``

    :RETURNS:

        :class:`pandas.DataFrame` of the asset_subclasses over period 
        interval

    .. note::

        In an effort to ensure spurious asset classes aren't chosen (for 
        instance, 'US Equity' to be chosen for one quarter and then 
        'Alternatives' to be chosen in a different quarter, simply 
        because of   "similar asset performance"), the process of the 
        algorithm is:

            1. Determine the "Overall Asset Class" for the entire 
            period of the asset's returns

            2. Determine the subclass attributions over the rolling 
            interval of time

    """
    a_class = asset_class(series)
    d_o = utils.first_valid_date(series)
    ac_dict = asset_class_dict(a_class)
    benchmarks = utils.tickers_to_frame(ac_dict.values(), api ='yahoo',
        start = d_o, join_col = 'Adj Close')
    return subclass_by_interval_helper_fn(series, benchmarks,  
        interval, asset_class)
    
def subclass_by_interval_helper_fn(series, benchmarks, 
                                   interval, asset_class):
    """
    Return asset su class weightings that explain the asset returns 
    over periods of "interval."

    :ARGS:

        asset_prices: :class:`pandas.Series` of the asset for which 
        attribution will be done

        asset_class_prices: :class:`pandas.DataFrame` of asset class 
        prices

        interval :class:string of the frequency interval 'quarterly' 
        or 'annual'

    :RETURNS:

        :class:`pandas.DataFrame` of proportions of each asset class 
        that most explain the returns of the individual security
    """    
    ind = utils.dtindex_clean_intersect(series, benchmarks)
    dts = utils.zipped_time_chunks(index = ind, interval = interval)
    weight_d  = {}
    for beg, fin in dts:
        weight_d[beg] = best_fitting_weights(
            series[beg:fin], benchmarks.loc[beg:fin, :]).rename(
            index = {v:k for k, v in AC_DICT.iteritems()})

    return pandas.DataFrame(weight_d).transpose()
    
def subclass_helper_fn(series, benchmarks, asset_class):
    """
    Given an the prices of a single asset an its overall asset class, 
    return the proportion of returns attribute to each asset class

    :ARGS:

        series: :class:`pandas.Series` of asset prices

        asset_class: :class:`string` of either ``US Equity``, 
        ``Foreign Equity``, ``Alternative``, or  ``Fixed Income``

    :RETURNS:

        :class:`pandas.DataFrame` of the subclasses and their 
        optimized proportion to explain the ``series`` returns over 
        the entire period.
    """
    ind = utils.dtindex_clean_intersect(series, benchmarks)
    ret_series = best_fitting_weights(series[ind], 
                benchmarks.loc[ind, :])
    pairs = asset_class_dict(asset_class)
    #change the column names to asset classes instead of tickers
    return ret_series.rename(index = {v:k for k, v in pairs.iteritems()})

    
