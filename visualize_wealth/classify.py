#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: visualize_wealth.classify.py

Created by Benjamin M. Gross

"""

import argparse
import datetime
import numpy
import pandas
import os

def classify_series_with_store(series, trained_series, store_path,
                               calc_meth = 'x-inv-x', n = None):
    """
    Determine the asset class of price series from an existing
    HDFStore with prices

    :ARGS:

        series: :class:`pandas.Series` or `pandas.DataFrame` of the
        price series to determine the asset class of

        trained_series: :class:`pandas.Series` of tickers
        and their respective asset classes

        store_path: :class:`string` of the location of the HDFStore
        to find asset prices

        calc_meth: :class:`string` of either ['x-inv-x', 'inv-x', 'exp-x']
        to determine which calculation method is used
                
        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

    :RETURNS:

        :class:`string` of the tickers that have been estimated
        based on the method provided
    """
    from .utils import index_intersect
    from .analyze import log_returns, r2_adj

    if series.name in trained_series.index:
        return trained_series[series.name]
    else:
        try:
            store = pandas.HDFStore(path = store_path, mode = 'r')
        except IOError:
            print store_path + " is not a valid path to HDFStore"
            return
        rsq_d = {}
        ys = log_returns(series)

        for key in store.keys():
            key = key.strip('/')
            p = store.get(key)
            xs = log_returns(p['Adj Close'])
            ind = index_intersect(xs, ys)
            rsq_d[key] = r2_adj(benchmark = ys[ind], series = xs[ind])
        rsq_df = pandas.Series(rsq_d)
        store.close()
        if not n:
            n = len(trained_series.unique()) + 1
        
    return __weighting_method_agg_fun(series = rsq_df,
                                      trained_series = trained_series,
                                      n = n, calc_meth = calc_meth)

def classify_series_with_online(series, trained_series,
                                calc_meth = 'x-inv-x', n = None):
    """
    Determine the asset class of price series from an existing
    HDFStore with prices

    :ARGS:

        series: :class:`pandas.Series` or `pandas.DataFrame` of the
        price series to determine the asset class of

        trained_series: :class:`pandas.Series` of tickers
        and their respective asset classes

        calc_meth: :class:`string` of either ['x-inv-x', 'inv-x', 'exp-x']
        to determine which calculation method is used
                
        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

    :RETURNS:

        :class:`string` of the tickers that have been estimated
        based on the method provided
    """
    from .utils import tickers_to_dict, index_intersect
    from .analyze import log_returns, r2_adj

    if series.name in trained_series.index:
        return trained_series[series.name]
    else:
        price_dict = tickers_to_dict(trained_series.index)
        rsq_d = {}
        ys = log_returns(series)
        
        for key in price_dict.keys():
            p = price_dict[key]
            xs = log_returns(p['Adj Close'])
            ind = index_intersect(xs, ys)
            rsq_d[key] = r2_adj(benchmark = xs[ind], series = ys[ind])
        rsq_df = pandas.Series(rsq_d)

        if not n:
            n = len(trained_series.unique()) + 1
        
    return __weighting_method_agg_fun(series = rsq_df,
                                      trained_series = trained_series,
                                      n = n, calc_meth = calc_meth)

def knn_exp_weighted(series, trained_series, n = None):
    """
    Training data is a m x n matrix with 'training_tickers' as columns
    and rows of r-squared for different tickers and asset_class is a
    n x 1 result of the asset class

    :ARGS:

        series: :class:`pandas.Series` or :class:`pandas.DataFrame` of
        r-squared values

        trained_series: :class:`pandas.Series` of the columns
        and their respective asset classes

        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

    :RETURNS:

        :class:`string` of the tickers that have been estimated
        based on the n closest neighbors
   """
    if not n:
        n = len(trained_series.unique()) + 1

    return __weighting_method_agg_fun(series, trained_series, n,
                                      calc_meth = 'exp-x')

def knn_inverse_weighted(series, trained_series, n = None):
    """
    Training data is a m x n matrix with 'training_tickers' as columns
    and rows of r-squared for different tickers and asset_class is a
    n x 1 result of the asset class

    :ARGS:

        series: :class:`pandas.Series` or :class:`pandas.DataFrame` of
        r-squared values

        trained_series: :class:`pandas.Series` of the columns
        and their respective asset clasnses

        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

    :RETURNS:

        :class:`string` of the tickers that have been estimated
        based on the n closest neighbors
   """
    if not n:
        n = len(trained_series.unique()) + 1

    return __weighting_method_agg_fun(series, trained_series, n,
                                      calc_meth = 'inv-x')

def knn_wt_inv_weighted(series, trained_series, n = None):
    """
    Training data is a m x n matrix with 'training_tickers' as columns
    and rows of r-squared for different tickers and asset_class is a
    n x 1 result of the asset class

    :ARGS:

        series: :class:`pandas.Series` or :class:`pandas.DataFrame` of
        r-squared values

        trained_series: :class:`pandas.Series` of the columns
        and their respective asset clasnses

        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

    :RETURNS:

        :class:`string` of the tickers that have been estimated
        based on the n closest neighbors
   """
    if not n:
        n = len(trained_series.unique()) + 1

    return __weighting_method_agg_fun(series, trained_series, n,
                                      calc_meth = 'x-inv-x')



def __weighting_method_agg_fun(series, trained_series, n, calc_meth):
    """
    Generator function for the different calcuation methods to determine
    the asset class based on a Series or DataFrame of r-squared values

    :ARGS:

        series: :class:`pandas.Series` or :class:`pandas.DataFrame` of
        r-squared values

        trained_series: :class:`pandas.Series` of the columns
        and their respective asset classes

        n: :class:`integer` of the number of highest r-squared assets
        to include when classifying a new asset

        calc_meth: :class:`string` of either ['x-inv-x', 'inv-x', 'exp-x']
        to determine which calculation method is used

    :RETURNS:

        :class:`string` of the asset class  been estimated
        based on the n closest neighbors, or 'series' in the case when
        a :class:`DataFrame` has been provided instead of a :class:`Series`

    """
    def weighting_method_agg_fun(series, trained_series, n, calc_meth):
        weight_map = {'x-inv-x': lambda x: x.div(1. - x),
                      'inv-x': lambda x: 1./(1. - x),
                      'exp-x': lambda x: numpy.exp(x)
                      }

        key_map = trained_series[series.index]
        series = series.rename(index = key_map)
        wts = weight_map[calc_meth](series)
        wts = wts.sort(ascending = False, inplace = False)
        grp = wts[:n].groupby(wts[:n].index).sum()
        return grp.argmax()

    if isinstance(series, pandas.DataFrame):
        return series.apply(
            lambda x: weighting_method_agg_fun(x,
            trained_series, n, calc_meth), axis = 1)
    else:
        return weighting_method_agg_fun(series, trained_series, n, calc_meth)

