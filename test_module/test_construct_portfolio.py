
#!/usr/bin/env python
# encoding: utf-8

"""
.. module:: visualize_wealth.test_module.test_construct_portfolio.py

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>

"""
import os
import pytest
import numpy
import pandas
import tempfile
import datetime
from pandas.util import testing
from visualize_wealth import construct_portfolio as cp

@pytest.fixture
def test_file():
    f = './test_data/panel from weight file test.xlsx'
    return pandas.ExcelFile(f)

@pytest.fixture
def tc_file():
    f = './test_data/transaction-costs.xlsx'
    return pandas.ExcelFile(f)

@pytest.fixture
def rebal_weights(test_file):
    return test_file.parse('rebal_weights', index_col = 0)

@pytest.fixture
def panel(test_file, rebal_weights):
    tickers = ['EEM', 'EFA', 'IYR', 'IWV', 'IEF', 'IYR', 'SHY']
    d = {}
    for ticker in tickers:
        d[ticker] = test_file.parse(ticker, index_col = 0)

    return cp.panel_from_weight_file(rebal_weights, 
                                  pandas.Panel(d), 
                                  1000.
    )

@pytest.fixture
def manual_index(panel, test_file):
    man_calc = test_file.parse('index_result',
                            index_col = 0
    )
    return man_calc

@pytest.fixture
def manual_tc_bps(tc_file):
    man_tcosts = tc_file.parse('tc_bps', index_col = 0)
    man_tcosts = man_tcosts.fillna(0.0)
    return man_tcosts

@pytest.fixture
def manual_tc_cps(tc_file):
    man_tcosts = tc_file.parse('tc_cps', index_col = 0)
    man_tcosts = man_tcosts.fillna(0.0)
    return man_tcosts

@pytest.fixture
def manual_mngmt_fee(tc_file):
    return tc_file.parse('mgmt_fee', index_col = 0)

def test_mngmt_fee(panel, tc_file, manual_mngmt_fee):
    index = cp.pfp_from_weight_file(panel)
    
    vw_mfee = cp.mngmt_fee(price_series = index['Close'],
                           bps_cost = 100.,
                           frequency = 'daily'
    )
    
    testing.assert_series_equal(manual_mngmt_fee['daily_index'],
                                vw_mfee
    )

def test_pfp(panel, manual_index):
    lib_calc = cp.pfp_from_weight_file(panel)
    testing.assert_series_equal(manual_index['Close'], 
                                lib_calc['Close']
    )
    return lib_calc

def test_tc_bps(rebal_weights, panel, manual_tc_bps):
    vw_tcosts = cp.tc_bps(weight_df = rebal_weights, 
                          share_panel = panel,
                          bps = 10.,
    )
    cols = ['EEM', 'EFA', 'IEF', 'IWV', 'IYR', 'SHY']
    testing.assert_frame_equal(manual_tc_bps[cols], vw_tcosts)

def test_net_bps(rebal_weights, panel, manual_tc_bps, manual_index):
    
    index = test_pfp(panel, manual_index)
    index = index['Close']

    vw_tcosts = cp.tc_bps(weight_df = rebal_weights, 
                          share_panel = panel,
                          bps = 10.,
    )

    net_tcs = cp.net_tcs(tc_df = vw_tcosts, 
                         price_index = index
    )

    testing.assert_series_equal(manual_tc_bps['adj_index'],
                                net_tcs
    )

def test_net_cps(rebal_weights, panel, manual_tc_cps, manual_index):
    index = test_pfp(panel, manual_index)
    index = index['Close']

    vw_tcosts = cp.tc_cps(weight_df = rebal_weights, 
                          share_panel = panel,
                          cps = 10.,
    )

    net_tcs = cp.net_tcs(tc_df = vw_tcosts, 
                         price_index = index
    )

    testing.assert_series_equal(manual_tc_cps['adj_index'],
                                net_tcs
    )

def test_tc_cps(rebal_weights, panel, manual_tc_cps):
    cols = ['EEM', 'EFA', 'IEF', 'IWV', 'IYR', 'SHY']
    vw_tcosts = cp.tc_cps(weight_df = rebal_weights, 
                          share_panel = panel,
                          cps = 10.,
    )

    testing.assert_frame_equal(manual_tc_cps[cols], vw_tcosts)

def test_funs():
    """
    >>> import pandas.util.testing as put
    >>> xl_file = pandas.ExcelFile('../tests/test_splits.xlsx')
    >>> blotter = xl_file.parse('blotter', index_col = 0)
    >>> cols = ['Close', 'Adj Close', 'Dividends']
    >>> price_df = xl_file.parse('calc_sheet', index_col = 0)
    >>> price_df = price_df[cols]
    >>> split_frame = calculate_splits(price_df)

    >>> shares_owned = blotter_and_price_df_to_cum_shares(blotter, 
    ...     split_frame)
    >>> test_vals = xl_file.parse(
    ...     'share_balance', index_col = 0)['cum_shares']
    >>> put.assert_almost_equal(shares_owned['cum_shares'].dropna(), 
    ...     test_vals)
    True
    >>> f = '../tests/panel from weight file test.xlsx'
    >>> xl_file = pandas.ExcelFile(f)
    >>> weight_df = xl_file.parse('rebal_weights', index_col = 0)
    >>> tickers = ['EEM', 'EFA', 'IYR', 'IWV', 'IEF', 'IYR', 'SHY']
    >>> d = {}
    >>> for ticker in tickers:
    ...     d[ticker] = xl_file.parse(ticker, index_col = 0)
    >>> panel = panel_from_weight_file(weight_df, pandas.Panel(d), 
    ...     1000.)
    >>> portfolio = pfp_from_weight_file(panel)
    >>> manual_calcs = xl_file.parse('index_result', index_col = 0)
    >>> put.assert_series_equal(manual_calcs['Close'], 
    ...     portfolio['Close'])
    """
    return None