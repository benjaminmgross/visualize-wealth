#!/usr/bin/env python
# encoding: utf-8

"""
.. module:: visualize_wealth.test_module.test_analyze.py

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>

"""

import pytest
import pandas
from pandas.util import testing
import visualize_wealth.analyze as analyze

@pytest.fixture
def test_file():
    return pandas.ExcelFile('../test_data/test_analyze.xlsx')

@pytest.fixture
def man_calcs(test_file):
    return test_file.parse('calcs', index_col = 0)

@pytest.fixture
def stat_calcs(test_file):
    return test_file.parse('results', index_col = 0)

@pytest.fixture
def prices(test_file):
    tmp = test_file.parse('calcs', index_col = 0)
    return tmp[['S&P 500', 'VGTSX']]

def test_active_returns(man_calcs, prices):
    active_returns = analyze.active_returns(series = prices['VGTSX'], 
                                            benchmark = prices['S&P 500'])

    testing.assert_series_equal(man_calcs['Active Return'], active_returns)

def test_log_returns(man_calcs, prices):
    testing.assert_series_equal(man_calcs['S&P 500 Log Ret'],
                                analyze.log_returns(prices['S&P 500'])
    )

def test_linear_returns(man_calcs, prices):
    testing.assert_series_equal(man_calcs['S&P 500 Lin Ret'],
                                analyze.linear_returns(prices['S&P 500'])
    )

def test_drawdown(man_calcs, prices):
    testing.assert_series_equal(man_calcs['VGTSX Drawdown'],
                                analyze.drawdown(prices['VGTSX'])
    )

def test_rsq(man_calcs, prices):
    log_rets = analyze.log_returns(prices).dropna()
    pandas_rsq = pandas.ols(x = log_rets['S&P 500'], 
                            y = log_rets['VGTSX']).r2

    analyze_rsq = analyze.r2(benchmark = log_rets['S&P 500'], 
                             series = log_rets['VGTSX'])

    testing.assert_almost_equal(pandas_rsq, analyze_rsq)

def test_rsq_adj(man_calcs, prices):
    log_rets = analyze.log_returns(prices).dropna()
    pandas_rsq = pandas.ols(x = log_rets['S&P 500'], 
                            y = log_rets['VGTSX']).r2_adj

    analyze_rsq = analyze.r2_adj(benchmark = log_rets['S&P 500'], 
                             series = log_rets['VGTSX'])

    testing.assert_almost_equal(pandas_rsq, analyze_rsq)

def test_cumulative_turnover(test_file, stat_calcs):
    alloc_df = test_file.parse('alloc_df', index_col = 0)
    cols = alloc_df.columns[alloc_df.columns!='Daily TO']
    alloc_df = alloc_df[cols].dropna()
    asset_wt_df = test_file.parse('asset_wt_df', index_col = 0)
    testing.assert_almost_equal(analyze.cumulative_turnover(alloc_df, asset_wt_df), 
                                stat_calcs.loc['cumulative_turnover', 'S&P 500']
    )

def test_marginal_contribution_to_risk(test_file):
    mctr_prices = test_file.parse('mctr', index_col = 0)
    mctr_manual = test_file.parse('mctr_results', index_col = 0)
    cols = ['BSV','VBK','VBR','VOE','VOT']
    mctr = analyze.mctr(mctr_prices[cols], mctr_prices['Portfolio'])
    testing.assert_series_equal(mctr, mctr_manual.loc['mctr', cols])

def test_risk_contribution(test_file):
    mctr_prices = test_file.parse('mctr', index_col = 0)
    mctr_manual = test_file.parse('mctr_results', index_col = 0)
    cols = ['BSV','VBK','VBR','VOE','VOT']
    mctr = analyze.mctr(mctr_prices[cols], mctr_prices['Portfolio'])
    weights = pandas.Series( [.2, .2, .2, .2, .2], index = cols, name = 'risk_contribution')
    
    testing.assert_series_equal(analyze.risk_contribution(mctr, weights), 
                             mctr_manual.loc['risk_contribution', :]
    )

def test_risk_contribution_as_proportion(test_file):
    mctr_prices = test_file.parse('mctr', index_col = 0)
    mctr_manual = test_file.parse('mctr_results', index_col = 0)
    cols = ['BSV','VBK','VBR','VOE','VOT']
    mctr = analyze.mctr(mctr_prices[cols], mctr_prices['Portfolio'])
    weights = pandas.Series( [.2, .2, .2, .2, .2], index = cols, name = 'risk_contribution')
    
    testing.assert_series_equal(
        analyze.risk_contribution_as_proportion(mctr, weights),
        mctr_manual.loc['risk_contribution_as_proportion']
    )

def test_alpha(prices, stat_calcs):
    man_alpha = stat_calcs.loc['alpha', 'VGTSX']

    testing.assert_almost_equal(man_alpha, analyze.alpha(series = prices['VGTSX'],
                                                         benchmark = prices['S&P 500'])
    )

def test_annualized_return(prices, stat_calcs):
    man_ar = stat_calcs.loc['annualized_return', 'VGTSX']
    
    testing.assert_almost_equal(
        man_ar, analyze.annualized_return(series = prices['VGTSX'], freq = 'daily')
    )

def test_annualized_vol(prices, stat_calcs):
    man_ar = stat_calcs.loc['annualized_vol', 'VGTSX']
    
    testing.assert_almost_equal(
        man_ar, analyze.annualized_vol(series = prices['VGTSX'], freq = 'daily')
    )

def test_beta(prices, stat_calcs):
    man_beta = stat_calcs.loc['beta', 'VGTSX']

    testing.assert_almost_equal(man_beta, analyze.beta(series = prices['VGTSX'],
                                                       benchmark = prices['S&P 500'])
    )

def test_cvar_cf(prices, stat_calcs):
    man_cvar_cf = stat_calcs.loc['cvar_cf', 'VGTSX']

    testing.assert_almost_equal(
        man_cvar_cf, analyze.cvar_cf(series = prices['VGTSX'], p = 0.01)
    )

def test_cvar_norm(prices, stat_calcs):
    man_cvar_norm = stat_calcs.loc['cvar_norm', 'VGTSX']

    testing.assert_almost_equal(
        man_cvar_norm, analyze.cvar_norm(series = prices['VGTSX'], p = 0.01)
    )

def test_downcapture(prices, stat_calcs):
    man_dc = stat_calcs.loc['downcapture', 'VGTSX']

    testing.assert_almost_equal(
        man_dc, analyze.downcapture(series = prices['VGTSX'], 
                                    benchmark = prices['S&P 500'])
    )

def test_downside_deviation(prices, stat_calcs):
    man_dd = stat_calcs.loc['downside_deviation', 'VGTSX']

    testing.assert_almost_equal(
        man_dd, analyze.downside_deviation(series = prices['VGTSX'])
    )

def test_geometric_difference():
    a, b = 1. , 1.
    assert analyze.geometric_difference(a, b) == 0.
    a, b = pandas.Series({'a': 1.}), pandas.Series({'a': 1.})
    assert analyze.geometric_difference(a, b).values == 0.

def test_idiosyncratic_as_proportion(prices, stat_calcs):
    man_iap = stat_calcs.loc['idiosyncratic_as_proportion', 'VGTSX']

    testing.assert_almost_equal(
        man_iap, analyze.idiosyncratic_as_proportion(
            series = prices['VGTSX'], benchmark = prices['S&P 500'])
    )

def test_idiosyncratic_risk(prices, stat_calcs):
    man_ir = stat_calcs.loc['idiosyncratic_risk', 'VGTSX']

    testing.assert_almost_equal(
        man_ir, analyze.idiosyncratic_risk(
            series = prices['VGTSX'], benchmark = prices['S&P 500'])
    )


def test_funs():
    """
    The testing function for ``analyze.py``

    >>> import pandas.util.testing as put
    >>> import inspect, analyze

    >>> f = pandas.ExcelFile('../test_data/test_analyze.xlsx')
    >>> man_calcs = f.parse('calcs', index_col = 0)
    >>> prices = man_calcs[['S&P 500', 'VGTSX']]
    >>> log_rets = prices.apply(numpy.log).diff().dropna()
    >>> stats = f.parse('results', index_col = 0)

    These functions are already calculated or aren't calculated in the spreadsheet

    >>> no_calc_list = ['rolling_ui', 'active_returns',
    ... 'test_funs', 'linear_returns', 'log_returns',
    ... 'cumulative_turnover', 'mctr', 'risk_contribution',
    ... 'risk_contribution_as_proportion', 'cvar_cf_ew', 'cvar_median_np',
    ... 'cvar_mu_np', 'var_np', 'var_cf', 'var_norm', 'consecutive',
    ... 'consecutive_downticks', 'consecutive_upticks', 
    ... 'consecutive_downtick_performance', 'consecutive_uptick_performance',
    ... 'consecutive_downtick_relative_performance', 
    ... 'consecutive_uptick_relative_performance',
    ... 'r_squared', 'r_squared_adjusted', 'drawdown', 'r2', 'r2_adj',
    ... 'r2_mv', 'r2_mv_adj']

    >>> d = {'series': prices['VGTSX'], 'benchmark':prices['S&P 500'], 
    ...    'freq': 'daily', 'rfr': 0.0, 'p': .01 }
    >>> funs = inspect.getmembers(analyze, inspect.isfunction)

    Instead of writing out each function, I use the ``inspect module`` to determine
    Function names, and number of args.  Because the names of the functions are
    the same as the statistic in the ``results`` tab of
    ``../test_data/test_analyze.xlsx`` I can use those key values to both call the
    function and reference the manually calculated value in the ``stats`` frame.

    >>> trus = []
    >>> for fun in funs:
    ...    if (fun[0][0] != '_') and (fun[0] not in no_calc_list):
    ...        arg_list = inspect.getargspec(fun[1]).args
    ...        in_vals = tuple([d[arg] for arg in arg_list])
    ...        numpy.testing.assert_almost_equal(fun[1](*in_vals),
    ...                                  stats.loc[fun[0], 'VGTSX'])

    """
    return None
