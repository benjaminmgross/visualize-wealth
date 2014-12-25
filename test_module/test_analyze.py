#!/usr/bin/env python
# encoding: utf-8

import pytest
import from pandas.util import testing
import visualize_wealth.analze as analyze

@pytest.fixture
def man_calcs():
	f = pandas.ExcelFile('../test_data/test_analyze.xlsx')
    return f.parse('calcs', index_col = 0)

@pytest.fixture
def prices():
	f = pandas.ExcelFile('../test_data/test_analyze.xlsx')
    tmp = f.parse('calcs', index_col = 0)
    return tmp[['S&P 500', 'VGTSX']]

def test_active_returns(man_calcs):
	return None

def test_log_returns(man_cals, prices):

	testing.assert_series_equal(man_calcs['S&P 500 Log Ret'],
								analyze.log_returns(prices['S&P 500'])
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
    >>> put.assert_series_equal(man_calcs['S&P 500 Log Ret'], 
    ...    log_returns(prices['S&P 500']))
    >>> put.assert_series_equal(man_calcs['VGTSX Lin Ret'], 
    ...    linear_returns(prices['VGTSX']))
    >>> put.assert_series_equal(man_calcs['Active Return'],
    ...    active_returns(series = prices['VGTSX'], 
    ...    benchmark = prices['S&P 500']))
    >>> put.assert_series_equal(man_calcs['VGTSX Drawdown'],
    ...    drawdown(prices['VGTSX']))
    >>> numpy.testing.assert_almost_equal(pandas.ols(x = log_rets['S&P 500'],
    ...    y = log_rets['VGTSX']).r2, r2(benchmark = log_rets['S&P 500'],
    ...    series = log_rets['VGTSX']))
    >>> numpy.testing.assert_almost_equal(pandas.ols(x = log_rets['S&P 500'],
    ...    y = log_rets['VGTSX']).r2_adj, r2_adj(benchmark = log_rets['S&P 500'],
    ...    series = log_rets['VGTSX']))

    Cumulative Turnover Calculation

    >>> alloc_df = f.parse('alloc_df', index_col = 0)
    >>> alloc_df = alloc_df[alloc_df.columns[alloc_df.columns!='Daily TO']].dropna()
    >>> asset_wt_df = f.parse('asset_wt_df', index_col = 0)
    >>> numpy.testing.assert_almost_equal(
    ...    cumulative_turnover(alloc_df, asset_wt_df), 
    ...    stats.loc['cumulative_turnover', 'S&P 500'])

    marginal contributions to risk and risk contribution calcs

    >>> mctr_prices = f.parse('mctr', index_col = 0)
    >>> mctr_manual = f.parse('mctr_results', index_col = 0)
    >>> cols = ['BSV','VBK','VBR','VOE','VOT']
    >>> mctr = mctr(mctr_prices[cols], mctr_prices['Portfolio'])
    >>> put.assert_series_equal(mctr, mctr_manual.loc['mctr', cols])
    >>> weights = pandas.Series( [.2, .2, .2, .2, .2], index = cols, 
    ... name = 'risk_contribution')
    >>> put.assert_series_equal(risk_contribution(mctr, weights), 
    ... mctr_manual.loc['risk_contribution', :])
    >>> put.assert_series_equal(risk_contribution_as_proportion(mctr, weights),
    ... mctr_manual.loc['risk_contribution_as_proportion'])

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
