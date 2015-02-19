#!/usr/bin/env python
# encoding: utf-8

"""
.. module:: visualize_wealth.test_module.test_analyze.py

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>

"""

import pytest
import pandas
import tempfile
import datetime
from pandas.util import testing
import visualize_wealth.utils as analyze

@pytest.fixture
def store_dir():
    return tempfile.mkstemp(suffix = '.h5', 
                            dir = '../test_data/',
                            text = False
    )

@pytest.fixture
def populaten_store(store_dir):
    store = pandas.HDFStore(store_dir[1], mode = 'w')

    #two weeks of data before today, delete one week, then update
    delta = datetime.timedelta(14)
    today = datetime.datetime.date(datetime.datetime.today())
    index = pandas.DatetimeIndex(start = today - delta,
                                 freq = 'b',
                                 periods = 10
    )

    store.put('TICK', pandas.Series(numpy.ones(len(index), ),
                                    index = index,
                                    name = 'Close')
    )

    store.put('TOCK', pandas.Series(numpy.ones(len(index), ),
                                    index = index,
                                    name = 'Close')
    )
    store.close()
    return index

def test_create_store_master_index(store_dir, populate_store):
    index = populate_store
    index = pandas.Series(index, index = index)

    master_index = utils.create_store_master_index(store_dir[1])
    testing.assert_index_equal(master_index, index)

def test_create_union_store_indexes(store_dir, populate_store):
    store = pandas.HDFStore(store_dir[1], mode = 'r+')
    index = populate_store
    union



    
    
    #create two tickers, test the master_index creation
    #test the cash creation

    #test the master_index update
    #test the cash update




@pytest.fixture
def test_file():
    return pandas.ExcelFile('./test_data/test_analyze.xlsx')

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

def test_active_return(prices, stat_calcs):
    man_ar = stat_calcs.loc['active_return', 'VGTSX']

    testing.assert_almost_equal(man_ar, analyze.active_return(
                                series = prices['VGTSX'],
                                benchmark = prices['S&P 500'],
                                freq = 'daily')
    )

def test_active_returns(man_calcs, prices):
    active_returns = analyze.active_returns(series = prices['VGTSX'], 
                                            benchmark = prices['S&P 500'])

    testing.assert_series_equal(man_calcs['Active Return'], active_returns)


