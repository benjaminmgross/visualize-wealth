#!/usr/bin/env python
# encoding: utf-8

"""
.. module:: visualize_wealth.test_module.test_analyze.py

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>

"""
import os
import pytest
import numpy
import pandas
import tempfile
import datetime
from pandas.util import testing
import visualize_wealth.utils as utils


@pytest.fixture
def populate_store():
    name = './test_data/tmp.h5'
    store = pandas.HDFStore(name, mode = 'w')

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
    return {'name': name, 'index': index}

def test_create_store_master_index(populate_store):
    index = populate_store['index']
    index = pandas.Series(index, index = index)

    utils.create_store_master_index(populate_store['name'])
    store = pandas.HDFStore(populate_store['name'], mode = 'r+')
    testing.assert_series_equal(store.get('IND3X'), index)
    store.close()
    os.remove(populate_store['name'])

def test_union_store_indexes(populate_store):
    store = pandas.HDFStore(populate_store['name'], mode = 'r+')
    index = populate_store['index']
    union = utils.union_store_indexes(store)
    testing.assert_index_equal(index, union)
    store.close()
    os.remove(populate_store['name'])

#def test_create_store_cash(store_dir
