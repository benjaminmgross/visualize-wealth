#!/usr/bin/env python
# encoding: utf-8

"""
.. module:: visualize_wealth.test_module.test_utils.py

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>

"""
import os
import pytest
import numpy
import pandas
import tempfile
import datetime

from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal,
                                 assert_index_equal,
                                 assert_almost_equal
)

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

@pytest.fixture
def populate_updated():
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

    #truncate the index for updating
    ind = index[:5]
    n = len(ind)

    #store the Master IND3X
    store.put('IND3X', pandas.Series(ind, 
                                     index = ind)
    )

    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    cash = pandas.DataFrame(numpy.ones([n, len(cols)]),
                            index = ind,
                            columns = cols
    )

    #store the CA5H
    store.put('CA5H', cash)
    store.close()
    return {'name': name, 'index': index}


def test_create_store_master_index(populate_store):
    index = populate_store['index']
    index = pandas.Series(index, index = index)

    utils.create_store_master_index(populate_store['name'])
    store = pandas.HDFStore(populate_store['name'], mode = 'r+')
    assert_series_equal(store.get('IND3X'), index)
    store.close()
    os.remove(populate_store['name'])


def test_union_store_indexes(populate_store):
    store = pandas.HDFStore(populate_store['name'], mode = 'r+')
    index = populate_store['index']
    union = utils.union_store_indexes(store)
    assert_index_equal(index, union)
    store.close()
    os.remove(populate_store['name'])


def test_create_store_cash(populate_store):
    index = populate_store['index']
    utils.create_store_cash(populate_store['name'])
    store = pandas.HDFStore(populate_store['name'], mode = 'r+')
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    n = len(index)

    cash = pandas.DataFrame(numpy.ones([n, len(cols)]),
                            index = index,
                            columns = cols
    )

    assert_frame_equal(store.get('CA5H'), cash)
    store.close()
    os.remove(populate_store['name'])


def test_update_store_master_and_cash(populate_updated):
    index = populate_updated['index']
    index = pandas.Series(index, index = index)

    utils.update_store_master_index(populate_updated['name'])
    utils.update_store_cash(populate_updated['name'])

    store = pandas.HDFStore(populate_updated['name'], mode = 'r+')
    assert_series_equal(store.get('IND3X'), index)

    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    n = len(index)
    cash = pandas.DataFrame(numpy.ones([n, len(cols)]),
                            index = index,
                            columns = cols
    )

    assert_frame_equal(store.get('CA5H'), cash)
    store.close()
    os.remove(populate_updated['name'])



def test_rets_to_price():
    dts = ['1/1/2000', '1/2/2000', '1/3/2000']

    index = pandas.DatetimeIndex(
                pandas.Timestamp(dt) for dt in dts
    )

    series = pandas.Series([numpy.nan, 0., 0.], 
                           index = index
    )

    log = utils.rets_to_price(
            series, 
            ret_typ = 'log', 
            start_value = 100.
    )

    lin = utils.rets_to_price(
            series, 
            ret_typ = 'linear', 
            start_value = 100.
    )
    
    man = pandas.Series([100., 100., 100.], 
                        index = index
    )

    assert_series_equal(log, man)
    assert_series_equal(lin, man)

    df = pandas.DataFrame({'a': series, 'b': series})
    log = utils.rets_to_price(
            df, 
            ret_typ = 'log', 
            start_value = 100.
    )
    
    lin = utils.rets_to_price(
            df, 
            ret_typ = 'linear', 
            start_value = 100.
    )
    
    man = pandas.DataFrame({'a': man, 'b': man})

    assert_frame_equal(log, man)
    assert_frame_equal(lin, man)

    with pytest.raises(TypeError):
        utils.rets_to_price(pandas.Panel(), 
                            ret_typ = 'log', 
                            start_value = 100.
        )

#@pytest.mark.newtest
def test_strip_vals():
    l = [' TLT', ' HYY ', 'IEF ']
    strpd = utils.strip_vals(l)
    res = ['TLT', 'HYY', 'IEF']
    assert strpd == res

@pytest.mark.newtest
def test_zipped_time_chunks():
    pts = pandas.Timestamp

    index = pandas.DatetimeIndex(
                start = '06/01/2000',
                freq = 'D',
                periods = 100
    )
    res = [('06-01-2000', '06-30-2000'), 
           ('07-01-2000', '07-31-2000'), 
           ('08-01-2000', '08-31-2000')]

    mc = list(((pts(x), pts(y)) for x, y in res))
    lc = utils.zipped_time_chunks(
            index = index,
            interval = 'monthly',
            incl_T = False
    )
    assert mc == lc

    res = [('06-01-2000', '06-30-2000'), 
           ('07-01-2000', '07-31-2000'), 
           ('08-01-2000', '08-31-2000'),
           ('09-01-2000', '09-08-2000')]

    mc = list(((pts(x), pts(y)) for x, y in res))
    lc = utils.zipped_time_chunks(
            index = index,
            interval = 'monthly',
            incl_T = True
    )
    assert mc == lc

    res = [('06-01-2000', '06-30-2000')]
    mc = list(((pts(x), pts(y)) for x, y in res))
    lc = utils.zipped_time_chunks(
            index = index,
            interval = 'quarterly',
            incl_T = False
    )
    assert mc == lc

    res = [('06-01-2000', '06-30-2000')]
    mc = list(((pts(x), pts(y)) for x, y in res))
    lc = utils.zipped_time_chunks(
            index = index,
            interval = 'quarterly',
            incl_T = False
    )
    assert mc == lc

    res = [('06-01-2000', '06-30-2000'),
           ('07-01-2000', '09-08-2000')]
    mc = list(((pts(x), pts(y)) for x, y in res))
    lc = utils.zipped_time_chunks(
            index = index,
            interval = 'quarterly',
            incl_T = True
    )
    assert mc == lc

    mc = []
    lc = utils.zipped_time_chunks(
            index = index,
            interval = 'yearly',
            incl_T = False
    )
    assert mc == lc

    res = [('06-01-2000', '09-08-2000')]
    mc = list(((pts(x), pts(y)) for x, y in res))
    lc = utils.zipped_time_chunks(
            index = index,
            interval = 'yearly',
            incl_T = True
    )
    assert mc == lc

"""
def test_update_store_cash(populate_updated):
    index = populate_updated['index']

    utils.update_store_cash(populate_updated['name'])
    store = pandas.HDFStore(populate_updated['name'], mode = 'r+')
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    n = len(index)
    cash = pandas.DataFrame(numpy.ones([n, len(cols)]),
                            index = index,
                            columns = cols
    )

    assert_frame_equal(store.get('CA5H'), cash)
    store.close()
    os.remove(populate_updated['name'])
"""