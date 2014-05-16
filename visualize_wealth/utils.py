#!/usr/bin/env python
# encoding: utf-8
"""
utils.py

Created by Benjamin Gross on 5/15/2014

A myriad of different utility functions often useful in the manipulation
construction, & examination of useful financial data
"""

import argparse
import pandas
import numpy

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
    
    
    if ind[0] == True: #The series started on the last day of period
        index = index.copy()[1:] #So we can't get a Period
        ind = time_d[interval](index[:-1]) != time_d[interval](index[1:])

    ldop = index[ind]
    fdop = index[numpy.append(True, ind[:-1])]
    return zip(fdop, ldop)

def scipt_function(arg_1, arg_2):
	return None

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
