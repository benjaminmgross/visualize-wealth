#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: analyze.py

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>

"""
import numpy
import pandas
import collections

def active_returns(series, benchmark):
    """
    Active returns is defined as the compound difference between linear returns i.e.

    :ARGS:

        series: ``pandas.Series`` of prices of the portfolio

        benchmark: ``pandas.Series`` of prices of the benchmark

    :RETURNS: 

        ``pandas.Series`` of active returns

    .. note:: Compound Linear Returns

         Linear returns are not simply subtracted, but rather the compound
         difference is taken such that

        .. math::

            r_a = (1 + r_p)/(1 + r_b) - 1
    """
    def _active_returns(series, benchmark):
        return (1 + linear_returns(series)).div(1 + linear_returns(benchmark)) - 1 
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _active_returns(series, x))
    else:
        return _active_returns(series, benchmark)

def alpha(series, benchmark, freq = 'daily', rfr = 0.0):
    """
    Alpha is defined as simply the geometric difference between the return of a 
    portfolio and a benchmark, less the risk free rate or:

    .. math::

        \\alpha = \\frac{(1 + R_p - r_f)}{(1 + R_b - r_f)}  - 1 \\\\
        
        \\textrm{where},

            R_p &= \\textrm{Portfolio Annualized Return} \\\\
            R_b &= \\textrm{Benchmark Annualized Return} \\\\
            r_f &= \\textrm{Risk Free Rate}
    """
    def _alpha(series, benchmark, freq = 'daily', rfr = 0.0):
        return numpy.divide(1 + annualized_return(series, freq = freq) - rfr,
                        1 + annualized_return(benchmark, freq = freq) - rfr) - 1

    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _alpha(series, x, freq = freq, rfr = rfr))
    else:
        return _alpha(series, benchmark, freq = freq, rfr = rfr)



def annualized_return(series, freq = 'daily'):
    """
    Returns the annualized linear return of a series, i.e. the linear compounding
    rate that would have been necessary, given the initial investment, to arrive at
    the final value

    :ARGS:
    
        series: ``pandas.Series`` of prices
        
        freq: ``str`` of either ``daily, monthly, quarterly, or yearly``    
        indicating the frequency of the data ``default=`` daily

    :RETURNS:
    
        ``float``: of the annualized linear return

    .. code:: python

        import visualize_wealth.performance as vwp

        linear_return = vwp.annualized_return(price_series, frequency = 'monthly')
    
    """
    def _annualized_return(series, freq = 'daily'):
        fac = _interval_to_factor(freq)
        series_rets = log_returns(series)
        return numpy.exp(series_rets.mean()*fac) - 1
    if isinstance(series, pandas.DataFrame):
        return series.apply(lambda x: _annualized_return(x, freq = freq))
    else:
        return _annualized_return(series, freq = freq)

def annualized_vol(series, freq = 'daily'):
    """
    Returns the annlualized volatility of the log changes of the price series, by
    calculating the volatility of the series, and then applying the square root of 
    time rule

    :ARGS:
    
        series: ``pandas.Series`` of prices

        freq: ``str`` of either ``daily, monthly, quarterly, or yearly``    
        indicating the frequency of the data ``default=`` daily

    :RETURNS:
    
        float: of the annualized volatility

    .. note:: Applying the Square root of time rule


        .. math::

            \\sigma = \\sigma_t \\cdot \\sqrt{k},\\: \\textrm{where},

            k &= \\textrm{Factor of annualization} \\\\
            \\sigma_t &= \\textrm{volatility of period log returns}
        
    .. code::

        import visualize_wealth.performance as vwp

        ann_vol = vwp.annualized_vol(price_series, frequency = 'monthly')
    """
    def _annualized_vol(series, freq = 'daily'):
        fac = _interval_to_factor(freq)
        series_rets = log_returns(series)
        return series_rets.std()*numpy.sqrt(fac)
    if isinstance(series, pandas.DataFrame):
        return series.apply(lambda x: _annualized_vol(x, freq = freq))
    else:
        return _annualized_vol(series, freq = freq)

def beta(series, benchmark):
    """
    Returns the sensitivity of one price series to a chosen benchmark:

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` of a benchmark to calculate the sensitivity

    :RETURNS:

        float: the sensitivity of the series to the benchmark

    .. note:: Calculating Beta

        
        .. math::

           \\beta \\triangleq \\frac{\\sigma_{s, b}}{\\sigma^2_{b}},
           \\: \\textrm{where},

           \\sigma^2_{b} &= \\textrm{Variance of the Benchmark} \\\\
           \\sigma_{s, b} &= \\textrm{Covariance of the Series & Benchmark}
    
    """
    def _beta(series, benchmark):
        series_rets = log_returns(series)
        bench_rets = log_returns(benchmark)
        return numpy.divide(bench_rets.cov(series_rets), bench_rets.var())
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _beta(series, x))
    else:
        return _beta(series, benchmark)

def downcapture(series, benchmark):
    """
    Returns the proportion of ``series``'s cumulative negative returns to
    ``benchmark``'s cumulative  returns, given benchmark's returns were negative in
    that period

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` of prices to compare ``series`` against

    :RETURNS:

        ``float`` of the downcapture of cumulative positive ret

    .. seealso:: :py:data:`median_downcapture(series, benchmark)`

    """
    def _downcapture(series, benchmark):
        series_rets = log_returns(series)
        bench_rets = log_returns(benchmark)
        index = bench_rets < 0.
        return series_rets[index].mean() / bench_rets[index].mean()
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _downcapture(series, x))
    else:
        return _downcapture(series, benchmark)

def downside_deviation(series, freq = 'daily'):
    """
    Returns the volatility of the returns that are less than zero

    :ARGS:

        series:``pandas.Series`` of prices

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    :RETURNS:

        float: of the downside standard deviation

    """
    def _downside_deviation(series, freq = 'daily'):
        fac = _interval_to_factor(freq)
        series_rets = log_returns(series)
        index = series_rets < 0.    
        return series_rets[index].std()*numpy.sqrt(fac)
    if isinstance(series, pandas.DataFrame):
        return series.apply(lambda x: _downside_deviation(x, freq = freq))
    else:
        return _downside_deviation(series, freq = freq)

def idiosyncratic_as_proportion(series, benchmark, freq = 'daily'):
    """
    Returns the idiosyncratic risk as proportion of total volatility

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` to compare ``series`` against

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    :RETURNS:

        ``float`` between (0, 1) representing the proportion of  volatility
        represented by idiosycratic risk
        
    """
    def _idiosyncratic_as_proportion(series, benchmark, freq = 'daily'):
        fac = _interval_to_factor(freq)
        series_rets = log_returns(series)
        return idiosyncratic_risk(series, benchmark, freq)**2 / (
            annualized_vol(series, freq)**2)
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _idiosyncratic_as_proportion(series, x, 
                                freq))
    else:
        return _idiosyncratic_as_proportion(series, benchmark, freq)

def idiosyncratic_risk(series, benchmark, freq = 'daily'):
    """
    Returns the idiosyncratic risk, i.e. unexplained variation between a price series
    and a chosen benchmark 

    :ARGS:

       series: ``pandas.Series`` of prices

       benchmark: ``pandas.Series`` to compare ``series`` against

       freq: ``str`` of frequency, either ``daily, monthly, quarterly, or yearly``

    :RETURNS:

        float: the idiosyncratic volatility (not variance)


    .. note:: Additivity of an asset's Variance

        An asset's variance can be broken down into systematic risk, i.e. that 
        proportion of risk that can be attributed to some benchmark or risk factor
        and idiosyncratic risk, or the unexplained variation between the series and 
        the chosen benchmark / factor.  

        Therefore, using the additivity of variances, we can calculate idiosyncratic
        risk as follows:

       .. math::

           \\sigma^2_{\\textrm{total}} = \\sigma^2_{\\beta} + \\sigma^2_{\\epsilon} +
           \\sigma^2_{\\epsilon, \\beta}, \\: \\textrm{where}, 

           \\sigma^2_{\\beta} &= \\textrm{variance attributable to systematic risk}
           \\\\
           \\sigma^2_{\\epsilon} &= \\textrm{idiosyncratic risk} \\\\
           \\sigma^2_{\\epsilon, \\beta} &= \\textrm{covariance between idiosyncratic
           and systematic risk, which by definition} = 0 \\\\

           \\Rightarrow \\sigma_{\\epsilon} = \\sqrt{\\sigma^2_{\\beta} + 
           \\sigma^2_{\\epsilon, \\beta}}

    """
    def _idiosyncratic_risk(series, benchmark, freq = 'daily'):
        fac = _interval_to_factor(freq)
        series_rets =log_returns(series)
        bench_rets = log_returns(benchmark)
        series_vol = annualized_vol(series, freq)
        benchmark_vol = annualized_vol(benchmark, freq)
        
        return numpy.sqrt(series_vol**2 - beta(series, benchmark)**2 * (
            benchmark_vol ** 2))

    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _idiosyncratic_risk(series, x, freq = freq))
    else:
        return _idiosyncratic_risk(series, benchmark, freq)

def jensens_alpha(series, benchmark, rfr = 0., freq = 'daily'):
    """
    Returns the `Jensen's Alpha <http://en.wikipedia.org/wiki/Jensen's_alpha>`_ or 
    the excess return based on the systematic risk of the ``series`` relative to
    the ``benchmark``

    :ARGS:

         series: ``pandas.Series`` of prices 

         benchmark: ``pandas.Series`` of the prices to compare ``series`` against

         rfr: ``float`` of the risk free rate

         freq: ``str`` of frequency, either daily, monthly, quarterly, or yearly

    :RETURNS:

        ``float`` representing the Jensen's Alpha

    .. note:: Calculating Jensen's Alpha

        .. math::

            \\alpha_{\\textrm{Jensen}} = r_p - \\beta \\cdot r_b 
            \\: \\textrm{where}

            r_p &= \\textrm{annualized linear return of the portfolio} \\\\
            \\beta &= \\frac{\\sigma_{s, b}}{\\sigma^2_{b}} \\\\
            r_b &= \\textrm{annualized linear return of the benchmark}

    """
    def _jensens_alpha(series, benchmark, rfr = 0., freq = 'daily'):
        fac = _interval_to_factor(freq)
        series_ret = annualized_return(series, freq)
        bench_ret = annualized_return(benchmark, freq)
        return series_ret - (rfr + beta(series, benchmark)*(bench_ret - rfr))

    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _jensens_alpha(series, x, rfr = rfr, 
            freq = freq))
    else:
        return _jensens_alpha(series, benchmark, rfr = rfr, freq = freq)

def linear_returns(series):
    """
    Returns a series of linear returns given a series of prices
    
    :ARGS:

        series: ``pandas.Series`` of prices

    :RETURNS:

        series: ``pandas.Series`` of linear returns

    .. note:: Calculating Linear Returns

            .. math::

                R_t = \\frac{P_{t+1}}{P_t} - 1
             
    """
    def _linear_returns(series):
        return series.div(series.shift(1)) - 1
    if isinstance(series, pandas.DataFrame):
        return series.apply(_linear_returns)
    else:
        return _linear_returns(series)    

def log_returns(series):
    """
    Returns a series of log returns given a series of prices where

    :ARGS:

        series: ``pandas.Series`` of prices

    :RETURNS:

        series: ``pandas.Series`` of log returns 

    .. note:: Calculating Log Returns

        .. math::

            R_t = \\log(\\frac{P_{t+1}}{P_t})
         
    """
    def _log_returns(series):
        return series.apply(numpy.log).diff()
    if isinstance(series, pandas.DataFrame):
        return series.apply(_log_returns)
    else:
        return _log_returns(series)

def max_drawdown(series):
    """
    Returns the maximum drawdown, or the maximum peak to trough linear distance, as 
    a positive drawdown value

    :ARGS:
    
        series: ``pandas.Series`` of prices

    :RETURNS:
    
        float: the maximum drawdown of the period, expressed as a positive number

    .. code::

        import visualize_wealth.performance as vwp

        max_dd = vwp.max_drawdown(price_series)
        """
    def _max_drawdown(series):
        return numpy.max(1 - series/series.cummax())
    if isinstance(series, pandas.DataFrame):
        return series.apply(_max_drawdown)
    else:
        return _max_drawdown(series)

def mean_absolute_tracking_error(series, benchmark, freq = 'daily'):
    """
    Returns Carol Alexander's calculation for Mean Absolute Tracking Error
    ("MATE").


    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` to compare ``series`` against

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly`` 


    :RETURNS:

        ``float`` of the mean absolute tracking error
        
    .. note:: Why Mean Absolute Tracking Error

        One of the downfalls of 
        `Tracking Error <http://en.wikipedia.org/wiki/Tracking_error>`_ ("TE") is
        that diverging price series that diverge at a constant rate **may** have low 
        TE.  MATE addresses this issue.
        
        .. math::
    
           \\sqrt{\\frac{(T-1)}{T}\\cdot \\tau^2 + \\bar{R}} \\: \\textrm{where}

           \\tau &= \\textrm{Tracking Error} \\\\
           \\bar{R} &= \\textrm{mean of the active returns}

    """
    def _mean_absolute_tracking_error(series, benchmark, freq = 'daily'):
        active_rets = active_returns(series = series, benchmark = benchmark)
        N = active_rets.shape[0]
        return numpy.sqrt((N - 1)/float(N) * tracking_error(series, benchmark, 
                freq)**2 + active_rets.mean()**2)

    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _mean_absolute_tracking_error(series, x,
                               freq = freq))
    else:
        return _mean_absolute_tracking_error(series, benchmark, freq = freq)

def median_downcapture(series, benchmark):
    """
    Returns the median downcapture of a ``series`` of prices against a ``benchmark`` 
    prices, given that the ``benchmark`` achieved negative returns in a given period

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` of prices to compare ``series`` against

    :RETURNS:

        ``float`` of the median downcapture

    .. warning:: About Downcapture

        Upcapture can be a difficult statistic to ensure validity.  As upcapture
        is :math:`\\frac{\\sum{r_{\\textrm{series}}}}{\\sum{r_{b|r_i \\geq 0}}}` or 
        the median values (in this case), dividing by small numbers can have 
        asymptotic effects to the overall value of this statistic.  Therefore, it's 
        good to do a "sanity check" between ``median_upcapture`` and ``upcapture``
    
    """
    def _median_downcapture(series, benchmark):
        series_rets = log_returns(series)
        bench_rets = log_returns(benchmark)
        index = bench_rets < 0.
        return series_rets[index].median() / bench_rets[index].median()
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _median_downcapture(series, x))
    else:
        return _median_downcapture(series, benchmark)

def median_upcapture(series, benchmark):
    """
    Returns the median upcapture of a ``series`` of prices against a ``benchmark`` 
    prices, given that the ``benchmark`` achieved positive returns in a given period

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` of prices to compare ``series`` against

    :RETURNS:

        float: of the median upcapture 

    .. warning:: About Upcapture

        Upcapture can be a difficult statistic to ensure validity.  As upcapture
        is :math:`\\frac{\\sum{r_{\\textrm{series}}}}{\\sum{r_{b|r_i \\geq 0}}}` or 
        the median values (in this case), dividing by small numbers can have 
        asymptotic effects to the overall value of this statistic.  Therefore, it's 
        good to do a "sanity check" between ``median_upcapture`` and ``upcapture``
        
    """
    def _median_upcapture(series, benchmark):
        series_rets = log_returns(series)
        bench_rets = log_returns(benchmark)
        index = bench_rets > 0.
        return series_rets[index].median() / bench_rets[index].median()
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _median_upcapture(series, x))
    else:
        return _median_upcapture(series, benchmark)

def r_squared(series, benchmark):
    """
    Returns the R-Squared or `Coefficient of Determination <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_ by squaring the
    correlation coefficient of the returns of the two series

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series of prices to regress ``series`` against

    :RETURNS:

        float: of the coefficient of variation

    .. note:: Calculating R-Squared

        .. math:: 

           R^2 = \\rho_{s, b}^2 \\: \\textrm{where},

           \\rho_{s, b} = \\textrm{correlation between series and benchmark log
           returns}

    """
    def _r_squared(series, benchmark):
        series_rets = log_returns(series)
        bench_rets = log_returns(benchmark)        
        return series_rets.corr(bench_rets)**2

    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _r_squared(series, x))
    else:
        return _r_squared(series, benchmark)

def risk_adjusted_excess_return(series, benchmark, rfr = 0., freq = 'daily'):
    """
    Returns the MMRAP or the `Modigliani Risk Adjusted Performance <http://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance>`_ that
    calculates the excess return from the `Capital
    Allocation Line <http://en.wikipedia.org/wiki/Capital_allocation_line>`_,
    at the same level of risk (or volatility), specificaly,
        
    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` from which to compare ``series``

        rfr: ``float`` of the risk free rate

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    :RETURNS:

        ``float`` of the risk adjusted excess performance

    .. note:: Calculating Risk Adjusted Excess Returns

        .. math::
    
            raer = r_p - \\left(\\textrm{SR}_b \\cdot \\sigma_p + r_f\\right), \\:
            \\textrm{where},

            r_p &= \\textrm{annualized linear return} \\\\
            \\textrm{SR}_b &= \\textrm{Sharpe Ratio of the benchmark} \\\\
            \\sigma_p &= \\textrm{volatility of the portfolio} \\\\
            r_f &= \\textrm{Risk free rate}
    
    """
    def _risk_adjusted_excess_return(series, benchmark, rfr = 0., freq = 'daily'):
        benchmark_sharpe = sharpe_ratio(benchmark, rfr, freq)
        annualized_ret = annualized_return(series, freq)
        series_vol = annualized_vol(series, freq)
        
        return annualized_ret - series_vol * benchmark_sharpe - rfr

    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _risk_adjusted_excess_return(series, 
                            x, rfr = rfr, freq = freq))
    else:
        return _risk_adjusted_excess_return(series, benchmark, 
                                            rfr = rfr, freq = freq)


def rolling_ui(series, window = 21):
    """   
    returns the rolling ulcer index over a series for a given ``window``
    (instead of the squared deviations from the mean).
    
    :ARGS:
    
        series: ``pandas.Series`` of prices

        window: ``int`` of the size of the rolling window
        
    :RETURNS:
    
        ``pandas.Series``: of the rolling ulcer index

    .. code::

        import visualize_wealth.performance as vwp

        ui = vwp.rolling_ui(price_series, window = 252)

    """
    def _rolling_ui(series, window = 21):
        rui = pandas.Series(numpy.tile(numpy.nan, [len(series),]), 
                            index = series.index, name = 'rolling UI')
        j = 0
        for i in numpy.arange(window, len(series)):
            rui[i] = ulcer_index(series[j:i])
            j += 1
        return rui

    if isinstance(series, pandas.DataFrame):
        return series.apply(lambda x: _rolling_ui(x, window = window))
    else:
        return _rolling_ui(series)

def sharpe_ratio(series, rfr = 0., freq = 'daily'):
    """
    Returns the `Sharpe Ratio <http://en.wikipedia.org/wiki/Sharpe_ratio>`_ of an 
    asset, given a price series, risk free rate of ``rfr``, and ``frequency`` of the 
    time series
    
    :ARGS:

        series: ``pandas.Series`` of prices

        rfr: ``float`` of the risk free rate

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    :RETURN:

        ``float`` of the Sharpe Ratio

    .. note:: Calculating Sharpe 

        .. math::

            \\textrm{SR} = \\frac{(R_p - r_f)}{\\sigma} \\: \\textrm{where},

            R_p &= \\textrm{series annualized return} \\\\
            r_f &= \\textrm{Risk free rate} \\\\
            \\sigma &= \\textrm{Portfolio annualized volatility}

    """
    def _sharpe_ratio(series, rfr = 0., freq = 'daily'):
        return (annualized_return(series, freq) - rfr)/annualized_vol(series, freq)
    if isinstance(series, pandas.DataFrame):
        return series.apply(lambda x: _sharpe_ratio(x, rfr = rfr, freq = freq))
    else:
        return _sharpe_ratio(series, rfr = 0., freq = freq)


def sortino_ratio(series, freq = 'daily', rfr = 0.0):
    """
    Returns the `Sortino Ratio <http://en.wikipedia.org/wiki/Sortino_ratio>`_, or
    excess returns per unit downside volatility

    :ARGS:
    
        series: ``pandas.Series`` of prices
    
        freq: ``str`` of either ``daily, monthly, quarterly, or yearly``    
        indicating the frequency of the data ``default=`` daily

    :RETURNS:
    
        float of the Sortino Ratio

    .. note:: Calculating the Sortino Ratio

        There are several calculation methodologies for the Sortino Ratio, this
        method using downside volatility, where
        
        .. math::

            \\textrm{Sortino Ratio} = \\frac{(R-r_f)}{\\sigma_\\textrm{downside}}
    
    .. code:: 

        import visualize_wealth.performance as vwp

        sortino_ratio = vwp.sortino_ratio(price_series, frequency = 'monthly')
        
    """
    def _sortino_ratio(series, freq = 'daily'):
        return annualized_return(series, freq = freq)/downside_deviation(series, 
                                                                     freq = freq)
    if isinstance(series, pandas.DataFrame):
        return series.apply(lambda x: _sortino_ratio(x, freq = freq))
    else:
        return _sortino_ratio(series, freq = freq)

def systematic_as_proportion(series, benchmark, freq = 'daily'):
    """
    Returns the systematic risk as proportion of total volatility

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` to compare ``series`` against

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    :RETURNS:

        ``float`` between (0, 1) representing the proportion of  volatility
        represented by systematic risk

    """
    def _systematic_as_proportion(series, benchmark, freq = 'daily'):
        fac = _interval_to_factor(freq)
        return systematic_risk(series, benchmark, freq) **2 / (
            annualized_vol(series, freq)**2)
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _systematic_as_proportion(series, x, freq))
    else:
        return _systematic_as_proportion(series, benchmark, freq)


def systematic_risk(series, benchmark, freq = 'daily'):
    """
    Returns the systematic risk, or the volatility that is directly attributable
    to the benchmark

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` to compare ``series`` against

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    :RETURNS:

        ``float`` of the systematic volatility (not variance)

    .. note::  Calculating Systematic Risk

        .. math::
            \\sigma_b &= \\textrm{Volatility of the Benchmark} \\\\
            \\sigma^2_{\\beta} &= \\textrm{Systematic Risk} \\\\
            \\beta &= \\frac{\\sigma^2_{s, b}}{\\sigma^2_{b}} \\: \\textrm{then,}

            \\sigma^2_{\\beta} &= \\beta^2 \\cdot \\sigma^2_{b}
            \\Rightarrow \\sigma_{\\beta} &= \\beta \\cdot \\sigma_{b}
    """
    def _systematic_risk(series, benchmark, freq = 'daily'):
        bench_rets = log_returns(benchmark)
        benchmark_vol = annualized_vol(benchmark)
        return benchmark_vol * beta(series, benchmark)
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _systematic_risk(series, x, freq))
    else:
        return _systematic_risk(series, benchmark, freq)

def tracking_error(series, benchmark, freq = 'daily'):
    """
    Returns a ``float`` of the `Tracking Error <http://en.wikipedia.org/wiki/Tracking_error>`_ or standard deviation of the 
    active returns
      
    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` to compare ``series`` against

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly`` 


    :RETURNS:

        ``float`` of the tracking error

    .. note:: Calculating Tracking Error

        Let :math:`r_{a_i} =` "Active Return" for period :math:`i`, to calculate the
        compound linear difference between :math:`r_s` and :math:`r_b` is,

        .. math::

          r_{a_i} = \\frac{(1+r_{s_i})}{(1+r_{b_i})}-1

          \\textrm{then, } \\textrm{TE} &= \\sigma_a \\cdot \\sqrt{k} \\\\
          k &= \\textrm{Annualization factor}

    """
    def _tracking_error(series, benchmark, freq = 'daily'):
        fac = _interval_to_factor(freq)
        series_rets = linear_returns(series)
        bench_rets = linear_returns(benchmark)
        return ((1 + series_rets).div(1 + bench_rets) - 1).std()*numpy.sqrt(fac)

    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _tracking_error(series, x, freq = freq))
    else:
        return _tracking_error(series, benchmark, freq = freq)

def cumulative_turnover(alloc_df, asset_wt_df):
    """
    Provided an allocation frame (i.e. the weights to which the portfolio was 
    rebalanced), and the historical asset weights,  return the cumulative turnover,
    where turnover is defined below.  The first period is excluded of the
    ``alloc_df`` is excluded as that represents the initial investment

    :ARGS:

        alloc_df: :class:`pandas.DataFrame` of the the weighting allocation that was
        provided to construct the portfolio

        asset_wt_df: :class:`pandas.DataFrame` of the actual historical weights of
        each asset

    :RETURNS:

        cumulative turnover

    .. note:: Calcluating Turnover

    Let :math:`\\tau_j = `"Single Period period turnover for period :math:`j`,
    and assets :math:`i = 1,:2,:...:,n`, each whose respective portfolio weight is
    represented by :math:`\\omega_i`.
    
    Then the single period :math:`j` turnover for all assets :math:`1,..,n` can be
     calculated as:
    
    .. math::

        \\tau_j = \\frac{\\sum_{i=1}^n|\omega_i - \\omega_{i+1}|  }{2}
        
    """
    ind = alloc_df.index[1:]
    return asset_wt_df.iloc[ind, :].sub(
        asset_wt_df.shift(-1).iloc[ind, :]).abs().sum(axis = 1).sum()
    
def ulcer_index(series):
    """
    Returns the ulcer index of  the series, which is defined as the squared drawdowns
    (instead of the squared deviations from the mean).  Further explanation can be 
    found at `Tanger Tools <http://www.tangotools.com/ui/ui.htm>`_
    
    :ARGS:
    
        series: ``pandas.Series`` of prices

    :RETURNS:
    
        :float: the maximum drawdown of the period, expressed as a positive number

    .. code::

        import visualize_wealth.performance as vwp

        ui = vwp.ulcer_index(price_series)

    """
    def _ulcer_index(series):
        dd = 1. - series/series.cummax()
        ssdd = numpy.sum(dd**2)
        return numpy.sqrt(numpy.divide(ssdd, series.shape[0] - 1))
    if isinstance(series, pandas.DataFrame):
        return series.apply(_ulcer_index)
    else:
        return _ulcer_index(series)


def upcapture(series, benchmark):
    """
    Returns the proportion of ``series``'s cumulative positive returns to
    ``benchmark``'s cumulative  returns, given benchmark's returns were positive 
    in that period

    :ARGS:

        series: ``pandas.Series`` of prices

        benchmark: ``pandas.Series`` of prices to compare ``series`` against

    :RETURNS:

        float: of the upcapture of cumulative positive returns

    .. seealso:: :py:data:`median_upcature(series, benchmark)`
    
    """
    def _upcapture(series, benchmark):
        series_rets = log_returns(series)
        bench_rets = log_returns(benchmark)
        index = bench_rets > 0.
        return series_rets[index].mean() / bench_rets[index].mean()
    if isinstance(benchmark, pandas.DataFrame):
        return benchmark.apply(lambda x: _upcapture(series, x))
    else:
        return _upcapture(series, benchmark)

def upside_deviation(series, freq = 'daily'):
    """
    Returns the volatility of the returns that are greater than zero

    :ARGS:

        series: ``pandas.Series`` of prices

        freq: ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    :RETURNS:

        ``float`` of the upside standard deviation
    """
    def _upside_deviation(series, freq = 'daily'):
        fac = _interval_to_factor(freq)
        series_rets = log_returns(series)
        index = series_rets > 0.
        return series_rets[index].std()*numpy.sqrt(fac)
    if isinstance(series, pandas.DataFrame):
        return series.apply(lambda x: _upside_deviation(x, freq = freq))
    else:
        return _upside_deviation(series, freq)

def value_at_risk(series, freq = 'weekly', percentile = 5.):
    """    
    Return the non-parametric VaR (non-parametric estimate) for a given percentile,
    i.e. the loss for which there is less than a ``percentile`` of exceeding in a 
    period `freq`.

    :ARGS:
    
        series: ``pandas.Series`` of prices

        freq:``str`` of either ``daily, monthly, quarterly, or yearly``    
        indicating the frequency of the data ``default = daily``

        percentile: ``float`` of the percentile at which to calculate VaR
        
    :RETURNS:
    
        float of the Value at Risk given a ``percentile``

    .. code::

        import visualize_wealth.performance as vwp

        var = vwp.value_at_risk(price_series, frequency = 'monthly', percentile =
        0.1)
    
    """
    def _value_at_risk(series, freq = 'weekly', percentile = 5.):
        ind = _bool_interval_index(series.index, interval = freq)
        series_rets = log_returns(series[ind])
        
        #transform to linear returns, and loss is always reported as positive
        return -1 * (numpy.exp(numpy.percentile(series_rets, percentile))-1)
    if isinstance(series, pandas.DataFrame):
        return series.apply(lambda x: _value_at_risk(x, freq = freq,
                                                     percentile = percentile))
    else:
        return _value_at_risk(series, freq = freq, percentile = percentile)


def return_by_year(series):
    """
    the annualized returns for each of the years

    :ARGS:

        series: ``pandas.Series`` of prices

    :RETURNS:

        ``pandas.Series`` where ``index is years and ``values`` are linear 
        annual returns for that year
    """
    end_of_year_series = series.ix[series.index[:-1].year != series.index[1:].year]
    years = numpy.unique(end_of_year_series.index.year)
    yearly_rets = end_of_year_series.apply(numpy.log).diff()
    
    return pandas.Series(yearly_rets[1:].values, index = years[1:],
                         name = series.name)
    
def generate_all_metrics(series, benchmark, freq = 'daily', rfr = 0.):
    """
    cumulative values of relevant statistics is calculated for the ``series`` and 
    ``benchmark`` for a given ``frequency`` and risk free rate

    :ARGS:
    
        series: ``pandas.Series`` or ``Pandas.DataFrame`` of prices to compare
         against benchmark

         benchmark: ``pandas.Series`` of prices to compare series against

         freq: ``str`` of the frequency of prices.  'daily', 'weekly',
          'monthly', 'quarterly','yearly'
    
    :RETURNS:
   
         ``pandas.DataFrame`` of columns ['series','benchmark'] with statistics 
         (index) of ['annualized_return', annualized_vol','sharpe','max DD','ulcer
         index','beta','alpha', 'upcapture','downcapture','tracking error']
    """
    def _gen_function(series, benchmark, freq, rfr):
        solo_stats = ['annual ret', 'annual vol', 'sharpe', 'max dd', 'ulcer index']
    
        functions = [lambda x, y: annualized_return(series = x, freq = freq),
                 lambda x, y: annualized_vol(series = x, freq = freq),
                 lambda x, y: sharpe_ratio(series = x, rfr = rfr, freq = freq), 
                 lambda x, y: max_drawdown(series = x),
                 lambda x, y: ulcer_index(series = x)]

        sname = series.name
        bname = benchmark.name
        d = {sname: [], bname: []}
        for fun in functions:
            d[sname].append(fun(series, None))
            d[bname].append(fun(benchmark, None))

        a = jensens_alpha(series, benchmark, rfr = rfr)   
        b = beta(series, benchmark)
        r  = risk_adjusted_excess_return(series, benchmark, rfr = rfr, freq = freq)
        up = upcapture(series, benchmark)
        down = downcapture(series, benchmark) 
        te = tracking_error(series, benchmark, freq = freq)
        d[sname].extend([a, b, r, up, down, te])
        d[bname].extend([0., 1., 0., 1., 1., 0.])
        solo_stats.extend(['alpha', 'beta', 'raer','upcapture','downcapture',
                           'tracking error'])
        return pandas.DataFrame(d, columns = [bname, sname],  index = solo_stats)
    
    if isinstance(series, pandas.Series):
        return  _gen_function(series = series, benchmark = benchmark, freq = freq
                              , rfr = rfr)
    
    elif isinstance(series, pandas.DataFrame):
        cols = series.columns
        frame = _gen_function(series[cols[0]], benchmark, freq, rfr)
        cols = numpy.delete(cols, 0)
        
        for col in cols:
            tmp = _gen_function(series[col], benchmark, freq, rfr)[col]
            frame = frame.join(tmp)
        return frame

def _monthly_indexed(series):
    """
    Returns the series, indexed by end of month values
    """
    ind = series.index[:-1].month != series.index[1:].month
    return series.ix[ind]

def _interval_to_factor(interval):
    factor_dict = {'daily': 252, 'monthly': 12, 'quarterly': 4, 'yearly': 1}
    return factor_dict[interval] 
    
def _bool_interval_index(pandas_index, interval = 'monthly'):
    """
    creates weekly, monthly, quarterly, or yearly intervals by creating a
    boolean index to be passed visa vie DataFrame.ix[bool_index, :]
    """
    weekly = lambda x: x.weekofyear[1:] != x.weekofyear[:-1]
    monthly = lambda x: x.month[1:] != x.month[:-1]
    yearly = lambda x: x.year[1:] != x.year[:-1]
    ldom = lambda x: x.month[1:] != x.month[:-1]
    fdom = lambda x: numpy.append(False, x.month[1:]!=x.month[:-1])
    qt = lambda x: numpy.append(False, x.quarter[1:]!=x.quarter[:-1])
    time_dict = {'weekly':weekly, 'monthly': monthly, 'quarterly': qt, 
                 'yearly': yearly, 'ldom':ldom, 'fdom':fdom}

    return time_dict[interval](pandas_index)

def test_funs():
    """
    The testing function for ``analyze.py``

    >>> import pandas.util.testing as put
    >>> import inspect, analyze
    
    >>> f = pandas.ExcelFile('../tests/test_analyze.xlsx')
    >>> man_calcs = f.parse('calcs', index_col = 0)
    >>> prices = man_calcs[['S&P 500', 'VGTSX']]
    >>> stats = f.parse('results', index_col = 0)
    >>> put.assert_series_equal(man_calcs['S&P 500 Log Ret'], 
    ...    log_returns(prices['S&P 500']))
    >>> put.assert_series_equal(man_calcs['VGTSX Lin Ret'], 
    ...    linear_returns(prices['VGTSX']))
    >>> put.assert_series_equal(man_calcs['Active Return'],
    ...    analyze.active_returns(series = prices['VGTSX'], 
    ...    benchmark = prices['S&P 500']))

    >>> no_calc_list = ['value_at_risk', 'rolling_ui', 'active_returns', 'test_funs',
    ...   'linear_returns', 'log_returns', 'generate_all_metrics', 'return_by_year']
    
    >>> d = {'series': prices['VGTSX'], 'benchmark':prices['S&P 500'], 
    ...    'freq': 'daily', 'rfr': 0.0}
    >>> funs = inspect.getmembers(analyze, inspect.isfunction)

    Instead of writing out each function, I use the ``inspect module`` to determine
    Function names, and number of args.  Because the names of the functions are
    the same as the statistic in the ``results`` tab of
    ``../tests/test_analyze.xlsx`` I can use those key values to both call the
    function and reference the manually calculated value in the ``stats`` frame.

    >>> for fun in funs:
    ...    if (fun[0][0] != '_') and (fun[0] not in no_calc_list):
    ...        arg_list = inspect.getargspec(fun[1]).args
    ...        in_vals = tuple([d[arg] for arg in arg_list])
    ...        put.assert_almost_equal(fun[1](*in_vals), stats.loc[fun[0], 'VGTSX'])
    """
    return None
