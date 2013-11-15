#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: analyze.py

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>

"""
import numpy
import pandas
import collections

def log_returns(series):
    """
    Returns a series of log returns given a series of prices where
    
    .. math::

        R_t = \\log(\\frac{P_{t+1}}{P_t})

    ** ARGS:**

        **series:** ``pandas.Series`` of prices

    **RETURNS:**

        **series:** ``pandas.Series`` of log returns 
         
    """
    return series.apply(numpy.log).diff()

def linear_returns(series):
    """
    Returns a series of linear returns given a series of prices
        
    .. math::

        R_t = \\frac{P_{t+1}}{P_t}) - 1
    
    ** ARGS:**

        **series:** ``pandas.Series`` of prices

    **RETURNS:**

        **series:** ``pandas.Series`` of linear returns
             
    """
    return series.div(series.shift(1)) - 1

def active_returns(series, benchmark):
    """
    Active returns is defined as the compound difference between linear returns i.e.

    .. math::

        (1 + r_p)/(1 + r_b) - 1
    """
    return (1 + linear_returns(series)).div(1 + linear_returns(benchmark)) - 1 

def annualized_return(series, freq = 'daily'):
    """
    Returns the annualized linear return of a series, i.e. the linear compounding
    rate that would have been necessary, given the initial investment, to arrive at
    the final value

    **ARGS:**
    
        **series:** ``pandas.Series`` of prices
        
        **freq:** ``str`` of either ``daily, monthly, quarterly, or yearly``    
        indicating the frequency of the data ``default=`` daily

    **RETURNS:**
    
        **float**: of the annualized linear return

    **USAGE:**::

        import visualize_wealth.performance as vwp

        linear_return = vwp.annualized_return(price_series, frequency = 'monthly')
    
    """
    fac = _interval_to_factor(freq)
    series_rets = log_returns(series)
    return numpy.exp(series_rets.mean()*fac)-1

def annualized_vol(series, freq = 'daily'):
    """
    Returns the annlualized volatility of the log changes of the price series, by
    calculating the volatility of the series, and then applying the square root of 
    time rule, where:

    .. math::
       :nowrap:

        \\sigma = \\sigma_t \\cdot \\sqrt{k}, \\textrm{where},

        k & \\triangleq \\textrm{Factor of annualization}
        
        \\sigma_t & \\triangleq \\textrm{volatility of period log returns}
        
    **ARGS:**
    
        **series:** ``pandas.Series`` of prices

        **freq:** ``str`` of either ``daily, monthly, quarterly, or yearly``    
        indicating the frequency of the data ``default=`` daily

    **RETURNS:**
    
        **float**: of the annualized volatility

    **USAGE:**::

        import visualize_wealth.performance as vwp

        ann_vol = vwp.annualized_vol(price_series, frequency = 'monthly')
    """
    fac = _interval_to_factor(freq)
    series_rets = log_returns(series)
    return series_rets.std()*numpy.sqrt(fac)

def sortino_ratio(series, freq = 'daily'):
    """
    Returns the Sortino Ratio, or excess returns per unit downside volatility

    .. math::

        \\frac{(\\mathbb{R}-r_f)}{\\sigma_\\textrm{downside}}

    **ARGS:**
    
        **series:** ``pandas.Series`` of prices
    
        **freq:** ``str`` of either ``daily, monthly, quarterly, or yearly``    
        indicating the frequency of the data ``default=`` daily

    **RETURNS:**
    
        **float** of the Sortino Ratio

    **USAGE:**::

        import visualize_wealth.performance as vwp

        sortino_ratio = vwp.sortino_ratio(price_series, frequency = 'monthly')
        
    """
    return annualized_return(series, freq = freq)/downside_deviation(series, 
                                                                     freq = freq)
    
def value_at_risk(series, freq = 'weekly', percentile = 5.):
    """    
    Return the non-parametric VaR (non-parametric estimate) for a given percentile

    **ARGS:**
    
        **series:** ``pandas.Series`` of prices

        **freq:** ``str`` of either ``daily, monthly, quarterly, or yearly``    
        indicating the frequency of the data ``default=`` daily

        **percentile:** ``float`` of the percentile at which to calculate VaR
        
    **RETURNS:**
    
        **float** of the Value at Risk given a ``percentile``

    **USAGE:**::

        import visualize_wealth.performance as vwp

        var = vwp.value_at_risk(price_series, frequency = 'monthly', percentile = 0.1)
    
    """
    ind = _bool_interval_index(series.index, interval = freq)
    series_rets = log_returns(series[ind])
    
    #transform to linear returns, and loss is always reported as positive
    return -1 * (numpy.exp(numpy.percentile(series_rets, percentile))-1)

    
def max_drawdown(series):
    """
    Returns the maximum drawdown, or the maximum peak to trough linear distance, as 
    a positive value

    **ARGS:**
    
        **series:** ``pandas.Series`` of prices

    **RETURNS:**
    
        **float**: the maximum drawdown of the period, expressed as a positive number

    **USAGE:**::

        import visualize_wealth.performance as vwp

        max_dd = vwp.max_drawdown(price_series)
        """
    return numpy.max(1 - series/series.cummax())
    
def ulcer_index(series):
    """
    Returns the ulcer index of  the series, which is defined as the squared drawdowns
    (instead of the squared deviations from the mean).  Further explanation can be 
    found at `Tanger Tools <http://www.tangotools.com/ui/ui.htm>`_
    
    **ARGS:**
    
        **series:** ``pandas.Series`` of prices

    **RETURNS:**
    
        **float**: the maximum drawdown of the period, expressed as a positive number

    **USAGE:**::

        import visualize_wealth.performance as vwp

        ui = vwp.ulcer_index(price_series)

    """
    dd = 1. - series/series.cummax()
    ssdd = numpy.sum(dd**2)
    return numpy.sqrt(numpy.divide(ssdd, series.shape[0] - 1))

def rolling_ui(series, window = 21):
    """   
    returns the rolling ulcer index over a series for a given ``window``
    (instead of the squared deviations from the mean).
    
    **ARGS:**
    
        **series:** ``pandas.Series`` of prices

        **window:** ``int`` of the size of the rolling window
        
    **RETURNS:**
    
        **pandas.Series**: of the rolling ulcer index

    **USAGE:**::

        import visualize_wealth.performance as vwp

        ui = vwp.rolling_ui(price_series, window = 252)

    """
    rui = pandas.Series(numpy.tile(numpy.nan, [len(series),]), index = series.index,
                        name = 'rolling UI')
    j = 0
    for i in numpy.arange(window, len(series)):
        rui[i] = ulcer_index(series[j:i])
        j += 1
    return rui
    
def sharpe_ratio(series, rfr = 0., freq = 'daily'):
    """
    Returns the `Sharpe Ratio <http://en.wikipedia.org/wiki/Sharpe_ratio>`_ of an 
    asset, given a price series, risk free rate of ``rfr``, and ``frequency`` of the 
    time series, where:

    .. math::

        \\textrm{SR} = \\frac{R_p - r_f}{\\sigma}

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **rfr:** ``float`` of the risk free rate

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    **RETURN:**

        **flaot:** of the Sharpe Ratio

    """
    
    return (annualized_return(series, freq) - rfr)/annualized_vol(series, freq)

def risk_adjusted_excess_return(series, benchmark, rfr = 0., freq = 'daily'):
    """
    Returns the MMRAP or the `Modigliani Risk Adjusted Performance <http://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance>`_ that
    calculates the excess return from the `Capital
    Allocation Line <http://en.wikipedia.org/wiki/Capital_allocation_line>`_,
    at the same level of risk (or volatility), specificaly,

    .. math::

        raer = r_p - (\\textr{SR}_b \\cdot \\sigma_p + r_f), \\texrm{where}

        r_p &=
        
    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` from which to compare ``series``

        **rfr:** ``float`` of the risk free rate

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    **RETURNS:**

        **float:** of the risk adjusted excess performance
    
    """
    benchmark_sharpe = sharpe_ratio(benchmark, rfr, freq)
    annualized_ret = annualized_return(series, freq)
    series_vol = annualized_vol(series, freq)

    return annualized_ret - series_vol * benchmark_sharpe - rfr

def jensens_alpha(series, benchmark, rfr = 0., freq = 'daily'):
    """
    Returns the `Jensen's Alpha <http://en.wikipedia.org/wiki/Jensen's_alpha>`_ or 
    the excess return based on the systematic risk of the ``series`` relative to
    the ``benchmark``

    **ARGS:**

         **series:** ``pandas.Series`` of prices 

         **benchmark:** ``pandas.Series`` of the prices to compare ``series`` against

         **rfr:** ``float`` of the risk free rate

         **freq:** ``str`` of frequency, either daily, monthly, quarterly, or yearly

    **RETURNS:**

        **float:** representing the Jensen's Alpha

    """
    fac = _interval_to_factor(freq)
    series_ret = annualized_return(series, freq)
    bench_ret = annualized_return(benchmark, freq)
    return series_ret - (rfr + beta(series, benchmark)*(
        bench_ret - rfr))

def beta(series, benchmark):
    """
    Returns the sensitivity of one price series to a chosen benchmark 

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` of a benchmark to calculate the sensitivity

    **RETURNS:**

        **float:** the sensitivity of the series to the benchmark
    """
    series_rets = log_returns(series)
    bench_rets = log_returns(benchmark)
    return numpy.divide(bench_rets.cov(series_rets), bench_rets.var())
    
def r_squared(series, benchmark):
    """
    Returns the R-Squared of `Coefficient of Determination <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series of prices to regress ``series`` against

    **RETURNS:**

        **float:** of the coefficient of variation
    
    """
    series_rets = log_returns(series)
    bench_rets = log_returns(benchmark)
    
    return series_rets.corr(bench_rets)**2
    
def tracking_error(series, benchmark, freq = 'daily'):
    """
    Returns a ``float`` of the `Tracking Error <http://en.wikipedia.org/wiki/Tracking_error>`_ or standard deviation of the 
   compound differences of linear returns

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` to compare ``series`` against

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly`` 


    **RETURNS:**

        **float:** of the tracking error
        
    """
    #This needs to be changed to MATE (Mean Absolute Tracking Error)
    fac = _interval_to_factor(freq)
    series_rets = linear_returns(series)
    bench_rets = linear_returns(benchmark)
    return ((1 + series_rets).div(1 + bench_rets) - 1).std()*numpy.sqrt(fac)

def mean_absolute_tracking_error(series, benchmark, freq = 'daily'):
    """
    Returns Carol Alexander's calculation for Mean Absolute Tracking Error which is
    $$\sqrt{\frac{(T-1)}{T}\cdot \tau^2 + \bar{R}}$$, where $$\tau = Tracking Error$$
    and $$\bar{R} = $$ mean of the active returns

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` to compare ``series`` against

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly`` 


    **RETURNS:**

        **float:** of the mean absolute tracking error
       
    """
    
    active_rets = active_returns(series = series, benchmark = benchmark)
    N = active_rets.shape[0]
    return numpy.sqrt((N - 1)/float(N) * tracking_error(series, benchmark, freq)**2
                      + active_rets.mean()**2)

def idiosyncratic_risk(series, benchmark, freq = 'daily'):
    """
    Returns the idiosyncratic risk, or (total variance - syst variance) ^ (1/2)

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` to compare ``series`` against

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    **RETURNS:**

        **float:** the idiosyncratic volatility (not variance)
    """
    fac = _interval_to_factor(freq)
    series_rets =log_returns(series)
    bench_rets = log_returns(benchmark)
    series_vol = annualized_vol(series, freq)
    benchmark_vol = annualized_vol(benchmark, freq)
    
    return numpy.sqrt(series_vol**2 - beta(series, benchmark)**2 * (
        benchmark_vol ** 2))

def idiosyncratic_as_proportion(series, benchmark, freq = 'daily'):
    """
    Returns the idiosyncratic risk as proportion of total volatility

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` to compare ``series`` against

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    **RETURNS:**

        **float:** (0, 1) representing the proportion of  volatility represented
        by idiosycratic risk
        
    """
    fac = _interval_to_factor(freq)
    series_rets = log_returns(series)
    return idiosyncratic_risk(series, benchmark, freq)**2 / (
        annualized_vol(series, freq)**2)

def systematic_risk(series, benchmark, freq = 'daily'):
    """
   Returns the systematic risk, or :math:`\beta^2 x \sigma^2`

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` to compare ``series`` against

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    **RETURNS:**

        **float:** the systematic volatility (not variance)
    """
    bench_rets = log_returns(benchmark)
    benchmark_vol = annualized_vol(benchmark)
    return benchmark_vol * beta(series, benchmark)

def systematic_as_proportion(series, benchmark, freq = 'daily'):
    """
    Returns the idiosyncratic risk as proportion of total volatility

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` to compare ``series`` against

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    **RETURNS:**

        **float:** (0, 1) representing the proportion of  volatility represented
        by systematic risk

    """
    fac = _interval_to_factor(freq)
    return systematic_risk(series, benchmark, freq) **2 / (
        annualized_vol(series, freq)**2)

def median_upcapture(series, benchmark):
    """
    Returns the median upcapture of a ``series`` of prices against a ``benchmark`` 
    prices, given that the ``benchmark`` achieved positive returns in a given period

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` of prices to compare ``series`` against

    **RETURNS:**

        **float:** of the median upcapture 

    """
    series_rets = log_returns(series)
    bench_rets = log_returns(benchmark)
    index = bench_rets > 0.
    return series_rets[index].median() / bench_rets[index].median()

def median_downcapture(series, benchmark):
    """
    Returns the median downcapture of a ``series`` of prices against a ``benchmark`` 
    prices, given that the ``benchmark`` achieved negative returns in a given period

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` of prices to compare ``series`` against

    **RETURNS:**

        **float:** of the median downcapture
    """
    series_rets = log_returns(series)
    bench_rets = log_returns(benchmark)
    index = bench_rets < 0.
    return series_rets[index].median() / bench_rets[index].median()

def upcapture(series, benchmark):
    """
    Returns the proportion of ``series``'s cumulative positive returns to
    ``benchmark``'s cumulative  returns, given benchmark's returns were positive 
    in that period

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` of prices to compare ``series`` against

    **RETURNS:**

        **float:** of the upcapture of cumulative positive returns
    """
    series_rets = log_returns(series)
    bench_rets = log_returns(benchmark)
    index = bench_rets > 0.
    return series_rets[index].mean() / bench_rets[index].mean()
    
def downcapture(series, benchmark):
    """
    Returns the proportion of ``series``'s cumulative negative returns to
    ``benchmark``'s cumulative  returns, given benchmark's returns were negative in
    that period

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **benchmark:** ``pandas.Series`` of prices to compare ``series`` against

    **RETURNS:**

        **float:** of the downcapture of cumulative positive ret
    """
    series_rets = log_returns(series)
    bench_rets = log_returns(benchmark)
    index = bench_rets < 0.
    return series_rets[index].mean() / bench_rets[index].mean()

def upside_deviation(series, freq = 'daily'):
    """
    Returns the volatility of the returns that are greater than zero

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    **RETURNS:**

        **float:** of the upside standard deviation
    """
    fac = _interval_to_factor(freq)
    series_rets = log_returns(series)
    index = series_rets > 0.
    return series_rets[index].std()*numpy.sqrt(fac)

def downside_deviation(series, freq = 'daily'):
    """
    Returns the volatility of the returns that are less than zero

    **ARGS:**

        **series:** ``pandas.Series`` of prices

        **freq:** ``str`` of frequency, either ``daily, monthly, quarterly, or 
        yearly``

    **RETURNS:**

        **float:** of the downside standard deviation
    """
    fac = _interval_to_factor(freq)
    series_rets = log_returns(series)
    index = series_rets < 0.    
    return series_rets[index].std()*numpy.sqrt(fac)

def return_by_year(series):
    """
    the annualized returns for each of the years

    **ARGS:**

        **series:** ``pandas.Series`` of prices

    **RETURNS:**

        ``**pandas.Series**`` where ``index is years and ``values`` are linear 
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

    **ARGS:**
    
        **series:** ``pandas.Series`` or ``Pandas.DataFrame`` of prices to compare
         against benchmark

         **benchmark:** ``pandas.Series`` of prices to compare series against

         **freq:**  ``str`` of the frequency of prices.  'daily', 'weekly',
          'monthly', 'quarterly','yearly'
    
   **RETURNS:**
   
         **``pandas.DataFrame``** of columns ['series','benchmark'] with statistics 
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
