#README.md

##Construct Portfolio

###`format_blotter(blotter_file)`
 Given a blotter file with positive values (and possible whitespace) for both the buy and sell, `format_blotter` transforms Sell values that are positive to negative values and
 returns a Series

####INPUTS:

* **blotter_file:** a file location where the blotter file is
  located

####RETURNS:

* **`pandas.DataFrame`:** where Sell values have been changed to negative
  and whitespace has been removed

###`append_price_frame_with_dividends(ticker, start_date, end_date=None):`
Given a ticker (type `str`), `start_date`, and `end_date` (of type
`datetime.datetime`, return a `pandas.DataFrame` by accessing Yahoo!'s
API to extract dividend payouts, and append that information to
'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume' (also obtained
from the Yahoo!'s API).

####INPUTS:

* **ticker:** a `str` of the ticker to query Yahoo!'s API
* **start_date:** a `datetime.datetime` of first price date desired
* **end_date:** a `datetime.datetime` of the last date desired, if
  left `null`, today's date (or the most recent trading day) is used
  

####RETURNS:

* `pandas.DataFrame` a `pandas.DataFrame` with columns 'Open', 'High',
  'Low', 'Close', 'Adj Close', 'Volume', 'Dividends.'

###`calculate_splits(price_df, tol = .1):`
Given a `pandas.DataFrame` of the format obtained from
`append_price_frame_with_dividends`, return a `pandas.DataFrame` with
an added column labeled 'Splits' that calculates the split factor.

The calculation estimates, using a specified tolerance (default
`tol=0.1`) between the ratio of the 'Close' and 'Adj Close'.

####INPUTS:

* **price_df:** 
  

####RETURNS:

* `pandas.DataFrame` a `pandas.DataFrame` with columns 'Open', 'High',
  'Low', 'Close', 'Adj Close', 'Volume', 'Dividends.'

###`append_price_frame_with_dividends(ticker, start_date, end_date=None):`
Given a ticker (type `str`), `start_date`, and `end_date` (of type
`datetime.datetime`, return a `pandas.DataFrame` by accessing Yahoo!'s
API to extract dividend payouts, and append that information to
'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume' (also obtained
from the Yahoo!'s API).

####INPUTS:

* **ticker:** a `str` of the ticker to query Yahoo!'s API
* **start_date:** a `datetime.datetime` of first price date desired
* **end_date:** a `datetime.datetime` of the last date desired, if
  left `null`, today's date (or the most recent trading day) is used
  

####RETURNS:

* `pandas.DataFrame` a `pandas.DataFrame` with columns 'Open', 'High',
  'Low', 'Close', 'Adj Close', 'Volume', 'Dividends.'




