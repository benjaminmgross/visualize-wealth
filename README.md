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
an added column labeled 'Splits' that calculates the split factor for
that day.

The calculation estimates when a split has occurred, using a specified
tolerance (default `tol=0.1`) between the ratio of the estimated 'Adj
Close' (using cumulative adjustment factors from dividends) and the
actual 'Adj Close'. Adjustment factors are then rounded to estimate
the closest n:1 split shares or 1:n reverse-split shares

####INPUTS:

* **price_df:** `pandas.DataFrame` with columns
    ['Close', 'Adj Close', 'Dividends'], as output in
    `append_price_frame_with_dividends` 
* **tol:** float tolerance to determine whether a split has occurred,
  default = 0.1
  

####RETURNS:

* `pandas.DataFrame` a `pandas.DataFrame` with columns 'Open', 'High',
  'Low', 'Close', 'Adj Close', 'Volume', 'Dividends' 'Splits'

###`blotter_to_split_adjusted_shares(blotter_series, price_df):`
Given a `blotter_series` of dates and purchases (+) / sales (-) and a DataFrame
with ['Close', 'Adj Close', 'Dividends', 'Splits'], as output from
`calculate_splits`, append the original `DataFrame` with a column 'cum_shares'

####INPUTS:

* **blotter_series:** a series of buys & sells of the format from `format_blotter`
* **price_df:** a `pandas.DataFrame` of the format output from `calculate_splits`
  

####RETURNS:

* `pandas.DataFrame` with columns 'Open', 'High', 'Low', 'Close', 'Adj
  Close', 'Volume', 'Dividends', 'Splits', 'cum_shares'

###`construct_random_trades(split_df, num_trades):`

Generate num_trades random trades in random amounts between -100 and 100 (rounded to
nearest 10) while making sure the shares never go below 0 (i.e. a
semi-realistic trade sequence is created), making sure those trades
occur on actual trading days

###INPUTS:

* **split_df:** a `pandas.DataFrame` of the asset's prices: used only
  to determine actual trading days

* **num_trades:** The number of random trades the user would like to
  generate

###RETURNS:

`pandas.Series` of random trades and trade dates that can be input
into functions requiring a `blotter_series`




