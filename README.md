#README.md

##License

This program is free software and is distrubuted under the
[GNU General Public License version 3](http://www.gnu.org/licenses/quick-guide-gplv3.html)("GNU
GPL v3"), which allow for four freedoms that every user should have:

* the freedom to use the software for any purpose,
* the freedom to change the software to suit your needs,
* the freedom to share the software with your friends and neighbors, and
* the freedom to share the changes you make.

(c) Benjamin M. Gross 2013


##Dependencies

`pandas`: extensively used
`urllib2`: for Yahoo! API calls to append price `DataFrame`s with
Dividends

##Installation

##ToDo List:

* Update the `README.md` file to include added portfolio construction
  functionaly completed today

* Currently the `generate_random_asset_path` is broken because of
  incorrect type passing (`Series` vs. `DataFrame`)

* ~~`tempfile` might be the more appropriate way to deal with price
  handling in aggregation functions, especially because at some point
  different methods of data calling will be used for price information.~~

* ~~Check out small error differences between the panel from weight
  program and the calculation sheet in '../bin/Adjustment Factor
  Test.xlsx' in `portfolio_panel_from_initial_weights`~~

* ~~After fixing the error calc, add testing of the
  `portfolio_panel_from_initial_weights` into `test_funs()`~~


