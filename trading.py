# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 09:59:26 2017

@author: Othmane Dirhoussi
Simple trading srategy
"""

#Bqsic trading strategy introduction


import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
# Import the `datetools` module from `pandas`
#from pandas.tseries import datetools




#data = pdr.get_data_yahoo( "SPY",start = "2017-01-01", end = "2017-04-30",as_panel = False,auto_adjust = True)


AAPL=pd.read_csv(r"C:\Users\Othmane Dirhoussi\Documents\kaagle\sandp500\individual_stocks_5yr\AAPL_data.csv",index_col='Date', parse_dates=True)
print(AAPL.head(5))

#CHecking the dataframe types
#AAPL.dtypes


#AAPL.Date=pd.to_datetime(AAPL.Date)

aapl = pdr.get_data_yahoo('AAPL', 
                          start=datetime.datetime(2006, 10, 1), 
                          end=datetime.datetime(2012, 1, 1))

AAPL.High.plot()
#AAPL.iloc[0] Date is not the index all though we can change it
#we check if our data containn any empty values
#AAPL.isnull().values.any()

#iloc pour aoir les indices qu' on a envie d'avoir comme pour les numpy array
print(AAPL.iloc[22, 3])

# Sample 20 rows
sample = AAPL.sample(20)
# Print `sample`
print(sample)

# Resample to monthly level 
monthly_aapl = AAPL.resample('M').mean()
# Print `monthly_aapl`
print(monthly_aapl.head(5))

AAPL['diff']=AAPL.Open-AAPL.Close


# Assign `Adj Close` to `daily_close`
daily_close = AAPL[['Close']]
# Daily returns
daily_pct_change =daily_close.pct_change()
# Replace NA values with 0
daily_pct_change.fillna(0, inplace=True)
# Inspect daily returns
print(daily_pct_change)
# Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)
# Print daily log returns
print(daily_log_returns)

# Resample `aapl` to business months, take last observation as value 
monthly = AAPL.resample('BM').apply(lambda x: x[-1])
monthly_close = monthly[['Close']]
# Calculate the monthly percentage change
monthly_close.pct_change()
# Resample `aapl` to quarters, take the mean as value per quarter
quarter = AAPL.resample("4M").mean()
quarter_close = monthly[['Close']]
# Calculate the quarterly percentage change
quarter_close.pct_change()
print quarter_close


# Daily returns manual calculation
daily_pct_change_ = daily_close / daily_close.shift(1) - 1
# Print `daily_pct_change`
print(daily_pct_change_)
daily_pct_change_.fillna(0, inplace=True)
# we do obtain the same results as the native pandas function 
# Plot the distribution of `daily_pct_c`
daily_pct_change.hist(bins=50)
# Show the plot
plt.show()
# Pull up summary statistics
print(daily_pct_change.describe())

# Calculate the cumulative daily returns
cum_daily_return = (1 + daily_pct_change).cumprod() 
# Print `cum_daily_return`
print(cum_daily_return)
cum_daily_return.plot(figsize=(12,8))

# Show the plot
plt.show()

# Resample the cumulative daily return to cumulative monthly return 
cum_monthly_return = cum_daily_return.resample("M").mean()

# Print the `cum_monthly_return`
print(cum_monthly_return)
cum_monthly_return.plot(figsize=(12,8))
# Show the plot
plt.show()



def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
    datas = map (data, tickers)
    
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))



# Isolate the `Adj Close` values and transform the DataFrame
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')
daily_close_px.head(3)
# Calculate the daily percentage change for `daily_close_px`
daily_pct_change = daily_close_px.pct_change()
# Plot the distributions
daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))
# Show the resulting plot
plt.show()



# Isolate the adjusted closing prices 
adj_close_px = aapl[['Adj Close']]
# Calculate the moving var
moving_var = adj_close_px.rolling(window=40).var()
# Inspect the result
print(moving_var[-10:])

moving_avg = adj_close_px.rolling(window=40).mean()
# Inspect the result
print(moving_avg[-10:])

aapl['252'] = adj_close_px.rolling(window=252).mean()
aapl['42'] = adj_close_px.rolling(window=40).mean()
aapl[['Adj Close', '42', '252']].plot()
plt.show()



#historical volatility

min_periods = 75 

# Calculate the volatility
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods) 

# Plot the volatility
vol.plot(figsize=(10, 8))

# Show the plot
plt.show()




# Isolate the adjusted closing price
all_adj_close = all_data[['Adj Close']]

# Calculate the returns 
all_returns = np.log(all_adj_close / all_adj_close.shift(1))

# Isolate the AAPL returns 
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
aapl_returns.index = aapl_returns.index.droplevel('Ticker')

# Isolate the MSFT returns
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')

# Build up a new DataFrame with AAPL and MSFT returns
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']

# Add a constant 
X = sm.add_constant(return_data['AAPL'])

# Construct the model
#we perform a linear regression on Y=MSFT ze try to explain MSFT returns using AAPL returns.

model = sm.OLS(return_data['MSFT'],X).fit()

# Print the summary
print(model.summary())


##ploting the regression


plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')

# Add an axis to the plot
ax = plt.axis()
# Initialize `x`
x = np.linspace(ax[0], ax[1] + 0.01)

# Plot the regression line
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)

# Customize the plot
plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft returns')
# Show the plot
plt.show()


long_window=10
short_window=40
vlong=1.0
vshort=0.0
signals=pd.DataFrame(index=aapl.index)
signals['signal']=0.0
signals['short_mavg']=aapl.Close.rolling(window = short_window, min_periods=1, center=False).mean()
signals['long_mavg']=aapl.Close.rolling(window = long_window, min_periods=1, center=False).mean()
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], vlong,vshort) 
signals['positions'] = signals['signal'].diff()

# Print `signals`
print(signals)

signals.iloc[39]


fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
aapl['Close'].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='k')
         
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='yellow')
         
# Show the plot
plt.show()

initial_capital=1000000
positions=pd.DataFrame(index=signals.index)
positions['AAPL']=100*signals['signal']
portfolio = positions.multiply(aapl['Adj Close'], axis=0)
pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()
print(portfolio.head())

fig = plt.figure()

ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2.)

ax1.plot(portfolio.loc[signals.positions == 1.0].index, 
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(portfolio.loc[signals.positions == -1.0].index, 
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='k')

# Show the plot
plt.show()




























