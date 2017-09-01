# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 14:57:30 2017

@author: Othmane Dirhoussi
"""

import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
import sklearn




def get(tickers, startdate, enddate):
    def get_data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
    datas = map (get_data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))
tickers==['AAPL','MSFT']

daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')


daily_close_px.columns
all_adj_close = all_data[['Adj Close']]
# Calculate the returns 
all_returns =daily_close_px.apply(lambda x :np.log(x / x.shift(1) ) ).dropna()
X = sm.add_constant(all_returns['AAPL'])
model = sm.OLS(all_returns['MSFT'],X).fit()
# Print the summary
#print(model.summary())
model.fittedvalues
X_k=[]
window=60
for i in xrange(0,len(all_returns.index)):
    if len(all_returns) - i >= window:
        X = sm.add_constant(all_returns['AAPL'].iloc[i:window+i])
        y=all_returns['MSFT'].iloc[i:window+i]
        model = sm.OLS(y,X).fit()
        epsilon=y-model.fittedvalues
        X_k.append(np.sum(epsilon))




len(all_returns['AAPL'].iloc[0:60])

def rolling_stats(df, window=5):
    #     dataframe to hold the results
    res = pd.DataFrame(index=df.index)

    for i in xrange(0,len(df.index)):

        if len(df) - i >= window:
            # break the df into smaller chunks
            chunk = df.iloc[i:window+i,:]
            # calc_stats is a function created from the code above,
            # refer to the Gist at the end of the article.
            beta,alpha,r2 = calc_stats(chunk)
            res.set_value(chunk.tail(1).index[0],"beta",beta)
            res.set_value(chunk.tail(1).index[0],"alpha",alpha)
            res.set_value(chunk.tail(1).index[0],"r2",r2)
            # print "%s beta: %.4f \t alpha: %.4f" % (chunk.tail(1).index[0],b,a)
    res = res.dropna()
    return res