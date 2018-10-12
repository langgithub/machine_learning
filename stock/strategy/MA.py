# Moving Averages Code

# Load the necessary packages and modules
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

'''
金叉与死叉
'''
# Simple Moving Average 
def SMA(data, ndays): 
 SMA = pd.Series(pd.rolling_mean(data['close'], ndays), name = 'SMA')
 data = data.join(SMA) 
 return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
 EMA = pd.Series(pd.ewma(data['close'], span = ndays, min_periods = ndays - 1),
 name = 'EWMA_' + str(ndays)) 
 data = data.join(EMA) 
 return data

# Retrieve the Nifty data from Yahoo finance:
code = '600867'
start_data = '2018-04-01'
end_data = '2018-07-07'
data = ts.get_hist_data(code, start=start_data, end=end_data)
data.sort_index(inplace=True)
print(data)
close = data['close']

# Compute the 50-day SMA for NIFTY
n = 30
SMA_NIFTY = SMA(data,n)
SMA_NIFTY = SMA_NIFTY.dropna()
SMA = SMA_NIFTY['SMA']

# Compute the 200-day EWMA for NIFTY
ew = 60
EWMA_NIFTY = EWMA(data,ew)
EWMA_NIFTY = EWMA_NIFTY.dropna()
EWMA = EWMA_NIFTY['EWMA_60']

# Plotting the NIFTY Price Series chart and Moving Averages below
plt.figure(figsize=(9,5))
plt.plot(data['close'],lw=1, label='NSE Prices')
plt.plot(SMA,'g',lw=1, label='30-day SMA (green)')
plt.plot(EWMA,'r', lw=1, label='60-day EWMA (red)')
plt.legend(loc=2,prop={'size':11})
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=90)
plt.show()