# Load the necessary packages and modules
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
 
# Ease of Movement 
def EVM(data, ndays): 
 dm = ((data['high'] + data['low'])/2) - ((data['high'].shift(1) + data['low'].shift(1))/2)
 br = (data['volume'] / 100000000) / ((data['high'] - data['low']))
 EVM = dm / br 
 EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name = 'EVM') 
 data = data.join(EVM_MA) 
 return data 
 
# Retrieve the AAPL data from Yahoo finance:
code = '600867'
start_data = '2018-01-01'
end_data = '2018-10-07'
data = ts.get_hist_data(code, start=start_data, end=end_data)
data.sort_index(inplace=True)

# Compute the 14-day Ease of Movement for AAPL
n = 14
AAPL_EVM = EVM(data, n)
EVM = AAPL_EVM['EVM']

# Plotting the Price Series chart and the Ease Of Movement below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['close'],lw=1)
plt.title('AAPL Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(EVM,'k',lw=0.75,linestyle='-',label='EVM(14)')
plt.legend(loc=2,prop={'size':9})
plt.ylabel('EVM values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()