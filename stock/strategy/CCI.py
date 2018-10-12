# Load the necessary packages and modules
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
# Commodity Channel Index 
def CCI(data, ndays): 
    TP = (data['high'] + data['low'] + data['close']) / 3
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),name = 'CCI')
    data = data.join(CCI)
    return data

# Retrieve the Nifty data from Yahoo finance:

code = '600867'
start_data = '2018-01-01'
end_data = '2018-10-07'
data = ts.get_hist_data(code, start=start_data, end=end_data)
data.sort_index(inplace=True)
print(data)


# Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
n = 20
NIFTY_CCI = CCI(data, n)
CCI = NIFTY_CCI['CCI']

# Plotting the Price Series chart and the Commodity Channel index below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['close'],lw=1)
plt.title('NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(CCI,'k',lw=0.75,linestyle='-',label='CCI')
plt.legend(loc=2,prop={'size':9.5})
plt.ylabel('CCI values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()