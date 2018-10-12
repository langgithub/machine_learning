# Rate of Change code

# Load the necessary packages and modules
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

# Rate of Change (ROC)
def ROC(data,n):
 N = data['close'].diff(n)
 D = data['close'].shift(n)
 ROC = pd.Series(N/D,name='Rate of Change')
 data = data.join(ROC)
 return data 
 
# Retrieve the NIFTY data from Yahoo finance:
code = '600867'
start_data = '2018-04-01'
end_data = '2018-07-07'
data = ts.get_hist_data(code, start=start_data, end=end_data)
data.sort_index(inplace=True)

# Compute the 5-period Rate of Change for NIFTY
n = 5
NIFTY_ROC = ROC(data,n)
ROC = NIFTY_ROC['Rate of Change']

# Plotting the Price Series chart and the Ease Of Movement below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['close'],lw=1)
plt.title('NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(ROC,'k',lw=0.75,linestyle='-',label='ROC')
plt.legend(loc=2,prop={'size':9})
plt.ylabel('ROC values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()