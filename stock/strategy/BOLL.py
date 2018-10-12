################ Bollinger Bands #############################

# Load the necessary packages and modules
import pandas as pd
import tushare as ts

'''
紧口变大，买入特征
'''
# Compute the Bollinger Bands 
def BBANDS(data, ndays):

    MA = pd.Series(pd.rolling_mean(data['close'], ndays))
    SD = pd.Series(pd.rolling_std(data['close'], ndays))
    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name = 'Upper BollingerBand')
    data = data.join(B1)
 
    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name = 'Lower BollingerBand')
    data = data.join(B2)
 
    return data
 
# Retrieve the Nifty data from Yahoo finance:
code = '600867'
start_data = '2018-01-01'
end_data = '2018-10-07'
data = ts.get_hist_data(code, start=start_data, end=end_data)
data.sort_index(inplace=True)

# Compute the Bollinger Bands for NIFTY using the 50-day Moving average
n = 50
NIFTY_BBANDS = BBANDS(data, n)
print(NIFTY_BBANDS)