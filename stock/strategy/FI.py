################# Force Index ########################################################

# Load the necessary packages and modules
import pandas as pd
import tushare as ts

# Force Index 
def ForceIndex(data, ndays): 
 FI = pd.Series(data['close'].diff(ndays) * data['volume'], name = 'ForceIndex')
 data = data.join(FI) 
 return data


# Retrieve the Apple data from Yahoo finance:
code = '600867'
start_data = '2018-01-01'
end_data = '2018-10-07'
data = ts.get_hist_data(code, start=start_data, end=end_data)
data.sort_index(inplace=True)

# Compute the Force Index for Apple 
n = 1
AAPL_ForceIndex = ForceIndex(data,n)
print(AAPL_ForceIndex)