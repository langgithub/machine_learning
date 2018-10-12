# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:44:23 2018
@author: M
"""

import tushare as ts
from matplotlib.pylab import date2num
import mpl_finance as mpf
import datetime
import matplotlib.pyplot as plt

code = '600273'
start_data = '2017-05-18'
end_data = '2018-05-18'
hist_data = ts.get_hist_data(code, start=start_data, end=end_data)

def draw_k(hist_data):
    data_list = []
    for dates, row in hist_data.iterrows():
        date_time = datetime.datetime.strptime(dates, '%Y-%m-%d')
        t = date2num(date_time)
        open, high, close, low = row[:4]
        datas = (t, open, high, low, close)  # tushare里的数据顺序为open,high,close,low注意
        data_list.append(datas)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(bottom=0.2)

    mpf.candlestick_ohlc(ax, data_list, width=1.5, colorup='r', colordown='green')
    plt.grid()
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()

draw_k(hist_data)
