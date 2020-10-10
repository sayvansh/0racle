from datetime import timedelta
# Import all the packages you need for your model below
import numpy as np
import sys
import pandas as pd
# Import Flask for creating API
from flask import Flask, request
import plotly.graph_objects as go
from keras.models import load_model

import matplotlib.pyplot as plt
import gc
# load model
model = load_model('model2.h5')
df = pd.read_csv('EURGBP-M15 - back test.csv')
df = df[df['Close'].notna()]
df['Date'] = pd.to_datetime(df['Date'])
lastDate = df["Date"].iloc[-1]
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
close_data = df['Close'].values
print(type(close_data))
print(close_data)
close_data = close_data.reshape((-1, 1))
# del df
# gc.collect()
look_back = 10

close_data = close_data.reshape((-1))
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
a = np.array(a)
a.reshape((-1))
print(type(a))
print(a)
def predict(close, num_prediction, model):
    prediction_list = close[-look_back:]
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back - 1:]
    return prediction_list


def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
    return prediction_dates

#Preparing Final DataFrame
def dataFrameMaker(startDate, length):
  #startDate : the date that dataframe going to start from
  #length : how many business days
  H4 = ['1:00', '5:00', '9:00', '13:00', '17:00', '21:00']
  columns = ['Date','Time']
  df_ = pd.DataFrame(columns=columns)
  dateList = []
  timeList = []
  j = 0
  dtNext = startDate
  for i in range(length*3):
    dtNext = dtNext + timedelta(days=1)
    if dtNext.dayofweek != 5 and dtNext.dayofweek != 6:
      for p in range(6):
        dateList.append(dtNext)
        timeList.append(H4[p])
      j+=1
    if j == length:
      break
  df_['Date'] = dateList
  df_['Time'] = timeList
  df_['Time'] = [x + ':00' for x in df_['Time']]
  df_['Date'] +=  pd.to_timedelta(df_.Time)
  dateList = df_['Date'].tolist()
  return dateList

num_prediction = 5
forecast = predict(a,num_prediction, model)
forecast_dates = predict_dates(num_prediction)
forecast = forecast.reshape((-1))
# Read all necessary request parameters
s = dataFrameMaker(lastDate, 1)
print(type(lastDate))
# print(s)
# x = forecast_dates
# y = forecast
# plt.plot(s, y)
# plt.show()