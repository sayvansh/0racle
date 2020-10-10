import numpy as np
import pandas as pd
from flask import Flask, request
from keras.models import load_model
from datetime import timedelta
from flask import jsonify

# load model
model = load_model('model2.h5')
look_back = 10


def close_prepare(close):
    close = np.array(close)
    close = close.reshape((-1))
    return close

def predict(close, num_prediction, model):
    num_prediction = num_prediction - 1
    prediction_list = close[-look_back:]
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back - 1:]

    return prediction_list

#Preparing Final DataFrame
def dataFrameMaker(startDate, length):
  #startDate : the date that dataframe going to start from
  #length : how many business days
  H4 = ['1:00', '5:00', '9:00', '13:00', '17:00', '21:00']
  columns = ['Date', 'Time']
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


app = Flask(__name__)


# Create an API endpoint
@app.route('/predict',  methods = ['POST'])
def predict_iris():
    req_data = request.get_json()
    closes = req_data['closes']
    date = req_data['date']

    closes = close_prepare(closes)
    num_prediction = 6
    forecast = predict(closes, num_prediction, model)
    forecast = list(forecast)

    date = pd.to_datetime(date)
    dates = dataFrameMaker(date, 1)
    res = {"close": forecast, "date": dates}

    return jsonify(res)

    # return jsonify(finalList)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')