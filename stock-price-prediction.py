# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:20:59 2020

@author: anims
"""

#%% Importing the libraries
'''
Thanks to @aranroussi for yfinance module
https://github.com/ranaroussi/yfinance

you can install it on anaconda using the following
> conda install -c ranaroussi yfinance

This model uses tensorflow 1.15 GPU for the computation, but it can easily be run on tensorflow 1.x on CPU
I still have to check for compatibility with tensorflow 2.x

'''

import pandas as pd
import numpy as np
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout


#%% getting the dataset
'''
for more information, visit finance.yahoo.com
That is where the data is fetched from
'''
ticker = str(input("Enter a ticker: "))
Ticker = yf.Ticker(ticker.upper())


'''This saves the dataset to a created directory'''

cwd = os.getcwd()
if not os.path.isdir(cwd+"/data/"+ticker):
    os.makedirs(cwd+"/data/"+ticker,0o755)
df = Ticker.history(period="max",interval="1d")
df.to_csv(cwd+"/data/"+ticker+"/"+ticker+"-"+str(date.today())+".csv")


#%% Splitting the training data and test data

'''
What the model basically does is that it takes the prices of shares from the day of inception to 30 days earlier from today.
That constitutes of the training data, and then it takes the next 30 days (until yesterday or today, whichever is available) and uses that for comparison
'''
test_period = 30 #days
train_data = df[:df.shape[0]-test_period]["Close"]
test_data = df[df.shape[0]-test_period:]["Close"]


#%% Scaling the data in periods

'''
The data is scaled for the usual reasons
'''
scaler = MinMaxScaler()
train_data = np.array(train_data).reshape(-1,1)
test_data = np.array(test_data).reshape(-1,1)

train_data = scaler.fit_transform(train_data)
train_data = train_data.reshape(-1)

test_data = scaler.transform(test_data).reshape(-1)

#%% Curve Smoothening (Exponential Moving Average)
'''
Exponential Moving Average is a concept of finance, which removes random noises from the data and gives a clearer picture of the trend of the stock price
'''
EMA = 0.0
gamma = 0.3
for ti in range(train_data.shape[0]):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

#%% Getting training and testing data

'''
Now, this code creates a series of matrices.
X_train is the matrix which contains prices of 80 consecutive days, starting from day 80th to the last day of training data
Y_train is the target value of 81st day.
The concept is to train 80 days of data to predict the 81st day price, now this is done again with shifting the date window by one day
'''

jump=1
lookback = 80
X_train,y_train = [],[]
for i in range(lookback,train_data.size,jump):
    X_train.append(train_data[i-lookback:i])
    y_train.append(train_data[i])
X_train,y_train = np.array(X_train),np.array(y_train)
 
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#%% Making the LSTM Model

'''
This model is rather complex, with eight LSTM layers. The data it is trained on is rather big and this takes up a lot of computational power
Take note that the final LSTM layer does NOT return a sequence.
'''
regressor = Sequential()

regressor.add(LSTM(units = 128, return_sequences = True, input_shape=(lookback,1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 64, return_sequences = True))
regressor.add(Dropout(0.15))

regressor.add(LSTM(units = 32, return_sequences = True))
regressor.add(Dropout(0.15))

regressor.add(LSTM(units = 64, return_sequences = False))
regressor.add(Dropout(0.15))

regressor.add(Dense(units=64,activation='relu'))
regressor.add(Dense(units=32,activation='relu'))
regressor.add(Dense(units=16,activation='relu'))
regressor.add(Dense(units=8,activation='tanh'))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss="mean_squared_error",metrics=["accuracy"])

regressor.summary()

#%% Fitting and saving the model

'''Execute this cell multiple times. This is done so as to keep saving the model in between. 
It has been observed that training on GPU is somewhat unstable and sometimes the kernel crashes halfway through training. 
I have tried this workaround.
'''

if os.path.isfile(cwd+"/data/"+ticker+"/"+ticker+".h5"):
    regressor = load_model(cwd+"/data/"+ticker+"/"+ticker+".h5")
    print("Loaded from checkpoint")
history = regressor.fit(X_train,y_train,batch_size=32,epochs=50,use_multiprocessing=True)
try:
    regressor.save(cwd+"/data/"+ticker+"/"+ticker+".h5")
    print("Checkpoint made")
except:
    pass

#%% Post processing the data and readying for prediction

'''
Now the prediction is made on Open prices of the stocks. What we do is we take last 80 days' data from the last day of training data, and we make the prediction of the 81st day (or, first day of the testing data).
Now this window shifts and we take 81st day's price (testing data) into training data and we make prediction for 82nd day.
This is done until today is reached. 
This is a one timestep prediction, which means that we will only predict prices up to one day in advance, we will need prices of next day to predict for day after tomorrow. 
'''

real_stock_price = df.iloc[df.shape[0]-test_period:df.shape[0],0:1]["Open"]
inputs =  df["Open"][(len(df)-len(real_stock_price)-lookback):].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(lookback, lookback+test_period):
    X_test.append(inputs[i-lookback:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (test_period, lookback ,1))

real_stock_price = np.array(real_stock_price)

#%%
'''Predicting and inverse-transform the prices for the 30 days'''
predicted = regressor.predict(X_test)
predicted = scaler.inverse_transform(predicted)

#%% converting to dataframe

predicted_stock_price=pd.DataFrame(data=predicted,columns=["Predicted Stock Price"])
real_stock_price=pd.DataFrame(data=real_stock_price,columns=["Real Stock Price"])
total = pd.concat([predicted_stock_price,real_stock_price],axis=1)

#%% setting date column

datecolumn = []
date = date.today() - timedelta(days=test_period)

for i in range(test_period):
    _ = str(date+timedelta(days=i))
    _ = _[len(_)-5:]
    datecolumn.append(_)
total["Date"]=datecolumn
total.set_index("Date",inplace=True)

total.to_csv("./data/"+ticker+"/"+ticker+"-predicted.csv")

#%% plotting

plot = total.plot(y=["Real Stock Price", "Predicted Stock Price"],figsize=(6,4),title="Stock Price Prediction",kind="line")
plot.set_xlabel("Date (MM-DD)")
plot.set_ylabel("Price")
fig=plot.get_figure()
fig.savefig("./data/"+ticker+"/"+ticker+".png",dpi=300)

