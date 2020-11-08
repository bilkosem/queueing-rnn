
###############################################################################
# Author: Bilgehan Kösem
# E-mail: bilkos92@gmail.com
# Date created: 07.18.2020
# Date last modified: 08.11.2020
# Python Version: 3.8
###############################################################################

###############################################################################
# References:
#
# "series_to_supervised" function is taken from a post of Jason Brownlee on May 8,2017:
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
#
# The preprocessing section is taken from a post of Jason Brownlee on August 14,2017:
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#
# Access to Bike Sharing Dataset:
# https://www.kaggle.com/medharawat/google-stock-price
###############################################################################


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

if __name__ == '__main__':
    
    dataset_train = pd.read_csv(r'datasets\Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:2].values
    
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    X_train = []
    y_train = []
    for i in range(60, 1258):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = False, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dense(units = 1))
    
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    history = regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)

    # Evaluate the performance of the model
    dataset_test = pd.read_csv(r'datasets\Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    
    scaled_real_stock_price = sc.transform(real_stock_price)  
    predicted_train_stock_price = regressor.predict(X_train)
    
    rmse_train = sqrt(mean_squared_error(y_train, predicted_train_stock_price))
    rmse_test = sqrt(mean_squared_error(scaled_real_stock_price, predicted_stock_price))
    
    print("Test RMSE:",str('%.3f'%rmse_test))
    print("Train RMSE:",str('%.3f'%rmse_train))
    
    plt.plot(scaled_real_stock_price, color = 'red', label = 'Real Google Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('LSTM Google Stock Price Test Set')
    plt.xlabel('Day')
    plt.ylabel('Normalized Google Stocking Price')
    plt.legend()
    plt.show()
    
    plt.plot(y_train, color = 'red', label = 'Real Google Stock Price')
    plt.plot(predicted_train_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('LSTMGoogle Stock Price Test Set')
    plt.xlabel('Day')
    plt.ylabel('Normalized Google Stocking Price')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train')
    plt.title("LSTM Google Stock Price Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Square Error")
    



    
    
    
    
    
    