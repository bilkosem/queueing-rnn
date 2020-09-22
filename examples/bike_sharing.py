
###############################################################################
# Author: Bilgehan Kösem
# E-mail: bilkos92@gmail.com
# Date created: 07.18.2020
# Date last modified: 07.18.2020
# Python Version: 3.7
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
# https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
###############################################################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from queueing_rnn import QRNN
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
    
    df = pd.read_csv(r"\datasets\bike_day.csv")
    df = df.set_index(pd.DatetimeIndex(df["dteday"]))
    values = df.cnt.values.reshape(-1,1).astype('float32')
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    n_days = 60
    n_features = 1
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_days, 1)
    print(reframed.shape)
    
    # split into train and test sets
    values = reframed.values
    n_train_hours = int(len(df)*0.85) # %85 of the dataset will be used for training 
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    n_obs = n_days * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    train_y=train_y.reshape(-1,1)
        
    qrnn = QRNN([train_X.shape[-1],50,1],#3D Shaped Input
                time_steps = n_days,
                weight_scaler = 0.05,
                loss_function='mse',
                firing_rate_scaler=0.25,
                learning_rate = 0.01,
                optimizer='adadelta',
                )

    outlist =[]
    outoutlist =[]
    batch_error_list = []
    error_list = []
    batch_number = 32
    for i in range(50):
        loss=0
        num_correct = 0 #Counter for right prediction for each epoch
        for j in range(len(train_X)):
            qrnn.calculate_rate()
            output = qrnn.feedforward(train_X[j])
            if qrnn.loss_func == 'cross-entropy':
                probs = qrnn.softmax(output.reshape(-1,1))
                loss -= np.log(probs[int(train_y[j])]) #Loss is calculated
                num_correct += int(np.argmax(probs) == int(train_y[j]))
                d_L_d_y = probs
                d_L_d_y[int(train_y[j])] -= 1
            elif qrnn.loss_func == 'mse':
                loss += float(0.5*sum([(output[o]-train_y[j][o])**2 for o in range(len(train_y[j]))]))
                d_L_d_y = output - train_y[j]
            
            if j % batch_number != (batch_number-1):
                pass
            else:
                qrnn.backpropagation(train_y[j],d_L_d_y)
                error = float(loss/batch_number)
                batch_error_list.append(error)
                loss=0

        batch_mean = np.mean(batch_error_list)
        error_list.append(batch_mean)
        batch_error_list.clear()
        print('Epoch: '+str(i)+' MSE: '+ str(batch_mean))

    y_prediction_test=[]
    y_prediction_train=[]
    for j in range(len(test_X)):
        y_prediction_test.append(float(qrnn.feedforward(test_X[j])))
    for j in range(len(train_X)):
        y_prediction_train.append(float(qrnn.feedforward(train_X[j])))
    
    y_prediction_test = np.atleast_2d(y_prediction_test).reshape(-1,1)
    y_prediction_train = np.atleast_2d(y_prediction_train).reshape(-1,1)
    
    rmse_train = sqrt(mean_squared_error(train_y, y_prediction_train))
    rmse_test = sqrt(mean_squared_error(test_y, y_prediction_test))
    
    print("Test RMSE:",str('%.3f'%rmse_test))
    print("Train RMSE:",str('%.3f'%rmse_train))# Inverse Transform
    
    plt.plot(test_y[:500], color = 'red', label = 'Real Bike Sharing Data')
    plt.plot(y_prediction_test[:500], color = 'blue', label = 'Predicted Bike Sharing Data')
    plt.title('Queueing RNN Bike Sharing Test Data')
    plt.xlabel('Hours')
    plt.ylabel('Normalised Bike Sharing Data')
    plt.legend()
    plt.show()
    
    plt.plot(train_y, color = 'red', label = 'Real Bike Sharing Data')
    plt.plot(y_prediction_train, color = 'blue', label = 'Predicted Bike Sharing Data')
    plt.title('Queueing RNN Bike Sharing Data Prediction')
    plt.xlabel('Time')
    plt.ylabel('Normalised Bike Sharing Data')
    plt.legend()
    plt.show()

    plt.plot(error_list, label='Train')
    plt.title("Queueing RNN Bike Sharing Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Square Error")