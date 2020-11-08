
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
    
    timestep = 60
    dataset_train = pd.read_csv(r'datasets\Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:2].values

    dataset_test = pd.read_csv(r'datasets\Google_Stock_Price_Test.csv')
    test_set = dataset_test.iloc[:, 1:2].values

    dataset_total = pd.DataFrame(pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0))

    sc = MinMaxScaler(feature_range = (0, 1))
    dataset_total_scaled = sc.fit_transform(dataset_total)
    
    training_set_scaled = dataset_total_scaled[:len(training_set)]
    
    X_train = []
    y_train = []
    for i in range(timestep, 1258):
        X_train.append(training_set_scaled[i-timestep:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, train_y = np.array(X_train), np.array(y_train)
    train_X = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    train_y = train_y.reshape(-1,1)

    inputs = dataset_total_scaled[len(dataset_total) - len(dataset_test) - 60:]
    
    X_test = []
    for i in range(timestep, 80):
        X_test.append(inputs[i-timestep:i, 0])
    X_test = np.array(X_test)
    test_X = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    test_y = sc.transform(test_set.reshape(-1,1))

    qrnn = QRNN([train_X.shape[-1],50,1],#3D Shaped Input
                time_steps = timestep,
                weight_scaler = 0.05,
                loss_function='mse',
                firing_rate_scaler=0.25,
                learning_rate = 0.01,
                optimizer='amsgrad')

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

    # Evaluate the performance of the model
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
    print("Train RMSE:",str('%.3f'%rmse_train))
    
    plt.plot(test_y, color = 'red', label = 'Real Google Stock Price')
    plt.plot(y_prediction_test[-len(test_y):], color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('Queueing RNN Google Stocking Price Test Set')
    plt.xlabel('Day')
    plt.ylabel('Normalised Google Stocking Price')
    plt.legend()
    plt.show()
    
    plt.plot(train_y, color = 'red', label = 'Real Google Stock Price')
    plt.plot(y_prediction_train, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('Queueing RNN Bike Sharing Data Training Set')
    plt.xlabel('Day')
    plt.ylabel('Normalised Google Stocking Price')
    plt.legend()
    plt.show()

    plt.plot(error_list, label='Train')
    plt.title("Queueing RNN Google Stock Price Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Square Error")
    



    
    
    
    
    
    