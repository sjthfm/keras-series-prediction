import warnings
import itertools
import os
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras import optimizers
from tensorflow import keras



def setRandomSeed(x):
	np.random.seed(x)

def loadFile(_fileName, _delimiter, _nameOfIndexColumn):
	pdFrame = pd.read_csv(_fileName, delimiter=_delimiter, index_col=_nameOfIndexColumn)
	return pdFrame

def createLists(amountOfSteps):
	trainingMSE = []
	testingMSE = []
	predictedForecast = []

	for i in range(amountOfSteps):
		trainingMSE.append(0)
		testingMSE.append(0)
		predictedForecast.append(0)

	return trainingMSE, testingMSE, predictedForecast



def run():
	setRandomSeed(123456789)
	trainingIterations = 5
	epochNumber = 5
	stepsToPredict = 7
	batchSize = 1

	datasetFrame = loadFile('dataset/sales_data.csv', ',', 'date')
	dates = datasetFrame.index

	trainingMSE, testingMSE, modelForecast = createLists(stepsToPredict)

	yTargetData = datasetFrame.iloc[:,6]

	yTargetShiftAmount = -1
	for i in range(stepsToPredict):
		print(f"\n\nIteration Round {i+1} out of {stepsToPredict}\n\n")

		yTargetData_Shifted = yTargetData.shift(yTargetShiftAmount-i)
		yDifferencedTarget = (yTargetData - yTargetData_Shifted)

		data = pd.concat([yTargetData, yTargetData_Shifted, yDifferencedTarget], axis=1)
		data.columns = ['yTargetData', 'yTargetData_Shifted', 'yDifferencedTarget']
		data = data.dropna()

		y = data ['yTargetData_Shifted']
		cols =['yTargetData', 'yDifferencedTarget']
		x = data [cols]

		scaler_x = preprocessing.MinMaxScaler ( feature_range =(0, 1))
		x = np. array (x).reshape ((len( x) ,len(cols)))
		x = scaler_x.fit_transform (x)
	 
		scaler_y = preprocessing. MinMaxScaler ( feature_range =(0, 1))
		y = np.array (y).reshape ((len( y), 1))
		y = scaler_y.fit_transform (y)

		xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, shuffle=False) #test_size = 0.4 train_size=0.6?
	

		if (i == 0):	 
			prediction_data=[]
			for j in range (len(yTest) - 0) :
				prediction_data.append(0)

		model = Sequential()
		model.add(Dense(300, activation='tanh', input_dim = 2))
		model.add(Dense(90, activation='tanh'))
		model.add(Dropout(0.2))
		model.add(Dense(30, activation='tanh'))
		model.add(Dropout(0.2))
		model.add(Dense(7, activation='tanh')) #relu etc. 
		model.add(Dropout(0.2))
		model.add(Dense(1, activation='relu')) #linear
		model.compile(optimizer='adam', loss='mean_squared_error')
		early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=2)

		model.fit(xTrain, yTrain, epochs=epochNumber, batch_size=batchSize, shuffle=False, callbacks=[early_stop])


		trainingMSE[i] = model.evaluate(xTrain, yTrain, batch_size=batchSize)
		testingMSE[i] = model.evaluate(xTest, yTest, batch_size=batchSize)
		pred = model.predict(xTest, batch_size=batchSize)
		pred = scaler_y.inverse_transform (np. array (pred). reshape ((len( pred), 1)))
		modelForecast[i]=pred[-1]

		if (i == 0):
			for j in range (len(pred) - 0 ) :
				prediction_data[j] = pred[j] 
			modelForecast[i]=pred[-1]
			answeryTestTest = scaler_y.inverse_transform (yTest)
			answerXTest = scaler_x.inverse_transform (np. array (xTest). reshape ((len( xTest), len(cols))))

		xTest = scaler_x.inverse_transform (np. array (xTest). reshape ((len( xTest), len(cols))))
		
	prediction_data = np.asarray(prediction_data)
	prediction_data = prediction_data.ravel()

	for j in range (len(prediction_data) - 1 ):
		prediction_data[len(prediction_data) - j - 1  ] =  prediction_data[len(prediction_data) - 1 - j - 1]

	prediction_data = np.append(prediction_data, modelForecast)

	modelForecastOutput = np.concatenate(modelForecast).ravel()

	df = pd.DataFrame({'predictions': prediction_data})
	df2 = pd.DataFrame({'forecasts': modelForecastOutput})
	df3 = pd.DataFrame({'actual':answerXTest[:,0]})


	abc = [df, df2, df3]#, df4]
	testdf = pd.concat(abc, sort=False)
	testdf = testdf.reset_index()
	testdf = testdf.drop(columns=['index'])


	testdf.forecasts = testdf.forecasts.shift(-len(modelForecast))
	

	actual_sales = testdf.actual.dropna().reset_index().drop(columns=['index'])
	predictionslist = testdf.predictions.dropna().reset_index().drop(columns=['index'])
	forecast_list = testdf.forecasts.dropna().reset_index().drop(columns=['index'])

	salesActual = actual_sales.values[:,0]
	salesPrediction = predictionslist.values[:,0]

	testDex = []
	a = (len(salesPrediction)-len(modelForecastOutput))
	for i in range(stepsToPredict):
		testDex.append(a+i)

	print(testDex)

	resultsDf_Actual = pd.DataFrame({'actualSales': salesActual})
	resultsDf_Preds = pd.DataFrame({'salesPrediction': salesPrediction})
	resultsDf_Forecasts = pd.DataFrame({'salesForecasts':modelForecastOutput}, index=testDex)

	resultsDfList = [resultsDf_Actual, resultsDf_Preds, resultsDf_Forecasts]
	resultsDf = pd.concat(resultsDfList, sort=False, axis=1)


#	resultsDf['salesForecasts'] = resultsDf['salesForecasts'].shift(len(salesPrediction))
	#resultsDf['salesForecasts'] = resultsDf['salesForecasts'].shift(len(modelForecastOutput))
###################################################
	#print(DatasetFrame['date'])
#	resultsDf['date'] = DatasetFrame['date'][-len(resultsDf):].values
#	resultsDf['date']

	print(resultsDf)

	plt.plot(resultsDf['actualSales'].values, label='real_sales')
	plt.plot(resultsDf['salesPrediction'].values, label='predicted_sales')
	plt.plot(resultsDf['salesForecasts'].values, label='forecasted_sales')

#	plt.plot(resultsDf['salesForecasts'], label='forecasted_sales')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
	plt.show()

	resultsDf.to_csv("resultsDf.csv")

run()