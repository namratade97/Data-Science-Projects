import flask
from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import sklearn as s
from sklearn import datasets,linear_model
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures #PolynomialRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('home1(edited).html')

@app.route('/predict1',methods=['POST'])
def predict1():
	df=pd.read_csv('Data.csv',sep=',',header=0)
	df = df[df.test_time >= 0] #df is the common corpus


	#Features and Labels
	#1. For NHR
	X1 = df.drop(["motor_UPDRS","total_UPDRS", "Jitter(Percent)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP", "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:DDA","NHR","RPDE","DFA","PPE"], axis=1)
	Y1 = df.NHR

	target_y_1 = Y1.as_matrix()
	features_x_1 = X1.as_matrix()

	X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(features_x_1, target_y_1, test_size=0.18, random_state=42)

	# import the regressor 
	 #DecisionTreeRegression

	# create a regressor object 
	regressor = DecisionTreeRegressor(random_state = 0) 

	# fit the regressor with X and Y data 
	regressor.fit(X_train_1, Y_train_1)
	#predictions1 = regressor.predict(X_test)

	if request.method == 'POST':
		result=request.form

		age = result['age']
		test_time = result['test_time']
		JitterAbsolute = result['JitterAbsolute']
		ShimmerAPQ11 = result['ShimmerAPQ11']
		HNR = result['HNR']


		user_input=pd.DataFrame([[age, test_time, JitterAbsolute, ShimmerAPQ11,HNR]],columns=['age', 'test_time', 'JitterAbsolute', 'ShimmerAPQ11','HNR'],dtype=float)
		#1. NHR
		my_prediction_nhr=regressor.predict(user_input)[0]
		return flask.render_template('result1(NHR).html',original_input={'age':age, 'test_time':test_time, 'JitterAbsolute':JitterAbsolute, 'ShimmerAPQ11':ShimmerAPQ11,'HNR':HNR},result=my_prediction_nhr)
		######################
	return render_template('result1(NHR).html',prediction=my_prediction_nhr)

		#########################################################################################################################

@app.route('/predict2',methods=['POST'])
def predict2():
	df=pd.read_csv('Data.csv',sep=',',header=0)
	df = df[df.test_time >= 0] #df is the common corpus


	#Features and Labels
	#2. For RPDE
	X2 = df.drop(["motor_UPDRS","total_UPDRS", "Jitter(Percent)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP", "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:DDA","NHR","RPDE","DFA","PPE"], axis=1)
	Y2 = df.RPDE

	target_y_2 = Y2.as_matrix()
	features_x_2 = X2.as_matrix()

	X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(features_x_2, target_y_2, test_size=0.18, random_state=42)

	#RandomForest
	regressor1 = RandomForestRegressor(n_estimators=100, random_state=1)  
	regressor1.fit(X_train_2, Y_train_2)

	
	if request.method == 'POST':
		result=request.form

		age = result['age']
		test_time = result['test_time']
		JitterAbsolute = result['JitterAbsolute']
		ShimmerAPQ11 = result['ShimmerAPQ11']
		HNR = result['HNR']


		user_input=pd.DataFrame([[age, test_time, JitterAbsolute, ShimmerAPQ11,HNR]],columns=['age', 'test_time', 'JitterAbsolute', 'ShimmerAPQ11','HNR'],dtype=float)
		#2.RPDE
		my_prediction_rpde=regressor1.predict(user_input)[0]
		#####################
		return flask.render_template('result2(RPDE).html',original_input={'age':age, 'test_time':test_time, 'JitterAbsolute':JitterAbsolute, 'ShimmerAPQ11':ShimmerAPQ11,'HNR':HNR},result=my_prediction_rpde)
		######################
	return render_template('result2(RPDE).html',prediction=my_prediction_rpde)

		#########################################################################################################################
@app.route('/predict3',methods=['POST'])
def predict3():
	df=pd.read_csv('Data.csv',sep=',',header=0)
	df = df[df.test_time >= 0] #df is the common corpus


	#Features and Labels
	#3. For DFA
	X3 = df.drop(["motor_UPDRS","total_UPDRS", "Jitter(Percent)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP", "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:DDA","NHR","RPDE","DFA","PPE"], axis=1)
	Y3 = df.DFA

	target_y_3 = Y3.as_matrix()
	features_x_3 = X3.as_matrix()

	X_train_3, X_test_3, Y_train_3, Y_test_3 = train_test_split(features_x_3, target_y_3, test_size=0.18, random_state=42)
	#RandomForest
	regressor2 = RandomForestRegressor(n_estimators=100, random_state=1)  
	regressor2.fit(X_train_3, Y_train_3)

	
	if request.method == 'POST':
		result=request.form

		age = result['age']
		test_time = result['test_time']
		JitterAbsolute = result['JitterAbsolute']
		ShimmerAPQ11 = result['ShimmerAPQ11']
		HNR = result['HNR']


		user_input=pd.DataFrame([[age, test_time, JitterAbsolute, ShimmerAPQ11,HNR]],columns=['age', 'test_time', 'JitterAbsolute', 'ShimmerAPQ11','HNR'],dtype=float)
		#3.DFA
		my_prediction_dfa=regressor2.predict(user_input)[0]
		#####################
		return flask.render_template('result3(DFA).html',original_input={'age':age, 'test_time':test_time, 'JitterAbsolute':JitterAbsolute, 'ShimmerAPQ11':ShimmerAPQ11,'HNR':HNR},result=my_prediction_dfa)
		######################
	return render_template('result3(DFA).html',prediction=my_prediction_dfa)

		###################################################################################################

@app.route('/predict4',methods=['POST'])
def predict4():
	df=pd.read_csv('Data.csv',sep=',',header=0)
	df = df[df.test_time >= 0] #df is the common corpus


	#4. For PPE
	X4 = df.drop(["motor_UPDRS","total_UPDRS", "Jitter(Percent)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP", "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:DDA","NHR","RPDE","DFA","PPE"], axis=1)
	Y4 = df.PPE

	target_y_4 = Y4.as_matrix()
	features_x_4 = X4.as_matrix()

	X_train_4, X_test_4, Y_train_4, Y_test_4 = train_test_split(features_x_4, target_y_4, test_size=0.18, random_state=42)

	#RandomForest
	regressor3 = RandomForestRegressor(n_estimators=100, random_state=1)  
	regressor3.fit(X_train_4, Y_train_4)
	
	if request.method == 'POST':
		result=request.form

		age = result['age']
		test_time = result['test_time']
		JitterAbsolute = result['JitterAbsolute']
		ShimmerAPQ11 = result['ShimmerAPQ11']
		HNR = result['HNR']


		user_input=pd.DataFrame([[age, test_time, JitterAbsolute, ShimmerAPQ11,HNR]],columns=['age', 'test_time', 'JitterAbsolute', 'ShimmerAPQ11','HNR'],dtype=float)
		#3.PPE
		my_prediction_ppe=regressor3.predict(user_input)[0]
		#####################
		return flask.render_template('result4(PPE).html',original_input={'age':age, 'test_time':test_time, 'JitterAbsolute':JitterAbsolute, 'ShimmerAPQ11':ShimmerAPQ11,'HNR':HNR},result=my_prediction_ppe)
		######################
	return render_template('result4(PPE).html',prediction=my_prediction_ppe)
	####################################################################################

	#######################################################################################
if __name__ == '__main__':
   app.run(debug=True)


