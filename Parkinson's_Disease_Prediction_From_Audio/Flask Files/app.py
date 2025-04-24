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

# load the built-in model 
#gbr = joblib.load('model.pkl')

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('2.html')


@app.route('/predict',methods=['POST'])
def predict():
	df=pd.read_csv('Data.csv',sep=',',header=0)
	df = df[df.test_time >= 0] 
	#Features and Labels
	X = df.drop(["motor_UPDRS","total_UPDRS", "Jitter(Percent)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP", "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:DDA"], axis=1)
	Y = df.total_UPDRS
	########
	
	# Converts target and features to numpy arrays for sklearn API
	target_y = Y.as_matrix()
	features_x = X.as_matrix()
	# Split into train and test sets
	X_train, X_test, Y_train, Y_test = train_test_split(features_x, target_y, test_size=0.18, random_state=42)
	#LinearRegression
	lm = RandomForestRegressor(n_estimators=300, random_state=20) #RandomForestRegression
	model = lm.fit(X_train, Y_train) #actual "learning" happens here
	lm.score(X_test,Y_test)
	#######
	 

	  
	

	if request.method == 'POST':
		result=request.form
		age = result['age']
		test_time = result['test_time']
		JitterAbsolute = result['JitterAbsolute']
		ShimmerAPQ11 = result['ShimmerAPQ11']
		NHR = result['NHR']
		HNR = result['HNR']
		RPDE = result['RPDE']
		DFA = result['DFA']
		PPE = result['PPE']




		# we create a json object that will hold data from user inputs
		#user_input = {'age':age, 'test_time':test_time, 'JitterAbsolute':JitterAbsolute, 'ShimmerAPQ11':ShimmerAPQ11, 'NHR':NHR,'HNR':HNR,'RPDE':RPDE,'DFA':DFA,'PPE':PPE}
		user_input=pd.DataFrame([[age, test_time, JitterAbsolute, ShimmerAPQ11, NHR,HNR,RPDE,DFA,PPE]],columns=['age', 'test_time', 'JitterAbsolute', 'ShimmerAPQ11', 'NHR','HNR','RPDE','DFA','PPE'],dtype=float)
		

		
		my_prediction=lm.predict(user_input)[0]
		
		return flask.render_template('result1.1.html',original_input={'age':age, 'test_time':test_time, 'JitterAbsolute':JitterAbsolute, 'ShimmerAPQ11':ShimmerAPQ11, 'NHR':NHR,'HNR':HNR,'RPDE':RPDE,'DFA':DFA,'PPE':PPE},result=my_prediction)


 
	return render_template('result1.1.html',prediction=my_prediction)
	

		 


if __name__ == '__main__':
   app.run(debug=True)