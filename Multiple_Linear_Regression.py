#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
df=pd.read_csv('50_Startups.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,4].values

#Transforming Categprical Variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
ct=ColumnTransformer([('country',OneHotEncoder(),[3])],remainder='passthrough')
X=ct.fit_transform(X)

#determining independent features
X=X[:,1:]

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Building the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test result
y_pred=regressor.predict(X_test)

#Performing Backward Elimination
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1),dtype=np.int),values=X,axis=1)
X_opt=X[:, [0, 1, 2, 3, 4, 5]]
X_opt=X_opt.astype(np.float64)
y=y.astype(np.float64)  
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()  
print(regressor_OLS.summary())


X_opt=X[:, [0, 1, 3, 4, 5]]
X_opt=X_opt.astype(np.float64)
y=y.astype(np.float64)  
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()  
print(regressor_OLS.summary())

X_opt=X[:, [0, 3, 4, 5]]
X_opt=X_opt.astype(np.float64)
y=y.astype(np.float64)  
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()  
print(regressor_OLS.summary())

X_opt=X[:, [0, 3, 5]]
X_opt=X_opt.astype(np.float64)
y=y.astype(np.float64)  
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()  
print(regressor_OLS.summary())

X_opt=X[:, [0, 3]]
X_opt=X_opt.astype(np.float64)
y=y.astype(np.float64)  
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()  
print(regressor_OLS.summary())
