import pandas as pd
import matplotlib.pyplot as plt

Data=pd.read_csv('Practice/Data/insurance.csv')
print (Data.shape)

plt.boxplot(Data['bmi'])
plt.show()

Q1=Data['bmi'].quantile(0.25)
Q3=Data['bmi'].quantile(0.75)
IQR=Q3-Q1

min_range=Q1 - (1.5*IQR)
max_range=Q1 + (1.5*IQR)

Data_new= Data[(Data['bmi']>min_range)&(Data['bmi']<max_range)]
#print (Data.columns)
print (Data_new.shape)

Male=pd.get_dummies(Data_new['sex'],drop_first=True)
Smoker=pd.get_dummies(Data_new['smoker'],drop_first=True)
Region=pd.get_dummies(Data_new['region'],drop_first=True)

Data_01=pd.concat([Data_new,Male,Smoker,Region],axis=1)

Encoded_Data=Data_01.drop(['sex','smoker','region'],axis=1)
print (Encoded_Data.corr())



from sklearn.model_selection import train_test_split

X=Encoded_Data.drop(['charges'],axis=1)
y=Encoded_Data['charges']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=1)


from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import r2_score
import numpy as np

LR=LinearRegression()
LR.fit(X_train,y_train)

X_train_Sm= sm.add_constant(X_train)
ls=sm.OLS(y_train,X_train_Sm).fit()
print(ls.summary())


Prediction = LR.predict(X_test)

print('Mean Absolute Error:: ', metrics.mean_absolute_error(y_test, Prediction))
print('Mean Squared Error:: ', metrics.mean_squared_error(y_test, Prediction))
print('Root Mean Squared Error:: ', np.sqrt(metrics.mean_squared_error(y_test, Prediction)))

print (X.columns)

print (LR.score(X_test,y_test))
print (r2_score(y_test,Prediction))
print (LR.predict([[26,120,0,1,1,0,0,0]]))
