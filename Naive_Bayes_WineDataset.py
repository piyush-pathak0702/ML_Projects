import pandas as pd
from sklearn import datasets

wine_data=datasets.load_wine()

print (wine_data['feature_names'])
print (wine_data['target_names'])

df=pd.DataFrame(wine_data['data'],columns=wine_data['feature_names'])

df['target']=wine_data['target']
df['names']=df['target'].apply(lambda x: wine_data['target_names'][x])
print (df.columns)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(),annot=True)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score

X=df.drop(['target', 'names'],axis=1)
y=df['target']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=1)

GNB=GaussianNB()
GNB.fit(X_train,y_train)

MNB=MultinomialNB()
MNB.fit(X_train,y_train)

Prediction_GNB=GNB.predict(X_test)
print ("GNB accuracy: ",accuracy_score(Prediction_GNB,y_test))
print ((cross_val_score(GaussianNB(),X,y,cv=20)).mean())

Prediction_MNB=MNB.predict(X_test)
print ("MNB accuracy: ",accuracy_score(Prediction_MNB,y_test))
print ((cross_val_score(MultinomialNB(),X,y,cv=20)).mean())
