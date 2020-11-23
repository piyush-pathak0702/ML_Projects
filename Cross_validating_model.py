import pandas as pd
from sklearn import datasets

iris=datasets.load_iris()

df=pd.DataFrame(iris['data'],columns=iris['feature_names'])
df['target']=iris['target']
df['flower_name']=df['target'].apply(lambda x: iris['target_names'][x])

print (df.columns)

X=df.drop(['flower_name','target'],axis=1)
y=df['target']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier




from sklearn.model_selection import cross_val_score

cross_val_score(LogisticRegression(max_iter=4000),X,y,cv=3)
cross_val_score(SVC(),X,y,cv=3)
cross_val_score(RandomForestClassifier(),X,y,cv=3)

print ("Logistic Regression:: " +str((cross_val_score(LogisticRegression(max_iter=1000),X,y,cv=3)).mean()))
print ("Support Vector Machine:: " +str((cross_val_score(SVC(),X,y,cv=3)).mean()))
print ("Random Forest:: " +str((cross_val_score(RandomForestClassifier(n_estimators=80),X,y,cv=3)).mean()))

