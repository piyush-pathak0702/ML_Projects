import pandas as pd
from sklearn import datasets

iris=datasets.load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)

df['target']=iris.target
df['flower_name']=df['target'].apply(lambda x: iris.target_names[x])

print (df.columns)


X=df.drop(['target', 'flower_name'],axis=1)
y=df['target']



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

model_selection={
    'Logistic_Regression':{
    'Model': LogisticRegression(max_iter=5000),
    'params':{
    'C': [1,5,10]
    }
    },
    'SVM':{
    'Model': SVC(),
    'params':{
    'C': [1,10,20],
    'kernel': ['rbf','linear']
    }
    },
    'Random_Forest':{
    'Model': RandomForestClassifier(),
    'params':{
    'n_estimators':[1,5,10]
    }
    }
}

from sklearn.model_selection import GridSearchCV

scores=[]
for model_name,mp in model_selection.items():
    clf=GridSearchCV(mp['Model'],mp['params'],cv=3,return_train_score=False)
    clf.fit(X,y)
    scores.append({
        'model':model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
        })

print (pd.DataFrame(scores,columns=['model','best_score','best_params']))
