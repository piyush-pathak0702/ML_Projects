import pandas as pd
from sklearn import datasets

Data=datasets.load_breast_cancer()
print (dir(Data))

df=pd.DataFrame(Data.data,columns=Data.feature_names)
df['cancer']=Data.target
df['cancer_name']=df['cancer'].apply(lambda x: Data.target_names[x])

print (df.columns)
print (df.info())

X=df.drop(['cancer','cancer_name'],axis=1)
y=df['cancer']


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(X.corr(),cmap='PuOr')
#plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Model Selection based on score
Model_selection={
    'Logistic Regression':{
        'Model':LogisticRegression(max_iter=5000),
        'Parameters':{
        'C':[1,5,10]
    }
    },
    'SVM':{
        'Model':SVC(),
        'Parameters':{
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }
    },
    'Random Forest':{
        'Model': RandomForestClassifier(),
        'Parameters':{
        'n_estimators':[1,5,10,15,20,25,30]
    }
    },
    'Decision Tree':{
    'Model': DecisionTreeClassifier(),
    'Parameters':{
    'criterion': ['entropy','gini']
    }
    }
}

from sklearn.model_selection import GridSearchCV
scores=[]

for model_name,par in Model_selection.items():
    GSV=GridSearchCV(par['Model'],par['Parameters'],cv=3,return_train_score=False)
    GSV.fit(X,y)
    scores.append({
        'Model':model_name,
        'Best_score': GSV.best_score_,
        'Best_parameters': GSV.best_params_
        })

print (pd.DataFrame(scores,columns=['Model','Best_score','Best_parameters']))


# Since Logistc Regression, SVM, Random Forrest all have a score around 95% will try to check the accuracy for this data with all the models


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=1)


from sklearn.metrics import confusion_matrix,accuracy_score


# Logistc Regression
LR=LogisticRegression(max_iter=5000,C=10)
LR.fit(X_train,y_train)

Prediction=LR.predict(X_test)
print ("Accuracy score with Logistic Regression:: " +str(accuracy_score(y_test,Prediction)))


#SVM
SVM=SVC(C=10,kernel='linear')
SVM.fit(X_train,y_train)

Prediction=SVM.predict(X_test)
print ("Accuracy score with SVM :: " +str(accuracy_score(y_test,Prediction)))


#Random Forest
RF=RandomForestClassifier(n_estimators=20)
RF.fit(X_train,y_train)

Prediction=RF.predict(X_test)
print ("Accuracy score with Random Forest :: " +str(accuracy_score(y_test,Prediction)))


