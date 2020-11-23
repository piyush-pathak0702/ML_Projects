import pandas as pd
iris_data=pd.read_csv('iris.data')
print (iris_data.columns)

X=iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y=iris_data[['class']]



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)

#CCP alpha derived from cost complexity prunning
dt=DecisionTreeClassifier(ccp_alpha=0.14)
dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)
print (accuracy_score(y_pred,y_test))



from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
tree.plot_tree(dt,filled=True)

print (tree.export_text(dt))


# Cost Complexity Punning
path=dt.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas, impurities=path.ccp_alphas,path.impurities

dts=[]
for ccp_alpha in ccp_alphas:
    dt=DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    dt.fit(X_train,y_train)
    dts.append(dt)
print (ccp_alphas)

train_score=[dt.score(X_train,y_train) for dt in dts]
test_score=[dt.score(X_test,y_test) for dt in dts]

fig,ax=plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.plot(ccp_alphas,train_score,label="Train",marker="o")
ax.plot(ccp_alphas,test_score,label="Test",marker="o")
plt.legend()
plt.show()
