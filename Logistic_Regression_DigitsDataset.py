import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

data_set=datasets.load_digits()

print (dir(data_set))
print (data_set['target'][0])

plt.gray()
for i in range(2):
    plt.matshow(data_set.images[i])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

X=data_set['data']
y=data_set['target']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)
LR=LogisticRegression(max_iter=4000)
LR.fit(X_train,y_train)

predict=LR.predict(X_test)
accuracy=accuracy_score(y_test,predict)
matrix=confusion_matrix(y_test,predict)

print (accuracy)
print (matrix)


import seaborn as sns
sns.heatmap(matrix,annot=True)
plt.show()
