import pandas as pd
from sklearn import datasets

iris=datasets.load_iris()
print (dir(iris))

df=pd.DataFrame(iris['data'],columns=iris['feature_names'])
df['target']=iris['target']
print (df.columns)


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(),annot=True,cmap='PuOr')
plt.scatter(df['petal length (cm)'],df['petal width (cm)'])



from sklearn.preprocessing import MinMaxScaler

Scaler=MinMaxScaler()
Scaler.fit(df[['petal length (cm)']])
df['petal length (cm)']=Scaler.transform(df[['petal length (cm)']])

Scaler.fit(df[['petal width (cm)']])
df['petal width (cm)']=Scaler.transform(df[['petal width (cm)']])


from sklearn.cluster import KMeans

sse=[]
k_range=range(1,11)
for k in k_range:
    Model=KMeans(n_clusters=k)
    Model.fit(df[['petal length (cm)','petal width (cm)']])
    sse.append(Model.inertia_)

plt.plot(k_range,sse)
plt.show()


# Selecting cluster=3 based on elbow curve
Model=KMeans(n_clusters=3)
Prediction=Model.fit_predict(df[['petal length (cm)','petal width (cm)']])
print (Model.cluster_centers_)

df['Prediction']=Prediction

df0=df[df['Prediction']==0]
df1=df[df['Prediction']==1]
df2=df[df['Prediction']==2]
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='red')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='green')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='blue')
plt.scatter(Model.cluster_centers_[:,0],Model.cluster_centers_[:,1],color='black',marker='*')
plt.show()



