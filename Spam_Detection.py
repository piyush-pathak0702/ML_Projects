import pandas as pd
from sklearn.preprocessing import LabelEncoder

spam_data=pd.read_csv('Practice/Data/spam.csv')
Encode=LabelEncoder()
spam_data['Spam']=Encode.fit_transform(spam_data['Category'])
label_maping=dict(zip(Encode.classes_,Encode.transform(Encode.classes_)))

spam_data=spam_data.drop(['Unnamed: 2','Category'],axis=1)
print (spam_data.columns)
print (label_maping)

X=spam_data['Message']
y=spam_data['Spam']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=1)


from sklearn.feature_extraction.text import CountVectorizer
CV=CountVectorizer()
X_train_counts=CV.fit_transform(X_train)
X_train_counts.toarray()
#print (X_train_counts.toarray())


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_counts,y_train)

X_test_count=CV.transform(X_test)
print ("Model_Score:: "+ str(model.score(X_test_count,y_test)))


emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

email_count=(CV.transform(emails)).toarray()
print ("Prediction:: " +str(model.predict(email_count)))

