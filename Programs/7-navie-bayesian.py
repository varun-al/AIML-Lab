import pandas as  pd


msg=pd.read_csv('Dataset/naivetext.csv',names=['message','label'])
print("the dimension of the dataset",msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
print(X)
print(y)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y)
print("\n The total no of training data",ytrain.shape)
print("\n The total no of testing data",ytest.shape)

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
Xtrain_dtm=count_vect.fit_transform(Xtrain)
Xtest_dtm=count_vect.transform(Xtest)
print("\n the words or token in text document\n")
print(count_vect.get_feature_names_out())
df=pd.DataFrame(Xtrain_dtm.toarray(),columns=count_vect.get_feature_names_out())

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(Xtrain_dtm,ytrain)
predicted=clf.predict(Xtest_dtm)


from sklearn import metrics
print("\n accuracy of classifier",metrics.accuracy_score(ytest,predicted))
print("\n confusion matrix")
print(metrics.confusion_matrix(ytest,predicted))
print("\n the value of precision",metrics.precision_score(ytest,predicted))
print("\n the value of recall",metrics.recall_score(ytest,predicted))
