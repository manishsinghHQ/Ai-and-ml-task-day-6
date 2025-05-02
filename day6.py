import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib_inline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
iris=load_iris()
iris.feature_names
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()
df.shape
df['target']=iris.target
df.head()
df.tail(10)
df[df.target==1].head()
#Normalizing features
df0=df[0:50]
df1=df[50:100]
df2=df[100:]
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='.')
x=pd.DataFrame(iris.data)
y=df.target
knn=KNeighborsClassifier(n_neighbors=2)#playing with n values and by giving 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
knn.fit(x_train,y_train)
accuracy=knn.score(x_test,y_test)
print(accuracy)
y_pred=knn.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)

