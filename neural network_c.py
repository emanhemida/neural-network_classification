#labraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
#read dataset
data=pd.read_csv("Iris.csv")
#analysing
print(data.head())
print(data.isnull().sum())
#scaling
x= data.iloc[:,:-1]
y= data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.89)
model=MLPClassifier (hidden_layer_sizes=(65,130,35),activation="relu",solver="adam",learning_rate="constant",learning_rate_init=0.002, max_iter=1000)

model.fit(x_train,y_train)

print(model.score(x_train,y_train))

print(model.score(x_test,y_test))



