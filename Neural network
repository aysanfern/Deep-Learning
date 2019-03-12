# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,-1].values



# Changing the categorical variables into binary variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder1=LabelEncoder()
X[:,1]=labelencoder1.fit_transform(X[:,1])
labelencoder2=LabelEncoder()
X[:,2]=labelencoder2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features= [1])
X=onehotencoder.fit_transform(X).toarray()

# To avoid dummy variable trap
X=X[:,1:]




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating the neural network with 1 input layer, 2 hidden layers and 1 output layer
from keras.models import Sequential
from keras.layers import Dense

classifier= Sequential()

classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)


# Predicting the results on the X test set
y_pred=classifier.predict(X_test)

# Changing the results into a binary outcome so it can be used for the confusion matrix
y_pred = (y_pred>0.5)

# Using a confusion matrix to see how well the classifier performed
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


