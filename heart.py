import numpy as np #importing the numpy module which will be used in this project
import pandas as pd#importing the pandas module which will be used in this project
import matplotlib.pyplot as plt#importing the matplotlib module which will be used in this project
#import seaborn as sns#importing the seaborn module which will be used in this project
#from sklearn.model_selection import train_test_split#importing the sklearn module which will be used in this project
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


dataframe = pd.read_csv("heart-disease-prediction/dataset.csv", encoding="latin1")#reading our dataset using read_csv function
dataframe.head() #printing the first 5 columns of our dataset using head function
dataframe.drop('education', axis=1, inplace=True)
dataframe.rename(columns={"TenYearCHD": "CHD"}, inplace=True)

x = dataframe.iloc[:,:-1]
y = dataframe.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
 
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

sns.countplot(x=train_data['male']) ##plotting a count plot of male using sns.countplot

sns.countplot(x=train_data['male'], hue=train_data['CHD']) ##plotting a count plot of CHD and male having disease or not using sns.countplot
plt.figure(figsize=(15,15))#plotting a figure of size 15 and 15
sns.heatmap(train_data.corr(), annot=True, linewidths=0.1)

plt.show()
train_data.drop(['currentSmoker', 'diaBP'], axis=1, inplace=True)

train_data = train_data[~(train_data['sysBP'] > 220)] #deleting the outliers values in sysBP of training data
train_data = train_data[~(train_data['BMI'] > 43)]#deleting the outliers values in BMI of training data
train_data = train_data[~(train_data['heartRate'] > 125)]#deleting the outliers values in heartRate of training data
train_data = train_data[~(train_data['glucose'] > 200)]#deleting the outliers values in glucose of training data
train_data = train_data[~(train_data['totChol'] > 450)]#deleting the outliers values in totChol of training data

from sklearn.preprocessing import StandardScaler #importing the standard scaler library
cols_to_standardise = ['age','totChol','sysBP','BMI','heartRate','glucose','cigsPerDay']
scaler = StandardScaler()
train_data[cols_to_standardise] = scaler.fit_transform(train_data[['age','totChol','sysBP','BMI', 'heartRate', 'glucose', 'cigsPerDay']])#taking all the columns which are to be standardise in an array

test_data.drop(['currentSmoker', 'diaBP'], axis=1, inplace=True)
#print(test_data.columns)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')#Creating an instance of simple Imputer which will be used to fill the null vlaues
#test_data = pd.DataFrame(imputer.fit_transform(test_data))#fitting the data and filling any null values in the test dataset


test_data = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)

test_data[cols_to_standardise] = scaler.fit_transform(test_data[['age','totChol','sysBP','BMI', 'heartRate', 'glucose', 'cigsPerDay']])#taking all the columns which are to be standardise in an array



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 #importing the descision tree classifier from the sklearn tree 

tree = LogisticRegression() #making an instance the descision tree with maxdepth = 3 as passing the input
clf = tree.fit(X_train,y_train) #here we are passing our training and the testing data to the tree and fitting it
y_pred = clf.predict(X_test) #predicting the value by passing the x_test datset to the tree 
accuracy_score(y_pred,y_test)# here we are printing the accuracy score of the prediction and the testing data


from sklearn.tree import DecisionTreeClassifier #importing the descision tree classifier from the sklearn tree 
tree = DecisionTreeClassifier(max_depth=3) #making an instance the descision tree with maxdepth = 3 as passing the input
clf = tree.fit(X_train,y_train) #here we are passing our training and the testing data to the tree and fitting it
y_pred = clf.predict(X_test) #predicting the value by passing the x_test datset to the tree 
accuracy_score(y_pred,y_test)# here we are printing the accuracy score of the prediction and the testing data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
 #importing the k nearest classifier from the sklearn neighbors 
rf = RandomForestClassifier(n_estimators=3, random_state=42) #making an instance the k nearest neighbors with neighbors = 3 as passing the input
rf.fit(X_train, y_train) #here we are passing our training and the testing data to the tree and fitting it
y_pred = rf.predict(X_test) #predicting the value by passing the x_test datset to the tree 
accuracy_score(y_pred,y_test)# here we are printing the accuracy score of the prediction and the testing data



print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

#print(test_data.columns)



#print(sns.catplot)
#print(train_data)
#print(test_data)
#print(sns.heatmap)
#print(plt.figure)
