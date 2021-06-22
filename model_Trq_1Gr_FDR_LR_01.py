# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('JL_JT_ VPTDuty_CycleData_vehdataTN.csv')

dataset.columns = ['Configuration', 'Torque','Power', '1ST Gear', 'FDR',
       'LAP','GVW','GCW','Tire','RearAxleTN', 'FrontAxleTN']

dataset['FrontAxleTN'].fillna(0, inplace=True)

#dataset['RearAxleTN'].replace(0,dataset['RearAxleTN'].median(), inplace=True)

#dataset['FrontAxleTN'].replace(0,dataset['FrontAxleTN'].median(), inplace=True)
dataset['FrontAxleTN'].replace(0,dataset.FrontAxleTN[dataset.FrontAxleTN >0].median(), inplace=True)


#Based on EDA & Features correlation found below 4 features as final
X = dataset[['Power', 'FDR','GVW','Tire']]

#y = dataset.iloc[:, -1]
y = dataset["RearAxleTN"]

# Split dataset in to train & test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



# Since we have a very small dataset, we will train our model with all availabe data.
# We will use Grid Serch Cross validation method to usitilize data well

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

ridge = Ridge()
para = {'alpha':[1e-15,1e-10, 1e-8, 1e-3, 1e-2,1,5,10,20,30,35,40,45,50,55,100]}

ridge_regre =GridSearchCV(ridge, para, scoring='neg_mean_squared_error', cv=8)


#Fitting model with trainig data
ridge_regre.fit(X,y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(ridge_regre.predict([[400, 4.1, 6450, 0.4033]]))
print(ridge_regre.predict([[400, 3.73, 6350, 0.398]]))


# Get model performance
y_pred = ridge_regre.predict(X_test)
score = r2_score(y_test,y_pred)
print("R2 value from  model: ", score)


# Adjusted R-square
p = X_train.shape[1]
n = X_train.shape[0]
adjr= 1-(1-score)*(n-1)/(n-p-1)
print("Adj-R2 value from  model: ",adjr)





