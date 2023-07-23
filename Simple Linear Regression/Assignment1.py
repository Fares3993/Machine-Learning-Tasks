import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
#Loading data
data = pd.read_csv('assignment1_dataset.csv')
print(data.describe())
def Linear_Regression(X,Y,name):

    cls = linear_model.LinearRegression()
    X = np.expand_dims(X, axis=1)
    Y = np.expand_dims(Y, axis=1)
    cls.fit(X, Y)  # Fit method is used for fitting your training data into the model
    prediction = cls.predict(X)
    plt.scatter(X, Y)
    plt.xlabel(name, fontsize=20)
    plt.ylabel('house price of unit area', fontsize=20)
    plt.plot(X, prediction, color='red', linewidth=3)
    plt.show()
    print('Co-efficient of linear regression', cls.coef_)
    print('Intercept of linear regression model', cls.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction))


transaction_date =[]
for i in range(len(data['transaction date'])):
    td = data['transaction date'][i][0:4]
    transaction_date.append(int(td))
Linear_Regression(transaction_date,data['house price of unit area'],'transaction date')
Linear_Regression(data['house age'],data['house price of unit area'],'house age')
Linear_Regression(data['distance to the nearest MRT station'],data['house price of unit area'],'distance to the nearest MRT station')
Linear_Regression(data['number of convenience stores'],data['house price of unit area'],'number of convenience stores')
Linear_Regression(data['latitude'],data['house price of unit area'],'latitude')
Linear_Regression(data['longitude'],data['house price of unit area'],'longitude')



