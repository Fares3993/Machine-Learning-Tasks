import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X
data = pd.read_csv('assignment2_dataset_cars.csv')
allData = data.iloc[:,:]
#Deal with missing values
print(data.isna().sum())
X=data.iloc[:,0:3] #Features
Y=data['price'] #Label
#Feature Encoding
lbl = LabelEncoder()
lbl.fit(list(X['car_maker'].values))
X['car_maker'] = lbl.transform(list(X['car_maker'].values))
#Normalization
X = featureScaling(X,0,1)

#split the data to training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,train_size=0.80,shuffle=False)

#Get the correlation between the features

corr = allData.corr()
selectedFeature = corr.index[abs(corr['price'])>0.0]
plt.subplots(figsize=(12,8))
top_corr = allData[selectedFeature].corr()
sns.heatmap(top_corr,annot = True)
plt.show()



#polynomial regression
polyFeature = PolynomialFeatures(degree=5)
X_train_poly= polyFeature.fit_transform(X_train)


cls = linear_model.LinearRegression()
cls.fit(X_train_poly,Y_train)
Y_train_predict = cls.predict(X_train_poly)

prediction = cls.predict(polyFeature.fit_transform(X_test))
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print("Mean Square Error",metrics.mean_squared_error(np.array(Y_test),prediction))
print("true first",str(np.asarray(Y_test)[0]))
print("predict first",str(prediction[0]))