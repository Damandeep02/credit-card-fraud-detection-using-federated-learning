# SMOTE
import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'C:\Users\DAMAN\Desktop\Pre-Project\creditcard_train.csv')
print(data.head())

pd.value_counts(data['Class']).plot.bar()
plt.title('Fraud Detection histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
print(data['Class'].value_counts())

print('***************************************************')

X = np.array(data.loc[:, data.columns != 'Class'])
y = np.array(data.loc[:, data.columns == 'Class'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

print('***************************************************')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print('***************************************************')

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

print('')
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print('')
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
print('')
print('***************************************************')



y_train_res = y_train_res.reshape(-1, 1) # reshaping y_train to (398038,1)
data_res = np.concatenate((X_train_res, y_train_res), axis = 1)
np.savetxt('creditcard_train_SMOTE_1.csv', data_res, delimiter=",")