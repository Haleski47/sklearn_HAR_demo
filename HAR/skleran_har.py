from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# sensor data
X_train = pd.read_csv('./UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None)
X_test = pd.read_csv('./UCI HAR Dataset/test/X_test.txt', sep='\s+', header=None)

# sensor labels, 现在版本中names应该是一个list
sensor_labels = pd.read_csv('./UCI HAR Dataset/features.txt', sep=' ', header=None, names=['Sensor'])

# activity class
y_train = pd.read_csv('./UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None, names=['Y'])
y_test = pd.read_csv('./UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None, names=['Y'])

X_train.columns = list(sensor_labels['Sensor'])
X_test.columns = list(sensor_labels['Sensor'])


svc = LinearSVC(random_state=42)
svc.fit(X_train, y_train.values.ravel())

train_pred = svc.predict(X_train)
y_pred = svc.predict(X_test)

print('Accuracy score TRAIN:{}'.format(accuracy_score(y_train, train_pred)))
print('Accuracy score TEST:{}'.format(accuracy_score(y_test, y_pred)))
