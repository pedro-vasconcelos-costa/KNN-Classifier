# import necessary modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Euclidean Distance = 2 dimensional x,y distance between points on a 2 dimensional plan
# Euclidean Distance (d) = √(x2 - x1)p2 + (y2 - y1)p2

# dataset downloaded from kaggle at https://www.kaggle.com/datasets/saurabh00007/iriscsv?resource=download

data = pd.read_csv('Iris.csv')
# create variable with all parameters minus species
x = data.iloc[:, 1:-1].values

# create variable with species only
y = data.iloc[:, -1].values

# encoding the categorical dependent variable
l_encode = LabelEncoder()
y = l_encode.fit_transform(y)

# splitting the data into training and testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# KNN classifier

from sklearn.neighbors import KNeighborsClassifier
# n_neighbors=5 = defines number of nearest neighbours to be accounted for

# metric="minkowski" = Minkowski Distance – It is a metric intended for real-valued vector spaces.
# We can calculate Minkowski distance only in a normed vector space, which means in a space where distances
# can be represented as a vector that has a length and the lengths cannot be negative.

# p=2 = 2 selects the 'Euclidean' distance, the p2 sets the Minkowski formula to generate Euclidean distances
# explained at https://www.kdnuggets.com/2020/11/most-popular-distance-metrics-knn.html

knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)

# training the algorithm

knn.fit(x_train, y_train)

# predicting test results
y_pred = knn.predict(x_test)

# reshape prediction data into a single vertical column
y_pred_vertical = y_pred.reshape(len(y_pred), 1)
# reshape y test data into a single vertical column
y_true_vertical = y_test.reshape(len(y_test), 1)
# concatenate y pred and y true + set for horizontal axis
true_pred = np.concatenate((y_true_vertical, y_pred_vertical), axis=1)

# plotting the confusion matrix

# generating confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# plotting a heat map with the confusion matrix
sns.heatmap(confusion_mat, annot= True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# calculate accuracy of the model
from sklearn.metrics import accuracy_score
accScore = accuracy_score(y_test, y_pred)

# evaluating the classifier on new data
prediction = knn.predict([[5, 3, 1.6, 0.2]])
