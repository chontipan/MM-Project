import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Genre dictionary
genres = {'Electronic':0,  'Folk':1,  'Pop' :2, 'Instrumental':3 }

# Read the dataset of audio features
data1 = np.load('suf_train_arr.npz')
data2 = np.load('suf_valid_arr.npz')
data3 = np.load('test_arr.npz')
x_tr = data1['arr_0']
y_tr = data1['arr_1']
x_te = data3['arr_0']
y_te = data3['arr_1']
x_cv = data2['arr_0']
y_cv = data2['arr_1']

print("Training X shape: " + str(x_tr.shape))
print("Training Y shape: " + str(y_tr.shape))
print("Validation X shape: " + str(x_cv.shape))
print("Validation Y shape: " + str(y_cv.shape))
print("Test X shape: " + str(x_te.shape))
print("Test Y shape: " + str(y_te.shape))

x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1]*x_tr.shape[2])
x_cv = x_cv.reshape(x_cv.shape[0], x_cv.shape[1]*x_cv.shape[2])
x_te = x_te.reshape(x_te.shape[0], x_te.shape[1]*x_te.shape[2])

X = np.vstack([x_tr, x_cv,x_te])
y = np.concatenate([y_tr, y_cv, y_te])
print("All data",X.shape)
print("All labels",y.shape)

"""# Visualization"""


X = skl.decomposition.PCA(n_components=2).fit_transform(X)
y = skl.preprocessing.LabelEncoder().fit_transform(y)

target_ids = range(4)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, genres):
    plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=label)
 
plt.legend()
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.9)
plt.show()

scale = StandardScaler()
x_scaled = scale.fit_transform(X)

X = skl.decomposition.PCA(n_components=2).fit_transform(X)
y = skl.preprocessing.LabelEncoder().fit_transform(y)

target_ids = range(4)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, genres):
    plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=label)
 
plt.legend()
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.9)
plt.show()

# Standardize features by removing the mean and scaling to unit variance.
scaler = skl.preprocessing.StandardScaler(copy=False)
x_tr = scaler.fit_transform(x_tr)
x_cv = scaler.transform(x_cv)
x_te = scaler.transform(x_te)

## Label encode y - data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_tr = le.fit_transform(y_tr)
y_cv = le.fit_transform(y_cv)
y_te = le.fit_transform(y_te)
le.classes_

#scale = StandardScaler()
#x_scaled = scale.fit_transform(X)

#scale.fit(x_tr)
#scale.fit(x_cv)
#scale.fit(x_te)
## Apply transform to both the training set and the test set.
train_sc = x_tr
cv_sc = x_cv
test_sc = x_te

print("Starting KNN")
neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')

neigh.fit(train_sc, y_tr)
train_preds = neigh.predict(train_sc)
train_acc = np.sum(train_preds == y_tr)
train_acc = train_acc / len(y_tr)

cv_preds = neigh.predict(cv_sc)
cv_acc = np.sum(cv_preds == y_cv)
cv_acc = cv_acc / len(y_cv)
test_preds = neigh.predict(test_sc)
test_acc = np.sum(test_preds == y_te)
test_acc = test_acc / len(y_te);

print('Train: ', train_acc, "\tCV: ", cv_acc, "\tTest: ", test_acc)

print("Starting SVM:")
svm = SVC(C=2, kernel='rbf', gamma="scale")
#cv_scores = cross_val_score(svm, X, y)
#print(cv_scores)
#print("SVM: cv_scores mean:{}".format(np.mean(cv_scores)))
svm.fit(train_sc,y_tr)
train_preds = svm.predict(train_sc)
train_acc = np.sum(train_preds == y_tr)
train_acc = train_acc / len(y_tr)

#svm.fit(cv_sc,y_cv)
cv_preds = svm.predict(cv_sc)
cv_acc = np.sum(cv_preds == y_cv)
cv_acc = cv_acc / len(y_cv)

#svm.fit(test_sc,y_te)
test_preds = svm.predict(test_sc)
test_acc = np.sum(test_preds == y_te)
test_acc = test_acc / len(y_te);

print('Train: ', train_acc, "\tCV: ", cv_acc, "\tTest: ", test_acc)


print("Starting DT")
cart = DecisionTreeClassifier(criterion = 'entropy')
cart.fit(train_sc,y_tr)
train_preds = cart.predict(train_sc)
train_acc = np.sum(train_preds == y_tr)
train_acc = train_acc / len(y_tr)

#svm.fit(cv_sc,y_cv)
cv_preds = cart.predict(cv_sc)
cv_acc = np.sum(cv_preds == y_cv)
cv_acc = cv_acc / len(y_cv)

#svm.fit(test_sc,y_te)
test_preds = cart.predict(test_sc)
test_acc = np.sum(test_preds == y_te)
test_acc = test_acc / len(y_te);
print("DT")
print('Train: ', train_acc, "\tCV: ", cv_acc, "\tTest: ", test_acc)


print("Starting LR")
lr = LogisticRegression(penalty='l2', C = 1)
lr.fit(train_sc,y_tr)
train_preds = lr.predict(train_sc)
train_acc = np.sum(train_preds == y_tr)
train_acc = train_acc / len(y_tr)

cv_preds = lr.predict(cv_sc)
cv_acc = np.sum(cv_preds == y_cv)
cv_acc = cv_acc / len(y_cv)

test_preds = lr.predict(test_sc)
test_acc = np.sum(test_preds == y_te)
test_acc = test_acc / len(y_te);print
print("LR")
('Train: ', train_acc, "\tCV: ", cv_acc, "\tTest: ", test_acc)
