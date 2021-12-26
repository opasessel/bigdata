import pandas as pd
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import neural_network
from sklearn.metrics import f1_score


# read training and test data
train = pd.read_csv('online_shoppers_intention_train.csv')
test = pd.read_csv('online_shoppers_intention_test.csv')

# select the last column as label (y)
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# create classifier and fit to the training set
# KNeighbors preset
K_MAX_NUM_NEIGHBORS = 15
K_WEIGHTS = 'uniform'
K_METRIC = 'euclidean'
K_classifier = neighbors.KNeighborsClassifier(n_neighbors=K_MAX_NUM_NEIGHBORS, weights=K_WEIGHTS, metric=K_METRIC)
K_classifier.fit(X_train, y_train)
# KNeighbors custom
K_MAX_NUM_NEIGHBORS = 15
K_WEIGHTS = 'distance'  # 'distance' 'uniform'
K_METRIC = 'euclidean'
K_classifier2 = neighbors.KNeighborsClassifier(n_neighbors=K_MAX_NUM_NEIGHBORS, weights=K_WEIGHTS, metric=K_METRIC)
K_classifier2.fit(X_train, y_train)
# DecisionTree
D_classifier = tree.DecisionTreeClassifier()
D_classifier.fit(X_train, y_train)
# Random Forest
R_classifier = ensemble.RandomForestClassifier()
R_classifier.fit(X_train, y_train)
# Support Vektor Machines
S_classifier = svm.SVC()
S_classifier.fit(X_train, y_train)
# Neural Network
N_classifier = neural_network.MLPClassifier()
N_classifier.fit(X_train, y_train)


# predict test set results
# KNeighbors preset
K_y_pred = K_classifier.predict(X_test)
# KNeighbors custom
K2_y_pred = K_classifier2.predict(X_test)
# DecisionTree
D_y_pred = D_classifier.predict(X_test)
# Random Forest
R_y_pred = R_classifier.predict(X_test)
# Support Vektor Machines
S_y_pred = S_classifier.predict(X_test)
# Neural Network
N_y_pred = N_classifier.predict(X_test)

# print the F1 score for the predicted test results
print('F1 Scores:')
# KNeighbors default
print('KNeighbors preset:', f1_score(y_test, K_y_pred))
# KNeighbors custom
print('KNeighbors custom:', f1_score(y_test, K2_y_pred))
# DecisionTree
print('DecisionTree:', f1_score(y_test, D_y_pred))
# Random Forest
print('Random Forest:', f1_score(y_test, R_y_pred))
# Support Vektor Machines
print('Support Vektor Machines:', f1_score(y_test, S_y_pred))
# Neural Network
print('Neural Network:', f1_score(y_test, N_y_pred))

