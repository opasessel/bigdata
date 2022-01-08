import pandas as pd
from sklearn import neighbors
from sklearn.metrics import f1_score

MAX_NUM_NEIGHBORS = 15
WEIGHTS = 'uniform'
METRIC = 'euclidean'

# read training and test data
train = pd.read_csv('online_shoppers_intention_train.csv')
test = pd.read_csv('online_shoppers_intention_test.csv')

# select the last column as label (y)
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# create classifier and fit to the training set
classifier = neighbors.KNeighborsClassifier(n_neighbors=MAX_NUM_NEIGHBORS, weights=WEIGHTS, metric=METRIC)
classifier.fit(X_train, y_train)

# predict test set results
y_pred = classifier.predict(X_test)

# print the F1 score for the predicted test results
print(f1_score(y_test, y_pred))
