import pandas as pd
import numpy as np
import sklearn
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import neural_network
from sklearn.metrics import f1_score


# read training and test data
train = pd.read_csv('online_shoppers_intention_train.csv')
test = pd.read_csv('online_shoppers_intention_test.csv')


def analyze_algo(algo_name, classifier, train, test):
    """ Analyzes ML algorithm performance.
    Create classifier during function call! """

    # select the last column as label (y)
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Training: Fit classifier to the training set
    classifier.fit(X_train, y_train)

    # predict test set results
    y_pred = classifier.predict(X_test)

    # print the performance results for the predicted test results
    print('')
    print(algo_name)
    print('F1 Score:', f1_score(y_test, y_pred))
    cross_val = np.mean(sklearn.model_selection.cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1_macro'))
    print('Cross Validation:', cross_val)


# create classifier and call analyze_algo function:
# KNeighbors preset
K_MAX_NUM_NEIGHBORS = 15
K_WEIGHTS = 'uniform'
K_METRIC = 'euclidean'
K_classifier = neighbors.KNeighborsClassifier(n_neighbors=K_MAX_NUM_NEIGHBORS, weights=K_WEIGHTS, metric=K_METRIC)
analyze_algo('KNeighbors preset', K_classifier, train, test)

# KNeighbors custom
K_MAX_NUM_NEIGHBORS = 15
K_WEIGHTS = 'distance'  # 'distance' 'uniform'
K_METRIC = 'euclidean'
K_classifier2 = neighbors.KNeighborsClassifier(n_neighbors=K_MAX_NUM_NEIGHBORS, weights=K_WEIGHTS, metric=K_METRIC)
analyze_algo('KNeighbors custom', K_classifier2, train, test)

# DecisionTree
D_classifier = tree.DecisionTreeClassifier()
# analyze_algo('DecisionTree', D_classifier, train, test)

# Random Forest
R_classifier = ensemble.RandomForestClassifier()
analyze_algo('Random Forest', R_classifier, train, test)

# Support Vektor Machines
S_classifier = svm.SVC()
# analyze_algo('Support Vektor Machines', S_classifier, train, test)

# Neural Network
N_classifier = neural_network.MLPClassifier()
# analyze_algo('Neural Network', N_classifier, train, test)

# KNeighbors loop 4.1
K_WEIGHTS = 'distance'  # 'distance' 'uniform'
K_METRIC = 'euclidean'
for i in range(15):
    K_MAX_NUM_NEIGHBORS = i + 1
    K_classifier3 = neighbors.KNeighborsClassifier(n_neighbors=K_MAX_NUM_NEIGHBORS, weights=K_WEIGHTS, metric=K_METRIC)
    # analyze_algo('KNeighbors custom', K_classifier3, train, test)


# Grid Search
def grid_search(algo_name, classifier, param_grid, train, test):
    """ Perform parameter analysis via Grid Search & run analysis of algo with best parameters found """

    # select the last column as label (y)
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values

    # Perform Grid Search
    search = sklearn.model_selection.GridSearchCV(classifier, param_grid, cv=10, scoring='f1_macro')
    search.fit(X_train, y_train)

    # Run analysis of algo with best parameters found
    classifier_gs = search.best_estimator_
    analyze_algo(algo_name, classifier_gs, train, test)


# KNeighbors Grid Search 4.2
K_classifier4 = neighbors.KNeighborsClassifier()
param_grid = {'n_neighbors': [1, 15], 'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan', 'minkowski']}
grid_search('KNeighbors Grid Search', K_classifier4, param_grid, train, test)

# Random Forest Grid Search 4.2
R_classifier2 = ensemble.RandomForestClassifier()
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
param_grid = {'n_estimators': [], 'max_features': [], 'max_depth': [], 'min_samples_leaf': [], 'bootstrap': []}
grid_search('Random Forest Grid Search', R_classifier2, param_grid, train, test)
