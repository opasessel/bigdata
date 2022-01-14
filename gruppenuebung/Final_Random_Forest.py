import pandas as pd
import numpy as np
import sklearn
from sklearn import ensemble
from sklearn.metrics import f1_score


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


# read training and test data
train = pd.read_csv('online_shoppers_intention_train.csv')
test = pd.read_csv('online_shoppers_intention_test.csv')


# Random Forest Grid Search 4.2
R_classifier2 = ensemble.RandomForestClassifier()
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [10, None], 'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]} # 'max_features': ['auto', 'int'],
grid_search('Random Forest Grid Search', R_classifier2, param_grid, train, test)
