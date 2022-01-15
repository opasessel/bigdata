import pandas as pd
import numpy as np
import sklearn
from sklearn import ensemble
from sklearn.metrics import f1_score


class OnlineShopAnalysis:

    def __init__(self, algo_name, classifier, grid_params):
        self.algo_name = algo_name
        self.classifier = classifier
        self.grid_params = grid_params

        # read training and test data
        self.train = pd.read_csv('online_shoppers_intention_train.csv')
        self.test = pd.read_csv('online_shoppers_intention_test.csv')

        # select the last column as label (y)
        self.x_train = self.train.iloc[:, :-1].values
        self.y_train = self.train.iloc[:, -1].values
        self.x_test = self.test.iloc[:, :-1].values
        self.y_test = self.test.iloc[:, -1].values

    # Grid Search
    def grid_search(self):
        """ Perform hyperparameter analysis via Grid Search & save best settings to object classifier """

        # perform Grid Search
        print('')
        print('Starting Grid Search for', self.algo_name)
        search = sklearn.model_selection.GridSearchCV(self.classifier, self.grid_params, cv=10, scoring='f1_macro')
        search.fit(self.x_train, self.y_train)

        # override classifier with best estimator
        self.classifier = search.best_estimator_
        print('Classifier was overridden with best estimator')
        # set & print best params
        self.best_params = search.best_params_
        print('Best params:')
        print(self.best_params)

    def analyze_algo(self):
        """ Train algorithm, predict results for test data & analyze performance via cross validation """
        print('')
        print('Starting Analysis with set classifier:')

        # Training: Fit classifier to the training set
        self.classifier.fit(self.x_train, self.y_train)

        # predict test set results
        self.y_pred = self.classifier.predict(self.x_test)

        # calculate cross validation
        self.cross_val = np.mean(
            sklearn.model_selection.cross_val_score(self.classifier, self.x_train, self.y_train, cv=10, scoring='f1_macro'))

    def print_result(self):
        """ Print the performance results for the predicted test results """
        print('Results for', self.algo_name, ':')
        print('F1 Score:', f1_score(self.y_test, self.y_pred))
        print('Cross Validation:', self.cross_val)


# initiating Random Forest Classifier
R_classifier2 = ensemble.RandomForestClassifier()
# specifying parameters for Grid Search (defaults included)
param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [10, 20, None],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]}
# executing the Onlineshop Analysis
analysis1 = OnlineShopAnalysis('Random Forest', R_classifier2, param_grid)
# start Grid Search
analysis1.grid_search()
# analyze
analysis1.analyze_algo()
# print the results
analysis1.print_result()
