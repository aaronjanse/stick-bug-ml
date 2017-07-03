import pandas as pd
from sklearn.cross_validation import train_test_split
from singleton import Singleton

class FrameworkManager(Singleton):
    all_X = None
    train = {'X': None, 'y': None}
    validation = {'X': None, 'y': None}
    test = {'X': None, 'y': None}
    features = pd.Dataset()

# Decorators
def dataset(train_valid_test=(0.6, 0.2, 0.2)):
    train_amnt, valid_amnt, test_amnt = train_valid_test

    def dataset_decorator(func):
        # Get the dataset from the user-provided function
        X, y = func()

        FrameworkManager.all_X = X

        # Divide up the dataset
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=test_amnt, random_state=137)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_amnt, random_state=137)

        FrameworkManager.train['X'] = X_train
        FrameworkManager.train['X'] = y_train

        FrameworkManager.validation['X'] = X_valid
        FrameworkManager.validation['X'] = y_valid

        FrameworkManager.test['X'] = X_test
        FrameworkManager.test['X'] = y_test

    return dataset_decorator
