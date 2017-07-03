import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from singleton import Singleton

class FrameworkManager(Singleton):
    all_X = None
    train = {'X': None, 'y': None}
    validation = {'X': None, 'y': None}
    test = {'X': None, 'y': None}
    features = pd.Dataset()
    models = {}

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

def feature(name):
    def feature_decorator(func):
        # The function is explicitly called with the keyword argument for end-user consistancy (note: is this a good thing? yes? no?)
        feature_output = func(X=X)

        # A (hopefully) informative error message
        assert isinstance(feature_output, np.array), "The output of the feature `{}` should be of type numpy.array, not {}. If it is a pandas DataFrame that has only one column (as it should), it can be converted into a numpy array via `my_dataframe.values`".format(name, type(feature_output))

        FrameworkManager.features[name] = feature_output

    return feature_decorator

def model(name):
    def model_decorator(func):
        define_func, train_func, predict_func = func()

        FrameworkManager.models[name] = {}
        FrameworkManager.models[name]['define_func'] = define_func
        FrameworkManager.models[name]['train_func'] = train_func
        FrameworkManager.models[name]['predict_func'] = predict_func

    return model_decorator
