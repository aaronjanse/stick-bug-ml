import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics
from singleton import Singleton

class FrameworkManager(Singleton):
    all_X = None
    train = {'X': None, 'y': None}
    validation = {'X': None, 'y': None}
    test = {'X': None, 'y': None}
    features = None
    train_valid_test_splits = None
    models = {}

# Decorators
def dataset(train_valid_test=(0.6, 0.2, 0.2)):
    train_amnt, valid_amnt, test_amnt = train_valid_test

    assert train_amnt + valid_amnt + test_amnt == 1, "the train_valid_test splits should all add up to 1.0"

    FrameworkManager.train_valid_test_splits = train_valid_test

    def dataset_decorator(func):
        # Get the dataset from the user-provided function
        X, y = func()

        FrameworkManager.all_X = X

        FrameworkManager.features = pd.DataFrame(index=X.index.copy())

        # Divide up the dataset
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=test_amnt, random_state=137)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=valid_amnt/(1-test_amnt), random_state=137)

        FrameworkManager.train['X'] = X_train
        FrameworkManager.train['y'] = y_train

        FrameworkManager.validation['X'] = X_valid
        FrameworkManager.validation['y'] = y_valid

        FrameworkManager.test['X'] = X_test
        FrameworkManager.test['y'] = y_test

    return dataset_decorator

def feature(name):
    def feature_decorator(func):
        # The function is explicitly called with the keyword argument for end-user consistancy (note: is this a good thing? yes? no?)
        feature_output = pd.DataFrame(func(X=FrameworkManager.all_X), index=FrameworkManager.features.index)

        FrameworkManager.features = FrameworkManager.features.join(feature_output)

    return feature_decorator

def model(name):
    def model_decorator(func):
        define_func, train_func, predict_func = func()

        FrameworkManager.models[name] = {}
        FrameworkManager.models[name]['define'] = define_func
        FrameworkManager.models[name]['train'] = train_func
        FrameworkManager.models[name]['predict'] = predict_func

        FrameworkManager.models['model'] = define_func()

    return model_decorator

def train(model_name, params):
    # Add in features
    _, valid_amnt, test_amnt = FrameworkManager.train_valid_test_splits

    f_train_valid, _ = train_test_split(FrameworkManager.features, test_size=test_amnt, random_state=137)

    f_train, f_valid = train_test_split(f_train_valid, test_size=valid_amnt/(1-test_amnt), random_state=137)

    train_X = pd.concat([FrameworkManager.train['X'], f_train], axis=1)
    validation_X = pd.concat([FrameworkManager.validation['X'], f_valid], axis=1)

    train_data = {'X': train_X, 'y': FrameworkManager.train['y']}
    validation_data = {'X': validation_X, 'y': FrameworkManager.validation['y']}

    # Train model
    model = FrameworkManager.models[model_name]

    FrameworkManager.models[model_name]['model'] = model['train'](model, params, train_data, validation_data)

def evaluate(model_name):
    _, _, test_amnt = FrameworkManager.train_valid_test_splits

    # Add in features
    _, f_test = train_test_split(FrameworkManager.features, test_size=test_amnt, random_state=137)
    test_X = pd.concat([FrameworkManager.test['X'], f_test], axis=1)
    test_data = {'X': test_X, 'y': FrameworkManager.test['y']}

    # Make prediction
    model = FrameworkManager.models[model_name]
    predictions = model['predict'](model['model'], test_data['X'])

    # Calculate log_loss score
    return sklearn.metrics.log_loss(list(test_data['y']), predictions)
