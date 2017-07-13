__copyright__ = 'Copyright 2017 Aaron Janse'
__license__ = 'Apache 2.0'

from sklearn.model_selection import train_test_split
import sklearn.metrics
import pandas as pd
from ._util import FrameworkManager

def train(model_name, params):
    # Add in features
    _, valid_amnt, test_amnt = FrameworkManager.train_valid_test_splits

    f_train_valid, _ = train_test_split(FrameworkManager.features, test_size=test_amnt, random_state=137)

    f_train, f_valid = train_test_split(f_train_valid, test_size=valid_amnt/(1-test_amnt), random_state=137)

    train_X = pd.concat([FrameworkManager.train['X'], f_train], axis=1)
    validation_X = pd.concat([FrameworkManager.validation['X'], f_valid], axis=1)

    train_data = {'X': train_X.copy(), 'y': FrameworkManager.train['y'].copy()}
    validation_data = {'X': validation_X.copy(), 'y': FrameworkManager.validation['y'].copy()}

    # Train model
    model = FrameworkManager.models[model_name]

    FrameworkManager.models[model_name]['model'] = model['train'](model['model'], params, train_data, validation_data)

def evaluate(model_name, all_classes=None):
    _, _, test_amnt = FrameworkManager.train_valid_test_splits

    # Add in features
    _, f_test = train_test_split(FrameworkManager.features, test_size=test_amnt, random_state=137)
    test_X = pd.concat([FrameworkManager.test['X'], f_test], axis=1)
    test_data = {'X': test_X.copy(), 'y': FrameworkManager.test['y'].copy()}

    # Make predictions
    model = FrameworkManager.models[model_name]
    predictions = model['predict'](model['model'], test_data['X'])

    if all_classes is None:
        labels_arg = {}
    else:
        labels_arg = {'labels': all_classes}

    # Calculate log_loss score
    return sklearn.metrics.log_loss(list(test_data['y']), predictions, **labels_arg)

# Used for applying preprocessing and adding features to data not in the training dataset (never-before-seen data)
def process(raw_data):
    X = FrameworkManager.preprocess_func(raw_data)

    features = pd.DataFrame(index=X.index.copy())
    for func in FrameworkManager.feature_funcs:
        feature_output = pd.DataFrame(func(X=X.copy()), index=features.index)
        features = features.join(feature_output)

    return pd.concat([X, features], axis=1) # join preprocessed X with its features

def predict(model_name, processed_X):
    model = FrameworkManager.models[model_name]
    predictions = model['predict'](model['model'], processed_X)

    return predictions
