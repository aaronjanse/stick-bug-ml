__copyright__ = 'Copyright 2017 Aaron Janse'
__license__ = 'Apache 2.0'

from sklearn.model_selection import train_test_split
from .singleton import Singleton

class FrameworkManager(Singleton):
    all_X = None
    all_y = None
    train = {'X': None, 'y': None}
    validation = {'X': None, 'y': None}
    test = {'X': None, 'y': None}
    features = None
    train_valid_test_splits = None
    models = {}
    preprocess_func = None
    feature_funcs = []

def _split_dataset():
    X = FrameworkManager.all_X
    y = FrameworkManager.all_y

    _, valid_amnt, test_amnt = FrameworkManager.train_valid_test_splits

    # Divide up the dataset
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=test_amnt, random_state=137)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=valid_amnt/(1-test_amnt), random_state=137)

    FrameworkManager.train['X'] = X_train
    FrameworkManager.train['y'] = y_train

    FrameworkManager.validation['X'] = X_valid
    FrameworkManager.validation['y'] = y_valid

    FrameworkManager.test['X'] = X_test
    FrameworkManager.test['y'] = y_test
