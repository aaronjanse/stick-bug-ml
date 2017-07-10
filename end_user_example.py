import ml_framework
from ml_framework import dataset, feature, model

# Load the dataset. Tell it how to divide up the data into train/valid/split
@dataset(train_valid_test=(0.6, 0.2, 0.2))
def get_dataset():
    data = pd.read_csv('my_dataset.csv')

    X = data.drop_columns('y')
    y = data['y']
    return X, y

@preprocess
def preprocess_data(X):
    # Do some preprocessing here
    return new_X

# Extract some features. The ground truth is kept locked away for this
@feature('word_share')
def get_word_share(X):
    return y

@feature('word_count')
def get_word_count(X):
    return y

# include some builtin features such as tsne and pca
ml_framework.include_builtin_features('tsne', 'pca')

# and start defining models
# The models are never given the test dataset in order to prevent hidden overfitting
@model('xgboost')
def xgboost_model():
    import xgboost as xgb

    def define(params):
        return model

    def train(model, params, train, valid):
        params['objective'] = 'binary:logloss'
        model.train(params, train.X, train.y, valid.X, valid.y)
        return model

    def predict(model, X):
        return model.predict(X)

    return define, train, predict

# and now that model can be trained
# This format is perfect for Jupyter Notebooks, since you can modify the parameters and try various sets of parameters
ml_framework.train('xgboost', {
    'depth': 7,
    'eta': 2.73
})

# and keep on defining models
@model('nn')
def nn_model():
    def define():
        pass

    def train(model, params, train, valid):
        model.train(params, train.X, train.y, valid.X, valid.y)
        return model

    def predict(model, X):
        return model.predict(X)

    return define, train, predict
