# ML Framework

A framework to ease the burden of modularizing—and organizing—the code of a machine learning system.

With the help of pythonic decorators, the global namespace can be kept clutter-free.

## Simple Example

First, import this library:

```python
import ml_framework
from ml_framework import dataset, feature, model
```

Load your dataset:

```python
import seaborn.apionly as sns
import pandas as pd

@dataset(train_valid_test=(0.6, 0.2, 0.2)) # define your train/test/validation data splits
def my_dataset():
    titanic_dataset = sns.load_dataset('titanic')

    # Drop NaN rows for simplicity
    titanic_dataset.dropna(inplace=True)

    # Extract X and y
    X = titanic_dataset.drop('survived', axis=1)
    y = titanic_dataset['survived']
    return X, y

print(my_dataset.head()) # the function's name is now a var that holds the evaluated output `X`
```

(Optionally) do some pre-processing:

```python
@preprocess
def preprocess_data(X):
    # Encode categorical columns
    categorical_column_names = [
            'sex', 'embarked', 'class',
            'who', 'adult_male', 'deck',
            'embark_town', 'alive', 'alone']

    X = pd.get_dummies(X,
                       columns=categorical_column_names,
                       prefix=categorical_column_names)

    return X
```

Generate some features:

```python
from sklearn import decomposition
import numpy as np

@feature('pca')
def pca_feature(X):
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    pca_out = pca.transform(X)

    pca_out = np.transpose(pca_out, (1, 0))
    return {'pca_0': pca_out[0], 'pca_1': pca_out[1], 'pca_2': pca_out[2]}

# let's preview
pca_feature.head() # once again, the function's name becomes a variable holding its output

# you can add more features, btw
```

And define your (machine learning) model(s):

```python
import xgboost as xgb

@model('xgboost')
def xgboost_model():
    def define():
        return None # xgboost models aren't pre-defined


    def train(model, params, train, validation):
        params['objective'] = 'binary:logistic' # Static parameters can be defined here
        params['eval_metric'] = 'logloss'

        d_train = xgb.DMatrix(train['X'], label=train['y'])
        d_valid = xgb.DMatrix(validation['X'], label=validation['y'])

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        trained_model = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50, verbose_eval=10)

        return trained_model

    def predict(model, X):
        return model.predict(xgb.DMatrix(X))

    return define, train, predict
```

Now you can train your model, trying out different parameters if your want:

```python
ml_framework.train('xgboost', {
    'max_depth': 7,
    'eta': 0.01
})
```

The library keeps the test data's ground truth values locked away so your models won't train on it.
After you train your model, have the framework evaluate it for you:

```python
logloss_score = ml_framework.evaluate('xgboost')
print(logloss_score)
```

You can add lots more models and features if so desired.

Since this library is built with reality in mind, you can easily get predictions for new/real-life data:

```python
raw_X = pd.read_csv('2018_titanic_manifesto.csv') # It will probably sink, but we don't know who will survive
processed_X = ml_framework.process(raw_X) # Process the data
del raw_X

y = ml_framework.predict('xgboost', processed_X) # Make predictions

print(y)
```
