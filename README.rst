stick-bug-ml
============

NOTE: see the Github page at https://github.com/aaronduino/stick-bug-ml

A framework to ease the burden of organizing code of a supervised
machine learning system.

It provides decorators that manage data & pass it between common steps
in building a machine learning system, such as: - loading the dataset -
preprocessing - feature generation - model definition

While doing this, it keeps the global namespace free of clutter such as
that from an endless chain of features and models.

In addition, it makes it easy to put new, real life, data through the
exact same process that training data goes through.

Installation
------------

Install simply via ``pip`` (Python 3):

.. code:: bash

    $ pip install stick-bug-ml

Dependencies: - Python 3 - sklearn - pandas - numpy

Example
-------

Note: there is also a great `example for use in Jupyter
Notebooks <demo.ipynb>`__

First, import this library:

.. code:: python

    import stickbugml
    from stickbugml.decorators import dataset, feature, model

Load your dataset:

.. code:: python

    import seaborn.apionly as sns
    import pandas as pd

    @dataset(train_valid_test=(0.6, 0.2, 0.2)) # define your train/test/validation data splits
    def raw_dataset():
        titanic_dataset = sns.load_dataset('titanic')

        # Drop NaN rows for simplicity
        titanic_dataset.dropna(inplace=True)

        # Extract X and y
        X = titanic_dataset.drop('survived', axis=1)
        y = titanic_dataset['survived']
        return X, y

    print(raw_dataset.head()) # yes, this does work! raw_dataset is now a pandas DataFrame

(Optionally) do some pre-processing:

.. code:: python

    @preprocess
    def preprocessed_dataset(X):
        # Encode categorical columns
        categorical_column_names = [
                'sex', 'embarked', 'class',
                'who', 'adult_male', 'deck',
                'embark_town', 'alive', 'alone']

        X = pd.get_dummies(X,
                           columns=categorical_column_names,
                           prefix=categorical_column_names)

        return X

    print(preprocessed_dataset.head()) # See the first code block for explaination

Generate some features:

.. code:: python

    from sklearn import decomposition
    import numpy as np

    @feature('pca')
    def pca_feature(X):
        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        pca_out = pca.transform(X)

        pca_out = np.transpose(pca_out, (1, 0))
        return pd.DataFrame(pca_out)

    # let's preview
    print(pca_feature.head()) # See the first code block for explaination

    # you can add more features, btw

And define your (machine learning) model(s):

.. code:: python

    import xgboost as xgb

    @model('xgboost')
    def xgboost_model():
        def define(num_columns):
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

Now you can train your model, trying out different parameters if you
want:

.. code:: python

    stickbugml.train('xgboost', {
        'max_depth': 7,
        'eta': 0.01
    })

The library keeps the test data's ground truth values locked away so
your models won't train on it. After you train your model, have the
framework evaluate it for you:

.. code:: python

    logloss_score = stickbugml.evaluate('xgboost')
    print(logloss_score)

You can add lots more models and features if so desired.

Since this library is built with reality in mind, you can easily get
predictions for new/real-life data:

.. code:: python

    raw_X = pd.read_csv('2018_titanic_manifesto.csv') # It will probably sink, but we don't know who will survive
    processed_X = stickbugml.process(raw_X) # Process the data
    del raw_X # Gotta keep that namespace clean, right?

    y = stickbugml.predict('xgboost', processed_X) # Make predictions

    print(y)

License
-------

This project uses the Apache 2.0 License
