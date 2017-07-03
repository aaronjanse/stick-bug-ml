import pandas as pd
from singleton import Singleton

class FrameworkManager(Singleton):
    train = {'X': None, 'y': None}
    validation = {'X': None, 'y': None}
    test = {'X': None, 'y': None}
    features = pd.Dataset()
