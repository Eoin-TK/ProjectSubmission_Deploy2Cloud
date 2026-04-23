import pytest

import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.data import process_data
#from ml.model import train_model
#from ml.model import inference

@pytest.fixture
def sample_data():
    return pd.read_csv("./census.csv", nrows=5000)

@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

class TestProcessData:
    def test_one(self):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=cat_features,
            label="salary",
            training=True
        )
        assert isinstance(X, pd.DataFrame)

    def test_two(self):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=cat_features,
            label="salary",
            training=True
        )
        assert isinstance(y, pd.DataFrame)

    def test_three(self):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=cat_features,
            label="salary",
            training=True
        )
        assert isinstance(encoder, OneHotEncoder)

    def test_four(self):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=cat_features,
            label="salary",
            training=True
        )
        assert isinstance(lb, LabelBinarizer) 

    def test_five(self):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=cat_features,
            label="salary",
            training=True
        )
        assert len(X) == len(sample_data)

    def test_six(self):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=cat_features,
            label="salary",
            training=True
        )
        assert len(y) == len(self.X)
