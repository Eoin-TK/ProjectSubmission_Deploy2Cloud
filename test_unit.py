import pytest

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model
#from ml.model import inference

@pytest.fixture
def sample_data():
    return pd.read_csv("./census.csv", nrows=5000)

@pytest.fixture
def sample_model(sample_data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features,
        label="salary",
        training=True
    )

    rf_model = train_model(X, y)

    return rf_model

class TestProcessData:
    def test_one(self, sample_data):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"],
            label="salary",
            training=True
        )
        assert isinstance(X, np.ndarray)

    def test_two(self, sample_data):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"],
            label="salary",
            training=True
        )
        assert isinstance(y, np.ndarray)

    def test_three(self, sample_data):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"],
            label="salary",
            training=True
        )
        assert isinstance(encoder, OneHotEncoder)

    def test_four(self, sample_data):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"],
            label="salary",
            training=True
        )
        assert isinstance(lb, LabelBinarizer) 

    def test_five(self, sample_data):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"],
            label="salary",
            training=True
        )
        assert len(X) == len(sample_data)

    def test_six(self, sample_data):
        X, y, encoder, lb = process_data(
            sample_data, categorical_features=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"],
            label="salary",
            training=True
        )
        assert len(y) == len(X)


class TestTrainModel:
    def test_one(self, sample_model):
        assert isinstance(sample_model, RandomForestClassifier)