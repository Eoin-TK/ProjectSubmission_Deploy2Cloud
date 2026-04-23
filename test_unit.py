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
    def __init__(self, sample_data, cat_features):
        self.f_in = sample_data
        self.f_out_X, self.f_out_y, self.f_out_encoder, self.f_out_lb = process_data(
            sample_data, categorical_features=cat_features,
            label="salary",
            training=True
        )

    def test_one(self):
        assert isinstance(self.f_out_X, pd.DataFrame)

    def test_two(self):
        assert isinstance(self.f_out_y, pd.DataFrame)

    def test_three(self):
        assert isinstance(self.f_out_encoder, OneHotEncoder)

    def test_four(self):
        assert isinstance(self.f_out_lb, LabelBinarizer) 

    def test_five(self):
        assert len(self.f_out_X) == len(self.f_in)

    def test_six(self):
        assert len(self.f_out_y) == len(self.f_out_X)
