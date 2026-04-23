import pytest

import pandas as pd
from ml.data import process_data
#from ml.model import train_model
#from ml.model import inference

@pytest.fixture
def sample_data():
    return pd.read_csv("./census.csv", nrows=5000)

def test_process_data(sample_data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert len(X) == len(sample_data)