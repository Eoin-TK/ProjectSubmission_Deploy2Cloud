import pytest

import pandas as pd
#from ml.data import process_data
#from ml.model import train_model
#from ml.model import inference

@pytest.fixture
def sample_data():
    return pd.read_csv("./census.csv", nrows=5000)

def test_process_data(sample_data):
    assert len(sample_data) == 5000