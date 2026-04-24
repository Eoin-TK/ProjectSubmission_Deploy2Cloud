# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import train_model

# Add code to load in the data.
def get_data(path2file="./assets/census.csv"):
    
    #load data from csv and split into train/test datasets
    data_all = pd.read_csv(path2file)
    data_train, data_test = train_test_split(data_all, test_size=0.20)

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

    X_train, y_train, encoder, lb = process_data(
        data_train, 
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    with open("./assets/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open("./assets/lb.pkl", "wb") as f:
        pickle.dump(encoder, f)

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        data_test, 
        categorical_features=cat_features, 
        label="salary", 
        training=False,
        encoder=encoder,
        lb=lb
    )

    return X_train, y_train, X_test, y_test,

# Train and save a model.
def get_model(X_train, y_train, train_new=True):
    if train_new:
        rf_model = train_model(X_train, y_train)

        with open("./assets/model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
    else:
        with open("./assets/model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        
    
    return rf_model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()

    rf_model = get_model(X_train, y_train)
