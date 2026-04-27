# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
def get_data(path2file="./assets/census.csv"):
    
    #load data from csv and split into train/test datasets
    data_all = pd.read_csv(path2file)
    data_train, data_test = train_test_split(data_all, test_size=0.20)

    return data_train, data_test

# Train and save a model.
def get_model(data_train, train_new=True):
    if train_new:

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
            pickle.dump(lb, f)
            
        rf_model = train_model(X_train, y_train)

        with open("./assets/model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
    else:
        with open("./assets/model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        
    return rf_model

#evaluate model performance on slices of categorical feature.
def eval_model(data_test, feature, model=None):
    if model is None:
        with open("./assets/model.pkl", "rb") as f:
            rf_model = pickle.load(f)

    with open("./assets/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("./assets/lb.pkl", "rb") as f:
        lb = pickle.load(f)

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

    print(f"Model Performance on Data Slices: {feature}")
    print("="*30)
    for value in data_test[feature].unique():
        print(f"Feature Value: {value}")
        print("-"*30)
        data_slice = data_test.loc[data_test[feature] == value]
        print(len(data_slice))

        # Process the test data with the process_data function.
        X_test, y_test, encoder, lb = process_data(
            data_slice, 
            categorical_features=cat_features, 
            label="salary", 
            training=False,
            encoder=encoder,
            lb=lb
        )

        y_preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"fbeta: {fbeta}")
        print(" ")

    
    

if __name__ == "__main__":
    data_train, data_test = get_data()

    rf_model = get_model(data_train)

    eval_model(data_test, "education", rf_model)
