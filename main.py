from fastapi import FastAPI
from pydantic import BaseModel, Field

import pickle
import pandas as pd
from ml.data import process_data
from ml.model import inference

class InputItem(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example=13)
    marital_status: str = Field(alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example = 2174)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country", example="United-States")

    model_config = {
        "populate_by_name":True
    }

app = FastAPI()

@app.get("/")
async def welcome():
    return {"greeting": "Hello!"}

@app.post("/model_inference")
async def model_inference(input: InputItem):
    with open("./assets/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("./assets/lb.pkl", "rb") as f:
        lb = pickle.load(f)
    with open("./assets/model.pkl", "rb") as f:
        rf_model = pickle.load(f)

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
    
    input_df = pd.DataFrame([input.model_dump(by_alias=True)])

    X, y, encoder, lb = process_data(input_df, cat_features, training=False, encoder=encoder, lb=lb)

    y_pred = inference(model=rf_model, X=X)
    res = lb.inverse_transform(y_pred).ravel()
    
    return res.tolist()