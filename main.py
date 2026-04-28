from fastapi import FastAPI
from pydantic import BaseModel

class InputItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

app = FastAPI()

@app.get("/")
async def welcome():
    return {"greeting": "Hello!"}

@app.post("/inference/")
async def model_inference(input: InputItem):
    return input