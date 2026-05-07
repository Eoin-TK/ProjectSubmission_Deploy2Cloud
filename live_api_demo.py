import requests

if __name__ == "__main__":
    # Example POST request
    payload = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }

    try:
        response = requests.post("https://projectsubmission-deploy2cloud.onrender.com/model_inference", json=payload)
        response.raise_for_status()
        print("POST Status Code:", response.status_code)
        print("POST Response:", response.json())
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)