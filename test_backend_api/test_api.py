import requests
import json

headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
url = 'http://localhost:5000/'
payload = {
    "gender":1,
    "Partner":1,
    "Dependents":1,
    "TechSupport":1,
    "Contract": 0,
    "PaperlessBilling": 0,
    "SeniorCitizen": 1,
    "tenure": 9,
    "TotalCharges": 2
}
response = requests.post(url, data = json.dumps(payload), headers=headers).json()

print(response)