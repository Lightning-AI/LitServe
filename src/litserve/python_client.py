import requests

response = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0})
print(response.json())
