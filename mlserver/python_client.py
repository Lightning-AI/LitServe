import requests

response = requests.post("http://127.0.0.1:8888/predict/", json={"input": 5.0})
print(response.json())