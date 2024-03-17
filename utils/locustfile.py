import time
from locust import HttpUser, task, between


class LitServeUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def hello_world(self):
        self.client.post("/predict", json={"input": 5.0})

