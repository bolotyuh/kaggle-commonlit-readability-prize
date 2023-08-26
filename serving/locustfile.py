import numpy as np
from locust import HttpUser, between, task


txt_samples = [
    "I am certain to go wrong," " I said to myself",
    "Numerous books have been written about debugging, as it involves numerous aspects",
    "When the meal was over all returned to the parlor, where they spent the next hour in desultory chat.",
    "Everybody had a kindly greeting for the captain, and Violet's bright face grew still brighter as she made room for",
]


class PredictUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        num_of_sample = np.random.randint(1, 10)
        samples = np.random.choice(txt_samples, size=num_of_sample, replace=True)
        self.client.post("/predict", json={"text": samples.tolist()})
