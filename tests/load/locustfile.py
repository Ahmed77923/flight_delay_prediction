from locust import HttpUser, task, between


class FlightDelayUser(HttpUser):
    """
    Simulates one user using the Flight Delay API.
    """

    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(2)
    def get_model(self):
        self.client.get("/model", name="/model")

    @task(5)
    def predict(self):
        payload = {
            "airline": "AA",
            "origin": "JFK",
            "destination": "LAX",
            "departure_hour": 14,
            "day_of_week": 2
        }

        self.client.post(
            "/predict",
            json=payload,
            name="/predict"
        )

