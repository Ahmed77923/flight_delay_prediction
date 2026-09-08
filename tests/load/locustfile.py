
from locust import HttpUser, task, between


class FlightDelayUser(HttpUser):
    """
    Simulates a user interacting with the Flight Delay API.
    """

    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        self.client.get(
            "/health",
            name="/health"
        )

    @task(5)
    def predict(self):
        payload = {
            "FL_DATE": "2026-09-08T14:30:00",
            "CRS_DEP_TIME": 1430,
            "CRS_ARR_TIME": 1630,
            "CRS_ELAPSED_TIME": 120.0,
            "DISTANCE": 760.0,
            "OP_UNIQUE_CARRIER": "AA",
            "ORIGIN": "JFK",
            "DEST": "LAX"
        }

        self.client.post(
            "/predict",
            json=payload,
            name="/predict"
        )