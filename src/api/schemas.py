from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FlightRequest(BaseModel):
    # Reject unknown fields so a client can never smuggle ARR_DELAY
    # (the prediction target) or any other feature into the request.
    model_config = ConfigDict(extra="forbid")

    FL_DATE: datetime
    CRS_DEP_TIME: int
    CRS_ARR_TIME: int
    CRS_ELAPSED_TIME: float
    DISTANCE: float
    OP_UNIQUE_CARRIER: str
    ORIGIN: str
    DEST: str


class PredictionResponse(BaseModel):
    prediction: float
    target: str
    model: str
    source: str