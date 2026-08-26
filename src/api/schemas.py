from datetime import datetime

from pydantic import BaseModel, Field


class FlightRequest(BaseModel):
    FL_DATE: datetime

    CRS_DEP_TIME: int
    CRS_ARR_TIME: int

    CRS_ELAPSED_TIME: float
    DISTANCE: float

    OP_UNIQUE_CARRIER: str
    ORIGIN: str
    DEST: str

    TAIL_NUM: str = Field(default="UNKNOWN")


class PredictionResponse(BaseModel):
    prediction: float
    target: str
    model: str
    source: str