from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FlightRequest(BaseModel):
    """Request schema for flight delay prediction.

    ARR_DELAY is NOT accepted from the client to prevent target leakage.
    It is created internally as NaN for feature engineering.
    """

    model_config = ConfigDict(extra="forbid")

    FL_DATE: date
    CRS_ELAPSED_TIME: float = Field(ge=0)
    CRS_DEP_TIME: float = Field(ge=0, le=2359)
    CRS_ARR_TIME: float = Field(ge=0, le=2359)
    OP_UNIQUE_CARRIER: str = Field(min_length=1)
    ORIGIN: str = Field(min_length=1)
    DEST: str = Field(min_length=1)
    DISTANCE: float = Field(ge=0)
    ARR_TIME: Optional[float] = Field(default=None, ge=0, le=2359)
    TAIL_NUM: str = Field(min_length=1)

    @field_validator("OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "TAIL_NUM")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


def request_to_dataframe(flight: FlightRequest) -> pd.DataFrame:
    """Convert validated request to DataFrame, ensuring ARR_DELAY exists as float64 NaN.

    This is the single point where request boundary ensures correct dtypes for feature engineering.
    ARR_DELAY is never accepted from the client; it is created internally as NaN.

    Args:
        flight: Validated FlightRequest from API client.

    Returns:
        DataFrame with ARR_DELAY column added as float64 NaN.
    """
    # Convert request to dictionary
    data = flight.model_dump()

    # Never use client-provided ARR_DELAY (prevent target leakage)
    if "ARR_DELAY" in data:
        del data["ARR_DELAY"]

    # Create DataFrame from client data
    df = pd.DataFrame([data])

    # The fitted pipeline also expects raw dataset fields that are not needed
    # from an API client. Derive stable values where possible and use the
    # training defaults for fields unavailable at prediction time.
    flight_date = pd.to_datetime(df["FL_DATE"])
    df["YEAR"] = flight_date.dt.year.astype("int64")
    df["QUARTER"] = flight_date.dt.quarter.astype("int64")
    df["MONTH"] = flight_date.dt.month.astype("int64")
    df["ORIGIN_AIRPORT_ID"] = 0
    df["DEST_AIRPORT_ID"] = 0
    df["CANCELLED"] = 0
    df["DIVERTED"] = 0

    # Ensure ARR_DELAY exists as float64 NaN for feature engineering
    # build_features() needs this column for historical delay calculations
    df["ARR_DELAY"] = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64",
    )

    return df


class PredictionResponse(BaseModel):
    prediction: float
    target: str
    model: str
    source: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
