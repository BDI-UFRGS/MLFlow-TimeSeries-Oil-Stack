from datetime import datetime, timedelta

import pandas as pd

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends

from app.service.src.forecast import forecast_darts

from decouple import config

ROOT_PATH = config("ROOT_PATH")

tags_metadata = [
    {
        "name": "forecast",
        "description": "Infer future points using forecasting models",
    },
]

description = """
API for forecasting petroleum production 🛢️📈

## Get Data

Available endpoints:

## Forecast

Forecast petroleum production.

Available endpoints:

* **forecast**:     Forecast using Temporal Fusion Transformer (TFT)
    * *origin_timestamp*:   Timestamp for when to start forecast (not-inclusive)
    * *forecast_length*:    Amount of future points to forecast
    * *frequency*:          At which frequency will the returned data be in (hour or daily)



Available models:

"""

app = FastAPI(  
    root_path=ROOT_PATH,
    title="MLFlow-TimeSeries-API",
    description=description,
    version="2.0.1",
    contact={
        "name": "PeTwin Project",
        "url": "https://www.petwin.org/contact/",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/license/mit/",
    },
    openapi_tags=tags_metadata
)

#app.include_router(auth.router)
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_MAX_AGE = int(timedelta(days=365).total_seconds())

@app.get("/health", tags=["system"])
def health():
    return {"is_healthy": True}


@app.get("/version", tags=["system"])
def health():
    return {"version": "0.0.1"}

@app.get(
    "/forecast",
    dependencies=[],
    tags=["forecast"],
)
async def forecast(
        origin_timestamp: datetime = datetime.fromisoformat('2018-10-01T22:00:00'),
        forecast_length: int = 6,
        frequency: str = '1h',
        choke: str = "100",
        target: str = "RATE_OIL_PROD"
    ):
    
    return Response(
        content=forecast_darts(origin_timestamp, forecast_length, frequency, choke, target),
        media_type="application/json",
        headers={"Cache-Control": f"public, max-age={CACHE_MAX_AGE}"}
    )
