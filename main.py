import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the event data and similarity matrix
try:
    event_df = pd.read_csv("event.csv")
    event_df['id'] = range(1, len(event_df) + 1)
    with open("event_data.pkl", "rb") as f:
        similarity = pickle.load(f)
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

class UserBookings(BaseModel):
    events: List[str]

@app.post("/recommend")
def recommend_events(user_bookings: UserBookings):
    event_indices = []
    for event in user_bookings.events:
        try:
            event_index = event_df[event_df['event_title'] == event].index[0]
            if event_index >= similarity.shape[0]:
                raise HTTPException(status_code=400, detail=f"Event index '{event_index}' is out of bounds for the similarity matrix")
            event_indices.append(event_index)
        except IndexError:
            raise HTTPException(status_code=404, detail=f"Event title '{event}' not found in the database")

    if not event_indices:
        raise HTTPException(status_code=400, detail="No valid events found for recommendation")

    logger.info(f"Event indices for booked events: {event_indices}")
    logger.info(f"Similarity matrix shape: {similarity.shape}")

    try:
        similarity_values = similarity[event_indices]
        logger.info(f"Similarity values shape: {similarity_values.shape}")
        mean_similarity = np.mean(similarity_values, axis=0)
    except Exception as e:
        logger.error(f"Failed to compute similarity: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        event_list = sorted(list(enumerate(mean_similarity)), reverse=True, key=lambda x: x[1])[1:21]
        recommendations = [event_df.iloc[i[0]].event_title for i in event_list]
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

@app.get("/events")
def get_events():
    return {"events": event_df['event_title'].tolist()}
