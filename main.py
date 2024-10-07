import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

# Load the event data and similarity matrix
event_df = pd.read_csv("event.csv")
event_df['id'] = range(1, len(event_df) + 1)

with open("event_data.pkl", "rb") as f:
    similarity = pickle.load(f)

class UserBookings(BaseModel):
    events: list[str]  # List of event titles the user has previously booked

@app.post("/recommend")
def recommend_events(user_bookings: UserBookings):
    event_indices = []
    
    # Collect indices for each event the user has booked
    for event in user_bookings.events:
        try:
            event_index = event_df[event_df['event_title'] == event].index[0]
            event_indices.append(event_index)
        except IndexError:
            # Handle case where event is not found
            return {"error": f"Event title '{event}' not found in database"}
    
    # Calculate average similarity across the booked events
    mean_similarity = np.mean([similarity[idx] for idx in event_indices], axis=0)
    
    # Sort by similarity to find the most relevant events
    event_list = sorted(list(enumerate(mean_similarity)), reverse=True, key=lambda x: x[1])[1:21]
    
    recommendations = [event_df.iloc[i[0]].event_title for i in event_list]
    return {"recommendations": recommendations}

@app.get("/events")
def get_events():
    return {"events": event_df['event_title'].tolist()}
