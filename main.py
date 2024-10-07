import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

# Load the event data and similarity matrix
event_df = pd.read_csv("event.csv")
event_df['id'] = range(1, len(event_df) + 1)

# Load similarity matrix
with open("event_data.pkl", "rb") as f:
    similarity = pickle.load(f)

class UserBookings(BaseModel):
    events: list[str]

@app.post("/recommend")
def recommend_events(user_bookings: UserBookings):
    event_indices = []
    
    # Collect indices for each event the user has booked
    for event in user_bookings.events:
        try:
            # Find the event index
            event_index = event_df[event_df['event_title'] == event].index[0]
            # Ensure the index is within the bounds of the similarity matrix
            if event_index >= similarity.shape[0]:
                raise ValueError(f"Event index '{event_index}' is out of bounds for the similarity matrix")
            event_indices.append(event_index)
        except IndexError:
            raise HTTPException(status_code=404, detail=f"Event title '{event}' not found in the database")
    
    # If no valid event indices are found, return an error
    if not event_indices:
        raise HTTPException(status_code=400, detail="No valid events found for recommendation")

    # Print debug information about event indices and similarity matrix shape
    print(f"Event indices for booked events: {event_indices}")
    print(f"Similarity matrix shape: {similarity.shape}")

    # Calculate average similarity across the booked events
    try:
        # Extract the similarity values for the event indices
        similarity_values = similarity[event_indices]  # Assuming similarity is a 2D array
        print(f"Similarity values for indices {event_indices}: {similarity_values}")  # Debug print
        # Calculate mean similarity across the events booked
        mean_similarity = np.nanmean(similarity_values, axis=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute similarity: {e}")

    # Sort by similarity to find the most relevant events
    try:
        # Get sorted event indices based on mean similarity
        event_list = sorted(list(enumerate(mean_similarity)), reverse=True, key=lambda x: x[1])[1:21]
        recommendations = [event_df.iloc[i[0]].event_title for i in event_list]
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {e}")

@app.get("/events")
def get_events():
    return {"events": event_df['event_title'].tolist()}
