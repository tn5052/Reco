import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

# Load the event data and similarity matrix
event_df = pd.read_csv("event.csv")
event_df['id'] = range(1, len(event_df) + 1)

# Load the similarity matrix from the pickle file
with open("event_data.pkl", "rb") as f:
    similarity = pickle.load(f)

# Define the request model
class UserBookings(BaseModel):
    events: list[str]  # List of event titles the user has previously booked


@app.post("/recommend")
def recommend_events(user_bookings: UserBookings):
    event_indices = []
    
    # Collect indices for each event the user has booked
    for event in user_bookings.events:
        try:
            event_index = event_df[event_df['event_title'] == event].index[0]
            # Check if event index is within bounds of the similarity matrix
            if event_index >= similarity.shape[0]:
                return {"error": f"Event index '{event_index}' is out of bounds for the similarity matrix"}
            event_indices.append(event_index)
        except IndexError:
            return {"error": f"Event title '{event}' not found in the database"}

    # If no valid event indices are found, return an error
    if not event_indices:
        return {"error": "No valid events found for recommendation"}

    # Print debug information about event indices
    print(f"Event indices for booked events: {event_indices}")
    print(f"Similarity matrix shape: {similarity.shape}")

    # Calculate average similarity across the booked events
    try:
        # Check the individual similarity values to identify the issue
        similarity_values = [similarity[idx] for idx in event_indices]
        print(f"Similarity values: {similarity_values}")  # Debug print
        mean_similarity = np.mean(similarity_values, axis=0)
    except Exception as e:
        return {"error": f"Failed to compute similarity: {e}"}

    # Sort by similarity to find the most relevant events
    try:
        event_list = sorted(list(enumerate(mean_similarity)), reverse=True, key=lambda x: x[1])[1:21]
        recommendations = [event_df.iloc[i[0]].event_title for i in event_list]
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": f"Failed to generate recommendations: {e}"}

@app.get("/events")
def get_events():
    # Return all available events
    return {"events": event_df['event_title'].tolist()}
