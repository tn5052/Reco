from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load and preprocess the event data
event_df = pd.read_csv("event.csv")  # Ensure event.csv has relevant columns like 'event_title', 'speaker', etc.
event_df['id'] = range(1, len(event_df) + 1)

# Combine relevant columns into a 'tags' column for text-based similarity
event_df['tags'] = event_df.apply(lambda row: f"{row['event_title']} {row['speaker']} {row['category']} {row['description']} {row['venue']}", axis=1)
event_df['tags'] = event_df['tags'].apply(lambda x: x.lower())  # Convert all text to lowercase for uniformity

# Create TF-IDF vectorizer for calculating similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(event_df['tags'])

# Compute cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix)

# Pydantic model to accept event bookings from users
class UserBookings(BaseModel):
    events: list[str]

@app.post("/recommend")
def recommend_events(user_bookings: UserBookings):
    event_indices = []
    
    # Collect indices for each event the user has booked
    for event in user_bookings.events:
        try:
            event_index = event_df[event_df['event_title'].str.lower() == event.lower()].index[0]
            event_indices.append(event_index)
        except IndexError:
            raise HTTPException(status_code=404, detail=f"Event title '{event}' not found in the database")
    
    # If no valid event indices are found, return an error
    if not event_indices:
        raise HTTPException(status_code=400, detail="No valid events found for recommendation")
    
    try:
        # Fetch the similarity values based on the indices of booked events
        similarity_values = similarity[event_indices]
        
        # Calculate mean similarity across the booked events
        mean_similarity = np.mean(similarity_values, axis=0)
        
        # Sort by similarity to find the most relevant events
        event_list = sorted(enumerate(mean_similarity), key=lambda x: x[1], reverse=True)
        
        # Filter out the events that the user has already booked
        recommendations = [
            event_df.iloc[i]['event_title'] 
            for i, _ in event_list 
            if i not in event_indices
        ][:20]  # Get top 20 recommendations
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.get("/events")
def get_events():
    # Return all available event titles
    return {"events": event_df['event_title'].tolist()}
