from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the event data
event_df = pd.read_csv("event.csv")
event_df['id'] = range(1, len(event_df) + 1)

# Create TF-IDF vectorizer on event titles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(event_df['event_title'])

# Compute similarity matrix
similarity = cosine_similarity(tfidf_matrix)

# Pydantic model for user input
class UserBookings(BaseModel):
    events: list[str]

@app.post("/recommend")
def recommend_events(user_bookings: UserBookings):
    event_indices = []
    
    # Get indices for each event the user has booked
    for event in user_bookings.events:
        try:
            event_index = event_df[event_df['event_title'].str.lower() == event.lower()].index[0]
            event_indices.append(event_index)
        except IndexError:
            raise HTTPException(status_code=404, detail=f"Event title '{event}' not found in the database")
    
    if not event_indices:
        raise HTTPException(status_code=400, detail="No valid events found for recommendation")
    
    try:
        # Fetch the similarity values based on the booked event indices
        similarity_values = similarity[event_indices]
        
        # Calculate mean similarity across the booked events
        mean_similarity = np.mean(similarity_values, axis=0)
        
        # Sort by similarity to find the most relevant events
        event_list = sorted(enumerate(mean_similarity), key=lambda x: x[1], reverse=True)
        
        # Exclude the booked events from recommendations
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
    return {"events": event_df['event_title'].tolist()}
