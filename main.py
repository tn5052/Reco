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

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(event_df['event_title'])

# Compute similarity matrix
similarity = cosine_similarity(tfidf_matrix)

class EventInput(BaseModel):
    title: str

@app.post("/recommend")
def recommend_events(event_input: EventInput):
    try:
        # Find the index of the input event
        event_index = event_df[event_df['event_title'] == event_input.title].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Event title '{event_input.title}' not found in the database")
    
    try:
        # Get similarity scores for the input event
        sim_scores = list(enumerate(similarity[event_index]))
        
        # Sort events by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 100 similar events (excluding the input event)
        sim_scores = sim_scores[1:101]
        
        # Get the indices of the top similar events
        event_indices = [i[0] for i in sim_scores]
        
        # Get the titles of the top similar events
        recommendations = event_df['event_title'].iloc[event_indices].tolist()
        
        # Remove any duplicate recommendations
        recommendations = list(dict.fromkeys(recommendations))
        
        # Limit to top 20 recommendations
        recommendations = recommendations[:20]
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.get("/events")
def get_events():
    return {"events": event_df['event_title'].tolist()}
