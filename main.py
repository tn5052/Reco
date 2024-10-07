from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import re

app = FastAPI()

# Load and preprocess the event data
event_df = pd.read_csv("event.csv")
event_df['id'] = range(1, len(event_df) + 1)

# Function to remove numbers from strings
def remove_numbers(text):
    return re.sub(r'\d+', '', text).strip()

event_df['event_title'] = event_df['event_title'].apply(remove_numbers)

# Create tags by combining relevant columns
event_df['tags'] = event_df.apply(lambda row: f"{row['speaker']}, {row['category']}, {row['description']}, {row['venue']}", axis=1)
event_df['tags'] = event_df['tags'].apply(lambda x: x.lower())

# Apply stemming
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

event_df['tags'] = event_df['tags'].apply(stem)

# Create CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(event_df['tags']).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

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
        
        # Get top 20 similar events (excluding the input event)
        sim_scores = sim_scores[1:21]
        
        # Get the indices of the top similar events
        event_indices = [i[0] for i in sim_scores]
        
        # Get the titles of the top similar events
        recommendations = event_df['event_title'].iloc[event_indices].tolist()
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.get("/events")
def get_events():
    return {"events": event_df['event_title'].tolist()}
