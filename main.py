from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the event data
event_df = pd.read_csv("event.csv")
event_df['id'] = range(1, len(event_df) + 1)

# Function to remove digits from strings
def remove_numbers(text):
    return re.sub(r'\d+', '', text).strip()

# Apply the function to 'event_title'
event_df['event_title'] = event_df['event_title'].apply(remove_numbers)

# Join specified columns to create tags
event_df['tags'] = event_df.apply(lambda row: f"{row['speaker']}, {row['category']}, {row['description']}, {row['venue']}", axis=1)

# Convert tags to lowercase
event_df['tags'] = event_df['tags'].apply(lambda x: x.lower())

# Stemming the tags
ps = PorterStemmer()
event_df['tags'] = event_df['tags'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))

# Create CountVectorizer and fit_transform the tags
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(event_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Pydantic model for user input
class UserBookings(BaseModel):
    events: list[str]

@app.post("/recommend")
def recommend_events(user_bookings: UserBookings):
    event_indices = []

    # Collect indices for each event the user has booked
    for event in user_bookings.events:
        try:
            # Use str.lower() for case-insensitive matching
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

        # Exclude already booked events and ensure unique recommendations
        recommendations = []
        for i, _ in event_list:
            if i not in event_indices and event_df.iloc[i]['event_title'] not in recommendations:
                recommendations.append(event_df.iloc[i]['event_title'])
            if len(recommendations) >= 20:  # Get top 20 recommendations
                break
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.get("/events")
def get_events():
    return {"events": event_df['event_title'].tolist()}
