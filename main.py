import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Initialize FastAPI
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and preprocess the event data
event_df = pd.read_csv("event.csv")
event_df['id'] = range(1, len(event_df) + 1)

# Function to remove numbers from strings
def remove_numbers(text):
    return re.sub(r'\d+', '', text).strip()

event_df['event_title'] = event_df['event_title'].apply(remove_numbers)

# Create tags by combining relevant columns
def create_tags(row):
    return f"{row['speaker']}, {row['category']}, {row['description']}, {row['venue']}".lower()

event_df['tags'] = event_df.apply(create_tags, axis=1)

# Apply stemming
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

event_df['tags'] = event_df['tags'].apply(stem)

# Create a CountVectorizer to convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(event_df['tags']).toarray()

# Compute the cosine similarity matrix
similarity = cosine_similarity(vectors)
logger.info("Cosine similarity matrix computed.")

# Pydantic model for input validation
class EventInput(BaseModel):
    title: str

# Helper function to find event index
def find_event_index(title):
    title = remove_numbers(title.lower())
    try:
        event_index = event_df[event_df['event_title'] == title].index[0]
        return event_index
    except IndexError:
        logger.error(f"Event title '{title}' not found in the database.")
        raise HTTPException(status_code=404, detail=f"Event title '{title}' not found in the database.")

# Endpoint to recommend events
@app.post("/recommend")
def recommend_events(event_input: EventInput):
    event_title = event_input.title
    logger.info(f"Received request to recommend events for: {event_title}")

    # Get the index of the input event
    event_index = find_event_index(event_title)
    
    try:
        # Get similarity scores for the input event
        sim_scores = list(enumerate(similarity[event_index]))
        
        # Sort events by similarity score, excluding the input event itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
        
        # Get the indices of the top similar events
        event_indices = [i[0] for i in sim_scores]
        
        # Get the titles of the top similar events
        recommendations = event_df['event_title'].iloc[event_indices].tolist()
        
        logger.info(f"Recommendations generated for '{event_title}': {recommendations}")
        return {"recommendations": recommendations}
    
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

# Endpoint to retrieve the list of available events
@app.get("/events")
def get_events():
    try:
        events = event_df['event_title'].tolist()
        return {"events": events}
    except Exception as e:
        logger.error(f"Failed to retrieve events: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve events: {str(e)}")
