from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the event data (make sure to adjust the path)
event_df = pd.read_csv("event.csv")

# Clean the column names
event_df.columns = event_df.columns.str.strip()

# Fill missing values
event_df.fillna('', inplace=True)

# Add an 'id' column
event_df['id'] = range(1, len(event_df) + 1)

# Define a function to remove digits from a string
def remove_numbers(text):
    return re.sub(r'\d+', '', text).strip()

# Apply the function to the 'event_title' column
event_df['event_title'] = event_df['event_title'].apply(remove_numbers)

# Check if required columns exist
required_columns = ['speaker', 'category', 'description', 'venue']
missing_columns = [col for col in required_columns if col not in event_df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

# Joining the specified columns
event_df['tags'] = event_df.apply(lambda row: f"{row['speaker']}, {row['category']}, {row['description']}, {row['venue']}", axis=1)

# Convert tags to lowercase
event_df['tags'] = event_df['tags'].apply(lambda x: x.lower())

# Stemming the tags
ps = PorterStemmer()

def stem(text):
    result = []
    for i in text.split():
        result.append(ps.stem(i))
    return " ".join(result)

event_df['tags'] = event_df['tags'].apply(stem)

# Convert words into array/vector
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(event_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Define the request model
class RecommendRequest(BaseModel):
    event_title: str

# Define the recommendation function
def recommend(event):
    try:
        event_index = event_df[event_df['event_title'] == event].index[0]
        distance = similarity[event_index]
        event_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:21]
        
        recommended_events = []
        for i in event_list:
            recommended_events.append(event_df.iloc[i[0]].event_title)
        
        return recommended_events
    except IndexError:
        return []  # Return an empty list if the event is not found

# Define the API endpoint for recommendations
@app.post("/recommend")
async def get_recommendations(request: RecommendRequest):
    recommendations = recommend(request.event_title)
    if not recommendations:
        raise HTTPException(status_code=404, detail="Event not found or no recommendations available.")
    return {"recommendations": recommendations}

# New Endpoint: Sentiment Analysis
class SentimentAnalysisRequest(BaseModel):
    reviews: List[str]

# Load the sentiment dataset
reviews = pd.read_csv("sentiment.csv")

# Define positive and negative words
positive_words = {
    "amazing", "exceptional", "fantastic", "loved", "great", "recommended", 
    "learned", "insightful", "attend", "speakers", "good", "excellent", 
    "enjoyed", "well", "informative", "interesting", "useful", "helpful", 
    "best", "awesome", "brilliant", "fabulous", "outstanding", "superb", 
    "wonderful", "perfect", "love", "like", "nice", "happy", "exciting", 
    "fun", "enjoy", "positive", "satisfied", "recommend", "inspiring", 
    "motivating", "insightful", "knowledgeable", "impressed", "beneficial", 
    "valuable", "productive", "effective", "efficient", "successful", 
    "satisfactory", "pleased", "glad", "comfortable", "confident", "relaxed"
}
negative_words = {"not", "bad", "average", "could", "better", "worth", "improve", "disappointed", "waste", "poor"}

# Function to calculate sentiment for a single review
def analyze_sentiment(review):
    review_words = review.lower().split()
    positive_score = sum(1 for word in review_words if word in positive_words)
    negative_score = sum(1 for word in review_words if word in negative_words)
    sentiment_score = positive_score - negative_score
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# API Endpoint for Sentiment Analysis
@app.post("/sentiment")
 reviews = request.reviews
    results = []

    # Example sentiment analysis (this is a placeholder for actual logic)
    for review in reviews:
        sentiment = "Positive" if "good" in review.lower() else "Negative"  # Simplified sentiment analysis
        results.append({
            "review": review,
            "sentiment": sentiment
        })
    
    return {"results": results}
