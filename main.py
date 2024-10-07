import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the event data
event_df = pd.read_csv("/path/to/your/event.csv")

# Check and clean the column names
event_df.columns = event_df.columns.str.strip()  # Trim whitespace from column names
print("Columns in DataFrame:", event_df.columns.tolist())  # Print columns for debugging

# Add an 'id' column
event_df['id'] = range(1, len(event_df) + 1)

# Define a function to remove digits from a string
def remove_numbers(text):
    return re.sub(r'\d+', '', text).strip()

# Apply the function to the 'event_title' column
event_df['event_title'] = event_df['event_title'].apply(remove_numbers)

# Reorder the columns to move 'id' to the first position
columns = ['id'] + [col for col in event_df.columns if col != 'id']
event_df = event_df[columns]

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

# Example usage
result = recommend('Data Science Pakistan Summit')
print("Recommendations:", result)
