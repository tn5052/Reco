import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# Load the event data and similarity matrix
event_df = pd.read_csv("event.csv")
event_df['id'] = range(1, len(event_df) + 1)

with open("event_data.pkl", "rb") as f:
    similarity = pickle.load(f)

class Event(BaseModel):
    title: str

@app.post("/recommend")
def recommend_events(event: Event):
    event_index = event_df[event_df['event_title'] == event.title].index[0]
    distances = similarity[event_index]
    event_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:21]
    
    recommendations = [event_df.iloc[i[0]].event_title for i in event_list]
    return {"recommendations": recommendations}

@app.get("/events")
def get_events():
    return {"events": event_df['event_title'].tolist()}
