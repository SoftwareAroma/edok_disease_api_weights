import pickle
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from mangum import Mangum
import pandas as pd

app = FastAPI()
handler = Mangum(app)

# Load the data
df = pd.read_csv('files/diseases_symptoms.csv')

def predict_top_diseases(symptoms):
    with open('files/tfidf_vectorizer_without_model.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    with open('files/weights.pkl', 'rb') as weights_file:
        loaded_weights = pickle.load(weights_file)

    with open('files/weighted_tfidf.pkl', 'rb') as weighted_tfidf_file:
        loaded_weighted_tfidf = pickle.load(weighted_tfidf_file)

    user_tfidf = loaded_vectorizer.transform([' '.join(symptoms)])
    user_weighted_tfidf = user_tfidf.multiply(loaded_weights)
    similarities = cosine_similarity(user_weighted_tfidf, loaded_weighted_tfidf).flatten()
    top_indices = similarities.argsort()[-10:][::-1]
    top_diseases = df['diseases'].iloc[top_indices].values
    top_similarities = similarities[top_indices]
    top_similarities = [round(similarity * 100, 2) for similarity in top_similarities]
    results = [(top_diseases[i], top_similarities[i]) for i in range(len(top_diseases))]
    return results

@app.get('/')
def read_root():
    return JSONResponse({
        'message': 'Welcome to the Disease Prediction API'
    })
    
@app.get('/predict_diseases')
def predict_diseases(symptoms: str):
    user_symptoms = symptoms.split(',')
    top_diseases = predict_top_diseases(user_symptoms)
    
    return JSONResponse({
        'top_diseases': top_diseases
    })
    
# Example usage
# user_symptoms = 'fever', 'weight loss', 'stomach ache'
# top_diseases = predict_top_diseases(user_symptoms)

# start the server
# uvicorn app:app --reload
