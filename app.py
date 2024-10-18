import pickle
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from mangum import Mangum
import pandas as pd
import json

app = FastAPI()
handler = Mangum(app)

# Load the data
df = pd.read_csv('files/diseases_symptoms.csv')

def predict_top_diseases(symptoms):
    try:
        with open('files/tfidf_vectorizer_without_model.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)

        with open('files/weights.pkl', 'rb') as weights_file:
            loaded_weights = pickle.load(weights_file)

        with open('files/weighted_tfidf.pkl', 'rb') as weighted_tfidf_file:
            loaded_weighted_tfidf = pickle.load(weighted_tfidf_file)
            
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model or vectorizer file not found.")
    except pickle.UnpicklingError:
        raise HTTPException(status_code=500, detail="Error loading the model or vectorizer.")

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
    # Validate input: Ensure symptoms are provided
    if not symptoms:
        raise HTTPException(status_code=400, detail="Please provide symptoms as a comma-separated string.")
    # Split the symptoms string into a list
    user_symptoms = [symptom.strip() for symptom in symptoms.split(',') if symptom.strip()]
    
    top_diseases = predict_top_diseases(user_symptoms)
    
    return JSONResponse({
        'top_diseases': top_diseases
    })
    
    
@app.get('/symptoms')
def get_symptoms():
    try:
        # Load the symptoms from the symptoms.json file
        with open('files/symptoms.json', 'r') as file:
            data = json.load(file)
        
        # Check if the 'symptoms' key exists
        if 'symptoms' not in data:
            raise HTTPException(status_code=500, detail="Symptoms data is incorrectly formatted.")
        
        # Return the unique symptoms list
        return JSONResponse({
            'symptoms': data['symptoms']
        })
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Symptoms file not found.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error reading the symptoms file.")
    
    
@app.get('/treatment')
def get_treatment(disease: str):
    # Load the treatment data from the JSON file
    try:
        with open('files/treatment.json', 'r') as file:
            treatments = json.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Treatment file not found.")
    
    # Find the treatment for the given disease
    for treatment in treatments:
        if treatment['Disease'].lower() == disease.lower():
            return JSONResponse({
                "Disease": treatment['Disease'],
                "Investigations": treatment['Investigations'],
                "Pharmacological Treatment": treatment['Pharmacological Treatment'],
                "Non-Pharmacological Treatment": treatment['Non-Pharmacological Treatment']
            })
    
    # If the disease is not found, return an error
    raise HTTPException(status_code=404, detail="Treatment for the specified disease not found.")
    
    
    
# Example usage
# user_symptoms = 'fever', 'weight loss', 'stomach ache'
# top_diseases = predict_top_diseases(user_symptoms)

# start the server
# uvicorn app:app --reload
