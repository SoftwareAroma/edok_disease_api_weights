import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pd.read_excel('diseases_symptoms.xlsx')

# Function to predict top 3 diseases based on user-inputted symptoms
def predict_top_diseases(user_symptoms):
    # Load the vectorizer from the pickle file
    with open('tfidf_vectorizer_without_model.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    # Load the weights from the pickle file
    with open('weights.pkl', 'rb') as weights_file:
        loaded_weights = pickle.load(weights_file)

    # Load the weighted TF-IDF matrix from the pickle file
    with open('weighted_tfidf.pkl', 'rb') as weighted_tfidf_file:
        loaded_weighted_tfidf = pickle.load(weighted_tfidf_file)

    # Transform user symptoms using the loaded vectorizer
    user_tfidf = loaded_vectorizer.transform([' '.join(user_symptoms)])
    
    # Apply weights to the user TF-IDF vector
    user_weighted_tfidf = user_tfidf.multiply(loaded_weights)

    # Calculate cosine similarity
    similarities = cosine_similarity(user_weighted_tfidf, loaded_weighted_tfidf).flatten()
    
    # Get the indices of the top 3 diseases
    top_indices = similarities.argsort()[-3:][::-1]
    top_diseases = df['diseases'].iloc[top_indices].values
    top_similarities = similarities[top_indices]
    
    results = [(top_diseases[i], top_similarities[i]) for i in range(len(top_diseases))]
    return results

# Example usage
user_symptoms = ['fever', 'weight loss', 'stomach ache']
top_diseases = predict_top_diseases(user_symptoms)

# Print the results
print("\nTop 3 Diseases based on input symptoms:")
for disease, similarity in top_diseases:
    print(f'{disease}: Similarity Score: {similarity:.4f}')