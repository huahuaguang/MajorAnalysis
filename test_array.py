from joblib import load
import pandas as pd

# Load the model and vectorizer
kmeans = load('./models/kmeans_model.pkl')
tfidf = load('./models/tfidf_vectorizer.pkl')

# New text data
new_texts = [
    "How to fix Python memory leak in multiprocessing?",  # Computer-related
    "The best chocolate chip cookie recipe", # Non-computer-related
    "Linux kernel vs Windows scheduler performance", # Computer-related
    "2024 Olympic swimming predictions", # Non-computer-related
    "Debugging React useEffect infinite loop", # Computer-related
    "Staying here tonight, no train ticket available.", # Non-computer-related
    # Following two indicate the unaccurancy of this model
    "Spring is the most vibrant and energetic season of the year,"
    "and it is like a gentle painter, who uses delicate brushstrokes to drape the earth with brilliant colors", # Non-computer-related
    "Spring is beautiful" # Non-computer-related or Computer-related


]

# Vectorize the new texts
X_new = tfidf.transform(new_texts)

# Predict clusters
clusters = kmeans.predict(X_new)

# Output the results
for text, cluster in zip(new_texts, clusters):
    print(f"Text: '{text}'\nPredicted Category: {'Computer-related' if cluster == 1 else 'Non-computer-related'}\n")