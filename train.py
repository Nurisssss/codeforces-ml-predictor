import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from gensim.models import Word2Vec

# 1. Load the dataset
df = pd.read_csv('problems.csv').dropna(subset=['text'])
print(f"Loaded {len(df)} valid problems.")

# 2. Tokenize text (split sentences into lists of words)
tokenized_text = df['text'].apply(lambda x: str(x).lower().split()).tolist()

# 3. Train a Word2Vec model on our problem text
print("Training Word2Vec model...")
w2v_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=2, workers=4)

# 4. Function to convert a problem text into an averaged vector
def get_document_vector(tokens, model, vector_size=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# 5. Transform all problems into numerical feature matrices
X = np.array([get_document_vector(tokens, w2v_model) for tokens in tokenized_text])
y = df['rating']

# 6. Split data and train Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest with Word2Vec embeddings...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Word2Vec Mean Absolute Error: {mae:.2f} rating points")