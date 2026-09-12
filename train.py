import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading dataset
df = pd.read_csv('problems.csv')

# Drop any problems where the scraper might have failed to get text
df = df.dropna(subset=['text'])

print(f"Loaded {len(df)} valid problems.")

# Initialize the Vectorizer
# We remove common English 'stop words' (the, and, is) and keep the top 1000 most meaningful words
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Transform the text into a numerical matrix (X)
X = vectorizer.fit_transform(df['text'])

# Isolate our target variable (y) - the rating we want to predict
y = df['rating']

print(f"Feature matrix shape: {X.shape}")