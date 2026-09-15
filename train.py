import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.sparse import hstack # Needed to glue text matrices and standard numbers together

# 1. Load the dataset
df = pd.read_csv('problems.csv').dropna(subset=['text'])
print(f"Loaded {len(df)} valid problems.")

# 2. Feature Engineering: Word Count
# We split the text by spaces and count the resulting list size
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# 3. Vectorize the text (Translation step)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_text = vectorizer.fit_transform(df['text'])

# 4. Combine Features
# We stack the TF-IDF matrix horizontally with the new 2D word_count column
X_combined = hstack((X_text, df[['word_count']].values))
y = df['rating']

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 6. Train the Model
print("Training Random Forest model with Text + Word Count...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate the Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"New Mean Absolute Error: {mae:.2f} rating points")