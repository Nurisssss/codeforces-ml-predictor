import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#we will use random forest for classifying

# 1. Load the small sample dataset
df = pd.read_csv('problems.csv')
df = df.dropna(subset=['text'])
print(f"Loaded {len(df)} valid problems.")

# 2. Vectorize the text (Translation step)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['text'])
y = df['rating']

# 3. Split data into training and testing sets
# We hold back 20% of the data to test the model on data it hasn't seen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training Random Forest model...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f} rating points")