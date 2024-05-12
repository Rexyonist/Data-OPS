import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('output/translated_cleaned.csv')
df.to_csv()
print("Dataset loaded.")

def assign_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['translated'].apply(assign_sentiment)

# Split the data into features (X) and labels (y)
X = df['translated']
sentiment_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
y = df['sentiment'].map(sentiment_mapping).astype(int)

# Create TfidfVectorizer and fit the data
print("Transforming data using TfidfVectorizer...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)
print("Data transformed.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define a Keras model
print("Building Keras model...")
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
print("Keras model built.")

# Compile the model
print("Compiling Keras model...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Keras model compiled.")

# Train the model
print("Training Keras model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
print("Keras model trained.")

# Evaluate the model
print("Evaluating Keras model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")

# Predict the labels for the test data
print("Predicting labels for the test data...")
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Calculate the confusion matrix
print("Calculating confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["positive", "neutral", "negative"])
cm_display.plot()
plt.title("Confusion Matrix - Multinomial Naive Bayes")
plt.savefig("output/Confusion_Matrix_Multinomial_Naive_Bayes.png")

# Interpret the confusion matrix
TP = conf_matrix[0, 0]  # True positives for class 0
FN = conf_matrix[0, 1:]  # False negatives for class 0
FP = conf_matrix[1:, 0]  # False positives for class 0
TN = conf_matrix[1:, 1:]  # True negatives for classes 1 and 2

print(f"True Positives: {TP}")
print(f"False Negatives: {FN}")
print(f"False Positives: {FP}")
print(f"True Negatives: {TN}")

# Save the model to an h5 file
print("Saving Keras model...")
model.save('output/models/tensorflow/keras_model.keras')

# Optionally, save the TfidfVectorizer for later use
print("Saving TfidfVectorizer...")
joblib.dump(tfidf, 'output/models/tensorflow/tfidf_vectorizer.pkl')