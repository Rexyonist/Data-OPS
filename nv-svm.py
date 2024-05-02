import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from textblob import TextBlob

# Load the dataset
df = pd.read_csv('output/labelled_data.csv')

# Function to assign sentiment category
def assign_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Apply the function to your DataFrame
df['sentiment'] = df['label'].apply(assign_sentiment)

# Calculate the counts for each sentiment category
sentiment_counts = df['sentiment'].value_counts()

# Calculate the total number of entries
total_entries = len(df)
sentiment_percentages = (sentiment_counts / total_entries) * 100
# Print the percentage of each sentiment category
print("Persentase sentimen:")
for sentiment, percentage in sentiment_percentages.items():
    print(f"{sentiment.capitalize()}: {percentage:.2f}%")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['label'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vectors, y_train)

# Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectors, y_train)

def evaluate_model(model, X_test_vectors, y_test):
    predictions = model.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, predictions) * 100
    precision = precision_score(y_test, predictions, average='macro', zero_division=0) * 100  # Adjusted here
    recall = recall_score(y_test, predictions, average='macro', zero_division=0) * 100  # And here
    return accuracy, precision, recall

# Evaluate Naive Bayes
nb_accuracy, nb_precision, nb_recall = evaluate_model(nb_model, X_test_vectors, y_test)
print(f"Naive Bayes - Accuracy: {nb_accuracy:.2f}%, Precision: {nb_precision:.2f}%, Recall: {nb_recall:.2f}%")

# Evaluate SVM
svm_accuracy, svm_precision, svm_recall = evaluate_model(svm_model, X_test_vectors, y_test)
print(f"SVM - Accuracy: {svm_accuracy:.2f}%, Precision: {svm_precision:.2f}%, Recall: {svm_recall:.2f}%")