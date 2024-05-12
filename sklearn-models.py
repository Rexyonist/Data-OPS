import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import joblib

# Load the dataset
df = pd.read_excel('output/excel-data.xlsx')
df.to_csv

def assign_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['cleared_translate'].apply(assign_sentiment)

# Split the data into features (X) and cleared_translates (y)
X = df['cleared_translate']
y = df['sentiment']  # Ensure this column exists and contains correct target cleared_translates
print("Setting up TfidfVectorizer and models...")

# Define the preprocessing and models pipelines
tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
nb_model = MultinomialNB()
svm_model = SVC()

# Create a pipeline
nb_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', nb_model)
])

svm_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', svm_model)
])
print("Pipelines set up.")

# Define the parameter grids for GridSearchCV
print("Setting up parameter grids for grid search...")
nb_param_grid = {
    'tfidf__max_features': [5000, 10000],
    'classifier__alpha': [0.1, 1.0, 10.0]
}

svm_param_grid = {
    'tfidf__max_features': [5000, 10000],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'poly', 'rbf']
}
print("Parameter grids set up.")

# Perform GridSearchCV for each model
cv = KFold(n_splits=5, shuffle=True, random_state=42)

nb_gs = GridSearchCV(nb_pipeline, param_grid=nb_param_grid, cv=cv, scoring='accuracy')
svm_gs = GridSearchCV(svm_pipeline, param_grid=svm_param_grid, cv=cv, scoring='accuracy')

# Fit the GridSearchCV models
nb_gs.fit(X, y)
svm_gs.fit(X, y)

print("Getting the best models...")
# Get the best model for each classifier
best_nb_model = nb_gs.best_estimator_
best_svm_model = svm_gs.best_estimator_

print("Best models obtained.")
# Evaluate the models on the testing set
def evaluate_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    return accuracy, precision, recall

# Evaluate best models
for model_name, model in [('Naive Bayes', best_nb_model), ('SVM', best_svm_model)]:
    accuracy, precision, recall = evaluate_model(model, X, y)
    print(f"{model_name} - Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")

print("Saving trained models and vectorizer...")
joblib.dump(best_nb_model, 'output/models/sklearn/naive_bayes_model.pkl')
joblib.dump(best_svm_model, 'output/models/sklearn/svm_model.pkl')
joblib.dump(tfidf, 'output/models/sklearn/tfidf_vectorizer.pkl')
print("Models and vectorizer saved.")
