import bisect
import os
import re
import gensim.downloader as api
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Load GloVe Twitter embeddings model
embeddings_model = api.load("glove-twitter-200")

vector_size = 200  # Dimension of the embeddings

# Load and sort stopwords from file
stopwords_list = []
with open("stop_words.txt", 'r', encoding='utf-8') as f:
    for line in f:
        w = line.strip()
        if w:
            stopwords_list.append(w)
stopwords_list = sorted(stopwords_list)

stemmer = PorterStemmer()  # Initialize Porter stemmer


def contains_url(tweet):
    """Check if the tweet contains a URL."""
    return len(re.findall(r"http[s]?\S+", tweet)) != 0


def is_retweet(tweet):
    """Check if the tweet is a retweet."""
    return len(re.findall(r"rt @?[a-zA-Z0-9_]+:? .*", tweet)) != 0


def contains_username(tweet):
    """Check if the tweet contains a username mention."""
    return '@' in tweet


def preprocess_tweet(tweet):
    """
    Preprocess the tweet by:
    - Lowercasing and stripping whitespace
    - Removing tweets with URLs, retweets, or usernames
    - Removing non-word characters
    - Removing stopwords
    - Stemming the words
    """
    tweet = tweet.lower().strip()

    if contains_url(tweet):
        return None

    if is_retweet(tweet):
        return None

    if contains_username(tweet):
        return None

    # Remove non-word characters and extra spaces
    tweet = re.sub(r'\W+', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    if not tweet:
        return None

    words = []
    for word in tweet.split(" "):
        # Use binary search to check if the word is a stopword
        pos = bisect.bisect(stopwords_list, word)
        if pos > 0 and stopwords_list[pos - 1] == word:
            continue
        words.append(word)

    if not words:
        return None

    # Apply stemming to each word
    words = [stemmer.stem(w) for w in words]

    if not words:
        return None

    return ' '.join(words)


def get_avg_embedding(tweet, model, vector_size=200):
    """
    Compute the average embedding for a tweet by averaging the embeddings of its words.
    If no words are in the model, return a zero vector.
    """
    words = tweet.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


# Load and concatenate all training CSV files
li = []
for filename in os.listdir("challenge_data/train_tweets"):
    train_df = pd.read_csv(os.path.join("challenge_data/train_tweets", filename))
    li.append(train_df)
train_df = pd.concat(li, ignore_index=True)

# Preprocess tweets and remove rows with invalid tweets
train_df['Tweet'] = train_df['Tweet'].apply(preprocess_tweet)
train_df = train_df[train_df['Tweet'].notna()].copy()
train_df = train_df.reset_index(drop=True)

# Compute average embeddings for each tweet
tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in train_df['Tweet']])
tweet_df = pd.DataFrame(tweet_vectors)

# Combine tweet embeddings with other features
period_features = pd.concat([train_df, tweet_df], axis=1)

# Drop irrelevant columns
period_features = period_features.drop(columns=['Timestamp', 'Tweet'])

# Aggregate features by MatchID, PeriodID, and ID using median
period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).median().reset_index()

# Prepare feature matrix X and target vector y
X = period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
y = period_features['EventType'].values.astype(int)

# Split data into training and validation sets with stratification
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)

# Initialize Logistic Regression classifier
logistic_clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

# Initialize Random Forest classifier
random_forest_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Create an ensemble Voting Classifier with soft voting
ensemble_clf = VotingClassifier(
    estimators=[
        ('logistic', logistic_clf),
        ('random_forest', random_forest_clf)
    ],
    voting='soft'
)

# Train the ensemble classifier on the training data
ensemble_clf.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = ensemble_clf.predict(X_val)

# Calculate and print validation accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')

predictions = []

# Process each evaluation CSV file
for fname in os.listdir("challenge_data/eval_tweets"):
    val_df = pd.read_csv(os.path.join("challenge_data/eval_tweets", fname))

    # Preprocess tweets and remove invalid ones
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_tweet)
    val_df = val_df[val_df['Tweet'].notna()].copy()
    val_df = val_df.reset_index(drop=True)

    # Compute average embeddings for evaluation tweets
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    # Combine embeddings with other features
    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])

    # Aggregate features by MatchID, PeriodID, and ID using median
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).median().reset_index()

    # Prepare feature matrix for evaluation
    X_eval = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    # Predict EventType for evaluation data
    y_pred = ensemble_clf.predict(X_eval).astype(float)

    # Assign predictions to the DataFrame
    period_features['EventType'] = y_pred
    predictions.append(period_features[['ID', 'EventType']])

# Concatenate all predictions and save to CSV
pred_df = pd.concat(predictions)
pred_df.to_csv('submission.csv', index=False)
