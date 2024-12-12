import bisect
import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('wordnet')

if not os.path.exists("glove-twitter-200.model"):
    embeddings_model = api.load("glove-twitter-200")
    embeddings_model.save("glove-twitter-200.model")
else:
    embeddings_model = KeyedVectors.load("glove-twitter-200.model")

vector_size = 200

stopwords_list = []
with open("stop_words.txt", 'r', encoding='utf-8') as f:
    for line in f:
        w = line.strip()
        if w:
            stopwords_list.append(w)
stopwords_list = sorted(stopwords_list)
stemmer = PorterStemmer()


def contains_url(tweet):
    return len(re.findall(r"http[s]?\S+", tweet)) != 0


def is_retweet(tweet):
    return len(re.findall(r"rt @?[a-zA-Z0-9_]+:? .*", tweet)) != 0


def contains_username(tweet):
    return '@' in tweet


def preprocess_tweet(tweet):
    tweet = tweet.lower().strip()

    if contains_url(tweet):
        return None

    if is_retweet(tweet):
        return None

    if contains_username(tweet):
        return None

    tweet = re.sub(r'\W+', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    if not tweet:
        return None

    words = []
    for word in tweet.split(" "):
        pos = bisect.bisect(stopwords_list, word)
        if pos > 0 and stopwords_list[pos - 1] == word:
            continue
        words.append(word)

    if not words:
        return None

    words = [stemmer.stem(w) for w in words]

    if not words:
        return None

    return ' '.join(words)


def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

# Create a sliding window median function
def compute_sliding_window_median(df):
    results = []
    grouped = df.groupby('MatchID')
    for match_id, group in tqdm(grouped, desc="Processing MatchIDs", total=len(grouped)):
        group = group.sort_values('PeriodID').reset_index(drop=True)
        numeric_columns = group.select_dtypes(include=np.number).columns
        new_rows = []
        for i in range(len(group)):
            if i < len(group) - 1:
                # Median of current and next period
                median_row = group.iloc[i:i+2][numeric_columns].median()
            else:
                # Median of the last period only
                median_row = group.iloc[i][numeric_columns]
            new_rows.append(median_row)
        results.append(pd.DataFrame(new_rows))
    return pd.concat(results, ignore_index=True)


def compute_sliding_window_median_with_id(df):
    results = []

    # Ensure the 'PeriodID' column is numeric for sorting and median computation
    df['PeriodID'] = pd.to_numeric(df['PeriodID'], errors='coerce')

    # Group by 'MatchID'
    grouped = df.groupby('MatchID')

    for match_id, group in tqdm(grouped, desc="Processing MatchIDs", total=len(grouped)):
        # Sort by 'PeriodID'
        group = group.sort_values('PeriodID').reset_index(drop=True)

        # Identify numeric columns for median calculation
        numeric_columns = group.select_dtypes(include=[np.number]).columns

        # Initialize a list to store the new rows
        new_rows = []

        for i in range(len(group)):
            if i < len(group) - 1:
                # Median of the current and next period
                median_row = group.iloc[i:i+2][numeric_columns].median()
            else:
                # Median of the last period only
                median_row = group.iloc[i][numeric_columns]

            # Preserve the same ID (MatchID_PeriodID) for the median row
            median_row["ID"] = group.iloc[i]["ID"]
            new_rows.append(median_row)

        # Append the resulting DataFrame for this group to the results
        results.append(pd.DataFrame(new_rows))

    # Concatenate all results into a single DataFrame
    return pd.concat(results, ignore_index=True)


li = []
for filename in os.listdir("challenge_data/train_tweets"):
    train_df = pd.read_csv(os.path.join("challenge_data/train_tweets", filename))
    li.append(train_df)
train_df = pd.concat(li, ignore_index=True)

train_df['Tweet'] = train_df['Tweet'].apply(preprocess_tweet)
train_df = train_df[train_df['Tweet'].notna()].copy()
train_df = train_df.reset_index(drop=True)

vector_size = 200
tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in train_df['Tweet']])
tweet_df = pd.DataFrame(tweet_vectors)

period_features = pd.concat([train_df, tweet_df], axis=1)

period_features = period_features.drop(columns=['Timestamp', 'Tweet'])

period_features = period_features = compute_sliding_window_median_with_id(period_features)

X = period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
y = period_features['EventType'].values.astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)

logistic_clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

random_forest_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

ensemble_clf = VotingClassifier(
    estimators=[
        ('logistic', logistic_clf),
        ('random_forest', random_forest_clf)
    ],
    voting='soft'
)

ensemble_clf.fit(X_train, y_train)


y_val_pred = ensemble_clf.predict(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')

predictions = []

for fname in os.listdir("challenge_data/eval_tweets"):
    val_df = pd.read_csv(os.path.join("challenge_data/eval_tweets", fname))
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_tweet)
    val_df = val_df[val_df['Tweet'].notna()].copy()
    val_df = val_df.reset_index(drop=True)

    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = compute_sliding_window_median_with_id(period_features)
    X_eval = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    y_pred = ensemble_clf.predict(X_eval).astype(float)

    period_features['EventType'] = y_pred
    predictions.append(period_features[['ID', 'EventType']])

pred_df = pd.concat(predictions)
pred_df.to_csv('slidingwindow_fasttext.csv', index=False)
