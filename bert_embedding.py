import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk
import logging
from transformers.utils.logging import set_verbosity_info

# Download resources for preprocessing
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)

set_verbosity_info()


# Load BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Function to compute the BERT embeddings for a tweet
def get_bert_embedding(tweet, model, tokenizer, max_length=128):
    # Tokenize the tweet
    tokens = tokenizer(tweet, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    # Get embeddings from BERT
    with torch.no_grad():
        outputs = model(**tokens)
    # The embeddings are the mean of the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenization and stopword removal
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Load and preprocess the training data
li = []
train_dir = "challenge_data/train_tweets"
for filename in os.listdir(train_dir):
    df = pd.read_csv(os.path.join(train_dir, filename))
    li.append(df)
df = pd.concat(li, ignore_index=True)
df['Tweet'] = df['Tweet'].apply(preprocess_text)

# Compute BERT embeddings for each tweet
tweet_vectors = np.vstack([get_bert_embedding(tweet, bert_model, tokenizer) for tweet in df['Tweet']])
tweet_df = pd.DataFrame(tweet_vectors)

# Combine embeddings with the original dataframe
period_features = pd.concat([df, tweet_df], axis=1)
period_features.to_csv("Preprocessed_Data/BERT_preprocessed_tweets.csv", index=False)

# Group features and prepare for classification
period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
period_features.to_csv("Preprocessed_Data/BERT_vectors.csv", index=False)

'''
X = period_features.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
y = period_features['EventType'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set accuracy:", accuracy_score(y_test, y_pred))

# Train the logistic regression model on the full dataset
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)
dummy_clf = DummyClassifier(strategy="most_frequent").fit(X, y)

# Predictions for Kaggle submission
predictions = []
dummy_predictions = []
eval_dir = "challenge_data/eval_tweets"
for fname in os.listdir(eval_dir):
    print(f"Processing file: {fname}")
    val_df = pd.read_csv(os.path.join(eval_dir, fname))
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)
    
    tweet_vectors = np.vstack([get_bert_embedding(tweet, bert_model, tokenizer) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)
    
    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    preds = clf.predict(X)
    dummy_preds = dummy_clf.predict(X)

    period_features['EventType'] = preds
    period_features['DummyEventType'] = dummy_preds

    predictions.append(period_features[['ID', 'EventType']])
    dummy_predictions.append(period_features[['ID', 'DummyEventType']])

# Save predictions
pred_df = pd.concat(predictions)
pred_df.to_csv("Predictions/logistic_predictions.csv", index=False)

dummy_pred_df = pd.concat(dummy_predictions)
dummy_pred_df.to_csv("Predictions/dummy_predictions.csv", index=False)
'''