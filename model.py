import os
import gensim.downloader as api
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')
# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings


# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


# Basic preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


'''
df = pd.read_csv("Preprocessed_Data/preprocessed_tweets.csv")
df = df.sample(5000, random_state=42)
df.to_csv("Preprocessed_Data/preprocessed_tweets_sample.csv", index = False)


# Drop the columns that are not useful anymore
df = df.drop(columns=['Timestamp', 'Tweet'])
# Group the tweets into their corresponding periods. This way we generate an average embedding vector for each period
df = df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

# Save the sampled dataframe to a new CSV file
df.to_csv("Preprocessed_Data/vectors.csv", index=False)
'''

df = pd.read_csv("Preprocessed_Data/vectors.csv")

df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
# We extract the labels of our training samples
y = df['EventType'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gradient Boosting
cf_gb = GradientBoostingClassifier()
cf_gb.fit(X_train, y_train)

y_pred = cf_gb.predict(X_test)
print("GB Test set: ", accuracy_score(y_test, y_pred))

cf_gb_full = GradientBoostingClassifier()
cf_gb_full.fit(X, y)

# XGBoost
cf_xgb = XGBClassifier()
cf_xgb.fit(X_train, y_train)

y_pred = cf_xgb.predict(X_test)
print("XGB Test set: ", accuracy_score(y_test, y_pred))

cf_xgb_full = XGBClassifier()
cf_xgb_full.fit(X, y)

# Cat Boost
cf_cb = CatBoostClassifier()
cf_cb.fit(X_train, y_train, verbose=False)

y_pred = cf_cb.predict(X_test)
print("CB Test set: ", accuracy_score(y_test, y_pred))

cf_cb_full = CatBoostClassifier()
cf_cb_full.fit(X, y)


predictions = []
vector_size = 200  

for fname in os.listdir("challenge_data/eval_tweets"):
    print('started')
    val_df = pd.read_csv("challenge_data/eval_tweets/" + fname)
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)

    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    preds_gb = cf_gb_full.predict(X)
    
    preds_xgb = cf_xgb_full.predict(X)
    
    preds_cb = cf_cb_full.predict(X)
    
    # Average the predictions
    avg_preds = (preds_gb + preds_xgb + preds_cb) / 2
    final_preds = (avg_preds > 0.5).astype(int)
    
    period_features['EventType'] = final_preds

    predictions.append(period_features[['ID', 'EventType']])
    

pred_df = pd.concat(predictions)
pred_df.to_csv('Predictions/GBC_predictions.csv', index=False)
