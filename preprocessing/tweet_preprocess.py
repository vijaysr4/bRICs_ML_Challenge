import os
import re
import bisect
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import gensim.downloader as api
from textblob import TextBlob
import fasttext
import fasttext.util
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm



fasttext.util.download_model('en', if_exists='ignore')  # Download English model
fasttext_model = fasttext.load_model('cc.en.300.bin')  # Load the 300-dimensional FastText model

nltk.download('stopwords')
nltk.download('wordnet')

if not os.path.exists("glove-twitter-200.model"):
    embeddings_model = api.load("glove-twitter-200")
    embeddings_model.save("glove-twitter-200.model")
else:
    embeddings_model = KeyedVectors.load("glove-twitter-200.model")

vector_size = 200

stopwords_list = []
with open("../stop_words.txt", 'r', encoding='utf-8') as f:
    for line in f:
        w = line.strip()
        if w:
            stopwords_list.append(w)
stopwords_list = sorted(stopwords_list)
stemmer = PorterStemmer()

# Define a list of event-related keywords to count.
# You can refine this list based on domain knowledge.
keywords = [
    'goal', 'score', 'kick', 'penalti', 'card',
    'half', 'time', 'red', 'yellow', 'own', 'full'
]

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

def get_avg_embedding_GloVe(tweet: str, model, vector_size=200) -> list:
    words = tweet.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

def get_avg_embedding_fasttext(tweet, model, vector_size=300):
    """
    Calculate the average embedding of a tweet using a FastText embedding model.

    Parameters:
    - tweet (str): Input text (e.g., tweet) to compute the average embedding.
    - model: Pre-trained FastText embedding model.
    - vector_size (int): Dimensionality of the embedding vectors (default 300 for FastText).

    Returns:
    - np.ndarray: Average embedding vector of the input text.
    """
    words = tweet.split()
    word_vectors = []

    for word in words:
        try:
            word_vectors.append(model.get_vector(word))  # Use get_vector for FastText compatibility
        except AttributeError:
            word_vectors.append(model[word])  # Fallback for FastText models without get_vector
        except KeyError:
            pass  # Skip OOV words

    if not word_vectors:
        return np.zeros(vector_size)

    return np.mean(word_vectors, axis=0)


def get_avg_embedding_graphofwords(tweet, gow_model, vector_size=300):
    """
    Calculate the average embedding of a tweet using a Graph-of-Words embedding model.

    Parameters:
    - tweet (str): Input text (e.g., tweet) to compute the average embedding.
    - gow_model: Pre-trained Graph-of-Words embedding model.
    - vector_size (int): Dimensionality of the embedding vectors (default 300 for GoW embeddings).

    Returns:
    - np.ndarray: Average embedding vector of the input text.
    """
    words = tweet.split()
    word_vectors = []

    for word in words:
        try:
            # Assume gow_model provides a method `get_embedding(word)` for word embeddings
            word_vectors.append(gow_model.get_embedding(word))
        except AttributeError:
            raise ValueError("Graph-of-Words model must implement 'get_embedding(word)' method.")
        except KeyError:
            pass  # Skip OOV words

    if not word_vectors:
        return np.zeros(vector_size)

    return np.mean(word_vectors, axis=0)



class GraphOfWordsModel:
    def __init__(self, window_size=2, vector_size=300):
        self.window_size = window_size
        self.vector_size = vector_size

    def build_graph(self, tweet):
        """
        Build a co-occurrence graph for the given text.
        
        Parameters:
        - tweet (str): Input text (e.g., tweet).
        
        Returns:
        - graph (networkx.Graph): Co-occurrence graph of the text.
        """
        words = tweet.split()
        G = nx.Graph()

        # Add nodes and edges within the sliding window
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + self.window_size + 1, len(words))):
                G.add_edge(word, words[j], weight=1.0)
        return G

    def get_embedding(self, word):
        """
        Placeholder for word embedding generation.
        This method should return the vector representation of the word.
        """
        # For demonstration, generate a random vector. Replace with a learned embedding lookup.
        np.random.seed(hash(word) % 10000)  # Ensure deterministic results for reproducibility
        return np.random.rand(self.vector_size)



def process_tweets(data_dir, embedding_func, embeddings_model, vector_size=200, preprocess_func=None, drop_columns=None):
    """
    Processes tweet data to generate features based on specified embeddings and aggregates them.

    Parameters:
    - data_dir (str): Directory containing tweet CSV files.
    - embedding_func (callable): Function to generate embeddings for tweets.
    - embeddings_model: Pre-trained embeddings model (e.g., GloVe, FastText).
    - vector_size (int): Dimensionality of the embedding vectors.
    - preprocess_func (callable, optional): Function to preprocess tweets. Defaults to None.
    - drop_columns (list, optional): List of columns to drop before aggregation. Defaults to ['Timestamp', 'Tweet'].

    Returns:
    - pd.DataFrame: Aggregated features dataframe with embeddings.
    """
    li = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        li.append(df)

    combined_df = pd.concat(li, ignore_index=True)

    if preprocess_func:
        combined_df['Tweet'] = combined_df['Tweet'].apply(preprocess_func)

    combined_df = combined_df[combined_df['Tweet'].notna()].copy()
    combined_df = combined_df.reset_index(drop=True)

    tweet_vectors = np.vstack([
        embedding_func(tweet, embeddings_model, vector_size)
        for tweet in tqdm(combined_df['Tweet'], desc="Generating tweet embeddings")
    ])


    tweet_df = pd.DataFrame(tweet_vectors)
    features_df = pd.concat([combined_df, tweet_df], axis=1)

    if drop_columns is None:
        drop_columns = ['Timestamp', 'Tweet']

    features_df = features_df.drop(columns=drop_columns)

    aggregated_features = features_df.groupby(['MatchID', 'PeriodID', 'ID']).median().reset_index()

    return aggregated_features


from nltk import pos_tag
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

def process_tweets_add_features(data_dir: pd.DataFrame, embedding_func, embeddings_model, vector_size: int, preprocess_func=None, drop_columns=None):
    """
    Processes tweet data to generate features based on specified embeddings and aggregates them.

    Parameters:
    - data_dir (str): Directory containing tweet CSV files.
    - embedding_func (callable): Function to generate embeddings for tweets.
    - embeddings_model: Pre-trained embeddings model (e.g., GloVe, FastText).
    - vector_size (int): Dimensionality of the embedding vectors.
    - preprocess_func (callable, optional): Function to preprocess tweets. Defaults to None.
    - drop_columns (list, optional): List of columns to drop before aggregation. Defaults to ['Timestamp', 'Tweet'].

    Returns:
    - pd.DataFrame: Aggregated features dataframe with embeddings, sentiment, and advanced NLP features.
    """
    li = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        li.append(df)

    combined_df = pd.concat(li, ignore_index=True)

    if preprocess_func:
        combined_df['Tweet'] = combined_df['Tweet'].apply(preprocess_func)

    combined_df = combined_df[combined_df['Tweet'].notna()].copy()
    combined_df = combined_df.reset_index(drop=True)

    # Generate embeddings for each tweet
    tweet_vectors = np.vstack([
        embedding_func(tweet, embeddings_model, vector_size)
        for tweet in tqdm(combined_df['Tweet'], desc="Generating tweet embeddings")
    ])

    tweet_df = pd.DataFrame(tweet_vectors)

    # Generate sentiment features (polarity and subjectivity)
    combined_df['Polarity'] = combined_df['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
    combined_df['Subjectivity'] = combined_df['Tweet'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    # Generate Emotion Scores using a pre-trained model
    #emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    #combined_df['Emotion_Scores'] = combined_df['Tweet'].apply(lambda x: emotion_analyzer(x))

    # Perform Latent Dirichlet Allocation (LDA) for topic modeling
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    tweet_term_matrix = vectorizer.fit_transform(combined_df['Tweet'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_topics = lda.fit_transform(tweet_term_matrix)
    lda_df = pd.DataFrame(lda_topics, columns=[f"Topic_{i}" for i in range(1, 6)])

    # Generate Named Entity Recognition (NER) features
    ner_pipeline = pipeline("ner", grouped_entities=True)
    combined_df['NER'] = combined_df['Tweet'].apply(lambda x: ner_pipeline(x))

    # Generate Part-of-Speech (POS) tag features
    combined_df['POS_Tags'] = combined_df['Tweet'].apply(lambda x: pos_tag(x.split()))

    # Combine all features
    features_df = pd.concat([combined_df, tweet_df, lda_df], axis=1)

    if drop_columns is None:
        drop_columns = ['Timestamp', 'Tweet']

    features_df = features_df.drop(columns=drop_columns)

    # Aggregate features by MatchID, PeriodID, and ID
    aggregated_features = features_df.groupby(['MatchID', 'PeriodID', 'ID']).median().reset_index()

    return aggregated_features


'''
fasttext_train_features = process_tweets(
     data_dir="challenge_data/train_tweets",
     embedding_func=get_avg_embedding_fasttext,  # or get_avg_embedding_GloVe
     embeddings_model=fasttext_model,            # or glove_model
     vector_size=300,
     preprocess_func=preprocess_tweet)

fasttext_train_features.to_csv("Preprocessed_Data/Train_fasttext.csv", index=False)

fasttext_test_features = process_tweets(
     data_dir="D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/challenge_data/eval_tweets",
     embedding_func=get_avg_embedding_fasttext,  # or get_avg_embedding_GloVe
     embeddings_model=fasttext_model,            # or glove_model
     vector_size=300,
     preprocess_func=preprocess_tweet)

fasttext_test_features.to_csv("Preprocessed_Data/Test_fasttext.csv", index=False)
'''

# GloVe 200
glove_train_features = process_tweets_add_features(
     data_dir="../challenge_data/train_tweets",
     embedding_func=get_avg_embedding_GloVe,  # or get_avg_embedding_GloVe
     embeddings_model=embeddings_model,            # or glove_model
     vector_size=200,
     preprocess_func=preprocess_tweet)

glove_train_features.to_csv("Preprocessed_Data/Train_glove_200_extra_features.csv", index=False)

glove_test_features = process_tweets_add_features(
     data_dir="../challenge_data/eval_tweets",
     embedding_func=get_avg_embedding_GloVe,  # or get_avg_embedding_GloVe
     embeddings_model=embeddings_model,            # or glove_model
     vector_size=200,
     preprocess_func=preprocess_tweet)

glove_test_features.to_csv("Preprocessed_Data/Test_glove200_extra_features.csv", index=False)

'''
gow_model = GraphOfWordsModel(window_size=2, vector_size=200)

gow_train_features = process_tweets(
    data_dir="challenge_data/train_tweets",
    embedding_func=get_avg_embedding_graphofwords,
    embeddings_model=gow_model,  # Your initialized Graph-of-Words model
    vector_size=200,
    preprocess_func=preprocess_tweet
)

gow_train_features.to_csv("Preprocessed_Data/Train_gow_300.csv", index=False)

gow_test_features = process_tweets(
    data_dir="challenge_data/eval_tweets",
    embedding_func=get_avg_embedding_graphofwords,
    embeddings_model=gow_model,  # Your initialized Graph-of-Words model
    vector_size=200,
    preprocess_func=preprocess_tweet
)

gow_test_features.to_csv("Preprocessed_Data/Test_gow_300.csv", index=False)
'''