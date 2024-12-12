from Preprocessor import Preprocessor
from baseline import get_avg_embedding, preprocess_text
from Tweet_PreProcess import preprocess_tweet
import pandas as pd 
import numpy as np
import os 
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')
# Load GloVe model with Gensim's API
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings



def process_and_save_embeddings(df: pd.DataFrame, embeddings_model, vector_size: int, output_file: str) -> pd.DataFrame:
    # Apply preprocessing to each tweet
    df['Tweet'] = df['Tweet'].apply(preprocess_tweet)
    
    # Compute embeddings
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
    
    # Create a DataFrame for embeddings
    tweet_df = pd.DataFrame(tweet_vectors)
    
    # Attach embeddings to the original dataframe
    processed_df = pd.concat([df, tweet_df], axis=1)
    
    # Save the processed DataFrame to a CSV file
    processed_df.to_csv(output_file, index=False)
    print(f"Processed embeddings saved to: {output_file}")



def sp_ss_preprocess_and_save_embeddings(df: pd.DataFrame, embeddings_model, vector_size: int, output_file: str) -> pd.DataFrame:
    # Apply preprocessing to each tweet
    df['Tweet'] = df['Tweet'].apply(preprocess_tweet)
    
    # Compute sentiment polarity and subjectivity
    df['Polarity'] = df['Tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    df['Subjectivity'] = df['Tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
    
    # Compute embeddings
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])
    
    # Create a DataFrame for embeddings
    tweet_df = pd.DataFrame(tweet_vectors)
    
    # Attach embeddings and sentiment features to the original dataframe
    processed_df = pd.concat([df, tweet_df], axis=1)
    
    # Save the processed DataFrame to a CSV file
    processed_df.to_csv(output_file, index=False)
    print(f"Processed embeddings and sentiment features saved to: {output_file}")

    return processed_df

def sp_ss_preprocess_and_save_embeddings_with_preprocess_class(df: pd.DataFrame, 
                                                               embeddings_model, 
                                                               vector_size: int, 
                                                               output_file: str) -> pd.DataFrame:
   

    # Preprocess each tweet using the Preprocessor class
    preprocessor = Preprocessor()
    df['Tweet'] = df['Tweet'].apply(lambda tweet: " ".join(preprocessor.preprocess([(None, tweet)])[1]))

    # Compute sentiment polarity and subjectivity
    df['Polarity'] = df['Tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    df['Subjectivity'] = df['Tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)

    # Compute embeddings
    tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size) for tweet in df['Tweet']])

    # Create a DataFrame for embeddings
    tweet_df = pd.DataFrame(tweet_vectors)

    # Attach embeddings and sentiment features to the original dataframe
    processed_df = pd.concat([df, tweet_df], axis=1)

    # Save the processed DataFrame to a CSV file
    processed_df.to_csv(output_file, index=False)
    print(f"Processed embeddings and sentiment features saved to: {output_file}")

    return processed_df




train_li = []

train_path = "../challenge_data/train_tweets"
# Read all training files and concatenate them into one dataframe
for filename in os.listdir(train_path):
    file_path = os.path.join(train_path, filename).replace("\\", "/")
    print(file_path)
    df = pd.read_csv(file_path)
    train_li.append(df)

# Combine all files into a single dataframe
df = pd.concat(train_li, ignore_index=True)

# Split the dataframe based on 'PeriodID'
df1 = df[df['PeriodID'] <= 96].reset_index(drop=True)
df2 = df[df['PeriodID'] > 96].reset_index(drop=True)



vector_size = 200  # Adjust based on the chosen GloVe model

# Process and save df1
sp_ss_preprocess_and_save_embeddings(
    df1, 
    embeddings_model, 
    vector_size, 
    "Preprocessed_Data/SP_SS_df1_preprocessed_tweets_96.csv"
)

# Process and save df2
sp_ss_preprocess_and_save_embeddings(
    df2, 
    embeddings_model, 
    vector_size, 
    "Preprocessed_Data/Sp_SS_df2_preprocessed_tweets_more_96.csv"
)

test_li = []

train_path = "../challenge_data/eval_tweets"
# Read all training files and concatenate them into one dataframe
for filename in os.listdir(train_path):
    file_path = os.path.join(train_path, filename).replace("\\", "/")
    print(file_path)
    df = pd.read_csv(file_path)
    test_li.append(df)

# Combine all files into a single dataframe
df = pd.concat(test_li, ignore_index=True)

# Split the dataframe based on 'PeriodID'
df1 = df[df['PeriodID'] <= 96].reset_index(drop=True)
df2 = df[df['PeriodID'] > 96].reset_index(drop=True)



vector_size = 200  # Adjust based on the chosen GloVe model

# Process and save df1
sp_ss_preprocess_and_save_embeddings_with_preprocess_class(
    df1, 
    embeddings_model, 
    vector_size, 
    "Preprocessed_Data/Test_SP_SS_df1_preprocessed_tweets_96.csv"
)

# Process and save df2
sp_ss_preprocess_and_save_embeddings_with_preprocess_class(
    df2, 
    embeddings_model, 
    vector_size, 
    "Preprocessed_Data/Test_Sp_SS_df2_preprocessed_tweets_more_96.csv"
)