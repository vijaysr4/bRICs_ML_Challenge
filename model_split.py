import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import os

from baseline import get_avg_embedding, preprocess_text

embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings

def save_vectors(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    df = df.drop(columns=['Timestamp', 'Tweet'])
    df = df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    df.to_csv(output_path, index=False)



def save_vectors_with_retweet_count(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    # Add a 'Retweet' column to indicate if a tweet is a retweet (starts with 'RT')
    df['Retweet'] = df['Tweet'].str.startswith('RT').astype(int)
    
    # Calculate the retweet count per group
    retweet_counts = df.groupby(['MatchID', 'PeriodID', 'ID'])['Retweet'].sum().reset_index(name='retweet_count')
    
    # Group other features (excluding 'Tweet') by taking the mean
    grouped_features = df.drop(columns=['Tweet', 'Retweet']).groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    
    # Merge retweet counts back with the grouped features
    result_df = pd.merge(grouped_features, retweet_counts, on=['MatchID', 'PeriodID', 'ID'])
    
    # Save the result to the output file
    result_df.to_csv(output_path, index=False)
    print(f"Processed dataframe with retweet count saved to: {output_path}")

    return result_df



# Function to train models and return trained classifiers
def train_models(df: pd.DataFrame):
    
    X = df.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
    y = df['EventType'].values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Gradient Boosting
    cf_gb = GradientBoostingClassifier()
    cf_gb.fit(X_train, y_train)
    print("GB Test set accuracy: ", cf_gb.score(X_test, y_test))
    cf_gb_full = GradientBoostingClassifier()
    cf_gb_full.fit(X, y)

    # XGBoost
    cf_xgb = XGBClassifier()
    cf_xgb.fit(X_train, y_train)
    print("XGB Test set accuracy: ", cf_xgb.score(X_test, y_test))
    cf_xgb_full = XGBClassifier()
    cf_xgb_full.fit(X, y)

    # CatBoost
    cf_cb = CatBoostClassifier(verbose=False)
    cf_cb.fit(X_train, y_train)
    print("CB Test set accuracy: ", cf_cb.score(X_test, y_test))
    cf_cb_full = CatBoostClassifier(verbose=False)
    cf_cb_full.fit(X, y)

    return cf_gb_full, cf_xgb_full, cf_cb_full

# Function to predict and combine results
def evaluate_and_predict(eval_path: str, models, vector_size=200) -> list:
    print(f"Processing: {eval_path}")
    # Load the preprocessed vectors
    df = pd.read_csv(eval_path)

    # Extract features (X) and IDs
    X = df.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    # Predictions from each model
    preds_gb = models[0].predict(X)
    preds_xgb = models[1].predict(X)
    preds_cb = models[2].predict(X)

    # Average predictions
    avg_preds = (preds_gb + preds_xgb + preds_cb) / 3
    final_preds = (avg_preds > 0.5).astype(int)

    # Create a DataFrame with IDs and predictions
    df['EventType'] = final_preds
    return df[['ID', 'EventType']]




df1_SPSS_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/SP_SS_df1_preprocessed_tweets_96.csv"
df2_SPSS_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Sp_SS_df2_preprocessed_tweets_more_96.csv"

df1_SPSS_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Sp_SS__df1_vectors_96.csv"
df2_SPSS_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Sp_SS__df2_vectors_more_96.csv"


df1_SPSS= pd.read_csv(df1_SPSS_path)
save_vectors(df1_SPSS, df1_SPSS_vec_path)

df2_SPSS = pd.read_csv(df2_SPSS_path)
save_vectors(df2_SPSS, df2_SPSS_vec_path)







test_df1_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Test_SP_SS_df1_preprocessed_tweets_96.csv"
test_df2_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Test_Sp_SS_df2_preprocessed_tweets_more_96.csv"

test_df1_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Test_Sp_SS_df1_vectors_96.csv"
test_df2_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Test_Sp_SS_df2_vectors_more_96.csv"


test_df1 = pd.read_csv(test_df1_path)
save_vectors(test_df1, test_df1_vec_path)

test_df2 = pd.read_csv(test_df2_path)
save_vectors(test_df2, test_df2_vec_path)

#df1_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/df1_preprocessed_tweets_96.csv"
#df2_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/df2_preprocessed_tweets_more_96.csv"

#df1_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/df1_vectors_96.csv"
#df2_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/df2_vectors_more_96.csv"

#df1 = pd.read_csv(df1_path)
#save_vectors(df1, df1_vec_path)

#df2 = pd.read_csv(df2_path)
#save_vectors(df2, df2_vec_path)

df1 = pd.read_csv(df1_SPSS_vec_path)
df2 = pd.read_csv(df2_SPSS_vec_path)



# Train models for df1 and df2
print("Training models for df1...")
cf_gb1, cf_xgb1, cf_cb1 = train_models(df1)

print("Training models for df2...")
cf_gb2, cf_xgb2, cf_cb2 = train_models(df2)

# Paths for precomputed vector CSVs
test_df1_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Test_Sp_SS_df1_vectors_96.csv"
test_df2_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Test_Sp_SS_df2_vectors_more_96.csv"

# Predictions for df1 and df2
print("Evaluating df1...")
df1_predictions = evaluate_and_predict(test_df1_vec_path, (cf_gb1, cf_xgb1, cf_cb1))

print("Evaluating df2...")
df2_predictions = evaluate_and_predict(test_df2_vec_path, (cf_gb2, cf_xgb2, cf_cb2))

# Combine predictions and save to CSV
final_predictions = pd.concat([df1_predictions, df2_predictions]).sort_values('ID')
final_predictions.to_csv("D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Predictions/SS_SP_96_split_3model_average_predictions.csv", index=False)

print("Final predictions saved.")