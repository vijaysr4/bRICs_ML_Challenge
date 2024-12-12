import pandas as pd

def save_vectors(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    df = df.drop(columns=['Timestamp', 'Tweet'])
    df = df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    df.to_csv(output_path, index=False)
'''
    
df1_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/df1_preprocessed_tweets_96.csv"
df2_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/df2_preprocessed_tweets_more_96.csv"

df1_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/df1_vectors_96.csv"
df2_vec_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/df2_vectors_more_96.csv"

df1 = pd.read_csv(df1_path)
save_vectors(df1, df1_vec_path)

df2 = pd.read_csv(df2_path)
save_vectors(df2, df2_vec_path)
'''

df_path = "D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/SP_SS_df1_preprocessed_tweets_96.csv"

df = pd.read_csv(df_path)



print(df.columns(['Polarity'] ))