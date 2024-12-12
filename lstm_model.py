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
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier


from dimensionality_reduction import isomap_reduction, tsne_reduction
from tensorflow.keras.regularizers import l2

from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.utils import to_categorical

import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore',SparseEfficiencyWarning)

from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Define the LSTM classifier creation function
def create_lstm_classifier(input_dim, output_dim, max_sequence_length, embedding_dim=300, lstm_units=128, dropout_rate=0.2):
    """
    Create and configure an LSTM classifier for text data.

    Parameters:
    - input_dim (int): Size of the vocabulary (for Embedding layer).
    - output_dim (int): Number of output classes.
    - max_sequence_length (int): Maximum sequence length for input text.
    - embedding_dim (int): Dimensionality of the embedding vectors. Default is 300.
    - lstm_units (int): Number of LSTM units. Default is 128.
    - dropout_rate (float): Dropout rate for regularization. Default is 0.2.

    Returns:
    - model (tf.keras.Model): Compiled LSTM model.
    """
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=max_sequence_length),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(output_dim, activation='softmax', kernel_regularizer=l2(0.01))
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# Define hyperparameters for LSTM
vocabulary_size = 20000  # Number of unique words in the dataset
output_classes = 2       # Number of output classes
sequence_length = 100    # Maximum length of input sequences
embedding_dimensions = 200

# Create the LSTM model
lstm_clf = create_lstm_classifier(
    input_dim=vocabulary_size,
    output_dim=output_classes,
    max_sequence_length=sequence_length,
    embedding_dim=embedding_dimensions,
    lstm_units=256,
    dropout_rate=0.2
)

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, keras_model, epochs=12, batch_size=15, verbose=1):
        self.keras_model = keras_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        # Remove to_categorical here
        self.keras_model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        predictions = self.keras_model.predict(X)
        return predictions.argmax(axis=1)

    def predict_proba(self, X):
        return self.keras_model.predict(X)


# Wrap the LSTM model
lstm_clf_wrapped = KerasClassifierWrapper(
    keras_model=lstm_clf,
    epochs=10, 
    batch_size=32, 
    verbose=1
)


train_df = pd.read_csv("D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Train_fasttext.csv")

X = train_df.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
y = train_df['EventType'].values.astype(int)

#isomap_X = isomap_reduction(X, 120)
print(X.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)

logistic_clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

xgb_clf = XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=100,
    use_label_encoder=False,  # Disable use_label_encoder for newer XGBoost versions
    eval_metric='logloss',   # Set evaluation metric to log loss for classification
    random_state=42
)

random_forest_clf = RandomForestClassifier(
    max_depth=10,
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

lgb_clf = lgb.LGBMClassifier(
    max_depth=10,
    learning_rate=0.02,
    n_estimators=300,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat_clf = CatBoostClassifier(
    depth=10,
    learning_rate=0.05,
    iterations=300,
    random_seed=42,
    verbose=100
)

ensemble_clf = VotingClassifier(
    estimators=[
        ('random_forest', random_forest_clf),

        ('cat_boost', cat_clf)
    ],
    voting='soft'
)
ensemble_clf.fit(X_train, y_train)

y_val_pred = ensemble_clf.predict(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')



predictions = []

test_df = pd.read_csv("D:/M1_DataAI/P1/Intro_ML_DL/Kaggle_challenge/Preprocessed_Data/Test_fasttext.csv")
X_eval = test_df.drop(columns=['MatchID', 'PeriodID', 'ID']).values

#isomap_X_eval = isomap_reduction(X_eval, 120)

y_pred = ensemble_clf.predict(X_eval).astype(float)

test_df['EventType'] = y_pred
predictions.append(test_df[['ID', 'EventType']])

pred_df = pd.concat(predictions)

pred_df.to_csv('fasttext_RF_cat.csv', index=False)