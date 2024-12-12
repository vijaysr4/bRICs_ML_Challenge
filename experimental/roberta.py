import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')


def load_embeddings(model_name="glove-twitter-200"):
    print(f"Loading embeddings model: {model_name}")
    return api.load(model_name)


def preprocess_text(text):
    """
    对输入文本进行预处理，包括小写化、移除标点符号和数字、分词、移除停用词及词形还原。
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def get_avg_embedding(tweet, model, vector_size=200):
    """
    计算一条推文的平均词向量。
    如果推文中没有词在模型的词汇表中，返回零向量。
    """
    words = tweet.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


def load_and_merge_train_data(train_dir):
    """
    读取train_tweets目录下的所有CSV文件并合并成一个DataFrame。
    """
    data_frames = []
    csv_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f)) and f.endswith('.csv')]
    for filename in tqdm(csv_files, desc="Loading training files"):
        file_path = os.path.join(train_dir, filename)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    merged_df = pd.concat(data_frames, ignore_index=True)
    return merged_df


def compute_tweet_count_change_rate(df):
    """
    计算每个MatchID下按PeriodID排序的推特数量变化率。
    缺失值填充为0。
    """
    tweet_counts = df.groupby(['MatchID', 'PeriodID']).size().reset_index(name='tweet_count')
    tweet_counts = tweet_counts.sort_values(['MatchID', 'PeriodID'])
    tweet_counts['tweet_count_change_rate'] = tweet_counts.groupby(['MatchID'])[
        'tweet_count'].pct_change().fillna(0)
    return tweet_counts[['MatchID', 'PeriodID', 'tweet_count_change_rate']]


def get_embeddings(texts, tokenizer, model, device, batch_size=32):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def prepare_features(df, tokenizer, model, device, vector_size):
    """
    对DataFrame中的推文进行预处理、计算平均词向量，并计算推特数量变化率。
    返回包含特征的DataFrame。
    """

    df['Tweet'] = df['Tweet'].apply(preprocess_text)

    tweet_vectors = get_embeddings(df['Tweet'].tolist(), tokenizer, model, device)

    tweet_df = pd.DataFrame(tweet_vectors, columns=[f'vec_{i}' for i in range(vector_size)])

    features_df = pd.concat([df, tweet_df], axis=1)

    features_df.to_csv('features_df_roberta.csv', index=False)

    features_df = features_df.drop(columns=['Timestamp', 'Tweet'])

    features_df = features_df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    tweet_count_change = compute_tweet_count_change_rate(df)

    features_df = features_df.merge(tweet_count_change, on=['MatchID', 'PeriodID'], how='left')

    features_df['tweet_count_change_rate'] = features_df['tweet_count_change_rate'].fillna(0)

    return features_df


def split_data(features_df, target_column='EventType', test_size=0.2, random_state=24):
    """
    划分数据为训练集和验证集。
    """
    X = features_df.drop(columns=[target_column, 'MatchID', 'PeriodID', 'ID']).values
    y = features_df[target_column].values.astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def initialize_ml_models():
    """
    初始化逻辑回归和随机森林分类器，并创建集成的VotingClassifier。
    """
    logistic_clf = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )

    random_forest_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
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

    return ensemble_clf


class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_dnn_model(model, train_loader, val_loader, device, epochs=50, learning_rate=1e-3, patience=5):
    """
    训练深度神经网络模型，并使用EarlyStopping防止过拟合。
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}", leave=False):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_dnn_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

    model.load_state_dict(torch.load('best_dnn_model.pth'))
    return model


def train_ml_model(model, X_train, y_train):
    """
    训练集成的传统机器学习模型。
    """
    model.fit(X_train, y_train)
    return model


def evaluate_models(ml_model, dnn_model, X_val, y_val, device):
    """
    评估传统机器学习模型和深度神经网络模型，并结合它们的预测结果。
    """

    ml_probs = ml_model.predict_proba(X_val)[:, 1]

    dnn_model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        dnn_probs = dnn_model(X_val_tensor).cpu().numpy().flatten()

    combined_probs = ml_probs

    y_pred = (combined_probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    print(f'Combined Validation Accuracy: {accuracy:.4f}')
    return accuracy


def predict_evaluation_data(ml_model, dnn_model, eval_dir, tokenizer, model, device):
    """
    遍历评估数据集中的每个文件，进行预处理、特征计算，并使用模型进行预测。
    返回所有预测结果的DataFrame。
    """
    predictions = []
    eval_files = [f for f in os.listdir(eval_dir) if os.path.isfile(os.path.join(eval_dir, f)) and f.endswith('.csv')]

    for filename in tqdm(eval_files, desc="Predicting evaluation files"):
        file_path = os.path.join(eval_dir, filename)
        eval_df = pd.read_csv(file_path)
        features_eval = prepare_features(eval_df, tokenizer, model, device, model.config.hidden_size)

        X_eval = features_eval.drop(columns=['MatchID', 'PeriodID', 'ID']).values

        ml_probs = ml_model.predict_proba(X_eval)[:, 1]

        dnn_model.eval()
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
        with torch.no_grad():
            dnn_probs = dnn_model(X_eval_tensor).cpu().numpy().flatten()

        combined_probs = (ml_probs + dnn_probs) / 2

        y_pred = (combined_probs >= 0.5).astype(int)

        eval_df_result = pd.DataFrame({
            'ID': features_eval['ID'],
            'EventType': y_pred
        })
        predictions.append(eval_df_result)

    if predictions:
        return pd.concat(predictions, ignore_index=True)
    else:
        return pd.DataFrame(columns=['ID', 'EventType'])


def save_predictions(predictions, output_file='submission.csv'):
    """
    将预测结果保存为CSV文件。
    """
    predictions.to_csv(output_file, index=False)
    print(f'Predictions saved to {output_file}')


def main():
    train_dir = "challenge_data/train_tweets"
    eval_dir = "challenge_data/eval_tweets"
    output_file = 'submission.csv'

    device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    download_nltk_data()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model.eval()
    model.to(device)

    print("Loading and merging training data...")
    train_df = load_and_merge_train_data(train_dir)

    print("Preparing training features...")
    train_features = prepare_features(train_df, tokenizer, model, device, model.config.hidden_size)

    train_features.to_csv('train_features_bert.csv', index=False)

    train_features = pd.read_csv('train_features_bert.csv')

    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = split_data(train_features, target_column='EventType', test_size=0.2,
                                                random_state=24)

    print("Initializing traditional machine learning models...")
    ensemble_clf = initialize_ml_models()

    print("Training traditional machine learning models...")
    ensemble_clf = train_ml_model(ensemble_clf, X_train, y_train)

    train_dataset = TweetDataset(X_train, y_train)
    val_dataset = TweetDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Initializing deep neural network model...")
    dnn_model = DNNModel(input_dim=X_train.shape[1])

    print("Training deep neural network model...")
    dnn_model = train_dnn_model(dnn_model, train_loader, val_loader, device, epochs=50, learning_rate=1e-3, patience=5)

    print("Evaluating combined models...")
    evaluate_models(ensemble_clf, dnn_model, X_val, y_val, device)

    print("Predicting on evaluation data...")
    predictions = predict_evaluation_data(ensemble_clf, dnn_model, eval_dir, tokenizer, model, device)

    print("Saving predictions...")
    save_predictions(predictions, output_file)


if __name__ == "__main__":
    main()
