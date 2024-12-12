import os
import pandas as pd
import re
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch


def preprocess_tweets(tweets):
    tokenizer = TweetTokenizer()
    cleaned_tweets = []
    for tweet in tweets:
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"@\S+", "", tweet)
        tweet = re.sub(r"#", "", tweet)
        tweet = re.sub(r"RT ", "", tweet)
        tweet = re.sub(r"\n", " ", tweet)
        tweet = re.sub(r"\s+", " ", tweet)
        tokens = tokenizer.tokenize(tweet)
        cleaned_tweets.append(" ".join(tokens))
    return " ".join(cleaned_tweets)


def load_and_preprocess_data(data_dir):
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)

            grouped = df.groupby(["PeriodID"]).apply(
                lambda x: {
                    "text": preprocess_tweets(x["Tweet"].tolist()),
                    "label": x["EventType"].iloc[0],
                }
            )
            all_data.extend(grouped.tolist())

    processed_data = pd.DataFrame(all_data)
    processed_data["label"] = processed_data["label"].astype(int)
    return processed_data


class TwitterDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


data_dir = "challenge_data/train_tweets"
processed_data = load_and_preprocess_data(data_dir)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    processed_data["text"].tolist(),
    processed_data["label"].tolist(),
    test_size=0.2,
    random_state=42,
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = TwitterDataset(tokenizer, train_texts, train_labels)
test_dataset = TwitterDataset(tokenizer, test_texts, test_labels)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

raw_predictions = trainer.predict(test_dataset)
predicted_labels = raw_predictions.predictions.argmax(axis=1)

accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy on test set: {accuracy:.4f}")
