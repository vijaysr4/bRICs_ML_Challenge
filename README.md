# bRICs Kaggle Challenge - CSC 51054 EP

Welcome to the bRICs Kaggle Challenge repository! This project contains Python scripts and data organized for
preprocessing, analysis, and modeling to tackle the Kaggle challenge. Among these, the `submission.py` script offers the
highest accuracy (```0.74218```) and can be run independently to produce replicable results.

---

## Repository Structure

This repository is structured as follows:

- **`challenge_data`**: Contains the dataset used for the challenge.
- **`analysis`**: Jupyter notebooks for exploratory data analysis (EDA) and visualization.
- **`preprocessing`**: Scripts used for cleaning and preparing the dataset for modeling.
- **`experimental`**: Code for various models and algorithms explored during the challenge.
- **`submission.py`**: The final script used to generate submission file for Kaggle, designed to run independently.

### Files and Directories:

```
challenge_data/
├── train_tweets/                # Folder containing training CSV files
├── eval_tweets/                 # Folder containing evaluation/test CSV files

analysis/
├── event_count.ipynb            # Jupyter notebook for analyzing EventType vs Tweet Count

preprocessing/
├── bert_embedding.py            # Script for generating BERT embeddings
├── dimensionality_reduction.py  # Feature optimization with dimensionality reduction
├── match_time_filtration.py     # Filters data based on time constraints
├── tweet_preprocess.py          # Preprocesses tweets for text analysis

experimental/
├── bert.py                      # Implements a BERT-based neural network
├── catboost_model.py            # Implements a CatBoost-based model
├── lstm_model.py                # Implements an LSTM-based neural network
├── roberta.py                   # Implements a RoBERTa-based neural network
├── sliding_window.py            # Sliding window approach for handling tweet delays
├── split_periods.py             # Model trained separately for PeriodID =< 96 and PeriodID > 96
                                 # assuming that events are more likely to occur in the extra time.

submission.py                    # Final script to generate submission files
submission.csv                   # Our final submission file
baseline.py                      # Baseline model for initial comparisons
requirements.txt                 # Python dependencies required
stop_words.txt                   # List of stop words for preprocessing
```

---

## Prerequisites

1. **Python**: Ensure you have Python 3.9 or higher installed.
2. **Libraries**: Install required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

3. **Dataset**: Ensure you have the dataset prepared and accessible in the required format.
    - Create a folder `challenge_data/train_tweets` and place all training CSV files in it.
    - Create a folder `challenge_data/eval_tweets` and place all test CSV files in it.

---

## How to Run the Scripts

### 1. **Baseline Model**

To train and evaluate a baseline model, execute:

   ```bash
   python baseline.py
   ```

### 2. **Best Model for Submission**

The most accurate model can be run independently. Execute:

   ```bash
   python submission.py
   ```

### 2. **Other Models or Scripts**

To run other models or scripts, execute the desired Python file:

   ```bash
   python <base_dir>/<script_name>.py
   ```

---

## Notes

- Ensure that all required dependencies are installed and the dataset is set up properly before running the scripts.
- The `submission.py` script is designed to operate standalone but requires the dataset and dependencies to be correctly
  set up.

---

## Contribution

- Vijay Venkatesh Murugan - M1 Data AI - IP Paris
- Yufei Zhou - M1 Data AI - IP Paris
- Stephan - M1 Data AI - IP Paris

---
