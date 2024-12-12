# bRICs Kaggle Challenge - CSC 51054 EP

Welcome to the bRICs Kaggle Challenge repository! This project contains multiple Python scripts designed to preprocess, analyze, and model data for the Kaggle challenge. Among these, the `best_model.py` script offers the highest accuracy ```0.74218```, and can be run independently with the necessary preprocessing in the same file.

---

## Repository Structure

### Files and Directories:

1. **`baseline.py`**: Implements a baseline model for comparison.
2. **`bert_embedding.py`**: Extracts embeddings using BERT for advanced feature representation.
3. **`best_model.py`**: The most accurate model implementation. Recommended for standalone execution.
4. **`dimensionality_reduction.py`**: Applies dimensionality reduction techniques to optimize feature sets.
5. **`lstm_model.py`**: Implements an LSTM-based neural network for sequence modeling.
6. **`match_time_filtration.py`**: Filters data based on time constraints for analysis.
7. **`model.py`**: Contains general ensemble models.
8. **`model_split.py`**: Model trained separately for ```PeriodID =< 96``` and ```PeriodID > 96```, assuming events are more likely to occur in the extra time.
9. **`shaply_valuespy.py`**: Calculates SHAP values for the random forest.
10. **`sliding_window_best_model.py`**: Implements a sliding window to account for tweet delays.
11. **`stop_words.txt`**: A list of stop words used for preprocessing.
12. **`test.py`**: Scripts for running tests and debugging.
13. **`tweet_preprocess.py`**: Preprocesses tweets for text analysis.
14. **`requirements.txt`**: Lists the Python dependencies required to run the scripts.

---

Prerequisites

Python: Ensure you have Python 3.8 or higher installed.

Libraries: Install required dependencies by running:

pip install -r requirements.txt

Dataset: Ensure you have the dataset prepared and accessible in the required format.

Folder Structure:

Create a folder challenge_data/train_tweets and place all training CSV files in it.

Create a folder challenge_data/eval_tweets and place all test CSV files in it.

Create a folder Preprocessed_Data to store all vector embeddings and cleaned CSV files.

Create a folder Predictions to save all submission CSV files generated after running the scripts.

How to Run the Scripts

1. Preprocessing Tweets

Run the preprocessing script to clean and prepare the dataset:

python tweet_preprocess.py

2. Exploratory Data Analysis

Use the dimensionality_reduction.py or match_time_filtration.py scripts for data analysis and feature preparation:

python dimensionality_reduction.py
python match_time_filtration.py

3. Baseline Model

To train and evaluate a baseline model, execute:

python baseline.py

4. Advanced Models

BERT Embedding:

python bert_embedding.py

LSTM Model:

python lstm_model.py

Sliding Window Approach:

python sliding_window_best_model.py

5. Best Model

The most accurate model can be run independently. Execute:

python best_model.py

This script automatically handles preprocessing, training, and evaluation, outputting the final results.

6. Model Testing

For testing the model on new data:

python test.py

7. SHAP Value Analysis

To interpret model predictions using SHAP values, run:

python shaply_valuespy.py

Notes

Ensure all input file paths are correctly specified within the scripts or passed as arguments.

The best_model.py script is designed to operate standalone but requires the dataset and dependencies to be correctly set up.
## Contribution
Vijay Venkatesh Murugan - M1 Data AI - IP Paris
Yufei Zhou - M1 Data AI - IP Paris
Stephan - M1 Data AI - IP Paris
---

## License

This repository is licensed under [MIT License](LICENSE).

---

For further assistance, contact the repository maintainer or raise an issue in this GitHub repository.

