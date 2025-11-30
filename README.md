# SMS Spam Detection Project

## Overview
This project is an SMS (Short Message Service) spam classification system built using machine learning. It analyzes SMS messages and classifies them as either "spam" or "ham" (legitimate messages).

## Project Structure
```
SMS-CLASSIFIER/
├── code/
│   ├── SMS_SPAM_DETECTION.ipynb    # Main Jupyter notebook with analysis and model training
│   └── spam.csv                     # Dataset containing labeled SMS messages
└── SMS_CLASS/
    └── app.py
    └── model.pkl
    └── vectorizer.pkl                           
```

## Dataset
- **File**: `code/spam.csv`
- **Format**: CSV with columns for label (ham/spam) and message content
- **Size**: Contains thousands of SMS messages with labels
- **Labels**: 
  - `ham` = Legitimate message
  - `spam` = Spam message

## Files Description

### `code/SMS_SPAM_DETECTION.ipynb`
The main Jupyter notebook containing:
- Data loading and exploration
- Text preprocessing and cleaning
- Feature extraction (TF-IDF vectorization with max_features=3000)
- Model training and evaluation with multiple classifiers
- Voting Classifier implementation

### `SMS_CLASS/app.py`
The Flask/Streamlit application file for:
- Running predictions on new SMS messages
- Providing a user interface for the spam classifier
- Demonstrating the trained model in action

## Model Performance Metrics

### Models Tested
| Algorithm | Accuracy | Precision |
|-----------|----------|-----------|
| SVC | 97.29% | 97.41% |
| K-Neighbors (KN) | 91.20% | 100.00% |
| **Multinomial Naive Bayes (MNB)** | **97.00%** | **100.00%** |
| Decision Tree (DT) | 92.84% | 81.37% |
| Logistic Regression (LR) | 95.55% | 96.94% |
| Random Forest (RF) | 97.00% | 96.52% |

### Selected Model: Multinomial Naive Bayes (MNB)
**Reason for Selection**: Perfect precision (100%) with excellent accuracy (97.00%)

**Performance Metrics for MNB**:
- **Accuracy**: 97.00%
- **Precision**: 100.00%
- **Confusion Matrix**:
  - True Negatives (Ham correctly classified): 896
  - False Positives (Ham classified as Spam): 0
  - False Negatives (Spam classified as Ham): 31
  - True Positives (Spam correctly classified): 107


## Key Features
- Text preprocessing (tokenization, stopword removal, stemming)
- TF-IDF feature extraction with 3000 max features
- Multiple classifier implementations and comparison
- Voting ensemble for improved robustness
- Model serialization using pickle for deployment

## Requirements
- Python 3.x
- pandas
- scikit-learn
- numpy
- nltk
- xgboost
- Jupyter Notebook (for .ipynb files)
- Flask or Streamlit (for app.py)

## Usage
1. Review the analysis in `code/SMS_SPAM_DETECTION.ipynb`
2. Train the model using the provided dataset
3. Run predictions using `SMS_CLASS/app.py`

## Model Deployment
The trained MNB model and vectorizer are saved as:
- `model.pkl` - Trained Multinomial Naive Bayes classifier
- `vectorizer.pkl` - TF-IDF vectorizer for text transformation

## Notes
- The dataset contains real SMS messages with natural language variations
- Models are trained to handle common spam patterns and legitimate message characteristics
- The system can be deployed as a real-time SMS filtering solution
- Perfect precision means zero false positives, ensuring legitimate messages are never marked as spam
