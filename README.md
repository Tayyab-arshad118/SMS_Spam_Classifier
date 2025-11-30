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
    └── app.py                       # Python application file
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
- Feature extraction (TF-IDF vectorization)
- Model training and evaluation
- Multiple classifier implementations

### `SMS_CLASS/app.py`
The Flask/Streamlit application file for:
- Running predictions on new SMS messages
- Providing a user interface for the spam classifier
- Demonstrating the trained model in action

## Key Features
- Text preprocessing (tokenization, stopword removal, etc.)
- Machine learning models for classification
- Performance metrics and model evaluation
- Ready-to-use prediction interface

## Requirements
- Python 3.x
- pandas
- scikit-learn
- numpy
- Jupyter Notebook (for .ipynb files)
- Flask or Streamlit (for app.py)

## Usage
1. Review the analysis in `code/SMS_SPAM_DETECTION.ipynb`
2. Train the model using the provided dataset
3. Run predictions using `SMS_CLASS/app.py`

## Notes
- The dataset contains real SMS messages with natural language variations
- Models are trained to handle common spam patterns and legitimate message characteristics
- The system can be deployed as a real-time SMS filtering solution

---

Feel free to modify this README as needed for your project!
