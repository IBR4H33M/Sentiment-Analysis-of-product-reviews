# Product Review Sentiment Analyzer

A machine learning system that classifies customer product reviews as **Positive** or **Negative** using an ensemble of four NLP models, deployed as a Flask REST API and integrated into a live React web application.

---

## Background

The initial phase of this project was a **collaborative work done as part of CSE437 – Data Science: Coding with Real World Data**, completed by a team of four. The group trained and evaluated multiple sentiment classification models on a dataset of 40,000 Amazon customer reviews inside a shared Jupyter notebook — [`project_code_initial.ipynb`](./project_code_initial.ipynb).

The models explored in the notebook were:
- Decision Tree
- Logistic Regression
- Support Vector Machine (SVM)
- AdaBoost

All subsequent work — including the prediction pipeline, Flask REST API, feature engineering improvements, model selection for deployment, and the web integration — was **independently implemented by me**.

---

## What Changed from Notebook → Deployment

| Aspect | Jupyter Notebook (Group Work) | Flask Deployment (Solo) |
|---|---|---|
| Models | Decision Tree, Logistic Regression, SVM, AdaBoost | Logistic Regression, SVM, Multinomial NB, Ridge Classifier |
| Removed | — | Decision Tree & AdaBoost (poor performance, high cost) |
| Added | — | Multinomial Naive Bayes, Ridge Classifier (better suited for TF-IDF) |
| Features | Basic TF-IDF | TF-IDF (unigrams + bigrams) + 5 hand-crafted lexical features |
| Prediction | Offline notebook | Live REST API with majority voting across all 4 models |
| Interface | None | React web page — users type any review and get instant results |

---

## Dataset

- **Source:** Amazon customer product reviews (FastText format) — [Kaggle: Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- **Size:** ~40,000 reviews
- **Format:** `__label__<1|2> <review text>` per line
- **Labels:** `1` = Negative, `2` = Positive
- **Note:** The raw dataset file (`test.ft.txt`) is **not included** in this repo due to its size. Download it from the link above and place it in this directory before running `train_and_save.py`.


---

## Feature Engineering

Each review is transformed into two feature groups, then horizontally stacked into a single sparse matrix:

**A. TF-IDF Features**
- Unigrams + bigrams (`ngram_range=(1, 2)`)
- Minimum document frequency: 2 (`min_df=2`)
- Maximum document frequency: 95% (`max_df=0.95`)
- English stop words removed

**B. Lexical Features** (5 hand-crafted)

| Feature | Description |
|---|---|
| `char_count` | Total character length of cleaned text |
| `word_count` | Total word count |
| `avg_word_length` | char_count / (word_count + 1) |
| `exclamation_count` | Number of `!` characters |
| `question_count` | Number of `?` characters |

---

## Models

| Model | Key Hyperparameters | Saved As |
|---|---|---|
| Logistic Regression | `max_iter=1000` | `model_lr.pkl` |
| LinearSVC (SVM) | `max_iter=2000` | `model_svm.pkl` |
| Multinomial Naive Bayes | defaults | `model_nb.pkl` |
| Ridge Classifier | defaults | `model_ridge.pkl` |

All models are evaluated using **accuracy score** on an 80/20 stratified validation split.

The `.pkl` files are loaded into memory at Flask server startup, allowing fast predictions without retraining on every request.

---

## Inference & Majority Voting

1. Input text is cleaned (lowercased, URLs/HTML stripped, special characters removed)
2. Features are extracted (TF-IDF + lexical) and stacked into a sparse matrix
3. All 4 models independently predict `1` (Negative) or `2` (Positive)
4. **Majority vote:** the label with >= 2 votes wins
5. **Confidence** = percentage of models that agreed with the majority

---

## API Endpoints

**Base URL:** `http://localhost:5002`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sentiment/predict` | Predict sentiment of a review |
| `GET` | `/sentiment/health` | Health check |

### Example Request

```json
POST /sentiment/predict
{
  "text": "This product is absolutely amazing, would buy again!"
}
```

### Example Response

```json
{
  "success": true,
  "text": "This product is absolutely amazing, would buy again!",
  "predictions": {
    "logistic_regression": "Positive",
    "svm": "Positive",
    "naive_bayes": "Positive",
    "ridge": "Positive"
  },
  "majority": "Positive",
  "confidence": 100.0,
  "timing": {
    "preprocessing_ms": 12.4,
    "model_ms": {
      "logistic_regression": 0.8,
      "svm": 0.5,
      "naive_bayes": 1.1,
      "ridge": 0.6
    },
    "total_ms": 15.4
  }
}
```

---

## Web Interface

This service is connected to a **React frontend** via an **Express.js proxy server**. Users can visit the Sentiment Analysis page on the website, type or paste any product review, and receive a live prediction showing:

- Each model's individual verdict (Positive / Negative)
- The overall majority sentiment
- Confidence percentage
- Per-model and total inference time in milliseconds
- A live log panel showing each step of the request

---

## Files in This Repository

```
Sentiment-Analysis-of-product-reviews/
|
|-- project_code_initial.ipynb  <- Original collaborative Jupyter notebook (group work, CSE437)
|-- train_and_save.py           <- Solo work: retraining pipeline for the Flask deployment
|-- app.py                      <- Solo work: Flask REST API server
|-- requirements.txt            <- Python dependencies
|-- .python-version             <- Python version pin
|-- README.md                   <- This file
|
|   -- Generated by train_and_save.py (NOT committed — too large for GitHub) --
|-- vectorizer.pkl       <- Fitted TF-IDF vectorizer   (~33 MB)
|-- model_lr.pkl         <- Logistic Regression         (~9.3 MB)
|-- model_svm.pkl        <- LinearSVC                   (~9.3 MB)
|-- model_nb.pkl         <- Multinomial Naive Bayes     (~37 MB)
|-- model_ridge.pkl      <- Ridge Classifier            (~9.3 MB)
```

**Commit to GitHub:** `app.py`, `train_and_save.py`, `requirements.txt`, `.python-version`, `README.md`

**Do NOT commit:** `.pkl` files and `test.ft.txt` — add them to `.gitignore`

---

## Recommended `.gitignore` Entries

```
*.pkl
test.ft.txt
__pycache__/
*.pyc
.env
```

---

## Local Setup

```bash
# 1. Clone the repo and enter this directory
cd ml-service-ProductReview

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place test.ft.txt (Amazon reviews dataset) in this directory

# 4. Train the models and generate .pkl files
python train_and_save.py

# 5. Start the Flask API server
python app.py
# Runs on http://localhost:5002
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML & NLP | scikit-learn 1.7.1, SciPy 1.14.1 |
| Data processing | pandas 2.2.3, NumPy 2.2.6 |
| API | Flask 3.0.0, flask-cors 4.0.0 |
| Production server | gunicorn 23.0.0 |
| Frontend | React (separate repo) |
| Proxy | Express.js (separate repo) |
