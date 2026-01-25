# Sentiment-Analysis-of-product-reviews
Reproducible pipeline for binary sentiment classification on Amazon product reviews using TF–IDF + lexical features and baseline ML models (Logistic Regression, Linear SVM, Decision Tree, AdaBoost).
Key features 
Preprocessing: text cleaning, tokenization, lexical stats (char/word counts, punctuation)
Features: TF–IDF (unigrams + bigrams) + lexical features, stored as CSR arrays
Models: Decision Tree, Logistic Regression, LinearSVC, AdaBoost with optional GridSearchCV tuning flags (RUN_*_TUNING)
Reproducibility: stratified 80/20 splits saved under splits, feature artifacts under features
Notebook-first: project_code_full.ipynb contains full pipeline and prediction examples
Quick start 
Clone the repo
Create and activate Python environment, then install deps:
pip install -r requirements.txt
Open and run project_code_full.ipynb (set RUN_*_TUNING flags as needed)
Artifacts: splits/*.csv, features/*.npz, and features/*_vectorizer.pkl
Files & structure 
project_code_full.ipynb — full pipeline (preproc, feature eng., models)
test.ft.txt — raw FastText-format dataset ([Kaggle source](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews))

splits, features — generated artifacts for reproducibility
