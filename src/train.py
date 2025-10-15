# src/train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

if __name__ == "__main__":

    # Charger les données d'entraînement
    train_df = pd.read_csv(os.path.join('data', 'train.csv'))
    X_train = train_df['text'].astype(str)  # S'assurer que tout est en string
    y_train = train_df['sentiment']

    # --- Expérience 1 : Régression Logistique ---
    print("Entraînement du modèle de Régression Logistique...")

    # Création d'un pipeline qui combine le vectoriseur TF-IDF et le classifieur
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Entraînement du pipeline
    lr_pipeline.fit(X_train, y_train)

    # Sauvegarde du modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_pipeline, os.path.join('models', 'logistic_regression_pipeline.joblib'))

    print("Modèle de Régression Logistique sauvegardé.")

    # --- Expérience 2 : Naive Bayes ---
    print("Entraînement du modèle Naive Bayes...")

    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
        ('clf', MultinomialNB())
    ])

    nb_pipeline.fit(X_train, y_train)
    joblib.dump(nb_pipeline, os.path.join('models', 'naive_bayes_pipeline.joblib'))

    print("Modèle Naive Bayes sauvegardé.")