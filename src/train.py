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
    train_df = pd.read_csv(os.path.join("data", "train.csv"))
    X_train = train_df["text"].astype(str)
    y_train = train_df["sentiment"]

    os.makedirs("models", exist_ok=True)

    # ------------------------------
    # Modèle 1 : Régression Logistique
    # ------------------------------
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200, random_state=42))
    ])

    print("Entraînement du modèle LogisticRegression...")
    lr_pipeline.fit(X_train, y_train)
    joblib.dump(lr_pipeline, "models/logistic_regression_pipeline.joblib")
    print("Modèle LogisticRegression sauvegardé.\n")

    # ------------------------------
    # Modèle 2 : Naive Bayes
    # ------------------------------
    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", MultinomialNB())
    ])

    print("Entraînement du modèle NaiveBayes...")
    nb_pipeline.fit(X_train, y_train)
    joblib.dump(nb_pipeline, "models/naive_bayes_pipeline.joblib")
    print("Modèle NaiveBayes sauvegardé.\n")

    print("Tous les modèles ont été entraînés et sauvegardés dans /models/")