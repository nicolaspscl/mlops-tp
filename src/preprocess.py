# src/preprocess.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

# ============================================================
# 🔧 Téléchargement des ressources NLTK nécessaires
# (obligatoire pour GitHub Actions)
# ============================================================

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")  # ← indispensable (corrige ton erreur actuelle)
nltk.download("wordnet")

# ============================================================
# 🔧 Fonction de prétraitement d'un tweet
# ============================================================

def preprocess_text(text):
    """
    Nettoie et prétraite un texte de tweet.
    """

    # 1. Passage en minuscules
    text = text.lower()

    # 2. Suppression des URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Suppression des mentions (@user) et hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    # 4. Suppression des caractères non alphabétiques
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Tokenisation
    tokens = word_tokenize(text)

    # 6. Suppression des stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# ============================================================
# 🔧 Exécution principale
# ============================================================

if __name__ == "__main__":
    print("Début du prétraitement du texte...")

    # Charger les données brutes
    raw_data_path = os.path.join("data", "raw_tweets.csv")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Fichier introuvable : {raw_data_path}")

    df = pd.read_csv(raw_data_path)

    # Prétraitement de la colonne "text"
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # Sauvegarde
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")

    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(train_path, index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(test_path, index=False)

    print(f"Prétraitement terminé.")
    print(f"Fichiers générés : {train_path} et {test_path}")
