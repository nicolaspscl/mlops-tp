# src/preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

# ---------------------------------------------------------
# Chargement des ressources NLTK (fix pour GitHub Actions)
# ---------------------------------------------------------

# Stopwords
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# Punkt tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Punkt_tab (nécessaire avec nltk>=3.9)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# WordNet
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


def preprocess_text(text: str) -> str:
    """
    Nettoie et prétraite un texte de tweet.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove @mentions and hashtags
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    # 4. Remove non alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 5. Tokenisation
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]

    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)


if __name__ == "__main__":

    raw_path = os.path.join("data", "raw_tweets.csv")
    print("Chargement des données brutes...")
    df = pd.read_csv(raw_path)

    print("Prétraitement du texte...")
    df["cleaned_text"] = df["text"].astype(str).apply(preprocess_text)

    print("Séparation train/test...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["sentiment"],
        random_state=42
    )

    os.makedirs("data", exist_ok=True)
    train_df.to_csv(os.path.join("data", "train.csv"), index=False)
    test_df.to_csv(os.path.join("data", "test.csv"), index=False)

    print("Fichiers enregistrés : data/train.csv & data/test.csv")