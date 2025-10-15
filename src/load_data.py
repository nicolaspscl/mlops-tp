# src/load_data.py
import os
import io
import zipfile
import requests
import pandas as pd


def load_and_prepare_data(url: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Télécharge, extrait et prépare le jeu de données Sentiment140.

    Paramètres
    ----------
    url : str
        Lien direct vers le fichier zip du dataset Sentiment140.
    data_dir : str
        Répertoire local où sauvegarder les données extraites.

    Retourne
    --------
    df : pd.DataFrame
        DataFrame Pandas contenant deux colonnes : 'sentiment' et 'text'.
    """

    # 1. Crée le dossier data/ s'il n'existe pas déjà
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "sentiment140.zip")
    csv_path = os.path.join(data_dir, "training.1600000.processed.noemoticon.csv")

    # 2. Télécharger le fichier si le CSV n'existe pas déjà
    if not os.path.exists(csv_path):
        print("Téléchargement du jeu de données...")
        response = requests.get(url)
        response.raise_for_status()  # Vérifie que la requête a réussi

        # 3. Extraire le contenu du zip dans data/
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(data_dir)

        print("Téléchargement et extraction terminés.\n")

    # 4. Définir les colonnes du dataset
    cols = ["sentiment", "id", "date", "query", "user", "text"]

    # 5. Charger le CSV dans un DataFrame Pandas
    df = pd.read_csv(csv_path, header=None, names=cols, encoding="latin-1")

    # 6. Ne conserver que les colonnes utiles
    df = df[["sentiment", "text"]]

    # 7. Convertir les labels : 0 = négatif, 4 = positif → devient 1
    df["sentiment"] = df["sentiment"].replace({4: 1})

    print("Préparation des données terminée.\n")
    return df


if __name__ == "__main__":
    # URL officielle du dataset Sentiment140
    dataset_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    # Charger et échantillonner pour alléger le TP
    data_df = load_and_prepare_data(dataset_url).sample(n=50000, random_state=42)

    # Sauvegarde dans data/raw_tweets.csv
    output_path = os.path.join("data", "raw_tweets.csv")
    data_df.to_csv(output_path, index=False)

    print(f"Échantillon de données sauvegardé dans {output_path}")