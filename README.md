---

## 🧩 Étapes du pipeline MLOps

### 1️⃣ Acquisition des données (`src/load_data.py`)
- Téléchargement du jeu **Sentiment140** (1,6 M de tweets).
- Extraction et filtrage des colonnes pertinentes (`sentiment`, `text`).
- Conversion des labels :  
  - `0 → négatif`  
  - `4 → positif`  
- Échantillonnage de 50 000 tweets pour des raisons de performance.

➡️ Résultat : `data/raw_tweets.csv`

---

### 2️⃣ Prétraitement du texte (`src/preprocess.py`)
- Nettoyage du texte (URLs, mentions, hashtags, chiffres, ponctuation).  
- Conversion en minuscules.  
- Tokenisation et suppression des stopwords (`nltk`).
- Lemmatisation (`WordNetLemmatizer`).
- Séparation **train/test (80/20)** de manière stratifiée.  

➡️ Résultat :  
`data/train.csv` et `data/test.csv`

---

### 3️⃣ Entraînement des modèles (`src/train.py`)
Deux modèles ont été entraînés à l’aide d’un **pipeline scikit-learn** combinant :
- **Vectorisation TF-IDF** : représentation des mots selon leur importance locale et globale.
- **Modèles comparés** :
  - **Régression Logistique (Logistic Regression)** → modèle *discriminatif*.
  - **Naive Bayes Multinomial (MultinomialNB)** → modèle *génératif*.

➡️ Résultat :  
`models/logistic_regression_pipeline.joblib`  
`models/naive_bayes_pipeline.joblib`

---

### 4️⃣ Évaluation des performances (`src/evaluate.py`)
- Chargement des jeux de test et des modèles sauvegardés.  
- Génération de rapports de classification (`precision`, `recall`, `f1-score`, `accuracy`).  
- Comparaison quantitative entre les deux modèles.

#### 🧮 Résultats obtenus :

| Modèle                   | Accuracy | F1-score (pondéré) |
|---------------------------|-----------|--------------------|
| Régression Logistique     | **0.7507** | **0.7506**         |
| Naive Bayes               | 0.7443    | 0.7443             |

Ces résultats confirment la supériorité de la Régression Logistique sur des données textuelles riches, en accord avec la théorie.

---

## ⚙️ Installation et exécution

### 🔧 1. Cloner le dépôt
```bash
git clone https://github.com/nicolaspscl/mlops-tp.git
cd mlops-tp
```

### 🔧 2. Créer l’environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# ou
venv\Scripts\activate         # Windows
```

### 🔧 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### ▶️ 4. Exécuter le pipeline complet
```bash
python src/load_data.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
```
