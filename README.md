# Démonstration de Retrieval-Augmented Generation (RAG) avec LLM

## 🚀 Introduction

Ce projet est une démonstration pratique de Retrieval-Augmented Generation (RAG), une technique innovante qui améliore les capacités des modèles de langage en leur fournissant un contexte supplémentaire lors de la génération de réponses.

## 📘 Concepts Clés

### Qu'est-ce que le RAG ?
Le RAG combine deux technologies puissantes :
1. **Recherche sémantique** : Trouver des informations pertinentes dans un document
2. **Génération de texte** : Produire une réponse basée sur ces informations

### Composants Principaux
- **Modèle de Langage** : `distilgpt2` (version allégée de GPT-2)
- **Embeddings** : Transformations vectorielles des documents
- **Base de Données Vectorielle** : ChromaDB pour le stockage et la recherche sémantique

## 🛠 Fonctionnalités Principales

### 1. Chargement du Modèle LLM
- Téléchargement du modèle `distilgpt2`
- Configuration optimisée pour les performances
- Gestion intelligente du cache

### 2. Extraction de Texte PDF
- Lecture automatique de documents PDF
- Extraction du contenu textuel

### 3. Indexation Sémantique
- Transformation du texte en vecteurs
- Création d'une base de données vectorielle
- Permet une recherche sémantique précise

### 4. Génération de Réponses
- Deux modes de génération :
  1. **Sans Contexte** : Réponse basique du modèle
  2. **Avec RAG** : Réponse enrichie par des informations contextuelles

## 🔍 Workflow Détaillé

1. **Préparation**
   - Charger le modèle de langage
   - Initialiser l'embedder
   - Extraire le texte du PDF

2. **Indexation**
   - Diviser le texte en segments
   - Créer des embeddings
   - Stocker dans ChromaDB

3. **Requête**
   - L'utilisateur pose une question
   - Recherche sémantique des passages pertinents
   - Génération de la réponse avec contexte

## 🔬 Architecture Technique Détaillée

### 1. Système de Logging Personnalisé 🌈

#### Objectif : Amélioration de la Lisibilité
- **Codes Couleurs Personnalisés** :
  - 🟣 Magenta : Chargement du modèle Hugging Face
  - 🔵 Bleu : Requêtes LLM
  - 🟢 Vert : Configuration du RAG
  - 🔷 Cyan : Requêtes RAG

#### Avantages
- Distinction visuelle des étapes d'exécution
- Amélioration du débogage
- Expérience utilisateur améliorée

### 2. Chargement du Modèle LLM 🤖

#### Étapes Principales
1. **Préparation du Cache**
   - Création d'un répertoire de cache personnalisé
   - Nettoyage du cache existant

2. **Chargement du Tokenizer**
   - Utilisation de `AutoTokenizer`
   - Configuration des tokens spéciaux
   - Logging détaillé :
     ```
     🤖 Téléchargement du tokenizer...
     ✅ Tokenizer téléchargé
     📊 Taille du vocabulaire
     🏷️ Tokens spéciaux
     ```

3. **Optimisations du Modèle**
   - `device_map='auto'` : Sélection automatique du périphérique
   - `torch_dtype=torch.float16` : Réduction de précision
   - `low_cpu_mem_usage=True` : Minimisation de l'utilisation CPU

### 3. Génération de Réponse 🔍

#### Stratégies Intelligentes
- Sélection dynamique du périphérique (CPU/GPU/MPS)
- Tokenization adaptative
- Gestion des erreurs de génération

### 4. Extraction de Texte PDF 📄

#### Workflow
1. Ouverture du fichier PDF
2. Lecture séquentielle des pages
3. Concaténation du texte
4. Gestion robuste des erreurs

### 5. Configuration du Système RAG 🏗️

#### Processus d'Indexation
1. **Préparation du Texte**
   - Division en segments intelligents
   - Utilisation de `RecursiveCharacterTextSplitter`

2. **Création d'Embeddings**
   - Transformation vectorielle
   - Modèle `SentenceTransformer`

3. **Indexation Vectorielle**
   - Collection ChromaDB
   - Stockage des représentations sémantiques

### 6. Requête RAG Intelligente 🕵️

#### Étapes de Recherche et Génération
1. Embedding de la requête utilisateur
2. Recherche sémantique dans ChromaDB
3. Récupération des passages contextuels
4. Génération de réponse enrichie

### 7. Comparaison Analytique 🔬

#### Objectif : Évaluation Comparative
- Comparaison des réponses :
  1. Sans contexte RAG
  2. Avec contexte RAG
- Mesure de la valeur ajoutée du RAG

### Principes de Conception 🎯

#### Approche Technique
- **Modularité** : Responsabilités uniques par fonction
- **Logging Exhaustif** : Traçabilité complète
- **Adaptabilité** : Configuration dynamique
- **Performance** : Optimisation des ressources

### Points Techniques Avancés 🚀

- Accélération GPU avec `torch`
- Gestion dynamique des périphériques
- Logging coloré et informatif
- Embeddings sémantiques avancés
- Recherche contextuelle intelligente

### Limitations Actuelles 🚧

- Modèle léger (`distilgpt2`)
- Performances pour tâches complexes
- Dépendance à la qualité du document source

## 🔮 Perspectives d'Amélioration

1. Intégration de modèles plus performants
2. Amélioration de la recherche sémantique
3. Développement de métriques d'évaluation
4. Support multi-documents
5. Interface utilisateur plus intuitive

## 💻 Prérequis Techniques

- Python 3.8+
- Bibliothèques :
  - `torch`
  - `transformers`
  - `chromadb`
  - `sentence-transformers`
  - `PyPDF2`

## 🚦 Comment Utiliser

1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. Préparez un fichier PDF

3. Lancez la démonstration :
   ```bash
   python llm_rag_demo.py
   ```

## 🎓 Pour les Débutants

- **Modèle de Langage** : Un "assistant" qui génère du texte
- **Embedding** : Transformer du texte en "coordonnées" pour la recherche
- **Sémantique** : Comprendre le sens, pas juste les mots

## 🔬 Points Techniques Avancés

- Utilisation de `torch` pour l'accélération GPU
- Gestion dynamique du périphérique (CPU/GPU)
- Logging coloré pour une meilleure lisibilité

## 🚧 Limitations

- Modèle léger (`distilgpt2`)
- Performances limitées pour des tâches complexes
- Nécessite un bon document source

## 🤝 Contributions

N'hésitez pas à ouvrir des issues ou proposer des améliorations !

## 📜 Licence

[Spécifiez votre licence]

## 🌐 API Flask avec Swagger

### Fonctionnalités de l'API

#### 1. Génération de Texte `/generate`
- **Méthode** : POST
- **Description** : Générer du texte avec le modèle LLM
- **Paramètres** :
  - `prompt` (requis) : Texte de départ pour la génération

#### 2. Requête RAG `/rag_query`
- **Méthode** : POST
- **Description** : Effectuer une recherche sémantique avec RAG
- **Paramètres** :
  - `query` (requis) : Question ou requête pour la recherche

### Démarrage de l'API

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. Lancer l'API :
   ```bash
   python app.py
   ```

3. Accéder à la documentation Swagger :
   - URL : `http://localhost:5000/apidocs/`

### Exemple de Requête cURL

#### Génération de Texte
```bash
curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explique le machine learning"}'
```

#### Requête RAG
```bash
curl -X POST http://localhost:5000/rag_query \
     -H "Content-Type: application/json" \
     -d '{"query": "Qu'est-ce que le RAG ?"}'
```

### Déploiement

- Serveur de production recommandé : Gunicorn
- Commande de démarrage :
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  ```

### Sécurité et Configuration

- Mode debug désactivé en production
- Configuration CORS si nécessaire
- Ajout potentiel d'authentification
