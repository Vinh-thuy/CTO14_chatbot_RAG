# LightRAG : Système de Recherche et Génération Augmentée 

## Démarrage Rapide

### Options de Lancement

#### Mode Démo (Sans API)
```bash
python '/Users/vinh/Documents/RAG/llm_rag_demo.py'
```
- Exécute le système RAG directement
- Idéal pour les tests rapides
- Interface en ligne de commande

#### Mode API
```bash
python app.py
```
- Lance le serveur Flask
- Expose les endpoints `/generate` et `/rag_query`
- Documentation Swagger disponible à `/apidocs/`
- Port par défaut : 5001

### Prérequis Avant de Commencer
- Python 3.8+
- Installer les dépendances : `pip install -r requirements.txt`

## Démonstration de Retrieval-Augmented Generation (RAG) avec LLM

## Introduction

Ce projet est une démonstration pratique de Retrieval-Augmented Generation (RAG), une technique innovante qui améliore les capacités des modèles de langage en leur fournissant un contexte supplémentaire lors de la génération de réponses.

## Concepts Clés

### Qu'est-ce que le RAG ?
Le RAG combine deux technologies puissantes :
1. **Recherche sémantique** : Trouver des informations pertinentes dans un document
2. **Génération de texte** : Produire une réponse basée sur ces informations

### Composants Principaux
- **Modèle de Langage** : `distilgpt2` (version allégée de GPT-2)
- **Embeddings** : Transformations vectorielles des documents
- **Base de Données Vectorielle** : ChromaDB pour le stockage et la recherche sémantique

## Fonctionnalités Principales

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

## Workflow Détaillé

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

## Architecture Technique Détaillée

### 1. Système de Logging Personnalisé 

#### Objectif : Amélioration de la Lisibilité
- **Codes Couleurs Personnalisés** :
  - : Chargement du modèle Hugging Face
  - : Requêtes LLM
  - : Configuration du RAG
  - : Requêtes RAG

#### Avantages
- Distinction visuelle des étapes d'exécution
- Amélioration du débogage
- Expérience utilisateur améliorée

### 2. Chargement du Modèle LLM 

#### Étapes Principales
1. **Préparation du Cache**
   - Création d'un répertoire de cache personnalisé
   - Nettoyage du cache existant

2. **Chargement du Tokenizer**
   - Utilisation de `AutoTokenizer`
   - Configuration des tokens spéciaux
   - Logging détaillé :
     ```
     Téléchargement du tokenizer...
     Tokenizer téléchargé
     Taille du vocabulaire
     Tokens spéciaux
     ```

3. **Optimisations du Modèle**
   - `device_map='auto'` : Sélection automatique du périphérique
   - `torch_dtype=torch.float16` : Réduction de précision
   - `low_cpu_mem_usage=True` : Minimisation de l'utilisation CPU

### 3. Génération de Réponse 

#### Stratégies Intelligentes
- Sélection dynamique du périphérique (CPU/GPU/MPS)
- Tokenization adaptative
- Gestion des erreurs de génération

### 4. Extraction de Texte PDF 

#### Workflow
1. Ouverture du fichier PDF
2. Lecture séquentielle des pages
3. Concaténation du texte
4. Gestion robuste des erreurs

### 5. Configuration du Système RAG 

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

### 6. Requête RAG Intelligente 

#### Étapes de Recherche et Génération
1. Embedding de la requête utilisateur
2. Recherche sémantique dans ChromaDB
3. Récupération des passages contextuels
4. Génération de réponse enrichie

### 7. Comparaison Analytique 

#### Objectif : Évaluation Comparative
- Comparaison des réponses :
  1. Sans contexte RAG
  2. Avec contexte RAG
- Mesure de la valeur ajoutée du RAG

### Principes de Conception 

#### Approche Technique
- **Modularité** : Responsabilités uniques par fonction
- **Logging Exhaustif** : Traçabilité complète
- **Adaptabilité** : Configuration dynamique
- **Performance** : Optimisation des ressources

### Points Techniques Avancés 

- Accélération GPU avec `torch`
- Gestion dynamique des périphériques
- Logging coloré et informatif
- Embeddings sémantiques avancés
- Recherche contextuelle intelligente

### Limitations Actuelles 

- Modèle léger (`distilgpt2`)
- Performances pour tâches complexes
- Dépendance à la qualité du document source

## Perspectives d'Amélioration

1. Intégration de modèles plus performants
2. Amélioration de la recherche sémantique
3. Développement de métriques d'évaluation
4. Support multi-documents
5. Interface utilisateur plus intuitive

## Implémentation Technique de l'API Flask

### Architecture de l'API

#### Endpoints Principaux
1. **`/generate`** : Génération de texte
   - **Méthode** : POST
   - **Objectif** : Générer du texte avec le modèle LLM
   - **Paramètres** :
     - `prompt` (requis) : Texte de départ pour la génération

2. **`/rag_query`** : Recherche Sémantique RAG
   - **Méthode** : POST
   - **Objectif** : Effectuer une recherche sémantique contextuelle
   - **Paramètres** :
     - `query` (requis) : Question ou requête pour la recherche

3. **`/model_info`** : Informations du Modèle
   - **Méthode** : GET
   - **Objectif** : Fournir des informations détaillées sur le modèle de langage utilisé
   - **Paramètres** : Aucun

### Format des Requêtes 

#### Structure JSON des Endpoints 

#### 1. Endpoint `/generate`

##### JSON d'Entrée
```json
{
  "prompt": "Votre texte de départ pour la génération"
}
```

##### JSON de Réponse (Succès)
```json
{
  "response": "Texte généré par le modèle LLM",
  "prompt": "Votre texte de départ"
}
```

##### JSON de Réponse (Erreur)
```json
{
  "error": "Message décrivant l'erreur"
}
```

#### 2. Endpoint `/rag_query`

##### JSON d'Entrée
```json
{
  "query": "Votre question ou requête sémantique"
}
```

##### JSON de Réponse (Succès)
```json
{
  "rag_response": "Réponse générée par le système RAG",
  "query": "Votre question originale"
}
```

##### JSON de Réponse (Erreur)
```json
{
  "error": "Message décrivant l'erreur"
}
```

### Validation des Requêtes 

#### Règles Générales
- Chaque endpoint attend un JSON valide
- Le champ requis doit être non vide
- Le Content-Type doit être `application/json`

#### Exemples de Requêtes Valides

```bash
# Génération de texte
curl -X POST http://localhost:5001/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explique l\'intelligence artificielle"}'

# Requête RAG
curl -X POST http://localhost:5001/rag_query \
     -H "Content-Type: application/json" \
     -d '{"query": "Qu\'est-ce que le RAG ?"}'
```

### Gestion des Erreurs 

#### Types d'Erreurs Courants
1. **400 Bad Request** : 
   - JSON invalide
   - Champ requis manquant
   - Champ vide
2. **500 Internal Server Error** :
   - Modèle non initialisé
   - Erreur système
   - Problème de traitement

### Débogage des Requêtes 

Si vous rencontrez des problèmes :
1. Vérifiez le format JSON
2. Assurez-vous que le champ `query` ou `prompt` est présent et non vide
3. Utilisez un validateur JSON en ligne
4. Consultez les logs du serveur pour des détails spécifiques

### Débogage des Requêtes de Génération 

Si vous rencontrez des problèmes :
1. Vérifiez le format JSON
2. Assurez-vous que le champ `prompt` est présent et non vide
3. Utilisez un validateur JSON en ligne
4. Consultez les logs du serveur pour des détails spécifiques

### Gestion des Erreurs et Logging 

#### Principes de Gestion des Erreurs
- Validation stricte des requêtes
- Messages d'erreur détaillés
- Logging complet des événements

#### Types de Validation
- Vérification de l'initialisation des modèles
- Contrôle de la présence et du format des paramètres
- Gestion des exceptions durant le traitement

### Exemple de Workflow de Requête

1. **Validation de la Requête**
   - Vérification de l'existence du modèle
   - Validation des paramètres d'entrée
   - Logging de la requête reçue

2. **Traitement**
   - Génération de réponse ou recherche sémantique
   - Capture des erreurs potentielles
   - Logging du résultat

3. **Réponse**
   - Retour JSON structuré
   - Codes de statut HTTP appropriés
   - Informations de débogage si nécessaire

### Sécurité et Configuration 

#### Bonnes Pratiques
- Mode debug désactivé en production
- Potential configuration CORS
- Préparation pour l'ajout d'authentification

#### Déploiement
- Serveur recommandé : Gunicorn
- Configuration multi-workers
- Binding sur toutes les interfaces réseau

### Exemples Avancés

#### Requête de Génération
```bash
curl -X POST http://localhost:5001/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explique le machine learning"}'
```

#### Requête RAG
```bash
curl -X POST http://localhost:5001/rag_query \
     -H "Content-Type: application/json" \
     -d '{"query": "Qu\'est-ce que le RAG ?"}'
```

#### Requête d'Informations du Modèle
```bash
curl http://localhost:5001/model_info
```

### Perspectives d'Amélioration 

1. Ajout de métriques de performance
2. Mise en place de la pagination pour les grandes réponses
3. Intégration de mécanismes de cache
4. Développement de tests unitaires et d'intégration
5. Mise en place de la surveillance des requêtes

## Prérequis Techniques

- Python 3.8+
- Bibliothèques :
  - `torch`
  - `transformers`
  - `chromadb`
  - `sentence-transformers`
  - `PyPDF2`

## Comment Utiliser

1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. Préparez un fichier PDF

3. Lancez la démonstration :
   ```bash
   python llm_rag_demo.py
   ```

## Pour les Débutants

- **Modèle de Langage** : Un "assistant" qui génère du texte
- **Embedding** : Transformer du texte en "coordonnées" pour la recherche
- **Sémantique** : Comprendre le sens, pas juste les mots

## Points Techniques Avancés

- Utilisation de `torch` pour l'accélération GPU
- Gestion dynamique du périphérique (CPU/GPU)
- Logging coloré pour une meilleure lisibilité

## Limitations

- Modèle léger (`distilgpt2`)
- Performances limitées pour des tâches complexes
- Nécessite un bon document source

## Contributions

N'hésitez pas à ouvrir des issues ou proposer des améliorations !

## Licence

[Spécifiez votre licence]

## API Flask avec Swagger

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

#### 3. Informations du Modèle `/model_info`
- **Méthode** : GET
- **Description** : Fournir des informations détaillées sur le modèle de langage utilisé
- **Paramètres** : Aucun

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
   - URL : `http://localhost:5001/apidocs/`

### Exemple de Requête cURL

#### Génération de Texte
```bash
curl -X POST http://localhost:5001/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explique le machine learning"}'
```

#### Requête RAG
```bash
curl -X POST http://localhost:5001/rag_query \
     -H "Content-Type: application/json" \
     -d '{"query": "Qu\'est-ce que le RAG ?"}'
```

#### Requête d'Informations du Modèle
```bash
curl http://localhost:5001/model_info
```

### Déploiement

- Serveur de production recommandé : Gunicorn
- Commande de démarrage :
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5001 app:app
  ```

### Sécurité et Configuration

- Mode debug désactivé en production
- Configuration CORS si nécessaire
- Ajout potentiel d'authentification

### Optimisation de la Base Vectorielle 

#### Stratégie d'Initialisation
- **Initialisation Unique** : La base vectorielle est créée une seule fois au démarrage de l'application
- **Persistance** : Stockage dans un répertoire local `./vectorstore`
- **Réutilisation** : La même collection est utilisée pour toutes les requêtes RAG

#### Avantages
1. **Performance Améliorée** : Évite la reconstruction à chaque requête
2. **Économie de Ressources** : Minimise la charge CPU et mémoire
3. **Cohérence** : Maintient un contexte constant entre les requêtes

#### Gestion des Erreurs
- Vérification de l'initialisation avant chaque requête
- Logs détaillés en cas d'échec
- Mécanisme de reprise en cas de problème

### Monitoring de la Base Vectorielle 

#### Informations Disponibles
- Nombre total de segments indexés
- Chemin de stockage
- Modèle d'embedding utilisé

#### Exemple de Log d'Initialisation
```
 Modèle d'embedding initialisé
 Collection vectorielle initialisée avec 78 segments
```

### Perspectives d'Amélioration 

1. Mise en place d'un mécanisme de rafraîchissement périodique
2. Ajout de métriques de performance
3. Implémentation d'un système de cache intelligent

### Compatibilité des Embeddings 

#### Problématique
- Changement récent dans l'interface ChromaDB
- Nécessité d'adapter la fonction d'embedding

#### Solution : Wrapper d'Embedding 
```python
class EmbeddingFunctionWrapper:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def __call__(self, input):
        # Conversion des embeddings pour ChromaDB
        embeddings = self.embedding_model.encode(input)
        return embeddings.tolist()
```

#### Avantages du Wrapper
1. **Compatibilité** : Interface standardisée avec ChromaDB
2. **Flexibilité** : Supporte différents modèles d'embedding
3. **Gestion des Erreurs** : Logging et remontée des exceptions

### Détails Techniques de l'Embedding 

#### Modèle Utilisé
- **Nom** : all-MiniLM-L6-v2
- **Type** : Sentence Transformer
- **Dimensionnalité** : 384 dimensions

#### Processus d'Embedding
1. Réception d'une liste de textes
2. Encodage en vecteurs numériques
3. Conversion en format compatible ChromaDB

### Optimisations Futures 

1. Support de modèles d'embedding dynamiques
2. Mise en cache des embeddings
3. Métriques de performance de l'embedding

### Endpoint d'Informations du Modèle 

#### `/model_info` - Détails Techniques du Modèle

##### Description
Fournit des informations détaillées sur le modèle de langage utilisé.

##### Exemple de Réponse
```json
{
  "model_name": "gpt2",
  "model_size": {
    "total_parameters": 124_000_000,
    "trainable_parameters": 124_000_000,
    "estimated_size_mb": 480.47,
    "device": "mps"
  }
}
```

##### Informations Retournées
- **Nom du Modèle** : Type de modèle utilisé
- **Nombre Total de Paramètres**
- **Paramètres Entraînables**
- **Taille Estimée** (en Mo)
- **Périphérique** (CPU, GPU, MPS)

#### Cas d'Usage
- Diagnostic de configuration
- Monitoring des ressources
- Compréhension des capacités du modèle

### Exemple de Requête
```bash
curl http://localhost:5001/model_info
```

### Interprétation des Résultats 

- **Paramètres** : Nombre de poids ajustables
- **Taille** : Estimation de l'empreinte mémoire
- **Périphérique** : Matériel d'exécution
