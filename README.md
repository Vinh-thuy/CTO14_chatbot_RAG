# Démonstration LLM et RAG sur Mac M1

## Prérequis
- Python 3.9+
- Processeur Apple Silicon (M1/M2)

## Installation
1. Créer un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Exécution
```bash
python llm_rag_demo.py
```

## Description
- Utilise un petit modèle LLM de 1B (Orca-2-1B)
- Implémente un système RAG simple avec ChromaDB
- Démontre la génération de texte et la recherche contextuelle

### Fonctionnalités
- Chargement de modèle LLM
- Génération de texte
- Création de base vectorielle
- Recherche contextuelle avec RAG

### Notes
- Adapté pour les systèmes Mac M1/M2
- Optimisé pour les ressources limitées
