import os
import torch
from flask import Flask, request, jsonify
from flasgger import Swagger
from flasgger.utils import swag_from
from llm_rag_demo import load_llm, setup_rag_system, rag_query, generate_response
from sentence_transformers import SentenceTransformer

# Configuration de l'application Flask
app = Flask(__name__)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}
Swagger(app, config=swagger_config)

# Chargement initial des modèles
try:
    llm_model, llm_tokenizer = load_llm()
    pdf_path = os.path.join(os.path.dirname(__file__), 'LightRAG.pdf')
    rag_collection = setup_rag_system(pdf_path)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Erreur lors du chargement initial des modèles : {e}")
    llm_model, llm_tokenizer, rag_collection, embedder = None, None, None, None

@app.route('/generate', methods=['POST'])
@swag_from({
    'tags': ['LLM Generation'],
    'description': 'Générer une réponse textuelle avec un modèle de langage',
    'parameters': [
        {
            'name': 'prompt',
            'in': 'body',
            'type': 'string',
            'required': True,
            'description': 'Texte de prompt pour générer une réponse'
        }
    ],
    'responses': {
        '200': {
            'description': 'Réponse générée avec succès',
            'schema': {
                'type': 'object',
                'properties': {
                    'response': {'type': 'string'}
                }
            }
        },
        '500': {
            'description': 'Erreur lors de la génération'
        }
    }
})
def generate_text():
    """
    Endpoint pour générer du texte avec le modèle LLM
    """
    if not llm_model or not llm_tokenizer:
        return jsonify({"error": "Modèle LLM non initialisé"}), 500
    
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "Prompt requis"}), 400
    
    try:
        response = generate_response(llm_model, llm_tokenizer, prompt)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rag_query', methods=['POST'])
@swag_from({
    'tags': ['RAG Query'],
    'description': 'Effectuer une requête RAG avec recherche sémantique',
    'parameters': [
        {
            'name': 'query',
            'in': 'body',
            'type': 'string',
            'required': True,
            'description': 'Question ou requête pour la recherche sémantique'
        }
    ],
    'responses': {
        '200': {
            'description': 'Réponse RAG générée avec succès',
            'schema': {
                'type': 'object',
                'properties': {
                    'rag_response': {'type': 'string'},
                    'context': {'type': 'array', 'items': {'type': 'string'}}
                }
            }
        },
        '500': {
            'description': 'Erreur lors de la requête RAG'
        }
    }
})
def semantic_search():
    """
    Endpoint pour effectuer une requête RAG
    """
    if not rag_collection or not embedder or not llm_model or not llm_tokenizer:
        return jsonify({"error": "Système RAG non initialisé"}), 500
    
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Requête requise"}), 400
    
    try:
        rag_response = rag_query(
            rag_collection, 
            embedder, 
            llm_model, 
            llm_tokenizer, 
            query
        )
        return jsonify({"rag_response": rag_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """
    Page d'accueil avec des informations sur l'API
    """
    return """
    <h1>🤖 LLM RAG API</h1>
    <p>Bienvenue sur l'API de démonstration RAG !</p>
    <ul>
        <li><a href="/apidocs/">📘 Documentation Swagger</a></li>
        <li><strong>/generate</strong>: Générer du texte avec un LLM</li>
        <li><strong>/rag_query</strong>: Effectuer une recherche sémantique</li>
    </ul>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
