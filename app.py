import os
import torch
from flask import Flask, request, jsonify
from flasgger import Swagger
from flasgger.utils import swag_from
from llm_rag_demo import load_llm, setup_rag_system, rag_query, generate_response
from sentence_transformers import SentenceTransformer
import logging
import chromadb

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
except Exception as e:
    print(f"Erreur lors du chargement initial des modèles : {e}")
    llm_model, llm_tokenizer = None, None

logger = logging.getLogger(__name__)

# Initialisation globale de la base vectorielle
rag_collection = None
embedder = None

# Wrapper pour la fonction d'embedding compatible ChromaDB
class EmbeddingFunctionWrapper:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def __call__(self, input):
        """
        Méthode compatible avec l'interface ChromaDB
        
        Args:
            input (List[str]): Liste de textes à encoder
        
        Returns:
            List[List[float]]: Liste d'embeddings
        """
        try:
            # Encoder tous les textes d'un coup
            embeddings = self.embedding_model.encode(input)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'embedding : {e}")
            raise

def initialize_rag_system():
    """
    Initialise le système RAG une seule fois au démarrage
    """
    global rag_collection, embedder
    
    try:
        # Charger l'embedder
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Modèle d'embedding initialisé")
        
        # Créer un wrapper compatible ChromaDB
        embedding_function = EmbeddingFunctionWrapper(embedder)
        
        # Créer ou charger la collection vectorielle
        rag_collection = chromadb.PersistentClient(path="./vectorstore").get_or_create_collection(
            name="document_collection", 
            embedding_function=embedding_function
        )
        logger.info(f"✅ Collection vectorielle initialisée avec {rag_collection.count()} segments")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erreur d'initialisation RAG : {e}")
        return False

# Initialiser le système RAG au démarrage
rag_initialization_success = initialize_rag_system()

def get_model_size(model):
    """
    Calcule la taille approximative d'un modèle
    
    Args:
        model: Modèle PyTorch ou Transformers
    
    Returns:
        dict: Informations sur la taille du modèle
    """
    try:
        # Calcul de la taille totale des paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimation de la taille mémoire
        param_size = total_params * 4 / (1024 ** 2)  # en Mo (float32)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_size_mb": round(param_size, 2),
            "device": str(next(model.parameters()).device)
        }
    except Exception as e:
        logger.error(f"❌ Erreur lors du calcul de la taille du modèle : {e}")
        return None

@app.route('/generate', methods=['POST'])
@swag_from({
    'tags': ['LLM Generation'],
    'description': 'Générer une réponse textuelle avec un modèle de langage',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'prompt': {
                        'type': 'string',
                        'description': 'Texte de départ pour la génération',
                        'example': 'Explique les bases de l\'intelligence artificielle'
                    }
                },
                'required': ['prompt']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Réponse générée avec succès',
            'schema': {
                'type': 'object',
                'properties': {
                    'response': {
                        'type': 'string',
                        'description': 'Texte généré par le modèle'
                    },
                    'prompt': {
                        'type': 'string',
                        'description': 'Prompt original'
                    }
                },
                'example': {
                    'response': 'L\'intelligence artificielle est un domaine...',
                    'prompt': 'Explique les bases de l\'intelligence artificielle'
                }
            }
        },
        '400': {
            'description': 'Requête invalide',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Message d\'erreur'
                    }
                },
                'example': {
                    'error': 'Paramètre \'prompt\' requis'
                }
            }
        },
        '500': {
            'description': 'Erreur serveur',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Message d\'erreur détaillé'
                    }
                },
                'example': {
                    'error': 'Modèle LLM non configuré'
                }
            }
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
        return jsonify({"response": response, "prompt": prompt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rag_query', methods=['POST'])
@swag_from({
    'tags': ['RAG Query'],
    'description': 'Effectuer une requête RAG avec recherche sémantique',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Question ou requête pour la recherche sémantique',
                        'example': 'Qu\'est-ce que le RAG ?'
                    }
                },
                'required': ['query']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Réponse RAG générée avec succès',
            'schema': {
                'type': 'object',
                'properties': {
                    'rag_response': {
                        'type': 'string',
                        'description': 'Réponse générée par le système RAG'
                    },
                    'query': {
                        'type': 'string',
                        'description': 'Requête originale'
                    }
                },
                'example': {
                    'rag_response': 'Le RAG (Retrieval-Augmented Generation) est une technique...',
                    'query': 'Qu\'est-ce que le RAG ?'
                }
            }
        },
        '400': {
            'description': 'Requête invalide',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Message d\'erreur'
                    }
                },
                'example': {
                    'error': 'Paramètre \'query\' requis'
                }
            }
        },
        '500': {
            'description': 'Erreur serveur',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Message d\'erreur détaillé'
                    }
                },
                'example': {
                    'error': 'Système RAG non configuré'
                }
            }
        }
    }
})
def semantic_search():
    """
    Endpoint pour effectuer une requête RAG
    """
    # Vérification de l'initialisation du système RAG
    if not rag_initialization_success:
        logger.error("🚨 Initialisation RAG a échoué")
        return jsonify({"error": "Système RAG non configuré"}), 500
    
    # Vérification des dépendances
    if not rag_collection:
        logger.error("🚨 Collection RAG non initialisée")
        return jsonify({"error": "Système RAG non configuré"}), 500
    
    if not embedder:
        logger.error("🚨 Modèle d'embedding non initialisé")
        return jsonify({"error": "Embedding non configuré"}), 500
    
    if not llm_model or not llm_tokenizer:
        logger.error("🚨 Modèle LLM non initialisé")
        return jsonify({"error": "Modèle LLM non configuré"}), 500
    
    # Récupération des données
    data = request.get_json()
    
    # Validation de la requête
    if not data:
        logger.warning("⚠️ Requête vide reçue")
        return jsonify({"error": "Corps de requête vide"}), 400
    
    query = data.get('query', '').strip()
    
    if not query:
        logger.warning("⚠️ Paramètre 'query' manquant ou vide")
        return jsonify({"error": "Paramètre 'query' requis"}), 400
    
    # Logging de la requête
    logger.info(f"🔍 Requête RAG reçue : {query}")
    
    try:
        # Exécution de la requête RAG
        rag_response = rag_query(
            rag_collection, 
            embedder, 
            llm_model, 
            llm_tokenizer, 
            query
        )
        
        # Logging du résultat
        logger.info(f"✅ Réponse RAG générée avec succès")
        
        return jsonify({
            "rag_response": rag_response,
            "query": query
        })
    
    except Exception as e:
        # Gestion des erreurs détaillée
        logger.error(f"❌ Erreur lors de la requête RAG : {str(e)}")
        return jsonify({
            "error": "Erreur lors du traitement de la requête RAG",
            "details": str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Endpoint pour obtenir les informations du modèle LLM
    """
    if not llm_model:
        return jsonify({"error": "Modèle non initialisé"}), 500
    
    model_details = get_model_size(llm_model)
    
    if not model_details:
        return jsonify({"error": "Impossible de récupérer les informations du modèle"}), 500
    
    return jsonify({
        "model_name": llm_model.config.model_type,
        "model_size": model_details
    })

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
        <li><strong>/model_info</strong>: Obtenir les informations du modèle</li>
    </ul>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
