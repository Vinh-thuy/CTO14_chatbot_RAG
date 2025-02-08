import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
import logging
import colorlog
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Codes de couleurs ANSI
class LogColors:
    MAGENTA = '\033[35m'  # Chargement du modèle Hugging Face
    BLUE = '\033[34m'     # Requête LLM
    GREEN = '\033[32m'    # Configuration du RAG
    CYAN = '\033[36m'     # Requête RAG
    RESET = '\033[0m'     # Réinitialiser la couleur

def log_hf_llm_load(message):
    """Log les messages de chargement du modèle Hugging Face en magenta"""
    print(f"{LogColors.MAGENTA}🤖 {message}{LogColors.RESET}")

def log_llm_query(message):
    """Log les messages de requête LLM en bleu"""
    print(f"{LogColors.BLUE}🔍 {message}{LogColors.RESET}")

def log_rag_setup(message):
    """Log les messages de configuration du RAG en vert"""
    print(f"{LogColors.GREEN}🏗️ {message}{LogColors.RESET}")

def log_rag_query(message):
    """Log les messages de requête RAG en cyan"""
    print(f"{LogColors.CYAN}🕵️ {message}{LogColors.RESET}")

# Configuration du logging avec couleurs et mise en forme améliorée
def setup_logger():
    """
    Configurer un logger personnalisé avec des couleurs et un formatage amélioré.
    
    Returns:
        logging.Logger: Logger configuré
    """
    # Créer un gestionnaire de console
    handler = colorlog.StreamHandler()
    
    # Définir le format du log avec des couleurs
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(levelname)s]%(reset)s '
        '%(blue)s%(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    handler.setFormatter(formatter)
    
    # Configurer le logger
    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

# Initialiser le logger
logger = setup_logger()

def log_section_separator(title, symbol='=', length=80):
    """
    Génère un séparateur visuel pour les logs
    
    Args:
        title (str): Titre de la section
        symbol (str, optional): Symbole de séparation. Defaults to '='.
        length (int, optional): Longueur du séparateur. Defaults to 80.
    """
    separator = symbol * length
    centered_title = title.center(length)
    logger.info("\n" + separator)
    logger.info(centered_title)
    logger.info(separator + "\n")

# Modèle LLM sélectionné
model_name = "distilgpt2"

def load_llm():
    """
    Charge le modèle de langage avec des paramètres optimisés et force le téléchargement
    
    Returns:
        tuple: (modèle, tokenizer)
    """
    import os
    import shutil
    import huggingface_hub

    # Chemin du cache personnalisé
    custom_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'custom_llm_cache')
    
    # Supprimer le cache existant pour forcer le téléchargement
    logger.info(f"🔄 Préparation du téléchargement forcé du modèle : {model_name}")
    if os.path.exists(custom_cache_dir):
        logger.info(f"🗑️ Suppression du cache existant : {custom_cache_dir}")
        shutil.rmtree(custom_cache_dir)
    
    os.makedirs(custom_cache_dir, exist_ok=True)
    
    # Étape 1 : Téléchargement et chargement du tokenizer
    logger.info("🤖 Téléchargement du tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=custom_cache_dir,
            use_fast=True,
            force_download=True,
            resume_download=False
        )
        
        # Configuration du tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✅ Tokenizer téléchargé : {tokenizer.__class__.__name__}")
        logger.info(f"📊 Taille du vocabulaire : {len(tokenizer.vocab)} tokens")
        
        # Détails des tokens spéciaux
        special_tokens = {
            'bos_token': tokenizer.bos_token, 
            'eos_token': tokenizer.eos_token, 
            'unk_token': tokenizer.unk_token,
            'pad_token': tokenizer.pad_token
        }
        logger.info("🏷️ Tokens spéciaux : " + str(special_tokens))
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement du tokenizer : {e}")
        raise
    
    # Étape 2 : Téléchargement et chargement du modèle
    logger.info("🤖 Téléchargement du modèle de langage...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=custom_cache_dir,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            force_download=True,
            resume_download=False,
            attn_implementation='eager'
        )
        
        # Configuration du modèle
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Informations détaillées sur le modèle
        logger.info(f"✅ Modèle téléchargé : {model.__class__.__name__}")
        logger.info(f"📊 Nombre de paramètres : {model.num_parameters():,}")
        
        # Vérification de la taille du modèle
        import glob
        model_files = glob.glob(os.path.join(custom_cache_dir, '*.safetensors')) + \
                      glob.glob(os.path.join(custom_cache_dir, 'pytorch_model.bin'))
        
        if model_files:
            model_size_mb = os.path.getsize(model_files[0]) / (1024 * 1024)
            logger.info(f"💾 Taille du modèle : {model_size_mb:.2f} Mo")
        else:
            logger.warning("⚠️ Impossible de déterminer la taille du modèle")
        
        # Informations sur le périphérique
        logger.info(f"💻 Périphérique : {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement du modèle : {e}")
        raise
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=500):
    """
    Génère une réponse textuelle avec des paramètres adaptés au modèle
    
    Args:
        model: Modèle de génération de texte
        tokenizer: Tokenizer associé
        prompt: Texte d'entrée
        max_length: Longueur maximale de la réponse
    
    Returns:
        str: Réponse générée
    """
    log_llm_query("Détails de la requête LLM")
    log_llm_query(f"Prompt original : {prompt}")
    
    try:
        # Configurer le périphérique
        if torch.backends.mps.is_available():
            device = torch.device("cpu")  # Forcer CPU pour éviter les problèmes MPS
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Déplacer le modèle sur le bon périphérique
        model = model.to(device)
        
        # Préparation de l'entrée
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        log_llm_query(f"Analyse du prompt :")
        log_llm_query(f"Nombre de tokens : {len(inputs['input_ids'][0])}")
        
        # Configuration de la génération
        log_llm_query("Paramètres de génération :")
        log_llm_query("Mode : Génération avec échantillonnage")
        log_llm_query(f"Longueur max : {max_length} tokens")
        
        # Génération de la réponse
        outputs = model.generate(
            **inputs, 
            max_length=max_length, 
            num_return_sequences=1, 
            do_sample=True, 
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Décodage de la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        log_llm_query("Résultat de la génération :")
        log_llm_query(f"Longueur : {len(response)} caractères")
        log_llm_query(f"Début de la réponse : {response[:100]}...")
        
        return response
    except Exception as e:
        logger.error(f"Erreur lors de la génération : {e}")
        return f"Erreur de génération : {str(e)}"

def extract_pdf_text(pdf_path):
    """
    Extrait le texte d'un fichier PDF.
    
    Étapes :
    1. Ouvrir le fichier PDF
    2. Lire chaque page
    3. Concaténer le texte
    
    Args:
        pdf_path: Chemin vers le fichier PDF
    
    Returns:
        str: Texte extrait du PDF
    """
    logger.info(f"Extraction du texte du PDF : {pdf_path}")
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        logger.info(f"Extraction réussie : {len(text)} caractères extraits")
        return text
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction PDF : {e}")
        return ""

def setup_rag_system(pdf_path):
    """
    Configuration du système RAG avec indexation de document PDF
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF à indexer
    
    Returns:
        chromadb.Collection: Collection vectorielle pour la recherche sémantique
    """
    logger.info("🏗️ Configuration détaillée du système RAG")
    logger.info(f"🏗️ Source de connaissances : {pdf_path}")
    
    # Initialisation de ChromaDB
    logger.info("🏗️ Initialisation de la base de données vectorielle")
    client = chromadb.PersistentClient(path="./chroma_storage")
    
    # Créer ou récupérer une collection
    collection_name = "lightrag_collection"
    try:
        collection = client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("🏗️ Collection ChromaDB :")
        logger.info("🏗️ Création : Réussi")
    except Exception as e:
        logger.error(f"Erreur lors de la création de la collection : {e}")
        raise
    
    # Extraction du texte du PDF
    text = extract_pdf_text(pdf_path)
    logger.info(f"[INFO] Extraction du texte du PDF : {pdf_path}")
    logger.info(f"[INFO] Extraction réussie : {len(text)} caractères extraits")
    
    # Configuration du modèle d'embedding
    logger.info("🏗️ Analyse du texte source :")
    logger.info(f"🏗️ Nombre total de caractères : {len(text):,}")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("🏗️ Modèle d'embedding :")
    logger.info("🏗️ Nom : all-MiniLM-L6-v2")
    
    # Segmentation du texte
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    
    logger.info("🏗️ Segmentation du texte :")
    logger.info(f"🏗️ Nombre de segments : {len(text_chunks)}")
    
    # Embedding et indexation des segments
    for i, chunk in enumerate(text_chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    
    logger.info("🏗️ Résumé de l'indexation RAG :")
    logger.info(f"🏗️ Segments indexés : {len(text_chunks)}")
    logger.info(f"🏗️ Taille de la collection : {len(text_chunks)} segments")
    
    return collection, embedder

def rag_query(collection, embedder, model, tokenizer, query, max_context_length=1000):
    """
    Effectue une requête RAG avec recherche sémantique et génération de réponse
    
    Args:
        collection (chromadb.Collection): Collection vectorielle
        embedder (SentenceTransformer): Modèle d'embedding
        model (transformers.PreTrainedModel): Modèle de génération
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer associé
        query (str): Question ou requête de l'utilisateur
        max_context_length (int, optional): Longueur maximale du contexte. Defaults to 1000.
    
    Returns:
        str: Réponse générée avec contexte
    """
    logger.info("🕵️ Requête RAG détaillée")
    logger.info(f"🕵️ Question : {query}")
    
    # Embedding de la requête
    query_embedding = embedder.encode(query).tolist()
    logger.info("🕵️ Embedding de la requête :")
    logger.info(f"🕵️ Dimension : {len(query_embedding)}")
    
    # Recherche de contexte
    logger.info("🕵️ Recherche de contexte :")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Récupérer les 3 segments les plus pertinents
    )
    
    # Extraction des segments pertinents
    context_segments = results['documents'][0]
    logger.info(f"🕵️ Segments trouvés : {len(context_segments)}")
    
    # Analyse du contexte
    context = " ".join(context_segments)
    logger.info("🕵️ Analyse du contexte :")
    logger.info(f"🕵️ Longueur : {len(context)} caractères")
    logger.info(f"🕵️ Début du contexte : {context[:100]}...")
    
    # Construction du prompt enrichi
    enriched_prompt = f"""Contexte: {context}

Question: {query}

Réponds de manière précise et détaillée en te basant uniquement sur le contexte fourni :"""
    
    logger.info("🕵️ Génération de réponse RAG :")
    logger.info("🕵️ Construction du prompt enrichi")
    logger.info(f"🕵️ Longueur du prompt : {len(enriched_prompt)} caractères")
    
    # Générer une réponse avec le contexte
    return generate_response(model, tokenizer, enriched_prompt, max_length=300)

def compare_responses(response_without_context, rag_response):
    """
    Compare les réponses générées sans et avec contexte RAG
    
    Args:
        response_without_context (str): Réponse générée sans contexte
        rag_response (str): Réponse générée avec contexte RAG
    """
    logger.info("\n🔍 Analyse comparative des réponses")
    
    # Calcul des métriques de base
    without_context_length = len(response_without_context)
    rag_response_length = len(rag_response)
    
    # Analyse de la pertinence
    logger.info(f"📏 Longueur de la réponse sans contexte : {without_context_length} caractères")
    logger.info(f"📏 Longueur de la réponse avec contexte RAG : {rag_response_length} caractères")
    
    # Comparaison qualitative
    if rag_response_length > without_context_length:
        logger.info("✅ La réponse RAG semble plus détaillée")
    elif rag_response_length < without_context_length:
        logger.info("🤔 La réponse RAG est plus concise que la réponse initiale")
    else:
        logger.info("🔄 Les réponses ont une longueur similaire")
    
    # Indication de la valeur ajoutée du RAG
    logger.info("\n💡 Valeur ajoutée du RAG :")
    logger.info("1. Contextualisation : Enrichissement de la réponse avec des informations spécifiques")
    logger.info("2. Précision : Réduction des hallucinations et des informations génériques")
    logger.info("3. Adaptabilité : Capacité à fournir des réponses basées sur des sources spécifiques")

def manual_step_validation(step_name, description=None):
    """
    Permet à l'utilisateur de valider manuellement le passage à l'étape suivante
    
    Args:
        step_name (str): Nom de l'étape en cours
        description (str, optional): Description supplémentaire de l'étape
    
    Returns:
        bool: True si l'utilisateur valide, False sinon
    """
    try:
        logger.info(f"\n🔍 Validation manuelle de l'étape : {step_name}")
        if description:
            logger.info(f"📝 Description : {description}")
        
        while True:
            user_input = input("❓ Voulez-vous continuer ? (o/n) : ").strip().lower()
            
            if user_input in ['o', 'oui', 'y', 'yes']:
                logger.info("✅ Étape validée par l'utilisateur")
                return True
            elif user_input in ['n', 'non', 'no']:
                logger.info("🛑 Étape annulée par l'utilisateur")
                return False
            else:
                logger.warning("❌ Réponse invalide. Veuillez répondre par 'o' ou 'n'.")
    
    except KeyboardInterrupt:
        logger.info("\n🚫 Interruption manuelle détectée")
        return False

def main():
    """
    Fonction principale orchestrant la démonstration LLM et RAG
    """
    # Initialisation et configuration du système
    logger.info("🚀 Initialisation du système de RAG avancé")
    logger.info("🔍 Préparation des composants essentiels : modèle LLM, RAG, et outils d'analyse")
    
    log_section_separator("Étape 1 : Chargement du modèle LLM")
    # Étape 1 : Chargement du modèle LLM
    logger.info("\n🤖 ÉTAPE 1 : CHARGEMENT DU MODÈLE DE LANGAGE")
    logger.info("💡 Logique : Initialiser un modèle de langage capable de comprendre et générer du texte.")
    logger.info("   - Sélection d'un modèle compact et performant")
    logger.info("   - Configuration optimisée pour les ressources limitées")
    model, tokenizer = load_llm()
    
    if not manual_step_validation(
        "Étape 1 : Chargement du modèle LLM", 
        description="Vérifiez les détails du modèle chargé : type, nombre de paramètres, tokens spéciaux"
    ):
        return
    
    log_section_separator("Étape 2 : Réponse LLM sans contexte")
    # Étape 2 : Réponse LLM sans contexte
    logger.info("\n🧠 ÉTAPE 2 : GÉNÉRATION DE RÉPONSE SANS CONTEXTE")
    logger.info("💡 Logique : Tester les capacités brutes du modèle sans information supplémentaire.")
    logger.info("   - Évaluer la compréhension générale du modèle")
    logger.info("   - Observer les limites et potentialités du modèle")
    prompt = "Explique moi la solution que LightRAG apporte et ses forces ?"
    response_without_context = generate_response(model, tokenizer, prompt)
    logger.info(f"Réponse sans contexte : {response_without_context}")
    
    if not manual_step_validation(
        "Étape 2 : Réponse LLM sans contexte", 
        description="Examinez la réponse générée sans contexte spécifique. Observez la qualité et la cohérence."
    ):
        return
    
    log_section_separator("Étape 3 : Configuration du système RAG")
    # Étape 3 : Configuration du système RAG
    logger.info("\n🌐 ÉTAPE 3 : CONFIGURATION DU SYSTÈME RAG")
    logger.info("💡 Logique : Préparer un système de Retrieval-Augmented Generation (RAG).")
    logger.info("   - Charger et indexer une source de connaissances")
    logger.info("   - Créer une base de données vectorielle pour la recherche sémantique")
    pdf_path = "LightRAG.pdf"
    rag_collection, embedder = setup_rag_system(pdf_path)
    
    if not manual_step_validation(
        "Étape 3 : Configuration du système RAG", 
        description="Vérifiez la configuration du système RAG : nombre de segments, modèle d'embedding"
    ):
        return
    
    log_section_separator("Étape 4 : Réponse LLM avec contexte RAG")
    # Étape 4 : Réponse LLM avec contexte RAG
    logger.info("\n🔬 ÉTAPE 4 : GÉNÉRATION DE RÉPONSE AVEC CONTEXTE RAG")
    logger.info("💡 Logique : Enrichir la génération de réponse avec des connaissances contextuelles.")
    logger.info("   - Rechercher des segments pertinents dans la base de connaissances")
    logger.info("   - Augmenter le prompt avec des informations contextuelles")
    rag_response = rag_query(rag_collection, embedder, model, tokenizer, prompt)
    logger.info(f"Réponse avec contexte RAG : {rag_response}")
    
    if not manual_step_validation(
        "Étape 4 : Réponse LLM avec contexte RAG", 
        description="Comparez la réponse avec contexte RAG à la réponse précédente. Notez les différences."
    ):
        return
    
    log_section_separator("Étape 5 : Analyse comparative")
    # Étape 5 : Analyse comparative
    logger.info("\n📊 ÉTAPE 5 : ANALYSE COMPARATIVE DES RÉPONSES")
    logger.info("💡 Logique : Comparer et évaluer les réponses générées.")
    logger.info("   - Mesurer l'impact du contexte sur la qualité de la réponse")
    logger.info("   - Identifier les améliorations apportées par le RAG")
    compare_responses(response_without_context, rag_response)
    
    log_section_separator("Conclusion : Démonstration RAG", symbol='*')
    logger.info("🌟 Résumé de la démonstration :")
    logger.info("   1. Modèle LLM : DistilGPT2")
    logger.info("   2. Système RAG : Indexation et recherche sémantique")
    logger.info("   3. Amélioration de la réponse : Contextualisation")
    
    log_section_separator("Merci !", symbol='~', length=40)
    logger.info("\n🏁 DÉMONSTRATION RAG TERMINÉE")
    logger.info("Merci d'avoir exploré les capacités de notre système RAG !")

if __name__ == "__main__":
    main()
