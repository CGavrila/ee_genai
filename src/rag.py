from typing import List, Tuple
from pymilvus import MilvusClient, utility, connections
import numpy as np
import pandas as pd

import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

COLLECTION_NAME = "ee_collection"
EMBEDDING_DIMENSION = 384
EMBEDDING_MAX_TOKENS = 256
MODEL_PATH = "./resources/llama-2-7b-chat.Q4_K_M.gguf"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize clients and models
client = MilvusClient("./resources/milvus_exercise.db")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
nltk.download('punkt_tab')

def generate_embeddings(text: str) -> List[Tuple[str, np.ndarray]]:
    """
    Generate embeddings for a given text. 
    
    It splits the text into sentences, and then build the chunks based on each sentence's token length.

    TODO:
    - Add the title as well, if available.
    """
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    for sentence in sent_tokenize(text):
        sentence_tokens = embedding_tokenizer.tokenize(sentence)
        logging.info(f"Chunking {len(sentence_tokens)} tokens, sentence: {sentence[:100]}...")
        if len(sentence_tokens) + current_chunk_tokens > EMBEDDING_MAX_TOKENS:
            chunks.append(current_chunk)
            current_chunk = ""
            current_chunk_tokens = 0
        current_chunk += sentence
        current_chunk_tokens += len(sentence_tokens)

    if current_chunk:
        chunks.append(current_chunk)

    logging.info(f"Generated {len(chunks)} chunks for {text[:100]}...")

    return [(chunk, embedding_model.encode(chunk)) for chunk in chunks]

def create_collection():
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIMENSION
    )
    text_csv = pd.read_csv("./resources/ee_case_studies.csv")
    docs = text_csv['text']
    
    embeddings: List[Tuple[int, str, np.ndarray]] = []
    for doc_id, doc in enumerate(docs):
        doc_embeddings = generate_embeddings(doc)
        embeddings.extend((doc_id, chunk, embedding) for chunk, embedding in doc_embeddings)
    logging.info(f"Generated {len(embeddings)} embeddings")

    data = [{
        "id": i,
        "doc_id": embeddings[i][0],
        "text": embeddings[i][1],
        "vector": embeddings[i][2],
        "subject": "equal experts"
    } for i in range(len(embeddings))]

    res = client.insert(collection_name=COLLECTION_NAME, data=data)

    return res

def query_rag(query: str, verbose=True):
    search_vector = generate_embeddings(query)[0][1]
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    res = client.search(COLLECTION_NAME, [search_vector], output_fields=['id', 'doc_id', 'vector', 'text', 'subject'],
                        params=search_params, limit=3)
    if verbose:
        for doc in res[0]:
            logging.info(f"id {doc['id']} doc_id {doc['entity']['doc_id']} distance {doc['distance']} {doc['entity']['text']} ")
    return res

def delete_collection():
    connections.connect("default", uri="./resources/milvus_exercise.db")
    utility.drop_collection(COLLECTION_NAME)

def init_collection():
    delete_collection()
    create_collection()

def some_test_queries():
    logging.info("Running test queries")
    query_rag("Did EE do any work with SpaceX?")

if __name__ == "__main__":
    init_collection()
    # some_test_queries()
