from sentence_transformers import SentenceTransformer

model_embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chain(chain):
    return model_embedder.encode(chain)
