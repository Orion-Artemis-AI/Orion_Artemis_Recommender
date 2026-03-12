from llama_cpp import Llama
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

model_path = r"D:\AITF_26_Model\kalm-embedding-multilingual-mini-instruct-v2.5-q8_0.gguf"

# model = SentenceTransformer(
#     "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
#     trust_remote_code=True,
#     model_kwargs={
#         "torch_dtype": torch.bfloat16,
#         "attn_implementation": "flash_attention_2",  # Optional
#     },
# )
# model.max_seq_length = 512

# queries = [
#     "What is the capital of China?",
#     "Explain gravity",
# ]
# documents = [
#     "The capital of China is Beijing.",
#     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
# ]

# query_embeddings = model.encode_query(queries)
# document_embeddings = model.encode_document(documents)

# similarities = model.similarity(query_embeddings, document_embeddings)
# print(similarities)

llm = Llama(
    model_path=model_path,
    embedding=True,
    flash_attn=True,
    n_ctx=512,
    n_threads=6,
    n_gpu_layers=0,
)

queries = [
    "What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

def embed(text):
    return np.array(llm.create_embedding(text)["data"][0]["embedding"])

query_embeddings = np.vstack([embed(q) for q in queries])
doc_embeddings = np.vstack([embed(d) for d in documents])

# normalize vectors
query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

# cosine similarity
similarities = np.matmul(query_embeddings, doc_embeddings.T)

print(similarities)