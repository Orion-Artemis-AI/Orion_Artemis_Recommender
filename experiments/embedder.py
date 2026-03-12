from sentence_transformers import SentenceTransformer
import torch
from llama_cpp import Llama
import numpy as np

# model_path = "D:\AITF_26_Model\kalm-embedding-multilingual-mini-instruct-v2.5-q8_0.gguf"

# model = SentenceTransformer(
#     model_path,
#     trust_remote_code=True,
#     model_kwargs={
#         "torch_dtype": torch.bfloat16,
#         "attn_implementation": "flash_attention_2",  # Optional
#     },
# )
# model.max_seq_length = 512

# sentences = ["This is an example sentence", "Each sentence is converted"]
# embeddings = model.encode(
#     sentences,
#     normalize_embeddings=True,
#     batch_size=256,
#     show_progress_bar=True,
# )
# print(embeddings)


model_path = r"D:\AITF_26_Model\kalm-embedding-multilingual-mini-instruct-v2.5-q8_0.gguf"

llm = Llama(
    model_path=model_path,
    embedding=True,
    flash_attn=True,
    n_ctx=512,
    n_threads=6,
    n_batch=512,
    n_gpu_layers=0  # >0 if GPU
)

sentences = [
    "This is an example sentence",
    "Each sentence is converted"
]

embeddings = []

for s in sentences:
    emb = llm.create_embedding(s)["data"][0]["embedding"]
    embeddings.append(emb)


# print("Embeddings: ", embeddings)


embeddings = np.array(embeddings)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print("Normalized Embeddings: ", embeddings)
