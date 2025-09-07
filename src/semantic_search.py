from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Neural networks learn patterns from data.",
    "Cognitive science explores the human mind and behavior.",
    "Reinforcement learning trains agents to make sequential decisions.",
    "Knowledge graphs connect entities and relations for reasoning.",
    "Vector databases enable fast semantic similarity search.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(documents, convert_to_numpy=True)

index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

def search(q, k=3):
    qv = model.encode([q], convert_to_numpy=True)
    D, I = index.search(qv, k)
    return [(documents[i], float(D[0][j])) for j, i in enumerate(I[0])]

if __name__ == "__main__":
    q = "How do machines learn from experience?"
    print("Query:", q)
    for doc, dist in search(q):
        print(f"- {doc}  (dist={dist:.4f})")
