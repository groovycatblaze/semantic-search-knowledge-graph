import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="Semantic Search Demo")
st.title("Semantic Search (SentenceTransformers + FAISS)")

docs_default = [
    "Neural networks learn patterns from data.",
    "Cognitive science explores the human mind and behavior.",
    "Reinforcement learning trains agents to make sequential decisions.",
    "Knowledge graphs connect entities and relations for reasoning.",
    "Vector databases enable fast semantic similarity search.",
]

docs = st.text_area("Documents (one per line):", value="\n".join(docs_default)).splitlines()
query = st.text_input("Query", "How do machines learn from experience?")
k = st.slider("Top-K", 1, min(5, len(docs)), 3)

if st.button("Build index & Search"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(docs, convert_to_numpy=True)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    qv = model.encode([query], convert_to_numpy=True)
    D, I = index.search(qv, k)
    st.subheader("Results")
    for rank, (i, d) in enumerate(zip(I[0], D[0]), start=1):
        st.write(f"**#{rank}** {docs[i]}  — distance: {float(d):.4f}")
