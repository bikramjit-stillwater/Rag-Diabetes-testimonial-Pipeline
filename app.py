from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# -----------------------------
# Gemini setup (keep key for now)
# -----------------------------
genai.configure(api_key="AIzaSyBwdKvlWsxyj1ZObCNZiKLA_oSe-YKj-3U")
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# 🔥 GLOBALS (IMPORTANT)
# -----------------------------
df = None
documents = None
embed_model = None
index = None

# -----------------------------
# 🔥 LAZY LOADING (FULL FIX)
# -----------------------------
def load_data_and_models():
    global df, documents, embed_model, index

    if df is None:
        print("🔄 Loading everything...")

        # Load CSV
        df_local = pd.read_csv("diabetes_testimonials_only.csv")
        df_local = df_local[["title", "url", "transcript"]].copy()

        df_local["title"] = df_local["title"].fillna("").astype(str).str.strip()
        df_local["url"] = df_local["url"].fillna("").astype(str).str.strip()
        df_local["transcript"] = df_local["transcript"].fillna("").astype(str).str.strip()

        df_local = df_local[df_local["transcript"] != ""].reset_index(drop=True)

        # Build documents
        docs = []
        for i, row in df_local.iterrows():
            doc_text = f"""TITLE: {row['title']}
URL: {row['url']}
TRANSCRIPT:
{row['transcript']}"""

            docs.append({
                "doc_id": i,
                "title": row["title"],
                "url": row["url"],
                "text": doc_text
            })

        # Load embedding model
        embed = SentenceTransformer("all-MiniLM-L6-v2")

        # Create embeddings
        doc_texts = [d["text"] for d in docs]
        doc_embeddings = embed.encode(
            doc_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Create FAISS index
        dim = doc_embeddings.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(doc_embeddings.astype("float32"))

        # Save globally
        df = df_local
        documents = docs
        embed_model = embed
        index = idx

        print("✅ Data + Model + FAISS ready")

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=3):
    load_data_and_models()

    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx_val in zip(scores[0], indices[0]):
        if idx_val == -1:
            continue
        item = documents[idx_val].copy()
        item["score"] = float(score)
        results.append(item)

    return results

# -----------------------------
# RAG
# -----------------------------
def ask_rag(query, top_k=3):
    results = retrieve(query, top_k=top_k)

    context = "\n\n".join([
        f"""SOURCE {i+1}
TITLE: {r['title']}
URL: {r['url']}
CONTENT:
{r['text']}"""
        for i, r in enumerate(results)
    ])

    prompt = f"""
You are a testimonial-based assistant.

Rules:
1. Answer only from the provided testimonial context.
2. If the answer is not clearly present, say: "Not found clearly in the testimonials."
3. Do not give medical advice.
4. Mention relevant source title and URL.
5. Keep the answer clear and short.

User question:
{query}

Context:
{context}
"""

    response = model.generate_content(prompt)

    return {
        "query": query,
        "answer": response.text,
        "sources": [
            {"title": r["title"], "url": r["url"], "score": r["score"]}
            for r in results
        ]
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    result = ask_rag(query)
    return jsonify(result)

# -----------------------------
# Run (local only)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)