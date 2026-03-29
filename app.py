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
genai.configure(api_key="AIzaSyAun80ANH3ykTZpl8oUhbtSE7JJ_qt-HIY")
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# Load CSV
# -----------------------------
csv_path = "diabetes_testimonials_only.csv"

df = pd.read_csv(csv_path)
df = df[["title", "url", "transcript"]].copy()

df["title"] = df["title"].fillna("").astype(str).str.strip()
df["url"] = df["url"].fillna("").astype(str).str.strip()
df["transcript"] = df["transcript"].fillna("").astype(str).str.strip()

df = df[df["transcript"] != ""].reset_index(drop=True)

# -----------------------------
# Create text file (optional)
# -----------------------------
all_docs = []

for i, row in df.iterrows():
    block = f"""TESTIMONIAL_ID: {i}
TITLE: {row['title']}
URL: {row['url']}
TRANSCRIPT:
{row['transcript']}

{'='*100}
"""
    all_docs.append(block)

with open("all_patient_testimonials.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_docs))

# -----------------------------
# Build documents
# -----------------------------
documents = []

for i, row in df.iterrows():
    doc_text = f"""TITLE: {row['title']}
URL: {row['url']}
TRANSCRIPT:
{row['transcript']}"""

    documents.append({
        "doc_id": i,
        "title": row["title"],
        "url": row["url"],
        "text": doc_text
    })

# -----------------------------
# 🔥 LAZY LOADING (IMPORTANT FIX)
# -----------------------------
embed_model = None
index = None

def load_models():
    global embed_model, index

    if embed_model is None:
        print("🔄 Loading embedding model...")

        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        doc_texts = [d["text"] for d in documents]
        doc_embeddings = embed_model.encode(
            doc_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = doc_embeddings.shape[1]
        index_local = faiss.IndexFlatIP(dimension)
        index_local.add(doc_embeddings.astype("float32"))

        index = index_local

        print("✅ Model + FAISS loaded")

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=3):
    load_models()  # 🔥 ensures model loads only when needed

    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = documents[idx].copy()
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

    result = ask_rag(query, top_k=3)
    return jsonify(result)

# -----------------------------
# Run app (for local only)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)