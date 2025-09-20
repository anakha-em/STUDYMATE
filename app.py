import fitz
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
import requests

# 🔹 Load Hugging Face Token
load_dotenv(".env")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("❌ HF_TOKEN not found. Please set it in your .env file.")

# Hugging Face API Setup
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

st.title("📘 StudyMate - AI-Powered Study Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload your academic PDF", type=["pdf"])

if uploaded_file is not None:
    # Extract text
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text += page.get_text()

    # ✅ Improved Chunking (word-based, avoids cutting mid-sentence)
    def chunk_text(text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    chunks = chunk_text(text)

    # Embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    # FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.success(f"✅ PDF processed into {len(chunks)} chunks and indexed successfully!")

    # Ask question
    query = st.text_input("Ask a question from the PDF:")
    if query:
        query_vector = model.encode([query])
        distances, indices = index.search(np.array(query_vector), k=3)

        # Retrieve top chunks and join into full context
        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = " ".join(retrieved_chunks)

        # Prompt for Hugging Face
        prompt = f"""
        You are a helpful study assistant. 
        Use the following context from a textbook or research paper to answer the question in detail.

        Context:
        {context}

        Question: {query}
        """

        # Send request to Hugging Face Inference API
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 400, "temperature": 0.2}}
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            generated_text = result[0]["generated_text"] if isinstance(result, list) else result
        else:
            generated_text = f"❌ Error: {response.status_code} - {response.text}"

        # ✅ Show Answer
        st.subheader("Answer:")
        st.write(generated_text)

        # ✅ Show References (cleaner)
        st.subheader("Sources from PDF:")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**Source {i}:** {chunk[:600]}{'...' if len(chunk) > 600 else ''}")
            st.write("---")

