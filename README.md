📘 StudyMate - Your AI-Powered Study Assistant
StudyMate is a powerful Streamlit application designed to help you study more effectively. It allows you to upload an academic PDF and then ask questions directly about its content. Using a robust Retrieval-Augmented Generation (RAG) system, StudyMate finds the most relevant information within your document to provide accurate, detailed answers.

✨ Key Features
PDF Processing: Seamlessly extracts text from any academic PDF.

Intelligent Chunking: Breaks down the document into optimized, manageable chunks for efficient information retrieval.

Vector Search (FAISS): Utilizes a high-performance FAISS index to quickly pinpoint the most relevant sections of the PDF for your query.

Contextual Q&A: Generates answers by using a Hugging Face LLM (Mistral-8x7B-Instruct) combined with the retrieved context from your PDF. This ensures the answers are highly relevant and accurate, and prevents the model from "hallucinating" or making things up.

Source Citation: Provides direct references to the specific sections of the PDF used to formulate the answer.

