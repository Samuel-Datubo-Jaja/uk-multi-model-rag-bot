---
title: Uk Building Regulations Bot
emoji: 📚
colorFrom: purple
colorTo: green
sdk: streamlit
sdk_version: 1.42.2
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


## 🚧 StructureGPT: AI Assistant for UK Building Regulations
A Multi-Model Retrieval-Augmented Generation (RAG) System Fine-Tuned for Domain-Specific Compliance Guidance

## 🔍 Overview
StructureGPT is a powerful, modular AI assistant designed to simplify access to UK Building Regulations. It compares three LLaMA-based language model configurations — including a LoRA fine-tuned model — in a shared RAG pipeline, enabling accurate, grounded, and efficient responses to complex regulatory queries.

## 🚀 Features
### ✅ Multi-Model RAG System: Supports LLaMA-3.3-70B, LLaMA3-8B via GROQ API, and a fine-tuned LLaMA-3.1-8B.
# ✅ Unified Streamlit Interface: Seamlessly switch between models and explore real-time comparisons.
✅ Hybrid Retrieval: 70% vector search (ChromaDB + all-mpnet-base-v2) + 30% BM25 keyword matching.
✅ Fine-Tuning with LoRA + Quantization: Efficient domain adaptation with 3,000+ curated Q&A pairs.
✅ Fully Transparent Output: Source attribution, response times, and performance metrics included.

## 📷 Interface Snapshot
![image](https://github.com/user-attachments/assets/bdae8ebc-1939-4c36-bacf-8e41ada9a7e2)
![image](https://github.com/user-attachments/assets/493a24e8-f178-4cf3-a28d-d2d3e4ef547b)



🧪 Try It Live:
👉 StructureGPT on Hugging Face Spaces

## ⚙️ How It Works
graph TD;
    A[📄 Official GOV.UK PDFs] --> B[🧠 Content Extraction<br>(Unstructured API with YOLOX)]
    B --> C[🧩 Paragraph-Based Chunking<br>+ Embedding (MPNet)]
    C --> D[🔎 Hybrid Retrieval Engine<br>(70% Vector Similarity + 30% BM25)]
    D --> E[💬 Unified Chat Interface<br>(Streamlit + LangChain)]
    E --> F[🧠 Model Selector:<br>LLaMA-3.3-70B (GROQ)<br>LLaMA3-8B (GROQ)<br>Fine-Tuned LLaMA3.1-8B]


## 📦 Stack & Tools
💬 LLMs: Meta’s LLaMA-3 series (via GROQ API & Hugging Face)

🔍 RAG Stack: LangChain, ChromaDB, BM25, Sentence Transformers

🧠 Fine-Tuning: LoRA (Rank=16, Alpha=32), 8-bit Quantization

🖥️ Deployment: Hugging Face Spaces + Streamlit

🧪 Evaluation: RAGAS, Giskard Toolkit

📊 Tracking: Weights & Biases (wandb)    

