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
✅ **Multi-Model RAG System**: Supports LLaMA-3.3-70B, LLaMA3-8B via GROQ API, and a fine-tuned LLaMA-3.1-8B.  <br>
✅ **Unified Streamlit Interface**: Seamlessly switch between models and explore real-time comparisons.  <br>
✅ **Hybrid Retrieval**: 70% vector search (ChromaDB + all-mpnet-base-v2) + 30% BM25 keyword matching.  <br>
✅ **Fine-Tuning with LoRA + Quantization**: Efficient domain adaptation with 3,000 curated Q&A pairs.  <br>
✅ **Fully Transparent Output**: Source attribution, response times, and performance metrics included.

## Workflow
<img width="701" alt="image" src="https://github.com/user-attachments/assets/194f1835-aa3b-4f33-ac78-a6e9ebcf3a9a" />

## 📷 Interface Snapshot
![image](https://github.com/user-attachments/assets/bdae8ebc-1939-4c36-bacf-8e41ada9a7e2)

![image](https://github.com/user-attachments/assets/493a24e8-f178-4cf3-a28d-d2d3e4ef547b)

🧪 Try It Live: [Active Link will be available soon, for now just run it locally]
👉 StructureGPT on Hugging Face Spaces

## 📦 Stack & Tools  
💬 **LLMs**: Meta’s LLaMA-3 series (via GROQ API & Hugging Face)  <br>
🔍 **RAG Stack**: LangChain, ChromaDB, BM25, Sentence Transformers  <br>
🧠 **Fine-Tuning**: LoRA (Rank=16, Alpha=32), 8-bit Quantization  <br>
🖥️ **Deployment**: Hugging Face Spaces + Streamlit  <br>
🧪 **Evaluation**: RAGAS, Giskard Toolkit  <br>
📊 **Tracking**: Weights & Biases (wandb)

