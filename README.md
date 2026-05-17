# Conversational RAG Chatbot using LangChain LCEL, Groq & ChromaDB

## Overview

This project demonstrates how to build a **Conversational RAG (Retrieval-Augmented Generation)** application using:

* LangChain LCEL
* Groq LLMs
* ChromaDB Vector Store
* HuggingFace Embeddings
* Web-based document loading
* Chat History Memory

The chatbot:

* Loads data from a website
* Splits documents into chunks
* Converts text into embeddings
* Stores embeddings in ChromaDB
* Retrieves relevant context for user questions
* Maintains conversational memory using session history
* Generates context-aware answers using Groq LLM

---

# Tech Stack

* Python
* LangChain
* Groq API
* HuggingFace Embeddings
* ChromaDB
* BeautifulSoup4
* WebBaseLoader

---

# Project Workflow

```text
Website Data
     ↓
Document Loader
     ↓
Text Splitting
     ↓
Embeddings Generation
     ↓
Chroma Vector Database
     ↓
Retriever
     ↓
Conversational RAG Chain
     ↓
Groq LLM Response
```

---

# Features

* Conversational RAG pipeline
* Memory-enabled chatbot
* Session-based chat history
* LCEL-based chain building
* Website document ingestion
* Context-aware retrieval
* Fast and lightweight embedding model

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/NeeraoGhadge34/Conversational-Chatbot-With-History/tree/main
cd Conversational-Chatbot-With-History
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```bash
pip install langchain
pip install langchain-community
pip install langchain-groq
pip install langchain-huggingface
pip install chromadb
pip install beautifulsoup4
pip install sentence-transformers
pip install python-dotenv
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

---

# Loading Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
```

---

# Initialize Groq LLM

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")
```

---

# Load Website Data

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

text = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = text.load()
```

---

# Split Documents into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)
```

---

# Create Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

---

# Store Embeddings in ChromaDB

```python
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

---

# Create Retriever

```python
retriever = db.as_retriever()
```

---

# Conversational RAG using LCEL

The project uses:

* Question rewriting
* Context retrieval
* Chat history memory
* Session-based conversations

Main components:

* `RunnableWithMessageHistory`
* `MessagesPlaceholder`
* `RunnablePassthrough`
* `StrOutputParser`

---

# Memory Management

The chatbot maintains chat history using:

```python
RunnableWithMessageHistory
```

This enables:

* Multi-turn conversations
* Context-aware follow-up questions
* Session-specific memory

---

# Example Questions

```text
What is Task Decomposition?
```

```text
What are common ways of doing it?
```

```text
Explain Self-Reflection in AI Agents.
```

---

# Sample Use Cases

* AI-powered chatbots
* Research assistants
* Document Q&A systems
* Conversational knowledge retrieval
* Educational AI tools

---

# Learning Concepts Covered

This project helps understand:

* RAG Architecture
* Vector Databases
* Embeddings
* Retrieval Pipelines
* Conversational AI
* LangChain LCEL
* Memory Handling
* LLM Integration

---

# Author

Neerao Ghadge

Aspiring Data Scientist | Machine Learning & Generative AI Enthusiast
