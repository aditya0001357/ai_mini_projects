# Simple RAG Agent

A small learning project exploring how to build a **Retrieval-Augmented Generation (RAG) agent using LangChain, LangGraph, OpenAI, and ChromaDB**.

The project loads a PDF knowledge base, splits it into chunks, generates embeddings, stores them in ChromaDB, and exposes document retrieval as a tool that an LLM-powered LangGraph agent can use to answer user questions.

## 🧠 What This Project Demonstrates

- PDF document loading with `PyPDFLoader`
- Document chunking with `RecursiveCharacterTextSplitter`
- Text embeddings using OpenAI `text-embedding-3-small`
- Vector storage and similarity search with ChromaDB
- Creating a retriever from a vector store
- Creating a custom LangChain retrieval tool
- Tool calling with `gpt-4o-mini`
- Building an agent workflow using LangGraph
- Managing agent state with `AgentState`
- Conditional routing based on LLM tool calls
- Grounding responses in retrieved document context

## 🔄 Architecture

```text
                    User Question
                         │
                         ▼
                  ┌─────────────┐
                  │     LLM     │
                  │ gpt-4o-mini │
                  └──────┬──────┘
                         │
                   Tool requested?
                    ┌────┴────┐
                   YES        NO
                    │          │
                    ▼          ▼
             ┌────────────┐   END
             │ Retriever  │
             │   Tool     │
             └─────┬──────┘
                   │
                   ▼
               ChromaDB
                   │
             Similarity Search
                   │
                   ▼
            Relevant Chunks
                   │
                   ▼
                  LLM
                   │
                   ▼
              Final Answer
📚 RAG Pipeline

The document processing pipeline follows:

PDF
 │
 ▼
PyPDFLoader
 │
 ▼
Document Pages
 │
 ▼
RecursiveCharacterTextSplitter
 │
 ▼
Document Chunks
 │
 ▼
OpenAI Embeddings
 │
 ▼
ChromaDB

When a user asks a question:

User Question
     │
     ▼
LLM
     │
     ▼
Retriever Tool
     │
     ▼
Similarity Search
     │
     ▼
Top 5 Relevant Chunks
     │
     ▼
LLM
     │
     ▼
Final Answer
🛠️ Tech Stack
Technology	Purpose
Python	Programming language
LangChain	LLM and RAG components
LangGraph	Agent workflow and state management
OpenAI GPT-4o-mini	LLM
OpenAI text-embedding-3-small	Text embeddings
ChromaDB	Vector database
PyPDF	PDF processing
python-dotenv	Environment variable management
📁 Project Structure
simple_RAG/
│
├── langgraph_RAG_agent.py
├── rag_knowledge_base.pdf
├── requirements.txt
├── .env
│
└── chroma_docs/
    └── ChromaDB data

.env and generated ChromaDB data should not be committed to GitHub.

⚙️ Environment Variables

Create a .env file in the project directory:

OPENAI_API_KEY=your_api_key_here

Make sure .env is included in .gitignore.

▶️ Running the Project

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Place the PDF knowledge base in the project directory with the following name:

rag_knowledge_base.pdf

Run the application:

python langgraph_RAG_agent.py

You can then ask questions about the information contained in the knowledge base.

To stop the application, type:

exit

or:

quit