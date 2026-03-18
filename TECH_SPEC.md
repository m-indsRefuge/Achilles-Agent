# Achilles Agent: Technical Specification & Installation Guide

## 1. Overview
The Achilles Agent is an AI-powered development companion for VS Code. It features high-performance semantic memory, autonomous task execution, and project-specific fine-tuning capabilities.

## 2. Architecture
- **Frontend**: VS Code Extension (TypeScript)
  - Manages UI (Sidebar Webview, Inline Suggestions).
  - Interfaces with VS Code APIs (File System, Diagnostics, Symbols).
- **Bridge**: Persistent JSON-RPC Bridge (TS/Python)
  - Maintains a long-running Python process for low-latency ML operations.
- **Backend**: Python Memory Layer
  - **Knowledge Base**: FAISS-backed vector database for project-wide semantic search.
  - **Quick Recall**: FAISS-backed working memory for recent context.
  - **Short-Term Memory**: SQLite-backed conversation history with automated summarization.
- **AI Integration**: OpenAI-compatible client targeting local Ollama (CodeLlama).

## 3. Requirements
- **Hardware**: 16GB+ RAM recommended for local LLM execution.
- **Software**:
  - [Node.js](https://nodejs.org/) (v18+) & NPM.
  - [Python](https://www.python.org/) (3.8+).
  - [Ollama](https://ollama.com/) (Running locally).

## 4. Installation

### Step 1: Install Dependencies
```bash
# Install Extension dependencies
npm install

# Install Python Backend dependencies
pip install -r requirements.txt
```

### Step 2: Setup AI Models
Ensure Ollama is running and pull the default models:
```bash
ollama pull codellama
```
*Note: The current implementation uses `SentenceTransformer` (MiniLM) for embeddings and re-ranking, which will be automatically downloaded on first use.*

### Step 3: Build the Extension
```bash
npm run compile
```

## 5. Running & Manual Testing

1.  Open the project in VS Code.
2.  Press **F5** to launch the **[Extension Development Host]**.
3.  **Analyze Project**: Run `Achilles: Analyze Project` from the Command Palette (`Ctrl+Shift+P`).
4.  **Chat**: Open the Achilles sidebar (Beaker icon) and ask questions about your code.
5.  **Autonomous Workflow**: Run `Achilles: Run Autonomous Workflow` and provide a goal (e.g., "Refactor this file").
6.  **Code Review**: Open a file and run `Achilles: Review Current File`.

## 6. Automated Testing

### Python Backend Tests
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m pytest src/memory_layer/tests/
```

### TypeScript Frontend Tests
```bash
npm test
```

## 7. Configuration
Settings can be customized in VS Code via `File > Preferences > Settings` (search for "Achilles"):
- `ollama.baseURL`: Endpoint for AI services.
- `ollama.model`: Text generation model name.
- `ollama.embeddingModel`: Embedding model name.
- `search.topK`: Number of context results retrieved.

## 8. Safety Features
- **Human-in-the-Loop**: All shell commands and file edits proposed by the agent require manual user approval via a VS Code dialogue.
- **Data Privacy**: All code indexing and AI processing (via Ollama) stay local to your machine.
