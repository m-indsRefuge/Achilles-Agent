# Achilles Agent

An AI-powered development agent with long-term memory for VS Code.

## Features
- **Semantic Memory**: Uses FAISS for lightning-fast retrieval of project context.
- **Autonomous Workflows**: Can plan and execute multi-step tasks (shell commands, file edits).
- **Inline Suggestions**: Context-aware code completions as you type.
- **Project Analyzer**: Deeply understands your codebase by indexing symbols and diagnostics.

## Getting Started

### Prerequisites
- [Ollama](https://ollama.com/) (running locally)
- Python 3.8+
- Node.js & NPM

### Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   npm install
   pip install -r requirements.txt
   ```
3. Pull the required models for Ollama:
   ```bash
   ollama pull codellama
   ollama pull nomic-embed-text
   ```

### Running & Testing
1. Open the project in VS Code.
2. Press **`F5`** to launch the **Extension Development Host**.
3. In the new window, use the **Achilles Sidebar** (Beaker icon) or the Command Palette (`Cmd/Ctrl+Shift+P`) to interact with the agent.

## License
MIT
