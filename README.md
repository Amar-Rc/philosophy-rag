# Vivekananda RAG

This project provides a Retrieval-Augmented Generation (RAG) system to answer questions based on the complete works of Swami Vivekananda.

## Features

- **Chat Interface:** A simple and intuitive Streamlit UI to interact with the RAG system.
- **Dual Backend:** Supports both Ollama and MLX for generation, with automatic detection of your system.
- **Model Selection:** Choose between a "Fast" and a "Deep" retrieval model for a trade-off between speed and accuracy.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- **For non-macOS users:** [Ollama](https://ollama.ai/) must be installed and running.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vivekananda-rag-github
```

### 2. Run the Setup Script

This will create a virtual environment and install all the necessary dependencies.

```bash
cd streamlit_app
bash setup.sh
```

### 3. Launch the App

```bash
source venv/bin/activate
streamlit run app.py
```

## For Developers

The RAG system can also be run from the command line. The project is divided into two main components: `ollama_rag` and `mlx_rag`.

### Ollama Version (CLI)

1.  **Navigate to the Ollama directory:**
    ```bash
    cd ollama_rag
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the RAG system:**
    ```bash
    python ollama_rag_system.py
    ```

### MLX Version (CLI, for Apple Silicon)

1.  **Navigate to the MLX directory:**
    ```bash
    cd mlx_rag
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the RAG system:**
    ```bash
    python mlx_rag_system.py
    ```

## Project Structure

- `streamlit_app/`: The main Streamlit application.
  - `app.py`: The main script for the Streamlit app.
  - `setup.sh`: The setup script for the Streamlit app.
- `ollama_rag/`: The Ollama-based RAG system.
  - `ollama_rag_system.py`: The main script for the Ollama RAG system.
  - `setup.sh`: The setup script for the Ollama version.
- `mlx_rag/`: The MLX-based RAG system for Apple Silicon.
  - `mlx_rag_system.py`: The main script for the MLX RAG system.
  - `setup_mlx.sh`: The setup script for the MLX version.
- `data/`: Contains the FAISS indices for the models.
- `embeddings/`: Contains the embeddings for the models.
- `models/`: Contains the models.