
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import requests
import json

class VivekanandaRAGSystem:
    """
    Production RAG system optimized for M1 MacBook.
    
    Design decisions:
    - FAISS for M1-optimized vector search
    - Lazy loading to manage 8GB RAM
    - External SSD for data storage
    """
    
    def __init__(self, model_name='e5-large-v2'):
        self.model_name = model_name
        self.data_path = Path(__file__).parent.parent / "data"
        self.encoder = None
        self.index = None
        self.chunks = None
        self.metadata = None
        
        print("🚀 Initializing Vivekananda RAG System...")
    
    def build_index(self):
        """
        Build FAISS index from embeddings.
        One-time operation, then save index to disk.
        """
        print(f"\n📊 Building FAISS index from {self.model_name}...")
        
        # Load embeddings from parquet
        embeddings_path = self.data_path.parent / f"embeddings/embeddings_{self.model_name}/"
        
        # Read parquet files
        df = pd.read_parquet(embeddings_path)
        
        print(f"   Loaded {len(df)} chunks")
        
        # Extract embeddings as numpy array
        embeddings = np.stack(df['embedding'].values).astype('float32')
        
        # Store metadata separately (more memory efficient)
        self.chunks = df['chunk_content'].tolist()
        self.metadata = df[[c for c in df.columns if c != 'embedding']].to_dict('records')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        # For M1, use IndexFlatIP (inner product, normalized vectors)
        # M1 has excellent SIMD support for this
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors
        index.add(embeddings)
        
        self.index = index
        
        print(f"   ✅ Index built: {index.ntotal} vectors, {dimension} dims")
        
        # Save index to disk
        self._save_index()
    
    def _save_index(self):
        """Save index and metadata to disk"""
        index_dir = self.data_path / "faiss_indices"
        index_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_dir / f"{self.model_name}.index"))
        
        # Save chunks and metadata
        with open(index_dir / f"{self.model_name}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(index_dir / f"{self.model_name}_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"   💾 Index saved to {index_dir}")
    
    def load_index(self):
        """Load pre-built index from disk (fast startup)"""
        print(f"\n📂 Loading index: {self.model_name}...")
        
        index_dir = self.data_path / "faiss_indices"
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_dir / f"{self.model_name}.index"))
        
        # Load chunks and metadata
        with open(index_dir / f"{self.model_name}_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        with open(index_dir / f"{self.model_name}_metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load encoder (lazy)
        if self.encoder is None:
            local_model_path = self.data_path.parent / "models" / self.model_name.split('_')[0]
        
            print(f"🔄 Loading local encoder from: {local_model_path}")
            
            # Load the model directly from the folder
            self.encoder = SentenceTransformer(str(local_model_path))
        
        print(f"   ✅ Loaded {self.index.ntotal} vectors")
    
    def search(self, query, top_k=5):
        """Search for relevant chunks"""
        
        # Encode query
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'rank': i + 1,
                'similarity': float(distances[0][i]),
                'text': self.chunks[idx],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def answer(self, question, top_k=5, llm_provider='ollama', max_tokens=1024):
        """Full RAG pipeline: Retrieve + Generate"""
        
        # 1. Retrieve
        results = self.search(question, top_k)
        
        # 2. Build context
        context = "\n\n".join([
            f"[Source {i+1}]: {r['text']}"
            for i, r in enumerate(results)
        ])
        
        # 3. Generate (using local LLM)
        if llm_provider == 'ollama':
            answer = self._generate_with_ollama(question, context, max_tokens)
            followup_questions = self._generate_followup_questions(answer, context)
        else:
            # This part is not used in the ollama_rag script
            # but kept for compatibility with the original script
            answer = "MLX provider not implemented in this version."
            followup_questions = []
        
        return {
            'answer': answer,
            'sources': results,
            'followup_questions': followup_questions
        }
    
    def _generate_with_ollama(self, question, context, max_tokens=1024):
        """Use Ollama for local inference via REST API"""
        
        prompt = f"""Based on Swami Vivekananda's teachings:

Context:
{context}

Question: {question}

Provide a thoughtful answer grounded in the context above."""

        try:
            # Check if Ollama server is running
            requests.get("http://localhost:11434")
        except requests.exceptions.ConnectionError:
            return "Ollama server not found. Please make sure Ollama is installed and running. You can download it from https://ollama.ai/"

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                },
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"

    def _generate_followup_questions(self, answer, context):
        """Generate followup questions based on the answer and context."""
        
        prompt = f"""Based on the following answer and context, generate 3 relevant follow-up questions that a user might ask.

Context:
{context}

Answer:
{answer}

Return the questions as a JSON list of strings. For example:
["What is the role of a Guru?", "How does Karma affect my life?", "What is the nature of the Atman?"]
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 100
                    }
                },
            )
            response.raise_for_status()
            
            # Extract the JSON list from the response
            response_text = response.json()["response"].strip()
            # Find the start and end of the JSON list
            start_index = response_text.find('[')
            end_index = response_text.rfind(']') + 1
            json_str = response_text[start_index:end_index]
            
            return json.loads(json_str)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"An error occurred while generating follow-up questions: {e}")
            return []

# Usage
if __name__ == "__main__":
    # To build index (one-time):
    # rag = VivekanandaRAGSystem(model_name='e5-large-v2')
    # rag.build_index()
    
    # Subsequent runs: just load
    rag = VivekanandaRAGSystem(model_name='e5-large-v2')
    rag.load_index()
    
    # Test
    response = rag.answer("What is the nature of the Atman?")
    print(response['answer'])