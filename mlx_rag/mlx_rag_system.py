import faiss
import numpy as np
import pandas as pd
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import os
import json

class VivekanandaRAGSystem:
    """
    Native Apple Silicon RAG system using MLX.
    Aligned with the Ollama implementation structure.
    """
    
    def __init__(self, model_name='e5-large-v2'):
        self.model_name = model_name
        self.data_path = Path(__file__).parent.parent / "data"
        self.encoder = None
        self.index = None
        self.chunks = None
        self.metadata = None
        
        # MLX Specifics
        self.model = None
        self.tokenizer = None
        self.model_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"
        
        print("🚀 Initializing Vivekananda RAG System (MLX Edition)...")

    def build_index(self):
        """Build FAISS index from embeddings on SSD."""
        print(f"\n📊 Building FAISS index from {self.model_name}...")
        
        # Consistent pathing as per your T7 structure
        embeddings_path = self.data_path.parent / f"embeddings/embeddings_{self.model_name}/"
        df = pd.read_parquet(embeddings_path)
        
        print(f"   Loaded {len(df)} chunks")
        
        # Memory-efficient conversion for 8GB RAM
        embeddings = np.stack(df['embedding'].values).astype('float32')
        
        self.chunks = df['chunk_content'].tolist()
        self.metadata = df[[c for c in df.columns if c != 'embedding']].to_dict('records')
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        self.index = index
        print(f"   ✅ Index built: {index.ntotal} vectors")
        
        self._save_index()

    def _save_index(self):
        """Save index and metadata to disk"""
        index_dir = self.data_path / "faiss_indices"
        index_dir.mkdir(exist_ok=True)
        
        faiss.write_index(self.index, str(index_dir / f"{self.model_name}.index"))
        
        with open(index_dir / f"{self.model_name}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(index_dir / f"{self.model_name}_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"   💾 Index saved to {index_dir}")

    def load_index(self):
        """Load pre-built index and local SentenceTransformer encoder"""
        print(f"\n📂 Loading index: {self.model_name}...")
        
        index_dir = self.data_path / "faiss_indices"
        self.index = faiss.read_index(str(index_dir / f"{self.model_name}.index"))
        
        with open(index_dir / f"{self.model_name}_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        with open(index_dir / f"{self.model_name}_metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Use the local path on your SSD to avoid 401 errors
        current_model_tag = getattr(self, 'current_model_name', None)
        
        if self.encoder is None or current_model_tag != self.model_name:
            # For chunk experiments (e.g., bge-small-v1.5_fine_grained), 
            # we only need the base model folder 'bge-small-v1.5'
            base_model = self.model_name.split('_')[0]
            local_model_path = self.data_path.parent / "models" / base_model
            
            print(f"   🔄 Swapping encoder to: {base_model}")
            self.encoder = SentenceTransformer(str(local_model_path))
            self.current_model_name = self.model_name # Track current state
        
        print(f"   ✅ Loaded {self.index.ntotal} vectors (Dim: {self.index.d})")

    def search(self, query, top_k=5):
        """Search for relevant chunks"""
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'rank': i + 1,
                'similarity': float(distances[0][i]),
                'text': self.chunks[idx],
                'metadata': self.metadata[idx]
            })
        return results

    def answer(self, question, top_k=5, max_tokens=1024):
        """Full RAG pipeline: Retrieve + Generate (MLX Native)"""
        results = self.search(question, top_k)
        
        context = "\n\n".join([
            f"[Source {i+1}]: {r['text']}"
            for i, r in enumerate(results)
        ])
        
        answer = self._generate_with_mlx(question, context, max_tokens)
        followup_questions = self._generate_followup_questions(answer, context)
        
        return {
            'answer': answer,
            'sources': results,
            'followup_questions': followup_questions
        }

    def _generate_with_mlx(self, question, context, max_tokens=1024):
        """Native MLX inference for Llama-3.2"""
        if self.model is None:
            print(f"⏳ Loading MLX model into Unified Memory...")
            self.model, self.tokenizer = load(self.model_id)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert on Swami Vivekananda's philosophy.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Context: {context}
        Question: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)

    def _generate_followup_questions(self, answer, context):
        """Generate followup questions based on the answer and context."""
        if self.model is None:
            print(f"⏳ Loading MLX model into Unified Memory...")
            self.model, self.tokenizer = load(self.model_id)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert on Swami Vivekananda's philosophy.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Based on the following answer and context, generate 3 relevant follow-up questions that a user might ask.

        Context:
        {context}

        Answer:
        {answer}

        Return the questions as a JSON list of strings. For example:
        ["What is the role of a Guru?", "How does Karma affect my life?", "What is the nature of the Atman?"]
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        response_text = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=100)
        
        try:
            # Extract the JSON list from the response
            start_index = response_text.find('[')
            end_index = response_text.rfind(']') + 1
            json_str = response_text[start_index:end_index]
            
            return json.loads(json_str)
        except (json.JSONDecodeError) as e:
            print(f"An error occurred while generating follow-up questions: {e}")
            return []

if __name__ == "__main__":
    # 1. First run: Build index
    # rag = VivekanandaRAGSystem(model_name='e5-large-v2')
    # rag.build_index()
    
    # 2. Subsequent runs: Load and Answer
    rag = VivekanandaRAGSystem(model_name='e5-large-v2')
    rag.load_index()
    response = rag.answer("What is the nature of the Atman?")
    
    print(f"\n💡 Answer: {response['answer']}")