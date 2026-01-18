#!/usr/bin/env python3
"""
Nu1lm RAG (Retrieval Augmented Generation)

Give Nu1lm unlimited knowledge through smart retrieval:
1. Store your knowledge in a vector database
2. When user asks a question, find relevant documents
3. Feed those documents to Nu1lm along with the question
4. Nu1lm generates answer based on retrieved knowledge

Nu1lm + RAG can outperform GPT-4 on YOUR specific domain!
"""

import argparse
from pathlib import Path
from typing import List, Optional
import json
import numpy as np

# For embeddings and vector storage
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimpleVectorStore:
    """Simple in-memory vector store. For production, use FAISS, Chroma, etc."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("pip install sentence-transformers")
        self.encoder = SentenceTransformer(embedding_model)
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[dict] = []

    def add_documents(self, documents: List[str], metadata: List[dict] = None):
        """Add documents to the store."""
        new_embeddings = self.encoder.encode(documents, show_progress_bar=True)

        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")

    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """Search for similar documents."""
        if self.embeddings is None:
            return []

        query_embedding = self.encoder.encode([query])[0]

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx]), self.metadata[idx]))

        return results

    def save(self, path: Path):
        """Save vector store to disk."""
        np.savez(
            path,
            embeddings=self.embeddings,
            documents=np.array(self.documents, dtype=object),
            metadata=np.array([json.dumps(m) for m in self.metadata], dtype=object),
        )
        print(f"Saved to {path}")

    def load(self, path: Path):
        """Load vector store from disk."""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.documents = data["documents"].tolist()
        self.metadata = [json.loads(m) for m in data["metadata"]]
        print(f"Loaded {len(self.documents)} documents")


class RAGSystem:
    """RAG-enhanced language model."""

    def __init__(self, model_path: Path, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for stability
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, query: str, top_k: int = 3, max_tokens: int = 512) -> str:
        """Generate answer using RAG."""
        # Retrieve relevant documents
        results = self.vector_store.search(query, top_k=top_k)

        # Build context
        context_parts = []
        for doc, score, meta in results:
            source = meta.get("source", "Unknown")
            context_parts.append(f"[Source: {source}]\n{doc}")

        context = "\n\n---\n\n".join(context_parts)

        # Build prompt
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question accurately.

### Context:
{context}

### Question:
{query}

### Answer:"""

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return response.strip()


def index_documents(docs_dir: Path, output_path: Path):
    """Index documents from a directory."""
    vector_store = SimpleVectorStore()

    documents = []
    metadata = []

    # Support txt and md files
    for ext in ["*.txt", "*.md"]:
        for file_path in docs_dir.glob(f"**/{ext}"):
            content = file_path.read_text()
            # Split into chunks (simple chunking)
            chunks = [content[i:i+500] for i in range(0, len(content), 400)]  # 100 char overlap
            for chunk in chunks:
                if chunk.strip():
                    documents.append(chunk)
                    metadata.append({"source": str(file_path.name)})

    if documents:
        vector_store.add_documents(documents, metadata)
        vector_store.save(output_path)
    else:
        print(f"No documents found in {docs_dir}")


def chat_with_rag(model_path: Path, index_path: Path):
    """Interactive chat with RAG."""
    vector_store = SimpleVectorStore()
    vector_store.load(index_path)

    rag = RAGSystem(model_path, vector_store)

    print("\n" + "="*50)
    print("Nu1lm RAG Chat (type 'quit' to exit)")
    print("="*50 + "\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue

        response = rag.generate(query)
        print(f"Nu1lm: {response}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--docs", type=Path, required=True, help="Documents directory")
    index_parser.add_argument("--output", type=Path, default=Path("data/index.npz"), help="Output index")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with RAG")
    chat_parser.add_argument("--model", type=Path, required=True, help="Model path")
    chat_parser.add_argument("--index", type=Path, required=True, help="Index path")

    args = parser.parse_args()

    if args.command == "index":
        index_documents(args.docs, args.output)
    elif args.command == "chat":
        chat_with_rag(args.model, args.index)
    else:
        parser.print_help()
