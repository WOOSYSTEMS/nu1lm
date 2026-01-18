#!/usr/bin/env python3
"""
Nu1lm Ask - Simple Q&A using RAG

Searches the microplastics knowledge base and returns relevant information.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.rag import SimpleVectorStore

BANNER = """
╔═══════════════════════════════════════════════╗
║           Nu1lm Microplastics Q&A             ║
║   Type your question, 'quit' to exit          ║
╚═══════════════════════════════════════════════╝
"""

def main():
    # Load knowledge base
    index_path = Path(__file__).parent / "data" / "microplastics_index.npz"

    if not index_path.exists():
        print("Knowledge base not found. Run:")
        print("  python scripts/rag.py index --docs training_data --output data/microplastics_index.npz")
        return

    vs = SimpleVectorStore()
    vs.load(index_path)

    print(BANNER)

    while True:
        try:
            query = input("\033[94mQuestion:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # Search knowledge base
        results = vs.search(query, top_k=2)

        print("\n\033[92mNu1lm:\033[0m")
        for doc, score, meta in results:
            if score > 0.3:  # Only show relevant results
                print(doc.strip())
                print()

        if not results or results[0][1] < 0.3:
            print("No relevant information found for that question.")
        print()


if __name__ == "__main__":
    # Check for command line question
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

        index_path = Path(__file__).parent / "data" / "microplastics_index.npz"
        vs = SimpleVectorStore()
        vs.load(index_path)

        results = vs.search(query, top_k=2)
        for doc, score, meta in results:
            if score > 0.3:
                print(doc.strip())
                print()
    else:
        main()
