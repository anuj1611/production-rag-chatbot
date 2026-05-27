"""
Test retrieval pipeline without calling any external LLM or embedding APIs.

This script:
- connects to local Qdrant (uses QDRANT_URL from .env)
- scrolls stored points and loads documents
- builds a BM25 retriever over the loaded documents
- runs a sample query and prints BM25 results
- prints Qdrant collection info (points_count)

Run:
cd backend
python test_retrieval.py
"""
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "no_l_data")

print(f"Connecting to Qdrant at: {QDRANT_URL}")
client = QdrantClient(url=QDRANT_URL)

# 1) show collection info (if exists)
try:
    info = client.get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}' info: points_count={info.points_count}")
except Exception as e:
    print(f"Warning: could not get collection info: {e}")

# 2) load all chunks via scroll
chunks = []
offset = None
while True:
    try:
        records, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
        )
    except Exception as e:
        print(f"Error while scrolling collection: {e}")
        break

    for record in records:
        payload = record.payload or {}
        page_content = payload.get("page_content", "")
        if page_content and page_content.strip():
            chunks.append({
                "page_content": page_content.strip(),
                "metadata": payload.get("metadata", {}),
            })

    if not next_offset:
        break
    offset = next_offset

print(f"Loaded {len(chunks)} chunks from Qdrant (collection='{COLLECTION}')")

# 3) build BM25 retriever and run a test query
if not chunks:
    print("No chunks found — ensure you ingested data or loaded the snapshot.")
    raise SystemExit(1)

docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in chunks]
print("Building BM25 retriever (this is lexical search only)...")
bm25 = BM25Retriever.from_documents(docs, bm25_impl="BM25Plus")
bm25.k = 5

query = "What is the organization and who runs it?"
print(f"\nRunning BM25 query: {query}\n")
results = bm25.invoke(query)
for i, doc in enumerate(results[:10], 1):
    snippet = doc.page_content[:300].replace('\n', ' ')
    source = doc.metadata.get('source_file', 'unknown')
    print(f"{i}. Source: {source} — {snippet[:200]}...")

print("\nRetrieval test complete. No LLM/embedding calls were made.")
