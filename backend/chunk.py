from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


@dataclass(slots=True)
class IngestionConfig:
	md_root: Path
	collection_name: str
	vector_name: str
	embedding_model: str
	google_api_key: str | None
	chunk_size: int
	chunk_overlap: int
	upsert_batch_size: int
	upsert_max_retries: int
	upsert_batch_pause_seconds: int
	qdrant_url: str | None
	qdrant_api_key: str | None
	qdrant_timeout: int
	qdrant_local_path: Path
	dry_run: bool


class IngestionState(TypedDict, total=False):
	config: IngestionConfig
	documents: list[Document]
	chunks: list[Document]
	chunk_ids: list[str]
	file_count: int
	upserted_count: int
	qdrant_target: str


def discover_markdown_files(markdown_source: Path) -> list[Path]:
	if markdown_source.is_file():
		return [markdown_source] if markdown_source.suffix.lower() == ".md" else []
	return sorted(path for path in markdown_source.rglob("*.md") if path.is_file())


def load_markdown_documents(files: list[Path], markdown_source: Path) -> list[Document]:
	documents: list[Document] = []
	for file_path in files:
		text = file_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
		text = text.strip()
		if not text:
			continue

		if markdown_source.is_file():
			rel_path = file_path.name
		else:
			rel_path = file_path.relative_to(markdown_source).as_posix()
		documents.append(
			Document(
				page_content=text,
				metadata={
					"source_file": rel_path,
					"file_name": file_path.name,
				},
			)
		)
	return documents


def summarize_chunk_text(text: str, max_chars: int = 280, max_sentences: int = 2) -> str:
	"""Create a short extractive summary while preserving full chunk content elsewhere."""
	normalized = text.replace("\r\n", "\n")
	normalized = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", normalized)
	normalized = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", normalized)
	normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", normalized)
	normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
	normalized = re.sub(r"\s+", " ", normalized).strip()
	if not normalized:
		return ""

	sentences = re.split(r"(?<=[.!?])\s+", normalized)
	selected: list[str] = []
	for sentence in sentences:
		candidate = sentence.strip()
		if not candidate:
			continue
		selected.append(candidate)
		if len(selected) >= max_sentences or len(" ".join(selected)) >= max_chars:
			break

	summary = " ".join(selected).strip() or normalized
	if len(summary) > max_chars:
		trimmed = summary[:max_chars].rsplit(" ", 1)[0].strip()
		summary = trimmed if trimmed else summary[:max_chars].strip()
	if len(summary) < len(normalized) and not summary.endswith((".", "!", "?")):
		summary = f"{summary}..."
	return summary


def node_load_documents(state: IngestionState) -> IngestionState:
	config = state["config"]
	files = discover_markdown_files(config.md_root)
	documents = load_markdown_documents(files, config.md_root)
	return {
		"documents": documents,
		"file_count": len(files),
	}


def node_chunk_documents(state: IngestionState) -> IngestionState:
	config = state["config"]
	documents = state.get("documents", [])

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=config.chunk_size,
		chunk_overlap=config.chunk_overlap,
		separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
	)

	chunks: list[Document] = []
	for doc in documents:
		# Split per file to avoid mixing unrelated sections from different markdown files.
		file_chunks = splitter.split_documents([doc])
		total = len(file_chunks)
		for idx, chunk in enumerate(file_chunks):
			summary = summarize_chunk_text(chunk.page_content)
			chunk.metadata["chunk_index"] = idx
			chunk.metadata["chunk_total"] = total
			chunk.metadata["chunk_chars"] = len(chunk.page_content)
			chunk.metadata["description"] = summary
			chunks.append(chunk)

	return {"chunks": chunks}


def make_chunk_ids(chunks: list[Document]) -> list[str]:
	ids: list[str] = []
	for chunk in chunks:
		source = str(chunk.metadata.get("source_file", "unknown"))
		idx = str(chunk.metadata.get("chunk_index", "0"))
		payload = f"{source}|{idx}|{chunk.page_content.strip()}"
		digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
		ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, digest)))
	return ids


def get_qdrant_client(config: IngestionConfig) -> tuple[QdrantClient, str]:
	if config.qdrant_url:
		return (
			QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=config.qdrant_timeout),
			config.qdrant_url,
		)

	config.qdrant_local_path.mkdir(parents=True, exist_ok=True)
	return (
		QdrantClient(path=str(config.qdrant_local_path), timeout=config.qdrant_timeout),
		str(config.qdrant_local_path),
	)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, vector_name: str) -> None:
	if client.collection_exists(collection_name=collection_name):
		return

	if vector_name:
		vectors_config: dict[str, VectorParams] | VectorParams = {
			vector_name: VectorParams(size=vector_size, distance=Distance.COSINE)
		}
	else:
		vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)

	client.create_collection(
		collection_name=collection_name,
		vectors_config=vectors_config,
	)


def upsert_in_batches(
	vector_store: QdrantVectorStore,
	chunks: list[Document],
	chunk_ids: list[str],
	config: IngestionConfig,
) -> int:
	total = len(chunks)
	upserted = 0

	for start in range(0, total, config.upsert_batch_size):
		end = min(start + config.upsert_batch_size, total)
		batch_chunks = chunks[start:end]
		batch_ids = chunk_ids[start:end]

		for attempt in range(config.upsert_max_retries + 1):
			try:
				vector_store.add_documents(
					batch_chunks,
					ids=batch_ids,
					batch_size=len(batch_chunks),
					timeout=config.qdrant_timeout,
				)
				upserted += len(batch_ids)
				print(f"Upserted batch {start + 1}-{end} / {total}")
				if end < total and config.upsert_batch_pause_seconds > 0:
					print(
						f"Sleeping {config.upsert_batch_pause_seconds}s before next batch "
						f"to avoid embedding rate limits..."
					)
					time.sleep(config.upsert_batch_pause_seconds)
				break
			except Exception as exc:
				is_last_attempt = attempt == config.upsert_max_retries
				if is_last_attempt:
					raise RuntimeError(
						f"Qdrant upsert failed for batch {start + 1}-{end} after {config.upsert_max_retries + 1} attempts: {exc}"
					) from exc

				wait_seconds = min(2 ** attempt, 8)
				print(
					f"Retrying batch {start + 1}-{end} in {wait_seconds}s "
					f"(attempt {attempt + 2}/{config.upsert_max_retries + 1}) due to: {exc}"
				)
				time.sleep(wait_seconds)

	return upserted


def node_upsert_qdrant(state: IngestionState) -> IngestionState:
	config = state["config"]
	chunks = state.get("chunks", [])

	chunk_ids = make_chunk_ids(chunks)
	if config.dry_run:
		return {
			"chunk_ids": chunk_ids,
			"upserted_count": 0,
			"qdrant_target": "dry-run",
		}

	if not chunks:
		return {
			"chunk_ids": [],
			"upserted_count": 0,
			"qdrant_target": "no-chunks",
		}

	embeddings = GoogleGenerativeAIEmbeddings(
		model=config.embedding_model,
		google_api_key=config.google_api_key,
	)
	vector_size = len(embeddings.embed_query("qdrant-dimension-check"))

	client, target = get_qdrant_client(config)
	try:
		ensure_collection(client, config.collection_name, vector_size, config.vector_name)

		vector_store = QdrantVectorStore(
			client=client,
			collection_name=config.collection_name,
			embedding=embeddings,
			vector_name=config.vector_name,
		)
		upserted_count = upsert_in_batches(vector_store, chunks, chunk_ids, config)
		print(
			f"Qdrant upload successful: created or updated {upserted_count} points "
			f"in collection '{config.collection_name}'."
		)
	finally:
		client.close()

	return {
		"chunk_ids": chunk_ids,
		"upserted_count": upserted_count,
		"qdrant_target": target,
	}


def build_graph() -> Any:
	graph = StateGraph(IngestionState)
	graph.add_node("load_documents", node_load_documents)
	graph.add_node("chunk_documents", node_chunk_documents)
	graph.add_node("upsert_qdrant", node_upsert_qdrant)

	graph.set_entry_point("load_documents")
	graph.add_edge("load_documents", "chunk_documents")
	graph.add_edge("chunk_documents", "upsert_qdrant")
	graph.add_edge("upsert_qdrant", END)
	return graph.compile()


def run_ingestion(config: IngestionConfig) -> dict[str, Any]:
	app = build_graph()
	result = app.invoke({"config": config})

	chunk_count = len(result.get("chunk_ids", []))
	print(f"Files scanned: {result.get('file_count', 0)}")
	print(f"Chunks built: {chunk_count}")
	if config.dry_run:
		print("Dry run complete: no vectors were upserted.")
	else:
		print(f"Vectors upserted: {result.get('upserted_count', 0)}")
		print(f"Qdrant target: {result.get('qdrant_target', 'unknown')}")

	return result


def parse_args() -> argparse.Namespace:
	script_dir = Path(__file__).resolve().parent
	default_md_root_candidates = [
		script_dir / "data.md",
		script_dir.parent / "data.md",
		script_dir / "md",
		script_dir.parent / "md",
	]
	default_md_root = next(
		(path for path in default_md_root_candidates if path.exists() and (path.is_file() or path.is_dir())),
		default_md_root_candidates[0],
	)
	default_local_qdrant = script_dir / "qdrant_data"

	parser = argparse.ArgumentParser(
		description="Ingest markdown files into Qdrant with recursive chunking and source metadata."
	)
	parser.add_argument("--md-root", default=str(default_md_root), help="Markdown file or root folder to ingest")
	parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "new_data"), help="Qdrant collection name")
	parser.add_argument("--vector-name", default=os.getenv("QDRANT_VECTOR_NAME", "dense"), help="Qdrant dense vector name (required for named-vector collections)")
	parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", ""), help="Qdrant URL (for server mode)")
	parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""), help="Qdrant API key")
	parser.add_argument("--qdrant-timeout", type=int, default=int(os.getenv("QDRANT_TIMEOUT", "180")), help="Qdrant request timeout in seconds")
	parser.add_argument("--qdrant-local-path", default=os.getenv("QDRANT_LOCAL_PATH", str(default_local_qdrant)), help="Local Qdrant path when URL is not provided")
	parser.add_argument(
		"--google-api-key",
		default=os.getenv("GOOGLE_API_KEY_INGESTION", os.getenv("GOOGLE_API_KEY", "")),
		help="Google API key for Gemini embeddings",
	)
	parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"), help="Google embedding model name")
	parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "1500")), help="Chunk size in characters")
	parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "300")), help="Chunk overlap in characters")
	parser.add_argument("--upsert-batch-size", type=int, default=int(os.getenv("UPSERT_BATCH_SIZE", "16")), help="Number of chunks to upsert per request")
	parser.add_argument("--upsert-max-retries", type=int, default=int(os.getenv("UPSERT_MAX_RETRIES", "3")), help="Retry attempts per failed upsert batch")
	parser.add_argument(
		"--upsert-batch-pause-seconds",
		type=int,
		default=int(os.getenv("UPSERT_BATCH_PAUSE_SECONDS", "60")),
		help="Pause between successful upsert batches to reduce rate limits",
	)
	parser.add_argument("--dry-run", action="store_true", help="Build chunks and metadata without upserting vectors")
	return parser.parse_args()


def main() -> int:
	load_dotenv()
	args = parse_args()

	md_root = Path(args.md_root).resolve()
	if not md_root.exists() or (not md_root.is_dir() and not md_root.is_file()):
		raise SystemExit(f"Invalid markdown source: {md_root}")

	if args.chunk_overlap >= args.chunk_size:
		raise SystemExit("chunk-overlap must be smaller than chunk-size")
	if args.upsert_batch_size <= 0:
		raise SystemExit("upsert-batch-size must be greater than 0")
	if args.upsert_max_retries < 0:
		raise SystemExit("upsert-max-retries cannot be negative")
	if args.upsert_batch_pause_seconds < 0:
		raise SystemExit("upsert-batch-pause-seconds cannot be negative")
	if args.qdrant_timeout <= 0:
		raise SystemExit("qdrant-timeout must be greater than 0")
	if not args.google_api_key.strip():
		raise SystemExit("Missing GOOGLE_API_KEY. Set it in .env or pass --google-api-key.")

	config = IngestionConfig(
		md_root=md_root,
		collection_name=args.collection,
		vector_name=args.vector_name.strip(),
		embedding_model=args.embedding_model,
		google_api_key=args.google_api_key.strip() or None,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
		upsert_batch_size=args.upsert_batch_size,
		upsert_max_retries=args.upsert_max_retries,
		upsert_batch_pause_seconds=args.upsert_batch_pause_seconds,
		qdrant_url=args.qdrant_url.strip() or None,
		qdrant_api_key=args.qdrant_api_key.strip() or None,
		qdrant_timeout=args.qdrant_timeout,
		qdrant_local_path=Path(args.qdrant_local_path).resolve(),
		dry_run=bool(args.dry_run),
	)
	run_ingestion(config)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
