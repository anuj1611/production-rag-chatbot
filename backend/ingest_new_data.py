import glob
import os
import sys
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Datatype, Distance, Filter, FilterSelector, HnswConfigDiff, VectorParams

from config import QDRANT_HOST, QDRANT_PORT

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "local_no_l_data")
SNAPSHOTS_DIR = Path(os.getenv("QDRANT_SNAPSHOTS_DIR", Path(__file__).resolve().parent / "snapshots"))


def get_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        https=False,   
        prefer_grpc=False
    )


def get_snapshot_path() -> str:
    files = glob.glob(str(SNAPSHOTS_DIR / "*.snapshot"))
    if not files:
        print(f"[ERROR] No .snapshot file found in '{SNAPSHOTS_DIR}'")
        sys.exit(1)
    if len(files) > 1:
        print(f"[WARN] Multiple snapshots found, using: {files[0]}")
    return files[0]


def collection_exists(client: QdrantClient, name: str) -> bool:
    existing = [c.name for c in client.get_collections().collections]
    return name in existing


def create_collection(client: QdrantClient, name: str) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(
                size=3072,
                distance=Distance.COSINE,
                on_disk=False,
                hnsw_config=HnswConfigDiff(
                    m=24,
                    payload_m=24,
                    ef_construct=256,
                ),
                datatype=Datatype.FLOAT32,
            )
        },
    )
    print(f"[OK] Collection '{name}' created.")


def clear_collection(client: QdrantClient, name: str) -> None:
    client.delete(
        collection_name=name,
        points_selector=FilterSelector(filter=Filter()),
    )
    print(f"[OK] All points deleted from '{name}'.")


def load_snapshot(client: QdrantClient, name: str, snapshot_path: str) -> None:
    print(f"[..] Uploading snapshot: {snapshot_path}")

    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{name}/snapshots/upload?priority=snapshot"
    headers = {}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    with open(snapshot_path, "rb") as snapshot_file:
        response = requests.post(
            url,
            headers=headers,
            files={
                "snapshot": (
                    os.path.basename(snapshot_path),
                    snapshot_file,
                    "application/octet-stream",
                )
            },
            timeout=300,
        )

    if response.status_code == 200:
        print(f"[OK] Snapshot loaded into '{name}'.")
    else:
        print(f"[ERROR] Snapshot upload failed: {response.status_code} - {response.text}")
        sys.exit(1)


def main() -> None:
    client = get_client()
    snapshot_path = get_snapshot_path()

    print("\n--- Qdrant Collection Setup ---")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Snapshot   : {snapshot_path}")
    print("--------------------------------\n")

    if collection_exists(client, COLLECTION_NAME):
        print(f"[INFO] Collection '{COLLECTION_NAME}' already exists, clearing data...")
        clear_collection(client, COLLECTION_NAME)
    else:
        print(f"[INFO] Collection '{COLLECTION_NAME}' not found, creating it...")
        create_collection(client, COLLECTION_NAME)

    load_snapshot(client, COLLECTION_NAME, snapshot_path)

    info = client.get_collection(COLLECTION_NAME)
    print(f"\n[DONE] Points in collection: {info.points_count}")


if __name__ == "__main__":
    main()
