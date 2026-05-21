import argparse
from urllib.parse import urlparse
from chromadb import PersistentClient
from tqdm import tqdm

from utils import slugify


def backfill_slugs(vector_db_path: str, collection_name: str = "lore") -> None:
    """Add slug metadata to all records in an existing Chroma collection.

    Parameters
    ----------
    vector_db_path: str
        Path to the persistent Chroma database directory.
    collection_name: str, default "lore"
        Name of the collection to update.
    """
    client = PersistentClient(path=vector_db_path)
    collection = client.get_collection(collection_name)
    items = collection.get(include=["metadatas"])

    ids = items["ids"]
    metas = items["metadatas"]
    for item_id, meta in tqdm(zip(ids, metas), total=len(ids), desc="Updating"):
        url = meta.get("source")
        if not url:
            continue
        slug = slugify(urlparse(url).path.rstrip("/").split("/")[-1])
        collection.update(ids=[item_id], metadatas=[{**meta, "slug": slug}])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill slug metadata in a Chroma collection")
    parser.add_argument("--vector-db-path", required=True, help="Path to Chroma database")
    parser.add_argument("--collection", default="lore", help="Collection name (default: lore)")
    args = parser.parse_args()
    backfill_slugs(args.vector_db_path, args.collection)
