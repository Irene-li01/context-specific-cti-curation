import json
import logging
import os
import glob
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Data_Ingestor")


class DataIngestor:
    def __init__(self):
        logger.info("Initializing Data Ingestor...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if self._check_cuda() else 'cpu'}
        )
        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten metadata to prevent ChromaDB insertion crashes on non-scalar values."""
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = "None"
            elif isinstance(value, list):
                sanitized[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = value
        return sanitized

    def reset_collection(self):
        """Wipe the collection before re-ingesting so duplicate runs don't stack up."""
        try:
            self.vector_store._client.delete_collection(Config.COLLECTION_NAME)
            logger.info("Cleared existing collection '%s'.", Config.COLLECTION_NAME)
        except Exception:
            pass
        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

    def _ingest_documents(self, raw_docs: List[Document], source_name: str):
        if not raw_docs:
            logger.warning("No documents found for %s. Skipping.", source_name)
            return

        logger.info("Chunking %d documents from %s...", len(raw_docs), source_name)
        chunked_docs = self.text_splitter.split_documents(raw_docs)

        logger.info("Inserting %d chunks into ChromaDB...", len(chunked_docs))
        batch_size = 100
        for i in range(0, len(chunked_docs), batch_size):
            self.vector_store.add_documents(documents=chunked_docs[i:i + batch_size])

        logger.info("Ingested %s.", source_name)

    def process_stix_bundle(self, filepath: str):
        logger.info("Processing STIX bundle: %s", filepath)
        if not os.path.exists(filepath):
            logger.error("STIX file not found: %s", filepath)
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            bundle = json.load(f)

        raw_documents = []
        for obj in bundle.get("objects", []):
            obj_type = obj.get("type")
            if obj_type not in ["indicator", "vulnerability"]:
                continue

            name = obj.get("name", "Unnamed Threat")
            desc = obj.get("description", "No description provided.")
            metadata = {"source_type": "global_stix", "stix_id": obj.get("id", "Unknown"), "threat_type": obj_type}

            if obj_type == "indicator":
                labels = obj.get("labels", [])
                content = (
                    f"GLOBAL THREAT (IoC)\nName: {name}\nDescription: {desc}\n"
                    f"Pattern: {obj.get('pattern', 'N/A')}\nTargeted Technologies: {', '.join(labels)}"
                )
                metadata["labels"] = labels
            else:
                content = f"GLOBAL VULNERABILITY\nCVE: {name}\nDescription: {desc}"

            raw_documents.append(Document(page_content=content, metadata=self._sanitize_metadata(metadata)))

        self._ingest_documents(raw_documents, "STIX Bundle")

    def process_recommendations(self, folder_path: str):
        logger.info("Processing recommendations from: %s", folder_path)
        if not os.path.exists(folder_path):
            logger.error("Recommendations folder not found: %s", folder_path)
            return

        # Group by org, keep only the newest file per org to avoid cross-run duplicates.
        all_files = glob.glob(os.path.join(folder_path, "recommendations_*.json"))
        latest: dict = {}
        for fp in all_files:
            basename = os.path.basename(fp)          # recommendations_ORG-XX-001_20260604T...json
            parts = basename.split("_", 2)
            org_key = parts[1] if len(parts) >= 2 else basename
            if org_key not in latest or fp > latest[org_key]:
                latest[org_key] = fp
        rec_files = list(latest.values())
        logger.info("Ingesting latest file for %d orgs (skipping %d older files).",
                    len(rec_files), len(all_files) - len(rec_files))
        raw_documents = []

        for filepath in rec_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            org_id = data.get("org_id", "Unknown-ORG")
            for rank, rec in enumerate(data.get("recommendations", []), start=1):
                content = (
                    f"ORGANIZATION SPECIFIC ADVICE FOR: {org_id}\n"
                    f"Threat Rank: #{rank}\n"
                    f"Threat Title: {rec.get('title', 'Unknown Threat')}\n"
                    f"Relevance Score: {rec.get('score', 0)}\n"
                    f"Technical Summary: {rec.get('entities', {}).get('summary', '')}\n"
                    f"AI Reasoning: {rec.get('llm_explanation', 'No AI reasoning provided.')}"
                )
                metadata = {
                    "source_type": "curated_recommendation",
                    "org_id": org_id,
                    "relevance_score": float(rec.get("score", 0)),
                    "severity": rec.get("severity", "unknown")
                }
                raw_documents.append(Document(page_content=content, metadata=self._sanitize_metadata(metadata)))

        self._ingest_documents(raw_documents, f"{len(rec_files)} Recommendation Profiles")

    def finish(self):
        # persist() was removed in ChromaDB 0.4.0 — auto-persists on write
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
        logger.info("Ingestion complete.")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    STIX_FILE = os.path.join(BASE_DIR, "curated_threats_stix3.json")
    RECOMMENDATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "recommendations")

    try:
        ingestor = DataIngestor()
        ingestor.reset_collection()
        ingestor.process_stix_bundle(STIX_FILE)
        ingestor.process_recommendations(RECOMMENDATIONS_DIR)
        ingestor.finish()
    except Exception as err:
        logger.error("Ingestion failed: %s", err, exc_info=True)
