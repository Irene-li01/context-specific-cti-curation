import logging
from typing import Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_Engine")

_SYSTEM_PROMPT = """You are a Level 3 SOC Analyst AI assistant for the CCTI (Curated Cyber Threat Intelligence) platform. You have access to curated threat intelligence for 12 organisations across 4 sectors.

ORGANISATION DIRECTORY (name → ID → sector):
- CommBank (ORG-FB-001, Finance) | PaySwift (ORG-FB-002, Finance) | SecureBank (ORG-SB-99, Finance)
- ElectraNet (ORG-EN-001, Energy) | PowerCo (ORG-EN-002, Energy) | SunPower (ORG-EN-003, Energy)
- Royal Adelaide Hospital (ORG-HC-001, Healthcare) | MedData (ORG-HC-002, Healthcare) | PharmaCoAus (ORG-HC-003, Healthcare)
- HarbourEats (ORG-CL-001, Other) | FreshMart (ORG-CL-002, Other) | FoodFirst (ORG-CL-003, Other)

The context contains two types of data:
1. GLOBAL THREATS: STIX 2.1 indicators and CVEs from the threat feed.
2. ORGANISATION RECOMMENDATIONS: AI-curated threats ranked by relevance score for each org.

RULES:
- Answer directly. NEVER ask for information the user already gave (e.g. org ID, sector).
- Use the organisation directory above to resolve org names to IDs automatically.
- When asked about a specific org, summarise its top ranked threats from context.
- For each threat include: what it is, why it matters to that org, and a concrete defensive action.
- For attack analysis queries: explain step-by-step how the attack works, what assets are at risk, and prioritised mitigations.
- Present findings in order of relevance score / threat rank — highest first.
- If context has no data for a named org, say so and suggest running the pipeline.
- Do not hallucinate. Only use what is in the provided context."""


# Map org names/aliases → org_id for smart filtering
_ORG_NAME_MAP = {
    "commbank": "ORG-FB-001", "comm bank": "ORG-FB-001",
    "payswift": "ORG-FB-002",
    "securebank": "ORG-SB-99", "secure bank": "ORG-SB-99",
    "electranet": "ORG-EN-001",
    "powerco": "ORG-EN-002", "power co": "ORG-EN-002",
    "sunpower": "ORG-EN-003", "sun power": "ORG-EN-003",
    "royal adelaide": "ORG-HC-001", "royal adelaide hospital": "ORG-HC-001",
    "meddata": "ORG-HC-002", "med data": "ORG-HC-002",
    "pharmacoaus": "ORG-HC-003", "pharmaco": "ORG-HC-003",
    "harboureats": "ORG-CL-001", "harbour eats": "ORG-CL-001",
    "freshmart": "ORG-CL-002", "fresh mart": "ORG-CL-002",
    "foodfirst": "ORG-CL-003", "food first": "ORG-CL-003",
    # also match raw IDs
    "org-fb-001": "ORG-FB-001", "org-fb-002": "ORG-FB-002",
    "org-sb-99": "ORG-SB-99",  "org-en-001": "ORG-EN-001",
    "org-en-002": "ORG-EN-002","org-en-003": "ORG-EN-003",
    "org-hc-001": "ORG-HC-001","org-hc-002": "ORG-HC-002",
    "org-hc-003": "ORG-HC-003","org-cl-001": "ORG-CL-001",
    "org-cl-002": "ORG-CL-002","org-cl-003": "ORG-CL-003",
}

def _detect_org(query: str):
    """Return org_id if any org name is mentioned in the query, else None."""
    q = query.lower()
    for name, org_id in _ORG_NAME_MAP.items():
        if name in q:
            return org_id
    return None


class ThreatIntelRAG:
    def __init__(self):
        if not Config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set — RAG chat unavailable.")

        logger.info("Initializing RAG Engine...")
        self.vector_store = self._initialize_store()
        self.llm = ChatGroq(
            model=Config.GROQ_MODEL,
            api_key=Config.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=2048,
        )
        logger.info("RAG Engine ready.")

    def _initialize_store(self):
        logger.info("Connecting to vector store at %s...", Config.CHROMA_PERSIST_DIR)
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        return Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

    def _retrieve(self, query: str, org_id=None):
        """Smart retrieval: semantic search for threat content + org top-N by score."""
        k = Config.RETRIEVER_K

        # Step 1: semantic search across ALL docs for the threat/actor/technique content
        all_docs = self.vector_store.similarity_search(query, k=k * 3)

        if org_id:
            # Step 2: fetch org's top recommendations directly by metadata (not similarity)
            # This guarantees we get the org's highest-scoring threats regardless of query wording
            raw = self.vector_store._collection.get(
                where={"$and": [{"source_type": {"$eq": "curated_recommendation"}}, {"org_id": {"$eq": org_id}}]},
                include=["documents", "metadatas"],
                limit=200
            )
            # Sort by relevance_score descending, take top-k
            org_pairs = sorted(
                zip(raw["documents"], raw["metadatas"]),
                key=lambda x: float(x[1].get("relevance_score") or 0),
                reverse=True
            )[:k]
            from langchain_core.documents import Document
            org_docs = [Document(page_content=doc, metadata=meta) for doc, meta in org_pairs]
        else:
            org_docs = []

        # Combine: org top threats first, then semantic results, deduplicate
        seen = set()
        docs = []
        for d in org_docs + all_docs:
            key = d.page_content[:120]
            if key not in seen:
                seen.add(key)
                docs.append(d)
        return docs[:k * 2]

    def query(self, user_query: str) -> Dict[str, Any]:
        logger.info("RAG query: '%s'", user_query)
        try:
            org_id = _detect_org(user_query)
            if org_id:
                logger.info("Org detected in query: %s", org_id)
            docs = self._retrieve(user_query, org_id)
            context = "\n\n---\n\n".join(d.page_content for d in docs)

            response = self.llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=f"--- THREAT CONTEXT ---\n{context}\n----------------------\n\nANALYST QUERY: {user_query}")
            ])

            formatted_sources = []
            for doc in docs:
                meta = doc.metadata
                s_type = meta.get("source_type", "unknown")
                label = (
                    f"[{meta.get('org_id')}] Rank #{meta.get('threat_rank')} | Score {meta.get('relevance_score')}"
                    if s_type == "curated_recommendation"
                    else f"[STIX] {meta.get('threat_type', 'Indicator')}"
                )
                formatted_sources.append({"label": label, "content": doc.page_content, "metadata": meta})

            return {"status": "success", "answer": response.content, "sources": formatted_sources}

        except Exception as e:
            logger.error("RAG query failed: %s", e)
            return {"status": "error", "answer": f"Query failed: {e}", "sources": []}
