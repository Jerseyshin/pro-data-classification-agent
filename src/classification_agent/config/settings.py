import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL", None)

    default_confidence_threshold: float = 0.7
    default_allow_multiple: bool = True

    # RAG settings
    enable_rag: bool = False
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.5

    # Embedding provider: "openai" or "bge"
    # - openai: use OpenAI API embeddings (good quality, requires API key)
    # - bge: use local BGE model (BAAI/bge-large-zh-v1.5, good for Chinese, runs locally)
    rag_embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "openai")

    # OpenAI embedding settings
    rag_embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    # Optional: separate OpenAI credentials just for RAG embeddings
    # Useful when you use a different provider for chat (e.g. MiniMax)
    rag_openai_api_key: str | None = os.getenv("RAG_OPENAI_API_KEY", None)
    rag_openai_base_url: str | None = os.getenv("RAG_OPENAI_BASE_URL", None)

    # BGE local embedding settings (for BAAI/bge-large-zh-v1.5)
    # You can use a local path instead of HuggingFace model name
    bge_model_name: str = os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
    bge_device: str = os.getenv("BGE_DEVICE", "cpu")
    bge_use_fp16: bool = os.getenv("BGE_USE_FP16", "true").lower() == "true"

    # RAG embedding cache settings (Wave 2 T2.1 optimization)
    # Cache backend options: "memory", "disk", "hybrid", or "none" to disable
    rag_cache_backend: str = os.getenv("RAG_CACHE_BACKEND", "memory")
    rag_cache_max_size: int = int(os.getenv("RAG_CACHE_MAX_SIZE", "10000"))
    rag_cache_ttl_seconds: int = int(os.getenv("RAG_CACHE_TTL_SECONDS", "3600"))
    rag_cache_dir: str = os.getenv("RAG_CACHE_DIR", ".rag_embeddings_cache")
    # Set to true to disable caching (not recommended for production)
    rag_disable_cache: bool = os.getenv("RAG_DISABLE_CACHE", "false").lower() == "true"

    # Speed optimization: fast mode merges feature analysis + preliminary classification
    # into one LLM call, reducing network round-trips by ~33%
    # Faster speed with slightly lower accuracy (usually acceptable)
    fast_mode: bool = os.getenv("FAST_MODE", "false").lower() == "true"

    class Config:
        env_file = ".env"
