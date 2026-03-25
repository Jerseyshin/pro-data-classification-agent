from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from openai import OpenAI
import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseEmbeddings(ABC):
    """Base interface for text embeddings"""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string"""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch"""
        pass


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings client for text embedding"""

    def __init__(
        self,
        client: OpenAI,
        model: str = "text-embedding-3-small",
    ):
        self.client = client
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch"""
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        if not response.data:
            return []
        return [data.embedding for data in response.data if data and hasattr(data, 'embedding')]


class BGEEmbeddings(BaseEmbeddings):
    """Local BGE embedding model (BAAI/bge-large-zh-v1.5) via HuggingFace transformers

    Runs locally on your machine, no API calls needed. Good for Chinese text.
    Requires: pip install transformers torch
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cpu",
        use_fp16: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name or path to local model
                Default: "BAAI/bge-large-zh-v1.5"
            device: Device to run on ("cpu", "cuda", "cuda:0", etc.)
            use_fp16: Use half precision for faster inference
        """
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "BGEEmbeddings requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto" if use_fp16 else None,
        ).to(device)

    @staticmethod
    def _average_pool(last_hidden_state, attention_mask):
        """Average pooling for BGE embeddings"""
        import torch
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string with BGE instruction prefix for retrieval"""
        # BGE uses "为这个句子生成表示以用于检索相关文章：" as instruction for retrieval
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch"""
        import torch
        if not texts:
            return []

        # Add BGE instruction prefix for retrieval
        # Reference: https://huggingface.co/BAAI/bge-large-zh-v1.5
        instruction = "为这个句子生成表示以用于检索相关文章："
        texts_with_instruction = [f"{instruction}{text}" for text in texts]

        # Tokenize
        encoded_input = self.tokenizer(
            texts_with_instruction,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        # Inference
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._average_pool(
                model_output.last_hidden_state,
                encoded_input["attention_mask"]
            )
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list[list[float]]
        return embeddings.cpu().tolist()
    

class MemoryCache:
    """Simple thread-safe in-memory LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get item from cache, return None if not found or expired"""
        with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None
            
            timestamp = self._timestamps.get(key, 0)
            current_time = time.time()
            if current_time - timestamp > self.ttl_seconds:
                # Expired
                self._evict_key(key)
                self._miss_count += 1
                return None
            
            # Move to end (most recently used)
            value = self._cache[key]
            self._cache.move_to_end(key)
            self._hit_count += 1
            return value
    
    def set(self, key: str, value: List[float]):
        """Set item in cache"""
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache[key] = value
                self._cache.move_to_end(key)
            else:
                # Add new
                self._cache[key] = value
                
            self._timestamps[key] = time.time()
            
            # Enforce max size
            if len(self._cache) > self.max_size:
                oldest = next(iter(self._cache))
                self._evict_key(oldest)
    
    def _evict_key(self, key: str):
        """Evict a key from cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hit_count = 0
            self._miss_count = 0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds,
            }


class DiskCache:
    """Disk-based cache using SQLite for persistence"""
    
    def __init__(self, db_path: str = ":memory:", max_size: int = 100000, ttl_seconds: int = 86400):
        self.db_path = db_path
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._conn = None
        self._init_db()
        self._hit_count = 0
        self._miss_count = 0
    
    def _init_db(self):
        """Initialize SQLite database"""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self._conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                timestamp REAL NOT NULL,
                model_name TEXT NOT NULL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON embeddings_cache(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings_cache(model_name)')
        self._conn.commit()
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get item from cache"""
        cursor = self._conn.cursor()
        cursor.execute(
            'SELECT embedding, timestamp FROM embeddings_cache WHERE key=?',
            (key,)
        )
        row = cursor.fetchone()
        
        if not row:
            self._miss_count += 1
            return None
        
        embedding_blob, timestamp = row
        current_time = time.time()
        
        if current_time - timestamp > self.ttl_seconds:
            # Expired, clean up
            cursor.execute('DELETE FROM embeddings_cache WHERE key=?', (key,))
            self._conn.commit()
            self._miss_count += 1
            return None
        
        # Deserialize embedding
        embedding = json.loads(embedding_blob)
        self._hit_count += 1
        return embedding
    
    def set(self, key: str, value: List[float], model_name: str = "unknown"):
        """Set item in cache"""
        cursor = self._conn.cursor()
        embedding_blob = json.dumps(value)
        cursor.execute(
            'INSERT OR REPLACE INTO embeddings_cache (key, embedding, timestamp, model_name) VALUES (?, ?, ?, ?)',
            (key, embedding_blob, time.time(), model_name)
        )
        
        # Clean up old entries if over max size
        cursor.execute('SELECT COUNT(*) FROM embeddings_cache')
        count = cursor.fetchone()[0]
        
        if count > self.max_size:
            # Delete oldest entries
            cursor.execute('''
                DELETE FROM embeddings_cache 
                WHERE key IN (
                    SELECT key FROM embeddings_cache 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                )
            ''', (count - self.max_size,))
        
        self._conn.commit()
    
    def clear(self, model_name: Optional[str] = None):
        """Clear cache entries, optionally filtered by model name"""
        cursor = self._conn.cursor()
        if model_name:
            cursor.execute('DELETE FROM embeddings_cache WHERE model_name=?', (model_name,))
        else:
            cursor.execute('DELETE FROM embeddings_cache')
        self._conn.commit()
        self._hit_count = 0
        self._miss_count = 0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cursor = self._conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings_cache')
        size = cursor.fetchone()[0]
        
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0
        
        return {
            "size": size,
            "max_size": self.max_size,
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
            "db_path": self.db_path,
        }


class CacheBackedEmbeddings(BaseEmbeddings):
    """Embeddings wrapper with caching to avoid recomputing embeddings for same text
    
    Features:
    1. Memory cache (LRU with TTL) for fast access
    2. Disk cache (SQLite) for persistence across runs
    3. Batch optimization: only compute embeddings for texts not in cache
    4. Thread-safe for concurrent access
    5. Performance monitoring and statistics
    """
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        cache_backend: str = "memory",
        max_cache_size: int = 10000,
        ttl_seconds: int = 3600,
        cache_dir: Optional[str] = None,
        enable_statistics: bool = True,
    ):
        """
        Args:
            embeddings: The underlying embeddings instance to wrap
            cache_backend: "memory", "disk", or "hybrid" (memory + disk)
            max_cache_size: Maximum number of items in cache
            ttl_seconds: Time-to-live for cache entries in seconds
            cache_dir: Directory for disk cache (if using disk backend)
            enable_statistics: Whether to collect cache usage statistics
        """
        self.embeddings = embeddings
        self.cache_backend = cache_backend
        self.ttl_seconds = ttl_seconds
        self.enable_statistics = enable_statistics
        
        # Initialize caches
        self.memory_cache = None
        self.disk_cache = None
        
        if cache_backend in ["memory", "hybrid"]:
            self.memory_cache = MemoryCache(
                max_size=max_cache_size // 2 if cache_backend == "hybrid" else max_cache_size,
                ttl_seconds=ttl_seconds
            )
        
        if cache_backend in ["disk", "hybrid"]:
            if cache_dir is None:
                cache_dir = ".embeddings_cache"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            db_path = str(Path(cache_dir) / "embeddings_cache.db")
            disk_max_size = max_cache_size // 2 if cache_backend == "hybrid" else max_cache_size
            self.disk_cache = DiskCache(
                db_path=db_path,
                max_size=disk_max_size,
                ttl_seconds=ttl_seconds
            )
        
        # Performance tracking
        self._total_embed_text_calls = 0
        self._total_embed_texts_calls = 0
        self._total_cached_embeddings = 0
        self._total_computed_embeddings = 0
        self._total_compute_time_ms = 0.0
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        Includes model information to avoid collisions between different embedding models.
        """
        # Normalize text (remove extra whitespace)
        normalized = ' '.join(text.strip().split())
        # Create hash of normalized text
        content_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:32]
        
        # Use model/configuration hash for versioning
        # This ensures different embedding models don't share cached values
        model_info = f"{type(self.embeddings).__name__}"
        if hasattr(self.embeddings, 'model'):
            model_info += f"_{self.embeddings.model}"
        
        return f"{model_info}:{content_hash}"
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string with caching"""
        start_time = time.time()
        self._total_embed_text_calls += 1
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached = self._get_from_cache(cache_key)
        
        if cached is not None:
            self._total_cached_embeddings += 1
            return cached
        
        # Not in cache, compute
        embedding = self.embeddings.embed_text(text)
        self._total_computed_embeddings += 1
        self._total_compute_time_ms += (time.time() - start_time) * 1000
        
        # Store in cache
        self._set_to_cache(cache_key, embedding)
        
        if self.enable_statistics:
            cache_stats = self.stats
            if cache_stats.get('hits', 0) + cache_stats.get('misses', 0) > 100:  # Log every 100 operations
                hit_rate = cache_stats.get('hit_rate', 0)
                if hit_rate < 0.3:
                    logger.info(f"Cache hit rate low ({hit_rate:.1%}), consider increasing cache size or TTL")
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch with caching optimization"""
        start_time = time.time()
        self._total_embed_texts_calls += 1
        
        if not texts:
            return []
        
        # Phase 1: Check cache for all texts
        results = []
        to_compute = []
        compute_indices = []
        cache_keys = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_keys.append(cache_key)
            
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                results.append(cached)
                self._total_cached_embeddings += 1
            else:
                results.append(None)  # Placeholder
                to_compute.append(text)
                compute_indices.append(i)
        
        # Phase 2: Compute only the texts not in cache
        if to_compute:
            computed = self.embeddings.embed_texts(to_compute)
            self._total_computed_embeddings += len(computed)
            
            # Phase 3: Fill in results and update cache
            for idx, computed_idx in enumerate(compute_indices):
                embedding = computed[idx]
                results[computed_idx] = embedding
                self._set_to_cache(cache_keys[computed_idx], embedding)
        
        self._total_compute_time_ms += (time.time() - start_time) * 1000
        return results
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache, checking memory first then disk"""
        # Try memory cache first
        if self.memory_cache:
            cached = self.memory_cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Try disk cache
        if self.disk_cache:
            cached = self.disk_cache.get(cache_key)
            if cached is not None and self.memory_cache:
                # Populate memory cache from disk cache
                self.memory_cache.set(cache_key, cached)
            return cached
        
        return None
    
    def _set_to_cache(self, cache_key: str, embedding: List[float]):
        """Store embedding in cache"""
        if self.memory_cache:
            self.memory_cache.set(cache_key, embedding)
        
        if self.disk_cache:
            # Determine model name for disk cache
            model_name = "unknown"
            if hasattr(self.embeddings, 'model'):
                model_name = self.embeddings.model
            elif hasattr(self.embeddings, 'model_name'):
                model_name = self.embeddings.model_name
                
            self.disk_cache.set(cache_key, embedding, model_name=model_name)
    
    def clear_cache(self, memory_only: bool = False):
        """Clear cache entries"""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.disk_cache and not memory_only:
            self.disk_cache.clear()
        
        # Reset performance counters
        self._total_cached_embeddings = 0
        self._total_computed_embeddings = 0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics and performance metrics"""
        total_operations = self._total_cached_embeddings + self._total_computed_embeddings
        hit_rate = self._total_cached_embeddings / total_operations if total_operations > 0 else 0
        avg_compute_time = self._total_compute_time_ms / self._total_computed_embeddings if self._total_computed_embeddings > 0 else 0
        
        memory_stats = self.memory_cache.stats if self.memory_cache else {}
        disk_stats = self.disk_cache.stats if self.disk_cache else {}
        
        combined_stats = {
            "backend": self.cache_backend,
            "hit_rate": hit_rate,
            "total_operations": total_operations,
            "cached_embeddings": self._total_cached_embeddings,
            "computed_embeddings": self._total_computed_embeddings,
            "embed_text_calls": self._total_embed_text_calls,
            "embed_texts_calls": self._total_embed_texts_calls,
            "avg_compute_time_ms": avg_compute_time,
            "total_compute_time_ms": self._total_compute_time_ms,
            "ttl_seconds": self.ttl_seconds,
        }
        
        if memory_stats:
            combined_stats["memory_cache"] = memory_stats
        if disk_stats:
            combined_stats["disk_cache"] = disk_stats
        
        return combined_stats
    
    def log_stats(self, level: str = "INFO"):
        """Log cache statistics at specified log level"""
        stats = self.stats
        log_func = getattr(logger, level.lower(), logger.info)
        
        log_func(f"CacheBackedEmbeddings statistics:")
        log_func(f"  Backend: {stats['backend']}")
        log_func(f"  Hit rate: {stats['hit_rate']:.1%} ({stats['cached_embeddings']}/{stats['total_operations']})")
        log_func(f"  Operations: {stats['embed_text_calls']} single, {stats['embed_texts_calls']} batch")
        log_func(f"  Compute time: {stats['avg_compute_time_ms']:.1f}ms avg, {stats['total_compute_time_ms']:.1f}ms total")
        
        if "memory_cache" in stats:
            mem = stats["memory_cache"]
            log_func(f"  Memory cache: {mem['size']}/{mem['max_size']} items")
        
        if "disk_cache" in stats:
            disk = stats["disk_cache"]
            log_func(f"  Disk cache: {disk['size']}/{disk['max_size']} items in {disk['db_path']}")
