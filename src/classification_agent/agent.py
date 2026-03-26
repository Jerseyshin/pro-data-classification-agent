from typing import List, Optional, Tuple, Dict, Callable, Generator, Any
import time
import os
import csv
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from langgraph.graph import StateGraph

# Performance measurement
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    import resource

from classification_agent.graph.state import ClassificationState, TableContextAnalysis
from classification_agent.graph.builder import build_graph
from classification_agent.types.schemas import (
    TableFieldInput,
    HierarchicalCategory,
    FeatureAnalysisResult,
    PreliminaryResult,
    VerificationResult,
    RetrievedExample,
    ClassificationResult,
    PredictedItem,
)
from classification_agent.llm.base import BaseLLM
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.config.settings import Settings
from classification_agent.utils.validation import validate_input, validate_categories
from classification_agent.rag.embeddings import OpenAIEmbeddings
from classification_agent.rag.vector_store import InMemoryVectorStore
from classification_agent.utils.logging import get_logger
from classification_agent.utils.data_reader import load_data_csv

logger = get_logger(__name__)


# Regex pattern for parsing "数据项(子数据项)" format in ground truth
_GT_PATTERN = re.compile(r"([^(]+)\(([^)]+)\)")


def _parse_ground_truth(label_text: str) -> List[str]:
    """
    Parse ground truth label text to extract data item names.

    Format example:
        "用户应用基本信息(应用包名\客户端版本号)" → ["用户应用基本信息"]
        "交易信息(交易记录)" → ["交易信息"]
        "A(a), B(b)" → ["A", "B"]

    Args:
        label_text: Raw label text

    Returns:
        List of extracted data item names
    """
    if not label_text or not label_text.strip():
        return []

    label_text = label_text.strip()
    matches = _GT_PATTERN.findall(label_text)

    data_items = [data_item.strip() for data_item, _ in matches]
    return list(dict.fromkeys(data_items))


def process_large_csv_stream(
    file_path: str | Path,
    batch_size: int = 50,
    encoding: str = "utf-8-sig",
    include_ground_truth: bool = True,
    skip_empty_gt: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Generator[
    Tuple[List[TableFieldInput], Optional[List[List[str]]], int, int], None, None
]:
    """
    Stream process large CSV file in chunks to avoid loading entire file into memory.

    This generator yields batches of records, allowing processing of large files
    with minimal memory footprint.

    CSV columns:
        数据域,表名,表中文名,表字段,字段描述,字段隐私四级分类

    Args:
        file_path: Path to CSV file
        batch_size: Number of records per batch (default 50)
        encoding: File encoding (default utf-8-sig, can use GB18030)
        include_ground_truth: Whether to parse and yield ground truth labels
        skip_empty_gt: Whether to skip records with empty ground truth
        progress_callback: Optional callback(batch_idx, total_batches) for progress

    Yields:
        Tuple of (inputs_batch, ground_truths_batch, batch_idx, total_batches)
        When include_ground_truth=False, ground_truths_batch is None
    """
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    batch_inputs: List[TableFieldInput] = []
    batch_gts: List[List[str]] = []
    batch_count = 0
    total_records = 0

    with open(csv_path, encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row_idx, row in enumerate(reader, 2):
            if len(row) < 5:
                continue

            table_name_raw = row[1].strip()
            table_cn_name = row[2].strip()
            field_name = row[3].strip()
            field_desc = row[4].strip() if len(row) > 4 else ""
            gt_text = row[5].strip() if len(row) > 5 else ""

            if table_cn_name:
                table_name = f"{table_name_raw} - {table_cn_name}"
            else:
                table_name = table_name_raw

            if not table_name or not field_name:
                continue

            input_obj: TableFieldInput = {
                "table_name": table_name,
                "field_name": field_name,
                "field_description": field_desc or None,
            }

            gt_data_items: Optional[List[str]] = None
            if include_ground_truth:
                gt_data_items = _parse_ground_truth(gt_text)
                if skip_empty_gt and not gt_data_items:
                    continue

            batch_inputs.append(input_obj)
            if include_ground_truth:
                batch_gts.append(gt_data_items or [])

            total_records += 1

            if len(batch_inputs) >= batch_size:
                batch_count += 1
                if progress_callback:
                    progress_callback(batch_count, -1)  # -1 indicates unknown total
                yield (
                    batch_inputs,
                    batch_gts if include_ground_truth else None,
                    batch_count,
                    total_records,
                )
                batch_inputs = []
                batch_gts = []

        if batch_inputs:
            batch_count += 1
            if progress_callback:
                progress_callback(batch_count, -1)
            yield (
                batch_inputs,
                batch_gts if include_ground_truth else None,
                batch_count,
                total_records,
            )


class ClassificationAgent:
    """表格字段层级分类Agent
    采用思考验证架构：特征分析 → [可选RAG检索] → 初步分类 → 自我验证 → 最终结果
    支持动态权衡表名和字段名的重要性，可选RAG检索相似标注样例辅助分类
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        hierarchical_categories: Optional[List[HierarchicalCategory]] = None,
        confidence_threshold: float = 0.7,
        allow_multiple: bool = True,
        settings: Optional[Settings] = None,
        # RAG options
        enable_rag: Optional[bool] = None,
        rag_examples: Optional[
            List[Tuple[TableFieldInput, HierarchicalCategory]]
        ] = None,
        rag_top_k: Optional[int] = None,
        rag_similarity_threshold: Optional[float] = None,
        rag_embedding_provider: Optional[str] = None,
        # BGE options
        bge_model_name: Optional[str] = None,
        bge_device: Optional[str] = None,
        bge_use_fp16: Optional[bool] = None,
        # Table context optimization
        enable_table_context: bool = True,
    ):
        """
        Args:
            llm: LLM实例，如果不提供则使用默认OpenAI配置
            hierarchical_categories: 预定义的层级分类体系
            confidence_threshold: 置信度阈值，低于此建议重分类
            allow_multiple: 是否允许多标签输出
            settings: 配置对象
            enable_rag: 是否启用RAG检索，默认从settings读取
            rag_examples: 初始化时添加的标注样例列表，每个元素是 (input, correct_label)
            rag_top_k: 检索返回的最相似结果数
            rag_similarity_threshold: 相似度阈值，低于此不返回
            rag_embedding_provider: Embedding 提供者 ("openai" or "bge")
            bge_model_name: BGE 模型名称或本地路径
            bge_device: BGE 运行设备 ("cpu" or "cuda")
            bge_use_fp16: BGE 是否使用半精度
            enable_table_context: 是否启用表级上下文分析，在字段分类前先分析整张表用途，
                提供全局上下文给后续字段分类，提高准确率，默认开启
        """
        self.settings = settings or Settings()

        # Instance options
        self.enable_table_context = enable_table_context

        # Apply RAG config overrides
        if enable_rag is not None:
            self.settings.enable_rag = enable_rag
        if rag_top_k is not None:
            self.settings.rag_top_k = rag_top_k
        if rag_similarity_threshold is not None:
            self.settings.rag_similarity_threshold = rag_similarity_threshold
        if rag_embedding_provider is not None:
            self.settings.rag_embedding_provider = rag_embedding_provider
        # Apply BGE config overrides
        if bge_model_name is not None:
            self.settings.bge_model_name = bge_model_name
        if bge_device is not None:
            self.settings.bge_device = bge_device
        if bge_use_fp16 is not None:
            self.settings.bge_use_fp16 = bge_use_fp16

        self.llm = llm or self._create_default_llm()
        self.hierarchical_categories = hierarchical_categories
        self.confidence_threshold = confidence_threshold
        self.allow_multiple = allow_multiple

        # RAG initialization
        self._init_rag(rag_examples)

        # 验证预定义分类
        if self.hierarchical_categories:
            validate_categories(self.hierarchical_categories)

        # 构建图
        self.graph = build_graph(
            self.llm,
            self.settings,
            self.vector_store,
            self.embeddings,
            enable_table_context=self.enable_table_context,
        ).compile()

    def _create_default_llm(self) -> BaseLLM:
        """Create default OpenAI LLM from settings"""
        return OpenAILLM(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
            base_url=self.settings.openai_base_url,
        )

    def _init_rag(
        self, rag_examples: Optional[List[Tuple[TableFieldInput, HierarchicalCategory]]]
    ) -> None:
        """Initialize RAG components if enabled"""
        if not self.settings.enable_rag:
            self.vector_store = None
            self.embeddings = None
            return

        # Initialize embeddings based on provider
        if self.settings.rag_embedding_provider == "bge":
            # Use local BGE model (BAAI/bge-large-zh-v1.5)
            from classification_agent.rag import BGEEmbeddings

            base_embeddings = BGEEmbeddings(
                model_name=self.settings.bge_model_name,
                device=self.settings.bge_device,
                use_fp16=self.settings.bge_use_fp16,
            )
        else:
            # Default to OpenAI embeddings
            from classification_agent.rag import OpenAIEmbeddings

            # Use separate OpenAI credentials for RAG if provided
            if self.settings.rag_openai_api_key:
                # RAG uses its own OpenAI client (for embedding only)
                client = OpenAI(
                    api_key=self.settings.rag_openai_api_key,
                    base_url=self.settings.rag_openai_base_url,
                )
            elif isinstance(self.llm, OpenAILLM):
                # Reuse the same OpenAI client from the main LLM
                client = self.llm.client
            else:
                # Fallback to settings main credentials
                client = OpenAI(
                    api_key=self.settings.openai_api_key,
                    base_url=self.settings.openai_base_url,
                )

            base_embeddings = OpenAIEmbeddings(
                client=client,
                model=self.settings.rag_embedding_model,
            )

        # Wrap with cache for performance optimization (Wave 2 T2.1)
        # Only enable cache if explicitly configured or not disabled
        from classification_agent.rag.embeddings import CacheBackedEmbeddings

        cache_backend = (
            self.settings.rag_cache_backend
            if hasattr(self.settings, "rag_cache_backend")
            else "memory"
        )
        cache_enabled = not getattr(self.settings, "rag_disable_cache", False)

        if cache_enabled:
            self.embeddings = CacheBackedEmbeddings(
                embeddings=base_embeddings,
                cache_backend=cache_backend,
                max_cache_size=getattr(self.settings, "rag_cache_max_size", 10000),
                ttl_seconds=getattr(self.settings, "rag_cache_ttl_seconds", 3600),
                cache_dir=getattr(self.settings, "rag_cache_dir", ".embeddings_cache"),
                enable_statistics=True,
            )
            logger.info(f"RAG embeddings caching enabled (backend: {cache_backend})")
        else:
            self.embeddings = base_embeddings
            logger.info("RAG embeddings caching disabled")

        # Initialize vector store
        self.vector_store = InMemoryVectorStore()

        # Add initial examples if provided
        if rag_examples and len(rag_examples) > 0:
            self.add_rag_examples(rag_examples)

    def add_rag_examples(
        self,
        examples: List[Tuple[TableFieldInput, HierarchicalCategory]],
    ) -> None:
        """Add labeled examples to RAG vector store dynamically

        Args:
            examples: List of (input, correct_label) tuples
        """
        if (
            not self.settings.enable_rag
            or self.vector_store is None
            or self.embeddings is None
        ):
            raise RuntimeError("Cannot add RAG examples: RAG is not enabled")

        if not examples:
            return

        # Precompute embeddings for all examples
        texts = []
        valid_examples = []
        for input_example, label in examples:
            # Create combined text same way as query
            parts = [
                f"Table: {input_example.get('table_name', '')}",
                f"Field: {input_example.get('field_name', '')}",
            ]
            if input_example.get("field_description"):
                parts.append(f"Description: {input_example['field_description']}")
            texts.append(" ".join(parts))
            valid_examples.append((input_example, label))

        # Get embeddings
        embeddings_result = self.embeddings.embed_texts(texts)
        if len(embeddings_result) != len(texts):
            # Mismatch - filter out any failed embeddings
            valid_pairs = [
                (np.array(emb), input_example, label)
                for emb, (input_example, label) in zip(
                    embeddings_result, valid_examples
                )
                if emb is not None
            ]
            if not valid_pairs:
                logger.warning("No valid embeddings obtained, nothing added")
                return
            embeddings_list = [p[0] for p in valid_pairs]
            valid_examples = [(p[1], p[2]) for p in valid_pairs]

        else:
            embeddings_list = [
                np.array(emb) for emb in embeddings_result if emb is not None
            ]

        # Add to vector store
        self.vector_store.add_examples(embeddings_list, valid_examples)

        logger.info("Added %d examples to RAG vector store", len(valid_examples))

    def clear_rag_examples(self) -> None:
        """Clear all RAG examples from the vector store"""
        if self.vector_store is not None:
            self.vector_store.clear()
            logger.info("Cleared all RAG examples")

    def get_rag_examples(self) -> List[Tuple[TableFieldInput, HierarchicalCategory]]:
        """Get all stored RAG examples"""
        if self.vector_store is None:
            return []
        return self.vector_store.get_all_examples()

    def classify(
        self,
        field_input: TableFieldInput,
        hierarchical_categories: Optional[List[HierarchicalCategory]] = None,
        confidence_threshold: Optional[float] = None,
        allow_multiple: Optional[bool] = None,
        enable_rag: Optional[bool] = None,
        ground_truth_data_items: Optional[List[str]] = None,
    ) -> ClassificationResult:
        """
        对表格字段进行分类

        Args:
            field_input: 输入，包含table_name, field_name, 可选field_description
            hierarchical_categories: 动态传入分类体系，覆盖初始化时的预定义
            confidence_threshold: 动态指定置信度阈值
            allow_multiple: 动态指定是否允许多标签
            enable_rag: 动态指定是否启用RAG，不指定则使用settings默认
            ground_truth_data_items: 真实数据项标签，如果提供会自动运行评估计算准确率

        Returns:
            ClassificationResult 包含完整推理链和最终结果，如果提供了ground_truth则包含evaluation
        """
        # 输入验证
        validate_input(field_input)

        # 使用分类体系
        categories = hierarchical_categories or self.hierarchical_categories
        if categories is None:
            raise ValueError(
                "hierarchical_categories must be provided either "
                "at initialization or classification time"
            )
        # 动态传入的 categories 才需要校验（初始化时传入的已在 __init__ 校验过）
        if hierarchical_categories is not None:
            validate_categories(categories)

        # 配置
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )
        multiple = allow_multiple if allow_multiple is not None else self.allow_multiple

        # Determine if RAG is enabled for this classification
        # If explicitly provided, use that; otherwise use settings default
        rag_enabled = enable_rag if enable_rag is not None else self.settings.enable_rag

        # Check if we actually have examples to use
        if rag_enabled and (self.vector_store is None or self.vector_store.size == 0):
            logger.warning(
                "RAG enabled but no examples available, disabling for this classification"
            )
            rag_enabled = False

        # 初始状态
        initial_state: ClassificationState = {
            "input": field_input,
            "hierarchical_categories": categories,
            "confidence_threshold": threshold,
            "allow_multiple": multiple,
            "rag_enabled": rag_enabled,
            "ground_truth_data_items": ground_truth_data_items,
            "evaluation": None,
            "retrieved_examples": None,
            "feature_analysis": None,
            "preliminary_classification": None,
            "verification": None,
            "reclassification_count": 0,
        }

        # 执行图
        result_state = self.graph.invoke(initial_state)

        # 提取结果
        final_predictions: List[PredictedItem] = result_state.get(
            "_final_predictions", []
        )
        final_labels = result_state.get("_final_labels", [])
        final_avg_confidence = result_state.get("_final_avg_confidence", 0.0)
        reasoning_chain = result_state.get("_reasoning_chain", [])

        # 构建返回结果
        # 用 dict 过滤多余字段，避免 LLM 返回额外 key 导致 TypedDict 展开报 TypeError
        known_feature_keys = {
            "table_name_keywords",
            "field_name_keywords",
            "description_keywords",
            "semantic_summary",
            "consistency_analysis",
            "dominant_source",
        }
        known_preliminary_keys = {"predictions", "total_confidence"}
        known_verification_keys = {
            "verified_predictions",
            "removed_false_positives",
            "added_missing",
            "average_confidence",
            "cross_validation_note",
            "suggests_reclassification",
        }

        classification_result: ClassificationResult = {
            "final_predictions": final_predictions,
            "final_labels": final_labels,
            "final_confidence": final_avg_confidence,
            "reasoning_chain": reasoning_chain,
            "feature_analysis": FeatureAnalysisResult(
                **{
                    k: v
                    for k, v in result_state["feature_analysis"].items()
                    if k in known_feature_keys
                }
            ),
            "preliminary_result": PreliminaryResult(
                **{
                    k: v
                    for k, v in result_state["preliminary_classification"].items()
                    if k in known_preliminary_keys
                }
            ),
            "verification_result": VerificationResult(
                **{
                    k: v
                    for k, v in (result_state.get("verification") or {}).items()
                    if k in known_verification_keys
                }
            ),
            "evaluation": result_state.get("evaluation"),
            "table_context_analysis": None,
        }

        return classification_result

    def classify_table(
        self,
        fields: List[TableFieldInput],
        table_chinese_name: Optional[str] = None,
        hierarchical_categories: Optional[List[HierarchicalCategory]] = None,
        confidence_threshold: Optional[float] = None,
        allow_multiple: Optional[bool] = None,
        enable_rag: Optional[bool] = None,
        ground_truth_list: Optional[List[List[str]]] = None,
    ) -> List[ClassificationResult]:
        """
        对整张表所有字段批量分类（仅支持Bulk模式）：

        **Bulk Mode (table-level processing with minimal LLM calls):**
        context_analysis (once) → (optional rag) → bulk_feature_analysis (ALL fields in one LLM call) →
        bulk_preliminary_classification (ALL fields in one LLM call) → bulk_self_verification (ALL in one) →
        bulk_final_result → done.

        **Total LLM calls for bulk mode: 3-4 calls TOTAL regardless of number of fields.**
        This is the fastest mode, drastically reduces API calls and total processing time.

        Args:
            fields: 整张表所有字段，每个元素包含 table_name, field_name, 可选 field_description
            table_chinese_name: 表中文名（可选），用于表级上下文分析
            hierarchical_categories: 动态传入分类体系，覆盖初始化时的预定义
            confidence_threshold: 动态指定置信度阈值
            allow_multiple: 动态指定是否允许多标签
            enable_rag: 动态指定是否启用RAG，不指定则使用settings默认
            ground_truth_list: 每个字段对应的真实数据项标签列表，如果提供会自动运行评估
                len(ground_truth_list) == len(fields)

        Returns:
            List[ClassificationResult] 每个字段对应一个分类结果，顺序和 fields 一致
        """
        # 性能测量开始
        perf_start_time = time.perf_counter()
        if _HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            perf_mem_before = process.memory_info().rss
            perf_mem_peak = perf_mem_before
        else:
            perf_mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            perf_mem_peak = perf_mem_before

        # 输入验证
        if not fields:
            raise ValueError("fields cannot be empty")

        for field in fields:
            validate_input(field)

        # 使用分类体系
        categories = hierarchical_categories or self.hierarchical_categories
        if categories is None:
            raise ValueError(
                "hierarchical_categories must be provided either "
                "at initialization or classification time"
            )
        # 动态传入的 categories 才需要校验（初始化时传入的已在 __init__ 校验过）
        if hierarchical_categories is not None:
            validate_categories(categories)

        # 配置
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )
        multiple = allow_multiple if allow_multiple is not None else self.allow_multiple
        rag_enabled = enable_rag if enable_rag is not None else self.settings.enable_rag

        # Check if we actually have examples to use
        if rag_enabled and (self.vector_store is None or self.vector_store.size == 0):
            logger.warning(
                "RAG enabled but no examples available, disabling for this classification"
            )
            rag_enabled = False

        def _update_mem_peak():
            if _HAS_PSUTIL:
                return process.memory_info().rss
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

        # Bulk Table Mode: entire table processed in one graph run, one LLM call per step
        logger.info(
            "Starting BULK table classification: %d fields will be processed in %d total LLM calls...",
            len(fields),
            3 + (1 if enable_rag else 0),
        )

        # Full initial state for bulk processing
        initial_state: ClassificationState = {
            "input": None,
            "inputs": fields,
            "table_chinese_name": table_chinese_name,
            "hierarchical_categories": categories,
            "confidence_threshold": threshold,
            "allow_multiple": multiple,
            "rag_enabled": rag_enabled,
            "bulk_mode": True,
            "ground_truth_data_items": None,
            "ground_truth_list": ground_truth_list,
            "remaining_ground_truth": None,
            "evaluation": None,
            "retrieved_examples": None,
            "table_context_analysis": None,
            # Bulk mode results
            "bulk_feature_analysis": None,
            "bulk_preliminary_classification": None,
            "bulk_verification": None,
            "bulk_final_results": None,
            # Legacy fields for compatibility
            "feature_analysis": None,
            "preliminary_classification": None,
            "verification": None,
            "completed_results": None,
            "reclassification_count": 0,
            "hallucinated_data_items": [],
            "_final_predictions": None,
            "_final_labels": None,
            "_final_avg_confidence": None,
            "_reasoning_chain": None,
        }

        # Run the entire bulk graph
        result_state = self.graph.invoke(initial_state)

        # Get the bulk final results
        completed_results = result_state.get("bulk_final_results", [])

        # If evaluation was run, attach the evaluation to the result (optional)
        evaluation = result_state.get("evaluation")

        logger.info(
            "Bulk classification complete: %d fields processed with just %d LLM calls total",
            len(completed_results),
            3 + (1 if enable_rag else 0) + (1 if evaluation else 0),
        )

        # Ensure we return the same number of results as input fields
        if len(completed_results) != len(fields) and len(completed_results) > 0:
            logger.warning(
                "Number of bulk results (%d) doesn't match number of input fields (%d)",
                len(completed_results),
                len(fields),
            )

        # Performance measurement end
        perf_elapsed = time.perf_counter() - perf_start_time
        perf_mem_after = _update_mem_peak()
        if perf_mem_after > perf_mem_peak:
            perf_mem_peak = perf_mem_after
        logger.info(
            "分类处理完成 [Bulk模式]: %d 个字段, 耗时: %.2f秒",
            len(fields),
            perf_elapsed,
        )
        logger.info(
            "内存使用: 开始=%dMB, 峰值=%dMB, 结束=%dMB",
            perf_mem_before // 1024 // 1024,
            perf_mem_peak // 1024 // 1024,
            perf_mem_after // 1024 // 1024,
        )

        # Return both results and evaluation (if any)
        return {"results": completed_results, "evaluation": evaluation}

    def _classify_single_with_context(
        self,
        field_input: TableFieldInput,
        table_context: Optional[TableContextAnalysis],
        retrieved_examples: Optional[List[RetrievedExample]],
        hierarchical_categories: Optional[List[HierarchicalCategory]] = None,
        confidence_threshold: Optional[float] = None,
        allow_multiple: Optional[bool] = None,
        enable_rag: Optional[bool] = None,
        ground_truth_data_items: Optional[List[str]] = None,
    ) -> ClassificationResult:
        """Classify a single field with pre-computed shared table context (for parallel processing)"""
        categories = hierarchical_categories or self.hierarchical_categories
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )
        multiple = allow_multiple if allow_multiple is not None else self.allow_multiple
        rag_enabled = enable_rag if enable_rag is not None else self.settings.enable_rag

        if rag_enabled and (self.vector_store is None or self.vector_store.size == 0):
            rag_enabled = False

        # Initial state with shared context already provided
        initial_state: ClassificationState = {
            "input": field_input,
            "hierarchical_categories": categories,
            "confidence_threshold": threshold,
            "allow_multiple": multiple,
            "rag_enabled": rag_enabled,
            "ground_truth_data_items": ground_truth_data_items,
            "evaluation": None,
            "retrieved_examples": retrieved_examples,
            "table_context_analysis": table_context,
            "feature_analysis": None,
            "preliminary_classification": None,
            "verification": None,
            "reclassification_count": 0,
            "hallucinated_data_items": [],
            "_final_predictions": None,
            "_final_labels": None,
            "_final_avg_confidence": None,
            "_reasoning_chain": None,
        }

        # Execute the graph from here (already has table context, goes straight to field processing)
        result_state = self.graph.invoke(initial_state)

        # Extract result
        final_predictions: List[PredictedItem] = result_state.get(
            "_final_predictions", []
        )
        final_labels = result_state.get("_final_labels", [])
        final_avg_confidence = result_state.get("_final_avg_confidence", 0.0)
        reasoning_chain = result_state.get("_reasoning_chain", [])

        known_feature_keys = {
            "table_name_keywords",
            "field_name_keywords",
            "description_keywords",
            "semantic_summary",
            "consistency_analysis",
            "dominant_source",
        }
        known_preliminary_keys = {"predictions", "total_confidence"}
        known_verification_keys = {
            "verified_predictions",
            "removed_false_positives",
            "added_missing",
            "average_confidence",
            "cross_validation_note",
            "suggests_reclassification",
        }

        classification_result: ClassificationResult = {
            "final_predictions": final_predictions,
            "final_labels": final_labels,
            "final_confidence": final_avg_confidence,
            "reasoning_chain": reasoning_chain,
            "feature_analysis": FeatureAnalysisResult(
                **{
                    k: v
                    for k, v in (result_state["feature_analysis"] or {}).items()
                    if k in known_feature_keys
                }
            )
            if result_state.get("feature_analysis")
            else None,
            "preliminary_result": PreliminaryResult(
                **{
                    k: v
                    for k, v in (
                        result_state.get("preliminary_classification") or {}
                    ).items()
                    if k in known_preliminary_keys
                }
            )
            if result_state.get("preliminary_classification")
            else None,
            "verification_result": VerificationResult(
                **{
                    k: v
                    for k, v in (result_state.get("verification") or {}).items()
                    if k in known_verification_keys
                }
            )
            if result_state.get("verification")
            else None,
            "evaluation": result_state.get("evaluation"),
            "table_context_analysis": table_context,
        }

        return classification_result

    def classify_table_streaming(
        self,
        csv_path: str | Path,
        batch_size: int = 50,
        table_chinese_name: Optional[str] = None,
        hierarchical_categories: Optional[List[HierarchicalCategory]] = None,
        confidence_threshold: Optional[float] = None,
        allow_multiple: Optional[bool] = None,
        enable_rag: Optional[bool] = None,
        max_workers: int = 3,
        bulk_mode: bool = True,
        encoding: str = "utf-8-sig",
        include_ground_truth: bool = True,
        skip_empty_gt: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        error_callback: Optional[Callable[[Exception, int], None]] = None,
    ) -> Generator[Tuple[List[ClassificationResult], int, int], None, None]:
        """
        Stream process large CSV file in batches, yielding results for each batch.

        This method avoids loading the entire CSV into memory by processing
        in batches. Each batch is classified independently and yielded as a tuple.

        Args:
            csv_path: Path to CSV file to process
            batch_size: Number of records per batch (default 50)
            table_chinese_name: Table Chinese name for context analysis
            hierarchical_categories: Dynamic classification hierarchy
            confidence_threshold: Confidence threshold override
            allow_multiple: Allow multiple labels override
            enable_rag: Enable RAG override
            max_workers: Max parallel workers (only for parallel mode)
            bulk_mode: Use bulk mode for faster processing
            encoding: CSV file encoding (default utf-8-sig)
            include_ground_truth: Whether to parse ground truth labels
            skip_empty_gt: Skip records with empty ground truth
            progress_callback: Callback(batch_description, batch_idx, total_records)
            error_callback: Callback(exception, batch_idx) for error handling

        Yields:
            Tuple of (results_batch, batch_idx, total_records_in_batch)
            Results are List[ClassificationResult] for the current batch

        Example:
            for results, batch_idx, count in agent.classify_table_streaming(
                "large_file.csv",
                batch_size=100,
                progress_callback=lambda desc, idx, total: print(f"Batch {idx}")
            ):
                print(f"Processed batch {batch_idx}: {len(results)} results")
        """
        if progress_callback:
            progress_callback("Starting CSV stream processing", 0, 0)

        try:
            for batch_idx, (inputs, gts, _, total) in enumerate(
                process_large_csv_stream(
                    csv_path,
                    batch_size=batch_size,
                    encoding=encoding,
                    include_ground_truth=include_ground_truth,
                    skip_empty_gt=skip_empty_gt,
                ),
                start=1,
            ):
                if progress_callback:
                    progress_callback(f"Processing batch {batch_idx}", batch_idx, total)

                try:
                    results = self.classify_table(
                        fields=inputs,
                        table_chinese_name=table_chinese_name,
                        hierarchical_categories=hierarchical_categories,
                        confidence_threshold=confidence_threshold,
                        allow_multiple=allow_multiple,
                        enable_rag=enable_rag,
                        ground_truth_list=gts,
                        max_workers=max_workers,
                        bulk_mode=bulk_mode,
                    )
                    yield (results, batch_idx, len(inputs))

                    if progress_callback:
                        progress_callback(
                            f"Completed batch {batch_idx}", batch_idx, total
                        )

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    if error_callback:
                        error_callback(e, batch_idx)
                    continue

        except Exception as e:
            logger.error(f"Fatal error in CSV stream processing: {e}")
            if error_callback:
                error_callback(e, 0)
            raise

        if progress_callback:
            progress_callback("CSV stream processing complete", -1, -1)

    def classify_table_from_csv(
        self,
        csv_path: str | Path,
        table_chinese_name: Optional[str] = None,
        hierarchical_categories: Optional[List[HierarchicalCategory]] = None,
        confidence_threshold: Optional[float] = None,
        allow_multiple: Optional[bool] = None,
        enable_rag: Optional[bool] = None,
        max_workers: int = 3,
        bulk_mode: bool = True,
        encoding: str = "utf-8-sig",
        include_ground_truth: bool = True,
        skip_empty_gt: bool = True,
        streaming_threshold: int = 500,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[ClassificationResult]:
        """
        Classify fields from a CSV file with automatic streaming for large files.

        For small files (< streaming_threshold records), loads all data at once.
        For large files (>= streaming_threshold records), uses streaming mode.

        Args:
            csv_path: Path to CSV file
            table_chinese_name: Table Chinese name for context analysis
            hierarchical_categories: Dynamic classification hierarchy
            confidence_threshold: Confidence threshold override
            allow_multiple: Allow multiple labels override
            enable_rag: Enable RAG override
            max_workers: Max parallel workers
            bulk_mode: Use bulk mode for faster processing
            encoding: CSV file encoding
            include_ground_truth: Whether to parse ground truth labels
            skip_empty_gt: Skip records with empty ground truth
            streaming_threshold: Threshold to switch to streaming mode (default 500)
            progress_callback: Callback(description, batch_idx, total) for progress

        Returns:
            List[ClassificationResult] for all classified fields
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

        inputs, gts = load_data_csv(
            csv_path,
            include_ground_truth=include_ground_truth,
            skip_empty_gt=skip_empty_gt,
            encoding=encoding,
        )

        if len(inputs) < streaming_threshold:
            if progress_callback:
                progress_callback(
                    f"Small file ({len(inputs)} records), loading directly",
                    1,
                    len(inputs),
                )
            return self.classify_table(
                fields=inputs,
                table_chinese_name=table_chinese_name,
                hierarchical_categories=hierarchical_categories,
                confidence_threshold=confidence_threshold,
                allow_multiple=allow_multiple,
                enable_rag=enable_rag,
                ground_truth_list=gts,
                max_workers=max_workers,
                bulk_mode=bulk_mode,
            )

        if progress_callback:
            progress_callback(
                f"Large file ({len(inputs)} records), using streaming", 0, len(inputs)
            )

        all_results = []
        for results, batch_idx, count in self.classify_table_streaming(
            csv_path=csv_path,
            batch_size=50,
            table_chinese_name=table_chinese_name,
            hierarchical_categories=hierarchical_categories,
            confidence_threshold=confidence_threshold,
            allow_multiple=allow_multiple,
            enable_rag=enable_rag,
            max_workers=max_workers,
            bulk_mode=bulk_mode,
            encoding=encoding,
            include_ground_truth=include_ground_truth,
            skip_empty_gt=skip_empty_gt,
            progress_callback=progress_callback,
        ):
            all_results.extend(results)

        return all_results


class AsyncClassificationAgent(ClassificationAgent):
    """
    异步版本的ClassificationAgent (Wave 2 T2.2 异步迁移原型)

    提供异步API，使用asyncio替代ThreadPoolExecutor，支持异步LLM调用
    保持与ClassificationAgent相同的API和行为
    """

    async def aclassify_table(
        self,
        fields: List[TableFieldInput],
        table_chinese_name: Optional[str] = None,
        hierarchical_categories: Optional[List[HierarchicalCategory]] = None,
        confidence_threshold: Optional[float] = None,
        allow_multiple: Optional[bool] = None,
        enable_rag: Optional[bool] = None,
        ground_truth_list: Optional[List[List[str]]] = None,
        max_workers: int = 3,
        bulk_mode: bool = True,
    ) -> List[ClassificationResult]:
        """
        异步版本的classify_table，使用asyncio.gather并行处理字段

        注意：对于bulk_mode=True（表级批量处理），异步版本与同步版本相同，
        因为bulk模式已经通过单次LLM调用处理整个表。异步优势主要在并行模式。

        Args:
            fields: 整张表所有字段
            (其余参数与同步版本相同)

        Returns:
            List[ClassificationResult]: 每个字段的分类结果，顺序和 fields 一致
        """
        import asyncio
        from typing import List

        # 使用缓存性能监控
        from classification_agent.async_utils import AsyncPerformanceMonitor

        perf_monitor = AsyncPerformanceMonitor()

        # 输入验证
        if not fields:
            raise ValueError("fields cannot be empty")

        for field in fields:
            validate_input(field)

        # 使用分类体系
        categories = hierarchical_categories or self.hierarchical_categories
        if categories is None:
            raise ValueError(
                "hierarchical_categories must be provided either "
                "at initialization or classification time"
            )

        # 动态传入的 categories 才需要校验（初始化时传入的已在 __init__ 校验过）
        if hierarchical_categories is not None:
            validate_categories(categories)

        # 应用动态参数
        confidence = confidence_threshold or self.confidence_threshold
        multiple = allow_multiple if allow_multiple is not None else self.allow_multiple
        use_rag = enable_rag if enable_rag is not None else self.settings.enable_rag

        # 确保图已构建
        if self.graph is None:
            self._ensure_graph()

        # 选择处理模式
        if bulk_mode:
            # Bulk模式：表级一次性处理（3-4次LLM调用）
            logger.info(
                f"Starting ASYNC BULK table classification: "
                f"{len(fields)} fields will be processed in ~3 total LLM calls..."
            )

            return await self._bulk_classify_async(
                fields=fields,
                table_chinese_name=table_chinese_name,
                categories=categories,
                confidence=confidence,
                multiple=multiple,
                use_rag=use_rag,
                ground_truth_list=ground_truth_list,
            )
        else:
            # Parallel模式：每个字段独立异步处理
            logger.info(
                f"Starting ASYNC PARALLEL table classification: "
                f"{len(fields)} fields will be processed in parallel"
            )

            return await self._parallel_classify_async(
                fields=fields,
                table_chinese_name=table_chinese_name,
                categories=categories,
                confidence=confidence,
                multiple=multiple,
                use_rag=use_rag,
                ground_truth_list=ground_truth_list,
                max_workers=max_workers,
            )

    async def _bulk_classify_async(
        self,
        fields: List[TableFieldInput],
        table_chinese_name: Optional[str],
        categories: List[HierarchicalCategory],
        confidence: float,
        multiple: bool,
        use_rag: bool,
        ground_truth_list: Optional[List[List[str]]],
    ) -> List[ClassificationResult]:
        """
        异步批量分类（表级一次性处理）

        Bulk模式已经是高效的，异步版本与同步版本基本相同，
        但可能使用异步版本的LLM调用（如果LLM支持）
        """
        # 创建初始状态
        initial_state = {
            "inputs": fields,
            "hierarchical_categories": categories,
            "confidence_threshold": confidence,
            "allow_multiple": multiple,
            "rag_enabled": use_rag,
            "bulk_mode": True,
        }

        if table_chinese_name:
            initial_state["table_chinese_name"] = table_chinese_name

        if ground_truth_list:
            initial_state["ground_truth_list"] = ground_truth_list

        # TODO: 如果LangGraph支持异步调用，使用 ainvoke()
        # 当前使用同步 invoke()，但LLM调用可能是异步的（如果LLM实现支持）
        result_state = self.graph.invoke(initial_state)

        results = result_state.get("results")
        if not results:
            raise ValueError("Processing completed but no results returned")

        # 提取并返回结果
        all_results: List[ClassificationResult] = []
        for idx, field in enumerate(fields):
            field_result = results[idx]
            res = ClassificationResult(
                table_name=field["table_name"],
                field_name=field["field_name"],
                predictions=field_result.get("predictions", []),
                verification_result=field_result.get("verification_result", {}),
                confidence=field_result.get("confidence", 0.0),
                suggests_reclassification=field_result.get(
                    "suggests_reclassification", False
                ),
                feature_analysis=field_result.get("feature_analysis"),
                retrieved_examples=field_result.get("retrieved_examples"),
            )
            if field.get("field_description"):
                res.field_description = field["field_description"]
            all_results.append(res)

        return all_results

    async def _parallel_classify_async(
        self,
        fields: List[TableFieldInput],
        table_chinese_name: Optional[str],
        categories: List[HierarchicalCategory],
        confidence: float,
        multiple: bool,
        use_rag: bool,
        ground_truth_list: Optional[List[List[str]]],
        max_workers: int = 3,
    ) -> List[ClassificationResult]:
        """
        异步并行分类（每个字段独立处理）

        使用asyncio.gather并行处理多个字段，每个字段运行完整的分类流程
        """
        import asyncio

        # 创建任务列表
        tasks = []

        for i, field in enumerate(fields):
            gt = (
                ground_truth_list[i]
                if ground_truth_list and i < len(ground_truth_list)
                else None
            )

            # 包装每个字段的分类作为异步任务
            task = self._process_single_field_async(
                field=field,
                ground_truth=gt,
                table_chinese_name=table_chinese_name,
                categories=categories,
                confidence=confidence,
                multiple=multiple,
                use_rag=use_rag,
            )
            tasks.append((i, asyncio.create_task(task)))

        # 收集结果并保持原始顺序
        results = [None] * len(fields)

        for i, task in tasks:
            try:
                result = await task
                results[i] = result
            except Exception as e:
                logger.error(f"字段 {fields[i]['field_name']} 处理失败: {e}")
                # 创建错误结果占位符
                results[i] = ClassificationResult(
                    table_name=fields[i]["table_name"],
                    field_name=fields[i]["field_name"],
                    predictions=[],
                    verification_result={},
                    confidence=0.0,
                    suggests_reclassification=False,
                )
                if fields[i].get("field_description"):
                    results[i].field_description = fields[i]["field_description"]

        return results

    async def _process_single_field_async(
        self,
        field: TableFieldInput,
        ground_truth: Optional[List[str]],
        table_chinese_name: Optional[str],
        categories: List[HierarchicalCategory],
        confidence: float,
        multiple: bool,
        use_rag: bool,
    ) -> ClassificationResult:
        """
        异步处理单个字段

        为每个字段创建独立的分类流程，使用异步执行
        """
        # 创建该字段的初始状态
        initial_state = {
            "input": field,
            "hierarchical_categories": categories,
            "confidence_threshold": confidence,
            "allow_multiple": multiple,
            "rag_enabled": use_rag,
            "bulk_mode": False,
        }

        if table_chinese_name:
            initial_state["table_chinese_name"] = table_chinese_name

        if ground_truth:
            initial_state["ground_truth_data_items"] = ground_truth

        # TODO: 使用异步图调用 ainvoke() 如果可用
        # 当前使用同步 invoke()，但在 async context 中的线程池执行
        result_state = self.graph.invoke(initial_state)

        # 提取结果
        res = ClassificationResult(
            table_name=field["table_name"],
            field_name=field["field_name"],
            predictions=result_state.get("predictions", []),
            verification_result=result_state.get("verification_result", {}),
            confidence=result_state.get("confidence", 0.0),
            suggests_reclassification=result_state.get(
                "suggests_reclassification", False
            ),
            feature_analysis=result_state.get("feature_analysis"),
            retrieved_examples=result_state.get("retrieved_examples"),
        )

        if field.get("field_description"):
            res.field_description = field["field_description"]

        return res

    async def aclassify_table_from_csv(
        self,
        csv_path: str,
        table_chinese_name: Optional[str] = None,
        hierarchical_categories: Optional[List[HierarchicalCategory]] = None,
        confidence_threshold: Optional[float] = None,
        allow_multiple: Optional[bool] = None,
        enable_rag: Optional[bool] = None,
        encoding: str = "utf-8-sig",
        include_ground_truth: bool = False,
        skip_empty_gt: bool = False,
        streaming_threshold: int = 500,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        max_workers: int = 3,
        bulk_mode: bool = True,
    ) -> List[ClassificationResult]:
        """
        异步从CSV文件分类

        自动选择流式或批量加载，然后使用异步分类
        """
        # 检查文件大小决定是否使用流式处理
        import os

        file_size = os.path.getsize(csv_path)

        # 使用同步助手函数加载数据（在async context中通过线程池执行）
        from classification_agent.utils.data_reader import load_data_csv

        async def load_data():
            import asyncio

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: load_data_csv(
                    csv_path,
                    include_ground_truth=include_ground_truth,
                    encoding=encoding,
                    skip_empty_gt=skip_empty_gt,
                ),
            )

        inputs, ground_truth_list = await load_data()

        if not inputs:
            return []

        # 决定是否使用流式处理
        if len(inputs) >= streaming_threshold:
            if progress_callback:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: progress_callback(
                        f"Large file ({len(inputs)} records), using streaming",
                        0,
                        len(inputs),
                    ),
                )

            return await self._streaming_classify_async(
                csv_path=csv_path,
                table_chinese_name=table_chinese_name,
                hierarchical_categories=hierarchical_categories,
                confidence_threshold=confidence_threshold,
                allow_multiple=allow_multiple,
                enable_rag=enable_rag,
                encoding=encoding,
                include_ground_truth=include_ground_truth,
                skip_empty_gt=skip_empty_gt,
                max_workers=max_workers,
                bulk_mode=bulk_mode,
                progress_callback=progress_callback,
            )
        else:
            # 直接处理全量数据
            return await self.aclassify_table(
                fields=inputs,
                table_chinese_name=table_chinese_name,
                hierarchical_categories=hierarchical_categories,
                confidence_threshold=confidence_threshold,
                allow_multiple=allow_multiple,
                enable_rag=enable_rag,
                ground_truth_list=ground_truth_list if include_ground_truth else None,
                max_workers=max_workers,
                bulk_mode=bulk_mode,
            )

    async def _streaming_classify_async(
        self,
        csv_path: str,
        table_chinese_name: Optional[str],
        hierarchical_categories: Optional[List[HierarchicalCategory]],
        confidence_threshold: Optional[float],
        allow_multiple: Optional[bool],
        enable_rag: Optional[bool],
        encoding: str,
        include_ground_truth: bool,
        skip_empty_gt: bool,
        max_workers: int,
        bulk_mode: bool,
        progress_callback: Optional[Callable[[str, int, int], None]],
    ) -> List[ClassificationResult]:
        """
        异步流式分类大文件

        逐批读取CSV并异步处理每批
        """
        import asyncio
        from classification_agent.utils.data_reader import load_data_csv_stream

        all_results = []

        # 使用流式读取器
        stream_reader = load_data_csv_stream(
            csv_path,
            include_ground_truth=include_ground_truth,
            encoding=encoding,
            skip_empty_gt=skip_empty_gt,
            batch_size=50,  # 每批大小
        )

        batch_idx = 0
        total_processed = 0

        for inputs, ground_truths in stream_reader:
            if not inputs:
                continue

            # 异步处理当前批
            batch_results = await self.aclassify_table(
                fields=inputs,
                table_chinese_name=table_chinese_name,
                hierarchical_categories=hierarchical_categories,
                confidence_threshold=confidence_threshold,
                allow_multiple=allow_multiple,
                enable_rag=enable_rag,
                ground_truth_list=ground_truths if include_ground_truth else None,
                max_workers=max_workers,
                bulk_mode=bulk_mode,
            )

            all_results.extend(batch_results)
            total_processed += len(inputs)

            # 进度回调
            if progress_callback:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: progress_callback(
                        f"Processing large file", batch_idx, total_processed
                    ),
                )

            batch_idx += 1

        return all_results
