from typing import List, Optional, Tuple
import numpy as np
from openai import OpenAI
from langgraph.graph import StateGraph

from classification_agent.graph.state import ClassificationState
from classification_agent.graph.builder import build_graph
from classification_agent.types.schemas import (
    TableFieldInput,
    HierarchicalCategory,
    FeatureAnalysisResult,
    PreliminaryResult,
    VerificationResult,
    ClassificationResult,
    PredictedItem
)
from classification_agent.llm.base import BaseLLM
from classification_agent.llm.openai_wrapper import OpenAILLM
from classification_agent.config.settings import Settings
from classification_agent.utils.validation import validate_input, validate_categories
from classification_agent.rag.embeddings import OpenAIEmbeddings
from classification_agent.rag.vector_store import InMemoryVectorStore
from classification_agent.utils.logging import get_logger

logger = get_logger(__name__)


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
        rag_examples: Optional[List[Tuple[TableFieldInput, HierarchicalCategory]]] = None,
        rag_top_k: Optional[int] = None,
        rag_similarity_threshold: Optional[float] = None,
        rag_embedding_provider: Optional[str] = None,
        # BGE options
        bge_model_name: Optional[str] = None,
        bge_device: Optional[str] = None,
        bge_use_fp16: Optional[bool] = None,
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
        """
        self.settings = settings or Settings()

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
            self.embeddings
        ).compile()

    def _create_default_llm(self) -> BaseLLM:
        """Create default OpenAI LLM from settings"""
        return OpenAILLM(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
            base_url=self.settings.openai_base_url
        )

    def _init_rag(self, rag_examples: Optional[List[Tuple[TableFieldInput, HierarchicalCategory]]]) -> None:
        """Initialize RAG components if enabled"""
        if not self.settings.enable_rag:
            self.vector_store = None
            self.embeddings = None
            return

        # Initialize embeddings based on provider
        if self.settings.rag_embedding_provider == "bge":
            # Use local BGE model (BAAI/bge-large-zh-v1.5)
            from classification_agent.rag import BGEEmbeddings
            self.embeddings = BGEEmbeddings(
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
                    base_url=self.settings.openai_base_url
                )

            self.embeddings = OpenAIEmbeddings(
                client=client,
                model=self.settings.rag_embedding_model,
            )

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
        if not self.settings.enable_rag or self.vector_store is None or self.embeddings is None:
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
            if input_example.get('field_description'):
                parts.append(f"Description: {input_example['field_description']}")
            texts.append(' '.join(parts))
            valid_examples.append((input_example, label))

        # Get embeddings
        embeddings_result = self.embeddings.embed_texts(texts)
        if len(embeddings_result) != len(texts):
            # Mismatch - filter out any failed embeddings
            valid_pairs = [
                (np.array(emb), input_example, label)
                for emb, (input_example, label) in zip(embeddings_result, valid_examples)
                if emb is not None
            ]
            if not valid_pairs:
                logger.warning("No valid embeddings obtained, nothing added")
                return
            embeddings_list = [p[0] for p in valid_pairs]
            valid_examples = [(p[1], p[2]) for p in valid_pairs]

        else:
            embeddings_list = [np.array(emb) for emb in embeddings_result if emb is not None]

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
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        multiple = allow_multiple if allow_multiple is not None else self.allow_multiple

        # Determine if RAG is enabled for this classification
        # If explicitly provided, use that; otherwise use settings default
        rag_enabled = enable_rag if enable_rag is not None else self.settings.enable_rag

        # Check if we actually have examples to use
        if rag_enabled and (self.vector_store is None or self.vector_store.size == 0):
            logger.warning("RAG enabled but no examples available, disabling for this classification")
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
            "reclassification_count": 0
        }

        # 执行图
        result_state = self.graph.invoke(initial_state)

        # 提取结果
        final_predictions: List[PredictedItem] = result_state.get("_final_predictions", [])
        final_labels = result_state.get("_final_labels", [])
        final_avg_confidence = result_state.get("_final_avg_confidence", 0.0)
        reasoning_chain = result_state.get("_reasoning_chain", [])

        # 构建返回结果
        # 用 dict 过滤多余字段，避免 LLM 返回额外 key 导致 TypedDict 展开报 TypeError
        known_feature_keys = {"table_name_keywords", "field_name_keywords", "description_keywords",
                              "semantic_summary", "consistency_analysis", "dominant_source"}
        known_preliminary_keys = {"predictions", "total_confidence"}
        known_verification_keys = {"verified_predictions", "removed_false_positives", "added_missing",
                                   "average_confidence", "cross_validation_note", "suggests_reclassification"}

        classification_result: ClassificationResult = {
            "final_predictions": final_predictions,
            "final_labels": final_labels,
            "final_confidence": final_avg_confidence,
            "reasoning_chain": reasoning_chain,
            "feature_analysis": FeatureAnalysisResult(
                **{k: v for k, v in result_state["feature_analysis"].items() if k in known_feature_keys}
            ),
            "preliminary_result": PreliminaryResult(
                **{k: v for k, v in result_state["preliminary_classification"].items() if k in known_preliminary_keys}
            ),
            "verification_result": VerificationResult(
                **{k: v for k, v in result_state["verification"].items() if k in known_verification_keys}
            ),
            "evaluation": result_state.get("evaluation"),
        }

        return classification_result
