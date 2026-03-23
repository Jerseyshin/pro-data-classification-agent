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
        # Speed optimization
        fast_mode: Optional[bool] = None,
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
            fast_mode: 是否开启快速模式，合并特征分析+初步分类为一次LLM调用，
                减少约33%的网络往返，速度更快但精度可能略有下降
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
        # Apply fast mode override
        if fast_mode is not None:
            self.settings.fast_mode = fast_mode

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
                **{k: v for k, v in (result_state.get("verification") or {}).items() if k in known_verification_keys}
            ),
            "evaluation": result_state.get("evaluation"),
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
        对整张表所有字段批量分类，一次LLM调用完成所有分类（大幅减少网络往返）

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
        from classification_agent.prompts.loader import load_prompt

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
        # 动态传入的 categories 才需要校验
        if hierarchical_categories is not None:
            validate_categories(categories)

        # 配置
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        multiple = allow_multiple if allow_multiple is not None else self.allow_multiple

        # Determine if RAG is enabled for this classification
        rag_enabled = enable_rag if enable_rag is not None else self.settings.enable_rag

        # Check if we actually have examples to use
        if rag_enabled and (self.vector_store is None or self.vector_store.size == 0):
            logger.warning("RAG enabled but no examples available, disabling for this classification")
            rag_enabled = False

        # Get table name from first field (all fields should have same table name)
        table_name = fields[0]["table_name"]

        # Step 1: table context analysis if enabled
        table_context = None
        if self.enable_table_context:
            prompt_ctx = load_prompt(
                "context_analysis.jinja2",
                table_name=table_name,
                table_chinese_name=table_chinese_name,
                fields=[{"field_name": f["field_name"], "field_description": f.get("field_description")} for f in fields],
            )
            result_ctx = self.llm.generate_json(prompt_ctx)
            table_context = {
                "table_name": table_name,
                "table_chinese_name": table_chinese_name,
                "inferred_purpose": result_ctx.get("inferred_purpose", ""),
                "key_business_concepts": result_ctx.get("key_business_concepts", []),
                "overall_data_category": result_ctx.get("overall_data_category", ""),
            }
            logger.info(
                "ClassificationAgent.table_context_analysis: purpose=%s, category=%s",
                table_context["inferred_purpose"][:50],
                table_context["overall_data_category"],
            )

        # Step 2: RAG retrieval if needed (retrieve once for whole table based on table context)
        retrieved_examples = None
        if rag_enabled and self.vector_store and self.embeddings:
            # Build query text from table context + all field names
            query_parts = [f"Table: {table_name}"]
            if table_context:
                query_parts.append(f"Purpose: {table_context['inferred_purpose']}")
                query_parts.extend(table_context["key_business_concepts"])
            for f in fields[:10]:  # limit to first 10 to avoid too long query
                query_parts.append(f["field_name"])
            query_text = " ".join(query_parts)

            query_emb = self.embeddings.embed_text(query_text)
            if query_emb is not None:
                retrieved_examples = self.vector_store.search(
                    np.array(query_emb),
                    self.settings.rag_top_k,
                    self.settings.rag_similarity_threshold,
                )
            else:
                logger.warning("Failed to get embedding for RAG query, proceeding without RAG")

        # Step 3: Batch classification for all fields
        prompt_batch = load_prompt(
            "batch_table_classification.jinja2",
            table_name=table_name,
            table_chinese_name=table_chinese_name,
            table_context=table_context,
            fields=fields,
            hierarchical_categories=categories,
            retrieved_examples=retrieved_examples,
            allow_multiple=multiple,
        )

        result_batch = self.llm.generate_json(prompt_batch)
        batch_results_data = result_batch.get("results", [])

        # Step 4: Process each result into ClassificationResult
        from classification_agent.types.schemas import (
            FeatureAnalysisResult, PreliminaryResult, VerificationResult,
        )
        known_feature_keys = {"table_name_keywords", "field_name_keywords", "description_keywords",
                              "semantic_summary", "consistency_analysis", "dominant_source"}
        known_preliminary_keys = {"predictions", "total_confidence"}

        final_results: List[ClassificationResult] = []

        for i, (field, result_data) in enumerate(zip(fields, batch_results_data)):
            # Get feature analysis from batch result
            feature_analysis_data = result_data.get("feature_analysis", {})
            feature_analysis = FeatureAnalysisResult(
                **{k: v for k, v in feature_analysis_data.items() if k in known_feature_keys}
            )

            # Get preliminary classification from batch result
            preliminary_data = {
                "predictions": result_data.get("predictions", []),
                "total_confidence": result_data.get("total_confidence", 0.0),
            }
            preliminary_result = PreliminaryResult(
                **{k: v for k, v in preliminary_data.items() if k in known_preliminary_keys}
            )

            # Collect predictions and labels for final result
            predictions = preliminary_data.get("predictions", [])
            final_labels = [p["data_item"] for p in predictions]
            final_avg_confidence = (
                sum(p["confidence"] for p in predictions) / len(predictions)
                if predictions else 0.0
            )

            # Self-verification is skipped in batch mode for speed
            # If you need verification, use individual classify() calls
            verification_result: VerificationResult = {
                "verified_predictions": [],
                "removed_false_positives": [],
                "added_missing": [],
                "average_confidence": final_avg_confidence,
                "cross_validation_note": "",
                "suggests_reclassification": False,
            }

            # Handle evaluation if ground truth provided
            evaluation_result = None
            if ground_truth_list and i < len(ground_truth_list):
                # Run simple evaluation right here for this field
                gt = ground_truth_list[i]
                pred_set = set(final_labels)
                gt_set = set(gt)
                correct = list(pred_set & gt_set)
                wrong = list(pred_set - gt_set)
                missing = list(gt_set - pred_set)
                exact_match = pred_set == gt_set

                from classification_agent.types.schemas import EvaluationResult, SingleEvaluationResult
                single_result: SingleEvaluationResult = {
                    "predicted_data_items": final_labels,
                    "ground_truth_data_items": gt,
                    "correct_predictions": correct,
                    "wrong_predictions": wrong,
                    "missing_predictions": missing,
                    "exact_match": exact_match,
                }
                evaluation_result: EvaluationResult = {
                    "total_samples": 1,
                    "exact_match_count": 1 if exact_match else 0,
                    "exact_match_accuracy": 1.0 if exact_match else 0.0,
                    "total_true_positives": len(correct),
                    "total_false_positives": len(wrong),
                    "total_false_negatives": len(missing),
                    "macro_precision": len(correct) / (len(correct) + len(wrong)) if (len(correct) + len(wrong)) > 0 else 0.0,
                    "macro_recall": len(correct) / (len(correct) + len(missing)) if (len(correct) + len(missing)) > 0 else 0.0,
                    "macro_f1": 0.0,
                    "per_sample_results": [single_result],
                }
                if (len(correct) + len(wrong)) > 0 and (len(correct) + len(missing)) > 0:
                    p = len(correct) / (len(correct) + len(wrong))
                    r = len(correct) / (len(correct) + len(missing))
                    if p + r > 0:
                        evaluation_result["macro_f1"] = 2 * p * r / (p + r)

            classification_result: ClassificationResult = {
                "final_predictions": predictions,
                "final_labels": final_labels,
                "final_confidence": final_avg_confidence,
                "reasoning_chain": [
                    f"table_context: {table_context['inferred_purpose'] if table_context else 'none'}",
                    f"batch_classification: {len(fields)} fields in one call",
                ],
                "feature_analysis": feature_analysis,
                "preliminary_result": preliminary_result,
                "verification_result": verification_result,
                "evaluation": evaluation_result,
            }

            final_results.append(classification_result)

        logger.info(
            "ClassificationAgent.classify_table: processed %d fields in one LLM call",
            len(fields)
        )

        return final_results
