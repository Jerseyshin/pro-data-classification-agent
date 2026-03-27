#!/usr/bin/env python3
"""
CLI入口 - 整张表字段批量分类工具，支持BGE RAG
Usage:
    python main.py --input data.csv [--output output] [--encoding GB18030] [--with-ground-truth] [--enable-rag] [--rag-top-k 10]

Modes:
    1. Bulk Mode (default): 整张表所有字段一次性处理，每个步骤只调用一次LLM。
       对于100个字段的表，也只需要 ~3-4次LLM调用，速度最快。
    2. RAG Mode: 使用BGE-large-zh-v1.5模型从data.csv加载相似示例，top_k=10

Requirements:
    1. Uses context_analysis: 启用表级上下文分析（默认开启）
    2. Whole table prediction: 对整张表所有字段批量分类
    3. Bulk mode only: 仅支持bulk模式，整表全字段一次性处理
    4. Output result documents: 输出 CSV 和 Markdown 结果文件
    5. RAG support: 使用本地BGE模型进行中文相似度检索
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加 src 目录到 Python 路径，以便导入模块
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from classification_agent import ClassificationAgent
    from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
    from classification_agent.llm.openai_wrapper import OpenAILLM
    from classification_agent.utils.data_reader import load_data_csv, load_rag_training_data
    from classification_agent.utils.result_exporter import export_batch_results
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"请确保在项目根目录运行: {Path(__file__).parent}")
    sys.exit(1)

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Whole table hierarchical classification with LangGraph\n"
        "Bulk Mode only: 3-4 LLM calls TOTAL for entire table\n"
        "RAG support with BGE-large-zh-v1.5 model, top_k=10"
    )
    parser.add_argument(
        "--input", required=True, help="Input CSV file with table fields to classify"
    )
    parser.add_argument(
        "--input-encoding",
        default="utf-8-sig",
        help="Input file encoding (default: utf-8-sig, can use GB18030)",
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Output directory for results"
    )
    parser.add_argument(
        "--output-base-name",
        default="classification_results",
        help="Base name for output files",
    )
    parser.add_argument(
        "--output-encoding",
        default="utf-8",
        help="Output file encoding (default: utf-8, can use GB18030)",
    )
    parser.add_argument(
        "--with-ground-truth",
        action="store_true",
        help="Input CSV contains ground truth, enable evaluation",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7, help="Confidence threshold"
    )
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        default=True,
        help="Allow multiple labels per field",
    )
    parser.add_argument(
        "--no-multiple",
        dest="allow_multiple",
        action="store_false",
        help="Disable multiple labels",
    )
    parser.add_argument(
        "--enable-rag", 
        action="store_true", 
        help="Enable RAG retrieval with BGE-large-zh-v1.5 model"
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=10,
        help="Number of similar examples to retrieve (default: 10)",
    )
    parser.add_argument(
        "--rag-similarity-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for RAG retrieval (default: 0.5)",
    )
    parser.add_argument(
        "--rag-data",
        default="data.csv",
        help="RAG training data CSV file (default: data.csv)",
    )
    parser.add_argument(
        "--rag-embedding-provider",
        default="bge",
        choices=["bge", "openai"],
        help="Embedding provider for RAG: 'bge' for local BGE model, 'openai' for OpenAI API (default: bge)",
    )

    args = parser.parse_args()

    # Initialize LLM from environment
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable must be set in .env")
        exit(1)

    llm = OpenAILLM(api_key=api_key, model=model, base_url=base_url)

    # Load default categories
    categories = DEFAULT_DATA_CATEGORIES

    # Initialize agent with requested configuration
    print(f"Initializing ClassificationAgent...")
    print(f"  - RAG enabled: {args.enable_rag}")
    
    if args.enable_rag:
        print(f"  - RAG embedding provider: {args.rag_embedding_provider}")
        print(f"  - RAG top_k: {args.rag_top_k}")
        print(f"  - RAG similarity threshold: {args.rag_similarity_threshold}")
        print(f"  - RAG training data: {args.rag_data}")
    
    agent = ClassificationAgent(
        llm=llm,
        hierarchical_categories=categories,
        confidence_threshold=args.confidence_threshold,
        allow_multiple=args.allow_multiple,
        enable_table_context=True,  # requirement 1: use context_analysis
        enable_rag=args.enable_rag,
        rag_top_k=args.rag_top_k,
        rag_similarity_threshold=args.rag_similarity_threshold,
        rag_embedding_provider=args.rag_embedding_provider,
        bge_model_name="BAAI/bge-large-zh-v1.5",  # Use BGE model for Chinese
    )

    # Load RAG training data if enabled
    if args.enable_rag:
        rag_data_path = Path(args.rag_data)
        if not rag_data_path.exists():
            print(f"Warning: RAG data file not found: {rag_data_path}")
            print("RAG will be disabled as no training examples are available.")
        else:
            print(f"Loading RAG training data from {rag_data_path}...")
            try:
                rag_examples = load_rag_training_data(
                    csv_path=rag_data_path,
                    hierarchical_categories=categories,
                    skip_if_not_found=True,
                    encoding=args.input_encoding,
                )
                print(f"Loaded {len(rag_examples)} RAG training examples")
                
                # Add examples to agent
                agent.add_rag_examples(rag_examples)
                print("RAG examples added to vector store")
            except Exception as e:
                print(f"Error loading RAG data: {e}")
                print("RAG will be disabled.")

    # Load input data from CSV
    print(f"Loading input from {args.input} (encoding={args.input_encoding})...")
    if args.with_ground_truth:
        inputs, ground_truth_list = load_data_csv(
            args.input,
            include_ground_truth=True,
            skip_empty_gt=True,
            encoding=args.input_encoding,
        )
    else:
        inputs, ground_truth_list = load_data_csv(
            args.input,
            include_ground_truth=False,
            encoding=args.input_encoding,
        )

    if not inputs:
        print("Error: No valid input fields loaded from CSV")
        exit(1)

    print(f"Loaded {len(inputs)} fields for classification")

    # Run whole table classification (only bulk mode)
    total_llm_calls = 3  # feature analysis + preliminary classification + verification
    if args.enable_rag:
        total_llm_calls += 1  # RAG retrieval
    
    print(
        f"Starting classification in BULK MODE: {len(inputs)} fields will be processed in ~{total_llm_calls} total LLM calls..."
    )

    classification_result = agent.classify_table(
        fields=inputs,
        ground_truth_list=ground_truth_list,
    )

    # Extract results list and evaluation from the returned dictionary
    results = classification_result.get("results", [])
    evaluation = classification_result.get("evaluation")

    print(f"Classification completed: {len(results)} fields processed")

    # Export results to CSV and Markdown
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if evaluation:
        print(f"\nEvaluation results:")
        print(f"  Total samples: {evaluation['total_samples']}")
        print(f"  Exact match accuracy: {evaluation['exact_match_accuracy']:.4f}")
        print(f"  Macro Precision: {evaluation['macro_precision']:.4f}")
        print(f"  Macro Recall: {evaluation['macro_recall']:.4f}")
        print(f"  Macro F1: {evaluation['macro_f1']:.4f}")

    # Export both CSV and Markdown
    csv_path, md_path = export_batch_results(
        outputs_dir=output_dir,
        inputs=inputs,
        results=results,
        evaluation=evaluation,
        base_name=args.output_base_name,
        encoding=args.output_encoding,
    )

    print(f"\nResults exported:")
    print(f"  CSV:  {csv_path}")
    print(f"  Markdown:  {md_path}")
    
    # Show table context analysis if available
    if results and 'table_context_analysis' in results[0] and results[0]['table_context_analysis'] is not None:
        ctx = results[0]['table_context_analysis']
        print(f"\nTable Context Analysis:")
        print(f"  - Business scenario: {ctx['business_scenario'][:100]}...")
        print(f"  - Table type/role: {ctx['table_type']}")
        print(f"  - Core business objects: {', '.join(ctx['core_business_objects'])}")
        print(f"  - Key business concepts: {', '.join(ctx['key_business_concepts'])}")
        print(f"  - Overall description: {ctx['overall_description'][:100]}...")


if __name__ == "__main__":
    main()
