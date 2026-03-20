"""
Result exporter: export batch evaluation results to structured table (CSV/Markdown).

Includes:
- Global metrics (total samples, accuracy, precision, recall, F1)
- Per-sample row with all intermediate node results
- Ground truth and match status when available
"""

import csv
from typing import List, Optional, TextIO
from pathlib import Path

from classification_agent.types.schemas import (
    ClassificationResult,
    EvaluationResult,
    SingleEvaluationResult,
    TableFieldInput,
)


def export_results_to_csv(
    file_path: str | Path,
    inputs: List[TableFieldInput],
    results: List[ClassificationResult],
    evaluation: Optional[EvaluationResult] = None,
) -> None:
    """
    Export batch classification results to CSV file.

    CSV columns:
        index, table_name, field_name, field_description,
        feature_analysis_summary, feature_dominant_source,
        preliminary_num_predictions, preliminary_avg_confidence,
        verification_num_kept, verification_avg_confidence,
        final_labels, final_confidence,
        ground_truth, is_exact_match, correct, wrong, missing

    Args:
        file_path: Output CSV file path
        inputs: List of input table fields
        results: List of corresponding classification results
        evaluation: Optional overall evaluation result (when ground truth provided)
    """
    file_path = Path(file_path)

    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = [
            'index',
            'table_name',
            'field_name',
            'field_description',
            'feature_analysis_summary',
            'feature_dominant_source',
            'preliminary_num_predictions',
            'preliminary_avg_confidence',
            'verification_num_kept',
            'verification_avg_confidence',
            'final_labels',
            'final_avg_confidence',
            'ground_truth',
            'is_exact_match',
            'correct_predictions',
            'wrong_predictions',
            'missing_predictions',
        ]
        writer.writerow(header)

        # Write each row
        for idx, (inp, result) in enumerate(zip(inputs, results)):
            # Extract feature analysis
            fa = result['feature_analysis']
            fa_summary = fa['semantic_summary']
            fa_dominant = fa['dominant_source']

            # Extract preliminary classification
            pc = result['preliminary_result']
            pc_num = len(pc['predictions'])
            pc_avg_conf = pc['total_confidence']

            # Extract verification with defaults for safety
            vr = result['verification_result']
            vr_num_kept = len(vr.get('verified_predictions', []))
            vr_avg_conf = vr.get('average_confidence', 0.0)

            # Final result
            final_labels = '; '.join(result['final_labels'])
            final_avg_conf = result['final_confidence']

            # Evaluation info
            gt = ''
            is_match = ''
            correct = ''
            wrong = ''
            missing = ''

            if evaluation and idx < len(evaluation['per_sample_results']):
                per_sample = evaluation['per_sample_results'][idx]
                gt = '; '.join(per_sample['ground_truth_data_items'])
                is_match = str(per_sample['exact_match'])
                correct = '; '.join(per_sample['correct_predictions'])
                wrong = '; '.join(per_sample['wrong_predictions'])
                missing = '; '.join(per_sample['missing_predictions'])

            row = [
                idx + 1,
                inp['table_name'],
                inp['field_name'],
                inp.get('field_description', ''),
                fa_summary,
                fa_dominant,
                pc_num,
                f"{pc_avg_conf:.4f}",
                vr_num_kept,
                f"{vr_avg_conf:.4f}",
                final_labels,
                f"{final_avg_conf:.4f}",
                gt,
                is_match,
                correct,
                wrong,
                missing,
            ]
            writer.writerow(row)


def export_results_to_markdown(
    file_path: str | Path,
    inputs: List[TableFieldInput],
    results: List[ClassificationResult],
    evaluation: Optional[EvaluationResult] = None,
) -> None:
    """
    Export batch classification results to Markdown file with:
    1. Global metrics summary section
    2. Per-sample detailed table

    Args:
        file_path: Output Markdown file path
        inputs: List of input table fields
        results: List of corresponding classification results
        evaluation: Optional overall evaluation result (when ground truth provided)
    """
    file_path = Path(file_path)

    with open(file_path, 'w', encoding='utf-8') as f:
        # Title
        f.write("# Classification Batch Evaluation Results\n\n")

        # Global metrics
        if evaluation:
            f.write("## Overall Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total samples | {evaluation['total_samples']} |\n")
            f.write(f"| Exact match count | {evaluation['exact_match_count']} |\n")
            f.write(f"| **Exact match accuracy** | **{evaluation['exact_match_accuracy']:.4f}** |\n")
            f.write(f"| Macro precision | {evaluation['macro_precision']:.4f} |\n")
            f.write(f"| Macro recall | {evaluation['macro_recall']:.4f} |\n")
            f.write(f"| **Macro F1** | **{evaluation['macro_f1']:.4f}** |\n")
            f.write(f"| Total true positives | {evaluation['total_true_positives']} |\n")
            f.write(f"| Total false positives | {evaluation['total_false_positives']} |\n")
            f.write(f"| Total false negatives | {evaluation['total_false_negatives']} |\n")
            f.write("\n---\n\n")

        # Detailed results table
        f.write("## Detailed Results by Sample\n\n")

        # Table header
        if evaluation:
            f.write("| # | Table | Field | Final Labels | GT | Match | Confidence |\n")
            f.write("|---|-------|-------|-------------|----|-------|------------|\n")
        else:
            f.write("| # | Table | Field | Final Labels | Confidence |\n")
            f.write("|---|-------|-------|-------------|------------|\n")

        # Table rows
        for idx, (inp, result) in enumerate(zip(inputs, results), 1):
            final_labels = '<br>'.join(result['final_labels'])
            conf = f"{result['final_confidence']:.4f}"
            table_name = inp['table_name']
            # Truncate long table names
            if len(table_name) > 30:
                table_name = table_name[:27] + '...'
            field_name = inp['field_name']

            if evaluation and idx-1 < len(evaluation['per_sample_results']):
                per_sample = evaluation['per_sample_results'][idx-1]
                gt = '<br>'.join(per_sample['ground_truth_data_items'])
                is_match = '✅' if per_sample['exact_match'] else '❌'
                f.write(f"| {idx} | {table_name} | {field_name} | {final_labels} | {gt} | {is_match} | {conf} |\n")
            else:
                f.write(f"| {idx} | {table_name} | {field_name} | {final_labels} | {conf} |\n")

        f.write("\n")

        # Full details with intermediate steps
        f.write("## Full Details with Intermediate Steps\n\n")
        for idx, (inp, result) in enumerate(zip(inputs, results), 1):
            f.write(f"### {idx}. {inp['table_name']}.{inp['field_name']}\n\n")

            f.write("- **Input:**\n")
            f.write(f"  - Table: `{inp['table_name']}`\n")
            f.write(f"  - Field: `{inp['field_name']}`\n")
            if inp.get('field_description'):
                f.write(f"  - Description: {inp['field_description']}\n")
            f.write("\n")

            fa = result['feature_analysis']
            f.write("- **Feature Analysis:**\n")
            f.write(f"  - Semantic summary: {fa['semantic_summary']}\n")
            f.write(f"  - Dominant source: {fa['dominant_source']}\n")
            f.write(f"  - Table keywords: {', '.join(fa['table_name_keywords'])}\n")
            f.write(f"  - Field keywords: {', '.join(fa['field_name_keywords'])}\n")
            f.write("\n")

            pc = result['preliminary_result']
            f.write("- **Preliminary Classification:**\n")
            f.write(f"  - Number of predictions: {len(pc['predictions'])}\n")
            f.write(f"  - Total average confidence: {pc['total_confidence']:.4f}\n")
            pred_list = [f"{p['data_item']} ({p['confidence']:.2f})" for p in pc['predictions']]
            f.write(f"  - Predictions: {', '.join(pred_list)}\n")
            f.write("\n")

            vr = result['verification_result']
            f.write("- **Self Verification:**\n")
            f.write(f"  - Kept predictions: {len(vr.get('verified_predictions', []))}\n")
            f.write(f"  - Average confidence after verification: {vr.get('average_confidence', 0.0):.4f}\n")
            if vr.get('removed_false_positives', []):
                f.write(f"  - Removed false positives: {', '.join(vr.get('removed_false_positives', []))}\n")
            if vr.get('added_missing', []):
                added = [p['data_item'] for p in vr.get('added_missing', [])]
                f.write(f"  - Added missing: {', '.join(added)}\n")
            if vr.get('suggests_reclassification', False):
                f.write(f"  - Suggests reclassification: Yes\n")
            f.write("\n")

            f.write("- **Final Result:**\n")
            f.write(f"  - Final labels: {', '.join(result['final_labels'])}\n")
            f.write(f"  - Final average confidence: {result['final_confidence']:.4f}\n")
            f.write("\n")

            if evaluation and idx-1 < len(evaluation['per_sample_results']):
                per_sample = evaluation['per_sample_results'][idx-1]
                f.write("- **Evaluation:**\n")
                f.write(f"  - Ground truth: {', '.join(per_sample['ground_truth_data_items'])}\n")
                f.write(f"  - Exact match: {'**YES**' if per_sample['exact_match'] else '**NO**'}\n")
                if per_sample['correct_predictions']:
                    f.write(f"  - Correct: {', '.join(per_sample['correct_predictions'])}\n")
                if per_sample['wrong_predictions']:
                    f.write(f"  - Wrong: {', '.join(per_sample['wrong_predictions'])}\n")
                if per_sample['missing_predictions']:
                    f.write(f"  - Missing: {', '.join(per_sample['missing_predictions'])}\n")
                f.write("\n")

            f.write("---\n\n")


def export_batch_results(
    outputs_dir: str | Path,
    inputs: List[TableFieldInput],
    results: List[ClassificationResult],
    evaluation: Optional[EvaluationResult] = None,
    base_name: str = "evaluation_results",
) -> tuple[Path, Path]:
    """
    Export results to both CSV and Markdown formats.

    Args:
        outputs_dir: Directory to save output files
        inputs: List of input table fields
        results: List of classification results
        evaluation: Optional evaluation result
        base_name: Base name for output files

    Returns:
        (csv_path, markdown_path) tuple
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = outputs_dir / f"{base_name}.csv"
    md_path = outputs_dir / f"{base_name}.md"

    export_results_to_csv(csv_path, inputs, results, evaluation)
    export_results_to_markdown(md_path, inputs, results, evaluation)

    return csv_path, md_path
