"""
从 data.csv 加载批量输入数据，支持读取真实标签用于评估。

CSV 列结构：
    数据域,表名,表中文名,表字段,字段描述,字段隐私四级分类

标签格式示例：
    用户应用基本信息(应用包名\客户端版本号)
    交易信息(交易记录)
    开发者账号ID(开发者账号ID), 应用基本信息(应用元数据)
"""

import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from classification_agent.types.schemas import TableFieldInput


# 正则匹配 "数据项(子数据项)" 格式
# 支持多个，用逗号分隔
_PATTERN = re.compile(r'([^(]+)\(([^)]+)\)')


@dataclass
class TableFieldInputWithGroundTruth:
    """输入加上真实标签（用于评估）"""
    input: TableFieldInput
    ground_truth_data_items: List[str]  # 真实数据项标签列表


def parse_ground_truth(label_text: str) -> List[str]:
    """
    解析 "字段隐私四级分类" 文本，提取数据项列表。

    格式示例：
        "用户应用基本信息(应用包名\客户端版本号)" → ["用户应用基本信息"]
        "交易信息(交易记录)" → ["交易信息"]
        "A(a), B(b)" → ["A", "B"]

    Args:
        label_text: 原始标签文本

    Returns:
        提取出的数据项名称列表
    """
    if not label_text or not label_text.strip():
        return []

    label_text = label_text.strip()
    matches = _PATTERN.findall(label_text)

    # 每个匹配是 (数据项, 子数据项)，只提取数据项
    data_items = [data_item.strip() for data_item, _ in matches]

    # 去重
    return list(dict.fromkeys(data_items))


def load_data_csv(
    csv_path: str | Path,
    include_ground_truth: bool = True,
    skip_empty_gt: bool = True,
) -> Tuple[List[TableFieldInput], Optional[List[List[str]]]]:
    """
    读取 data.csv，返回批量输入。

    Args:
        csv_path: CSV 文件路径
        include_ground_truth: 是否返回真实标签，False 只返回输入
        skip_empty_gt: 是否跳过真实标签为空的行，只在 include_ground_truth=True 生效

    Returns:
        (inputs, ground_truths) 元组：
            - inputs: List[TableFieldInput]，所有输入字段
            - ground_truths: 如果 include_ground_truth=True，返回对应每个输入的真实标签列表；否则 None
    """
    from pathlib import Path
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"data.csv 未找到: {csv_path.resolve()}")

    inputs: List[TableFieldInput] = []
    ground_truths: List[List[str]] = []

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头

        for row_idx, row in enumerate(reader, 2):  # row_idx 从2开始（表头是1）
            if len(row) < 5:
                continue  # 跳过不完整行

            # 提取列
            # 索引（0-based）：
            # 0: 数据域, 1: 表名, 2: 表中文名, 3: 表字段, 4: 字段描述, 5: 字段隐私四级分类
            table_name_raw = row[1].strip()
            table_cn_name = row[2].strip()
            field_name = row[3].strip()
            field_desc = row[4].strip() if len(row) > 4 else ""
            gt_text = row[5].strip() if len(row) > 5 else ""

            # 拼接 table_name: 表名 + 表中文名
            if table_cn_name:
                table_name = f"{table_name_raw} - {table_cn_name}"
            else:
                table_name = table_name_raw

            # 跳过关键字段为空
            if not table_name or not field_name:
                continue

            # 创建输入
            input_obj: TableFieldInput = {
                "table_name": table_name,
                "field_name": field_name,
                "field_description": field_desc or None,
            }

            # 不需要标签，直接加进去
            if not include_ground_truth:
                inputs.append(input_obj)
                continue

            # 需要标签，解析之
            gt_data_items = parse_ground_truth(gt_text)

            if skip_empty_gt and not gt_data_items:
                continue  # 跳过标签为空

            inputs.append(input_obj)
            ground_truths.append(gt_data_items)

    if include_ground_truth:
        return inputs, ground_truths
    else:
        return inputs, None


def load_data_csv_with_gt(
    csv_path: str | Path,
    skip_empty_gt: bool = True,
) -> List[TableFieldInputWithGroundTruth]:
    """
    便捷方法：读取 data.csv 直接返回带真实标签的列表。

    Args:
        csv_path: CSV 文件路径
        skip_empty_gt: 是否跳过真实标签为空的行

    Returns:
        List[TableFieldInputWithGroundTruth]
    """
    inputs, gts = load_data_csv(csv_path, include_ground_truth=True, skip_empty_gt=skip_empty_gt)
    if gts is None:
        return []
    return [
        TableFieldInputWithGroundTruth(input=inp, ground_truth_data_items=gt)
        for inp, gt in zip(inputs, gts)
    ]
