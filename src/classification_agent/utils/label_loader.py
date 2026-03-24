"""
从 label.csv 加载数据分类体系，构建 HierarchicalCategory 列表。

CSV 列映射关系：
  二级分类 (col 1) → level1
  三级分类 (col 2) → level2
  数据项   (col 3) → data_item
  数据子项 (col 4) → DataSubitem.name
  数据子项隐私声明描述 (col 7) → DataSubitem.description

同一 (level1, level2, data_item) 三元组的多行会合并到同一个
HierarchicalCategory，data_subitems 按行追加，去重。
"""

import csv
from pathlib import Path
from typing import List

from classification_agent.types.schemas import DataSubitem, HierarchicalCategory


# CSV 列索引（0-based）
_COL_LEVEL1       = 1   # 二级分类
_COL_LEVEL2       = 2   # 三级分类
_COL_DATA_ITEM    = 3   # 数据项
_COL_SUBITEM_NAME = 4   # 数据子项
_COL_DESCRIPTION  = 7   # 数据子项隐私声明描述


def load_label_csv(csv_path: str | Path, encoding: str = "utf-8-sig") -> List[HierarchicalCategory]:
    """
    读取 label.csv，返回去重合并后的 HierarchicalCategory 列表。

    Args:
        csv_path: label.csv 的路径（绝对或相对路径均可）。
        encoding: 文件编码，默认 "utf-8-sig"，可指定为 "GB18030"

    Returns:
        HierarchicalCategory 列表，每个 data_item 只出现一次，
        其下所有 data_subitems 均已合并。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"label.csv 未找到: {csv_path.resolve()}")

    # 用有序 dict 保证输出顺序与 CSV 行顺序一致
    # key: (level1, level2, data_item)  value: HierarchicalCategory
    category_map: dict[tuple[str, str, str], HierarchicalCategory] = {}

    # 用第二层 set 追踪已存在的子项名称，避免重复追加
    # key: (level1, level2, data_item)  value: set of subitem names
    subitem_seen: dict[tuple[str, str, str], set[str]] = {}

    with open(csv_path, encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头

        for row in reader:
            # 跳过列数不足或全空的行
            if len(row) <= _COL_DESCRIPTION:
                continue

            level1       = row[_COL_LEVEL1].strip()
            level2       = row[_COL_LEVEL2].strip()
            data_item    = row[_COL_DATA_ITEM].strip()
            subitem_name = row[_COL_SUBITEM_NAME].strip()
            description  = row[_COL_DESCRIPTION].strip() or None

            # 任何关键字段为空则跳过
            if not (level1 and level2 and data_item and subitem_name):
                continue

            key = (level1, level2, data_item)

            if key not in category_map:
                category_map[key] = HierarchicalCategory(
                    level1=level1,
                    level2=level2,
                    data_item=data_item,
                    data_subitems=[],
                )
                subitem_seen[key] = set()

            # 子项去重：同名子项只追加一次
            if subitem_name not in subitem_seen[key]:
                category_map[key]["data_subitems"].append(
                    DataSubitem(name=subitem_name, description=description)
                )
                subitem_seen[key].add(subitem_name)

    return list(category_map.values())
