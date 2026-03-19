"""从 label.csv 动态加载真实数据分类体系"""
from pathlib import Path

from classification_agent.utils.label_loader import load_label_csv

# label.csv 位于项目根目录（src/../../../label.csv）
_LABEL_CSV = Path(__file__).parent.parent.parent.parent / "label.csv"

DEFAULT_DATA_CATEGORIES = load_label_csv(_LABEL_CSV)
