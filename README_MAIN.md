# 更新后的 main.py 使用说明

## 新功能

### 1. BGE RAG 支持
- 使用 `BAAI/bge-large-zh-v1.5` 模型进行中文语义相似度检索
- 本地运行，无需额外API费用
- 专门为中文优化

### 2. 从 data.csv 加载 RAG 训练数据
- 自动从指定的 CSV 文件加载标注示例
- 支持与输入文件相同的格式
- 自动构建向量检索库

### 3. 可配置的 RAG 参数
- `--rag-top-k 10`: 检索最相似的10个示例（默认）
- `--rag-similarity-threshold 0.5`: 相似度阈值（默认0.5）
- `--rag-data data.csv`: RAG训练数据文件路径
- `--rag-embedding-provider bge`: 嵌入模型提供者（bge或openai）

## 使用示例

### 基本使用（无RAG）
```bash
python main.py --input data.csv --with-ground-truth
```

### 启用BGE RAG（推荐）
```bash
python main.py --input data.csv --enable-rag --rag-top-k 10 --with-ground-truth
```

### 使用OpenAI嵌入（需要API key）
```bash
python main.py --input data.csv --enable-rag --rag-embedding-provider openai --with-ground-truth
```

### 完整参数示例
```bash
python main.py \
  --input data.csv \
  --input-encoding GB18030 \
  --output-dir outputs \
  --output-base-name my_results \
  --with-ground-truth \
  --enable-rag \
  --rag-top-k 10 \
  --rag-similarity-threshold 0.6 \
  --rag-data data.csv \
  --rag-embedding-provider bge \
  --confidence-threshold 0.8 \
  --allow-multiple
```

## 环境配置

创建 `.env` 文件：
```bash
# OpenAI API 配置（用于分类的LLM）
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=optional-base-url

# BGE 模型配置（用于RAG嵌入，本地运行）
BGE_MODEL_NAME=BAAI/bge-large-zh-v1.5
BGE_DEVICE=cpu  # 或 cuda
BGE_USE_FP16=true
```

## RAG 工作原理

1. **训练阶段**：
   - 从 `data.csv` 加载标注示例
   - 使用BGE模型为每个示例生成中文向量嵌入
   - 存储在内存向量库中

2. **推理阶段**：
   - 对新字段生成向量嵌入
   - 检索最相似的10个标注示例（top_k=10）
   - 将相似示例作为上下文提供给LLM

3. **优势**：
   - 提高分类准确性（参考相似标注示例）
   - 减少幻觉（LLM看到真实存在的类别）
   - 保持一致性（相似字段得到相似分类）

## 输出结果

### 控制台输出
- 分类进度和LLM调用次数
- 评估指标（如果提供真实标签）
- 表级上下文分析摘要

### 文件输出
- `outputs/classification_results.csv`: 详细CSV结果
- `outputs/classification_results.md`: 可读的Markdown报告

## 性能说明

### BULK模式（默认）
- 无论字段数量多少，都只需要3-4次LLM调用
- 100个字段的表也只需要3-4次调用，不是100次

### RAG模式
- BGE嵌入：本地运行，无API延迟
- 向量检索：内存中快速相似度计算
- top_k=10: 平衡准确性和计算成本

## 故障排除

### BGE模型下载失败
```bash
# 手动下载BGE模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"
```

### 内存不足
- 减少 `--rag-top-k` 值（如改为5）
- 使用 `BGE_DEVICE=cpu` 避免GPU内存问题
- 分批处理大型数据集

### CSV编码问题
- 尝试 `--input-encoding GB18030` 或 `utf-8-sig`
- 检查CSV文件是否包含BOM头
