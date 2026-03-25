#!/bin/bash
cd /home/xzx/projects/pro-data-classification-agent

echo "=== 测试虚拟环境 ==="
echo "1. 直接使用虚拟环境Python:"
classificationEnv/bin/python -c "print('✅ 直接调用成功')"

echo ""
echo "2. 尝试激活:"
source classificationEnv/bin/activate
echo "激活后PATH: $PATH" | head -1
which python 2>/dev/null || echo "❌ which python 未找到"

echo ""
echo "3. 检查PYTHONPATH:"
echo "PYTHONPATH: $PYTHONPATH"

echo ""
echo "4. 最简单的方法：使用虚拟环境中的命令"
ENV_PYTHON="classificationEnv/bin/python"
$ENV_PYTHON -c "
import sys
print(f'虚拟环境Python: {sys.executable}')
print(f'系统路径: {sys.path[:2]}')
"