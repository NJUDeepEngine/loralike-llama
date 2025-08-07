import register
from lm_eval.__main__ import cli_evaluate

# LAMBADA (LD)：评估语言建模预测能力（通常是句子最后一个词）
# HellaSwag (HS)：常识推理填空题
# PIQA (PQ)：物理常识推理
# WinoGrande (WG)：代词消解任务
# ARC (Easy 和 Challenge)：小学科学题（常用于语言模型推理能力评估）
# MMLU：多任务语言理解（大量学科领域的知识问答）

if __name__ == "__main__":
    cli_evaluate()