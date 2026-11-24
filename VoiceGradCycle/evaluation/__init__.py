# 本包只提供客观评价与结果核对入口，不组织主观听测。
"""客观评价工具包：UTMOS、DNSMOS、CER、SVA 与分片合并。

这个文件把 ``evaluation`` 标记成可用 ``python -m`` 运行的 Python 包。它自己不加载
评价模型，也不计算分数；真正入口是 ``evaluate_metrics.py``。
"""
