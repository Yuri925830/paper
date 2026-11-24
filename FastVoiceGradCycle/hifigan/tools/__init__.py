# 此子包提供固定声码器加载和 mel 到 WAV 的处理工具，导入时不启动生成。
"""HiFi-GAN 加载、mel 转 WAV 和检查点审计工具。

正式实现位于 ``vocode.py``。保持 ``__init__`` 轻量，可以让其他模块明确写
``from hifigan.tools.vocode import ...``，同时不会因导入包名就加载模型。
"""
