# 严格对齐论文设定的 FastVoiceGrad 学生模型训练包。
"""严格对齐论文设定的 FastVoiceGrad 学生模型训练包。

这个文件告诉 Python：``fastvoicegrad`` 文件夹是一个可以导入的代码包。
它只向外公开配置类和已经锁定的配置对象，不会在导入时偷偷启动训练。
"""

# 从 config.py 取出“配置的数据结构”和“本次正式实验使用的固定配置”。
from .config import PAPER_STUDENT_CONFIG, FastVoiceGradStudentConfig

# __all__ 明确规定别人执行 ``from fastvoicegrad import *`` 时能看到的两个名字。
__all__ = ["FastVoiceGradStudentConfig", "PAPER_STUDENT_CONFIG"]
