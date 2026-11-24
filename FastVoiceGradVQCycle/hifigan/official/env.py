# 保留官方配置包装与复制逻辑；复制的配置供后续权重加载和来源核对。
"""HiFi-GAN 官方辅助模块：让 JSON 配置支持“点号取值”，并复制实验配置。

这是上游官方代码的本地副本，只增加中文教学说明，不改变执行逻辑。

* ``AttrDict`` 继承普通字典；``self.__dict__ = self`` 后，``h["n_fft"]`` 与
  ``h.n_fft`` 会指向同一个值。
* ``build_env`` 把本次训练使用的 JSON 复制到检查点目录，日后才能知道权重对应
  哪套网络参数。
* ``*args`` 收集任意位置参数，``**kwargs`` 收集任意带名字参数，再原样交给 dict。
"""

# os 负责跨平台拼路径和建目录；shutil.copyfile 只复制文件内容。
import os
# shutil负责原样复制配置文件，为检查点保存对应的参数快照。
import shutil


# 给普通配置字典增加点号访问方式，兼容上游模型读取h字段的写法。
class AttrDict(dict):
    # 同一配置既可用键名取值，也可按官方网络要求通过属性访问。
    """既能用方括号、又能用点号访问的字典。"""

    # 把收到的键值传给普通字典构造器，再建立属性访问别名。
    def __init__(self, *args, **kwargs):
        # super 调用父类 dict 的初始化，确保所有普通字典行为仍然存在。
        super(AttrDict, self).__init__(*args, **kwargs)
        # 属性命名空间与字典自身绑定，因此 h.key 会查询 h["key"]。
        self.__dict__ = self


# 把训练配置复制到输出目录，记录该权重使用的设置。
def build_env(config, config_name, path):
    # 源配置与输出目录不同时复制配置快照，保留官方实验目录布局。
    """若源配置不在目标位置，就建立目录并复制一份可追溯快照。"""

    # 拼出检查点目录内应保存的配置快照路径。
    t_path = os.path.join(path, config_name)
    # 源文件与目标路径不同时才复制，避免对同一文件自我覆盖。
    if config != t_path:
        # 建立存放配置和模型的目录，已存在时可继续使用。
        os.makedirs(path, exist_ok=True)
        # 按指定名称复制配置内容，后续加载模型可找到配套设置。
        shutil.copyfile(config, os.path.join(path, config_name))
