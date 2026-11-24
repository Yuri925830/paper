"""仅扩展独立MC2继续观察的准入范围；旧文件和生产训练数学均保持冻结。"""

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
FROZEN_HELPER = TOOLS / 'acdd_mc2_train_batch_v1.py'
FROZEN_HELPER_SHA = 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16'
START, STOP = 3447, 3927


def _load_frozen_math():
    if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
        raise RuntimeError('Run only on the registered Matpool rental')
    source = FROZEN_HELPER.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_HELPER_SHA:
        raise RuntimeError('Frozen MC2 helper identity mismatch')
    # SHA先锁定全部实现；再明确证明更改两个范围常量只影响这两个准入函数。
    tree = ast.parse(source)
    users = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                     and item.id in ('START', 'STOP') for item in ast.walk(node))}
    if users != {'validate_request', 'second_score_draws'}:
        raise RuntimeError('Frozen range constants have an unexpected consumer')
    spec = importlib.util.spec_from_file_location('_mc2_frozen_math_for_3448_3927_v1', FROZEN_HELPER)
    core = importlib.util.module_from_spec(spec)
    # 不注册/替换公开旧模块；每个函数的globals属于这个新建私有对象。
    spec.loader.exec_module(core)
    if hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() != FROZEN_HELPER_SHA:
        raise RuntimeError('Frozen MC2 helper changed while importing')
    if (core.START, core.STOP, core.MC2_SEED) != (3327, 3447, 74000000):
        raise RuntimeError('Frozen MC2 range or private seed contract changed')
    core.START, core.STOP = START, STOP
    return core


_core = _load_frozen_math()
_original_validate_request = _core.validate_request


def validate_request(global_step, config, device, receipt):
    # 给新范围准确的诊断；有效请求仍把真实绝对步交回旧函数的所有配置/设备/receipt门。
    if type(global_step) is not int or not START < global_step <= STOP:
        raise RuntimeError('Require the next declared absolute step3448..3927')
    return _original_validate_request(global_step, config, device, receipt)


_core.validate_request = validate_request
# 以下均是冻结文件实际生成的函数对象；没有复制、改写或重新组合训练函数体。
train_batch_mc1 = _core.train_batch_mc1
train_batch_mc2 = _core.train_batch_mc2
second_score_draws = _core.second_score_draws
mc2_score_distillation_loss = _core.mc2_score_distillation_loss
original_input_identity = _core.original_input_identity
peek_production_draws = _core.peek_production_draws
record_first_draws = _core.record_first_draws
score_record = _core.score_record
tensor_identity = _core.tensor_identity
production = _core.production
MC2_SEED = _core.MC2_SEED
PRODUCTION_SHA = _core.PRODUCTION_SHA
OBJECTIVES_SHA = _core.OBJECTIVES_SHA

__all__ = ['train_batch_mc1', 'train_batch_mc2', 'validate_request', 'second_score_draws',
           'mc2_score_distillation_loss', 'original_input_identity', 'peek_production_draws',
           'record_first_draws', 'score_record', 'tensor_identity', 'production',
           'START', 'STOP', 'MC2_SEED', 'PRODUCTION_SHA', 'OBJECTIVES_SHA',
           'FROZEN_HELPER', 'FROZEN_HELPER_SHA']
