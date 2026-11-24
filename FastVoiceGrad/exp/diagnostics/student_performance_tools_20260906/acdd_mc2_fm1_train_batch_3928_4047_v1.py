"""独立FM=1开发候选；仅更改私有准入，冻结MC2训练函数原样复用。"""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
RANGE_FILE = TOOLS / 'acdd_mc2_train_batch_3928_4047_v1.py'
RANGE_SHA = '6f04abb3838f574419986829c68196eddc72b3fe545d98a3e6a9f4162beffec3'
START, STOP, FM_WEIGHT = 3927, 4047, 1.0


def _load_private_core():
    if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
        raise RuntimeError('Run only on the registered Matpool rental')
    if hashlib.sha256(RANGE_FILE.read_bytes()).hexdigest() != RANGE_SHA:
        raise RuntimeError('Frozen3928 range helper changed')
    spec = importlib.util.spec_from_file_location('_fm1_private_range_loader', RANGE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hashlib.sha256(RANGE_FILE.read_bytes()).hexdigest() != RANGE_SHA:
        raise RuntimeError('Frozen range helper changed during import')
    # 此module及其_core都是新私有对象，绝不改变公开FM2 helper的globals。
    core = module._core
    if (core.START, core.STOP, core.MC2_SEED) != (START, STOP, 74000000):
        raise RuntimeError('Wrong private range or MC2 seed')
    return core


def validate_request(global_step, config, device, receipt):
    # 真实训练config必须是FM1；其余冻结准入与原函数相同，不伪造FM2配置过门。
    if type(global_step) is not int or not START < global_step <= STOP:
        raise RuntimeError('Require the next declared absolute step3928..4047')
    expected = {'batch_size': 32, 'diffusion_steps': 1000, 'student_start_step': 950,
                'score_target': 'clean_x0', 'score_weight': 'sqrt_alpha_bar',
                'lambda_fm': FM_WEIGHT, 'lambda_dist': 45.0}
    if any(getattr(config, name) != value for name, value in expected.items()):
        raise RuntimeError('Only the explicitly registeredFM1 MC2 configuration is allowed')
    if device.type != 'cuda' and not (device.type == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''):
        raise RuntimeError('CPU is permitted only for explicitly GPU-hidden rental tests')
    if not isinstance(receipt, dict) or receipt:
        raise RuntimeError('Supply one fresh empty MC receipt for every complete batch')


_core = _load_private_core()
_core.validate_request = validate_request
# 以下函数体/梯度/Adam顺序没有变化，系数来自实际传入的FM1 config。
train_batch_mc2 = _core.train_batch_mc2
original_input_identity = _core.original_input_identity
second_score_draws = _core.second_score_draws
mc2_score_distillation_loss = _core.mc2_score_distillation_loss
peek_production_draws = _core.peek_production_draws
tensor_identity = _core.tensor_identity
production = _core.production
MC2_SEED = _core.MC2_SEED
PRODUCTION_SHA, OBJECTIVES_SHA = _core.PRODUCTION_SHA, _core.OBJECTIVES_SHA
