# 把两个学生结果与两个当时的教师目标交叉配对，检查目标变化对loss读数的影响。
"""Read-only crossed clean-teacher targets for the fixed TRAIN3108 batch."""

import argparse  # 导入脚本不启动GPU；执行时必须显式批准。
import hashlib  # 冻结辅助文件通过字节哈希检查后才允许导入。
import json  # 只保存小型结果和运行身份，不保存模型或新噪声文件。
import math  # 重放与拆分计算遇到NaN/Inf立即拒绝。
import os  # 检查真实可用磁盘空间和本进程编号。
import platform  # 只使用原训练环境，避免换库带来数值差异。
import subprocess  # nvidia-smi只查询其他GPU进程和剩余显存。
import sys
import time  # 超时后不执行下一次网络前向。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
PAIR_FILE = TOOLS / 'run_acdd_paired_lr120_v1.py'
PAIR_SHA = '6bdf773152950226e8db6716ab64a977f30b3b72588322dd5aa54aee2b318bc0'
if hashlib.sha256(PAIR_FILE.read_bytes()).hexdigest() != PAIR_SHA:
    raise RuntimeError('Frozen monitoring helper changed')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch  # main之外只允许租用服务器上的小型CPU数学测试。
import run_acdd_paired_lr120_v1 as pair

base, training, resume, probe = pair.base, pair.training, pair.resume, pair.probe
DEV = pair.DEV
PAIRED = DEV / 'acdd_lr_paired120_from3107_v1'
CONTINUED = DEV / 'acdd_half_lr_continuation480_v1'
PROTOCOL = DEV / 'ACDD3247_TARGET_CROSS_PROTOCOL_CN.md'
PROTOCOL_SHA = '224b6f5ec5c8cbc5184e5d46dfeba9a6186aa564b02317d9cc68bdb41f77cc86'
STATES = {
    3227: {'directory': PAIRED / 'primary_s1e4', 'file': 'student_step_003227.pt',
           'sha256': '88da1f14234c9dcb164cdd249d47862b9ba105f1b7a1a858b4a18f9fa921b1dc',
           'contract_sha256': '25c6e14693a940d94e18a85f50e8e3c08bbfbf1e15f53580117662a0f86ad909'},
    3247: {'directory': CONTINUED, 'file': 'student_step_003247.pt',
           'sha256': '97dc8c97e692736a79f1c278c287d82db668567e2c5f17a0e3fd7bcefdbf64eb',
           'contract_sha256': '90e2c835fb4092ca72e4ad702c6967dc790a933ba2210958a811d239da626489'},
}  # 只接受已经完成、保存并验证的这两个学生端点。
FILES = {
    str(PAIR_FILE): PAIR_SHA,
    str(PROTOCOL): PROTOCOL_SHA,
    str(PAIRED / 'run_contract.json'): 'f60a052e712080d48207be35a9595804a579ac152acdbb8c9b450108a8fde303',
    str(PAIRED / 'primary_s1e4/arm_completion.json'): '50eee0fadb5b25de06e06beec23fa2e4b4f6e9aa1ab38f6aee1128d17ce17dcb',
    str(PAIRED / 'fixed_teacher_target.pt'): '2aedd82fb269ed07c98b206c41e071c73f64555c57ed4ce3faf1eb8db12144bb',
    str(CONTINUED / 'run_contract.json'): '9396b0609d5a77a6203d928339d6acd07ac02e51899e500c126da736e51517e8',
    str(CONTINUED / 'completion.json'): '0781f8b8928b850215c0b30892e8cedc106db9cc0c331637d487a586460e857b',
    str(pair.INPUT_DIR / 'fixed_inputs.pt'): '296a7b4a7c289f0d19409947cd2c4dcb54754ae5636a58af43e658bc12b17da8',
}
MAX_JSON_BYTES = 1024**2  # 完整报告上限1MiB；不会另存几十万项噪声或任何权重。
REPLAY_ATOL = 2e-5  # 运行前冻结的协议容差；不根据结果临时放宽。


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(tensor):
    return {'sha256': probe.tensor_sha(tensor), 'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}


def l1_readout(prediction, target, weights):
    # 每条语音先平均80个频带与128帧，再乘自己的sqrt(alpha_bar_t)，最后平均32句。
    if prediction.shape != target.shape or prediction.ndim != 3 or weights.shape != (prediction.shape[0],):
        raise RuntimeError('L1 tensors or per-utterance weights have incompatible shapes')
    if not all(bool(torch.isfinite(v).all()) for v in (prediction, target, weights)) or bool((weights < 0).any()):
        raise RuntimeError('Nonfinite values or negative weights cannot enter the comparison')
    per = (prediction - target).abs().flatten(1).mean(1)
    weighted = weights.to(per.dtype) * per  # 与生产函数相同精度，不把某个高t样本重复加权。
    return {'weighted': float(weighted.mean()), 'unweighted': float(per.mean()),
            'per_utterance_weighted': weighted.cpu().tolist(), 'per_utterance_unweighted': per.cpu().tolist()}


def split_scalar(c00, c10, c01, c11):
    # Cij的行是学生端点，列是教师目标端点；两条分解顺序都保留，避免挑有利顺序。
    values = (c00, c10, c01, c11)
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError('Nonfinite crossed score')
    student_first = c10 - c00  # 先只换学生，仍用R3227。
    target_after = c11 - c10  # 再只把目标换成R3247。
    target_first = c01 - c00  # 反过来，先只换目标。
    student_after = c11 - c01  # 再在R3247下换学生。
    total = c11 - c00
    student_mean = .5 * (student_first + student_after)
    target_mean = .5 * (target_first + target_after)
    if abs(student_mean + target_mean - total) > 1e-12 * max(1., *(abs(x) for x in values)):
        raise RuntimeError('Crossed decomposition does not sum to the actual diagonal difference')
    return {'diagonal_change': total, 'student_change_at_R3227': student_first,
            'target_change_at_y3247': target_after, 'target_change_at_y3227': target_first,
            'student_change_at_R3247': student_after, 'symmetric_student_component': student_mean,
            'symmetric_target_component': target_mean,
            'interaction': c11 - c10 - c01 + c00}


def crossed_readouts(predictions, targets, weights):
    cells = {f'y{i}_R{j}': l1_readout(predictions[i], targets[j], weights)
             for i in (3227, 3247) for j in (3227, 3247)}
    order = ('y3227_R3227', 'y3247_R3227', 'y3227_R3247', 'y3247_R3247')
    decomposition = {}
    for metric in ('weighted', 'unweighted'):
        values = [cells[key][metric] for key in order]
        decomposition[metric] = split_scalar(*values)
        # 逐句分解与总均值使用同一四格，32句全部保留，不删除难例。
        key = 'per_utterance_' + metric
        decomposition[metric]['per_utterance'] = [split_scalar(*(cells[name][key][i] for name in order))
                                                 for i in range(weights.numel())]
    return cells, decomposition


def verify_replay(actual, archived):
    # A仍是旧R3107；B是当时学生对应的新目标，不能交换名字或重定基线。
    differences = {key: abs(actual[key] - archived[key]) for key in actual}
    if any(not math.isfinite(actual[key]) or not math.isfinite(archived[key]) or delta > REPLAY_ATOL
           for key, delta in differences.items()):
        raise RuntimeError('Archived A/B or fixed TRAIN reference did not replay: ' + json.dumps(differences))
    return differences


def validate_payload(payload, step, config):
    item = STATES[step]
    if payload['global_step'] != step or payload['run_contract_sha256'] != item['contract_sha256']:
        raise RuntimeError('Student checkpoint step or contract mismatch')
    if payload['configuration'] != config.as_dict() or payload['effective_optimizer_lrs'] != {'student': .0001, 'discriminator': .0002}:
        raise RuntimeError('Student mathematical configuration or inherited LR identity changed')
    if payload['score_semantics'] != 'reconstruction_clean_x0_sqrt_alpha_bar':
        raise RuntimeError('Unexpected score objective in the checkpoint')


def capacity(directory):
    stat = os.statvfs(directory)
    available = stat.f_bavail * stat.f_frsize
    required = pair.SYSTEM_RESERVE + pair.REPORT_RESERVE
    if available < required:
        raise RuntimeError(f'Diagnostic disk reserve not met: available={available}, required={required}')
    return {'available_bytes': available, 'required_bytes': required}


def gpu_idle():
    # 在创建CUDA模型前检查整个设备；已有其他任务时拒绝并行，也不主动结束它们。
    command = ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits']
    rows = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    if any(line.split(',')[0].strip() != str(os.getpid()) for line in rows.splitlines() if line.strip()):
        raise RuntimeError('Another GPU process is active: ' + rows)
    memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.splitlines()
    if len(memory) != 1 or int(memory[0]) < 8192:
        raise RuntimeError('Require the original single GPU and at least8GiB free VRAM')
    return {'compute_processes': rows, 'free_mib': int(memory[0])}


@torch.no_grad()
def endpoint_forward(student, teacher, diffusion, inputs, config):
    # 一次学生前向和一次教师前向；不建立梯度图，不创建判别器、声码器或优化器。
    frozen = {'teacher': teacher, 'student': student}
    before = {name: pair.model_sha(model) for name, model in frozen.items()}
    pair.assert_frozen(frozen, before)
    prediction, q950 = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'],
                                                       inputs['bnf'], 950, inputs['source_noise'])
    if not torch.equal(q950, inputs['q950']):
        raise RuntimeError('Student did not receive the original exactq950 input')
    loss, target = training.score_distillation_loss(prediction, teacher, diffusion, inputs['score_timesteps'],
        inputs['score_noise'], inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
    weights = diffusion.sqrt_alphas_cumprod[inputs['score_timesteps']]
    checked = l1_readout(prediction, target, weights)
    if abs(checked['weighted'] - float(loss)) > 1e-12:
        raise RuntimeError('Crossed weighting differs from the production score function')
    if prediction.shape != inputs['x0'].shape or target.shape != inputs['x0'].shape or prediction.requires_grad or target.requires_grad:
        raise RuntimeError('Prediction/target shape or no-gradient boundary changed')
    pair.assert_frozen(frozen, before)  # 参数和全部named_buffers都必须逐字节不变。
    return prediction, target, before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--run-approved', action='store_true')
    args = parser.parse_args()
    if not args.run_approved or not 1 <= args.max_seconds <= 300:
        raise RuntimeError('Explicit approval and at most300 seconds required')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)  # 句柄保持到main结束，覆盖训练和评估的三把全局锁。
    if training.git_head(ROOT) != base.EXPECTED_HEAD:
        raise RuntimeError('Frozen clean production source changed')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Use the original myconda runtime')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Use the registered myconda CUDA interpreter')
    output = args.output_dir.resolve()
    if output.exists() or output != DEV / 'acdd_teacher_target_cross_3227_3247_v1':
        raise RuntimeError('Use the exact new output directory declared before the experiment')
    disk_before, gpu_before = capacity(DEV), gpu_idle()
    started = time.monotonic()
    def check_budget(stage):
        print('TARGET_CROSS ' + stage, flush=True)
        capacity(DEV)
        if time.monotonic() - started > args.max_seconds:
            raise RuntimeError('Read-only target comparison exceeded its time budget')
    hashes = dict(FILES)
    hashes.update({str(item['directory'] / item['file']): item['sha256'] for item in STATES.values()})
    resume.verify_files(hashes)
    parent, child = read_json(PAIRED / 'run_contract.json'), read_json(CONTINUED / 'run_contract.json')
    for step, contract in ((3227, parent), (3247, child)):
        if training.canonical_json_sha256(contract) != STATES[step]['contract_sha256']:
            raise RuntimeError('Archived run contract identity changed')
        hashes.update(contract['file_sha256'])  # 只读核查原teacher/G/BNF/stats依赖，绝不重写这些文件。
    if child['lineage']['parent_checkpoint_sha256'] != 'f98b0763ec5a2edbd23d5d493e9dac549f85cf40cfac71e79f0b462f6fc6beb0':
        raise RuntimeError('3247 is not the declared continuation of3227')
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if config.as_dict() != parent['configuration'] or config.as_dict() != child['configuration']:
        raise RuntimeError('Archived mathematical configuration changed')
    if (config.score_target, config.score_weight, config.lambda_fm, config.lambda_dist) != ('clean_x0', 'sqrt_alpha_bar', 2, 45):
        raise RuntimeError('Use the unchanged clean score and original coefficients')
    grand = read_json(resume.PARENT / 'run_contract.json')
    if training.collect_source_fingerprints(ROOT) != grand['source_sha256']:
        raise RuntimeError('Production source fingerprints changed')
    hashes[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
    resume.verify_files(hashes)
    archived = {3227: read_json(PAIRED / 'primary_s1e4/arm_completion.json'), 3247: read_json(CONTINUED / 'completion.json')}
    for step, receipt in archived.items():
        if receipt['completed_global_step'] != step or receipt['student_snapshot']['sha256'] != STATES[step]['sha256']:
            raise RuntimeError('Actual completed checkpoint receipt changed')
    raw_inputs = resume.load_torch_checkpoint(pair.INPUT_DIR / 'fixed_inputs.pt')
    input_identities = {name: identity(value) for name, value in raw_inputs.items()}
    if {k: v['sha256'] for k, v in input_identities.items()} != parent['input_sha256'] or parent['input_sha256'] != child['fixed_train_inputs_sha256']:
        raise RuntimeError('Original fixed inputs changed between runs')
    for name in ('x0', 'source_noise', 'score_noise', 'q950'):
        if raw_inputs[name].shape != (32, 80, 128) or raw_inputs[name].dtype != torch.float32:
            raise RuntimeError('Require the complete original32x80x128 TRAIN batch')
    if raw_inputs['bnf'].shape != (32, 144, 128) or raw_inputs['spk_emb'].shape != (32, 256):
        raise RuntimeError('Original BNF/speaker conditions changed')
    t = raw_inputs['score_timesteps']
    if t.shape != (32,) or t.dtype != torch.long or int(t.min()) < 1 or int(t.max()) > 1000:
        raise RuntimeError('Original32 discrete teacher timesteps required')
    if not all(bool(torch.isfinite(value).all()) for value in raw_inputs.values()):
        raise RuntimeError('Nonfinite archived input tensor')
    old_payload = resume.load_torch_checkpoint(PAIRED / 'fixed_teacher_target.pt')
    old_target = old_payload['teacher_target']
    if old_payload['global_source_step'] != 3107 or identity(old_target)['sha256'] != parent['fixed_teacher_target_sha256'] or old_payload['sha256'] != parent['fixed_teacher_target_sha256']:
        raise RuntimeError('OldA anchor must remain the exactR3107 target')
    torch.set_num_threads(2)  # 延续原开发驱动的设置，不另外改TF32、cudnn或归一化。
    device = torch.device('cuda')
    inputs = {name: value.to(device) for name, value in raw_inputs.items()}
    old_target = old_target.to(device)
    teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
    teacher.eval()
    student.train()  # 原固定TRAIN监测保留学生train模式；只关闭梯度，不额外改变模式来争取重放。
    training.set_requires_grad(teacher, False)
    training.set_requires_grad(student, False)
    if pair.model_sha(teacher) != parent['fixed_frozen_models_sha256']['teacher']:
        raise RuntimeError('Frozen teacher parameters or buffers changed')
    diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    weights = diffusion.sqrt_alphas_cumprod[inputs['score_timesteps']]
    predictions, targets, states, replay = {}, {}, {}, {}
    for step in (3227, 3247):
        check_budget('loading_student_' + str(step))
        payload = resume.load_torch_checkpoint(STATES[step]['directory'] / STATES[step]['file'])
        validate_payload(payload, step, config)
        student.load_state_dict(payload['student'], strict=True)  # 同一网络对象依次装入两份只读权重，前向前后分别核对。
        del payload
        with pair.preserve_rng(device):
            predictions[step], targets[step], frozen_hashes = endpoint_forward(student, teacher, diffusion, inputs, config)
        current = l1_readout(predictions[step], targets[step], weights)
        old = l1_readout(predictions[step], old_target, weights)
        drift = l1_readout(targets[step], old_target, weights)
        gt = l1_readout(predictions[step], inputs['x0'], weights)
        actual = {'primary_moving_score': current['weighted'], 'diagnostic_frozen_target_score': old['weighted'],
                  'teacher_target_weighted_l1_drift': drift['weighted'], 'train_gt_l1': gt['unweighted'],
                  'train_prediction_rms': float(predictions[step].square().mean().sqrt())}
        replay[step] = verify_replay(actual, archived[step]['last_monitor']['fixed_train'])
        states[step] = {'model_before_after_sha256': frozen_hashes, 'prediction': identity(predictions[step]),
                       'teacher_target': identity(targets[step]), 'old_R3107_score': old, 'moving_score': current,
                       'target_drift_from_R3107': drift, 'student_vs_GT': gt,
                       'teacher_target_vs_GT': l1_readout(targets[step], inputs['x0'], weights)}
    check_budget('crossing_targets_without_more_model_forwards')
    cells, decomposition = crossed_readouts(predictions, targets, weights)
    if {name: identity(value) for name, value in inputs.items()} != input_identities:
        raise RuntimeError('Read-only comparison changed an input tensor')
    if identity(old_target)['sha256'] != parent['fixed_teacher_target_sha256']:
        raise RuntimeError('Read-only comparison changed the archivedR3107 target')
    pair.assert_frozen({'teacher': teacher, 'student': student}, states[3247]['model_before_after_sha256'])
    resume.verify_files(hashes)  # 全部依赖文件在读取前后都相同；只装权重到隔离内存。
    check_budget('writing_small_report')
    report = {'status': 'completed', 'kind': 'readonly_crossed_teacher_targets', 'file_sha256': hashes, 'states': states, 'replay_absolute_differences': replay,
              'input_archive': str(pair.INPUT_DIR / 'fixed_inputs.pt'), 'input_identities': input_identities,
              'fixed_training_batch': {'global_step': 3108, 'crop_seed': 104342, 'noise_seed': 204342,
                  'indices': resume.GlobalBatchPlan(parent['training_indices']).for_step(3108),
                  'not_independent_development': True},
              'score_timesteps': inputs['score_timesteps'].cpu().tolist(), 'sqrt_alpha_bar_weights': weights.cpu().tolist(),
              'old_target_anchor_step': 3107, 'old_teacher_target': identity(old_target),
              'cells': cells, 'decomposition': decomposition, 'new_target_drift': l1_readout(targets[3247], targets[3227], weights),
              'student_output_change': l1_readout(predictions[3247], predictions[3227], weights),
              'actual_forward_counts': {'student': 2, 'teacher': 2, 'vocoder': 0, 'discriminator': 0},
              'optimizer_created': False, 'parameters_updated': False, 'new_checkpoint_written': False,
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'executable': str(Path(sys.executable).resolve()), 'gpu': torch.cuda.get_device_name(0),
                          'student_training_mode': student.training, 'teacher_training_mode': teacher.training},
              'initial_disk': disk_before, 'initial_gpu': gpu_before, 'elapsed_seconds': time.monotonic() - started,
              'limits': ['Only the original32 TRAIN examples and one fixednoise/t draw; no formal metrics or independent generalization claim.',
                         'Cij holds student output i and teacher target j; symmetric components are descriptive algebra, not causal training-gradient attribution.',
                         'A remainsR3107. The reported scores are unmultiplied by45; unweightedL1 removes onlysqrt_alpha_bar, not mel normalization.',
                         'The teacher uses the production stop-gradient target. No teacher Jacobian, optimizer update or coefficient change is tested.']}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_JSON_BYTES:
        raise RuntimeError('Refuse an unexpectedly large report')
    output.mkdir()  # 所有验真和前向完成后才创建新的报告目录，绝不覆盖旧实验。
    with (output / 'results.json.tmp').open('xb') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(output / 'results.json.tmp', output / 'results.json')  # 写完才出现正式结果，失败临时文件不能冒充完成。
    print('TARGET_CROSS_COMPLETED ' + str(output / 'results.json'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # 失败信息进入root保存的启动日志；不向已有输出目录补写或覆盖任何旧材料。
        print('TARGET_CROSS_FAILED ' + json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                                                 'error': str(error)}, ensure_ascii=False), flush=True)
        raise
