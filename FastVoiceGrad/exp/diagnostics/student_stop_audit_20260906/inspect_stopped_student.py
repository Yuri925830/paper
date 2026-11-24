# 停训后的只读诊断：不创建优化器，不反向传播，不更新任何模型。
"""Compare fixed reconstructions and crossed frozen critics after user stop."""
import gc  # 及时释放上一组不再使用的大张量。
import hashlib  # 给输入和权重留下可核对的身份记录。
import json  # 保存逐句结果和汇总，不修改原训练日志。
import random  # 重放短程诊断当时的固定裁剪。
import subprocess  # 只读记录远程源码版本。
import sys  # 将远程项目根目录加入导入路径。
import time  # 记录实际诊断时间。
from collections import defaultdict  # 汇总判别器逐层统计。
from dataclasses import replace  # 从原配置派生只读诊断设置。
from pathlib import Path  # 使用明确的绝对目录。
import numpy as np  # 读取固定样本索引及计算结果均值。
import torch  # 所有模型运算均在租用服务器执行。
from torch.utils.data import default_collate  # 保持原批量中样本的顺序。

ROOT = Path('/root/FsatVoiceGrad')  # 当前租用服务器的权威项目目录。
sys.path.insert(0, str(ROOT))  # 导入服务器现场代码，不导入本地旧副本。
from voicegrad.runtime_guard import assert_rented_server  # 阻止在其他主机运行。
from voicegrad.dataset import VoiceGradDataset  # 与训练共用预处理和帧对齐。
from voicegrad.diffusion import VoiceGradDiffusion  # 与训练共用扩散日程。
from fastvoicegrad.config import PAPER_STUDENT_CONFIG  # 保持已登记的参数。
from fastvoicegrad.components import load_teacher_and_student, load_frozen_vocoder, build_paper_discriminator, load_mel_statistics, load_torch_checkpoint, denormalize_mel, sha256_file  # 复用已校验的组件。
from fastvoicegrad.train import seed_everything, build_test_loss_loader, acquire_runtime_locks  # 独占诊断并复用固定测试裁剪。
from fastvoicegrad.objectives import make_student_prediction, score_distillation_loss  # 只前向测量原训练目标。


def tensor_sha(tensor):  # 固定输入也记录字节哈希，便于重放检查。
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, data):  # 仅生成本次诊断的新结果文件。
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def mel_measure(prediction, reference):  # 以不变的真实mel为参照，不经过判别器。
    return {  # 所有误差按句保留，避免只看一个平均数。
        'mel_l1': (prediction-reference).abs().flatten(1).mean(1).cpu().tolist(),  # 越低越接近真实mel。
        'mel_rmse': (prediction-reference).square().flatten(1).mean(1).sqrt().cpu().tolist(),  # 对较大误差更敏感。
        'prediction_rms': prediction.square().flatten(1).mean(1).sqrt().cpu().tolist(),  # 检查输出是否幅度塌缩。
        'reference_rms': reference.square().flatten(1).mean(1).sqrt().cpu().tolist(),  # 相同真实语音的幅度参照。
        'cosine_reference': torch.nn.functional.cosine_similarity(prediction.flatten(1), reference.flatten(1), dim=1).cpu().tolist(),  # 查看频谱形状是否接近。
    }


@torch.inference_mode()  # 整个诊断关闭梯度，绝不训练。
def main():
    assert_rented_server(ROOT)  # 只允许当前Matpool租用机。
    locks = acquire_runtime_locks(ROOT)  # 防止与其他训练任务同时占用。
    out = ROOT/'exp/diagnostics/student_stop_audit_20260906'  # 与训练产物分离。
    if (out/'results.json').exists():  # 不覆盖已经完成的诊断证据。
        raise RuntimeError('Refusing to overwrite completed diagnostic')
    out.mkdir(parents=True, exist_ok=True)  # 此处只建立诊断结果目录。
    torch.set_num_threads(4)  # 控制CPU线程占用。
    device = torch.device('cuda')  # 模型前向在出租GPU上执行。
    config = replace(PAPER_STUDENT_CONFIG, score_target='clean_x0', score_weight='sqrt_alpha_bar', num_workers=0)  # 不改变目标，只改读取线程。
    seed_everything(config.seed)  # 重建正式启动时的随机初始化顺序。
    teacher_path = ROOT/'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt'  # 同一教师起点。
    teacher, student, _ = load_teacher_and_student(teacher_path, config, device)  # 严格核对教师身份并初始化学生。
    initial = {k:v.detach().cpu().clone() for k,v in student.state_dict().items()}  # 在CPU保留共同起点。
    vd = ROOT/'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen'  # 同一自训声码器。
    vocoder, vconfig, _ = load_frozen_vocoder(ROOT, vd/'generator_step_2500000.pt', vd/'config.json', vd/'final_checkpoint_attestation_comments_20260906.json', config, device)  # 使用注释等价性验证后的独立证明。
    discriminator = build_paper_discriminator(ROOT, vconfig, config, device)  # 按启动顺序重建随机判别器。
    scratch = {k:v.detach().cpu().clone() for k,v in discriminator.state_dict().items()}  # 仅用于固定判别器交叉比较。
    diffusion = VoiceGradDiffusion(config.diffusion_steps, offset=config.cosine_offset).to(device)  # 固定扩散系数。
    mean, std = load_mel_statistics(ROOT/'data', device)  # 使用原训练统计。
    teacher.eval(); student.eval(); vocoder.eval(); discriminator.eval()  # 关闭所有训练模式。
    run = ROOT/'exp/checkpoints/fastvoicegrad_vctk_text_unseen_repair_x0_sqrt_20260906'  # 已停止的修复实验。
    checkpoint = load_torch_checkpoint(run/'latest.pt')  # 只读加载最后完整断点。
    pilot_dir = ROOT/'exp/diagnostics/acdd_repair_20260906'  # 先前120步对照的只读来源。
    pilot = load_torch_checkpoint(pilot_dir/'clean_x0_sqrt_alpha_bar.pt')  # 不再运行旧诊断脚本中的训练。
    manifest = json.loads((pilot_dir/'diagnostic_manifest.json').read_text())  # 使用原来的32句索引。
    dataset = VoiceGradDataset(ROOT/'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)  # 与原诊断相同数据视图。
    random.seed(1234)  # 原短程dev是在第一次随机裁剪时取得的。
    pilot_batch = default_collate([dataset[i] for i in manifest['dev_indices']])  # 精确重放这32条裁剪。
    batches = {'training_source_32': [pilot_batch], 'unseen_self_reconstruction_100': list(build_test_loss_loader(ROOT/'data', config))}  # 第二组保持正式固定测试裁剪。
    noise_banks = {}  # 各模型使用完全相同输入、噪声和随机时刻。
    identities = {}  # 保存每句说话人、句号及裁剪位置。
    for name, group in batches.items():  # 分别构造两套不变输入。
        seeds = (55001,66002,77003) if name=='training_source_32' else (config.test_loss_source_noise_seed,config.test_loss_score_noise_seed,config.test_loss_timestep_seed)  # 测试严格沿用原种子。
        generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]  # 噪声流互相独立。
        noise_banks[name] = []  # 每个批次保存一组噪声。
        identities[name] = []  # 保持清单顺序。
        for batch in group:  # 样本不打乱。
            shape = batch['mel'].shape  # 与真实mel完全同形。
            source = torch.randn(shape, device=device, generator=generators[0])  # 学生固定输入噪声。
            score = torch.randn(shape, device=device, generator=generators[1])  # 教师固定再加噪噪声。
            ticks = torch.randint(1,1001,(shape[0],),device=device,generator=generators[2])  # 与原测试相同均匀随机时刻。
            noise_banks[name].append((source,score,ticks))  # 多个模型直接复用张量。
            identities[name].append({'speakers':batch['spk_name'],'utterances':batch['utt_id'],'mel_sha256':tensor_sha(batch['mel']),'bnf_sha256':tensor_sha(batch['bnf']),'source_noise_sha256':tensor_sha(source),'crop_start':batch.get('fixed_crop_start',torch.tensor([])).tolist()})  # 记录实际输入身份。
    result = {'time':time.time(),'purpose':'Read-only diagnosis after user stop; no optimizer or backward; self-reconstruction is not formal cross-speaker VC evaluation.','checkpoint_epoch':checkpoint['epoch'],'checkpoint_step':checkpoint['global_step'],'checkpoint_sha256':sha256_file(run/'latest.pt'),'run_contract_sha256':checkpoint['run_contract_sha256'],'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'limitations':['The training-source 32 sentences were excluded only from the old 120-step pilot, not from formal full training.','Scratch critic is recreated from the documented startup order, not loaded from an archived step-0 critic.','Epoch 1 weights were overwritten by the original latest-checkpoint policy, so this audit cannot isolate epoch 1 to 2 output changes.'],'input_identities':identities,'models':{},'critic_cross':{}}  # 明确证据范围。
    waves = {}  # 缓存固定波形，交叉测量时不再重新合成。
    for label, state in [('teacher_initialized_one_step',initial),('pilot_120',pilot['student']),('repair_epoch2',checkpoint['student'])]:  # 比较相同结构的三个权重状态。
        student.load_state_dict(state, strict=True)  # 不继承其他组参数。
        result['models'][label] = {}  # 本模型的逐组测量。
        waves[label] = {}  # 本模型的生成波形。
        for name, group in batches.items():  # 使用两组完全固定的输入。
            totals = defaultdict(list)  # 按句积累固定误差。
            dist_sum = 0.0  # 按样本数加权蒸馏目标。
            pred_energy = ref_energy = 0.0  # 复现原pilot整体RMS定义。
            sample_count = element_count = 0  # 正确处理最后四句小批。
            waves[label][name] = []  # 固定模型、固定组别的波形。
            for batch, (source,score,ticks) in zip(group,noise_banks[name]):  # 噪声与语音一一对应。
                x,s,b = [batch[k].to(device) for k in ('mel','spk_emb','bnf')]  # 不改变任何归一化和对齐。
                prediction,_ = make_student_prediction(student,diffusion,x,s,b,950,source)  # 只进行一次前向预测。
                assert torch.isfinite(prediction).all()  # 确认没有NaN或Inf。
                for key, values in mel_measure(prediction,x).items(): totals[key].extend(values)  # 保存逐句固定参考误差。
                dist,_ = score_distillation_loss(prediction,teacher,diffusion,ticks,score,s,b,target_mode='clean_x0',weight_mode='sqrt_alpha_bar')  # 额外测量原移动目标，便于核对日志。
                dist_sum += float(dist)*x.shape[0]  # 使用样本数加权。
                sample_count += x.shape[0]; element_count += x.numel()  # 保留两个不同分母。
                pred_energy += float(prediction.square().sum()); ref_energy += float(x.square().sum())  # 不把逐句RMS平均冒充整体RMS。
                fake = vocoder(denormalize_mel(prediction,mean,std))  # 所有组使用同一个冻结声码器。
                real = vocoder(denormalize_mel(x,mean,std))  # 参考是GTmel声码器重建，不是原录音。
                waves[label][name].append((real.cpu(),fake.cpu()))  # 释放GPU上的长期缓存。
            summary = {k:float(np.mean(v)) for k,v in totals.items()}  # 逐句等权平均。
            summary.update({'samples':sample_count,'prediction_global_rms':(pred_energy/element_count)**0.5,'reference_global_rms':(ref_energy/element_count)**0.5,'moving_target_dist':dist_sum/sample_count})  # 原目标仅单独记录。
            result['models'][label][name] = {'summary':summary,'per_utterance':dict(totals)}  # 保留均值与明细。
            write_json(out/'partial_results.json',result)  # 生成独立中间结果，便于实时核查。
            print('FIXED_RESULT '+json.dumps({'model':label,'set':name,**summary}),flush=True)  # 简明打印进度。
    for dlabel,dstate in [('recreated_scratch',scratch),('epoch2',checkpoint['discriminator'])]:  # 固定D之后再比较学生，拆开两种变化。
        discriminator.load_state_dict(dstate,strict=True)  # 每个D状态使用其独立权重。
        result['critic_cross'][dlabel] = {}  # 保存交叉测量表。
        for label in waves:  # 同一D衡量三组已经缓存的相同输入波形。
            name = 'unseen_self_reconstruction_100'  # 交叉实验覆盖100个固定测试片段。
            accum = defaultdict(float); layers = defaultdict(lambda:defaultdict(float)); count=0  # 累计样本加权分项及层幅度。
            for real_cpu,fake_cpu in waves[label][name]:  # 不再改变学生或声码器输出。
                for start in range(0,len(real_cpu),4):  # 每次仅4条前向，降低特征图显存占用。
                    real,fake = real_cpu[start:start+4].to(device),fake_cpu[start:start+4].to(device)  # 成对固定波形。
                    n=len(real); count+=n  # 小批仍按真实样本数加权。
                    for bi,branch in enumerate(discriminator.loss_branches()):  # 8个分支分别只读测量。
                        rs,rf=branch(real); fs,ff=branch(fake)  # 获得分数和各层特征。
                        accum['adv']+=float((fs-1).square().mean())*n  # 与原训练目标一致。
                        accum['D']+=float((rs-1).square().mean()+fs.square().mean())*n  # 判别器误差。
                        for li,(r,f) in enumerate(zip(rf,ff)):  # 同层成对比较。
                            err=float((r-f).abs().mean()); scale=float(r.abs().mean())  # 同时记录误差和参考特征幅度。
                            accum['fm_raw']+=err*n  # 原目标逐层相加。
                            layers[f'{bi}:{li}']['abs_difference']+=err*n  # 逐层原始误差。
                            layers[f'{bi}:{li}']['real_abs_mean']+=scale*n  # 查参考尺子的放大。
                            layers[f'{bi}:{li}']['fake_abs_mean']+=float(f.abs().mean())*n  # 查生成特征幅度。
                        del rs,rf,fs,ff  # 及时释放当前分支特征图。
            row={k:v/count for k,v in accum.items()}  # 转为100句均值。
            row['layers']={k:{j:v/count for j,v in a.items()} for k,a in layers.items()}  # 保存逐层尺度证据。
            result['critic_cross'][dlabel][label]=row  # 相同学生配上不同D的交叉表。
            write_json(out/'partial_results.json',result)  # 不影响任何旧训练产物。
            print('CRITIC_RESULT '+json.dumps({'critic':dlabel,'model':label,**{k:v for k,v in row.items() if k!='layers'}}),flush=True)  # 显示核心交叉结果。
    write_json(out/'results.json',result)  # 标记只读诊断完整结束。
    print('READ_ONLY_DIAGNOSTIC_COMPLETE',flush=True)  # 明确没有启动训练。


if __name__=='__main__':  # 导入此文件不会执行任何任务。
    main()  # 只读前向与结果保存，不含优化器、backward或参数更新。
