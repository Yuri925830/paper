# --------------------------- 导入包逐项解释 ---------------------------
# csv 读取/写入 manifest；argparse 解析 --workers/--limit/--force。
import csv
# 定义终端参数，使数据目录、检查点和运行选项明确记录。
import argparse
# os.environ 用来读取 VOICEGRAD_PROJECT_ROOT，避免把私人服务器路径写死。
import os
# subprocess 从 Python 启动外部 ffmpeg 程序，并检查它是否成功退出。
import subprocess
# tempfile 在目标目录创建唯一临时名；wave 用标准库审核输出 WAV 头。
import tempfile
# 读取 WAV 文件头与末帧，确认采样率、声道和实际文件完整性。
import wave
# Path 负责路径拼接、建目录、判断文件存在。
from pathlib import Path
# Pool 创建多个 Python 子进程并行转码；partial 预先固定 force 参数。
from multiprocessing import Pool
# 把统一的强制重算选项绑定到每条音频的处理函数。
from functools import partial

# =============================================================================
# 预处理第 01 步：把 VCTK 原始 wav 统一为论文要求的 22.05 kHz 单声道
# =============================================================================
# 为什么必须重采样？
# VCTK 原始录音通常是 48 kHz，而论文明确使用 22.05 kHz。如果采样率不同，
# 同样的 n_fft/hop 会代表不同的真实时间长度，mel 特征和 HiFi-GAN 都会不匹配。
#
# 本脚本的输入来自 00_build_vctk_manifest.py 生成的 vctk_all_pairs.csv，其中每行
# 记录 speaker、utt_id 和原始 wav48_path。输出目录结构为：
#   data/wav22k/<speaker>/<utt_id>.wav
# 同时写成功清单 vctk_wav22k_manifest.csv 和失败清单。
#
# 这里调用系统 ffmpeg，而不是用 Python 手写重采样滤波器。ffmpeg 的重采样实现
# 稳定、支持多种 wav 编码，并可明确指定单声道、采样率和 16-bit PCM。

# 本脚本位于 preprocess/，所以 parents[1] 正好是仓库根目录。
PROJECT_ROOT = Path(
    # 优先读取显式工程根目录；没有环境变量时回到本文件向上两级的仓库目录。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[1])
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()
# 下面三个常量是输入清单、输出音频目录和输出清单目录。
MANIFEST_CSV = PROJECT_ROOT / "data/metadata/vctk_all_pairs.csv"
# 全部重采样 WAV 保存到 data/wav22k，按说话人建立子目录。
OUT_ROOT = PROJECT_ROOT / "data/wav22k"
# 重采样清单和失败清单统一保存到元数据目录。
META_ROOT = PROJECT_ROOT / "data/metadata"


# 检查生成 WAV 的采样率、单声道格式和文件末尾完整性；下面的参数分别限定输入数据和处理约束。
def validate_resampled_wav(wav_path: Path) -> None:
    # 检查生成 WAV 的采样率、单声道格式和文件末尾完整性。
    """Raise when ``wav_path`` is not a non-empty mono 22.05-kHz WAV."""
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在受控文件句柄中完成检查生成 WAV 的采样率、单声道格式和文件末尾完整性，退出此段后自动关闭文件。
        with wave.open(str(wav_path), "rb") as wav_file:
            # 音频文件的声道数。
            channels = wav_file.getnchannels()
            # 音频文件实际或预期的采样率。
            sample_rate = wav_file.getframerate()
            # 该文件的真实波形帧数或特征时间帧数。
            frames = wav_file.getnframes()
            # 每个波形采样值占用的字节数。
            sample_width = wav_file.getsampwidth()
            # 从文件尾实际读取的一帧，用于识别截断音频。
            last_frame = b""
            # 按该文件的真实波形帧数或特征时间帧数决定是否进入此分支；判断条件为 frames > 0。
            if frames > 0:
                # 只读最后一帧即可发现“头部已写、数据被截断”的中断文件。
                wav_file.setpos(frames - 1)
                # 从文件尾实际读取的一帧，用于识别截断音频。
                last_frame = wav_file.readframes(1)
    # 捕获检查生成 WAV 的采样率、单声道格式和文件末尾完整性中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except (OSError, EOFError, wave.Error) as exc:
        # 检查生成 WAV 的采样率、单声道格式和文件末尾完整性未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Unreadable WAV: {wav_path}: {exc}") from exc

    # 按音频文件的声道数决定是否进入此分支；判断条件为 channels != 1。
    if channels != 1:
        # 检查生成 WAV 的采样率、单声道格式和文件末尾完整性未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Expected mono WAV, got {channels} channels: {wav_path}")
    # 按音频文件实际或预期的采样率决定是否进入此分支；判断条件为 sample_rate != 22050。
    if sample_rate != 22050:
        # 检查生成 WAV 的采样率、单声道格式和文件末尾完整性未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 报错列出实际 WAV 采样率及路径，要求固定为 22.05 kHz。
            f"Expected 22050-Hz WAV, got {sample_rate} Hz: {wav_path}"
        )
    # 按该文件的真实波形帧数或特征时间帧数决定是否进入此分支；判断条件为 frames <= 0。
    if frames <= 0:
        # 检查生成 WAV 的采样率、单声道格式和文件末尾完整性未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"WAV contains no audio frames: {wav_path}")
    # 按每个波形采样值占用的字节数决定是否进入此分支；判断条件为 sample_width <= 0 or len(last_frame) != channels * sample_width。
    if sample_width <= 0 or len(last_frame) != channels * sample_width:
        # 检查生成 WAV 的采样率、单声道格式和文件末尾完整性未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"WAV payload is truncated: {wav_path}")


# 为单条重采样结果保留同目录唯一临时 WAV；下面的参数分别限定输入数据和处理约束。
def _new_wav_temp_path(dst: Path) -> Path:
    # 为单条重采样结果保留同目录唯一临时 WAV。
    """Reserve a unique temporary WAV beside ``dst`` for atomic publication."""
    # 当前打开的文件句柄。
    handle = tempfile.NamedTemporaryFile(
        # 临时文件名包含目标语句编号，便于定位并避免不同语音互相覆盖。
        prefix=f".{dst.stem}.",
        # 保留 WAV 扩展名，让转换工具按 WAV 格式写入临时文件。
        suffix=".tmp.wav",
        # 临时文件与最终 WAV 位于同一目录，保证之后可做同文件系统原子替换。
        dir=str(dst.parent),
        # 退出临时文件句柄时先保留路径，供外部音频转换程序写入。
        delete=False,
    )
    # 当前操作专用的临时文件路径。
    temp_path = Path(handle.name)
    # 关闭预先创建的空临时文件句柄，避免转换程序同时持有冲突句柄。
    handle.close()
    # 返回当前操作专用的临时文件路径，供调用方继续处理。
    return temp_path

# 用 ffmpeg 将一条原音频转换为 22.05 kHz 单声道；下面的参数分别限定输入数据和处理约束。
def convert_one(row, force=False):
    # 这个函数只处理 CSV 中的一行，也就是一条 utterance。
    # multiprocessing.Pool 会让多个子进程并行调用它，因此不要在这里保存跨语音状态。
    spk = row["speaker"]
    # 同时包含说话人和句号的样本编号。
    utt_id = row["utt_id"]
    # 原始音频路径。
    src = Path(row["wav48_path"])
    # 把重采样结果归入当前说话人的目录。
    dst_dir = OUT_ROOT / spk
    # parents=True 自动创建上级目录；exist_ok=True 表示目录已经存在也不报错。
    dst_dir.mkdir(parents=True, exist_ok=True)
    # 重采样后的目标 WAV。
    dst = dst_dir / (utt_id + ".wav")

    # 断点续跑不能只看文件名：中断留下的空文件/截断 WAV 必须重做。
    if dst.exists() and (not force):
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # 复用已有 WAV 前检查头信息、采样率、声道和文件尾，拒绝中断残留的半文件。
            validate_resampled_wav(dst)
            # 返回可信复用标记和输出位置，汇入正式可用音频清单。
            return ("skip", spk, utt_id, str(dst), "")
        # 捕获用 ffmpeg 将一条原音频转换为 22.05 kHz 单声道中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except ValueError:
            # 保留旧的无效文件，直到新结果验证通过并原子替换它。
            pass

    # 当前操作专用的临时文件路径。
    temp_path = _new_wav_temp_path(dst)

    # subprocess 推荐用“字符串列表”传命令，每个元素就是一个命令行参数，
    # 不需要自己处理空格和引号，也不会经过 shell 字符串拼接。
    cmd = [
        # 调用音频转换程序完成重采样。
        "ffmpeg",
        # -y：允许覆盖已有输出；真正是否覆盖已由上面的 force 逻辑控制。
        "-y",
        # -v error：只打印错误，避免四万条音频的普通日志刷满终端。
        "-v", "error",
        # 把当前库存记录中的源音频作为 ffmpeg 输入。
        "-i", str(src),
        # -ac 1：转单声道；-ar 22050：采样率；-sample_fmt s16：16-bit PCM。
        "-ac", "1",
        # 将目标 WAV 采样率固定为 22,050 Hz。
        "-ar", "22050",
        # 以 16 位有符号整数格式保存重采样 WAV。
        "-sample_fmt", "s16",
        # 将转换结果先写入当前专属临时 WAV，不直接覆盖旧的有效结果。
        str(temp_path)
    ]

    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # check=True 表示 ffmpeg 返回非零退出码时自动抛 CalledProcessError。
        subprocess.run(cmd, check=True)
        # 只有临时文件通过单声道/22.05 kHz/非空检查才发布。
        validate_resampled_wav(temp_path)
        # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
        os.replace(temp_path, dst)
        # 重采样、完整性检查和原子发布均成功后，返回完成记录。
        return ("ok", spk, utt_id, str(dst), "")
    # ffmpeg 不存在、转码失败或结果验证失败都记入失败清单。
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        # 将当前语音及具体异常返回主进程，失败文件不加入可用清单。
        return ("fail", spk, utt_id, str(dst), str(e))
    # 无论处理成功或失败，都执行后面的临时资源清理。
    finally:
        # 成功 os.replace 后路径已不存在；失败时清理本次临时文件。
        try:
            # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
            temp_path.unlink()
        # 捕获用 ffmpeg 将一条原音频转换为 22.05 kHz 单声道中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except FileNotFoundError:
            # 此异常属于允许忽略的清理或兼容情形，后续流程继续执行。
            pass

# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main():
    # --workers 控制并行子进程数；--limit 方便先取少量数据试运行；
    # --force 是布尔开关，出现时 True，不写时 False。
    parser = argparse.ArgumentParser()
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--workers", type=int, default=6)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--limit", type=int, default=0,
                        # 下面的命令行帮助说明此选项的用途和实验限制。
                        help="Only process the first N utterances for sanity check")
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--force", action="store_true",
                        # 下面的命令行帮助说明此选项的用途和实验限制。
                        help="Overwrite existing wav22k files")
    # 终端传入并经解析的运行设置。
    args = parser.parse_args()

    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    META_ROOT.mkdir(parents=True, exist_ok=True)

    # DictReader 把 CSV 每行转换为字典，例如 row["wav48_path"]。
    rows = []
    # 读取原始 VCTK 库存清单，按记录中的路径寻找源音频。
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        # 按表头返回字典的 CSV 读取器。
        reader = csv.DictReader(f)
        # 逐项遍历按表头返回字典的 CSV 读取器，保持每条记录都经过相同检查。
        for row in reader:
            # 保留该条库存记录，后续交给同一重采样流程。
            rows.append(row)

    # 列表切片 rows[:N] 取前 N 条。正式全量运行时 limit 默认 0，不截断。
    if args.limit > 0:
        # 本次选定的语音清单。
        rows = rows[:args.limit]

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("Total rows to process:", len(rows))
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("Workers:", args.workers)
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("Output root:", OUT_ROOT)

    # partial 预先固定 force 参数，生成只需要接收 row 的新函数，便于 Pool 调用。
    fn = partial(convert_one, force=args.force)

    # 收集本轮新转换成功的音频记录。
    ok_rows = []
    # 收集已存在并通过完整性检查的可复用 WAV。
    skip_rows = []
    # 单独保存无法完成重采样的语音及失败原因。
    fail_rows = []

    # with 会在代码块结束后自动关闭进程池。imap_unordered 哪条先完成就先返回，
    # 不强求原 CSV 顺序，所以并行效率比等待顺序更高。
    with Pool(processes=args.workers) as pool:
        # 多进程按完成先后收取转换结果；正式清单稍后重新排序。
        for i, result in enumerate(pool.imap_unordered(fn, rows), 1):
            # 拆出处理状态、语音身份、目标路径与错误信息。
            status, spk, utt_id, dst_path, err = result

            # 根据 convert_one 返回的三种状态分别统计，失败时额外保存 error。
            if status == "ok":
                # 为本轮新生成的有效 WAV 建立清单记录。
                ok_rows.append({
                    # 记录当前语音所属说话人。
                    "speaker": spk,
                    # 记录同时包含说话人和句号的样本编号。
                    "utt_id": utt_id,
                    # 记录已经重采样到 22.05 kHz 的音频路径。
                    "wav22k_path": dst_path
                })
            # 按本轮处理完成、部分完成或失败的状态决定是否进入此分支；判断条件为 status == "skip"。
            elif status == "skip":
                # 为已经存在且核对通过的 WAV 建立相同字段的清单记录。
                skip_rows.append({
                    # 记录当前语音所属说话人。
                    "speaker": spk,
                    # 记录同时包含说话人和句号的样本编号。
                    "utt_id": utt_id,
                    # 记录已经重采样到 22.05 kHz 的音频路径。
                    "wav22k_path": dst_path
                })
            # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
            else:
                # 失败条目只进入独立错误清单，不伪装为可训练音频。
                fail_rows.append({
                    # 记录当前语音所属说话人。
                    "speaker": spk,
                    # 记录同时包含说话人和句号的样本编号。
                    "utt_id": utt_id,
                    # 记录已经重采样到 22.05 kHz 的音频路径。
                    "wav22k_path": dst_path,
                    # 记录本次操作捕获的具体错误。
                    "error": err
                })

            # % 是取余；每处理 500 条打印一次进度，最后一条也强制打印。
            if i % 500 == 0 or i == len(rows):
                # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
                print("Processed {}/{} | ok={} skip={} fail={}".format(
                    # 进度日志列出已处理、总数、新成功、可信复用和失败数量。
                    i, len(rows), len(ok_rows), len(skip_rows), len(fail_rows)
                ))

    # 成功和已跳过都代表目标 wav 存在，所以两类都写入后续可用清单。
    # imap_unordered 的完成顺序每次可能不同；写盘前按 speaker/utt_id 排序，
    # 让同一输入始终生成字节顺序稳定、便于审计和比较的 manifest。
    ready_rows = sorted(
        # 合并新生成与验证后复用的 WAV，二者共同构成下一步数据划分的有效输入。
        ok_rows + skip_rows,
        # 当前语音的唯一身份。
        key=lambda row: (row["speaker"], row["utt_id"]),
    )
    # 按说话人和句号排序失败记录，便于重复运行之间逐条比较。
    fail_rows.sort(key=lambda row: (row["speaker"], row["utt_id"]))
    # 确定下一步划分脚本要读取的 22.05 kHz 有效音频清单。
    out_manifest = META_ROOT / "vctk_wav22k_manifest.csv"
    # 以固定 UTF-8 CSV 格式写出本轮可用音频清单。
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        # 固定表头顺序的 CSV 写入器。
        writer = csv.DictWriter(f, fieldnames=["speaker", "utt_id", "wav22k_path"])
        # 先写列名，使后续行能按相同字段重新读取。
        writer.writeheader()
        # 按稳定排序逐条写出可用音频，不保留多进程随机完成顺序。
        for r in ready_rows:
            # 把已整理的清单记录按固定字段顺序写入 CSV。
            writer.writerow(r)

    # 将转换失败原因保存到单独文件，便于补齐数据后重试。
    fail_csv = META_ROOT / "vctk_wav22k_failures.csv"
    # 写入本轮失败清单，避免错误信息只留在终端滚动输出中。
    with open(fail_csv, "w", newline="", encoding="utf-8") as f:
        # 固定表头顺序的 CSV 写入器。
        writer = csv.DictWriter(f, fieldnames=["speaker", "utt_id", "wav22k_path", "error"])
        # 先写列名，使后续行能按相同字段重新读取。
        writer.writeheader()
        # 逐条保存失败语音身份、路径和异常说明。
        for r in fail_rows:
            # 把已整理的清单记录按固定字段顺序写入 CSV。
            writer.writerow(r)

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print()
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("===== Resample Summary =====")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("ok   :", len(ok_rows))
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("skip :", len(skip_rows))
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("fail :", len(fail_rows))
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("saved manifest :", out_manifest)
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("saved failures :", fail_csv)

    # 只打印失败但返回成功状态会让后续脚本继续处理残缺数据，所以有失败就以
    # 非零退出码停止整个 shell 流水线。失败详情已保存在 fail_csv。
    if fail_rows:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise SystemExit(
            # 报错说明重采样失败总数及逐条失败清单位置。
            f"Resampling failed for {len(fail_rows)} utterances; see {fail_csv}."
        )

# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 直接运行本脚本才执行 main；被其他 Python 文件 import 时不自动重采样。
    main()
