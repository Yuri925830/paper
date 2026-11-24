# 持久保存可比较的损失窗口，并对持续上升及时写出报警记录。
"""Persist comparable loss windows and promptly flag sustained increases.

Alerts diagnose trends; they do not redefine losses or change optimizer steps.
The thread heartbeat reads loss_alerts.jsonl to notify the user.
"""

# 把每个损失窗口和报警保存为 JSON，方便后续监控程序逐条读取。
import json
# 用 isfinite 识别 NaN 和无穷大，防止把无效损失当成趋势。
import math
# 取得训练进程号，并用 fsync/replace 确保监控记录及时、安全地落盘。
import os
# 给窗口、报警和本次运行记录时间戳。
import time
# 用 Path 组织当前实验目录中的三份监控文件。
from pathlib import Path


# 收集训练批次的损失；满足持续上涨条件时写报警，训练更新由外部训练器负责。
class LossMonitor:
    # 创建监控器：默认每 50 个训练批次形成一个独立窗口；resume_step 是断点实际恢复的步数。
    def __init__(self, output_dir, window_steps=50, resume_step=0):
        # 窗口至少要包含一个批次，才能计算平均损失。
        if window_steps < 1:
            # 拒绝零或负窗口长度，避免除零和永远无法完成窗口。
            raise ValueError("window_steps must be positive")
        # 所有曲线窗口、报警和状态都写进本次实验的输出目录。
        self.output_dir = Path(output_dir)
        # 目录尚未建立时连同上级目录创建；已存在时继续使用。
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 保存每个窗口需要累积的批次数，后续据此决定何时计算均值。
        self.window_steps = int(window_steps)
        # 窗口文件每行保存一条完整的训练损失均值记录。
        self.window_path = self.output_dir / "loss_windows.jsonl"
        # 报警文件记录持续上涨或非有限损失，供线程监控及时提醒用户。
        self.alert_path = self.output_dir / "loss_alerts.jsonl"
        # 状态文件只保留最近状态，用于快速查看当前进程、epoch 和 step。
        self.status_path = self.output_dir / "training_status.json"
        # 按损失名称累加当前窗口内的数值，例如总损失和原始蒸馏损失。
        self.totals = {}
        # 当前窗口已接收的训练批次数从零开始。
        self.count = 0
        # 只需保留最近四个完整窗口，就能检查是否连续上涨三次。
        self.windows = []
        # 记录上一窗口是否已处于报警状态，避免同一次持续上涨重复通知。
        self.active_alert = False
        # 保存最近看到的优化器更新步数，便于退出时报告进度。
        self.last_global_step = 0
        # 保存最近看到的训练轮次。
        self.last_epoch = 0
        # 用纳秒时间和进程号标记这次启动，区分断电前后留下的记录。
        self.session_id = f"{time.time_ns()}-{os.getpid()}"
        # 断点步数不能为负，负值意味着调用参数错误。
        if resume_step < 0:
            # 在读取历史窗口之前拒绝非法断点，避免错误回退监控记录。
            raise ValueError("resume_step cannot be negative")
        # 续训目录已有窗口日志时，尝试恢复与检查点一致的历史。
        if self.window_path.exists():
            # 用 global_step 作键，若同一步曾写过多次就保留最后一条有效记录。
            restored = {}
            # 逐行读取历史 JSON 窗口；一行代表一个已经统计完成的窗口。
            for line in self.window_path.read_text(encoding="utf-8").splitlines():
                # 单条日志可能因突然断电只写了一半，因此单独处理解析异常。
                try:
                    # 把当前行恢复成含 step 和各项均值的字典。
                    item = json.loads(line)
                    # Optimizer state rolls back to the checkpoint after an
                    # interruption. Never mix later abandoned windows with it.
                    # 只恢复不晚于模型断点的窗口，避免把已经回滚的训练统计混进新曲线。
                    if item["global_step"] <= resume_step:
                        # 将这一条与恢复步数一致的窗口按更新步数登记。
                        restored[item["global_step"]] = item
                # 忽略无法解析的半行日志；它不能成为有效的损失观察。
                except json.JSONDecodeError:
                    # A partial append cannot be treated as a valid observation.
                    # 跳过损坏的这一行，继续读取后面的完整窗口。
                    continue
            # 按真实训练步数排序，并取最新四个窗口恢复趋势判断。
            self.windows = [restored[key] for key in sorted(restored)][-4:]
            # 检查恢复后的总损失或蒸馏损失是否已经处于持续上涨状态。
            self.active_alert = any(
                # 所有历史窗口都必须包含同一项损失，才能直接比较它们。
                all(name in item["mean"] for item in self.windows)
                # 对这项损失的四个窗口均值应用相同的上涨判断规则。
                and self.sustained_rise([item["mean"][name] for item in self.windows])
                # 分别关注完整学生损失和未乘权重的蒸馏损失，便于分辨异常来源。
                for name in ("student_total", "dist_raw")
            )
        # 保存监控器的启动时间，作为本次会话的时间参考。
        self.started = time.time()

    # 趋势判断只依赖给定数列，无须读取或改变监控器状态。
    @staticmethod
    # 判断最近四个窗口是否逐个上涨，而且首尾累计涨幅达到 2%。
    def sustained_rise(values):
        # 四个完整窗口连续三次上升且累计至少增加 2% 才触发持续上升条件。
        """Four complete windows, three rises, at least 2% overall growth."""
        # 少于四个完整窗口时，还没有足够记录判断连续三次上涨。
        if len(values) < 4:
            # 证据不足时不产生持续上涨报警。
            return False
        # 只考察最新四个窗口，早期训练的变化不干扰当前判断。
        values = values[-4:]
        # 以下三个条件必须同时满足，才认定出现持续上涨。
        return (
            # 四个均值都必须是有效数字；NaN/Inf 由专门的异常报警处理。
            all(math.isfinite(v) for v in values)
            # 逐个比较相邻窗口，要求三次变化全部向上。
            and all(b > a for a, b in zip(values, values[1:]))
            # 首尾差至少为首值绝对值的 2%；1e-8 下限过滤接近零时的浮点微差。
            and values[-1] - values[0] >= max(abs(values[0]) * 0.02, 1e-8)
        )

    # 向指定 JSONL 文件追加一条记录，并保证旧文件末尾的半行不会吞掉新记录。
    def _append(self, path, record):
        # 复制待写记录并附上当前会话标识，保留调用方原字典不受修改。
        record = dict(record, session_id=self.session_id)
        # 已有且非空的日志需要先检查最后一个字节是否为换行。
        if path.exists() and path.stat().st_size:
            # 以字节形式查看末尾，避免文本编码和换行转换干扰判断。
            with path.open("rb") as previous:
                # 把读取位置移到文件最后一个字节，不必重新读完整日志。
                previous.seek(-1, 2)
                # 末尾没有换行时，旧记录可能尚未完整写入。
                partial_line = previous.read(1) != b"\n"
        # 新文件或空文件没有需要隔离的末尾半行。
        else:
            # 允许直接追加本次 JSON 记录。
            partial_line = False
        # 以追加模式写入，保留同一次实验已有的全部窗口和报警。
        with path.open("a", encoding="utf-8") as handle:
            # 先把没有换行的旧尾部与新记录隔开。
            if partial_line:
                # 补一个换行，使后续读取仍能找到新写入的完整 JSON 行。
                handle.write("\n")
            # 序列化为一行 UTF-8 JSON；禁止 NaN/Inf，保证监控文件可严格解析。
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            # 把 Python 的写入缓冲及时送到操作系统。
            handle.flush()
            # 要求操作系统将记录同步到磁盘，降低再次意外关机时丢日志的风险。
            os.fsync(handle.fileno())

    # 用原子替换更新当前状态，避免监控读取到正在写一半的 JSON。
    def _status(self, record):
        # 复制待写记录并附上当前会话标识，保留调用方原字典不受修改。
        record = dict(record, session_id=self.session_id)
        # 先在同目录准备临时状态文件，完整写好后再替换正式文件。
        temporary = self.status_path.with_suffix(".json.tmp")
        # 重写临时文件，只存当前状态快照。
        with temporary.open("w", encoding="utf-8") as handle:
            # 写成便于人工查看的缩进 JSON，同时拒绝非法浮点值。
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            # 把 Python 的写入缓冲及时送到操作系统。
            handle.flush()
            # 要求操作系统将记录同步到磁盘，降低再次意外关机时丢日志的风险。
            os.fsync(handle.fileno())
        # 一次替换正式状态文件，让读者看到旧版或新版完整状态。
        os.replace(temporary, self.status_path)

    # 每完成一个训练批次调用一次，接收当前步数、轮次和分项损失。
    def observe(self, global_step, epoch, losses):
        # 更新最近一次成功观察到的训练步数。
        self.last_global_step = int(global_step)
        # 更新这批损失所属的训练轮次。
        self.last_epoch = int(epoch)
        # 把张量或数值统一转为普通浮点数，便于累计并写入 JSON。
        values = {name: float(value) for name, value in losses.items()}
        # 只要任何一项出现 NaN/Inf，就立即记录异常，不能等到窗口结束。
        if not all(math.isfinite(value) for value in values.values()):
            # 把异常发生的位置写入报警日志，供外部通知程序读取。
            self._append(self.alert_path, {
                # 标记为非有限损失报警，并记录发生异常的优化器步数。
                "kind": "nonfinite_loss", "global_step": int(global_step),
                # 保存异常轮次和墙上时钟时间，方便与训练日志对应。
                "epoch": int(epoch), "time": time.time(),
            })
            # 中止包含无效损失的训练调用，避免继续更新出损坏权重。
            raise FloatingPointError("Loss monitor received NaN/Inf")
        # 逐项累计总损失、对抗损失、特征匹配和蒸馏等统计值。
        for name, value in values.items():
            # 把本批对应损失加到当前窗口；首次出现的项从零开始。
            self.totals[name] = self.totals.get(name, 0.0) + value
        # 将本批计入当前窗口的样本批次数。
        self.count += 1
        # 本次观察共用同一个时间戳，便于关联状态、窗口和报警。
        now = time.time()
        # 窗口尚未收满时，暂时只更新轻量进度状态。
        if self.count < self.window_steps:
            # 第一步及每十步刷新状态，兼顾及时查看进度和写盘开销。
            if global_step == 1 or global_step % 10 == 0:
                # 刷新外部监控读取的当前训练状态。
                self._status({
                    # 标记仍在训练，附上进程号和状态更新时间。
                    "status": "running", "pid": os.getpid(), "time": now,
                    # 记录当前总更新步数和 epoch，便于核对恢复进度。
                    "global_step": int(global_step), "epoch": int(epoch),
                    # 未形成完整窗口时展示最近批次损失，并声明窗口长度。
                    "last_batch": values, "window_steps": self.window_steps,
                })
            # 本批记录完成；尚未收满窗口时不执行趋势判断。
            return
        # 为刚完成的非重叠窗口建立一条可长期保存的均值记录。
        record = {
            # 记录窗口结束位置及时间，不把窗口均值误当成单批损失。
            "global_step": int(global_step), "epoch": int(epoch), "time": now,
            # 写入本窗口实际累积的批次数。
            "window_steps": self.count,
            # 每项累计损失除以窗口批次数，得到可比较的窗口均值。
            "mean": {name: total / self.count for name, total in self.totals.items()},
        }
        # 将完整窗口持久化，以便画趋势和断点恢复。
        self._append(self.window_path, record)
        # 加入新窗口并只保留最近四条，限制监控内存占用。
        self.windows = (self.windows + [record])[-4:]
        # 收集满足持续上涨条件的损失名称，报警可同时包含多项。
        rising = [
            # 只对学生总损失和原始蒸馏损失执行预先规定的持续上涨检测。
            name for name in ("student_total", "dist_raw")
            # 要求四个窗口都有当前损失，防止缺字段造成不可靠比较。
            if all(name in item["mean"] for item in self.windows)
            # 对这项损失的四个窗口均值应用相同的上涨判断规则。
            and self.sustained_rise([item["mean"][name] for item in self.windows])
        ]
        # 从正常状态首次进入上涨状态时通知，持续同一报警期间不重复轰炸。
        if rising and not self.active_alert:
            # 组织本次上涨报警的原因、位置和原始窗口证据。
            alert = {
                # 明确报警类别，并列出上涨的是总损失还是蒸馏损失。
                "kind": "sustained_loss_rise", "metrics": rising,
                # 标记检测到上涨的时间和训练位置。
                "time": now, "epoch": int(epoch), "global_step": int(global_step),
                # 把报警阈值直接写进记录，教授或用户可核对触发依据。
                "definition": "4 non-overlapping windows; 3 successive rises; >=2% total growth",
                # 附上四个完整窗口的数据，保留可审查的数值证据。
                "windows": self.windows,
            }
            # 先把报警可靠地写到文件，让线程监控能读取。
            self._append(self.alert_path, alert)
            # 终端同步打印醒目的固定前缀并立即刷新，方便实时发现异常。
            print("LOSS_TREND_ALERT " + json.dumps(alert, ensure_ascii=False), flush=True)
        # 更新当前报警状态；趋势恢复正常后，后续新一轮上涨可再次通知。
        self.active_alert = bool(rising)
        # 刷新外部监控读取的当前训练状态。
        self._status({
            # 标记仍在训练，附上进程号和状态更新时间。
            "status": "running", "pid": os.getpid(), "time": now,
            # 记录当前总更新步数和 epoch，便于核对恢复进度。
            "global_step": int(global_step), "epoch": int(epoch),
            # 状态快照展示最新完整窗口及当前正在上涨的损失项。
            "last_window": record, "active_rise": rising,
            # 公开配置的窗口批次数，便于理解监控的时间分辨率。
            "window_steps": self.window_steps,
        })
        # 清空本窗口累计值，使下一个窗口不与这一窗口重叠。
        self.totals, self.count = {}, 0

    # 训练正常结束、被中断或失败时，写入最终状态供外部确认。
    def finish(self, status, global_step, epoch):
        # 刷新外部监控读取的当前训练状态。
        self._status({
            # 记录调用方给出的结束状态、进程号及结束时间。
            "status": status, "pid": os.getpid(), "time": time.time(),
            # 记录当前总更新步数和 epoch，便于核对恢复进度。
            "global_step": int(global_step), "epoch": int(epoch),
        })
