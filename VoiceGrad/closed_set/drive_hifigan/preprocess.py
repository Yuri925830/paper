import os
import glob
import random

def create_file_lists(wav_dir, output_dir):
    """
    扫描 wav_dir 下所有的 .wav 文件，按 95:5 划分训练集和验证集。
    生成的 txt 文件中每行是一个 wav 文件的绝对路径。
    """
    print(f"正在搜索音频文件: {wav_dir} ...")
    
    # 递归搜索所有 .wav 文件 (支持 wav/bdl/xxx.wav 这种嵌套结构)
    # 使用 recursive=True 配合 ** 模式
    search_pattern = os.path.join(wav_dir, '**', '*.wav')
    all_files = glob.glob(search_pattern, recursive=True)
    
    # 获取绝对路径，防止相对路径在训练时出错
    all_files = [os.path.abspath(f) for f in all_files]
    
    # 必须排序以保证每次运行结果一致（在打乱前）
    all_files.sort()
    
    # 打乱顺序
    random.seed(1234)
    random.shuffle(all_files)
    
    print(f"总共找到 {len(all_files)} 个音频文件。")
    
    if len(all_files) == 0:
        print("错误：未找到 .wav 文件，请检查路径是否正确。")
        return

    # 划分训练集和验证集 (95% 训练, 5% 验证)
    split_idx = int(len(all_files) * 0.95)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # 写入 txt 文件
    train_txt_path = 'train_files.txt'
    val_txt_path = 'val_files.txt'
    
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        for path in train_files:
            f.write(f"{path}\n")
            
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        for path in val_files:
            f.write(f"{path}\n")

    print(f"✅ 已生成文件列表：")
    print(f"   - 训练集: {train_txt_path} ({len(train_files)} 条)")
    print(f"   - 验证集: {val_txt_path} ({len(val_files)} 条)")

if __name__ == '__main__':
    # --- 配置 ---
    # 请修改为你存放 wav 文件的实际路径
    # 根据你之前的描述，路径应该是这个：
    WAV_DATA_ROOT = '/root/lanyun-tmp/code3/data/wavs'
    
    create_file_lists(WAV_DATA_ROOT, '.')