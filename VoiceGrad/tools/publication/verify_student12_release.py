"""学生12号发布的静态文件核验；不导入模型、不执行训练、推理或评分。"""  # 仅处理发布元数据。
import argparse  # 读取仓库位置及本地可选的基准树清单。
import ast  # 只解析Python语法，不导入项目模块。
import hashlib  # 核对源文件字节身份。
import json  # 校验机器可读记录和发布清单。
from pathlib import Path  # 限定所有文件在指定仓库内。
import re  # 检查当前说明中的相对文件链接。
from urllib.parse import unquote  # 解码Markdown路径，不进行网络访问。

def inspect(root, base_tree=None):  # 静态核验不接受任何模型或服务器参数。
    root = Path(root).resolve()  # 固定当前本地发布目录。
    manifest = json.loads((root/'artifacts/student12_122900/publication_manifest.json').read_text())  # 读取明确的发布范围。
    errors = []  # 汇总所有不一致而非只报告第一个。
    metadata = json.loads(Path(base_tree).read_text()) if base_tree else []  # 可在无完整clone时使用已经读取的Git树。
    known = {row['path'] for row in metadata}  # 此集合只用于相对链接存在性，不代替内容哈希核验。
    checked_sources = 0  # 记录实际读取的源文件数量。
    for entry in manifest['promoted_source_files']:  # 所有新提升文件必须与源快照逐字节一致。
        path = root/entry['path']  # 使用发布的相对路径。
        if not path.is_file():  # 不以清单声明代替本地文件。
            errors.append('missing source: '+entry['path'])  # 记录缺文件。
            continue  # 继续检查其他来源。
        data = path.read_bytes()  # 只读取文本/元数据字节，不反序列化checkpoint。
        git_sha = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()  # 计算标准Git blob身份。
        if git_sha != entry['git_blob_sha']: errors.append('source blob differs: '+entry['path'])  # 不隐式改写历史运行源码。
        if path.suffix == '.py': ast.parse(data.decode('utf-8-sig'), filename=entry['path'])  # 新提升的Python只做语法解析。
        checked_sources += 1  # 成功读取后计入实际核验数。
    for entry in manifest['authored_files']:  # 文档/摘要与静态核验工具也必须绑定实际内容。
        path = root/entry['path']  # 不扫描无关用户目录。
        data = path.read_bytes()  # 不执行任何项目代码。
        if hashlib.sha256(data).hexdigest() != entry['sha256']: errors.append('authored file differs: '+entry['path'])  # 防止检查后内容变化。
        if path.suffix == '.json': json.loads(data)  # 当前新增JSON必须可解析。
        if path.suffix == '.py': ast.parse(data.decode('utf-8-sig'), filename=entry['path'])  # 不将语法检查称为运行时测试。
        if path.suffix == '.md':  # 当前文档的文件链接需可到达。
            text = re.sub(r'```.*?```', '', data.decode('utf-8'), flags=re.S)  # 不把示例代码误当Markdown链接。
            for target in re.findall(r'\]\(([^)]+)\)', text):  # 检查项目内文件，不假装验证外部URL可用性。
                target = target.strip().strip('<>')  # 支持含空格的规范Markdown路径。
                if not target or target.startswith(('#','https://','http://','mailto:')): continue  # 外部链接不在静态范围。
                location = (path.parent/unquote(target.split('#')[0])).resolve()  # 锚点单独留给渲染器。
                try: relative = location.relative_to(root).as_posix()  # 不允许相对链接逃离仓库。
                except ValueError: errors.append('link escapes root: '+target); continue  # 不读取工作区之外的文件。
                if not location.exists() and relative not in known: errors.append('missing link '+entry['path']+' -> '+target)  # 同时支持现存基准树文件。
    results = json.loads((root/'artifacts/student12_122900/results.json').read_text())  # 只读取本次明确发布的学生终点。
    if results['scope']['endpoint_global_step'] != 122900 or results['scope']['subsequent_experiments_included']: errors.append('wrong experiment cutoff')  # 拒绝混入后续实验。
    counts = results['cer_counts']  # 保留语料级分母。
    if abs(100*counts['character_errors']/counts['reference_characters']-results['primary_metrics']['corpus_cer_percent']) > 1e-12: errors.append('CER count mismatch')  # 验证汇总定义。
    sva = results['separate_sva_rescore']  # SVA仅作为带资格重评分。
    if abs(100*sva['passes']/results['evaluated_pairs']-sva['acceptance_percent']) > 1e-12: errors.append('SVA count mismatch')  # 不把通过率当相似度百分比。
    if not sva.get('qualification'): errors.append('missing SVA precision qualification')  # 不丢弃历史限制。
    forbidden = ('cycle_post','base_post','vq_reproduction.py','fastvoicegrad_formal.py')  # 明确排除本次截止之后的已知实验入口。
    for item in manifest['promoted_source_files']+manifest['authored_files']:  # 只核对本次实际发布的路径。
        if any(word in item['path'] for word in forbidden): errors.append('out-of-scope file: '+item['path'])  # 不将其他实验文件夹一起上传。
    return {'status':'pass' if not errors else 'fail','source_files_checked':checked_sources,'authored_files_checked':len(manifest['authored_files']),'python_validation':'AST only; no imports or runtime tests','link_validation':'repository-relative file targets only; external URLs/anchors not certified','model_execution':False,'server_access':False,'errors':errors}  # 明确核验范围。

if __name__ == '__main__':  # 本工具不启动任何生产入口。
    parser = argparse.ArgumentParser(description=__doc__)  # 明确为发布静态检查。
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])  # 可在普通checkout或暂存目录使用。
    parser.add_argument('--base-tree', type=Path)  # 本地无完整clone时传入已经读取的Git目录清单。
    args = parser.parse_args()  # 不接受训练设备或模型参数。
    report = inspect(args.root, args.base_tree)  # 不导入任何项目模型模块。
    print(json.dumps(report, ensure_ascii=False, indent=2))  # 只输出真实静态结果。
    raise SystemExit(0 if report['status']=='pass' else 1)  # 任一不一致都会阻止发布。
