from TranslatorLib import zipfile, json, eb, defaultdict, Path, uuid
from TranslatorCore import Translator
import TranslatorPersistence

class TranslatorTool:
    def __init__(Self, Config: dict = None):
        Self.Translator = Translator(Config=Config or {})
        Self.Config = Self.Translator.Config
        Self.Module = Self.Translator.Module
        Self.Locale = Self.Translator.Locale
        Self.Quantization = Self.Translator.Quantization
        Self.Lang = Self.Translator.Lang
        Self.日志 = Self.Module.写入日志
        Self.tqdm = Self.Locale.Tqdm
    def 语言文件对转DictMini(Self, File0: str, DictMini: str = None, OutputPath: str = "./"):
        Self.日志("log.core.file.settle.start", info_level=0)
        try:
            File0 = Path(File0).resolve()
            if File0.suffix == ".zip":
                缓存路径 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}"
                Path(缓存路径).mkdir(parents=True, exist_ok=True)
                for _ in Self.tqdm(range(1), desc="tqdm.file.read"): 
                    with zipfile.ZipFile(File0, "r") as f:
                        f.extractall(path=缓存路径)
            else: 缓存路径 = File0
            缓存路径 = Path(缓存路径)
            符合条件的文件夹 = []
            for 子目录 in 缓存路径.rglob("*"):
                if 子目录.is_dir():
                    文件列表 = set(子目录.iterdir())
                    文件名集合 = {f.name for f in 文件列表 if f.is_file()}
                    文件名集合 = {name.lower() for name in 文件名集合}
                    真实文件名映射 = {}
                    for f in 子目录.iterdir():
                        if f.is_file():
                            真实文件名映射[f.name.lower()] = f.name
                    输入语言文件名称 = f"{Self.Config.LANGUAGE_INPUT.strip().lower()}"
                    输出语言文件名称 = f"{Self.Config.LANGUAGE_OUTPUT.strip().lower()}"
                    for ext in [".json", ".lang"]:
                        输入语言文件名 = f"{输入语言文件名称}{ext}"
                        输出语言文件名 = f"{输出语言文件名称}{ext}"
                        if 输入语言文件名 in 真实文件名映射 and 输出语言文件名 in 真实文件名映射:
                            符合条件的文件夹.append((子目录, 真实文件名映射[输入语言文件名], 真实文件名映射[输出语言文件名]))
                            break
            符合条件的文件夹 = list(dict.fromkeys(符合条件的文件夹))
            DictMini附加 = defaultdict(list)
            for 子目录, 输入路径, 输出路径 in 符合条件的文件夹:
                语言文件A = Self.Module.读取语言文件(str(子目录/输入路径))[0]
                语言文件B = Self.Module.读取语言文件(str(子目录/输出路径))[0]
                语言文件B = {index[0]: index[1] for index in 语言文件B}
                for index in 语言文件A:
                    try:
                        键, 值 = index[0], index[1]
                        语言文件B[键]
                        DictMini附加[值].append(语言文件B[键])
                    except: pass
            if DictMini:
                with open(DictMini, "r", encoding="utf-8") as f:
                    Dict文件 = json.load(f)
                for 原文, 值列表 in DictMini附加.items():
                    if 原文 in Dict文件:
                        for 单个值 in 值列表:
                            if 单个值 not in Dict文件[原文]:
                                Dict文件[原文].append(单个值)
                    else:
                        Dict文件[原文] = 值列表.copy()
                with open(DictMini, "w", encoding="utf-8") as f:
                    json.dump(Dict文件, f, ensure_ascii=False)
            else:
                with open(f"{OutputPath}/Dict-Mini.json", "w", encoding="utf-8") as f:
                    json.dump(DictMini附加, f, ensure_ascii=False)
            Self.日志("log.core.file.settle.end", info_level=0)
        except Exception:
            Self.日志("log.core.file.settle.error", e=eb.format_exc(), info_level=3)
    def 分离语言文件更新(Self, file0: str, file1: str = "", output_path: str = ""):
        Self.日志("log.core.lang.separate.start", info_level=0)
        output_path = Self.Module.输出路径处理(output_path)
        缺失列表 = []
        文件0, _, 文件1, _, 输出扩展名, _ = Self.Module.读取资源文件(file0, file1)
        if 文件1:
            参考字典 = {}
            for item in 文件1:
                try:
                    参考字典[item[0]] = item[1]
                except Exception:
                    Self.日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
                    pass
            for index in 文件0:
                key = index[0]
                path = index[2]
                if key in 参考字典:
                    缺失列表.append([key, 参考字典[key], path])
        else:
            缺失列表 = 文件0.copy()
        if not Path(output_path).suffix:
            导出路径 = str(Path(f"{output_path}/{Self.Config.LANGUAGE_INPUT}{输出扩展名}"))
        else:
            导出路径 = str(Path(output_path))
        Self.Module.保存语言文件(导出路径, [f"{index[0]}={index[1]}" for index in 缺失列表])
        Self.日志("log.core.settle.succeed", path=Path(导出路径).resolve(), info_level=0)
        返回路径 = Path(导出路径).resolve()
        Self.日志("log.core.lang.separate.end", info_level=0)
        return 返回路径
    def 合并语言文件更新(Self, file0: str, notlang_file: str, file1: str = "", output_path: str = ""):
        Self.日志("log.core.lang.merge.start", info_level=0)
        output_path = Self.Module.输出路径处理(output_path)
        合并列表 = []
        输出列表 = []
        缺失列表 = []
        文件0, 文件0源文件, 文件1, 压缩路径, 输出扩展名, file2 = Self.Module.读取资源文件(file0, file1)
        if 文件1:
            参考字典 = {}
            for item in 文件1:
                try:
                    参考字典[item[0]] = item[1]
                except Exception:
                    Self.日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
                    pass
            for index in 文件0:
                key = index[0]
                path = index[2]
                if key in 参考字典:
                    合并列表.append([key, 参考字典[key], path])
        else:
            合并列表 = 文件0.copy()
        未翻译文件路径 = notlang_file 
        if Path(未翻译文件路径).suffix == ".translang":
            缺失列表 = [[index[0], index[1], ""] for index in Self.Module.读取审查文件(未翻译文件路径)]
        else:
            缺失列表 = Self.Module.读取语言文件(notlang_file)[0]
        缺失列表 = {index[0]: index[1] for index in 缺失列表}
        分组 = defaultdict(dict)
        for a, b, c in 合并列表:
            分组[c][a] = b
        合并列表 = 分组.copy()
        for index in 文件0源文件:
            输出列表缓存 = []
            for index1 in index[0]:
                if index1.strip().startswith(('#', '//')):
                    输出列表缓存.append(index1)
                else:
                    index1键 = index1.split('=', 1)[0]
                    if index1键 in 缺失列表:
                        输出列表缓存.append(f"{index1键}={缺失列表[index1键]}")
                    elif index1键 in 合并列表[index[1]]:
                        输出列表缓存.append(f"{index1键}={合并列表[index[1]][index1键]}")
                    else:
                        输出列表缓存.append(index1)
            输出列表.append([index[1], 输出列表缓存])
        if 压缩路径:
            for index in 输出列表:
                Self.Module.保存语言文件(f"{Path(index[0]).parent}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}", index[1])
            压缩文件夹Path = Path(压缩路径)
            if file2[0] == False:
                压缩文件夹Path = 压缩文件夹Path.parent
                文档内容 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(name=Path(file0).stem, lang=Self.Config.LANGUAGE_OUTPUT, author=Self.Config.PACK_AUTHOR or "海盐青茫")
                with open(压缩文件夹Path/"pack.mcmeta", "w+", encoding="utf-8") as f:
                    f.write(json.dumps({"pack": {"description": 文档内容, "pack_format": 9999,"supported_formats": [0, 9999],"min_format": 0,"max_format": 9999}}, ensure_ascii=False, indent=4))
            with zipfile.ZipFile(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                for 压缩文件 in 压缩文件夹Path.rglob('*'):
                    if 压缩文件.is_file():
                        f.write(压缩文件, arcname=压缩文件.relative_to(压缩文件夹Path))
            Self.日志("log.core.translator.succeed", path=Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve(), info_level=0)
            返回路径 = Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve()
        else:
            输出路径 = str(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}")
            Self.Module.保存语言文件(输出路径, 输出列表)
            Self.日志("log.core.settle.succeed", path=Path(输出路径).resolve(), info_level=0)
            返回路径 = Path(输出路径).resolve()
        Self.日志("log.core.lang.merge.end", info_level=0)
        return 返回路径
    def 导入DictMini缓存(Self, file: str, mode="index"):
        Self.日志("log.core.file.settle.start", info_level=0)
        文本列表 = []
        for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
            with open(file, "rb") as f:
                Dict文件 = json.load(f)
        if mode == "index":
            for index in Self.tqdm(Dict文件, desc="tqdm.file.processing"):
                文本列表.append([index, Dict文件[index][0]])
        elif mode == "rerank":
            文本列表 = Self.Module.获取相似度最高译文(Dict文件)
            文本列表 = [[index[0], index[1]] for index in 文本列表]
        Self.Module.翻译缓存(文本列表)
        Self.日志("log.core.file.settle.end", info_level=0)
    def DictMini转换数据集(Self, file: str, mode: str = "Alpaca", output_file: str = "dataset.jsonl"):
        Self.日志("log.core.file.settle.start", info_level=0)
        待处理列表 = []
        for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
            with open(file, "rb") as f:
                Dict文件 = json.load(f)
        for index in Self.tqdm(Dict文件, desc="tqdm.file.processing"):
            for index2 in Dict文件[index]:
                待处理列表.append([index, index2])
        导出列表 = []
        for index in Self.tqdm(待处理列表, desc="tqdm.progress.encoding"):
            if mode == "ChatML":
                导出列表.append({"messages": [{"role": "system", "content": f"将下列文本翻译为{Self.Config.LANGUAGE_OUTPUT}语言"}, {"role": "user", "content": index[0]}, {"role": "assistant", "content": index[1]}]})
            elif mode == "Alpaca":
                导出列表.append({"instruction": f"翻译为{Self.Config.LANGUAGE_OUTPUT}语言", "input": index[0], "output": index[1], "system": f"将下列文本翻译为{Self.Config.LANGUAGE_OUTPUT}语言"})
        with open(output_file, 'w+', encoding='utf-8') as f:
            f.write('\n'.join(json.dumps(item, ensure_ascii=False, separators=(',', ':')) for item in Self.tqdm(导出列表, desc="tqdm.progress.encoding")))
        Self.日志("log.core.file.settle.end", info_level=0)
    def 导入DictMini参考词(Self, file: str = None, mode: str = "dense", max_len: int = 80, reversal: bool = False):
        Self.日志("log.core.file.settle.start", info_level=0)
        待处理列表 = []
        过滤集 = set()
        if Path(file).suffix == ".json":
            for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
                with open(file, "rb") as f:
                    Dict文件 = json.load(f)
            if mode == "dense":
                for index in Self.tqdm(Dict文件, desc="tqdm.file.processing"):
                    待处理列表.append([index, ", ".join(list({str(item) for item in Dict文件[index] if len(str(item)) <= max_len})), ""])
            elif mode == "sparse":
                for index in Self.tqdm(Dict文件, desc="tqdm.file.processing"):
                    for index1 in Dict文件[index]:
                        if len(index1) <= max_len:
                            if index1 not in 过滤集:
                                过滤集.add(index1)
                                待处理列表.append([index, index1, ""])
                                if reversal:
                                    待处理列表.append([index1, index, ""])
        elif file == None:
            for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
                待处理列表 = [[k, v, ""] for k, v in Self.Module.翻译缓存()]
        TranslatorPersistence.参考词预处理(Self=Self, texts=待处理列表)
        Self.日志("log.core.file.settle.end", info_level=0)
        
测试 = False
if __name__ == "__main__" and 测试:
    参数 = {
        "EMB_API_URL": "http://127.0.0.1:25564/v1/embeddings",
        "EMB_MODEL": "text-embedding-bge-large-en-v1.5",
        "LANGUAGE": "zh_CN",
        "VEC_FILE_NAME": "Vectors1",
        "EMB_MAX_WORKERS": 2,
        "DEBUG_MODE": True,
        "EMB_MAX_TOKENS": 512
    }
    翻译 = TranslatorTool(参数)
    #翻译.语言文件对转DictMini(r"C:\Users\FengMang\Downloads\Minecraft-Shaders-zh_CN-Lang-Files-Surisen.zip", r"C:\Users\FengMang\Downloads\Dict-Mini.json")
    翻译.导入DictMini参考词(r"C:\Users\FengMang\Downloads\Dict-Mini.json")
    