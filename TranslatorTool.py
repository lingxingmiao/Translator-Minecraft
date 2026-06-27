from TranslatorLib import (zipfile, json, eb, defaultdict, Path, uuid, random, ThreadPoolExecutor, partial,
                           Translator, TranslatorPersistence)

class TranslatorTool:
    def __init__(Self, Config: dict = None):
        Self.Translator = Translator(Config=Config or {})
        Self.Config = Self.Translator.Config
        Self.Module = Self.Translator.Module
        Self.Locale = Self.Translator.Locale
        Self.Quantization = Self.Translator.Quantization
        Self.Lang = Self.Translator.Lang
        Self.File = Self.Translator.File
        Self.Builder = Self.Translator.Builder
        Self.日志 = Self.Translator.日志
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
                    for ext in [".json", ".lang", ".local"]:
                        输入语言文件名 = f"{输入语言文件名称}{ext}"
                        输出语言文件名 = f"{输出语言文件名称}{ext}"
                        if 输入语言文件名 in 真实文件名映射 and 输出语言文件名 in 真实文件名映射:
                            符合条件的文件夹.append((子目录, 真实文件名映射[输入语言文件名], 真实文件名映射[输出语言文件名]))
                            break
            符合条件的文件夹 = list(dict.fromkeys(符合条件的文件夹))
            DictMini附加 = defaultdict(list)
            for 子目录, 输入路径, 输出路径 in 符合条件的文件夹:
                输入文件全路径 = str(子目录 / 输入路径)
                输出文件全路径 = str(子目录 / 输出路径)
                文件0, _, 文件1, _, _, _ = Self.File.读取资源文件(输入文件全路径, 输出文件全路径, read_error=False)
                语言文件B = {index[0]: index[1] for index in 文件1}
                for index in 文件0:
                    try:
                        键, 值 = index[0], index[1]
                        if 键 in 语言文件B:
                            DictMini附加[值].append(语言文件B[键])
                    except Exception:
                        pass
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
        文件0, _, 文件1, _, 输出扩展名, _ = Self.File.读取资源文件(file0, file1)
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
        Self.File.保存语言文件(导出路径, [f"{index[0]}={index[1]}" for index in 缺失列表])
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
        文件0, 文件0源文件, 文件1, 压缩路径, 输出扩展名, file2 = Self.File.读取资源文件(file0, file1)
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
            缺失列表 = [[index[0], index[1], ""] for index in Self.File.读取审查文件(未翻译文件路径)]
        else:
            缺失列表 = Self.File.读取语言文件(notlang_file)[0]
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
                Self.File.保存语言文件(f"{Path(index[0]).parent}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}", index[1])
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
            Self.File.保存语言文件(输出路径, 输出列表)
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
            文本列表 = Self.Builder.获取相似度最高译文(Dict文件)
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
        random.shuffle(待处理列表)
        if mode == "Alpaca-EX":
            文本列表, 参考列表 = Self.Translator.翻译语言列表([[索引[1], 索引[0], ""] for 索引 in 待处理列表], 获取参考文本=True)
            提示词 = Self.Translator.Config.TRANSLATOR_SYSTEM_PROMPT[1].format(LANGUAGE_OUTPUT=Self.Translator.Config.LANGUAGE_OUTPUT)
            所有翻译对 = []
            for 源文本, 目标参考 in zip(文本列表, 参考列表):
                src = 源文本 if isinstance(源文本, str) else str(源文本)
                tgt = 目标参考[0] if isinstance(目标参考[0], str) else str(目标参考[0])
                # 安全获取术语表
                术语表原始数据 = 目标参考[1][1] if len(目标参考) > 1 and isinstance(目标参考[1][1], list) else []
                所有翻译对.append({"src": src, "tgt": tgt, "glossary": 术语表原始数据})
            random.shuffle(所有翻译对)
            新待处理列表 = []
            i = 0
            while i < len(所有翻译对):
                剩余数量 = len(所有翻译对) - i
                抽样数量 = random.randint(1, min(16, 剩余数量))
                当前批次 = 所有翻译对[i : i + 抽样数量]
                i += 抽样数量
                选中的源文本 = [item["src"] for item in 当前批次]
                选中的目标文本 = [item["tgt"] for item in 当前批次]
                合并术语表 = []
                见过的术语 = set()
                for item in 当前批次:
                    for 术语 in item["glossary"]:
                        if isinstance(术语, (list, tuple)) and len(术语) >= 2:
                            术语键 = f"{术语[0]} --> {术语[1]}"
                            if 术语键 not in 见过的术语:
                                见过的术语.add(术语键)
                                合并术语表.append(术语)
                            
                术语表文本 = "\n".join([f"{术语[0]} --> {术语[1]}" for 术语 in 合并术语表])
                if 抽样数量 == 1:
                    源文本字符串 = 选中的源文本[0]
                    目标文本字符串 = 选中的目标文本[0]
                else:
                    源文本字符串 = json.dumps(选中的源文本, ensure_ascii=False)
                    目标文本字符串 = json.dumps(选中的目标文本, ensure_ascii=False)
                if 术语表文本:
                    输入字符串 = f"{源文本字符串}\n\n术语参考：\n{术语表文本}"
                else:
                    输入字符串 = 源文本字符串
                新待处理列表.append([提示词, 输入字符串, 目标文本字符串])
            待处理列表 = 新待处理列表
        for index in Self.tqdm(待处理列表, desc="tqdm.progress.encoding"):
            if mode == "ChatML":
                导出列表.append({"messages": [{"role": "system", "content": f"将下列文本翻译为{Self.Config.LANGUAGE_OUTPUT}语言"}, {"role": "user", "content": index[0]}, {"role": "assistant", "content": index[1]}]})
            elif mode == "Alpaca":
                导出列表.append({"instruction": f"翻译为{Self.Config.LANGUAGE_OUTPUT}语言", "input": index[0], "output": index[1]})
            elif mode == "Alpaca-EX":
                导出列表.append({"instruction": index[0], "input": index[1], "output": index[2]})
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
        
    def 翻译流程转DictMini(Self, path1, path2, DictMini, 文件匹配, 读取方法, 过滤方法, 读取并发, 日志类型, OutputPath = r"./"):
        文件列表, 翻译列表, 参考列表, 参考文件列表 = [], [], [], []
        参考字典 = {}
        翻译参考列表 = defaultdict(list)
        path1 = Path(path1)
        path2 = Path(path2)
        if isinstance(文件匹配, str):
            文件匹配 = [文件匹配]
        for index in 文件匹配:
            文件列表.extend([p for p in path1.rglob(index)] if Path(path1).is_dir() else [path1])
            参考文件列表.extend([p for p in path2.rglob(index)] if Path(path2).is_dir() else [path2])
        Self.日志(f"log.core.file.{日志类型}.read.start", info_level=0)
        with ThreadPoolExecutor(max_workers=读取并发) as 执行器:
            结果集 = 执行器.map(读取方法, 文件列表)
            for 结果 in Self.tqdm(结果集, total=len(文件列表), desc="tqdm.file.read"):
                翻译列表.extend(结果)
        with ThreadPoolExecutor(max_workers=读取并发) as 执行器:
            结果集 = 执行器.map(读取方法, 参考文件列表)
            for 结果 in Self.tqdm(结果集, total=len(参考文件列表), desc="tqdm.file.read"):
                参考列表.extend(结果)
        Self.日志(f"log.core.file.{日志类型}.read.end", info_level=0)
        过滤后 = []
        try:
            for 条目 in 翻译列表:
                if 过滤方法(条目):
                    过滤后.append(条目)
        except Exception:
            Self.日志(f"log.module.{日志类型}.clean.error", index=条目, e=eb.format_exc(), info_level=2)
        if 参考列表 != None:
            for item in 参考列表:
                try:
                    参考字典[str(item[0])] = item[1]
                except Exception:
                    Self.日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
            for index in 翻译列表:
                key = str(index[0])
                if key in 参考字典:
                    翻译参考列表[index[1]].append(参考字典[key])
        if DictMini:
            with open(DictMini, "r", encoding="utf-8") as f:
                Dict文件 = json.load(f)
            for 原文, 值列表 in 翻译参考列表.items():
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
                json.dump(翻译参考列表, f, ensure_ascii=False)
                
    def 导入FTB任务DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        Self.翻译流程转DictMini(path, path2, DictMini, "*.snbt", Self.File.读取单个FTBQ_Snbt文件, Self.Module.过滤键文本, Self.Config.QUESTS_READ_MAX_CONCURRENT, "quests", OutputPath)
    def 导入BQ任务DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        Self.翻译流程转DictMini(path, path2, DictMini, "*.json", Self.File.读取单个BQ_Json文件, Self.Module.过滤键文本, Self.Config.QUESTS_READ_MAX_CONCURRENT, "quests", OutputPath)
    def 导入HQM任务DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        Self.翻译流程转DictMini(path, path2, DictMini, ["*.hqm", "*.json"], partial(Self.File.读取单个HQM文件, mode="L" if any(Path(path).rglob("*.hqm")) else "H"), Self.Module.过滤键文本, Self.Config.QUESTS_READ_MAX_CONCURRENT, "quests", OutputPath)
    def 导入ZS脚本DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        Self.翻译流程转DictMini(path, path2, DictMini, "*.zs", Self.File.读取单个ZS文件, lambda 条目: 条目[1] and not Self.正则表达式预编译.翻译剔除方法.match(条目[1]), Self.Config.SCRIPT_READ_MAX_CONCURRENT, "script", OutputPath)
    def 导入CMM菜单DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        Self.翻译流程转DictMini(path, path2, DictMini, "*.json", Self.File.读取单个CMM文件, Self.Module.过滤键文本, Self.Config.MENU_READ_MAX_CONCURRENT, "menu", OutputPath)
    def 导入FM菜单DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        if Path(f"{path}/customization").is_dir(): 
            Self.翻译流程转DictMini(f"{path}/customization", path2, DictMini, "*.txt", Self.File.读取单个FM文件, Self.Module.过滤键文本, Self.Config.MENU_READ_MAX_CONCURRENT, "menu", OutputPath)
        if Path(f"{path}/locals").is_dir():
            Self.语言文件对转DictMini(f"{path}/locals/", DictMini, OutputPath)
    def 导入帕秋莉手册DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        Self.翻译流程转DictMini(path, path2, DictMini, "*.json", Self.File.读取单个帕秋莉手册文件, Self.Module.过滤键文本, Self.Config.BOOK_READ_MAX_CONCURRENT, "book", OutputPath)
    def 导入数据包DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        path = Path(path)
        if path.is_file():
            缓存文件夹 = Path(f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
            with zipfile.ZipFile(path, 'r') as zf: zf.extractall(缓存文件夹)
        else: 
            缓存文件夹 = path
        Self.翻译流程转DictMini(缓存文件夹, path2, DictMini, ["*.json", "*.mcmeta", "*.mcfunction"], Self.File.读取单个数据包文件, Self.Module.过滤键文本, Self.Config.DATA_READ_MAX_CONCURRENT, "data", OutputPath)
    def 导入未知伤亡语言文件DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        Self.翻译流程转DictMini(path, path2, DictMini, ["*.json"], Self.File.读取未知伤亡语言文件, Self.Module.过滤键文本, Self.Config.LANG_READ_MAX_CONCURRENT, "lang", OutputPath)
    def 导入未知伤亡dll模组DictMini(Self, path, path2, DictMini, OutputPath = r"./"):
        Self.翻译流程转DictMini(path, path2, DictMini, "*.dll", Self.File.读取单个DLL文件, Self.Module.过滤DLL文本, Self.Config.DLL_READ_MAX_CONCURRENT, "dll", OutputPath)
                    
        
测试 = True
if __name__ == "__main__" and 测试:
    参数 = {
        "EMB_API_URL": "http://127.0.0.1:25564/v1/embeddings",
        "EMB_MODEL": "text-embedding-bge-large-en-v1.5",
        "LANGUAGE": "zh_CN",
        "VEC_FILE_NAME": "Vectors2",
        "TRANSLATOR_CACHE_READ": False,
        "VEC_QUANTIZATION": "Float32",
        "EMB_MAX_WORKERS": 2,
        "DEBUG_MODE": True,
        "EMB_MAX_TOKENS": 512
    }
    翻译 = TranslatorTool(参数)
    #翻译.语言文件对转DictMini(r"C:\Users\FengMang\Downloads\Minecraft-Shaders-zh_CN-Lang-Files-Surisen.zip", r"C:\Users\FengMang\Downloads\Dict-Mini.json")
    #翻译.导入未知伤亡语言文件DictMini(r"C:\Users\FengMang\Downloads\EN.json") #byd保留所有权利不敢用
    #翻译.导入DictMini参考词(r"C:\Users\FengMang\Downloads\Dict-Mini.json")
    翻译.DictMini转换数据集(r"C:\Users\FengMang\Downloads\Dict-Mini.json", mode="Alpaca-EX")
