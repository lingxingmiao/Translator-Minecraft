from TranslatorLib import (zipfile, json, eb, defaultdict, Path, uuid, random, ThreadPoolExecutor, partial, asyncio, datasets, Mods, Loop,
                           TranslatorPersistence, Config, faiss, np, time, gguf)

class Tool:
    def __init__(Self, App: Config):
        Self.Translator = App.get_translator()
        Self.Config = App.Config
        Self.Module = App.Module
        Self.Locale = App.Locale
        Self.Quantization = App.Quantization
        Self.Lang = App.Lang
        Self.File = App.File
        Self.Builder = App.Builder
        Self.日志 = App.日志
        Self.tqdm = App.RichTqdm
        Self.CacheTranslator = App.CacheTranslator
        Self.正则表达式预编译 = Self.Translator.正则表达式预编译
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
                Self.去重DictMini(DictMini)
            else:
                with open(f"{OutputPath}/Dict-Mini.json", "w", encoding="utf-8") as f:
                    json.dump(DictMini附加, f, ensure_ascii=False)
                Self.去重DictMini(f"{OutputPath}/Dict-Mini.json")
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
        合并列表, 输出列表, 缺失列表 = [], [], []
        文件0, 文件0源文件, 文件1, 压缩路径, 输出扩展名, file2 = Self.File.读取资源文件(file0, file1)
        
        保存语言代码 = Self.Config.LANGUAGE_OUTPUT
        模组版本 = Self.File.从模组读取游戏版本(file0)
        if 模组版本: # 检查是否为1.6-1.10 因为这几个版本要用zh_CN en_US
            try:
                版本段 = str(模组版本).split('.')
                主, 次 = (int(版本段[0]), int(版本段[1])) if len(版本段) >= 2 else (int(版本段[0]), 0)
                if (1, 6) <= (主, 次) <= (1, 10):
                    保存语言代码 = Self.Module.语言代码区域转大写(保存语言代码)
            except Exception: pass
        
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
            缺失列表 = [[index[0], index[2], ""] for index in Self.File.读取审查文件(未翻译文件路径)]
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
                Self.File.保存语言文件(f"{Path(index[0]).parent}/{保存语言代码}{输出扩展名}", index[1])
            压缩文件夹Path = Path(压缩路径)
            if file2[0] == False:
                文档内容 = Self.Config.PACK_META_TEMPLATE_MERGE.format(name=Path(file0).stem, lang=保存语言代码, author=Self.Config.PACK_AUTHOR or "海盐青茫")
                with open(压缩文件夹Path/"pack.mcmeta", "w+", encoding="utf-8") as f:
                    f.write(json.dumps({"pack": {"description": 文档内容, "pack_format": 9999,"supported_formats": [0, 9999],"min_format": 0,"max_format": 9999}}, ensure_ascii=False, indent=4))
            with zipfile.ZipFile(f"{output_path}/{Path(file0).stem}-{保存语言代码}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                for 压缩文件 in 压缩文件夹Path.rglob('*'):
                    if 压缩文件.is_file():
                        f.write(压缩文件, arcname=压缩文件.relative_to(压缩文件夹Path))
            Self.日志("log.core.translator.succeed", path=Path(f"{output_path}/{Path(file0).stem}-{保存语言代码}.zip").resolve(), info_level=0)
            返回路径 = Path(f"{output_path}/{Path(file0).stem}-{保存语言代码}.zip").resolve()
        else:
            输出路径 = str(f"{output_path}/{保存语言代码}{输出扩展名}")
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
            文本列表 = Loop.运行(Self.Builder.获取相似度最高译文(Dict文件))
            文本列表 = [[index[0], index[1]] for index in 文本列表]
        Self.CacheTranslator.翻译缓存(文本列表, 语言=Self.Config.LANGUAGE_OUTPUT)
        Self.日志("log.core.file.settle.end", info_level=0)
    def DictMini转换数据集(Self, file: str, mode: str = "Alpaca", max_length: int = 2147483647, output_file: str = "dataset.jsonl"):
        Self.日志("log.core.file.settle.start", info_level=0)
        待处理列表 = []
        for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
            with open(file, "rb") as f:
                Dict文件 = json.load(f)
        for index in Self.tqdm(Dict文件, desc="tqdm.file.processing"):
            for index2 in Dict文件[index]:
                待处理列表.append([index, index2])
        Self.转换数据集(待处理列表[:max_length], mode, output_file)
    def 翻译缓存转换数据集(Self, mode: str = "Alpaca", output_file: str = "dataset.jsonl"):
        Self.日志("log.core.file.settle.start", info_level=0)
        待处理列表 = []
        for key0, value0 in Self.tqdm(Self.Translator.CacheTranslator.翻译缓存().items(), desc="tqdm.file.processing"):
            for key1, value1 in value0.items():
                 待处理列表.append([key1, value1])
        Self.转换数据集(待处理列表, mode, output_file)
    def 转换数据集(Self, 待处理列表, mode, output_file):
        导出列表 = []
        random.shuffle(待处理列表)
        系统提示词 = Self.Config.TRANSLATOR_SYSTEM_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT)
        预编译输入文本 = Self.Config.TRANSLATOR_USER_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT, text="{text}")
        if mode in ["Alpaca-EX", "DPO-Mini", "DPO-EX"]:
            翻译映射 = Loop.运行(Self.Translator.翻译语言列表([[索引[1], 索引[0], ""] for 索引 in 待处理列表], 获取参考文本=True))
        if mode == "Alpaca-EX":
            提示词 = Self.Config.TRANSLATOR_SYSTEM_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT)
            所有翻译对 = []
            for 原文, 译文 in 待处理列表:
                参考列表 = 翻译映射.get(原文, [])
                术语表原始数据 = 参考列表 if isinstance(参考列表, list) else []
                所有翻译对.append({
                    "src": 原文, 
                    "tgt": 译文, 
                    "glossary": 术语表原始数据
                })
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
                if 合并术语表:
                    参考词 = "<rag-input>" + " ".join([f"{t[0]}={t[1]}" for t in 合并术语表]) + "</rag-input>"
                else:
                    参考词 = ""
                if 抽样数量 == 1:
                    输入文本 = f"<rt>{选中的源文本[0]}</rt>"
                    输出文本 = f"<rt>{选中的目标文本[0]}</rt>"
                else:
                    输入文本 = "\n".join([f"<rt>{t}</rt>" for t in 选中的源文本])
                    输出文本 = "\n".join([f"<rt>{t}</rt>" for t in 选中的目标文本])
                输入文本 = (参考词 + "\n" if 参考词 else "") + 预编译输入文本.format(text=输入文本)
                新待处理列表.append([提示词, 输入文本, 输出文本])
            待处理列表 = 新待处理列表
        elif mode == "Alpaca-EX-Mini":
            提示词 = Self.Config.TRANSLATOR_SYSTEM_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT)
            预编译输入文本 = Self.Config.TRANSLATOR_USER_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT, text="{text}")
            所有翻译对 = [{"src": 原文, "tgt": 译文} for 原文, 译文 in 待处理列表]
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
                if 抽样数量 == 1:
                    输入文本 = f"<rt>{选中的源文本[0]}</rt>"
                    输出文本 = f"<rt>{选中的目标文本[0]}</rt>"
                else:
                    输入文本 = "\n".join([f"<rt>{t}</rt>" for t in 选中的源文本])
                    输出文本 = "\n".join([f"<rt>{t}</rt>" for t in 选中的目标文本])
                新待处理列表.append([提示词, 预编译输入文本.format(text=输入文本), 输出文本])
            待处理列表 = 新待处理列表
        elif mode in ["DPO-Mini", "DPO-EX"]:
            所有翻译对 = []
            for 原文, 译文 in 待处理列表:
                参考列表 = 翻译映射.get(原文, [])
                术语表原始数据 = 参考列表 if isinstance(参考列表, list) else []
                所有翻译对.append({"src": 原文, "tgt": 译文, "glossary": 术语表原始数据})
            random.shuffle(所有翻译对)
            参考池 = [[item["src"], item["tgt"]] for item in 所有翻译对]
            if mode == "DPO-Mini":
                for item in Self.tqdm(所有翻译对, desc="tqdm.file.processing"):
                    原文, 译文 = item["src"], item["tgt"]
                    conversations = [
                        {"role": "system", "content": 系统提示词},
                        {"role": "user", "content": Self.Config.TRANSLATOR_USER_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT, text=原文)},
                    ]
                    chosen = {"role": "assistant", "content": 译文}
                    for 污染 in Self.构造DPO污染(原文, 译文, item["glossary"], 系统提示词, 参考池, f"<rt>{原文}</rt>"):
                        if 污染 == 译文: continue
                        导出列表.append({"conversations": conversations, "chosen": chosen, "rejected": {"role": "assistant", "content": 污染}})
                    术语表 = [t for t in (item["glossary"] or []) if isinstance(t, (list, tuple)) and len(t) >= 2]
                    rag块 = None
                    if 术语表:
                        rag块 = "<rag-input>" + " ".join([f"{t[0]}={t[1]}" for t in 术语表]) + "</rag-input>"
                    else:
                        rag候选 = [r for r in (参考池 or []) if r[0] != 原文]
                        if rag候选:
                            rag块 = "<rag-input>" + " ".join([f"{r[0]}={r[1]}" for r in random.sample(rag候选, min(3, len(rag候选)))]) + "</rag-input>"
                    if rag块:
                        带RAG用户 = (rag块 + "\n") + Self.Config.TRANSLATOR_USER_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT, text=f"{原文}")
                        conversationsRAG = [
                            {"role": "system", "content": 系统提示词},
                            {"role": "user", "content": 带RAG用户},
                        ]
                        chosenRAG = {"role": "assistant", "content": 译文}
                        for rag污染 in [rag块 + "\n" + 译文, rag块, f"<rt>{译文}</rt>\n" + rag块]:
                            if rag污染 == 译文:
                                continue
                            导出列表.append({"conversations": conversationsRAG, "chosen": chosenRAG, "rejected": {"role": "assistant", "content": rag污染}})
            else:
                i = 0
                while i < len(所有翻译对):
                    剩余数量 = len(所有翻译对) - i
                    抽样数量 = random.randint(1, min(16, 剩余数量))
                    当前批次 = 所有翻译对[i : i + 抽样数量]
                    i += 抽样数量
                    选中的源文本 = [item["src"] for item in 当前批次]
                    选中的目标文本 = [item["tgt"] for item in 当前批次]
                    输入文本 = "\n".join([f"<rt>{t}</rt>" for t in 选中的源文本])
                    输出文本 = "\n".join([f"<rt>{t}</rt>" for t in 选中的目标文本])
                    合并术语表 = []
                    见过的术语 = set()
                    for item in 当前批次:
                        for 术语 in item["glossary"]:
                            if isinstance(术语, (list, tuple)) and len(术语) >= 2:
                                术语键 = f"{术语[0]} --> {术语[1]}"
                                if 术语键 not in 见过的术语:
                                    见过的术语.add(术语键)
                                    合并术语表.append(术语)
                    conversations = [
                        {"role": "system", "content": 系统提示词},
                        {"role": "user", "content": Self.Config.TRANSLATOR_USER_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT, text=输入文本)},
                    ]
                    chosen = {"role": "assistant", "content": 输出文本}
                    for 污染 in Self.构造DPO污染(输入文本, 输出文本, 合并术语表, 系统提示词, 参考池, 输入文本):
                        if 污染 == 输出文本: continue
                        导出列表.append({"conversations": conversations, "chosen": chosen, "rejected": {"role": "assistant", "content": 污染}})
        for index in Self.tqdm(待处理列表, desc="tqdm.progress.encoding"):
            if mode in ["DPO-Mini", "DPO-EX"]: continue
            if mode == "ChatML":
                导出列表.append({"messages": [{"role": "system", "content": f"将下列文本翻译为{Self.Config.LANGUAGE_OUTPUT}语言"}, {"role": "user", "content": index[0]}, {"role": "assistant", "content": index[1]}]})
            elif mode == "Alpaca":
                导出列表.append({"instruction": f"翻译为{Self.Config.LANGUAGE_OUTPUT}语言", "input": index[0], "output": index[1]})
            elif mode == "Alpaca-Instruction":
                导出列表.append({"instruction": Self.Config.TRANSLATOR_SYSTEM_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT), "input": index[0], "output": index[1]})
            elif mode in ["Alpaca-EX", "Alpaca-EX-Mini"]:
                导出列表.append({"instruction": index[0], "input": index[1], "output": index[2]})
        with open(output_file, 'w+', encoding='utf-8') as f:
            f.write('\n'.join(json.dumps(item, ensure_ascii=False, separators=(',', ':')) for item in Self.tqdm(导出列表, desc="tqdm.progress.encoding")))
        Self.日志("log.core.file.settle.end", info_level=0)
    def 构造DPO污染(Self, 原文: str, 译文: str, 参考列表, 系统提示词: str, 参考池, 输入文本: str = ""):
        污染列表 = []
        rag项 = [f"{t[0]}={t[1]}" for t in (参考列表 or []) if isinstance(t, (list, tuple)) and len(t) >= 2]
        if rag项:
            污染列表.append("<rag-input>" + " ".join(rag项) + "</rag-input>\n" + 译文)
        候选 = [r for r in (参考池 or []) if r[0] != 原文]
        if 候选:
            上下文内容 = " ".join([f"{r[0]}={r[1]}" for r in random.sample(候选, min(3, len(候选)))])
            污染列表.append(f"<context>{上下文内容}</context>\n" + 译文)
        if 输入文本:
            污染列表.append(输入文本 + "\n" + 译文)
        污染列表.append(系统提示词 + "\n" + 译文)
        污染列表.append(原文)
        污染列表.append(f"这是译文:{译文}")
        return [p for p in 污染列表 if p and p != 译文]
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
                待处理列表 = [[k, v, ""] for k, v in Self.CacheTranslator.翻译缓存(语言=Self.Config.LANGUAGE_OUTPUT).items()]
        Loop.运行(TranslatorPersistence.参考词预处理(Self=Self, texts=待处理列表, use_cache=False, 查询=False))
        Self.日志("log.core.file.settle.end", info_level=0)
    def 去重DictMini(Self, file: str):
        Self.日志("log.core.file.settle.start", info_level=0)
        with open(file, "r", encoding="utf-8") as f:
            数据 = json.load(f)
        去重前 = sum(len(v) for v in 数据.values())
        for 键 in 数据:
            数据[键] = list(dict.fromkeys(数据[键]))
        去重后 = sum(len(v) for v in 数据.values())
        with open(file, "w", encoding="utf-8") as f:
            json.dump(数据, f, ensure_ascii=False)
        Self.日志("log.core.file.settle.end", info_level=0)
        Self.日志("log.core.dictmini.dedup", before=去重前, after=去重后, info_level=0)
        
    def 翻译数据集(Self, path: str, field: str, input_suffix: str=".parquet", instruction: str=None, output_path: str=None):
        输入路径 = Path(path)
        处理路径 = []
        if instruction is None: instruction = f"翻译为{Self.Config.LANGUAGE_OUTPUT}语言"
        if 输入路径.is_dir():
            处理路径 = [p for p in 输入路径.rglob('*') if p.suffix == input_suffix]
        elif 输入路径.is_file():
            处理路径 = [输入路径]
        else:
            Self.日志("log.module.datasets.un.path.err", info_level=3, path=输入路径)
            raise FileNotFoundError(Self.Lang("log.module.datasets.not.path.err", path=输入路径))
        for 文件路径 in Self.tqdm(处理路径, desc="tqdm.translator.datasets"):
            数据集 = []
            if 文件路径.suffix == ".parquet":
                数据集 = list(datasets.load_dataset("parquet", 文件路径, split="train"))
            elif 文件路径.suffix == ".jsonl":
                with open(文件路径, "r", encoding="utf-8") as f:
                    for 行 in f:
                        行 = 行.strip()
                        if 行:
                            数据集.append(json.loads(行))
            elif 文件路径.suffix == ".json":
                with open(文件路径, "r", encoding="utf-8") as f:
                    数据 = json.load(f)
                数据集 = 数据 if isinstance(数据, list) else (list(数据.values()) if isinstance(数据, dict) else [数据])
            else:
                Self.日志("log.module.datasets.un.suffix.err", file=文件路径, info_level=2)
                continue
            输入列表 = [[str(idx), i[field], ""] for idx, i in enumerate(数据集)]
            输出列表 = Loop.运行(Self.Translator.翻译语言列表(输入列表))
            翻译映射 = {键: 译文 for 键, 译文, _ in 输出列表}
            导出列表 = []
            for idx, 行 in enumerate(数据集):
                原文 = 行[field]
                译文 = 翻译映射.get(str(idx), 原文)
                if not 译文:
                    译文 = 原文
                导出列表.append({"instruction": instruction, "input": 原文, "output": 译文})
            保存路径 = output_path if output_path else str(Path(文件路径).parent / f"{Path(文件路径).stem}_SFT.parquet")
            导出数据集 = datasets.Dataset.from_list(导出列表)
            导出数据集.to_parquet(保存路径)
            Self.日志("log.core.translator.succeed", path=Path(保存路径).resolve(), info_level=0)

    def 剪枝向量库(Self, 向量库: str, 缓存库: str, k: int, 输出路径: str = "", 量化: str = None, 批量: int = 4096, 设备: str = "cpu"):
        # 向量库: 要被剪枝的库(如 Vectors/Vector.gguf)   缓存库: 实际命中过的向量(如 VectorCaches/VectorCache.gguf)
        # k: 每条缓存向量在向量库里取前 k 个候选(全库精确检索), 全部候选的并集就是"被激活过"的向量
        # 输出路径: 目录(自动命名) 或 .gguf 文件           量化: None=沿用源文件张量类型  ""=原始张量  其它=Q8_0/F16
        # 设备: faiss 检索设备 cpu/gpu                    返回: 输出文件路径
        开始时间 = time.time()
        Self.日志("log.core.vector.prune.start", file=向量库, probe=缓存库, k=k, info_level=0)
        try:
            if k < 1: raise ValueError(Self.Lang("log.core.vector.prune.k.err", k=k))
            向量库, 缓存库 = Path(向量库).resolve(), Path(缓存库).resolve()
            输出路径 = Path(输出路径) if 输出路径 else Path(f"./{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}")
            if 输出路径.suffix.lower() != ".gguf": 输出路径 = 输出路径 / f"{向量库.stem}.pruned.gguf"
            输出路径 = 输出路径.resolve() # ↓ 必须先解析成绝对路径再比对, 否则 Vectors/Vector.gguf 和 C:\...\Vector.gguf 判不相等, 会覆盖掉输入
            输出路径.parent.mkdir(parents=True, exist_ok=True)
            if 输出路径 in (向量库, 缓存库): raise ValueError(Self.Lang("log.core.vector.prune.same.err", path=输出路径))
            
            def 取矩阵(向量文件): # ↓ 向量库 {"Vector":(N,D)} / 缓存库 {"vec":(N,D)} / 其它取第一个二维张量
                if isinstance(向量文件, dict):
                    for 键 in ("vec", "Vector"):
                        值 = 向量文件.get(键)
                        if isinstance(值, np.ndarray) and 值.ndim == 2: return 值
                    for 值 in 向量文件.values():
                        if isinstance(值, np.ndarray) and 值.ndim == 2: return 值
                if isinstance(向量文件, np.ndarray) and 向量文件.ndim == 2: return 向量文件
                raise ValueError(Self.Lang("log.core.vector.prune.matrix.err", file=向量库))
            
            def 取文本(快照): # ↓ 向量库 [[原文, 译文], ...] / 缓存库 {"texts": [...]}
                if isinstance(快照, dict): return list(快照.get("texts", []) or [])
                if isinstance(快照, (list, tuple)): return list(快照)
                return []
            
            def 读库(路径):
                向量文件, 快照 = TranslatorPersistence.向量GGUF读取(str(路径))
                if not 向量文件: raise ValueError(Self.Lang("log.core.vector.prune.matrix.err", file=路径))
                矩阵 = 取矩阵(向量文件)
                if 矩阵.dtype != np.float32: 矩阵 = 矩阵.astype(np.float32)
                return np.ascontiguousarray(矩阵), 取文本(快照), 向量文件
            
            def 检索(库矩阵, 探针):
                faiss.normalize_L2(库矩阵) # ↓ 原地归一化, 内积即余弦
                索引器 = faiss.IndexFlatIP(库矩阵.shape[1]) # ↓ 全库暴力检索, 不近似
                if 设备 == "gpu": 索引器 = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, 索引器)
                索引器.add(库矩阵)
                命中, 首名 = set(), np.empty(len(探针), dtype=np.float32)
                for 起 in Self.tqdm(range(0, len(探针), 批量), desc="tqdm.vector.prune"):
                    块 = np.ascontiguousarray(探针[起:起 + 批量], dtype=np.float32)
                    faiss.normalize_L2(块)
                    相似度, 编号 = 索引器.search(块, k)
                    首名[起:起 + len(块)] = 相似度[:, 0]
                    命中.update(int(号) for 号 in 编号.ravel() if 号 >= 0)
                return 命中, 首名
            
            def 检索Numpy(库矩阵, 探针): # ↓ 没有 faiss 时的兜底, 口径与 IP 一致
                块大小 = max(1, min(批量, 256)) # ↓ 相似度矩阵是 (批, 全库) float32, 别一次铺满内存
                库矩阵 /= np.maximum(np.linalg.norm(库矩阵, axis=1, keepdims=True), 1e-12)
                探针 = 探针 / np.maximum(np.linalg.norm(探针, axis=1, keepdims=True), 1e-12)
                命中, 首名 = set(), np.empty(len(探针), dtype=np.float32)
                for 起 in Self.tqdm(range(0, len(探针), 块大小), desc="tqdm.vector.prune"):
                    相似度 = 探针[起:起 + 块大小] @ 库矩阵.T
                    前k = np.argpartition(-相似度, k - 1, axis=1)[:, :k]
                    首名[起:起 + len(相似度)] = 相似度[np.arange(len(相似度)), 前k[:, 0]]
                    命中.update(int(号) for 号 in 前k.ravel())
                return 命中, 首名
            
            库矩阵, 库文本, 库字典 = 读库(向量库)
            探针矩阵, _, _ = 读库(缓存库)
            if 探针矩阵.shape[1] != 库矩阵.shape[1]:
                raise ValueError(Self.Lang("log.core.vector.prune.dim.err", lib=库矩阵.shape[1], probe=探针矩阵.shape[1]))
            if len(探针矩阵) == 0: raise ValueError(Self.Lang("log.core.vector.prune.empty.err"))
            k = min(k, len(库矩阵))
            
            命中集合, 检索器 = None, None
            if 设备 == "gpu":
                try: 命中集合, 首名 = 检索(库矩阵, 探针矩阵); 检索器 = "faiss-gpu"
                except Exception: 设备 = "cpu"
            if 命中集合 is None:
                try: 命中集合, 首名 = 检索(库矩阵, 探针矩阵); 检索器 = "faiss-cpu"
                except Exception: 命中集合, 首名 = 检索Numpy(库矩阵, 探针矩阵); 检索器 = "numpy"
            保留索引 = np.array(sorted(命中集合), dtype=np.int64)
            子矩阵 = np.ascontiguousarray(库矩阵[保留索引])
            子文本 = [库文本[i] for i in 保留索引] if 库文本 else []
            del 库矩阵, 探针矩阵 # ↓ 全库矩阵先放掉, 给重编码/量化腾内存
            Self.日志("log.core.vector.prune.hit", keep=len(保留索引), total=len(库文本),
                      rate=f"{len(保留索引) / max(len(库文本), 1):.2%}", info_level=0)
            
            目标量化 = 量化
            if 目标量化 is None: # ↓ 默认沿用源文件张量类型; F32/F16/BF16 走原始张量
                try:
                    读取器 = gguf.GGUFReader(str(向量库))
                    try: 目标量化 = next((张量项.tensor_type.name for 张量项 in 读取器.tensors if 张量项.name != "tv.text"), "F32")
                    finally: del 读取器
                except Exception: 目标量化 = "F32"
                if 目标量化 in ("F32", "F16", "BF16"): 目标量化 = ""
            输出字典 = Self.Quantization.编码向量(子矩阵)
            if isinstance(输出字典, dict): # ↓ PCA 投影/均值是全局参数, 行切片后依然有效, 原样带上
                for 键, 值 in (库字典.items() if isinstance(库字典, dict) else []):
                    if 键 not in 输出字典 and str(键).startswith("PCA"): 输出字典[键] = 值
            临时路径 = Path(str(输出路径) + ".tmp")
            TranslatorPersistence.向量GGUF写入(str(临时路径), 输出字典, 子文本 if 子文本 else None, 目标量化, Self.Lang)
            临时路径.replace(输出路径)
            Self.日志("log.core.vector.prune.end", path=输出路径, size=f"{输出路径.stat().st_size / 1024 / 1024:.1f}",
                      force=目标量化 or "原始", engine=检索器, elapsed=f"{time.time() - 开始时间:.1f}", info_level=0)
            return 输出路径
        except Exception:
            Self.日志("log.core.vector.prune.error", e=eb.format_exc(), info_level=3)
            raise