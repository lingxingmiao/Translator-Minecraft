from TranslatorLib import HARDWARE_INFO, np, mp, threading, zipfile, pickle, json, ast, os, eb, re, partial, defaultdict, Path, ThreadPoolExecutor, as_completed, Callable, Dict, Any, faiss, requests, GPU_ACC, time, uuid
from TranslatorConfig import RuntimeConfig
from TranslatorQuantization import Quantization
from TranslatorLocale import Locale
from TranslatorModule import Module
import TranslatorPersistence

class Translator:
    def __init__(Self, Config: dict = None):
        global tqdm
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Path(Self.Config.LOGS_FILE_PATH).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.VEC_FILE_PATH).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.PATH_CACHE).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.TRANSLATOR_CACHE_PATH).mkdir(parents=True, exist_ok=True)
        Self.Module = Module(Config=Config)
        Self.Locale = Locale(Config=Config)
        Self.Lang = Self.Locale.Lang
        tqdm = Self.Locale.Tqdm
        Self.Quantization = Quantization(Config=Config)
        Self.Module.写入日志("log.core.numpy", type=HARDWARE_INFO['type'], version=HARDWARE_INFO['version'], error=HARDWARE_INFO['error'],info_level=0)
        Self.上下文 = []
        Self.线程锁 = threading.Lock()
        Self.函数库: Dict[str, Callable] = {}
        Self.扫描模块函数(Self.Module)
    def 扫描模块函数(Self, 模块):
        for 属性名 in dir(模块):
            if 属性名.startswith('_'):
                continue
            属性对象 = getattr(模块, 属性名)
            if callable(属性对象):
                Self.函数库[属性名] = 属性对象
    def 调用额外函数(Self, 函数名: str, *参数, **关键字参数) -> Any:
        函数 = Self.函数库[函数名]
        return 函数(*参数, **关键字参数)
    def 文本生成向量(Self, text: list, outputs: list = None) -> np.float32:
        重试次数 = 0
        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            with Self.线程锁:
                try:
                    向量列表 = Self.嵌入模型.encode(text)
                    向量列表 = np.array(向量列表).astype(np.float32)
                    return [向量列表, [text, outputs]]
                except Exception:
                    Self.Module.写入日志("log.core.locally.generate.vectors.error", e=eb.format_exc(), info_level=2)
                    return [None, [text, outputs]]
        else:
            while 重试次数 < Self.Config.EMB_MAX_RETRY:
                try:
                    请求结果 = requests.post(
                        url=Self.Config.EMB_API_URL,
                        headers={"Content-Type": "application/json","Authorization": f"Bearer {Self.Config.EMB_API_KEY}"},
                        json={"input": text,"model": Self.Config.EMB_MODEL},
                    )
                    请求结果.raise_for_status()
                    请求结果 = 请求结果.json()
                    向量列表 = []
                    for index in range(len(text)):
                        向量列表.append(请求结果['data'][index]['embedding'])
                    向量列表 = np.array(向量列表).astype(np.float32)
                    return [向量列表, [text, outputs]]
                except Exception:
                    重试次数 += 1
                    if 重试次数 >= Self.Config.EMB_MAX_RETRY:
                        Self.Module.写入日志("log.core.api.generate.vectors.error", e=eb.format_exc(), info_level=3)
                        return [None, [text, outputs]]
                    else:
                        Self.Module.写入日志("log.core.api.generate.vectors.retry", e=eb.format_exc(), info_level=2)
                        time.sleep(Self.Config.EMB_RETRY_INTERVAL)
    def 并行生成向量(Self, texts: list,) -> list:
        Self.Module.写入日志("log.core.vector.generate.start", info_level=0)
        if (not Self.嵌入模型) and (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            Self.嵌入模型 = TranslatorPersistence.获取嵌入模型(Self=Self)
        if texts:
            最大字符数 = Self.Config.EMB_MAX_TOKENS * Self.Config.EMB_TOKENSTOTEXT_RATIO
            分组结果 = []
            当前组 = []
            当前总长 = 0.0
            for index in texts:
                首字符串 = index[0]
                长度 = len(首字符串)
                if 当前总长 + 长度 > 最大字符数:
                    分组结果.append(当前组)
                    当前组 = []
                    当前总长 = 0.0
                当前组.append(index)
                当前总长 += 长度
            if 当前组:
                分组结果.append(当前组)
            待处理文本列表原文 = [[item[0] for item in group] for group in 分组结果]
            待处理文本列表额外输出 = [[item[1] for item in group] for group in 分组结果]
            返回内容向量 = []
            with ThreadPoolExecutor(max_workers=Self.Config.EMB_MAX_WORKERS) as 执行器:
                未来任务映射 = {
                    执行器.submit(
                        Self.文本生成向量,
                        text=原文组,
                        outputs=额外输出
                    ): 原文组
                    for 原文组, 额外输出 in zip(待处理文本列表原文, 待处理文本列表额外输出)
                }
                for 单个任务 in tqdm(as_completed(未来任务映射), total=len(未来任务映射), desc="tqdm.vectors.generate"):
                    返回内容向量.append(单个任务.result())
        if 返回内容向量:
            合并向量 = []
            合并请求文本 = []
            合并额外返回 = []
            for 结果 in 返回内容向量:
                if isinstance(结果[0], np.ndarray):
                    合并向量.extend(结果[0])
                    合并请求文本.extend(结果[1][0])
                    合并额外返回.extend(结果[1][1])
            Self.Module.写入日志("log.core.vector.generate.end", info_level=0)
            return [np.array(合并向量).astype(np.float32), [合并请求文本, 合并额外返回]]
        Self.Module.写入日志("log.core.generated.vector.nan", texts=texts, info_level=3)

    def 参考词预处理(Self, texts: list = None,) -> tuple[np.ndarray, list]:
        return TranslatorPersistence.参考词预处理(Self=Self, texts=texts)
    def 生成翻译(Self, texts: list, other_input: str):
        额外内容 = str([index[1] for index in other_input])
        messages = []
        if Self.Config.LLM_CONTEXTS:
            with Self.线程锁:
                if Self.上下文:
                    messages.extend(Self.上下文[-Self.Config.LLM_CONTEXTS_LENGTH*2:])
        请求文本长度 = len(texts)
        括号分离方式 = re.compile(r'^(?:[&§][0-9a-fk-or])*\s*\{([^}]+)\}(.*)', re.DOTALL)
        请求文本 = []
        分离文本 = []
        for index in texts:
            括号分离结果 = 括号分离方式.match(index)
            if 括号分离结果:
                分离文本.append(括号分离结果.group(2))
                请求文本.append(括号分离结果.group(1))
            else:
                分离文本.append("")
                请求文本.append(index)
        请求文本 = 请求文本[0] if 请求文本长度 == 1 else str(请求文本)
        if Self.Config.LLM_PROMPT_LOCATION == "user":
            messages.insert(0, {"role": "system", "content": (Self.Config.LLM_SYSTEM_PROMPT + "" if 请求文本长度 == 1 else Self.Config.LLM_SYSTEM_PROMPTEX2).replace('\n', '')})
            messages.append({"role": "user", "content": 请求文本 + Self.Config.LLM_SYSTEM_PROMPTEX1 + 额外内容})
        elif Self.Config.LLM_PROMPT_LOCATION == "system":
            messages.insert(0, {"role": "system", "content": (Self.Config.LLM_SYSTEM_PROMPT + "" if 请求文本长度 == 1 else Self.Config.LLM_SYSTEM_PROMPTEX2).replace('\n', '') + Self.Config.LLM_SYSTEM_PROMPTEX1 + 额外内容})
            messages.append({"role": "user", "content": 请求文本})
        json = {
            "model": Self.Config.LLM_MODEL,
            "messages": messages,
            "top_p": Self.Config.LLM_TOP_P,
            "top_k": Self.Config.LLM_TOP_K,
            "temperature": Self.Config.LLM_TEMP,
            "stream": False,
        } | Self.Config.LLM_API_KWARGS
        请求次数 = 0
        while 请求次数 < Self.Config.LLM_MAX_RETRY:
            try:
                请求结果 = requests.post(
                    url=Self.Config.LLM_API_URL,
                    headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {Self.Config.LLM_API_KEY}"
                    },
                    json=json
                )
                请求结果.raise_for_status()
                请求结果 = 请求结果.json()
                请求结果 = 请求结果["choices"][0]["message"]["content"]
                Self.Module.写入日志("log.core.debug.request.outputs", messages=texts, item=请求结果, promptex=额外内容, info_level=4)
                添加上下文结果 = 请求结果
                请求结果 = re.sub(r'<think>.*?</think>\s*', '', 请求结果, flags=re.DOTALL)
                返回的请求结果 = texts.copy()
                返回的请求结果 = [请求结果] if 请求文本长度 == 1 else ast.literal_eval(请求结果)
                处理后的请求结果 = []
                for index in range(len(texts)):
                    处理后的请求结果.append(f"{分离文本[index]} {返回的请求结果[index]}")
                返回结果 = []
                for index in range(len(texts)):
                    返回结果.append([other_input[index][0], texts[index], 返回的请求结果[index]])
                if Self.Config.LLM_CONTEXTS:
                    with Self.线程锁:
                        if Self.Config.LLM_PROMPT_LOCATION == "user":
                            Self.上下文.append({"role": "user", "content": 处理后的请求结果 + 额外内容})
                        elif Self.Config.LLM_PROMPT_LOCATION == "system":
                            Self.上下文.append({"role": "user", "content": 处理后的请求结果 })
                        Self.上下文.append({"role": "assistant", "content": 添加上下文结果})
                return 返回结果
            except Exception:
                Self.Module.写入日志("log.core.generate.translator.messages.error", promptex=额外内容, messages=texts, info_level=1)
                返回结果 = [[other_input[index][0], texts[index], texts[index]] for index in range(len(texts))]
                请求次数 += 1
                if 请求次数 >= Self.Config.LLM_MAX_RETRY:
                    Self.Module.写入日志("log.core.generate.translator.error", e=eb.format_exc(), output=请求结果, info_level=2)
                    return 返回结果
                else:
                    Self.Module.写入日志("log.core.generate.translator.retry", e=eb.format_exc(), output=请求结果, info_level=1)
                    time.sleep(Self.Config.LLM_RETRY_INTERVAL)
    def 构建索引(Self, 向量文件):
        Self.Module.写入日志("log.core.index.generate.start", info_level=0)
        向量文件 = 向量文件.get() if GPU_ACC else 向量文件
        if Self.Config.INDEX_SQ == "Q4":
            量化类型 = faiss.ScalarQuantizer.QT_4bit
        elif Self.Config.INDEX_SQ == "Q6":
            量化类型 = faiss.ScalarQuantizer.QT_6bit
        elif Self.Config.INDEX_SQ == "Q8":
            量化类型 = faiss.ScalarQuantizer.QT_8bit
        elif Self.Config.INDEX_SQ == "F16":
            量化类型 = faiss.ScalarQuantizer.QT_fp16
        elif Self.Config.INDEX_SQ == "BF16":
            量化类型 = faiss.ScalarQuantizer.QT_bf16
        if Self.Config.INDEX_MODE == "HNSWSQ":
            向量索引 = faiss.IndexHNSWSQ(向量文件.shape[1], 量化类型, Self.Config.INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH
            for _ in tqdm(range(1), desc="tqdm.index.train"):
                向量索引.train(向量文件)
            for _ in tqdm(range(1), desc="tqdm.index.build"):
                向量索引.add(向量文件)
        elif Self.Config.INDEX_MODE == "RefineFlat":
            向量索引 = faiss.IndexRefineFlat(faiss.IndexFlatL2(向量文件.shape[1]))
            向量索引.k_factor = Self.Config.INDEX_REFINEFLAT_K_FACTOR
            for _ in tqdm(range(1), desc="tqdm.index.build"):
                向量索引.add(向量文件)
        Self.Module.写入日志("log.core.index.generate.end", info_level=0)
        return 向量索引
    def 缓存索引(Self, 向量文件, 文本文件):
        return TranslatorPersistence.缓存索引(Self=Self, 向量文件=向量文件, 文本文件=文本文件)
    def 翻译语言列表(Self, texts: list) -> list:
        输入列表 = []
        返回列表 = []
        命中缓存 = []
        QuestsMode = False
        if texts == []:
            return []
        try:
            if isinstance(texts[0][0], list):
                QuestsMode = True
        except: pass
        剔除方法 = re.compile(r'^\{[^}]+\}$')
        texts = [texts[index] for index in range(len(texts)) if not bool(剔除方法.match(texts[index][1]))] if QuestsMode else texts
        参考字典 = {str(item[0]) for item in texts}
        if Self.Config.TRANSLATOR_CACHE_READ:
            翻译缓存, 匹配字典 = Self.Module.翻译缓存()
            原始长度 = len(texts)
            待翻译 = []
            for item in tqdm(texts, desc="tqdm.translations.cache.use"):
                if item[1] in 匹配字典:
                    命中缓存.append([item[0], item[1], 翻译缓存[item[1]]])
                else:
                    待翻译.append(item)
            texts[:] = 待翻译
            成功缓存 = len(命中缓存)
            命中率 = (成功缓存 / 原始长度) if 原始长度 > 0 else 0.0
            Self.Module.写入日志("log.core.translator.cache.hit", hit=f"{命中率:.4%}", info_level=0)
        for index in texts:
            try:
                输入列表.append([index[1], index[0]])
            except Exception:
                Self.Module.写入日志("log.core.parsing.parameters.error", e=eb.format_exc(), index=index, info_level=2)
                pass
        try:
            if texts:
                向量文件, 文本文件 = Self.参考词预处理()
                if 文本文件:
                    Self.Module.写入日志("log.core.index.search.start", info_level=0)
                    向量文件 = Self.Quantization.解码向量(向量文件)
                    Self.Module.写入日志("log.core.debug.vector.shape", shape=向量文件.shape, info_level=4)
                    Self.Module.写入日志("log.core.debug.vector.range", range=(向量文件.min(), 向量文件.max()), info_level=4)
                    向量索引 = Self.缓存索引(向量文件, 文本文件)
                    输入列表 = Self.并行生成向量(输入列表)
                    向量列表 = np.array(输入列表[0]).astype(np.float32)
                    向量列表 = 向量列表.get() if GPU_ACC else 向量列表
                    for _ in tqdm(range(1), desc="tqdm.index.search"):
                        索引结果矩阵 = 向量索引.search(向量列表, Self.Config.INDEX_K)[1]
                    返回请求内容 = []
                    返回其他内容 = []
                    文本索引 = {index2[0]: index for index, index2 in enumerate(文本文件)}
                    原文文本文件 = [index[0] for index in 文本文件]
                    for index in range(len(向量列表)):
                        请求内容 = 输入列表[1][0][index]
                        键 = 输入列表[1][1][index]
                        其他内容 = [键, f"Reference:"] if QuestsMode else [键, f"Key: {键}|Reference:"]
                        for index2 in 索引结果矩阵[index]:
                            原文 = 原文文本文件[index2]
                            if 原文 in 文本索引:
                                其他内容[1] += f"{文本文件[文本索引[原文]]}|"
                        返回请求内容.append(请求内容)
                        返回其他内容.append(其他内容)
                    Self.Module.写入日志("log.core.index.search.end", info_level=0)
                else:
                    返回请求内容 = [row[1] for row in texts]
                    返回其他内容 = [[row[0], f"Key:{row[0]}"] for row in texts]
                其他列表 = [返回其他内容[i:i + Self.Config.LLM_MAX_BATCH] for i in range(0, len(返回其他内容), Self.Config.LLM_MAX_BATCH)]
                请求列表 = [返回请求内容[i:i + Self.Config.LLM_MAX_BATCH] for i in range(0, len(返回请求内容), Self.Config.LLM_MAX_BATCH)]
                Self.Module.写入日志("log.core.translator.generate.start", item=len(请求列表), info_level=0)
                with ThreadPoolExecutor(max_workers=Self.Config.LLM_MAX_WORKERS) as 执行器:
                    未来任务映射 = {
                        执行器.submit(
                            Self.生成翻译,
                            texts = index,
                            other_input = index2,
                        ): index
                        for index, index2 in zip(请求列表, 其他列表)
                    }
                    for 单个任务 in tqdm(as_completed(未来任务映射), total=len(未来任务映射), desc="tqdm.translations.generate"):
                        返回列表.extend(单个任务.result())
                Self.Module.写入日志("log.core.translator.generate.end", info_level=0)
            if Self.Config.TRANSLATOR_CACHE_WRITE:
                Self.Module.翻译缓存([[b, c] for a, b, c in 返回列表])
            返回列表.extend(命中缓存)
            返回列表 = [[a, f"{c}({b})"] for a, b, c in 返回列表 if str(a) in 参考字典] if Self.Config.LLM_ORIGINAL_REFERENCE else [[a, c] for a, b, c in 返回列表 if str(a) in 参考字典]
        except Exception:
            Self.Module.写入日志("log.core.translator.error", e=eb.format_exc(), texts=texts, info_level=3)
            raise eb.format_exc()
        return 返回列表
    def 翻译语言文件(Self, file0: str,  file1: str = "", output_path: str = "", export_inspection: bool = False, output_lang_str: bool = False, read_error: bool = True):
        output_path = Self.Module.输出路径处理(output_path)
        未翻译列表 = []
        去翻译列表 = []
        输出列表 = []
        翻译输出列表 = []
        可翻译源文件, 源文件, 参考文件, 压缩路径, 输出扩展名, file2 = Self.Module.读取资源文件(file0, file1, read_error)
        if 参考文件:
            参考字典 = {}
            for item in 参考文件:
                try:
                    参考字典[item[0]] = item[1]
                except Exception:
                    Self.Module.写入日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
                    pass
            for index1 in 可翻译源文件:
                key = index1[0]
                if key in 参考字典:
                    去翻译列表.append([key, 参考字典[key]])
                else:
                    未翻译列表.append(index1)
        else:
            未翻译列表 = 可翻译源文件.copy()
        翻译列表 = Self.翻译语言列表(未翻译列表)
        if export_inspection:
            for index in tqdm(range(len(未翻译列表)), desc="tqdm.progress.encoding"):
                行数据 = [{翻译列表[index][0]: 翻译列表[index][1]}, 未翻译列表[index][1]]
                输出列表.append(repr(行数据))
            with open(str(Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translatorlang")), 'w+', encoding='utf-8') as f:
                f.write("\n".join(输出列表))
            Self.Module.写入日志("log.core.translator.succeed", path=Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translatorlang").resolve(), info_level=0)
            return Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translatorlang")
        else:
            所有翻译结果 = 翻译列表 + 去翻译列表
            for index in 源文件:
                if index.strip().startswith(('#', '//')):
                    翻译输出列表.append(index)
                else:
                    索引成功 = False
                    for index1 in 所有翻译结果:
                        if index.split('=', 1)[0] == index1[0]:
                            翻译输出列表.append(f"{index1[0]}={index1[1]}")
                            索引成功 = True
                            break
                    if not 索引成功:
                        翻译输出列表.append(index)
            if 压缩路径 and (not output_lang_str):
                输出路径 = f"{Path(压缩路径).parent}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}"
                Self.Module.保存语言文件(输出路径, 翻译输出列表)
                压缩文件夹Path = Path(压缩路径).parent.parent.parent
                if file2[0] == False:
                    压缩文件夹Path = 压缩文件夹Path.parent
                    with open(f"{str(压缩文件夹Path)}/pack.mcmeta", "w+", encoding="utf-8") as f:
                        f.write(json.dumps({"pack": {"description": f"{Path(file0).stem}的{Self.Config.LANGUAGE_OUTPUT}语言资源包, 由 海盐青茫 制作, 由 {Self.Config.LLM_MODEL} 翻译","pack_format": 9999,"supported_formats": [0, 9999],"min_format": 0,"max_format": 9999}}, ensure_ascii=False, indent=4))
                压缩文件夹 = str(压缩文件夹Path)
                with zipfile.ZipFile(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                    for 压缩文件 in 压缩文件夹Path.rglob('*'):
                        if 压缩文件.is_file():
                            f.write(压缩文件, arcname=压缩文件.relative_to(压缩文件夹))
                Self.Module.写入日志("log.core.translator.succeed", path=Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve(), info_level=0)
                return Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip")
            else:
                if not Path(output_path).suffix:
                    output_path = str(Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}"))
                Self.Module.保存语言文件(output_path, 翻译输出列表)
                Self.Module.写入日志("log.core.translator.succeed", path=Path(output_path).resolve(), info_level=0)
                return Path(f"{output_path}")
    def 导入参考词(Self, path: str):
        def 打开文件(file: str):
            with open(file, "r", encoding="utf-8") as f:
                if Path(file).suffix == ".lang":
                    文件 = f.read().splitlines()
                elif Path(file).suffix == ".json":
                    文件 = "[" + ",".join(f"{k}={v}" for k, v in json.load(f).items()) + "]"
                文件 = [line.split('=', 1)   for line in 文件   if (stripped := line.strip()) and not stripped.startswith('#') and not stripped.startswith('//')]
            f.close()
            return 文件
        待处理列表 = []
        for index in ["*.jar", "*.zip", "*.pkl"]:
            for 文件路径 in list(Path(path).rglob(index)):
                文件路径 = str(文件路径)
                缓存路径 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}"
                if Path(文件路径).suffix == ".pkl":
                    with open(文件路径, "rb") as f:
                        待处理列表.extend(pickle.load(f))
                else:
                    处理列表 = []
                    文件列表 = Self.Module.读取压缩文件(文件路径, 缓存路径, Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT)
                    if not any(isinstance(item, list) and len(item) > 0 and item[0] == Self.Config.LANGUAGE_INPUT.lower() for item in 文件列表[1:]):
                        Self.写入日志("log.module.read.file0.not.lang.error", info_level=3)
                        raise FileNotFoundError(Self.Lang("log.module.read.file0.not.lang.error"))
                    if not any(isinstance(item, list) and len(item) > 0 and item[0] == Self.Config.LANGUAGE_OUTPUT.lower() for item in 文件列表[1:]):
                        Self.写入日志("log.module.read.file1.not.lang.error", info_level=3)
                        raise FileNotFoundError(Self.Lang("log.module.read.file1.not.lang.error"))
                    for index1 in 文件列表[1:]:
                        if index1[0].lower() == Self.Config.LANGUAGE_INPUT.lower():
                            原文文件 = index1[1]
                        if index1[0].lower() == Self.Config.LANGUAGE_OUTPUT.lower():
                            目标文件 = index1[1]
                    原文文件, 目标文件 = 打开文件(原文文件), 打开文件(目标文件)
                    目标映射 = {}
                    for item in 目标文件:
                        if len(item) >= 2:
                            目标映射[item[0]] = item[1]
                    处理列表.extend([[原文[1], 目标映射[原文[0]]] for 原文 in 原文文件 if 原文[0] in 目标映射])
                    待处理列表 += 处理列表
        Self.参考词预处理(待处理列表)
    def 翻译FTB任务(Self, path: str):
        翻译列表 = []
        snbt文件 = [str(index) for index in Path(path).rglob("*.snbt")]
        Self.Module.写入日志("log.core.file.quests.read.start", info_level=0)
        with mp.Pool(processes=Self.Config.QUESTS_FTB_READ_MAX_CONCURRENT) as 解释器:
            任务结果 = 解释器.imap(Self.Module.读取单个FTBQ_Snbt文件, snbt文件)
            for 单个任务 in tqdm(任务结果, total=len(snbt文件), desc="tqdm.file.read"):
                翻译列表.extend(单个任务)
        Self.Module.写入日志("log.core.file.quests.read.end", info_level=0)
        翻译列表2 = []
        try:
            for index in 翻译列表:
                if index[1] and not (re.match(r'^[a-z0-9._-]+$', index[1]) and '.' in index[1]):
                    翻译列表2.append(index)
        except Exception: 
            Self.Module.写入日志("log.module.quests.clean.error", index=index, info_level=2)
        翻译列表 = Self.翻译语言列表(翻译列表2)
        分组 = defaultdict(list)
        for item in 翻译列表:
            key = item[0][0]
            分组[key].append(item)
        翻译列表 = [分组[k] for k in sorted(分组.keys())]
        with mp.Pool(processes=Self.Config.QUESTS_FTB_WRITE_MAX_CONCURRENT) as 解释器:
            任务结果 = 解释器.imap(partial(Self.Module.应用FTBQ翻译, mode=("H" if os.path.isdir(os.path.join(path, "quests")) else "L")), 翻译列表)
            for 单个任务 in tqdm(任务结果, total=len(翻译列表), desc="tqdm.translations.use"): pass
        Self.Module.写入日志("log.core.translator.succeed", path=Path(path).resolve(), info_level=0)
    def 翻译BQ任务(Self, path: str):
        翻译列表 = []
        nbt文件 = [str(index) for index in Path(path).rglob("*.json")]
        Self.Module.写入日志("log.core.file.quests.read.start", info_level=0)
        with mp.Pool(processes=Self.Config.QUESTS_BQ_READ_MAX_CONCURRENT) as 解释器:
            任务结果 = 解释器.imap(Self.Module.读取单个BQ_Json文件, nbt文件)
            for 单个任务 in tqdm(任务结果, total=len(nbt文件), desc="tqdm.file.read"):
                翻译列表.extend(单个任务)
        Self.Module.写入日志("log.core.file.quests.read.end", info_level=0)
        翻译列表2 = []
        try:
            for index in 翻译列表:
                if index[1] and not (re.match(r'^[a-z0-9._-]+$', index[1]) and '.' in index[1]):
                    翻译列表2.append(index)
        except Exception: 
            Self.Module.写入日志("log.module.quests.clean.error", index=index, info_level=2)
        翻译列表 = Self.翻译语言列表(翻译列表2)
        分组 = defaultdict(list)
        for item in 翻译列表:
            key = item[0][0]
            分组[key].append(item)
        翻译列表 = [分组[k] for k in sorted(分组.keys())]
        with mp.Pool(processes=Self.Config.QUESTS_BQ_WRITE_MAX_CONCURRENT) as 解释器:
            任务结果 = 解释器.imap(partial(Self.Module.应用BQ翻译), 翻译列表)
            for 单个任务 in tqdm(任务结果, total=len(翻译列表), desc="tqdm.translations.use"): pass
        Self.Module.写入日志("log.core.translator.succeed", path=Path(path).resolve(), info_level=0)
    def 导入DictMini参考词(Self, file: str, mode: str = "dense"):
        Self.Module.写入日志("log.core.file.settle.start", info_level=0)
        待处理列表 = []
        for _ in tqdm(range(1), desc="tqdm.file.read"):
            with open(file, "rb") as f:
                Dict文件 = json.load(f)
        if mode == "dense":
            for index in tqdm(Dict文件, desc="tqdm.file.processing"):
                待处理列表.append([index, ", ".join(Dict文件[index])])
        elif mode == "sparse":
            for index in tqdm(Dict文件, desc="tqdm.file.processing"):
                for index1 in Dict文件[index]:
                    待处理列表.append([index, index1])
        Self.参考词预处理(待处理列表)
        Self.Module.写入日志("log.core.file.settle.end", info_level=0)
    def 导入DictMini缓存(Self, file: str):
        Self.Module.写入日志("log.core.file.settle.start", info_level=0)
        文本列表 = []
        for _ in tqdm(range(1), desc="tqdm.file.read"):
            with open(file, "rb") as f:
                Dict文件 = json.load(f)
        for index in tqdm(Dict文件, desc="tqdm.file.processing"):
            文本列表.append([index, Dict文件[index][0]])
        Self.Module.翻译缓存(文本列表)
        Self.Module.写入日志("log.core.file.settle.end", info_level=0)
    def DictMini转换数据集(Self, file: str, mode: str = "Alpaca", output_file: str = "dataset.jsonl"):
        Self.Module.写入日志("log.core.file.settle.start", info_level=0)
        待处理列表 = []
        for _ in tqdm(range(1), desc="tqdm.file.read"):
            with open(file, "rb") as f:
                Dict文件 = json.load(f)
        for index in tqdm(Dict文件, desc="tqdm.file.processing"):
            for index2 in Dict文件[index]:
                待处理列表.append([index, index2])
        导出列表 = []
        for index in tqdm(待处理列表, desc="tqdm.progress.encoding"):
            if mode == "ChatML":
                导出列表.append({"messages": [{"role": "system", "content": f"将下列文本翻译为{Self.Config.LANGUAGE_OUTPUT}语言"}, {"role": "user", "content": index[0]}, {"role": "assistant", "content": index[1]}]})
            elif mode == "Alpaca":
                导出列表.append({"instruction": f"翻译为{Self.Config.LANGUAGE_OUTPUT}语言", "input": index[0], "output": index[1], "system": f"将下列文本翻译为{Self.Config.LANGUAGE_OUTPUT}语言"})
        with open(output_file, 'w+', encoding='utf-8') as f:
            f.write('\n'.join(json.dumps(item, ensure_ascii=False, separators=(',', ':')) for item in tqdm(导出列表, desc="tqdm.progress.encoding")))
        Self.Module.写入日志("log.core.file.settle.end", info_level=0)
    def 分离语言文件更新(Self, file0: str, file1: str = "", output_path: str = "", mode: str = "none"):
        Self.Module.写入日志("log.core.lang.separate.start", info_level=0)
        output_path = Self.Module.输出路径处理(output_path)
        缺失列表 = []
        输出列表 = []
        文件0, _, 文件1, _, 输出扩展名, _ = Self.Module.读取资源文件(file0, file1)
        if 文件1:
            参考字典 = {}
            for item in 文件1:
                try:
                    参考字典[item[0]] = item[1]
                except Exception:
                    Self.Module.写入日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
                    pass
            for index1 in 文件0:
                key = index1[0]
                if not key in 参考字典:
                    缺失列表.append(index1)
        else:
            缺失列表 = 文件0.copy()
        if not Path(output_path).suffix:
            导出路径 = str(Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}"))
        else:
            导出路径 = str(Path(output_path))
        if mode == "none":
            Self.Module.保存语言文件(导出路径, [f"{index[0]}={index[1]}" for index in 缺失列表])
            Self.Module.写入日志("log.core.settle.succeed", path=Path(导出路径).resolve(), info_level=0)
            返回路径 = Path(导出路径).resolve()
        elif mode == "extra":
            输入列表 = []
            for index in 缺失列表:
                try:
                    输入列表.append([index[1], index[0]])
                except Exception:
                    Self.Module.写入日志("log.core.parsing.parameters.error", e=eb.format_exc(), index=index, info_level=2)
                    pass
            向量文件, 文本文件 = Self.参考词预处理()
            向量文件 = Self.Quantization.解码向量(向量文件)
            Self.Module.写入日志("log.core.debug.vector.shape", shape=向量文件.shape, info_level=4)
            if 文本文件:
                向量索引 = Self.缓存索引(向量文件, 文本文件)
                输入列表 = Self.并行生成向量(输入列表)
                向量列表 = np.array(输入列表[0]).astype(np.float32)
                向量列表 = 向量列表.get() if GPU_ACC else 向量列表
                for _ in tqdm(range(1), desc="tqdm.index.search"):
                    索引结果矩阵 = 向量索引.search(向量列表, Self.Config.INDEX_K)[1]
                返回请求内容 = []
                返回其他内容 = []
                文本索引 = {index2[0]: index for index, index2 in enumerate(文本文件)}
                原文文本文件 = [index[0] for index in 文本文件]
                for index in range(len(向量列表)):
                    其他内容 = ""
                    for index2 in 索引结果矩阵[index]:
                        原文 = 原文文本文件[index2]
                        if 原文 in 文本索引:
                            其他内容 += (f"{文本文件[文本索引[原文]]}|")
                    返回请求内容.append({输入列表[1][1][index]: 输入列表[1][0][index]})
                    返回其他内容.append(f"{其他内容}")
            for index in tqdm(range(len(返回请求内容)), desc="tqdm.progress.encoding"):
                行数据 = [返回请求内容[index], 返回其他内容[index]]
                输出列表.append(repr(行数据))
            with open(str(Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translatorlang").resolve()), 'w+', encoding='utf-8') as f:
                f.write("\n".join(输出列表))
            Self.Module.写入日志("log.core.settle.succeed", path=Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translatorlang").resolve(), info_level=0)
            返回路径 = Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translatorlang").resolve()
        Self.Module.写入日志("log.core.lang.separate.end", info_level=0)
        return 返回路径
    def 合并语言文件更新(Self, file0: str, notlang_file: str, file1: str = "", output_path: str = ""):
        Self.Module.写入日志("log.core.lang.merge.start", info_level=0)
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
                    Self.Module.写入日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
                    pass
            for index1 in 文件0:
                key = index1[0]
                if key in 参考字典:
                    合并列表.append(index1)
        else:
            合并列表 = 文件0.copy()
        if Path(notlang_file).suffix == ".translatorlang":
            with open(notlang_file, 'r', encoding='utf-8') as f:
                缺失列表文件 = f.read().splitlines()
            for index in 缺失列表文件:
                index = ast.literal_eval(index)[0]
                for index1, index2 in index.items():
                    缺失列表.append([index1, index2])
        else:
            缺失列表 = Self.Module.读取语言文件(notlang_file)[0]
        合并列表 = 合并列表 + 缺失列表
        for index in 文件0源文件:
            if index.strip().startswith(('#', '//')):
                输出列表.append(index)
            else:
                索引成功 = False
                for index1 in 合并列表:
                    if index.split('=', 1)[0] == index1[0]:
                        输出列表.append(f"{index1[0]}={index1[1]}")
                        索引成功 = True
                        break
                if not 索引成功:
                    输出列表.append(index)
        if 压缩路径:
            输出路径 = f"{Path(压缩路径).parent}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}"
            Self.Module.保存语言文件(输出路径, 输出列表)
            压缩文件夹Path = Path(压缩路径).parent.parent.parent
            if file2[0] == False:
                压缩文件夹Path = 压缩文件夹Path.parent
                with open(f"{str(压缩文件夹Path)}/pack.mcmeta", "w+", encoding="utf-8") as f:
                    f.write(json.dumps({"pack": {"description": f"{Path(file0).stem}的{Self.Config.LANGUAGE_OUTPUT}语言资源包","pack_format": 9999,"supported_formats": [0, 9999],"min_format": 0,"max_format": 9999}}, ensure_ascii=False, indent=4))
            压缩文件夹 = str(压缩文件夹Path)
            with zipfile.ZipFile(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                for 压缩文件 in 压缩文件夹Path.rglob('*'):
                    if 压缩文件.is_file():
                        f.write(压缩文件, arcname=压缩文件.relative_to(压缩文件夹))
            Self.Module.写入日志("log.core.settle.succeed", path=Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve(), info_level=0)
            返回路径 = Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve()
        else:
            输出路径 = str(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}")
            Self.Module.保存语言文件(输出路径, 输出列表)
            Self.Module.写入日志("log.core.settle.succeed", path=Path(输出路径).resolve(), info_level=0)
            返回路径 = Path(输出路径).resolve()
        Self.Module.写入日志("log.core.lang.merge.end", info_level=0)
        return 返回路径
    def 翻译整合包(Self, path: str, all_mode: bool = False):
        翻译列表路径 = {}
        if Path(f"{path}/mods").is_dir():
            I18n模组ID = [] if all_mode else Self.Module.从资源包文件夹获取I18n翻译模组ID(path)
            模组ID = Self.Module.从模组文件夹获取模组ID(path)
            模组ID字典 = {item[0]: item[1] for item in 模组ID}
            I18n缺失模组ID = []
            for index in 模组ID字典:
                if index not in I18n模组ID:
                    I18n缺失模组ID.append([index, 模组ID字典[index]]) 
            缓存路径 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}/"
            for index in tqdm(I18n缺失模组ID, desc="tqdm.translations.mod"):
                try:
                    保存路径 = Path(f"{缓存路径}/assets/{index[0]}/lang/")
                    保存路径.mkdir(parents=True, exist_ok=True)
                    Self.翻译语言文件(file0=f"{path}/mods/{index[1]}", file1="", output_path=保存路径, output_lang_str=True, read_error=False)
                except FileNotFoundError:
                    Self.Module.写入日志("log.core.translator.modpack.error.mod", e="", mod=index[0], info_level=0)
                except Exception:
                    Self.Module.写入日志("log.core.translator.modpack.error.mod", e=eb.format_exc(), mod=index[0], info_level=1)
            for p in sorted(Path(f"{缓存路径}/assets/").rglob('*'), reverse=True):
                try: p.rmdir()
                except: pass
            with open(f"{str(缓存路径)}/pack.mcmeta", "w+", encoding="utf-8") as f:
                f.write(json.dumps({"pack": {"description": f"{Self.Config.LANGUAGE_OUTPUT}语言资源包, 由 海盐青茫 制作, 由 {Self.Config.LLM_MODEL} 翻译","pack_format": 9999,"supported_formats": [0, 9999],"min_format": 0,"max_format": 9999}}, ensure_ascii=False, indent=4))
            Path(f"{path}/resourcepacks/").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(f"{path}/resourcepacks/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                for 压缩文件 in Path(缓存路径).rglob('*'):
                    if 压缩文件.is_file():
                        f.write(压缩文件, arcname=压缩文件.relative_to(缓存路径))
            翻译列表路径[f"/resourcepacks/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}.zip"] = ["file"]
        if Path(f"{path}/config/ftbquests").is_dir():
            Self.翻译FTB任务(f"{path}/config/ftbquests")
            翻译列表路径[f"/config/ftbquests"] = ["path"]
        if Path(f"{path}/config/betterquesting").is_dir():
            Self.翻译BQ任务(f"{path}/config/betterquesting")
            翻译列表路径[f"/config/betterquesting"] = ["path"]
        for index in ["resources", "kubejs/assets"]:
            文件夹路径 = f"{path}/{index}"
            if Path(文件夹路径).is_dir():
                所有文件夹 = [p.name for p in Path(文件夹路径).iterdir() if p.is_dir()]
                for 文件夹 in tqdm(所有文件夹, desc="tqdm.translations.resource"):
                    无后缀语言文件名 = f"{文件夹路径}/{文件夹}/lang/{Self.Config.LANGUAGE_INPUT}"
                    if Path(f"{无后缀语言文件名}.lang").is_file():
                        Self.翻译语言文件(file0=f"{无后缀语言文件名}.lang", output_path=f"{文件夹路径}/{文件夹}/lang")
                    if Path(f"{无后缀语言文件名}.json").is_file():
                        Self.翻译语言文件(file0=f"{无后缀语言文件名}.json", output_path=f"{文件夹路径}/{文件夹}/lang")
            翻译列表路径[f"/{index}"] = ["path"]
        Self.Module.写入日志("log.core.translator.succeed", path=Path(f"{path}/resourcepacks/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve(), info_level=0)
        return 翻译列表路径
    def 翻译通用文件(Self, file0, file1 = None, all_mode: bool = False, export_inspection = False):
        file0 = Path(file0).resolve()
        if file1:
            file1 = Path(file1).resolve()
        缓存文件夹 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/"
        Path(缓存文件夹).mkdir(parents=True, exist_ok=True)
        Self.Module.写入日志("log.core.translator.general.generate.start", info_level=0)
        返回内容 = None
        try:
            if Path(file0).is_file():
                文件0扩展名 = Path(file0).suffix
                if 文件0扩展名 in [".lang", ".json", ".jar"]:
                    Self.Module.写入日志("log.core.translator.general.model", model="Mod" if 文件0扩展名 == ".jar" else "Language File", info_level=0)
                    返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                    Self.Module.写入日志("log.core.translator.succeed", path=Path(返回路径).resolve(), info_level=0)
                    返回内容 = Path(返回路径)
                elif 文件0扩展名 in [".zip", ".mrpack"]:
                    with zipfile.ZipFile(file0, 'r') as zf:
                        namelist = zf.namelist()
                        def has_dir(prefix: str) -> bool:
                            return any(name.startswith(prefix + '/') or name == prefix for name in namelist)
                        if has_dir('shaders'):
                            Self.Module.写入日志("log.core.translator.general.model", model="Shaders", info_level=0)
                            返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                            Self.Module.写入日志("log.core.translator.succeed", path=Path(返回路径).resolve(), info_level=0)
                            返回内容 = Path(返回路径)
                        elif has_dir('ftbquests'):
                            Self.Module.写入日志("log.core.translator.general.model", model="FTBQuests", info_level=0)
                            zf.extractall(缓存文件夹)
                            Self.翻译FTB任务(f"{缓存文件夹}/ftbquests")
                            with zipfile.ZipFile(f"{缓存文件夹}/FTBQuests-Translation.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                                for 压缩文件 in Path(f"{缓存文件夹}/ftbquests").rglob('*'):
                                    if 压缩文件.is_file():
                                        f.write(压缩文件, arcname=压缩文件.relative_to(str(缓存文件夹)))
                            返回内容 = Path(f"{缓存文件夹}/FTBQuests-Translation.zip")
                        elif has_dir('betterquesting'):
                            Self.Module.写入日志("log.core.translator.general.model", model="BetterQuesting", info_level=0)
                            zf.extractall(缓存文件夹)
                            Self.翻译BQ任务(f"{缓存文件夹}/betterquesting")
                            with zipfile.ZipFile(f"{缓存文件夹}/BetterQuesting-Translation.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                                for 压缩文件 in Path(f"{缓存文件夹}/betterquesting").rglob('*'):
                                    if 压缩文件.is_file():
                                        f.write(压缩文件, arcname=压缩文件.relative_to(str(缓存文件夹)))
                            返回内容 = Path(f"{缓存文件夹}/BetterQuesting-Translation.zip")
                        else:
                            roots = {n.split('/')[0] for n in namelist if not n.startswith('__MACOSX/')}
                            整合包模式 = "General ModPack"
                            if has_dir('overrides'):
                                roots = ["overrides"]
                                整合包模式 = "CurseForge/Modrint/General ModPack"
                            if has_dir('minecraft'):
                                roots = ["minecraft"]
                                整合包模式 = "MultiMC/General ModPack"
                            if len(roots) == 1:
                                root = roots.pop()
                                if has_dir(f'{root}/mods') or has_dir(f'{root}/config') or has_dir(f'{root}/kubejs') or has_dir(f'{root}/resources'):
                                    Self.Module.写入日志("log.core.translator.general.model", model=整合包模式, info_level=0)
                                    zf.extractall(缓存文件夹)
                                    解压根目录完整路径 = f"{缓存文件夹}/{root}"
                                    压缩路径映射 = Self.翻译整合包(解压根目录完整路径, all_mode=all_mode)
                                    输出Zip路径 = f"{缓存文件夹}/ModPack-Translation-Addion.zip"
                                    with zipfile.ZipFile(输出Zip路径, 'w', zipfile.ZIP_DEFLATED) as modpackzf:
                                        for 相对路径, 类型列表 in 压缩路径映射.items():
                                            类型 = 类型列表[0] if 类型列表 else ""
                                            清理后的相对路径 = 相对路径.lstrip('/')
                                            真实文件路径 = os.path.join(解压根目录完整路径, 清理后的相对路径)
                                            if 类型 == "file":
                                                modpackzf.write(真实文件路径, arcname=相对路径.lstrip('/'))
                                            elif 类型 == "path":
                                                for 根目录, 子目录, 文件名列表 in os.walk(真实文件路径):
                                                    for 文件名 in 文件名列表:
                                                        文件完整路径 = os.path.join(根目录, 文件名)
                                                        文件在Zip中的相对路径 = os.path.relpath(文件完整路径, 解压根目录完整路径)
                                                        文件在Zip中的相对路径 = 文件在Zip中的相对路径.replace(os.sep, '/')
                                                        modpackzf.write(文件完整路径, arcname=文件在Zip中的相对路径)
                                    Self.Module.写入日志("log.core.translator.succeed", path=Path(输出Zip路径).resolve(), info_level=0)
                                    返回内容 = Path(输出Zip路径)
                                
                                else:
                                    Self.Module.写入日志("log.core.translator.general.modpack.translate.file.no", info_level=2)
                                    返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
                            else:
                                Self.Module.写入日志("log.core.translator.general.structure.unknown", info_level=3)
                                返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
                else:
                    Self.Module.写入日志("log.core.translator.general.structure.unknown", info_level=3)
                    返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
            elif Path(file0).is_dir():
                文件夹名称 = Path(file0).name
                if 文件夹名称 == "ftbquests":
                    Self.Module.写入日志("log.core.translator.general.model", model="FTBQuests", info_level=0)
                    Self.翻译FTB任务(path=file0)
                elif 文件夹名称 == "betterquesting":
                    Self.Module.写入日志("log.core.translator.general.model", model="BetterQuesting", info_level=0)
                    Self.翻译BQ任务(path=file0)
                else:
                    Self.Module.写入日志("log.core.translator.general.model", model="General ModPack", info_level=0)
                    Self.翻译整合包(path=file0, all_mode=all_mode)
        except Exception:
            Self.Module.写入日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=3)
            返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
        Self.Module.写入日志("log.core.translator.succeed", path=返回内容.resolve(), info_level=0)
        return 返回内容.resolve()
    def 检索缓存(Self, text: str):
        return Self.Module.翻译缓存()[0][text]

测试 = False
if __name__ == "__main__" and 测试:
    参数 = {
        "LLM_API_URL": "http://127.0.0.1:25564/v1/chat/completions",
        "LLM_MODEL": "qwen3-30b-a3b-instruct-2507-2.69bpw",
        "LOGS_FILE_NAME": "测试",
        "VEC_QUANTIZATION": "Float32",
        "LLM_MAX_BATCH": 1,
        "LLM_CONTEXTS": False,
        "EMB_MODEL": r"C:\Users\FengMang\Desktop\TranslatorMinecraft\nomic-embed-text-v1.5",
        "DEBUG_MODE": True,
        "LLM_MAX_WORKERS": 8,
        "TRANSLATOR_CACHE_WRITE": False
    }
    翻译 = Translator(参数)
    翻译2 = Translator(参数)
    #翻译.导入DictMini参考词(r"C:\Users\FengMang\Downloads\Dict-Mini.json")
    #翻译.翻译整合包(r"C:\Users\FengMang\AppData\Roaming\PrismLauncher\instances\Star Technology 翻译测试\minecraft")
    #翻译.翻译FTB任务(r"C:\Users\FengMang\AppData\Roaming\PrismLauncher\instances\Star Technology\minecraft\config\ftbquests")
    翻译.翻译通用文件(r"C:\Users\FengMang\Downloads\SEUS PTGI HRR Test 2.1.zip")
    翻译2.翻译通用文件(r"C:\Users\FengMang\Downloads\SEUS PTGI HRR Test 2.1.zip")
    #print(翻译.检索缓存('Divides capacity equally among all types. Use it if you increased the "max types" in config.'))
    #翻译.导入DictMini缓存(r"C:\Users\FengMang\Downloads\Dict-Mini.json")