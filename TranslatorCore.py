from TranslatorLib import HARDWARE_INFO, np, threading, zipfile, json, ast, os, eb, re, partial, defaultdict, Path, ThreadPoolExecutor, as_completed, Callable, Dict, Any, requests, GPU_ACC, time, uuid, faiss, HTTPAdapter, SimpleNamespace
from TranslatorConfig import RuntimeConfig
from TranslatorQuantization import Quantization
from TranslatorLocale import Locale
from TranslatorModule import Module
import TranslatorPersistence

class 翻译上下文管理器:
    def __init__(Self, 初始字典=None):
        Self.数据字典 = {}
        Self.已翻译键列表 = []
        Self.线程锁 = threading.Lock()
        Self.键索引缓存 = {}
        
        if 初始字典:
            for 索引, (键, 值) in enumerate(初始字典.items()):
                Self.数据字典[键] = 值
                Self.键索引缓存[键] = 索引
                if len(值) > 1:
                    译文 = 值[1]
                    if 译文 is not None and 译文 != "":
                        Self.已翻译键列表.append(键)

    def add(Self, 键, 译文):
        with Self.线程锁:
            if 键 in Self.数据字典:
                Self.数据字典[键][1] = 译文
                if 译文 and 译文 != "" and 键 not in Self.已翻译键列表:
                    Self.已翻译键列表.append(键)
            else:
                Self.数据字典[键] = [译文]
                Self.键索引缓存[键] = len(Self.数据字典) - 1

    def get(Self, 当前键, 数量):
        with Self.线程锁:
            if 当前键 not in Self.数据字典:
                return []
            当前全局索引 = Self.键索引缓存[当前键]
            结果列表 = []
            计数 = 0
            for 遍历键 in reversed(Self.已翻译键列表):
                if len(Self.数据字典[遍历键]) < 2 or Self.数据字典[遍历键][1] is None:
                    continue
                遍历键索引 = Self.键索引缓存[遍历键]
                if 遍历键索引 < 当前全局索引:
                    结果列表.append({"role": "assistant", "content": Self.数据字典[遍历键][1]})
                    结果列表.append({"role": "user", "content": Self.数据字典[遍历键][0]})
                    计数 += 1
                    if 计数 >= 数量:
                        break
            return 结果列表[::-1]
    
class Translator:
    def __init__(Self, Config: dict = None):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Path(Self.Config.LOGS_FILE_PATH).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.VEC_FILE_PATH).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.PATH_CACHE).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.TRANSLATOR_CACHE_PATH).mkdir(parents=True, exist_ok=True)
        Self.Module = Module(Config=Config)
        Self.Locale = Locale(Config=Config)
        Self.Lang = Self.Locale.Lang
        Self.日志 = Self.Module.写入日志
        Self.tqdm = Self.Locale.Tqdm
        Self.Quantization = Quantization(Config=Config)
        if GPU_ACC:
            Self.日志("log.core.numpy.gpu", type=HARDWARE_INFO['type'], version=HARDWARE_INFO['version'], deviceid=HARDWARE_INFO['device_id'], count=HARDWARE_INFO["device_count"], info_level=0)
        else:
            Self.日志("log.core.numpy.cpu", type=HARDWARE_INFO['type'], version=HARDWARE_INFO['version'], e=HARDWARE_INFO['error'], info_level=0)
        Self.上下文 = defaultdict(list)
        Self.嵌入模型 = None
        Self.重排序模型 = None
        Self.本次翻译模型列表 = []
        Self.线程锁 = threading.Lock()
        Self.函数库: Dict[str, Callable] = {}
        Self.扫描模块函数(Self.Module)
        Self.Config.TRANSLATOR_SYSTEM_PROMPT = Self.Config.TRANSLATOR_SYSTEM_PROMPT.format(LANGUAGE_OUTPUT=Self.Config.LANGUAGE_OUTPUT)
        Self.会话 = SimpleNamespace()
        EMB适配器 = HTTPAdapter(pool_maxsize=Self.Config.EMB_MAX_WORKERS)
        Self.会话.EMB = requests.Session()
        Self.会话.EMB.mount('https://', EMB适配器)
        Self.会话.EMB.mount('http://', EMB适配器)
        Self.会话.EMB.headers.update({"Authorization": f"Bearer {Self.Config.EMB_API_KEY}"})
        LLM适配器 = HTTPAdapter(pool_maxsize=Self.Config.LLM_MAX_WORKERS)
        Self.会话.LLM = requests.Session()
        Self.会话.LLM.mount('https://', LLM适配器)
        Self.会话.LLM.mount('http://', LLM适配器)
        Self.会话.LLM.headers.update({"Authorization": f"Bearer {Self.Config.LLM_API_KEY}"})
        RERANKER适配器 = HTTPAdapter(pool_maxsize=Self.Config.RERANKER_MAX_WORKERS)
        Self.会话.RERANKER = requests.Session()
        Self.会话.RERANKER.mount('https://', RERANKER适配器)
        Self.会话.RERANKER.mount('http://', RERANKER适配器)
        Self.会话.RERANKER.headers.update({"Authorization": f"Bearer {Self.Config.RERANKER_API_KEY}"})
        Self.正则表达式预编译 = SimpleNamespace()
        Self.正则表达式预编译.括号分离方式 = re.compile(r'^(?:[&§][0-9a-fk-or])*\s*\{([^}]+)\}(.*)', re.DOTALL)
        Self.正则表达式预编译.翻译剔除方法 = re.compile(r'^\{[^}]+\}$|^.{1,2}$')
        Self.正则表达式预编译.MMTQM格式化 = re.compile(r'%(\d+\$)?[sdf]|§[0-9a-fk-or]|&[0-9a-fk-or]', re.IGNORECASE)
        Self.owolib解析缓存 = {}
    def __enter__(Self):
        return Self
    def __exit__(Self, *args):
        for session_name in ['EMB', 'LLM', 'RERANKER']:
            getattr(Self.会话, session_name).close()
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
    def 生成向量(Self, text: list, outputs: list = None, outputs1: list = None) -> np.float32:
        重试次数 = 0
        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            with Self.线程锁:
                try:
                    向量列表 = Self.嵌入模型.encode(text)
                    向量列表 = np.asarray(向量列表, dtype=np.float32)
                    return [向量列表, [text, outputs, outputs1]]
                except Exception:
                    Self.日志("log.core.locally.generate.vectors.error", e=eb.format_exc(), info_level=2)
                    return [None, [text, outputs, outputs1]]
        else:
            while 重试次数 < Self.Config.EMB_MAX_RETRY:
                try:
                    请求结果 = Self.会话.EMB.post(url=Self.Config.EMB_API_URL, json={"input": text,"model": Self.Config.EMB_MODEL}, timeout=(Self.Config.EMB_CONN_TIMEOUT, Self.Config.EMB_TIMEOUT))
                    请求结果.raise_for_status()
                    请求结果 = 请求结果.json()
                    向量列表 = []
                    for index in range(len(text)):
                        向量列表.append(请求结果['data'][index]['embedding'])
                    向量列表 = np.asarray(向量列表, dtype=np.float32)
                    return [向量列表, [text, outputs, outputs1]]
                except Exception:
                    重试次数 += 1
                    if 重试次数 >= Self.Config.EMB_MAX_RETRY:
                        Self.日志("log.core.api.generate.vectors.error", e=eb.format_exc(), info_level=3)
                        return [None, [text, outputs, outputs1]]
                    else:
                        Self.日志("log.core.api.generate.vectors.retry", e=eb.format_exc(), info_level=2)
                        time.sleep((Self.Config.EMB_RETRY_COEF ** (重试次数 - 1)) * Self.Config.EMB_RETRY_TIME)
    def 并行生成向量(Self, texts: list) -> list:
        Self.日志("log.core.vector.generate.start", info_level=0)
        if not texts:
            Self.日志("log.core.generated.vector.nan", texts=texts, info_level=3)
            return [np.array([], dtype=np.float32).reshape(0, 0), [[], [], []]]
        最大字符数 = Self.Config.EMB_MAX_TOKENS * Self.Config.EMB_TOKENSTOTEXT_RATIO
        分组结果, 当前组, 当前总长 = [], [], 0.0
        for index in texts:
            长度 = len(index[0])
            if 当前总长 + 长度 > 最大字符数:
                分组结果.append(当前组)
                当前组, 当前总长 = [], 0.0
            当前组.append(index)
            当前总长 += 长度
        if 当前组: 分组结果.append(当前组)
        待处理文本列表原文 = [[item[0] for item in group] for group in 分组结果]
        待处理文本列表额外输出 = [[item[1] for item in group] for group in 分组结果]
        待处理文本列表额外输出1 = [[item[2] for item in group] for group in 分组结果]
        总条数 = sum(len(g) for g in 分组结果)
        合并向量 = None
        维度 = None
        当前偏移 = 0
        偏移锁 = threading.Lock()
        提前完成缓存 = []
        合并请求文本 = []
        合并额外返回 = []
        合并额外返回1 = []
        with ThreadPoolExecutor(max_workers=Self.Config.EMB_MAX_WORKERS) as 执行器:
            未来任务映射 = {
                执行器.submit(Self.生成向量, 原文组, 额外输出, 额外输出1): 原文组
                for 原文组, 额外输出, 额外输出1 in zip(待处理文本列表原文, 待处理文本列表额外输出, 待处理文本列表额外输出1)
            }
            for 单个任务 in Self.tqdm(as_completed(未来任务映射), total=len(未来任务映射), desc="tqdm.vectors.generate"):
                结果 = 单个任务.result()
                if 结果[0] is None: continue
                块 = 结果[0]
                块长度 = 块.shape[0]
                with 偏移锁:
                    if 维度 is None:
                        维度 = 块.shape[1]
                        合并向量 = np.empty((总条数, 维度), dtype=np.float32)
                        for 缓存块, 缓存元数据 in 提前完成缓存:
                            合并向量[当前偏移:当前偏移+缓存块.shape[0]] = 缓存块
                            当前偏移 += 缓存块.shape[0]
                            合并请求文本.extend(缓存元数据[0])
                            合并额外返回.extend(缓存元数据[1])
                            合并额外返回1.extend(缓存元数据[2])
                        提前完成缓存.clear()
                    elif 块.shape[1] != 维度:
                        continue
                    合并向量[当前偏移:当前偏移+块长度] = 块
                    当前偏移 += 块长度
                合并请求文本.extend(结果[1][0])
                合并额外返回.extend(结果[1][1])
                合并额外返回1.extend(结果[1][2])
        最终向量 = 合并向量[:当前偏移] if 合并向量 is not None else np.array([], dtype=np.float32).reshape(0, 维度 or 0)
        Self.日志("log.core.vector.generate.end", info_level=0)
        return [最终向量, [合并请求文本, 合并额外返回, 合并额外返回1]]

    def 参考词预处理(Self, texts: list = None,) -> tuple[np.ndarray, list]:
        return TranslatorPersistence.参考词预处理(Self=Self, texts=texts)
    def 生成翻译(Self, texts: list, other_input: str, tier_cfg: dict = None, context: 翻译上下文管理器 = None):
        if tier_cfg is None:
            api_url = Self.Config.LLM_API_URL
            api_key = Self.Config.LLM_API_KEY
            model = Self.Config.LLM_MODEL
            session = Self.会话.LLM
            top_p = Self.Config.LLM_TOP_P
            top_k = Self.Config.LLM_TOP_K
            temperature = Self.Config.LLM_TEMP
            presence_penalty = Self.Config.LLM_PP
            frequency_penalty = Self.Config.LLM_FP
            seed = Self.Config.LLM_SEED
            max_retry = Self.Config.LLM_MAX_RETRY
            timeout = (Self.Config.LLM_CONN_TIMEOUT, Self.Config.LLM_TIMEOUT)
            retry_time = Self.Config.LLM_RETRY_TIME
            retry_coef = Self.Config.LLM_RETRY_COEF
            extra_kwargs = Self.Config.LLM_API_KWARGS
        else:
            api_url = tier_cfg.get("url", Self.Config.LLM_API_URL)
            api_key = tier_cfg.get("key", Self.Config.LLM_API_KEY)
            model = tier_cfg.get("model", Self.Config.LLM_MODEL)
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {api_key}"})
            top_p = tier_cfg.get("top_p", Self.Config.LLM_TOP_P)
            top_k = tier_cfg.get("top_k", Self.Config.LLM_TOP_K)
            temperature = tier_cfg.get("temperature", Self.Config.LLM_TEMP)
            presence_penalty = tier_cfg.get("presence_penalty", Self.Config.LLM_PP)
            frequency_penalty = tier_cfg.get("frequency_penalty", Self.Config.LLM_FP)
            seed = tier_cfg.get("seed", Self.Config.LLM_SEED)
            max_retry = tier_cfg.get("max_retry", Self.Config.LLM_MAX_RETRY)
            timeout = (
                tier_cfg.get("conn_timeout", Self.Config.LLM_CONN_TIMEOUT),
                tier_cfg.get("timeout", Self.Config.LLM_TIMEOUT)
            )
            retry_time = tier_cfg.get("retry_time", Self.Config.LLM_RETRY_TIME)
            retry_coef = tier_cfg.get("retry_coef", Self.Config.LLM_RETRY_COEF)
            extra_kwargs = tier_cfg.get("api_kwargs", Self.Config.LLM_API_KWARGS)
        额外内容 = [[index[1][0], index[1][1]] for index in other_input]
        额外提示词 = [f"{index1[0]} --> {index1[1]}" for index in 额外内容 for index1 in index[1]]
        额外提示词 = Self.Module.列表去重(额外提示词)
        额外提示词 = " | ".join(额外提示词)
        键提示词 = []
        for idx in other_input:
            所有键 = idx[3] if len(idx) > 3 else [idx[0]]
            键提示词.append("Key:" + ",".join(所有键))
        键提示词 = " | ".join(键提示词)
        附加内容 = f"{键提示词}\n{额外提示词}"
        消息结果 = ""
        messages = [{"role": "system", "content": Self.Config.TRANSLATOR_SYSTEM_PROMPT + 附加内容}]
        if Self.Config.LLM_CONTEXTS != False:
            for index in 额外内容:
                messages.extend(context.get(index[0], Self.Config.LLM_CONTEXTS))
        请求文本 = []
        分离文本 = []
        for index in texts:
            括号分离结果 = Self.正则表达式预编译.括号分离方式.match(index)
            if 括号分离结果:
                分离文本.append(括号分离结果.group(2))
                请求文本.append(括号分离结果.group(1))
            else:
                分离文本.append("")
                请求文本.append(index)
        请求文本长度 = len(请求文本)
        请求文本 = 请求文本[0] if 请求文本长度 == 1 else str(请求文本)
        请求文本 = Self.Config.TRANSLATOR_USER_PROMPT.format(
            text=请求文本, LANGUAGE_OUTPUT=Self.Config.LANGUAGE_OUTPUT, COUNT=len(请求文本)
        )
        messages.append({"role": "user", "content": 请求文本})
        请求内容 = {
            "model": model,
            "messages": messages,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "stream": False,
        } | extra_kwargs
        请求次数 = 0
        while 请求次数 < max_retry:
            try:
                请求结果 = ""
                请求结果 = session.post(url=api_url, json=请求内容, timeout=timeout)
                请求结果.raise_for_status()
                请求结果 = 请求结果.json()
                Token结果 = 请求结果.get("usage", {})
                消息结果 = 请求结果["choices"][0]["message"]["content"]
                Self.Config.LLM_TOKEN_USAGE += Token结果.get("total_tokens", 0)
                Self.日志("log.core.translator.generate.request.outputs.debug", messages=texts, item=消息结果, promptex=附加内容, info_level=4)
                消息结果 = re.sub(r'<think>.*?</think>\s*', '', 消息结果, flags=re.DOTALL)
                消息结果 = re.sub(r'[think].*?[/think]\s*', '', 消息结果, flags=re.DOTALL)
                返回的请求结果 = [消息结果] if 请求文本长度 == 1 else ast.literal_eval(消息结果)
                处理后的请求结果 = []
                for index in range(len(texts)):
                    处理后的请求结果.append(f"{分离文本[index]}{返回的请求结果[index]}")
                返回结果 = []
                for index in range(len(texts)):
                    返回结果.append([other_input[index][0], texts[index], 处理后的请求结果[index], other_input[index][2]])
                    if Self.Config.LLM_CONTEXTS != False:
                        context.add(other_input[index][0], 处理后的请求结果[index])
                    Self.日志("log.core.translator.generate", input=texts[index], output=返回的请求结果[index])
                return 返回结果
            except Exception:
                Self.日志("log.core.translator.generate.messages.error", promptex=附加内容, messages=texts, e=eb.format_exc(), info_level=1)
                返回结果 = [[other_input[index][0], texts[index], texts[index], other_input[index][2]] for index in range(len(texts))]
                请求次数 += 1
                if 请求次数 >= max_retry:
                    Self.日志("log.core.translator.generate.error", e=eb.format_exc(), output=消息结果, info_level=2)
                    return 返回结果
                else:
                    Self.日志("log.core.translator.generate.retry", e=eb.format_exc(), output=消息结果, info_level=1)
                    time.sleep((retry_coef ** (请求次数 - 1)) * retry_time)
    def 构建索引(Self, 向量文件, 模式=None):
        Self.日志("log.core.index.generate.start", info_level=0)
        向量文件 = 向量文件.get() if GPU_ACC else 向量文件
        训练, 构建 = False, False
        量化映射 = {
            "Q4": faiss.ScalarQuantizer.QT_4bit,
            "Q6": faiss.ScalarQuantizer.QT_6bit,
            "Q8": faiss.ScalarQuantizer.QT_8bit,
            "F16": faiss.ScalarQuantizer.QT_fp16,
            "BF16": faiss.ScalarQuantizer.QT_bf16,
        }
        量化类型 = 量化映射[Self.Config.INDEX_SQ]
        if not 模式:
            模式 = Self.Config.INDEX_MODE
        if 模式 == "HNSWSQ":
            向量索引 = faiss.IndexHNSWSQ(向量文件.shape[1], 量化类型, Self.Config.INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH
            训练, 构建, SQ = True, True, True
        elif 模式 == "HNSWPQ":
            向量索引 = faiss.IndexHNSWPQ(向量文件.shape[1], Self.Config.INDEX_HNSW_M, Self.Config.INDEX_PQ_M, Self.Config.INDEX_NLITS)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH
            训练, 构建, SQ = True, True, False
        elif 模式 == "HNSWFlat":
            向量索引 = faiss.IndexHNSWFlat(向量文件.shape[1], Self.Config.INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH
            训练, 构建, SQ = True, True, False
        elif 模式 == "RefineFlat":
            向量索引 = faiss.IndexRefineFlat(faiss.IndexFlatL2(向量文件.shape[1]))
            向量索引.k_factor = Self.Config.INDEX_REFINEFLAT_K_FACTOR
            训练, 构建, SQ = False, True, False
        elif 模式 == "IVFSQ":
            向量索引 = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatL2(向量文件.shape[1]), 向量文件.shape[1], Self.Config.INDEX_NLIST, 量化类型)
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE
            训练, 构建, SQ = True, True, True
        elif 模式 == "IVFPQ":
            向量索引 = faiss.IndexIVFPQ(faiss.IndexFlatL2(向量文件.shape[1]), 向量文件.shape[1], Self.Config.INDEX_NLIST, Self.Config.INDEX_PQ_M, Self.Config.INDEX_NLITS)
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE
            训练, 构建, SQ = True, True, False
        elif 模式 == "IVFFlat":
            向量索引 = faiss.IndexIVFFlat(faiss.IndexFlatL2(向量文件.shape[1]), 向量文件.shape[1], Self.Config.INDEX_IVF_NLIST, faiss.METRIC_L2)
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE
            训练, 构建, SQ = True, True, False
        elif 模式 == "FlatL2":
            向量索引 = faiss.IndexFlatL2(向量文件.shape[1])
            训练, 构建, SQ = False, True, False
        elif 模式 == "FlatIP":
            向量索引 = faiss.IndexFlatIP(向量文件.shape[1])
            训练, 构建, SQ = False, True, False
        if SQ:
            if Self.Config.INDEX_RE_MINMAX:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_minmax
                向量索引.sq.rangestat_arg = Self.Config.INDEX_RE_MINMAX
            if Self.Config.INDEX_RE_MEANSTD:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_meanstd
                向量索引.sq.rangestat_arg = Self.Config.INDEX_RE_MEANSTD
            if Self.Config.INDEX_RE_QUANTILES:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_quantiles
                向量索引.sq.rangestat_arg = Self.Config.INDEX_RE_QUANTILES
            if Self.Config.INDEX_RE_OPTIM:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_optim
                向量索引.sq.rangestat_arg = Self.Config.INDEX_RE_OPTIM
        if 训练:
            for _ in Self.tqdm(range(1), desc="tqdm.index.train"):
                向量索引.train(向量文件)
        if 构建:
            for _ in Self.tqdm(range(1), desc="tqdm.index.build"):
                向量索引.add(向量文件)
        Self.日志("log.core.index.generate.end", info_level=0)
        return 向量索引
    def 缓存索引(Self, 向量文件, 文本文件, 模式=None, 存储=True):
        return TranslatorPersistence.缓存索引(Self=Self, 向量文件=向量文件, 文本文件=文本文件, 模式=模式, 存储=存储)
    def 任务分配器(Self, 总数: int):
        层级列表 = Self.Config.get_active_tiers()
        if not 层级列表:
            return [(None, 1.0)]
        有效层级 = [t for t in 层级列表 if 总数 >= t.get("min_count", 0)]
        if not 有效层级:
            有效层级 = [层级列表[0]]
        if Self.Config.LLM_TIER_MULTI_OVERLAP:
            权重列表 = [t.get("weight", 1.0) for t in 有效层级]
            总权重 = sum(权重列表)
            return [(t, w / 总权重) for t, w in zip(有效层级, 权重列表)]
        if Self.Config.LLM_TIER_CASCADE:
            r = Self.Config.LLM_TIER_CASCADE_RATIO
            分配结果 = []
            剩余比例 = 1.0
            for i in range(len(有效层级)-1, 0, -1):
                层级 = 有效层级[i]
                if i == len(有效层级)-1:
                    比例 = r
                else:
                    比例 = 剩余比例 * r
                分配结果.append((层级, 比例))
                剩余比例 -= 比例
            分配结果.append((有效层级[0], 剩余比例))
            分配结果.reverse()
            return 分配结果
        选定层级 = 有效层级[-1]
        if not Self.Config.LLM_TIER_OVERLAP:
            return [(选定层级, 1.0)]
        索引值 = 层级列表.index(选定层级)
        if 索引值 == 0:
            return [(选定层级, 1.0)]
        上一级 = 层级列表[索引值 - 1]
        低比例, 高比例 = Self.Config.LLM_TIER_OVERLAP_RATIO
        return [(上一级, 低比例), (选定层级, 高比例)]
    def 翻译语言列表(Self, texts: list, 翻译参考列表: list, 使用模型: list=None) -> list:
        输入列表 = []
        返回列表 = []
        命中缓存 = []
        翻译缓存输入 = []
        返回请求内容 = []
        返回其他内容 = []
        完整返回列表 = []
        QuestsMode = False
        if texts == []:
            return []
        try:
            if isinstance(texts[0][0], list):
                QuestsMode = True
        except: pass
        texts = [texts[index] for index in range(len(texts)) if not bool(Self.正则表达式预编译.翻译剔除方法.match(texts[index][1]))] if QuestsMode else texts
        texts = [index for index in texts if not f"{index[0]}" == f"{index[1]}"]
        输入复制 = texts.copy()
        if Self.Config.TRANSLATOR_CACHE_READ:
            翻译缓存 = Self.Module.翻译缓存()
            原始长度 = len(texts)
            待翻译 = []
            for item in Self.tqdm(texts, desc="tqdm.translator.cache.use"):
                if item[1] in 翻译缓存:
                    命中缓存.append([item[0], item[1], 翻译缓存[item[1]], item[2]])
                else:
                    待翻译.append(item)
            texts[:] = 待翻译
            成功缓存 = len(命中缓存)
            命中率 = (成功缓存 / 原始长度) if 原始长度 > 0 else 0.0
            Self.日志("log.core.translator.cache.hit", hit=f"{命中率:.4%}", info_level=0)
        for index in texts:
            try:
                输入列表.append([index[1], index[0], index[2]])
            except Exception:
                Self.日志("log.core.parsing.parameters.error", e=eb.format_exc(), index=index, info_level=2)
                pass
        try:
            if texts:
                向量文件, 文本文件 = Self.参考词预处理()
                索引聚合缓存 = {}
                单词结果暂存 = {}
                if 文本文件:
                    if Self.Config.INDEX_TEXT_K + Self.Config.INDEX_WORD_K + Self.Config.INDEX_LANG_K != 0 and Self.Config.INDEX_TEXT_K != 0:
                        向量索引 = Self.缓存索引(向量文件, 文本文件)
                        文本索引 = {index2[0]: index for index, index2 in enumerate(文本文件)}
                        原文文本文件 = [index[0] for index in 文本文件]
                        Self.日志("log.core.debug.vector.shape", shape=向量文件.shape, info_level=4)
                        Self.日志("log.core.debug.vector.range", range=(向量文件.min(), 向量文件.max()), info_level=4)
                        if Self.Config.INDEX_TEXT_K != 0:
                            Self.日志("log.core.index.search.start", info_level=0)
                            输入列表 = Self.并行生成向量(输入列表)
                            向量列表 = np.asarray(输入列表[0], dtype=np.float32)
                            向量列表 = 向量列表.get() if GPU_ACC else 向量列表
                            for _ in Self.tqdm(range(1), desc="tqdm.index.search"):
                                索引结果矩阵 = 向量索引.search(向量列表, Self.Config.INDEX_TEXT_K)[1]
                            if Self.Config.INDEX_LANG_K != 0 and 翻译参考列表 and len(翻译参考列表) >= Self.Config.INDEX_LANG_K:
                                参考字典 = {index0: index2 for index0, index1, index2 in 翻译参考列表}
                                参考文本文件 = [[index0, index2] for index0, index1, index2 in 翻译参考列表]
                                参考原文文本文件 = [index[0] for index in 翻译参考列表]
                                参考输入列表 = Self.并行生成向量(翻译参考列表)
                                参考向量列表 = np.asarray(参考输入列表[0], dtype=np.float32)
                                参考向量列表 = 参考向量列表.get() if GPU_ACC else 参考向量列表
                                参考向量索引 = Self.缓存索引(参考向量列表, 参考文本文件, Self.Config.INDEX_LANG_MODE, False)
                                for _ in Self.tqdm(range(1), desc="tqdm.index.search"):
                                    参考索引结果矩阵 = 参考向量索引.search(向量列表, Self.Config.INDEX_TEXT_K)[1]
                            for index in range(len(向量列表)):
                                键 = 输入列表[1][1][index]
                                文件路径 = 输入列表[1][2][index]
                                其他内容 = [键, ["None", []], 文件路径] if QuestsMode else [键, [键, []], 文件路径]
                                for index2 in 索引结果矩阵[index]:
                                    原文 = 原文文本文件[index2]
                                    if 原文 in 文本索引:
                                        其他内容[1][1].append(文本文件[文本索引[原文]])
                                if Self.Config.INDEX_LANG_K != 0 and 翻译参考列表 and len(翻译参考列表) >= Self.Config.INDEX_LANG_K:
                                    for index2 in 参考索引结果矩阵[index]:
                                        参考原文 = 参考原文文本文件[index2]
                                        其他内容[1][1].append([参考原文, 参考字典[参考原文]])
                                if 键 not in 索引聚合缓存:
                                    索引聚合缓存[键] = {"请求": 输入列表[1][0][index], "数据": 其他内容}
                                else:
                                    索引聚合缓存[键]["数据"][1][1].extend(其他内容[1][1])
                            Self.日志("log.core.index.search.end", info_level=0)
                        if Self.Config.INDEX_WORD_K != 0:
                            Self.日志("log.core.index.search.start", info_level=0)
                            单词列表 = []
                            for index in texts:
                                单词列表 += index[1].split()
                            单词列表 = Self.Module.列表去重(单词列表)
                            单词列表 = [w for w in 单词列表 if len(w.strip()) > 1]
                            单词列表 = [w for w in 单词列表 if w.lower() not in {w.lower() for w in Self.Config.INDEX_QUESTS_BASIC_WORDS}]
                            单词列表 = [[index, "", ""] for index in 单词列表]
                            单词输入列表 = Self.并行生成向量(单词列表)
                            单词向量列表 = np.asarray(单词输入列表[0], dtype=np.float32)
                            单词向量列表 = 单词向量列表.get() if GPU_ACC else 单词向量列表
                            for _ in Self.tqdm(range(1), desc="tqdm.index.search"):
                                单词索引结果矩阵 = 向量索引.search(单词向量列表, Self.Config.INDEX_WORD_K)[1]
                            for index in range(len(单词向量列表)):
                                请求内容 = 单词输入列表[1][0][index]
                                其他内容 = [请求内容, ["None", []], "WordIndex"] if QuestsMode else [请求内容, [请求内容, []], "WordIndex"]
                                for index2 in 单词索引结果矩阵[index]:
                                    原文 = 原文文本文件[index2]
                                    if 原文 in 文本索引:
                                        其他内容[1][1].append(文本文件[文本索引[原文]])
                                单词结果暂存[请求内容] = 其他内容[1][1]
                            if 单词结果暂存 and 索引聚合缓存:
                                for 缓存键, 缓存数据 in 索引聚合缓存.items():
                                    查询文本 = 缓存数据["请求"]
                                    for 分词 in 查询文本.split():
                                        if 分词 in 单词结果暂存:
                                            缓存数据["数据"][1][1].extend(单词结果暂存[分词])
                            Self.日志("log.core.index.search.end", info_level=0)
                        for 缓存项 in 索引聚合缓存.values():
                            返回请求内容.append(缓存项["请求"])
                            返回其他内容.append(缓存项["数据"])
                    else:
                        返回请求内容 = [row[1] for row in texts]
                        返回其他内容 = [[row[0], ["None", []], row[2]] for row in texts] if QuestsMode else [[row[0], [row[0], []], row[2]] for row in texts]
                else:
                    返回请求内容 = [row[1] for row in texts]
                    返回其他内容 = [[row[0], ["None", []], row[2]] for row in texts] if QuestsMode else [[row[0], [row[0], []], row[2]] for row in texts]
                处理后的请求内容 = []
                处理后的其他内容 = []
                额外列表 = defaultdict(list)
                def 深度优先搜索(组件, 当前路径):
                    if isinstance(组件, str):
                        提取记录.append((当前路径, 组件))
                        return
                    if isinstance(组件, dict):
                        if "text" in 组件:
                            提取记录.append((当前路径 + ["text"], 组件["text"]))
                        if "extra" in 组件 and isinstance(组件["extra"], list):
                            for i, 子组件 in enumerate(组件["extra"]):
                                深度优先搜索(子组件, 当前路径 + ["extra", i])
                        return
                    if isinstance(组件, list):
                        for i, 项目 in enumerate(组件):
                            深度优先搜索(项目, 当前路径 + [i])
                        return
                for index0, index1 in zip(返回请求内容, 返回其他内容):
                    try:
                        解析数据 = ast.literal_eval(index0)
                        Self.owolib解析缓存[index0] = 解析数据
                        提取记录 = []
                        深度优先搜索(解析数据, [])
                        额外列表[(index1[0], index0)] = []
                        for 路径, 文本 in 提取记录:
                            处理后的请求内容.append(文本)
                            其他内容值 = index1.copy()
                            路径键 = "|".join(str(p) for p in 路径)
                            其他内容值[0] = f"{index1[0]}{路径键}"
                            处理后的其他内容.append(其他内容值)
                            额外列表[(index1[0], index0)].append((路径, 文本))
                    except Exception:
                        处理后的请求内容.append(index0)
                        处理后的其他内容.append(index1)
                        
                原始请求内容 = 处理后的请求内容
                原始其他内容 = 处理后的其他内容
                原文到索引映射 = defaultdict(list)
                原文到键列表映射 = defaultdict(list)
                for idx, 原文 in enumerate(原始请求内容):
                    标识符 = 原始其他内容[idx][0]
                    原文到索引映射[原文].append(idx)
                    原文到键列表映射[原文].append(标识符)
                唯一请求内容 = []
                唯一其他内容 = []
                for 原文, 索引列表 in 原文到索引映射.items():
                    唯一请求内容.append(原文)
                    基础条目 = 原始其他内容[索引列表[0]]
                    所有键列表 = 原文到键列表映射[原文]
                    唯一其他内容.append(基础条目 + [所有键列表])
                处理后的请求内容 = 唯一请求内容
                处理后的其他内容 = 唯一其他内容
                上下文初始化字典 = defaultdict(list)
                
                for index0, index1 in zip(处理后的请求内容, 处理后的其他内容):
                    上下文初始化字典[index1[0]] = [index0, ""]
                上下文管理器 = 翻译上下文管理器(上下文初始化字典)
                总条目数 = len(处理后的请求内容)
                分配结果 = Self.任务分配器(总条目数)
                使用模型.append([层级配置.get('model', "") for 层级配置, 比例 in 分配结果 if 比例 > 0])
                任务列表 = []
                if Self.Config.LLM_TIER_INTERLEAVE:
                    条目计数 = []
                    剩余 = 总条目数
                    for 索引, (层级配置, 比例) in enumerate(分配结果):
                        if 索引 == len(分配结果) - 1:
                            条目计数.append(剩余)
                        else:
                            条数 = int(总条目数 * 比例)
                            条目计数.append(条数)
                            剩余 -= 条数
                    权重 = [w for w in 条目计数]
                    总权重 = sum(权重)
                    误差 = [0.0] * len(权重)
                    条目归属 = [0] * 总条目数
                    for i in range(总条目数):
                        最佳 = 0
                        for j in range(1, len(权重)):
                            if 误差[j] < 误差[最佳]:
                                最佳 = j
                        条目归属[i] = 最佳
                        误差[最佳] += 权重[最佳] / 总权重
                    索引 = 0
                    while 索引 < 总条目数:
                        当前层 = 条目归属[索引]
                        批次请求 = []
                        批次其他 = []
                        while 索引 < 总条目数 and len(批次请求) < Self.Config.TRANSLATOR_BATCH and 条目归属[索引] == 当前层:
                            批次请求.append(处理后的请求内容[索引])
                            批次其他.append(处理后的其他内容[索引])
                            索引 += 1
                        if 批次请求:
                            任务列表.append((批次请求, 批次其他, 分配结果[当前层][0]))
                else:
                    起始索引 = 0
                    for 索引, (层级配置, 比例) in enumerate(分配结果):
                        if 索引 == len(分配结果) - 1:
                            条目数量 = 总条目数 - 起始索引
                        else:
                            条目数量 = int(总条目数 * 比例)
                        结束索引 = 起始索引 + 条目数量
                        子请求 = 处理后的请求内容[起始索引:结束索引]
                        子其他 = 处理后的其他内容[起始索引:结束索引]
                        for 步进 in range(0, len(子请求), Self.Config.TRANSLATOR_BATCH):
                            任务列表.append((子请求[步进:步进 + Self.Config.TRANSLATOR_BATCH], 子其他[步进:步进 + Self.Config.TRANSLATOR_BATCH], 层级配置))
                        起始索引 = 结束索引
                Self.日志("log.core.translator.generate.start", item=len(任务列表), info_level=0)
                if Self.Config.LLM_TIER_INTERLEAVE:
                    with ThreadPoolExecutor(max_workers=Self.Config.LLM_MAX_WORKERS) as 执行器:
                        未来任务映射 = {执行器.submit(Self.生成翻译, texts=任务[0], other_input=任务[1], tier_cfg=任务[2], context=上下文管理器): 任务 for 任务 in 任务列表}
                        总进度 = Self.tqdm(total=总条目数, desc="tqdm.translator.generate")
                        for 单个任务 in as_completed(未来任务映射):
                            result = 单个任务.result()
                            返回列表.extend(result)
                            总进度.update(len(result))
                        总进度.close()
                else:
                    层级任务映射 = defaultdict(list)
                    for 任务 in 任务列表:
                        cfg = 任务[2]
                        层级_id = cfg.get("id", -1) if cfg else -1
                        层级任务映射[层级_id].append(任务)
                    总进度 = Self.tqdm(total=总条目数, desc="tqdm.translator.generate")
                    进度锁 = threading.Lock()
                    结果总收集 = []
                    层级线程列表 = []
                    def 执行层级任务(层级_id, 任务组):
                        tier_cfg = 任务组[0][2]
                        if tier_cfg is None:
                            最大并发 = Self.Config.LLM_MAX_WORKERS
                        else:
                            最大并发 = tier_cfg.get("max_workers", Self.Config.LLM_MAX_WORKERS)
                        层级返回 = []
                        with ThreadPoolExecutor(max_workers=最大并发) as 执行器:
                            未来任务映射 = {
                                执行器.submit(Self.生成翻译, texts=t[0], other_input=t[1], tier_cfg=t[2], context=上下文管理器): t
                                for t in 任务组
                            }
                            for 单个任务 in as_completed(未来任务映射):
                                result = 单个任务.result()
                                层级返回.extend(result)
                                with 进度锁:
                                    总进度.update(len(result))
                        return 层级返回
                    for 层级_id, 任务组 in 层级任务映射.items():
                        t = threading.Thread(target=lambda q, lid=层级_id, tasks=任务组: q.append(执行层级任务(lid, tasks)), args=(结果总收集,))
                        t.start()
                        层级线程列表.append(t)
                    for t in 层级线程列表:
                        t.join()
                    总进度.close()
                    for 层级结果 in 结果总收集:
                        返回列表.extend(层级结果)
                        
                唯一结果映射 = {}
                for res in 返回列表:
                    a, b, c, d = res
                    唯一结果映射[b] = (c, d)
                展开返回列表 = []
                for idx in range(len(原始请求内容)):
                    原文 = 原始请求内容[idx]
                    if 原文 in 唯一结果映射:
                        译文, 其他 = 唯一结果映射[原文]
                        新条目 = [原始其他内容[idx][0], 原文, 译文, 原始其他内容[idx][2]]
                    else:
                        新条目 = [原始其他内容[idx][0], 原文, 原文, 原始其他内容[idx][2]]
                    展开返回列表.append(新条目)
                返回列表 = 展开返回列表
                
            返回列表.extend(命中缓存)
            if QuestsMode:
                参考字典 = {str(item[0]) for item in 输入复制}
                完整返回列表 = [[a, f"{c}({b})", d] for a, b, c, d in 返回列表 if str(a) in 参考字典] if Self.Config.TRANSLATOR_ORIGINAL_REFERENCE else [[a, c, d] for a, b, c, d in 返回列表 if str(a) in 参考字典]
                翻译缓存输入 = [[b, c] for a, b, c, d in 返回列表]
            else:
                返回列表 = {a: [b, c, d] for a, b, c, d in 返回列表}
                for 原始条目 in 输入复制:
                    try:
                        解析数据 = Self.owolib解析缓存[原始条目[1]]
                        记录列表 = 额外列表.get((原始条目[0], 原始条目[1]), [])
                        for 路径, _ in 记录列表:
                            路径键 = "|".join(str(p) for p in 路径)
                            完整键 = f"{原始条目[0]}{路径键}"
                            翻译项 = 返回列表.get(完整键)
                            if 翻译项 and 翻译项[1]:
                                目标对象 = 解析数据
                                for 键 in 路径[:-1]:
                                    目标对象 = 目标对象[键]
                                目标对象[路径[-1]] = f"{翻译项[1]}({翻译项[0]})" if Self.Config.TRANSLATOR_ORIGINAL_REFERENCE else 翻译项[1]
                        完整返回列表.append([原始条目[0], json.dumps(解析数据, ensure_ascii=False), 原始条目[2]])
                    except Exception:
                        基础键 = 原始条目[0]
                        翻译项 = 返回列表[基础键]
                        翻译缓存输入.append([翻译项[0], 翻译项[1]])
                        if Self.Config.TRANSLATOR_ORIGINAL_REFERENCE:
                            完整返回列表.append([基础键, f"{翻译项[1]}({翻译项[0]})", 原始条目[2]])
                        else:
                            完整返回列表.append([基础键, 翻译项[1], 原始条目[2]])
            if Self.Config.TRANSLATOR_CACHE_WRITE:
                Self.Module.翻译缓存(翻译缓存输入)
        except Exception:
            Self.日志("log.core.translator.error", e=eb.format_exc(), texts=texts, info_level=3)
            raise
        return 完整返回列表
    def 翻译语言文件(Self, file0: str,  file1: str = "", output_path: str = "", export_inspection: bool = False, output_lang_str: bool = False, read_error: bool = True):
        output_path = Self.Module.输出路径处理(output_path)
        未翻译列表 = []
        去翻译列表 = []
        输出列表 = []
        参考字典 = {}
        翻译参考列表 = []
        可翻译源文件, 源文件, 参考文件, 压缩路径, 输出扩展名, file2 = Self.Module.读取资源文件(file0, file1, read_error)
        if 参考文件:
            for item in 参考文件:
                try:
                    参考字典[item[0]] = item[1]
                except Exception:
                    Self.日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
                    pass
            for entry in 可翻译源文件:
                key = entry[0]
                path = entry[2]
                if key in 参考字典:
                    去翻译列表.append([key, 参考字典[key], path])
                    翻译参考列表.append([entry[1], key, 参考字典[key]])
                else:
                    未翻译列表.append(entry)
        else:
            未翻译列表 = 可翻译源文件.copy()
        使用模型 = []
        翻译列表 = Self.翻译语言列表(未翻译列表, 翻译参考列表, 使用模型) #翻译核心
        if export_inspection:
            未翻译列表字典 = {index[0]: index[1] for index in 未翻译列表}
            for index in Self.tqdm(翻译列表, desc="tqdm.progress.encoding"):
                行数据 = {index[0]: [index[1], 未翻译列表字典[index[0]]]}
                输出列表.append(repr(行数据))
            with open(str(Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translang")), 'w+', encoding='utf-8') as f:
                f.write("\n".join(输出列表))
            Self.日志("log.core.translator.succeed", path=Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translang").resolve(), info_level=0)
            return Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translang")
        else:
            所有翻译结果 = 翻译列表 + 去翻译列表
            分组 = defaultdict(list)
            for a, b, c in 所有翻译结果:
                分组[c].append([a, b])
            所有翻译结果 = dict(分组)
            输出列表 = []
            for index in 源文件:
                翻译输出列表 = []
                for index1 in index[0]:
                    if index1.strip().startswith(('#', '//')):
                        翻译输出列表.append(index1)
                    else:
                        索引成功 = False
                        for index2 in 所有翻译结果[index[1]]:
                            if index1.split('=', 1)[0] == index2[0]:
                                翻译输出列表.append(f"{index2[0]}={index2[1]}")
                                索引成功 = True
                                break
                        if not 索引成功:
                            翻译输出列表.append(index1)
                输出列表.append([index[1], 翻译输出列表])
            if 压缩路径 and (not output_lang_str):
                for index in 输出列表:
                    Self.Module.保存语言文件(f"{Path(index[0]).parent}/{Self.Config.LANGUAGE_OUTPUT}{Path(index[0]).suffix}", index[1])
                压缩文件夹Path = Path(压缩路径)
                if file2[0] == False:
                    文档内容 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(name=Path(file0).stem, lang=Self.Config.LANGUAGE_OUTPUT, model=", ".join(使用模型[0]) or Self.Config.LLM_MODEL or Self.Lang("log.core.package.zip.hit"), author=Self.Config.PACK_AUTHOR or "海盐青茫")
                    with open(压缩文件夹Path/"pack.mcmeta", "w+", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "pack": {
                                "description": 文档内容,
                                "pack_format": 9999,
                                "supported_formats": [0, 9999],
                                "min_format": 0,
                                "max_format": 9999
                            }
                        }, ensure_ascii=False, indent=4))
                with zipfile.ZipFile(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                    for 压缩文件 in 压缩文件夹Path.rglob('*'):
                        if 压缩文件.is_file():
                            f.write(压缩文件, arcname=压缩文件.relative_to(压缩文件夹Path))
                Self.日志("log.core.translator.succeed", path=Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve(), info_level=0)
                return Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip")
            else:
                if not Path(output_path).suffix:
                    output_path = str(Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}"))
                Self.Module.保存语言文件(output_path, 翻译输出列表)
                Self.日志("log.core.translator.succeed", path=Path(output_path).resolve(), info_level=0)
                return Path(f"{output_path}")
    def 翻译FTB任务(Self, path: str):
        翻译列表 = []
        snbt文件 = [str(index) for index in Path(path).rglob("*.snbt")]
        Self.日志("log.core.file.quests.read.start", info_level=0)
        with ThreadPoolExecutor(max_workers=Self.Config.QUESTS_FTB_READ_MAX_CONCURRENT) as 解释器:
            任务结果 = 解释器.map(Self.Module.读取单个FTBQ_Snbt文件, snbt文件)
            for 单个任务 in Self.tqdm(任务结果, total=len(snbt文件), desc="tqdm.file.read"):
                翻译列表.extend(单个任务)
        Self.日志("log.core.file.quests.read.end", info_level=0)
        翻译列表2 = []
        try:
            for index in 翻译列表:
                if index[1] and not (re.match(r'^[a-z0-9._-]+$', index[1]) and '.' in index[1]):
                    翻译列表2.append(index)
        except Exception: 
            Self.日志("log.module.quests.clean.error", index=index, info_level=2)
        翻译列表2 = [[index[0], index[1], ""] for index in 翻译列表2]
        翻译列表 = Self.翻译语言列表(翻译列表2)
        分组 = defaultdict(list)
        for item in 翻译列表:
            key = item[0][0]
            分组[key].append(item)
        翻译列表 = [分组[k] for k in sorted(分组.keys())]
        mode = "H" if os.path.isdir(os.path.join(path, "quests")) else "L"
        with ThreadPoolExecutor(max_workers=Self.Config.QUESTS_FTB_WRITE_MAX_CONCURRENT) as 解释器:
            任务结果 = 解释器.map(partial(Self.Module.应用FTBQ翻译, mode=mode), 翻译列表)
            for 单个任务 in Self.tqdm(任务结果, total=len(翻译列表), desc="tqdm.translator.use"): 
                pass
        Self.日志("log.core.translator.succeed", path=Path(path).resolve(), info_level=0)
    def 翻译BQ任务(Self, path: str):
        翻译列表 = []
        nbt文件 = [str(index) for index in Path(path).rglob("*.json")]
        Self.日志("log.core.file.quests.read.start", info_level=0)
        with ThreadPoolExecutor(max_workers=Self.Config.QUESTS_BQ_READ_MAX_CONCURRENT) as 解释器:
            任务结果 = 解释器.map(Self.Module.读取单个BQ_Json文件, nbt文件)
            for 单个任务 in Self.tqdm(任务结果, total=len(nbt文件), desc="tqdm.file.read"):
                翻译列表.extend(单个任务)
        Self.日志("log.core.file.quests.read.end", info_level=0)
        翻译列表2 = []
        try:
            for index in 翻译列表:
                if index[1] and not (re.match(r'^[a-z0-9._-]+$', index[1]) and '.' in index[1]):
                    翻译列表2.append(index)
        except Exception: 
            Self.日志("log.module.quests.clean.error", index=index, info_level=2)
        翻译列表2 = [[index[0], index[1], ""] for index in 翻译列表2]
        翻译列表 = Self.翻译语言列表(翻译列表2)
        分组 = defaultdict(list)
        for item in 翻译列表:
            key = item[0][0]
            分组[key].append(item)
        翻译列表 = [分组[k] for k in sorted(分组.keys())]
        with ThreadPoolExecutor(max_workers=Self.Config.QUESTS_BQ_WRITE_MAX_CONCURRENT) as 解释器:
            任务结果 = 解释器.map(partial(Self.Module.应用BQ翻译), 翻译列表)
            for 单个任务 in Self.tqdm(任务结果, total=len(翻译列表), desc="tqdm.translator.use"): 
                pass
        Self.日志("log.core.translator.succeed", path=Path(path).resolve(), info_level=0)
    def 导入DictMini参考词(Self, file: str = None, mode: str = "dense", max_len: int = 80):
        Self.日志("log.core.file.settle.start", info_level=0)
        待处理列表 = []
        if Path(file).suffix == ".json":
            for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
                with open(file, "rb") as f:
                    Dict文件 = json.load(f)
            if mode == "dense":
                for index in Self.tqdm(Dict文件, desc="tqdm.file.processing"):
                    待处理列表.append([index, ", ".join([str(item) for item in Dict文件[index] if len(str(item)) <= max_len]), ""])
            elif mode == "sparse":
                for index in Self.tqdm(Dict文件, desc="tqdm.file.processing"):
                    for index1 in Dict文件[index]:
                        if len(index1) <= max_len:
                            待处理列表.append([index, index1, ""])
        elif file == None:
            for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
                待处理列表 = [[k, v, ""] for k, v in Self.Module.翻译缓存()]
        Self.参考词预处理(待处理列表)
        Self.日志("log.core.file.settle.end", info_level=0)
    def 选择相似度最高译文(Self, 请求消息: list):
        请求次数 = 0
        if (not Self.Config.RERANKER_API_URL) and (Self.Config.RERANKER_MODEL):
            with Self.线程锁:
                try:
                    相似度 = Self.重排序模型.predict([(请求消息[0], 候选) for 候选 in 请求消息[1]], show_progress_bar=False)
                    return [请求消息[0], 请求消息[1][相似度.argmax()], 相似度]
                except Exception:
                    Self.日志("log.core.translator.cache.locally.error", e=eb.format_exc(), info_level=2)
                    return [请求消息[0], 请求消息[1][0], [0 for _ in range(len(请求消息[1]))]]
        else:
            请求内容 = {
                "model": Self.Config.RERANKER_MODEL,
                "documents": 请求消息[1],
                "query": 请求消息[0],
                "instruct": Self.Config.RERANKER_INSTRUCT
            }
            while 请求次数 < Self.Config.RERANKER_MAX_RETRY:
                try:
                    相似度 = []
                    请求结果 = Self.会话.RERANKER.post(url=Self.Config.RERANKER_API_URL, json=请求内容, timeout=(Self.Config.RERANKER_CONN_TIMEOUT, Self.Config.RERANKER_TIMEOUT))
                    请求结果.raise_for_status()
                    请求结果 = 请求结果.json()
                    请求结果 = 请求结果["output"]["results"]
                    for index in 请求结果:
                        相似度.append(index["document"]["text"])
                    return [请求消息[0], 请求结果[0]["document"]["text"], 相似度]
                except Exception:
                    Self.日志("log.core.translator.cache.generate.messages.error", messages=请求消息[1], e=eb.format_exc(), info_level=1)
                    请求次数 += 1
                    if 请求次数 >= Self.Config.RERANKER_MAX_RETRY:
                        Self.日志("log.core.translator.cache.generate.error", e=eb.format_exc(), output=请求结果, info_level=2)
                        return [请求消息[0], 请求消息[1][0], [0 for _ in range(len(请求消息[1]))]]
                    else:
                        Self.日志("log.core.translator.cache.generate.retry", e=eb.format_exc(), output=请求结果, info_level=1)
                        time.sleep((Self.Config.RERANKER_RETRY_COEF ** (请求次数 - 1)) * Self.Config.RERANKER_RETRY_TIME)
    def 获取相似度最高译文(Self, 输入字典: dict, 强制重排: bool=False):
        请求列表 = []
        剔除列表 = []
        返回列表 = []
        for index in 输入字典:
            if len(输入字典[index]) == 1 and 强制重排 == False:
                剔除列表.append([index, 输入字典[index][0], [0]])
            else:
                请求列表.append([index, 输入字典[index]])
        if (not Self.重排序模型) and (not Self.Config.RERANKER_API_URL) and (Self.Config.RERANKER_MODEL) and (请求列表):
            Self.重排序模型 = TranslatorPersistence.获取重排模型(Self=Self)
        Self.日志("log.core.translator.cache.generate.start", item=len(请求列表), info_level=0)
        with ThreadPoolExecutor(max_workers=Self.Config.RERANKER_MAX_WORKERS) as 执行器:
            未来任务映射 = {
                执行器.submit(
                    Self.选择相似度最高译文,
                    请求消息 = index,
                ): index
                for index in 请求列表
            }
            for 单个任务 in Self.tqdm(as_completed(未来任务映射), total=len(未来任务映射), desc="tqdm.translator.cache.generate"):
                返回列表.append(单个任务.result())
        Self.日志("log.core.translator.cache.generate.end", info_level=0)
        返回列表 += 剔除列表
        return 返回列表
        
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
            文本列表 = Self.获取相似度最高译文(Dict文件)
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
            for index in Self.tqdm(I18n缺失模组ID, desc="tqdm.translator.mod"):
                try:
                    保存路径 = Path(f"{缓存路径}/assets/{index[0]}/lang/")
                    保存路径.mkdir(parents=True, exist_ok=True)
                    Self.翻译语言文件(file0=f"{path}/mods/{index[1]}", file1="", output_path=保存路径, output_lang_str=True, read_error=False)
                except FileNotFoundError:
                    Self.日志("log.core.translator.modpack.error.mod", e="", mod=index[0], info_level=0)
                except Exception:
                    Self.日志("log.core.translator.modpack.error.mod", e=eb.format_exc(), mod=index[0], info_level=1)
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
        for index in frozenset(["resources", "kubejs/assets"]):
            文件夹路径 = f"{path}/{index}"
            if Path(文件夹路径).is_dir():
                所有文件夹 = [p.name for p in Path(文件夹路径).iterdir() if p.is_dir()]
                if "nuclearcraft" in frozenset(所有文件夹) and Path(f"{文件夹路径}/nuclearcraft/addons/").is_dir():
                    for index2 in Self.tqdm(Path(f"{文件夹路径}/nuclearcraft/addons/").glob("*.zip"), desc="tqdm.translator.nuclearcraftaddonspack"):
                        Self.翻译语言文件(file0=index2, output_path=f"{文件夹路径}/nuclearcraft/addons/")
                for 文件夹 in Self.tqdm(所有文件夹, desc="tqdm.translator.resource"):
                    无后缀语言文件名 = f"{文件夹路径}/{文件夹}/lang/{Self.Config.LANGUAGE_INPUT}"
                    if Path(f"{无后缀语言文件名}.lang").is_file():
                        Self.翻译语言文件(file0=f"{无后缀语言文件名}.lang", output_path=f"{文件夹路径}/{文件夹}/lang")
                    if Path(f"{无后缀语言文件名}.json").is_file():
                        Self.翻译语言文件(file0=f"{无后缀语言文件名}.json", output_path=f"{文件夹路径}/{文件夹}/lang")
            翻译列表路径[f"/{index}"] = ["path"]
        Self.日志("log.core.translator.succeed", path=Path(f"{path}/resourcepacks/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve(), info_level=0)
        return 翻译列表路径
    def 翻译通用文件(Self, file0, file1 = None, all_mode: bool = False, export_inspection = False):
        file0 = Path(file0).resolve()
        if file1:
            file1 = Path(file1).resolve()
        Self.日志("log.core.translator.general.generate.file.input", file0=file0, file1=file1, info_level=0)
        缓存文件夹 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/"
        Path(缓存文件夹).mkdir(parents=True, exist_ok=True)
        Self.日志("log.core.translator.general.generate.start", info_level=0)
        返回内容 = None
        try:
            if Path(file0).is_file():
                文件0扩展名 = Path(file0).suffix
                if 文件0扩展名 in frozenset([".lang", ".json", ".jar"]):
                    Self.日志("log.core.translator.general.model", model="Mod" if 文件0扩展名 == ".jar" else "Language File", info_level=0)
                    返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                    Self.日志("log.core.translator.succeed", path=Path(返回路径).resolve(), info_level=0)
                    返回内容 = Path(返回路径)
                elif 文件0扩展名 in frozenset([".zip", ".mrpack"]):
                    with zipfile.ZipFile(file0, 'r') as zf:
                        namelist = zf.namelist()
                        def has_dir(prefix: str) -> bool:
                            return any(name.startswith(prefix + '/') or name == prefix for name in namelist)
                        def 是否仅含指定根文件夹(目标文件夹名: str) -> bool:
                            if not namelist:
                                return False
                            根目录集合 = {f.split('/', 1)[0] + '/' for f in namelist if '/' in f}
                            if len(根目录集合) != 1:
                                return False
                            根前缀 = 根目录集合.pop()
                            目标完整前缀 = 根前缀 + 目标文件夹名.rstrip('/') + '/'
                            return any(路径.startswith(目标完整前缀) for 路径 in namelist)
                        if has_dir('shaders'):
                            Self.日志("log.core.translator.general.model", model="Shaders", info_level=0)
                            返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                            Self.日志("log.core.translator.succeed", path=Path(返回路径).resolve(), info_level=0)
                            返回内容 = Path(返回路径)
                        elif has_dir('assets'):
                            Self.日志("log.core.translator.general.model", model="ResourcePacks", info_level=0)
                            返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                            Self.日志("log.core.translator.succeed", path=Path(返回路径).resolve(), info_level=0)
                            返回内容 = Path(返回路径)
                        elif has_dir('ftbquests'):
                            Self.日志("log.core.translator.general.model", model="FTBQuests", info_level=0)
                            zf.extractall(缓存文件夹)
                            Self.翻译FTB任务(f"{缓存文件夹}/ftbquests")
                            with zipfile.ZipFile(f"{缓存文件夹}/FTBQuests-Translation.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                                for 压缩文件 in Path(f"{缓存文件夹}/ftbquests").rglob('*'):
                                    if 压缩文件.is_file():
                                        f.write(压缩文件, arcname=压缩文件.relative_to(str(缓存文件夹)))
                            返回内容 = Path(f"{缓存文件夹}/FTBQuests-Translation.zip")
                        elif has_dir('betterquesting'):
                            Self.日志("log.core.translator.general.model", model="BetterQuesting", info_level=0)
                            zf.extractall(缓存文件夹)
                            Self.翻译BQ任务(f"{缓存文件夹}/betterquesting")
                            with zipfile.ZipFile(f"{缓存文件夹}/BetterQuesting-Translation.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                                for 压缩文件 in Path(f"{缓存文件夹}/betterquesting").rglob('*'):
                                    if 压缩文件.is_file():
                                        f.write(压缩文件, arcname=压缩文件.relative_to(str(缓存文件夹)))
                            返回内容 = Path(f"{缓存文件夹}/BetterQuesting-Translation.zip")
                        elif 是否仅含指定根文件夹("contenttweaker"):
                            Self.日志("log.core.translator.general.model", model="NuclearCraft: Overhauled Addons Pack", info_level=0)
                            返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                            Self.日志("log.core.translator.succeed", path=Path(返回路径).resolve(), info_level=0)
                            返回内容 = Path(返回路径)
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
                                    Self.日志("log.core.translator.general.model", model=整合包模式, info_level=0)
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
                                    Self.日志("log.core.translator.succeed", path=Path(输出Zip路径).resolve(), info_level=0)
                                    返回内容 = Path(输出Zip路径)
                                
                                else:
                                    Self.日志("log.core.translator.general.modpack.translate.file.no", info_level=2)
                                    返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
                            else:
                                Self.日志("log.core.translator.general.structure.unknown", info_level=3)
                                返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
                else:
                    Self.日志("log.core.translator.general.structure.unknown", info_level=3)
                    返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
            elif Path(file0).is_dir():
                文件夹名称 = Path(file0).name
                if 文件夹名称 == "ftbquests":
                    Self.日志("log.core.translator.general.model", model="FTBQuests", info_level=0)
                    Self.翻译FTB任务(path=file0)
                elif 文件夹名称 == "betterquesting":
                    Self.日志("log.core.translator.general.model", model="BetterQuesting", info_level=0)
                    Self.翻译BQ任务(path=file0)
                else:
                    Self.日志("log.core.translator.general.model", model="General ModPack", info_level=0)
                    Self.翻译整合包(path=file0, all_mode=all_mode)
                返回内容 = Path(file0)
        except Exception:
            Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=3)
            返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
        Self.日志("log.core.translator.succeed", path=返回内容.resolve(), info_level=0)
        return 返回内容.resolve()
    def 检索缓存(Self, text: str):
        return Self.Module.翻译缓存()[0][text]
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
                            符合条件的文件夹.append((子目录,真实文件名映射[输入语言文件名],真实文件名映射[输出语言文件名]))
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
        
        
测试 = True
if __name__ == "__main__" and 测试:
    参数 = {
        "LLM1_API_URL": "https://api.deepseek.com/chat/completions",
        "LLM1_API_KEY": "sk-51b8fc3395e24edeb7ab485ff94e1bd5",
        "LLM1_MODEL": "deepseek-v4-flash",
        "LLM1_API_KWARGS": {"extra_body": {"thinking": {"type": "disabled"}}},
        "LLM1_MAX_WORKERS": 1,
        "LLM0_API_URL": "http://127.0.0.1:25564/v1/chat/completions",
        "LLM0_MODEL": "Gemma4-26B-A4B",
        "LLM_TIER_CASCADE": True,
        "LLM0_MIN_COUNT": 2500,
        "LLM_TIER_CASCADE_RATIO": 0.8,
        "TRANSLATOR_BATCH": 1,
        "LLM_CONTEXTS": 3,
        "EMB_API_URL": "http://127.0.0.1:25564/v1/embeddings",
        "EMB_MODEL": "text-embedding-nomic-embed-text-v1.5",
        "TRANSLATOR_ORIGINAL_REFERENCE": False,
        "LANGUAGE": "zh_CN",
        "TRANSLATOR_CACHE_NAME": "Translator_Cache",
        "VEC_FILE_NAME": "Vectors2",
        "VEC_QUANTIZATION": "Float32",
        "EMB_MAX_WORKERS": 2,
        "DEBUG_MODE": True,
        "TRANSLATOR_CACHE_READ": False,
        "TRANSLATOR_CACHE_WRITE": False,
        "LLM_TIER_INTERLEAVE": False
    }
    翻译 = Translator(参数)
    #翻译2 = Translator(参数)
    #翻译.导入DictMini参考词(r"C:\Users\FengMang\Downloads\Dict-Mini.json")
    #翻译.翻译整合包(r"C:\Users\FengMang\AppData\Roaming\PrismLauncher\instances\Star Technology 翻译测试\minecraft")
    #翻译.翻译FTB任务(r"C:\Users\FengMang\AppData\Roaming\PrismLauncher\instances\Star Technology\minecraft\config\ftbquests")
    #翻译.翻译通用文件(r"C:\Users\FengMang\Downloads\GTConsolidate-1.12.2-1.1.4.1-beta.jar", r"C:\Users\FengMang\Downloads\19293(GTConsolidate-1.12.2-1.1.3.3-beta).zip", export_inspection=True)
    #翻译.合并语言文件更新(r"C:\Users\FengMang\Downloads\GTConsolidate-1.12.2-1.1.4.1-beta.jar", r"C:\Users\FengMang\Desktop\TranslatorMinecraft\Cache\d740bcc6e0b24d5daf558614c3508141\zh_cn.translang", r"C:\Users\FengMang\Downloads\19293(GTConsolidate-1.12.2-1.1.3.3-beta).zip")
    #翻译.翻译通用文件(r"C:\Users\FengMang\Downloads\GTConsolidate-1.12.2-1.1.4.1-beta.jar", export_inspection=True)
    #翻译.MMTQM(r"C:\Users\FengMang\Desktop\TranslatorMinecraft\Cache\16a036b342644beb8f22735ea1d324f7\zh_cn.translang", r"C:\Users\FengMang\Downloads\Dict-Mini.json") #Q3.6 9B
    #翻译.MMTQM(r"C:\Users\FengMang\Desktop\TranslatorMinecraft\Cache\0995a1a52a2540a5aec95684cfa83fea\zh_cn.translang", r"C:\Users\FengMang\Downloads\Dict-Mini.json") #小米研究院
    #翻译.翻译通用文件(r"en_us.json")
    #翻译.分离语言文件更新(r"fzzy_config-0.7.6+1.21+neoforge.jar")
    翻译.翻译通用文件(r"C:\Users\FengMang\Downloads\primitivemobs-1.2.3a.jar")
    #print(翻译.Config.LLM_TOKEN_USAGE)
    #print(翻译.检索缓存('Divides capacity equally among all types. Use it if you increased the "max types" in config.'))
    #翻译.导入DictMini缓存(r"C:\Users\FengMang\Downloads\Dict-Mini.json", mode="rerank")
    #翻译.语言文件对转DictMini(r"C:\Users\FengMang\Downloads\Minecraft-Shaders-zh_CN-Lang-Files-Surisen.zip", r"C:\Users\FengMang\Downloads\Dict-Mini.json")