from TranslatorLib import (eb, threading, Path, pickle, numpy, np, hashlib, faiss, GPU_ACC, aiohttp, requests, HTTPAdapter, Retry, Callable, asyncio, atexit,
                           IndexGSQ, time)

global 向量文本嵌入缓存
模型缓存 = {}
向量文本缓存 = {}
索引缓存 = {}
数据包指令缓存 = {}
会话缓存 = {}
异步会话缓存 = {}
向量文本嵌入缓存: dict = {}
向量文本嵌入频率: dict = {}    # {text: 累计命中次数}
向量文本嵌入代数: dict = {}    # {text: 最后命中轮次}
_缓存保存轮次 = 0
线程锁 = threading.Lock()
索引线程锁 = threading.Lock()
向量线程锁 = threading.Lock()
异步会话锁 = threading.Lock()
向量文本嵌入锁 = threading.Lock()
向量缓存脏标记 = False
向量缓存保存锁 = threading.Lock()
_向量缓存线程锁 = threading.Lock()
_向量缓存刷新线程 = None
_向量缓存刷新停止 = threading.Event()
_向量缓存持有者 = None
翻译文本缓存: dict = {}
翻译缓存脏标记 = False
翻译缓存锁 = threading.Lock()
翻译缓存保存锁 = threading.Lock()
_翻译缓存线程锁 = threading.Lock()
_翻译缓存加载锁 = threading.Lock()
_翻译缓存刷新线程 = None
_翻译缓存刷新停止 = threading.Event()
_翻译缓存持有者 = None
_翻译缓存已加载 = False


def 获取嵌入模型(Self):
    缓存键 = f"{Self.Config.EMB_MODEL}|{Self.Config.EMB_MODEL_ACC_MODE}"
    if 缓存键 in 模型缓存:
        return 模型缓存[缓存键]
    with 线程锁:
        if 缓存键 in 模型缓存:
            return 模型缓存[缓存键]
        try:
            for _ in Self.Locale.Tqdm(range(1), desc=f"tqdm.model.load"):
                传入参数 = dict(Self.Config.EMB_LOADER_KWARGS)
                if Self.Config.EMB_REASONING_FRAME.lower() == "sentencetransformer":
                    from sentence_transformers import SentenceTransformer # type: ignore
                    Self.日志("log.core.debug.load.embedded.model", model=Self.Config.EMB_MODEL, info_level=0)
                    模型参数 = dict(Self.Config.EMB_LOADER_MODEL_KWARGS)
                    if Self.Config.EMB_MODEL_ACC_MODE.lower() == "onnx":
                        if Self.Config.EMB_MODEL_ACC_MODE:
                            模型参数["dtype"] = Self.Config.EMB_MODEL_ACC_MODE
                        模型 = SentenceTransformer(Self.Config.EMB_MODEL, trust_remote_code=True, backend="onnx", model_kwargs=模型参数, **传入参数)
                    elif Self.Config.EMB_MODEL_ACC_MODE.lower() == "openvino":
                        模型 = SentenceTransformer(Self.Config.EMB_MODEL, trust_remote_code=True, backend="openvino", model_kwargs=模型参数, **传入参数)
                    else:
                        if Self.Config.EMB_MODEL_ACC_MODE:
                            模型参数["dtype"] = Self.Config.EMB_MODEL_ACC_MODE
                        模型 = SentenceTransformer(Self.Config.EMB_MODEL, trust_remote_code=True, device=Self.Config.EMB_MODEL_DEVICE, model_kwargs=模型参数, **传入参数)
                elif Self.Config.EMB_REASONING_FRAME.lower() == "fastembed":
                    from fastembed import TextEmbedding # type: ignore
                    if "cuda" in Self.Config.EMB_MODEL_DEVICE.lower():
                        传入参数["providers"] = ["CUDAExecutionProvider"]
                    模型 = TextEmbedding(Self.Config.EMB_MODEL, normalize=Self.Config.EMB_MODEL_NORMALIZE, **传入参数)
            模型缓存[缓存键] = 模型
            Self.日志("log.core.load.embedded.model.succeed", model=Self.Config.EMB_MODEL, info_level=0)
            return 模型
        except Exception:
            Self.日志("log.core.load.embedded.model.error", model=Self.Config.EMB_MODEL, e=eb.format_exc(), info_level=3)
            raise RuntimeError(Self.Lang("log.core.load.embedded.model.error", model=Self.Config.EMB_MODEL, e=eb.format_exc()))
def 获取重排模型(Self):
    缓存键 = f"{Self.Config.RERANKER_MODEL}|{Self.Config.RERANKER_INSTRUCT}"
    if 缓存键 in 模型缓存:
        return 模型缓存[缓存键]
    with 线程锁:
        if 缓存键 in 模型缓存:
            return 模型缓存[缓存键]
        try:
            for _ in Self.Locale.Tqdm(range(1), desc=f"tqdm.model.load"):
                from sentence_transformers import CrossEncoder # type: ignore
                Self.日志("log.core.load.rerank.model.debug", model=Self.Config.RERANKER_MODEL, info_level=0)
                模型参数 = {}
                if Self.Config.RERANKER_MODEL_DEVICE:
                    模型参数["device"] = Self.Config.RERANKER_MODEL_DEVICE
                if Self.Config.RERANKER_INSTRUCT:
                    模型参数["prompts"] = {"classification": Self.Config.RERANKER_INSTRUCT}
                    模型参数["default_prompt_name"] = "classification"
                模型 = CrossEncoder(Self.Config.RERANKER_MODEL, trust_remote_code=True, **模型参数)
            模型缓存[缓存键] = 模型
            Self.日志("log.core.load.rerank.model.succeed", model=Self.Config.RERANKER_MODEL, info_level=0)
            return 模型
        except Exception:
            Self.日志("log.core.load.rerank.model.error", model=Self.Config.RERANKER_MODEL, e=eb.format_exc(), info_level=3)
            raise RuntimeError(Self.Lang("log.core.load.rerank.model.error", model=Self.Config.RERANKER_MODEL, e=eb.format_exc()))
class 参考词预处理向量懒加载:
    def __init__(Self, 编码数据: dict, 解码函数: Callable, VEC_READ_CACHE: bool):
        Self._编码数据 = 编码数据
        Self._解码函数 = 解码函数
        Self._解码结果 = None
        Self.VEC_READ_CACHE = VEC_READ_CACHE
        Self.PCA_M = 编码数据.get("PCA_M", None)
        Self.PCA_P = 编码数据.get("PCA_P", None)
    def get(Self) -> np.ndarray:
        if Self._解码结果 is not None:
            return Self._解码结果
        解码结果 = Self._解码函数(Self._编码数据)
        if Self.VEC_READ_CACHE:
            Self._解码结果 = 解码结果
            Self._编码数据 = None
            Self._解码函数 = None
        return 解码结果
    def __getstate__(Self):
        if Self._解码结果 is not None:
            return {"_编码数据": None, "_解码结果": Self._解码结果}
        else:
            return {"_编码数据": Self._编码数据, "_解码结果": None}
    def __setstate__(Self, state):
        Self._编码数据 = state["_编码数据"]
        Self._解码结果 = state["_解码结果"]
        Self._解码函数 = None
def 参考词预处理(Self, texts: list = None, uuid = None, use_cache: bool = True) -> 参考词预处理向量懒加载|list: #Core
    检索词, 待处理文本 = [], []
    PCA均值, PCA投影矩阵 = None, None
    文件路径 = Self.Config.VEC_FILE_PATH
    文件名 = uuid if uuid else Self.Config.VEC_FILE_NAME
    缓存键 = f"{文件路径}/{文件名}"
    if texts:
        if use_cache and Path(f"{文件路径}/{文件名}.pkl").is_file():
            with 向量线程锁:
                with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                    检索词 = [item[0] for item in pickle.load(f)]
        检索词_set = set(检索词)
        待处理文本 = [index for index in texts if index[0] not in 检索词_set]
    elif 缓存键 in 向量文本缓存:
        return 向量文本缓存[缓存键][0], 向量文本缓存[缓存键][1]
    if (not 待处理文本) and texts and (not use_cache): 待处理文本 = texts
    Self.日志("log.core.vector.cache.start")
    if 待处理文本 and Self.Config.EMB_MODEL:
        返回内容向量 = Self.Builder.并行生成向量(待处理文本, use_cache=use_cache)
        向量结果列表 = 返回内容向量[0]
        if Self.Config.VEC_PCA_DIM != -1:
            向量结果列表, PCA均值, PCA投影矩阵 = Self.Quantization.PCA降维(向量结果列表)
        Self.日志("log.core.debug.vector.range", range=(向量结果列表.min(), 向量结果列表.max()), info_level=4)
        文本结果列表 = [[返回内容向量[1][0][i], 返回内容向量[1][1][i]] for i in range(len(返回内容向量[1][0]))]
        if not (Path(f"{文件路径}/{文件名}.npz").is_file() and Path(f"{文件路径}/{文件名}.pkl").is_file()):
            if Self.Config.VEC_RERANKER:
                向量结果列表, 文本结果列表 = Self.Quantization.向量重排(向量结果列表, 文本结果列表)
        with 向量线程锁:
            for _ in Self.tqdm(range(1), desc="tqdm.vectors.write"):
                if Path(f"{文件路径}/{文件名}.npz").is_file() and Path(f"{文件路径}/{文件名}.pkl").is_file():
                    旧向量文件 = numpy.load(f"{文件路径}/{文件名}.npz", allow_pickle=True)
                    旧向量文件 = {key: np.asarray(旧向量文件[key]) for key in 旧向量文件.files}
                    with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                        文本文件 = pickle.load(f)
                        
                    if "PCA_M" in 旧向量文件 and "PCA_P" in 旧向量文件:
                        向量结果列表 = Self.Quantization.PCA应用(向量结果列表, 旧向量文件["PCA_M"], 旧向量文件["PCA_P"])
                    文本文件.extend(文本结果列表)
                    向量文件, 文本文件 = Self.Quantization.叠加量化向量(旧向量文件, 向量结果列表, 文本文件)
                    
                    np.savez_compressed(f"{文件路径}/{文件名}.npz", **向量文件)
                    with open(f"{文件路径}/{文件名}.pkl", "wb") as f:
                        pickle.dump(文本文件, f)
                else:
                    向量文件 = Self.Quantization.编码向量(向量结果列表)
                    if PCA均值 is not None: 向量文件["PCA_M"] = PCA均值
                    if PCA投影矩阵 is not None: 向量文件["PCA_P"] = PCA投影矩阵
                    np.savez_compressed(f"{文件路径}/{文件名}.npz", **向量文件)
                    with open(f"{文件路径}/{文件名}.pkl", "wb") as f:
                        pickle.dump(文本结果列表, f)
                    文本文件 = 文本结果列表
    else:
        try:
            with 向量线程锁:
                for _ in Self.Locale.Tqdm(range(1), desc="tqdm.vectors.read"):
                    向量文件 = numpy.load(f"{文件路径}/{文件名}.npz", allow_pickle=True)
                    向量文件 = {key: np.asarray(向量文件[key]) for key in 向量文件.files}
                    with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                        文本文件 = pickle.load(f)
        except Exception:
            Self.日志("log.core.read.vevtor.error", e=eb.format_exc(), info_level=2)
            向量文件, 文本文件 = False, False
    Self.日志("log.core.vector.cache.end")
    向量文件 = 参考词预处理向量懒加载(向量文件, Self.Quantization.解码向量, Self.Config.VEC_READ_CACHE)
    向量文本缓存[缓存键] = [向量文件, 文本文件]
    return 向量文件, 文本文件

def 缓存索引(Self, 向量文件: 参考词预处理向量懒加载, 文本文件, 模式 = None, 存储 = True): #Core
    索引库 = faiss
    Self.日志("log.core.index.cache.start", info_level=0)
    if not 模式:
        模式 = Self.Config.INDEX_MODE
    if 存储:
        索引库 = IndexGSQ if 模式 == "GSQFast" else faiss
        索引配置 = [getattr(Self.Config, key) for key in Self.Config.INDEX_CONFIG]
        参考词哈希 = hashlib.md5(pickle.dumps((向量文件, 文本文件, 索引配置))).hexdigest()
        if 参考词哈希 in 索引缓存:
            return 索引缓存[参考词哈希]
        with 索引线程锁:
            if Path(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index-md5").is_file():
                with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index-md5", "r") as f:
                    参考词哈希文件 = f.read()
                if 参考词哈希文件 == 参考词哈希:
                    for _ in Self.Locale.Tqdm(range(1), desc="tqdm.index.read"):
                        向量索引 = 索引库.read_index(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index")
                else:
                    向量索引 = Self.Index.构建索引(向量文件.get())
                    for _ in Self.Locale.Tqdm(range(1), desc="tqdm.index.write"):
                        with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index-md5", "w+") as f:
                            f.write(参考词哈希)
                        索引库.write_index(向量索引, f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index")
            else:
                向量索引 = Self.Index.构建索引(向量文件.get())
                for _ in Self.Locale.Tqdm(range(1), desc="tqdm.index.write"):
                    with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index-md5", "w+") as f:
                        f.write(参考词哈希)
                    索引库.write_index(向量索引, f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index")
        索引缓存[参考词哈希] = 向量索引
    else:
        向量索引 = Self.Index.构建索引(向量文件.get(), 模式)
    Self.日志("log.core.index.cache.end", info_level=0)
    try:
        index = 索引库.index_cpu_to_gpu(index)
    except:
        try:
            index = 索引库.index_gpu_to_cpu(index)
        except: pass
    return 向量索引

def 缓存数据包指令表(Self): #Module
    缓存键 = f"{Self.Config.DATA_COMMAND_PATH}/{Self.Config.DATA_COMMAND_FILE}"
    Path(缓存键).parent.mkdir(parents=True, exist_ok=True)
    if 缓存键 in 数据包指令缓存:
        return 数据包指令缓存[缓存键]
    with 线程锁:
        if 缓存键 in 数据包指令缓存:
            return 数据包指令缓存[缓存键]
        规则列表 = []
        try:
            Self.日志("log.core.command.rule.load.start", info_level=0)
            文件路径 = Path(Self.Config.DATA_COMMAND_PATH) / Self.Config.DATA_COMMAND_FILE
            if 文件路径.is_file():
                with open(文件路径, "r", encoding="utf-8") as f:
                    for 行 in f:
                        行 = 行.strip()
                        if 行 and not 行.startswith("#"):
                            规则列表.append(行)
            数据包指令缓存[缓存键] = 规则列表
            Self.日志("log.core.command.rule.load.succeed", count=len(规则列表), info_level=0)
        except Exception:
            Self.日志("log.core.command.rule.load.error", e=eb.format_exc(), info_level=3)
        return 规则列表
def 高并发会话(api_key, 并发数, 避退指数, 重试次数): # 获取会话 依赖
    会话 = requests.Session()
    会话.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    })
    adapter = HTTPAdapter(
        pool_connections=并发数,
        pool_maxsize=并发数,
        max_retries=Retry(
            total=重试次数,
            backoff_factor=避退指数,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        ),
        pool_block=False
    )
    会话.mount('https://', adapter)
    会话.mount('http://', adapter)
    return 会话

def 获取会话(api_url, api_key, model, 最大并发, 避退指数, 重试次数): #Core Module
    缓存键 = (api_url, api_key, model)
    if 缓存键 in 会话缓存:
        return 会话缓存[缓存键]
    with 线程锁:
        if 缓存键 in 会话缓存:
            return 会话缓存[缓存键]
        新会话 = 高并发会话(api_key, 最大并发, 避退指数, 重试次数)
        会话缓存[缓存键] = 新会话
        return 新会话

def _当前事件循环id():
    try:
        return id(asyncio.get_running_loop())
    except RuntimeError:
        return None

def 获取异步会话(api_url: str, api_key: str, model: str, 最大并发: int, 超时时间: tuple = (3, 300), keepalive_timeout: float = 15, ttl_dns_cache: int = 300) -> aiohttp.ClientSession:
    循环id = _当前事件循环id()
    缓存键 = (api_url, api_key, model)
    with 异步会话锁:
        本循环会话 = 异步会话缓存.get(循环id)
        if 本循环会话 is not None:
            已有会话 = 本循环会话.get(缓存键)
            if 已有会话 is not None and not 已有会话.closed:
                return 已有会话
        连接超时, 总超时 = 超时时间
        timeout = aiohttp.ClientTimeout(
            total=总超时,
            connect=连接超时,
            sock_read=None,
            sock_connect=连接超时,
        )
        安全并发 = max(1, int(最大并发))
        connector = aiohttp.TCPConnector(
            limit=安全并发,
            limit_per_host=安全并发,
            ttl_dns_cache=ttl_dns_cache,
            force_close=False,
            enable_cleanup_closed=True,
            keepalive_timeout=keepalive_timeout,
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        会话 = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers,
        )
        异步会话缓存.setdefault(循环id, {})[缓存键] = 会话
        return 会话

async def 关闭所有异步会话():
    循环id = _当前事件循环id()
    with 异步会话锁:
        本循环会话 = 异步会话缓存.pop(循环id, None)
    if not 本循环会话:
        return
    需要小睡 = False
    for 会话 in list(本循环会话.values()):
        try:
            if not 会话.closed:
                await 会话.close()
                需要小睡 = True
        except Exception:
            pass
    if 需要小睡:
        try:
            await asyncio.sleep(0.1)
        except Exception:
            pass

def 运行异步(协程):
    async def _包装运行():
        try:
            return await 协程
        finally:
            await 关闭所有异步会话()
    return asyncio.run(_包装运行())

def 加载向量缓存(Self):
    global 向量文本嵌入缓存
    try:
        基础路径 = Path(Self.Config.VEC_CACHE_PATH) / Self.Config.VEC_CACHE_NAME
        文本路径 = Path(f"{基础路径}.pkl")
        向量路径 = Path(f"{基础路径}.npz")
        if 文本路径.is_file() and 向量路径.is_file():
            with open(文本路径, "rb") as f:
                文本列表 = pickle.load(f)
            向量数据 = numpy.load(向量路径, allow_pickle=False)
            向量矩阵 = 向量数据["vec"]
            for i, 文本 in enumerate(文本列表):
                if i >= len(向量矩阵): break
                向量文本嵌入缓存[文本] = np.asarray(向量矩阵[i]) if GPU_ACC else 向量矩阵[i].copy()
            Self.日志("log.core.vector.cache.load", count=len(向量文本嵌入缓存), info_level=0)
    except Exception:
        向量文本嵌入缓存 = {}
    启动向量缓存定时刷新(Self)

def 保存向量缓存(Self):
    global 向量文本嵌入缓存, 向量缓存脏标记
    try:
        基础路径 = Path(Self.Config.VEC_CACHE_PATH) / Self.Config.VEC_CACHE_NAME
        基础路径.parent.mkdir(parents=True, exist_ok=True)
        with 向量文本嵌入锁:
            global _缓存保存轮次
            _缓存保存轮次 += 1
            宽限期 = Self.Config.VEC_CACHE_DECAY_GRACE
            衰减阈值 = Self.Config.VEC_CACHE_DECAY_THRESHOLD
            if 向量文本嵌入频率:
                过期条目 = []
                for 文本 in list(向量文本嵌入缓存.keys()):
                    频率 = 向量文本嵌入频率.get(文本, 0)
                    代数差 = _缓存保存轮次 - 向量文本嵌入代数.get(文本, 0)
                    if 代数差 <= 宽限期:
                        continue
                    if 频率 / (代数差 + 1) < 衰减阈值:
                        过期条目.append(文本)
                for 文本 in 过期条目:
                    向量文本嵌入缓存.pop(文本, None)
                    向量文本嵌入频率.pop(文本, None)
                    向量文本嵌入代数.pop(文本, None)
                if 过期条目:
                    Self.日志("log.core.vector.cache.evict", evicted=len(过期条目), remain=len(向量文本嵌入缓存), info_level=0)
            上限 = Self.Config.VEC_CACHE_MAX_SIZE
            if len(向量文本嵌入缓存) > 上限:
                def _分数(t):
                    频率 = 向量文本嵌入频率.get(t, 0)
                    代数差 = _缓存保存轮次 - 向量文本嵌入代数.get(t, 0) + 1
                    return 频率 / 代数差
                排序 = sorted(向量文本嵌入缓存.keys(), key=_分数)
                淘汰数 = len(向量文本嵌入缓存) - 上限
                for 文本 in 排序[:淘汰数]:
                    向量文本嵌入缓存.pop(文本, None)
                    向量文本嵌入频率.pop(文本, None)
                    向量文本嵌入代数.pop(文本, None)
                Self.日志("log.core.vector.cache.evict", evicted=淘汰数, remain=len(向量文本嵌入缓存), info_level=0)
            文本列表 = list(向量文本嵌入缓存.keys())
            if not 文本列表:
                向量缓存脏标记 = False
                return
            向量列表 = numpy.stack([arr.get() if GPU_ACC and hasattr(arr, 'get') else numpy.asarray(arr) for arr in (向量文本嵌入缓存[t] for t in 文本列表)])
            向量缓存脏标记 = False
        with 向量缓存保存锁:
            with open(f"{基础路径}.pkl", "wb") as f:
                pickle.dump(文本列表, f)
            numpy.savez_compressed(f"{基础路径}.npz", vec=向量列表)
    except Exception:
        向量缓存脏标记 = True

def 查询向量缓存(texts: list) -> tuple:
    with 向量文本嵌入锁:
        命中 = {}
        for item in texts:
            if item[0] in 向量文本嵌入缓存:
                命中[item[0]] = 向量文本嵌入缓存[item[0]]
                向量文本嵌入频率[item[0]] = 向量文本嵌入频率.get(item[0], 0) + 1
                向量文本嵌入代数[item[0]] = _缓存保存轮次
        未命中 = [item for item in texts if item[0] not in 向量文本嵌入缓存]
    return 命中, 未命中

def 更新向量缓存(新增条目: dict):
    global 向量缓存脏标记
    with 向量文本嵌入锁:
        向量文本嵌入缓存.update(新增条目)
        向量缓存脏标记 = True 

def _向量缓存刷新循环(间隔: float):
    while not _向量缓存刷新停止.wait(间隔):
        try:
            if 向量缓存脏标记 and _向量缓存持有者 is not None:
                保存向量缓存(_向量缓存持有者)
        except Exception:
            pass

def _退出刷新向量缓存():
    _向量缓存刷新停止.set()
    try:
        if 向量缓存脏标记 and _向量缓存持有者 is not None:
            保存向量缓存(_向量缓存持有者)
    except Exception:
        pass

def 启动向量缓存定时刷新(Self):
    global _向量缓存刷新线程, _向量缓存持有者
    _向量缓存持有者 = Self
    with _向量缓存线程锁:
        if _向量缓存刷新线程 is not None and _向量缓存刷新线程.is_alive():
            return
        间隔 = float(getattr(Self.Config, "VEC_CACHE_SAVE_INTERVAL", 30.0))
        if 间隔 <= 0:
            间隔 = 30.0
        _向量缓存刷新停止.clear()
        _向量缓存刷新线程 = threading.Thread(target=_向量缓存刷新循环, args=(间隔,), daemon=True, name="VectorCacheFlusher")
        _向量缓存刷新线程.start()
        atexit.register(_退出刷新向量缓存)

def 增量索引(Self, 翻译参考列表, 索引ID): #Core
    if 索引ID not in Self.增量索引缓存:
        Self.增量索引缓存[索引ID] = {
            "faiss_index": None,
            "texts": [],
            "key": [],
            "ids": []}
    缓存 = Self.增量索引缓存[索引ID]
    for index in 翻译参考列表:
        缓存["texts"].append(index[0])
        缓存["key"].append(index[1])
        缓存["ids"].append(index[2])
    if 翻译参考列表 and Self.Config.INDEX_LANG_K != 0:
        生成结果 = Self.Builder.并行生成向量(翻译参考列表)
        新向量 = np.asarray(生成结果[0], dtype=np.float32)
        if 新向量.ndim == 1:
            新向量 = 新向量.reshape(1, -1)
        if GPU_ACC:
            新向量 = 新向量.get()
        if 缓存["faiss_index"] is None:
            缓存["faiss_index"] = Self.Index.构建索引(新向量, Self.Config.INDEX_LANG_MODE)
        else:
            缓存["faiss_index"].add(新向量)
    return 缓存["faiss_index"], 缓存["key"], 缓存["texts"], 缓存["ids"]

def 加载翻译缓存(Self):
    global 翻译文本缓存, _翻译缓存已加载
    with _翻译缓存加载锁:
        需要加载 = not _翻译缓存已加载
        _翻译缓存已加载 = True
    if 需要加载:
        try:
            基础路径 = Path(Self.Config.TRANSLATOR_CACHE_PATH) / Self.Config.TRANSLATOR_CACHE_NAME
            pkl路径 = Path(f"{基础路径}.pkl")
            if pkl路径.is_file():
                with open(pkl路径, "rb") as f:
                    原始数据 = pickle.load(f)
                if isinstance(原始数据, dict):
                    新缓存 = dict(原始数据)
                else:
                    新缓存 = {item[0]: item[1] for item in 原始数据 if isinstance(item, (list, tuple)) and len(item) >= 2}
                with 翻译缓存锁:
                    翻译文本缓存.update(新缓存)
        except Exception:
            pass
    启动翻译缓存定时刷新(Self)

def 查询翻译缓存(原文: str = None):
    with 翻译缓存锁:
        if 原文 is None:
            return dict(翻译文本缓存)
        return 翻译文本缓存.get(原文)

def 更新翻译缓存(新增条目):
    global 翻译缓存脏标记
    if not 新增条目:
        return
    with 翻译缓存锁:
        if isinstance(新增条目, dict):
            for 原文, 译文 in 新增条目.items():
                if 原文 and 译文:
                    翻译文本缓存[原文] = 译文
        else:
            for 条目 in 新增条目:
                if isinstance(条目, (list, tuple)) and len(条目) >= 2 and 条目[0] and 条目[1]:
                    翻译文本缓存[条目[0]] = 条目[1]
        翻译缓存脏标记 = True

def 保存翻译缓存(Self):
    global 翻译缓存脏标记
    try:
        基础路径 = Path(Self.Config.TRANSLATOR_CACHE_PATH) / Self.Config.TRANSLATOR_CACHE_NAME
        基础路径.parent.mkdir(parents=True, exist_ok=True)
        with 翻译缓存锁:
            if not 翻译文本缓存:
                翻译缓存脏标记 = False
                return
            快照 = [[原文, 译文] for 原文, 译文 in 翻译文本缓存.items()]
            翻译缓存脏标记 = False
        with 翻译缓存保存锁:
            with open(f"{基础路径}.pkl", "wb") as f:
                pickle.dump(快照, f)
    except Exception:
        翻译缓存脏标记 = True

def _翻译缓存刷新循环(间隔: float):
    while not _翻译缓存刷新停止.wait(间隔):
        try:
            if 翻译缓存脏标记 and _翻译缓存持有者 is not None:
                保存翻译缓存(_翻译缓存持有者)
        except Exception:
            pass

def _退出刷新翻译缓存():
    _翻译缓存刷新停止.set()
    try:
        if 翻译缓存脏标记 and _翻译缓存持有者 is not None:
            保存翻译缓存(_翻译缓存持有者)
    except Exception:
        pass

def 启动翻译缓存定时刷新(Self):
    """后台定时线程，按 TRANSLATOR_CACHE_SAVE_INTERVAL 间隔批量落盘（幂等，重复调用只启动一次）。"""
    global _翻译缓存刷新线程, _翻译缓存持有者
    _翻译缓存持有者 = Self
    with _翻译缓存线程锁:
        if _翻译缓存刷新线程 is not None and _翻译缓存刷新线程.is_alive():
            return
        间隔 = float(getattr(Self.Config, "TRANSLATOR_CACHE_SAVE_INTERVAL", 45.0))
        if 间隔 <= 0:
            间隔 = 45.0
        _翻译缓存刷新停止.clear()
        _翻译缓存刷新线程 = threading.Thread(target=_翻译缓存刷新循环, args=(间隔,), daemon=True, name="TranslatorCacheFlusher")
        _翻译缓存刷新线程.start()
        atexit.register(_退出刷新翻译缓存)