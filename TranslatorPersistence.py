from TranslatorLib import *

模型缓存 = {}
向量文本缓存 = {}
索引缓存 = {}
数据包指令缓存 = {}
会话缓存 = {}
异步会话缓存 = {}
线程锁 = threading.Lock()
嵌入模型线程锁 = threading.Lock()
重排模型线程锁 = threading.Lock()
语言模型线程锁 = threading.Lock()
索引线程锁 = threading.Lock()
向量线程锁 = threading.Lock()
异步会话锁 = threading.Lock()
Token估算器线程锁 = threading.Lock()
增量索引锁 = threading.Lock()
持久化管理器注册表 = {}
持久化管理器注册表锁 = threading.Lock()

def 获取嵌入模型(Self):
    缓存键 = f"{Self.Config.EMB_MODEL}|{Self.Config.EMB_MODEL_ACC_MODE}"
    if 缓存键 in 模型缓存:
        return 模型缓存[缓存键]
    设备设置 = [Self.Config.EMB_MODEL_DEVICE] if isinstance(Self.Config.EMB_MODEL_DEVICE, str) else Self.Config.EMB_MODEL_DEVICE
    with 嵌入模型线程锁:
        if 缓存键 in 模型缓存:
            return 模型缓存[缓存键]
        try:
            for _ in Self.tqdm(range(1), desc=f"tqdm.model.load"):
                传入参数 = dict(Self.Config.EMB_LOADER_KWARGS)
                if Self.Config.EMB_REASONING_FRAME.lower() == "sentencetransformer":
                    from sentence_transformers import SentenceTransformer # type: ignore
                    Self.日志("log.core.debug.load.embedded.model", model=Self.Config.EMB_MODEL, info_level=0)
                    模型参数 = dict(Self.Config.EMB_LOADER_MODEL_KWARGS)
                    if Self.Config.EMB_MODEL_ACC_MODE is None:
                        if Self.Config.EMB_MODEL_ACC_MODE:
                            模型参数["dtype"] = Self.Config.EMB_MODEL_ACC_MODE
                        模型 = SentenceTransformer(Self.Config.EMB_MODEL, trust_remote_code=True, device=Self.Config.EMB_MODEL_DEVICE, model_kwargs=模型参数, **传入参数)
                    elif Self.Config.EMB_MODEL_ACC_MODE.lower() == "onnx":
                        if any("cuda" in d.lower() for d in 设备设置):
                            模型参数["provider"] = "CUDAExecutionProvider"
                            传入参数["device_ids"] = [i.lower().split(":")[1] for i in 设备设置]
                    elif Self.Config.EMB_MODEL_ACC_MODE.lower() == "openvino":
                        模型 = SentenceTransformer(Self.Config.EMB_MODEL, trust_remote_code=True, backend="openvino", model_kwargs=模型参数, **传入参数)
                elif Self.Config.EMB_REASONING_FRAME.lower() == "fastembed":
                    from fastembed import TextEmbedding # type: ignore
                    if any("cuda" in d.lower() for d in 设备设置):
                        传入参数["providers"] = ["CUDAExecutionProvider"]
                        传入参数["device_ids"] = [i.lower().split(":")[1] for i in 设备设置]
                    模型 = TextEmbedding(Self.Config.EMB_MODEL, normalize=Self.Config.EMB_MODEL_NORMALIZE, **传入参数)
            模型缓存[缓存键] = 模型
            Self.日志("log.core.load.embedded.model.succeed", model=Self.Config.EMB_MODEL, info_level=0)
            return 模型
        except Exception:
            Self.日志("log.core.load.embedded.model.error", model=Self.Config.EMB_MODEL, e=eb.format_exc(), info_level=3)
            raise RuntimeError(Self.Lang("log.core.load.embedded.model.error", model=Self.Config.EMB_MODEL, e=eb.format_exc()))
def 获取图像嵌入模型(Self):
    模型名 = Self.Config.EMB_MODEL
    缓存键 = f"img|{模型名}|{Self.Config.EMB_MODEL_ACC_MODE}"
    if 缓存键 in 模型缓存:
        return 模型缓存[缓存键]
    设备设置 = [Self.Config.EMB_MODEL_DEVICE] if isinstance(Self.Config.EMB_MODEL_DEVICE, str) else Self.Config.EMB_MODEL_DEVICE
    with 嵌入模型线程锁:
        if 缓存键 in 模型缓存:
            return 模型缓存[缓存键]
        try:
            for _ in Self.tqdm(range(1), desc=f"tqdm.model.load"):
                传入参数 = dict(Self.Config.EMB_LOADER_KWARGS)
                if Self.Config.EMB_REASONING_FRAME.lower() == "sentencetransformer":
                    from sentence_transformers import SentenceTransformer # type: ignore
                    Self.日志("log.core.debug.load.embedded.model", model=模型名, info_level=0)
                    模型参数 = dict(Self.Config.EMB_LOADER_MODEL_KWARGS)
                    if Self.Config.EMB_MODEL_ACC_MODE:
                        模型参数["dtype"] = Self.Config.EMB_MODEL_ACC_MODE
                    模型 = SentenceTransformer(模型名, trust_remote_code=True, device=Self.Config.EMB_MODEL_DEVICE, model_kwargs=模型参数, **传入参数)
                elif Self.Config.EMB_REASONING_FRAME.lower() == "fastembed":
                    from fastembed import ImageEmbedding # type: ignore
                    if any("cuda" in d.lower() for d in 设备设置):
                        传入参数["providers"] = ["CUDAExecutionProvider"]
                        传入参数["device_ids"] = [i.lower().split(":")[1] for i in 设备设置]
                    模型 = ImageEmbedding(模型名, **传入参数)
            模型缓存[缓存键] = 模型
            Self.日志("log.core.image.load.embedded.model.succeed", model=模型名, info_level=0)
            return 模型
        except Exception:
            Self.日志("log.core.image.load.embedded.model.error", model=模型名, e=eb.format_exc(), info_level=3)
            raise RuntimeError(Self.Lang("log.core.image.load.embedded.model.error", model=模型名, e=eb.format_exc()))
def 获取重排模型(Self):
    缓存键 = f"{Self.Config.RERANKER_MODEL}|{Self.Config.RERANKER_INSTRUCT}"
    if 缓存键 in 模型缓存:
        return 模型缓存[缓存键]
    with 重排模型线程锁:
        if 缓存键 in 模型缓存:
            return 模型缓存[缓存键]
        try:
            for _ in Self.tqdm(range(1), desc=f"tqdm.model.load"):
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
def 创建语言模型实例(Self, 配置, 模型路径):
    import xllamacpp
    for _ in Self.tqdm(range(1), desc=f"tqdm.model.load"):
        模型 = xllamacpp.CommonParams()
        模型.model.path = str(Path(模型路径))
        Self.Module.设置实例参数(模型, 配置)
        模型 = xllamacpp.Server(模型)
    return 模型
    
def 获取语言模型同步(Self, 层级):
    缓存键 = str(层级["model"])
    if 缓存键 in 模型缓存:
        return 模型缓存[缓存键]
    模型路径 = Path(层级["model"])
    with 语言模型线程锁:
        if 缓存键 in 模型缓存:
            return 模型缓存[缓存键]
        try:
            for _ in range(1):
                Self.日志("log.core.load.llm.model.debug", info_level=0, model=层级["model"])
                加载传参 = 层级["loader_kwargs"].copy()
                加载传参.setdefault("cpuparams", {})["n_threads"] = Self.Module.采样器(加载传参.get("cpuparams", {}).get("n_threads", numpy.float32(1.0)), os.cpu_count())
                if 模型路径.is_file():
                    模型 = 创建语言模型实例(Self, 加载传参, 模型路径)
                    break
                if 模型路径.is_absolute():
                    raise FileNotFoundError(Self.Lang("log.core.load.llm.model.file.err", path=模型路径.resolve()))
                else:
                    仓库ID, 文件名, 修订版本 = Self.Module.解析HF引用(层级["model"])
                    if not 仓库ID or not 文件名:
                        raise ValueError(Self.Lang("log.core.load.llm.model.hf.file.err", path=层级["model"]))
                    for _ in Self.tqdm(range(1), desc=f"tqdm.model.download"):
                        本地文件 = huggingface_hub.hf_hub_download(repo_id=仓库ID, filename=文件名, revision=修订版本 or "main", **Self.Config.LLM_HF_DOWNLOAD_KWARGS)
                    模型 = 创建语言模型实例(Self, 加载传参, 本地文件)
            模型缓存[缓存键] = 模型
            Self.日志("log.core.load.llm.model.succeed", model=层级["model"], info_level=0)
            return 模型
        except Exception:
            Self.日志("log.core.load.llm.model.err", model=层级["model"], e=eb.format_exc(), info_level=3)
            raise RuntimeError(Self.Lang("log.core.load.llm.model.err", model=层级["model"], e=eb.format_exc()))
async def 获取语言模型(Self, 层级):
    return await asyncio.to_thread(获取语言模型同步, Self, 层级)
class 参考词预处理向量懒加载:
    def __init__(Self, 编码数据: dict, 解码函数: Callable, VEC_READ_CACHE: bool):
        Self._编码数据 = 编码数据
        Self._解码函数 = 解码函数
        Self._解码结果 = None
        Self.VEC_READ_CACHE = VEC_READ_CACHE
        Self.PCA_M = 编码数据.get("PCA_M", None)
        Self.PCA_P = 编码数据.get("PCA_P", None)
        Self.TT_Cores = 编码数据.get("TT_Cores", None)
        Self.TT_Mean = 编码数据.get("TT_Mean", None)
        Self.TT_Shape = 编码数据.get("TT_Shape", None)
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
async def 参考词预处理(Self, texts: list = None, uuid = None, use_cache: bool = True, 查询: bool = False, 图像: bool = False) -> tuple[参考词预处理向量懒加载, list]: #Core
    检索词, 待处理文本 = [], []
    PCA均值, PCA投影矩阵 = None, None
    文件路径 = Self.Config.VEC_FILE_PATH
    文件名 = uuid if uuid else Self.Config.VEC_FILE_NAME
    缓存键 = f"{文件路径}/{文件名}"
    if texts:
        if use_cache and Path(f"{文件路径}/{文件名}.pkl").is_file():
            with 向量线程锁:
                with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                    检索词 = [(item[1] if 图像 else item[0]) for item in pickle.load(f)]
        检索词_set = set(检索词)
        待处理文本 = [index for index in texts if (index[1] if 图像 else index[0]) not in 检索词_set]
    elif 缓存键 in 向量文本缓存:
        return 向量文本缓存[缓存键][0], 向量文本缓存[缓存键][1]
    if (not 待处理文本) and texts and (not use_cache): 待处理文本 = texts
    Self.日志("log.core.vector.cache.start")
    if 待处理文本 and Self.Config.EMB_MODEL:
        if 图像:
            返回内容向量 = await Self.Builder.并行生成图像向量(待处理文本, use_cache=use_cache)
        else:
            返回内容向量 = await Self.Builder.并行生成向量(待处理文本, use_cache=use_cache, 查询=查询)
        向量结果列表 = 返回内容向量[0]
        if Self.Config.VEC_PCA_DIM != -1:
            向量结果列表, PCA均值, PCA投影矩阵 = Self.Quantization.PCA降维(向量结果列表)
        TT核心列表, TT均值, TT形状 = None, None, None
        if Self.Config.VEC_TT_RANK > 0 and len(Self.Config.VEC_TT_SHAPE) > 0:
            TT核心列表, TT均值, TT形状 = Self.Quantization.TT分解(向量结果列表, Self.Config.VEC_TT_SHAPE, Self.Config.VEC_TT_RANK)
            向量结果列表 = np.array([Self.Quantization.TT压缩(v, TT核心列表, TT均值, TT形状) for v in 向量结果列表], dtype=np.float32)
        Self.日志("log.core.debug.vector.range", range=(向量结果列表.min(), 向量结果列表.max()), info_level=4)
        if 图像:
            文本结果列表 = [[None, 返回内容向量[1][1][i]] for i in range(len(返回内容向量[1][1]))]
        else:
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
                    if "TT_Cores" in 旧向量文件 and "TT_Mean" in 旧向量文件:
                        向量结果列表 = np.array([Self.Quantization.TT压缩(v, 旧向量文件["TT_Cores"], 旧向量文件["TT_Mean"], 旧向量文件["TT_Shape"]) for v in 向量结果列表], dtype=np.float32)
                    文本文件.extend(文本结果列表)
                    向量文件, 文本文件 = Self.Quantization.叠加量化向量(旧向量文件, 向量结果列表, 文本文件)
                    
                    np.savez_compressed(f"{文件路径}/{文件名}.npz", **向量文件)
                    with open(f"{文件路径}/{文件名}.pkl", "wb") as f:
                        pickle.dump(文本文件, f)
                else:
                    向量文件 = Self.Quantization.编码向量(向量结果列表)
                    if PCA均值 is not None: 向量文件["PCA_M"] = PCA均值
                    if PCA投影矩阵 is not None: 向量文件["PCA_P"] = PCA投影矩阵
                    if TT核心列表 is not None: 向量文件["TT_Cores"] = TT核心列表
                    if TT均值 is not None: 向量文件["TT_Mean"] = TT均值
                    if TT形状 is not None: 向量文件["TT_Shape"] = TT形状
                    np.savez_compressed(f"{文件路径}/{文件名}.npz", **向量文件)
                    with open(f"{文件路径}/{文件名}.pkl", "wb") as f:
                        pickle.dump(文本结果列表, f)
                    文本文件 = 文本结果列表
    else:
        if not (Path(f"{文件路径}/{文件名}.npz").is_file() and Path(f"{文件路径}/{文件名}.pkl").is_file()):
            向量文件, 文本文件 = False, False
        else:
            try:
                with 向量线程锁:
                    for _ in Self.tqdm(range(1), desc="tqdm.vectors.read"):
                        向量文件 = numpy.load(f"{文件路径}/{文件名}.npz", allow_pickle=True)
                        向量文件 = {key: np.asarray(向量文件[key]) for key in 向量文件.files}
                        with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                            文本文件 = pickle.load(f)
            except Exception:
                Self.日志("log.core.read.vevtor.error", e=eb.format_exc(), info_level=2)
                向量文件, 文本文件 = False, False
    Self.日志("log.core.vector.cache.end")
    if 向量文件:
        向量文件 = 参考词预处理向量懒加载(向量文件, Self.Quantization.解码向量, Self.Config.VEC_READ_CACHE)
    向量文本缓存[缓存键] = [向量文件, 文本文件]
    return (向量文件, 文本文件)

def 缓存索引(Self, 向量文件: 参考词预处理向量懒加载, 文本文件, 模式 = None, 存储 = True): #Core
    索引库 = faiss
    Self.日志("log.core.index.cache.start", info_level=0)
    if not 模式:
        模式 = Self.Config.INDEX_MODE
    if 存储:
        索引库 = TranslatorIndex if "RefineLowDim" in 模式  else IndexGSQ if "GSQFast" in 模式 else faiss
        索引配置 = [getattr(Self.Config, key) for key in Self.Config.INDEX_CONFIG]
        参考词哈希 = hashlib.md5(pickle.dumps((向量文件, 文本文件, 索引配置))).hexdigest()
        if 参考词哈希 in 索引缓存:
            return 索引缓存[参考词哈希]
        with 索引线程锁:
            if Path(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index-md5").is_file():
                with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index-md5", "r") as f:
                    参考词哈希文件 = f.read()
                if 参考词哈希文件 == 参考词哈希:
                    for _ in Self.tqdm(range(1), desc="tqdm.index.read"):
                        向量索引 = 索引库.read_index(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index")
                else:
                    向量索引 = Self.Index.构建索引(向量文件.get())
                    for _ in Self.tqdm(range(1), desc="tqdm.index.write"):
                        with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index-md5", "w+") as f:
                            f.write(参考词哈希)
                        索引库.write_index(向量索引, f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index")
            else:
                向量索引 = Self.Index.构建索引(向量文件.get())
                for _ in Self.tqdm(range(1), desc="tqdm.index.write"):
                    with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index-md5", "w+") as f:
                        f.write(参考词哈希)
                    索引库.write_index(向量索引, f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.index")
        索引缓存[参考词哈希] = 向量索引
    else:
        向量索引 = Self.Index.构建索引(向量文件.get(), 模式)
    Self.日志("log.core.index.cache.end", info_level=0)
    try:
        向量索引 = 索引库.index_cpu_to_gpu(向量索引)
    except:
        try:
            向量索引 = 索引库.index_gpu_to_cpu(向量索引)
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
async def 增量索引(Self, 翻译参考列表, 索引ID, 索引模式, 索引k): #class: 上下文管理器 Core
    # ↓线程锁保护缓存创建与文本追加 防止翻译整合包多线程共享同一索引ID时竞态
    with 增量索引锁:
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
    if 翻译参考列表 and 索引k != 0:
        生成结果 = await Self.Builder.并行生成向量(翻译参考列表, 查询=False)
        新向量 = np.asarray(生成结果[0], dtype=np.float32)
        if 新向量.ndim == 1:
            新向量 = 新向量.reshape(1, -1)
        if GPU_ACC:
            新向量 = 新向量.get()
        with 增量索引锁: # ↓faiss add与search互斥 防止并发add损坏索引或与文本列表错位
            if 缓存["faiss_index"] is None:
                缓存["faiss_index"] = Self.Index.构建索引(新向量, 索引模式)
            else:
                缓存["faiss_index"].add(新向量)
    with 增量索引锁: # ↓返回快照 防止返回后其他线程继续修改共享列表导致索引越界
        return 缓存["faiss_index"], list(缓存["key"]), list(缓存["texts"]), list(缓存["ids"])

class 持久化管理器:
    
    def __init__(Self, ID: str, 加载回调, 保存回调, 查询回调, 更新回调, 保存间隔: float = 60.0):
        Self.name = ID
        Self.保存间隔 = max(0.1, 保存间隔)
        Self.加载回调 = 加载回调
        Self.保存回调 = 保存回调
        Self.查询回调 = 查询回调 or Self.默认查询
        Self.更新回调 = 更新回调 or Self.默认更新
        Self.主数据: dict = {}
        Self.辅数据: list = []
        Self.脏标记 = False
        Self.已加载标志 = False
        Self.已失效标志 = False
        Self.线程锁 = threading.Lock()
        Self.保存锁 = threading.Lock()
        Self.持有者 = None
        Self.刷新线程 = None
        Self.停止事件 = threading.Event()

    def 默认查询(Self, key=None):
        with Self.线程锁:
            if key is None: return dict(Self.主数据)
            if isinstance(key, (set, list, tuple)):
                return {k: Self.主数据.get(k) for k in key}
            return Self.主数据.get(key)
    def 默认更新(Self, items):
        changed = False
        with Self.线程锁:
            if isinstance(items, dict):
                for k, v in items.items():
                    if k and v: Self.主数据[k] = v; changed = True
            else:
                for it in (items or []):
                    if isinstance(it, (list, tuple)) and len(it) >= 2 and it[0]:
                        Self.主数据[it[0]] = it[1]; changed = True
        return changed


    def _是最新实例(Self) -> bool:
        if Self.已失效标志: return False
        with 持久化管理器注册表锁:
            当前实例 = 持久化管理器注册表.get(Self.name)
            return 当前实例 is None or 当前实例 is Self
    def 加载(Self, app):
        Self.持有者 = app
        if Self.加载回调: Self.加载回调(app)
        Self.已加载标志 = True
        with 持久化管理器注册表锁:
            旧实例 = 持久化管理器注册表.get(Self.name)
        if 旧实例 is not None and 旧实例 is not Self:
            try: 旧实例.失效()
            except Exception: pass
        with 持久化管理器注册表锁:
            持久化管理器注册表[Self.name] = Self
        Self.启动定时刷新(app)
    def 添加(Self, items):
        if Self.更新回调:
            changed = Self.更新回调(items)
        else:
            changed = Self.默认更新(items)
        if changed: Self.脏标记 = True
    def 查询(Self, key=None):
        return Self.查询回调(key)
    def 保存(Self, app=None):
        if not Self._是最新实例():
            return
        if not Self.已加载标志 and Self.加载回调 is not None:
            try:
                Self.加载回调(app or Self.持有者)
            except Exception:
                return
            Self.已加载标志 = True
        if Self.保存回调:
            with Self.保存锁:
                Self.保存回调(app or Self.持有者)
            Self.脏标记 = False


    def 失效(Self):
        Self.停止事件.set()
        try:
            if Self.持有者 is not None and Self.脏标记 and Self._是最新实例():
                Self.保存(Self.持有者)
        except Exception: pass
        Self.已失效标志 = True
        with 持久化管理器注册表锁:
            if 持久化管理器注册表.get(Self.name) is Self:
                del 持久化管理器注册表[Self.name]
        try:
            atexit.unregister(Self.退出回调)
        except Exception:
            pass
    def 关闭(Self):
        Self.失效()
    def 刷新循环(Self):
        while not Self.停止事件.wait(Self.保存间隔):
            try:
                if Self.脏标记 and Self.持有者 is not None and Self._是最新实例():
                    Self.保存()
            except Exception: pass
    def 退出回调(Self):
        Self.停止事件.set()
        try:
            if Self.持有者 is not None and Self.脏标记 and Self._是最新实例():
                Self.保存(Self.持有者)
        except Exception: pass
    def 启动定时刷新(Self, app):
        Self.持有者 = app
        with threading.Lock():
            if Self.刷新线程 is not None and Self.刷新线程.is_alive(): return
            Self.停止事件.clear()
            Self.刷新线程 = threading.Thread(target=Self.刷新循环, daemon=True, name=f"CacheFlusher-{Self.name}")
            Self.刷新线程.start()
            atexit.register(Self.退出回调)



class VectorCache:
    
    def __init__(Self, app):
        Self.App = app
        Self.向量嵌入数据: dict = {} # {text: np.ndarray}
        Self.向量嵌入频率: dict = {} # {text: count}
        Self.向量嵌入代数: dict = {} # {text: round}
        Self.向量保存轮次: int = 0 # 每轮保存递增，用于衰减淘汰
        try:
            Self.初始化缓存实例().加载(Self.App)
        except Exception: pass
    def 向量加载回调(Self, app):
        try:
            基础路径 = Path(app.Config.VEC_CACHE_PATH) / app.Config.VEC_CACHE_NAME
            if not Path(f"{基础路径}.pkl").is_file() or not Path(f"{基础路径}.npz").is_file():
                return
            with open(f"{基础路径}.pkl", "rb") as f:
                原始数据 = pickle.load(f)
            文本列表       = 原始数据.get("texts", [])
            Self.向量嵌入频率.update(原始数据.get("frequency", {}))
            Self.向量嵌入代数.update(原始数据.get("algebra", {}))
            Self.向量保存轮次    = 原始数据.get("save_round", 0)

            向量矩阵 = numpy.load(f"{基础路径}.npz", allow_pickle=False)["vec"]
            for i, 文本 in enumerate(文本列表):
                if i >= len(向量矩阵): break
                Self.向量嵌入数据[文本] = np.asarray(向量矩阵[i]) if GPU_ACC else 向量矩阵[i].copy()
            app.日志("log.core.vector.cache.load", count=len(Self.向量嵌入数据), info_level=0)
        except Exception:
            Self.向量嵌入数据.clear()
            app.日志("log.core.vector.cache.load.error", e=eb.format_exc(), info_level=2)
    def 向量保存回调(Self, app):
        try:
            基础路径 = Path(app.Config.VEC_CACHE_PATH) / app.Config.VEC_CACHE_NAME
            基础路径.parent.mkdir(parents=True, exist_ok=True)
            Self.向量保存轮次 += 1
            if Self.向量嵌入频率:
                过期条目 = []
                for t in list(Self.向量嵌入数据.keys()):
                    f_ = Self.向量嵌入频率.get(t, 0)
                    g  = Self.向量保存轮次 - Self.向量嵌入代数.get(t, 0)
                    if g <= app.Config.VEC_CACHE_DECAY_GRACE: continue
                    if f_ / (g + 1) < app.Config.VEC_CACHE_DECAY_THRESHOLD: 过期条目.append(t)
                for t in 过期条目:
                    Self.向量嵌入数据.pop(t, None)
                    Self.向量嵌入频率.pop(t, None)
                    Self.向量嵌入代数.pop(t, None)
                if 过期条目 and hasattr(app, "日志"):
                    app.日志("log.core.vector.cache.evict", evicted=len(过期条目), remain=len(Self.向量嵌入数据), info_level=0)
            if len(Self.向量嵌入数据) > app.Config.VEC_CACHE_MAX_SIZE:
                def _计算评分(t):
                    频次 = Self.向量嵌入频率.get(t, 0)
                    代数差 = Self.向量保存轮次 - Self.向量嵌入代数.get(t, 0) + 1
                    return 频次 / 代数差
                排序列表 = sorted(Self.向量嵌入数据.keys(), key=_计算评分)
                淘汰数   = len(Self.向量嵌入数据) - app.Config.VEC_CACHE_MAX_SIZE
                for t in 排序列表[:淘汰数]:
                    Self.向量嵌入数据.pop(t, None)
                    Self.向量嵌入频率.pop(t, None)
                    Self.向量嵌入代数.pop(t, None)
                if 淘汰数 > 0 and hasattr(app, "日志"):
                    app.日志("log.core.vector.cache.evict", evicted=淘汰数, remain=len(Self.向量嵌入数据), info_level=0)
            文本列表 = list(Self.向量嵌入数据.keys())
            if not 文本列表: return
            向量列表 = numpy.stack([arr.get() if GPU_ACC and hasattr(arr, 'get') else np.asarray(arr) for arr in (Self.向量嵌入数据[t] for t in 文本列表)])
            快照 = {
                "texts":        文本列表,
                "frequency":    dict(Self.向量嵌入频率),
                "algebra":      dict(Self.向量嵌入代数),
                "save_round":   Self.向量保存轮次,
            }
            with open(f"{基础路径}.pkl", "wb") as f:
                pickle.dump(快照, f)
            numpy.savez_compressed(f"{基础路径}.npz", vec=向量列表)
        except Exception:
            app.日志("log.core.vector.cache.save.error", e=eb.format_exc(), info_level=2)
    def 向量查询回调(Self, texts=None):
        命中, 未命中 = {}, []
        for item in (texts or []):
            k = item[0] if isinstance(item, (list, tuple)) else item
            if not isinstance(k, str) and isinstance(item, (list, tuple)) and len(item) > 1:
                k = item[1]
            if k in Self.向量嵌入数据:
                命中[k] = Self.向量嵌入数据[k]
                Self.向量嵌入频率[k] = Self.向量嵌入频率.get(k, 0) + 1
                Self.向量嵌入代数[k] = Self.向量保存轮次
            else:
                未命中.append(item)
        return 命中, 未命中
    def 向量更新回调(Self, items):
        for k, v in items.items():
            Self.向量嵌入数据[k] = v
        return True
    def 初始化缓存实例(Self):
        Self.向量缓存实例 = 持久化管理器(
            ID="vector",
            保存间隔=Self.App.Config.VEC_CACHE_SAVE_INTERVAL,
            加载回调=Self.向量加载回调,
            保存回调=Self.向量保存回调,
            查询回调=Self.向量查询回调,
            更新回调=Self.向量更新回调,
        )
        Self.向量缓存实例.主数据 = Self.向量嵌入数据
        Self.向量缓存实例.辅数据 = [Self.向量嵌入频率, Self.向量嵌入代数]
        return Self.向量缓存实例
    
    # ↓接口
    def 失效(Self):
        Self.向量缓存实例.失效()
    def 保存向量缓存(Self=None):
        app = Self if Self is not None else Self.向量缓存实例.持有者
        Self.向量缓存实例.保存(app)
    def 查询向量缓存(Self, texts: list) -> tuple:
        return Self.向量缓存实例.查询(texts)
    def 更新向量缓存(Self, 新增条目: dict):
        for k, v in 新增条目.items():
            Self.向量嵌入数据[k] = v
        Self.向量缓存实例.脏标记 = True

class TranslationCache:
    def __init__(Self, app):
        Self.App = app
        Self.翻译缓存数据: dict = {} # {语言: {原文: 译文}} 按目标语言隔离的嵌套缓存
        Self.翻译已加载标志: bool = False
        try:
            Self.初始化缓存实例().加载(Self.App)
        except Exception: pass
    def 翻译加载回调(Self, app):
        with threading.Lock():
            if Self.翻译已加载标志: return # 已经加载过直接回家
            需要加载 = not Self.翻译已加载标志 # 确认是否需要加载
            Self.翻译已加载标志 = True # 标记已加载
        if 需要加载:
            try:
                基础路径 = Path(app.Config.TRANSLATOR_CACHE_PATH) / app.Config.TRANSLATOR_CACHE_NAME # 构建文件路径
                pkl路径 = Path(f"{基础路径}.pkl") # pickle数据路径
                if pkl路径.is_file(): # 检查文件是否存在
                    with open(pkl路径, "rb") as f: # 读取pickle数据
                        原始数据 = pickle.load(f) # 加载原始数据
                    当前语言 = app.Config.LANGUAGE_OUTPUT
                    if isinstance(原始数据, dict) and all(isinstance(v, dict) for v in 原始数据.values()):
                        Self.翻译缓存数据.update(原始数据) # 新版 {语言: {原文: 译文}}
                    else: # 旧版 {原文: 译文} 或 [[原文, 译文], ...] 迁移到当前语言
                        语言缓存 = Self.翻译缓存数据.setdefault(当前语言, {})
                        if isinstance(原始数据, dict):
                            语言缓存.update({k: v for k, v in 原始数据.items() if k and v})
                        else:
                            for it in (原始数据 or []):
                                if isinstance(it, (list, tuple)) and len(it) >= 2 and it[0]:
                                    语言缓存[it[0]] = it[1]
            except Exception: pass # 加载失败忽略
    def 翻译保存回调(Self, app):
        try:
            基础路径 = Path(app.Config.TRANSLATOR_CACHE_PATH) / app.Config.TRANSLATOR_CACHE_NAME # 构建文件路径
            基础路径.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
            if not Self.翻译缓存数据: return # 空缓存不保存
            临时路径 = Path(f"{基础路径}.pkl.tmp")
            with open(临时路径, "wb") as f: # 写入pickle临时文件
                pickle.dump(Self.翻译缓存数据, f) # 保存 {语言: {原文: 译文}}
            临时路径.replace(Path(f"{基础路径}.pkl")) # 原子替换 保证读者永远拿到完整文件
        except Exception: pass # 保存失败忽略
    def 翻译查询回调(Self, key=None, 语言: str = None):
        if 语言 is None: return dict(Self.翻译缓存数据) # 未指定语言返回全部嵌套缓存
        语言缓存 = Self.翻译缓存数据.get(语言) # 取目标语言子缓存
        if 语言缓存 is None: return {} if key is None else None # 无该语言缓存
        if key is None: return dict(语言缓存) # 返回该语言全部缓存副本
        return 语言缓存.get(key) # 返回指定原文对应的译文
    def 翻译更新回调(Self, items, 语言: str = None):
        语言 = 语言 or Self.App.Config.LANGUAGE_OUTPUT # 默认使用当前配置的目标语言
        语言缓存 = Self.翻译缓存数据.setdefault(语言, {}) # 取或建目标语言子缓存
        if isinstance(items, dict): # 如果是字典格式
            for k, v in items.items(): # 遍历键值对
                if k and v: 语言缓存[k] = v # 非空原文→译文存入当前语言缓存
        else: # 如果是列表/元组格式
            for it in (items or []): # 遍历条目
                if isinstance(it, (list, tuple)) and len(it) >= 2 and it[0]: # 检查是否是二元组且原文非空
                    语言缓存[it[0]] = it[1] # 存入原文→译文映射
        return True # 标记有更新
    def 初始化缓存实例(Self):
        Self.翻译缓存实例 = 持久化管理器( # 创建新的持久化管理器
            ID="translation", # 标识符用于日志和线程名
            保存间隔=Self.App.Config.TRANSLATOR_CACHE_SAVE_INTERVAL, # 定时刷新间隔秒数（从配置读取）
            加载回调=Self.翻译加载回调, # 加载数据回调
            保存回调=Self.翻译保存回调, # 保存数据回调
            查询回调=Self.翻译查询回调, # 查询数据回调
            更新回调=Self.翻译更新回调, # 更新数据回调
        )
        Self.翻译缓存实例.主数据 = Self.翻译缓存数据 # 映射主数据区到缓存字典
        return Self.翻译缓存实例 # 返回创建的实例
    
    # ↓接口
    def 失效(Self):
        Self.翻译缓存实例.失效()
    def 查询翻译缓存(Self, key: str = None, 语言: str = None):
        return Self.翻译查询回调(key, 语言=语言) # 直接查询（支持按目标语言隔离）
    def 更新翻译缓存(Self, 新增条目, 语言: str = None):
        if not 新增条目: return # 空条目直接回家
        Self.翻译更新回调(新增条目, 语言=语言) # 更新到目标语言子缓存
        Self.翻译缓存实例.脏标记 = True # 标记为未保存
    def 翻译缓存(Self, 输入列表: list = None, 语言: str = None):
        if 输入列表:
            过滤条目 = []
            if isinstance(输入列表, dict):
                过滤条目 = {k: v for k, v in 输入列表.items() if k and v and k != v}
            else:
                过滤条目 = [[it[0], it[1]] for it in 输入列表
                            if isinstance(it, (list, tuple)) and len(it) >= 2
                            and it[0] and it[1] and it[0] != it[1]]
            if 过滤条目:
                Self.更新翻译缓存(过滤条目, 语言=语言)
        return Self.查询翻译缓存(语言=语言)

class TokenCalibratorCache:
    def __init__(Self, app):
        Self.App = app
        Self.估算器数据: dict = {}    # 格式: {"模型名称": accumulator} 持久化矩阵
        Self.校准器集合: dict = defaultdict(token_calibrator.TokenCalibrator)    # 格式: {"模型名称": TokenCalibrator} 内存训练器
        Self.估算器集合: dict = defaultdict(token_calibrator.TokenEstimator)    # 格式: {"模型名称": TokenEstimator} 内存估算器
        Self.已加载标志: bool = False
        Self.统一模型预编译 = re.compile(r'[@:].*|[ _\-]+')
        try:
            Self.初始化缓存实例().加载(Self.App)
        except Exception: pass
    def 统一模型名称(Self, model):
        return Self.统一模型预编译.sub('', model).lower()
    def 估算器加载回调(Self, app):
        with threading.Lock():
            if Self.已加载标志: return # 已经加载过直接回家
            需要加载 = not Self.已加载标志 # 确认是否需要加载
            Self.已加载标志 = True # 标记已加载
        if 需要加载:
            try:
                基础路径 = Path(app.Config.TOKEN_CALIBRATOR_CACHE_PATH) / app.Config.TOKEN_CALIBRATOR_CACHE_NAME
                pkl路径 = Path(f"{基础路径}.pkl") # pickle数据路径
                if pkl路径.is_file(): # 检查文件是否存在
                    with open(pkl路径, "rb") as f: # 读取pickle数据
                        原始数据 = pickle.load(f) # 加载原始数据
                    if isinstance(原始数据, dict): # 如果是字典格式
                        Self.估算器数据.update(原始数据) # 直接更新缓存
            except Exception: pass
    def 估算器保存回调(Self, app):
        try:
            基础路径 = Path(app.Config.TOKEN_CALIBRATOR_CACHE_PATH) / app.Config.TOKEN_CALIBRATOR_CACHE_NAME
            基础路径.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
            if not Self.估算器数据: return # 空缓存不保存
            临时路径 = Path(f"{基础路径}.pkl.tmp")
            with open(临时路径, "wb") as f:
                pickle.dump(dict(Self.估算器数据), f)
            临时路径.replace(Path(f"{基础路径}.pkl"))
        except Exception: pass
    def 估算器查询回调(Self, key=None):
        if key is None: return dict(Self.估算器数据) # key为空返回全部缓存副本
        return Self.估算器数据.get(key) # 返回指定模型名对应的accumulator矩阵
    def 估算器更新回调(Self, items):
        if isinstance(items, dict): # 如果是字典格式
            for k, v in items.items(): # 遍历键值对
                if k and v: Self.估算器数据[k] = v # 非空模型名→矩阵存入缓存
        return True
    def 初始化缓存实例(Self):
        Self.Token估算器缓存实例 = 持久化管理器(
            ID="token_calibrator",
            保存间隔=Self.App.Config.TOKEN_CALIBRATOR_CACHE_SAVE_INTERVAL,
            加载回调=Self.估算器加载回调, # 加载数据回调
            保存回调=Self.估算器保存回调, # 保存数据回调
            查询回调=Self.估算器查询回调, # 查询数据回调
            更新回调=Self.估算器更新回调, # 更新数据回调
        )
        Self.Token估算器缓存实例.主数据 = Self.估算器数据 # 映射主数据区到缓存字典
        return Self.Token估算器缓存实例 # 返回创建的实例
    
    # ↓接口
    def 失效(Self):
        Self.Token估算器缓存实例.失效()
    def 添加Token(Self, model: str, text: str, Token: int):
        模型名称 = Self.统一模型名称(model)
        with Token估算器线程锁:
            if 模型名称 not in Self.校准器集合: # 检查是否已有该模型的校准器
                if 模型名称 in Self.估算器数据: # 如果持久化缓存中有数据
                    新校准器 = token_calibrator.TokenCalibrator() # 新建校准器实例
                    新校准器._acc = Self.估算器数据[模型名称] # 恢复累加器状态
                else: # 如果持久化缓存中没有数据
                    新校准器 = token_calibrator.TokenCalibrator() # 新建空校准器实例
                Self.校准器集合[模型名称] = 新校准器 # 存入校准器集合
            Self.校准器集合[模型名称].observe(str(text), Token) # 传入原文+Token数→训练累加器
            Self.估算器数据[模型名称] = Self.校准器集合[模型名称].to_matrix() # 将校准器的矩阵状态写入持久化缓存
            Self.估算器集合.pop(模型名称, None) # 清除该模型的内存估算器（需重新训练）
        Self.Token估算器缓存实例.脏标记 = True # 标记为未保存
    def 估算Token(Self, model: str, text: str):
        模型名称 = Self.统一模型名称(model)
        if 模型名称 not in Self.估算器集合: # 检查是否已有该模型的估算器 不在就创建
            矩阵字典 = {模型名称: Self.估算器数据[模型名称]} if 模型名称 in Self.估算器数据 else None # 从持久化缓存取矩阵（或None）
            Self.估算器集合[模型名称] = token_calibrator.TokenEstimator(矩阵字典) # 新建估算器实例
        return Self.估算器集合[模型名称].estimate(模型名称, str(text)) # 调用估算器输出Token数量