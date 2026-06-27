from TranslatorLib import (eb, threading, Path, pickle, numpy, np, hashlib, faiss, GPU_ACC, requests, HTTPAdapter, Retry, Callable,
                           IndexGSQ)

模型缓存 = {}
向量文本缓存 = {}
索引缓存 = {}
数据包指令缓存 = {}
会话缓存 = {}
线程锁 = threading.Lock()
索引线程锁 = threading.Lock()
向量线程锁 = threading.Lock()
def 获取嵌入模型(Self):
    缓存键 = f"{Self.Config.EMB_MODEL}|{Self.Config.EMB_MODEL_ACC_MODE}"
    if 缓存键 in 模型缓存:
        return 模型缓存[缓存键]
    with 线程锁:
        if 缓存键 in 模型缓存:
            return 模型缓存[缓存键]
        try:
            for _ in Self.Locale.Tqdm(range(1), desc=f"tqdm.model.load"):
                from sentence_transformers import SentenceTransformer # type: ignore
                Self.日志("log.core.debug.load.embedded.model", model=Self.Config.EMB_MODEL, info_level=0)
                if Self.Config.EMB_MODEL_ACC_MODE == "onnx":
                    模型 = SentenceTransformer(Self.Config.EMB_MODEL, trust_remote_code=True, backend="onnx")
                else:
                    模型参数 = {}
                    if Self.Config.EMB_MODEL_ACC_MODE:
                        模型参数["dtype"] = Self.Config.EMB_MODEL_ACC_MODE
                    if Self.Config.EMB_MODEL_DEVICE:
                        模型参数["device"] = Self.Config.EMB_MODEL_DEVICE
                    模型 = SentenceTransformer(Self.Config.EMB_MODEL, trust_remote_code=True, model_kwargs=模型参数)
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
    def __init__(self, 编码数据: dict, 解码函数: Callable, VEC_CACHE: bool):
        self._编码数据 = 编码数据
        self._解码函数 = 解码函数
        self._解码结果 = None
        self.VEC_CACHE = VEC_CACHE
    def get(self) -> np.ndarray:
        if self._解码结果 is None and self.VEC_CACHE:
            self._解码结果 = self._解码函数(self._编码数据)
            return self._解码结果
        else:
            return self._解码函数(self._编码数据)
    def __getstate__(self):
        if self._解码结果 is not None:
            return {"_编码数据": None, "_解码结果": self._解码结果}
        else:
            return {"_编码数据": self._编码数据, "_解码结果": None}
    def __setstate__(self, state):
        self._编码数据 = state["_编码数据"]
        self._解码结果 = state["_解码结果"]
        self._解码函数 = None
def 参考词预处理(Self, texts: list = None, uuid = None) -> 参考词预处理向量懒加载|list: #Core
    with 向量线程锁:
        检索词 = []
        待处理文本 = []
        文件路径 = Self.Config.VEC_FILE_PATH
        if uuid:
            文件名 = uuid
        else:
            文件名 = Self.Config.VEC_FILE_NAME
        缓存键 = f"{文件路径}/{文件名}"
        if texts:
            if Path(f"{文件路径}/{文件名}.pkl").is_file():
                with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                    检索词 = [item[0] for item in pickle.load(f)]
            检索词_set = set(检索词)
            待处理文本 = [index for index in texts if index[0] not in 检索词_set]
        elif 缓存键 in 向量文本缓存:
            return 向量文本缓存[缓存键][0], 向量文本缓存[缓存键][1]
        Self.日志("log.core.vector.cache.start")
        if 待处理文本 and Self.Config.EMB_MODEL:
            返回内容向量 = Self.Builder.并行生成向量(待处理文本)
            向量结果列表 = 返回内容向量[0]
            Self.日志("log.core.debug.vector.range", range=(向量结果列表.min(), 向量结果列表.max()), info_level=4)
            文本结果列表 = [[返回内容向量[1][0][i], 返回内容向量[1][1][i]] for i in range(len(返回内容向量[1][0]))]
            if Self.Config.VEC_RERANKER: 向量结果列表, 文本结果列表 = Self.Quantization.向量重排(向量结果列表, 文本结果列表)
            for _ in Self.Locale.Tqdm(range(1), desc="tqdm.vectors.write"):
                叠加状态 = False
                if Path(f"{文件路径}/{文件名}.npz").is_file():
                    向量文件 = numpy.load(f"{文件路径}/{文件名}.npz", allow_pickle=True)
                    向量文件 = {key: np.asarray(向量文件[key]) for key in 向量文件.files}
                    向量文件, 叠加状态 = Self.Quantization.叠加量化向量(向量文件, Self.Quantization.编码向量(向量结果列表))
                    np.savez_compressed(f"{文件路径}/{文件名}.npz", **向量文件)
                else:
                    向量文件 = Self.Quantization.编码向量(向量结果列表)
                    np.savez_compressed(f"{文件路径}/{文件名}.npz", **向量文件)

                if Path(f"{文件路径}/{文件名}.pkl").is_file():
                    with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                        文本文件 = pickle.load(f)
                    if 叠加状态:
                        文本文件.extend(文本结果列表)
                        with open(f"{文件路径}/{文件名}.pkl", "wb") as f:
                            pickle.dump(文本文件, f)
                else:
                    with open(f"{文件路径}/{文件名}.pkl", "wb") as f:
                        pickle.dump(文本结果列表, f)
                    文本文件 = 文本结果列表
        else:
            try:
                for _ in Self.Locale.Tqdm(range(1), desc="tqdm.vectors.read"):
                    向量文件 = numpy.load(f"{文件路径}/{文件名}.npz", allow_pickle=True)
                    向量文件 = {key: np.asarray(向量文件[key]) for key in 向量文件.files}
                    with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                        文本文件 = pickle.load(f)
            except Exception:
                Self.日志("log.core.read.vevtor.error", e=eb.format_exc(), info_level=2)
                向量文件, 文本文件 = False, False
        Self.日志("log.core.vector.cache.end")
        向量文件 = 参考词预处理向量懒加载(向量文件, Self.Quantization.解码向量, Self.Config.VEC_CACHE)
        向量文本缓存[缓存键] = [向量文件, 文本文件]
        return 向量文件, 文本文件

def 缓存索引(Self, 向量文件: 参考词预处理向量懒加载, 文本文件, 模式 = None, 存储 = True): #Core
    with 索引线程锁:
        Self.日志("log.core.index.cache.start", info_level=0)
        if not 模式:
            模式 = Self.Config.INDEX_MODE
        if 存储:
            索引库 = IndexGSQ if 模式 in ["GSQ", "GSQFast", "GSQMoE", "GSQMoEPlus"] else faiss
            索引配置 = [getattr(Self.Config, key) for key in Self.Config.INDEX_CONFIG]
            参考词哈希 = hashlib.md5(pickle.dumps((向量文件, 文本文件, 索引配置))).hexdigest()
            if 参考词哈希 in 索引缓存:
                return 索引缓存[参考词哈希]
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