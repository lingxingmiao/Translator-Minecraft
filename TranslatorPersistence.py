from TranslatorLib import eb, threading, Path, pickle, numpy, np, hashlib, faiss

模型缓存 = {}
向量文本缓存 = {}
索引缓存 = {}
线程锁 = threading.Lock()
def 获取嵌入模型(Self):
    缓存键 = f"{Self.Config.EMB_MODEL}|{Self.Config.EMB_MODEL_ACC_MODE}"
    if 缓存键 in 模型缓存:
        return 模型缓存[缓存键]
    with 线程锁:
        if 缓存键 in 模型缓存:
            return 模型缓存[缓存键]
        try:
            from sentence_transformers import SentenceTransformer
            Self.Module.写入日志("log.core.debug.load.embedded.model", model=Self.Config.EMB_MODEL, info_level=0)
            for _ in Self.Locale.Tqdm(range(1), desc=f"tqdm.model.load"):
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
            Self.Module.写入日志("log.core.load.embedded.model.succeed", model=Self.Config.EMB_MODEL, info_level=0)
            return 模型
        except Exception:
            Self.Module.写入日志("log.core.load.embedded.model.error", model=Self.Config.EMB_MODEL, e=eb.format_exc(), info_level=3)
            raise RuntimeError(Self.Lang("log.core.load.embedded.model.error", model=Self.Config.EMB_MODEL, e=eb.format_exc()))
        
def 参考词预处理(Self, texts: list = None,) -> tuple[np.ndarray, list]:
    检索词 = []
    pkl文件内容 = []
    待处理文本 = []
    文件路径 = Self.Config.VEC_FILE_PATH
    文件名 = Self.Config.VEC_FILE_NAME
    缓存键 = f"{文件路径}/{文件名}"
    if texts:
        if Path(f"{文件路径}/{文件名}.pkl").is_file():
            with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                pkl文件内容.extend(pickle.load(f))
                for index in pkl文件内容:
                    检索词.append(index[0])
        检索词_set = set(检索词)
        待处理文本 = [index for index in texts if index[0] not in 检索词_set]
    elif 缓存键 in 向量文本缓存:
        return 向量文本缓存[缓存键][0], 向量文本缓存[缓存键][1]
    Self.Module.写入日志("log.core.vector.cache.start")
    if 待处理文本 and Self.Config.EMB_MODEL:
        返回内容向量 = Self.并行生成向量(待处理文本)
        向量结果列表 = 返回内容向量[0]
        Self.Module.写入日志("log.core.debug.vector.range", range=(向量结果列表.min(), 向量结果列表.max()), info_level=4)
        文本结果列表 = [[返回内容向量[1][0][i], 返回内容向量[1][1][i]] for i in range(len(返回内容向量[1][0]))]
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
                if 叠加状态:
                    with open(f"{文件路径}/{文件名}.pkl", "rb") as f:
                        文本文件 = pickle.load(f)
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
            Self.Module.写入日志("log.core.read.vevtor.error", e=eb.format_exc(), info_level=2)
            向量文件, 文本文件 = False, False
    Self.Module.写入日志("log.core.vector.cache.end")
    向量文本缓存[缓存键] = [向量文件, 文本文件]
    return 向量文件, 文本文件

def 缓存索引(Self, 向量文件, 文本文件):
    索引配置 = [Self.Config.INDEX_MODE, Self.Config.INDEX_SQ, Self.Config.INDEX_HNSW_CONSTRUCTION, Self.Config.INDEX_HNSW_SEARCH, Self.Config.INDEX_HNSW_M, Self.Config.INDEX_REFINEFLAT_K_FACTOR]
    参考词哈希 = hashlib.sha3_256(pickle.dumps((向量文件, 文本文件, 索引配置))).hexdigest()
    if 参考词哈希 in 索引缓存:
        return 索引缓存[参考词哈希]
    Self.Module.写入日志("log.core.index.cache.start", info_level=0)
    if Path(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.faiss-sha3").is_file():
        with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.faiss-sha3", "r") as f:
            参考词哈希文件 = f.read()
        if 参考词哈希文件 == 参考词哈希:
            for _ in Self.Locale.Tqdm(range(1), desc="tqdm.index.read"):
                向量索引 = faiss.read_index(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.faiss")
        else:
            向量索引 = Self.构建索引(向量文件)
            for _ in Self.Locale.Tqdm(range(1), desc="tqdm.index.write"):
                with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.faiss-sha3", "w+") as f:
                    f.write(参考词哈希)
                faiss.write_index(向量索引, f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.faiss")
    else:
        向量索引 = Self.构建索引(向量文件)
        for _ in Self.Locale.Tqdm(range(1), desc="tqdm.index.write"):
            with open(f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.faiss-sha3", "w+") as f:
                f.write(参考词哈希)
            faiss.write_index(向量索引, f"{Self.Config.VEC_FILE_PATH}/{Self.Config.VEC_FILE_NAME}.faiss")
    Self.Module.写入日志("log.core.index.cache.end", info_level=0)
    索引缓存[参考词哈希] = 向量索引
    return 向量索引