from TranslatorLib import (os, faiss, numpy, pickle, njit, numba, _prange,
                           IndexGSQ, GPU_ACC, Config)
@njit(cache=True, parallel=True, fastmath=True) # 上一世不用Numba耗时增加1600% 这一世-10%
def 索引重排(原始向量, 粗排索引, 查询矩阵, 截取数量):
    查询数量 = 查询矩阵.shape[0]
    得分矩阵 = numpy.empty((查询数量, 截取数量), dtype=numpy.float32)
    索引矩阵 = numpy.empty((查询数量, 截取数量), dtype=numpy.int64)
    索引矩阵[:] = -1
    for i in _prange(查询数量):
        候选索引 = 粗排索引[i]
        候选数量 = 候选索引.shape[0]
        if 候选数量 == 0: continue
        当前查询 = 查询矩阵[i]
        相似度得分 = numpy.empty(候选数量, dtype=numpy.float32)
        for j in range(候选数量):
            向量索引 = 候选索引[j]
            if 向量索引 < 0: 相似度得分[j] = -1e30
            else: 相似度得分[j] = numpy.dot(原始向量[向量索引], 当前查询)
        实际截取数 = min(截取数量, 候选数量)
        排序后索引 = numpy.argsort(相似度得分)[::-1][:实际截取数]
        得分矩阵[i, :实际截取数] = 相似度得分[排序后索引]
        索引矩阵[i, :实际截取数] = 候选索引[排序后索引]
    return 得分矩阵, 索引矩阵


class IndexRefineLowDim:
    def __init__(Self, index: faiss.Index = None, low_dim: int = None, reduce_dim_mode: str = "mrl"):
        Self.索引 = index
        Self.低维维度 = low_dim
        Self.高维维度 = None
        Self.降维模式 = reduce_dim_mode.lower()      # "mrl" or "pca"
        Self.低维向量 = None
        Self.原始向量 = None
        Self.PCA均值 = None
        Self.PCA投影矩阵 = None

        Self.k_factor = 10
    def reduce_dim(Self, x):
        if Self.降维模式 == "pca":
            if Self.PCA均值 is None or Self.PCA投影矩阵 is None:
                目标维度 = int(Self.低维维度)
                均值 = numpy.mean(x, axis=0, keepdims=True)
                数据中心 = x - 均值
                if x.shape[0] > x.shape[1]:
                    协方差 = 数据中心.T @ 数据中心
                    特征值, 特征向量 = numpy.linalg.eigh(协方差)
                    排序 = numpy.argsort(特征值)[::-1]
                    投影矩阵 = 特征向量[:, 排序[:目标维度]]
                else:
                    _, _, Vt = numpy.linalg.svd(数据中心, full_matrices=False)
                    投影矩阵 = Vt[:目标维度, :].T
                x = numpy.dot(数据中心, 投影矩阵)
                Self.PCA均值 = 均值
                Self.PCA投影矩阵 = 投影矩阵
            else:
                x = numpy.dot(x - Self.PCA均值, Self.PCA投影矩阵)
        elif Self.降维模式 == "mrl":
            x = x[:, :Self.低维维度]
        return x
    def search(Self, queries, k):
        返回D = numpy.empty((len(queries), k), dtype='float32')
        返回I = numpy.empty((len(queries), k), dtype='int64')
        返回I.fill(-1)
        总向量数 = Self.原始向量.shape[0] if Self.原始向量 is not None else 0
        if 总向量数 == 0:
            return 返回D, 返回I
        粗排k = int(min(k * Self.k_factor, 总向量数))
        _, 粗排索引 = Self.索引.search(Self.reduce_dim(queries), 粗排k)
        return 索引重排(Self.原始向量, 粗排索引, queries, k)    
    def add(Self, x):
        if Self.原始向量 is None:
            Self.原始向量 = x
            Self.高维维度 = x.shape[1]
            Self.低维向量 = Self.reduce_dim(x)
        else:
            Self.原始向量 = numpy.concatenate([Self.原始向量, x], axis=0)
            新低维 = Self.reduce_dim(x)
            Self.低维向量 = numpy.concatenate([Self.低维向量, 新低维], axis=0)
        Self.索引.add(Self.reduce_dim(x))
    def train(Self, x):
        Self.索引.train(Self.reduce_dim(x))
    def save(Self, filename: str):
        if isinstance(Self.索引, faiss.Index):
            序列化索引 = {"type": "faiss", "data": faiss.serialize_index(Self.索引)}
        else:
            序列化索引 = {"type": "gsq", "data": IndexGSQ.serialize_index(Self.索引)}
        with open(filename, 'wb') as f:
            pickle.dump({
                "模式": "IndexRefineLowDim",
                "索引": 序列化索引,
                "低维维度": Self.低维维度,
                "高维维度": Self.高维维度,
                "降维模式": Self.降维模式,
                "低维向量": Self.低维向量,
                "原始向量": Self.原始向量,
                "PCA均值": Self.PCA均值,
                "PCA投影矩阵": Self.PCA投影矩阵,
                "k_factor": Self.k_factor,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def load(filename: str):
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    index = globals()[d["模式"]]()
    for k, v in d.items():
        if k == "索引" and v is not None:
            if isinstance(v, dict) and "type" in v and "data" in v:
                if v["type"] == "faiss":
                    v = faiss.deserialize_index(v["data"])
                elif v["type"] == "gsq":
                    v = IndexGSQ.deserialize_index(v["data"])
            elif isinstance(v, bytes):
                if v.startswith(b'\x80'):
                    v = IndexGSQ.deserialize_index(v)
                else:
                    v = faiss.deserialize_index(v)
            elif isinstance(v, str):
                v = faiss.read_index(v)
            elif isinstance(v, numpy.ndarray) and v.dtype == numpy.uint8:
                v = faiss.deserialize_index(v)
        setattr(index, k, v)
    return index
def read_index(filename: str):
    return load(filename)
def write_index(index, filename: str):
    index.save(filename)
    
class Index:
    def __init__(Self, App: Config):
        Self.Config = App.Config
        Self.日志 = App.日志
        Self.Locale = App.Locale
        Self.Lang = App.Lang
        Self.tqdm = App.RichTqdm
        Self.量化映射 = {
            "Q4": faiss.ScalarQuantizer.QT_4bit,
            "Q6": faiss.ScalarQuantizer.QT_6bit,
            "Q8": faiss.ScalarQuantizer.QT_8bit,
            "F16": faiss.ScalarQuantizer.QT_fp16,
            "BF16": faiss.ScalarQuantizer.QT_bf16,
            "GSQ2": 2,
            "GSQ3": 3,
            "GSQ4": 4,
            "GSQ5": 5,
            "GSQ6": 6,
            "GSQ8": 8,
        }
        Self.嵌套包装类型 = frozenset(Self.Config.INDEX_CONFIG_NEST)
        Self.独立训练类型 = frozenset(Self.Config.INDEX_CONFIG_TRAIN)

    def _取配置(Self, 主, 主键, 副键):
        return getattr(Self.Config, 主键) if 主 else getattr(Self.Config, 副键)

    def _应用SQ范围统计(Self, sq, 主):
        _RE_MINMAX = Self._取配置(主, "INDEX_RE_MINMAX", "VEC_RERANKER_INDEX_RE_MINMAX")
        if _RE_MINMAX:
            sq.rangestat = faiss.ScalarQuantizer.RS_minmax
            sq.rangestat_arg = _RE_MINMAX
        _RE_MEANSTD = Self._取配置(主, "INDEX_RE_MEANSTD", "VEC_RERANKER_INDEX_RE_MEANSTD")
        if _RE_MEANSTD:
            sq.rangestat = faiss.ScalarQuantizer.RS_meanstd
            sq.rangestat_arg = _RE_MEANSTD
        _RE_QUANTILES = Self._取配置(主, "INDEX_RE_QUANTILES", "VEC_RERANKER_INDEX_RE_QUANTILES")
        if _RE_QUANTILES:
            sq.rangestat = faiss.ScalarQuantizer.RS_quantiles
            sq.rangestat_arg = _RE_QUANTILES
        _RE_OPTIM = Self._取配置(主, "INDEX_RE_OPTIM", "VEC_RERANKER_INDEX_RE_OPTIM")
        if _RE_OPTIM:
            sq.rangestat = faiss.ScalarQuantizer.RS_optim
            sq.rangestat_arg = _RE_OPTIM

    def 构建索引节点(Self, 规格, 向量维度, 向量数量, 主, 深度=0):
        """递归构建（可任意嵌套）的索引节点。
        规格支持:
          - 字符串:            "IP" / "L2" / "HNSW" / "IVFSQ" ...
          - 数组(任意嵌套):    [类型, 子规格?, 量化名?]
            例: ["Refine", ["IVFPQ", "IP"]]  ->  Refine 包 IVFPQ, IVFPQ 用 IP 做粗量化器
        返回 (faiss索引对象, 需要训练: bool)
        """
        if 深度 > 64:
            Self.日志("log.index.mode.not", info_level=4)
            return faiss.IndexFlatIP(向量维度), False

        if isinstance(规格, (list, tuple)):
            项 = list(规格)
            类型 = 项[0] if 项 else None
            子规格 = 项[1] if len(项) > 1 else None
            量化名 = 项[2] if len(项) > 2 else None
        else:
            类型, 子规格, 量化名 = 规格, None, None

        专家总数nlist = max(1, int(numpy.sqrt(向量数量)))
        激活专家数nprobe = min(专家总数nlist, max(1, int(专家总数nlist / Self._取配置(主, "INDEX_IVF_NLIST", "VEC_RERANKER_INDEX_IVF_NLIST"))))
        HNSW_M = Self._取配置(主, "INDEX_HNSW_M", "VEC_RERANKER_INDEX_HNSW_M")
        HNSW_EFC = Self._取配置(主, "INDEX_HNSW_CONSTRUCTION", "VEC_RERANKER_INDEX_HNSW_CONSTRUCTION")
        HNSW_EFS = Self._取配置(主, "INDEX_HNSW_SEARCH", "VEC_RERANKER_INDEX_HNSW_SEARCH")
        默认子规格 = "IP"
        if 量化名 and 量化名 in Self.量化映射:
            节点量化类型 = Self.量化映射[量化名]
        else:
            节点量化类型 = Self.量化映射[Self._取配置(主, "INDEX_SQ", "VEC_RERANKER_INDEX_SQ")]

        if 类型 == "IP":
            return faiss.IndexFlatIP(向量维度), False
        if 类型 == "L2":
            return faiss.IndexFlatL2(向量维度), False
        if 类型 == "HNSW":
            向量索引 = faiss.IndexHNSWFlat(向量维度, HNSW_M)
            向量索引.hnsw.efConstruction = HNSW_EFC
            向量索引.hnsw.efSearch = HNSW_EFS
            return 向量索引, True
        if 类型 == "HNSWSQ":
            向量索引 = faiss.IndexHNSWSQ(向量维度, 节点量化类型, HNSW_M)
            向量索引.hnsw.efConstruction = HNSW_EFC
            向量索引.hnsw.efSearch = HNSW_EFS
            Self._应用SQ范围统计(向量索引.sq, 主)
            return 向量索引, True
        if 类型 == "HNSWPQ":
            PQ_M = Self._取配置(主, "INDEX_HNSW_PQ_M", "VEC_RERANKER_INDEX_HNSW_PQ_M")
            NBITS = Self._取配置(主, "INDEX_HNSW_NBITS", "VEC_RERANKER_INDEX_HNSW_NBITS")
            向量索引 = faiss.IndexHNSWPQ(向量维度, HNSW_M, PQ_M, NBITS)
            向量索引.hnsw.efConstruction = HNSW_EFC
            向量索引.hnsw.efSearch = HNSW_EFS
            return 向量索引, True
        if 类型 == "NSGFlat":
            NSG_R = int(Self._取配置(主, "INDEX_NSG_R", "VEC_RERANKER_INDEX_NSG_R"))
            向量索引 = faiss.IndexNSGFlat(向量维度, NSG_R, faiss.METRIC_L2)
            try: 向量索引.nsg.search_L = int(Self._取配置(主, "INDEX_NSG_SEARCH", "VEC_RERANKER_INDEX_NSG_SEARCH"))
            except Exception: pass
            return 向量索引, False
        if 类型 == "NSGSQ":
            NSG_R = int(Self._取配置(主, "INDEX_NSG_R", "VEC_RERANKER_INDEX_NSG_R"))
            向量索引 = faiss.IndexNSGSQ(向量维度, 节点量化类型, NSG_R, faiss.METRIC_L2)
            try: 向量索引.nsg.search_L = int(Self._取配置(主, "INDEX_NSG_SEARCH", "VEC_RERANKER_INDEX_NSG_SEARCH"))
            except Exception: pass
            return 向量索引, True
        if 类型 == "NSGPQ":
            NSG_R = int(Self._取配置(主, "INDEX_NSG_R", "VEC_RERANKER_INDEX_NSG_R"))
            NSG_PQ_M = int(Self._取配置(主, "INDEX_NSG_PQ_M", "VEC_RERANKER_INDEX_NSG_PQ_M"))
            NSG_NBITS = int(Self._取配置(主, "INDEX_NSG_NBITS", "VEC_RERANKER_INDEX_NSG_NBITS"))
            向量索引 = faiss.IndexNSGPQ(向量维度, NSG_PQ_M, NSG_R, NSG_NBITS)
            try: 向量索引.nsg.search_L = int(Self._取配置(主, "INDEX_NSG_SEARCH", "VEC_RERANKER_INDEX_NSG_SEARCH"))
            except Exception: pass
            return 向量索引, True
        if 类型 == "Refine":
            降维模式 = Self._取配置(主, "INDEX_REFINE_LOW_DIM_MODE", "VEC_RERANKER_INDEX_REFINE_LOW_DIM_MODE")
            降维模式 = 降维模式 if 降维模式 is None else 降维模式.lower()
            粗排维度 = int(Self._取配置(主, "INDEX_REFINE_LOW_DIM_DIM", "VEC_RERANKER_INDEX_REFINE_LOW_DIM_DIM")) if (降维模式 is not None) or (降维模式 == "mrl") else 向量维度
            基础索引对象, 基础训练 = Self.构建索引节点(子规格 if 子规格 is not None else 默认子规格, 粗排维度, 向量数量, 主, 深度 + 1)
            if 降维模式 == "pca":
                PCA = faiss.PCAMatrix(向量维度, 粗排维度)
                基础索引对象 = faiss.IndexPreTransform(PCA, 基础索引对象)
            向量索引 = faiss.IndexRefineFlat(基础索引对象)
            向量索引.k_factor = Self._取配置(主, "INDEX_REFINEFLAT_K_FACTOR", "VEC_RERANKER_INDEX_REFINEFLAT_K_FACTOR")
            return 向量索引, 基础训练
        if 类型 == "RefineLowDim": # Fiass包装
            低维维度 = int(Self._取配置(主, "INDEX_REFINE_LOW_DIM_DIM", "VEC_RERANKER_INDEX_REFINE_LOW_DIM_DIM"))
            基础索引对象, 基础训练 = Self.构建索引节点(子规格 if 子规格 is not None else 默认子规格, 低维维度, 向量数量, 主, 深度 + 1)
            向量索引 = IndexRefineLowDim(基础索引对象, 低维维度, Self._取配置(主, "INDEX_REFINE_LOW_DIM_MODE", "VEC_RERANKER_INDEX_REFINE_LOW_DIM_MODE"))
            向量索引.k_factor = Self._取配置(主, "INDEX_REFINEFLAT_K_FACTOR", "VEC_RERANKER_INDEX_REFINEFLAT_K_FACTOR")
            return 向量索引, 基础训练
        if 类型 == "IVFPQR":
            量化器, _ = Self.构建索引节点(子规格 if 子规格 is not None else 默认子规格, 向量维度, 向量数量, 主, 深度 + 1)
            PQ_M = int(Self._取配置(主, "INDEX_IVF_PQ_M", "VEC_RERANKER_INDEX_IVF_PQ_M"))
            NBITS = int(Self._取配置(主, "INDEX_IVF_NLITS", "VEC_RERANKER_INDEX_IVF_NLITS"))
            M_REFINE = int(Self._取配置(主, "INDEX_IVFPQR_M_REFINE", "VEC_RERANKER_INDEX_IVFPQR_M_REFINE"))
            NBITS_REFINE = int(Self._取配置(主, "INDEX_IVFPQR_NBITS_REFINE", "VEC_RERANKER_INDEX_IVFPQR_NBITS_REFINE"))
            向量索引 = faiss.IndexIVFPQR(量化器, 向量维度, 专家总数nlist, PQ_M, NBITS, M_REFINE, NBITS_REFINE)
            向量索引.nprobe = 激活专家数nprobe
            return 向量索引, True
        if 类型 in ("IVF", "IVFSQ", "IVFPQ"):
            量化器, _ = Self.构建索引节点(子规格 if 子规格 is not None else 默认子规格, 向量维度, 向量数量, 主, 深度 + 1)
            度量 = 量化器.metric_type
            if 类型 == "IVF":
                向量索引 = faiss.IndexIVFFlat(量化器, 向量维度, 专家总数nlist, 度量)
            elif 类型 == "IVFSQ":
                by_res = Self._取配置(主, "INDEX_IVF_RQ", "VEC_RERANKER_INDEX_IVF_RQ")
                if 度量 == faiss.METRIC_INNER_PRODUCT:
                    by_res = False
                向量索引 = faiss.IndexIVFScalarQuantizer(量化器, 向量维度, 专家总数nlist, 节点量化类型, 度量, by_res)
                Self._应用SQ范围统计(向量索引.sq, 主)
            else:
                PQ_M = Self._取配置(主, "INDEX_IVF_PQ_M", "VEC_RERANKER_INDEX_IVF_PQ_M")
                NBITS = Self._取配置(主, "INDEX_IVF_NLITS", "VEC_RERANKER_INDEX_IVF_NLITS")
                向量索引 = faiss.IndexIVFPQ(量化器, 向量维度, 专家总数nlist, PQ_M, NBITS, 度量)
                if 度量 == faiss.METRIC_INNER_PRODUCT:
                    try:
                        向量索引.by_residual = False
                    except Exception:
                        pass
            向量索引.nprobe = 激活专家数nprobe
            return 向量索引, True
        if 类型 == "GSQFast":
            return IndexGSQ.IndexGSQKCosineFast(app=Self, quantization=Self.量化映射[Self.Config.INDEX_SQ if 主 else Self.Config.VEC_RERANKER_INDEX_SQ]), True

        Self.日志("log.index.mode.not", info_level=2)
        return faiss.IndexFlatIP(向量维度), False

    def 基础索引(Self, 向量维度, 训练, 模式, 向量重排模式=True):
        向量索引, 需要训练 = Self.构建索引节点(模式, 向量维度, 1, 向量重排模式)
        return 向量索引, (训练 or 需要训练)

    def 构建索引(Self, 向量文件, 模式=None, 量化=None, 向量重排模式=False):
        if hasattr(向量文件, 'get'): 向量文件 = 向量文件.get()
        if GPU_ACC and hasattr(向量文件, 'get'): 向量文件 = 向量文件.get()
        向量文件 = numpy.ascontiguousarray(向量文件, dtype=numpy.float32).copy()
        faiss.normalize_L2(向量文件)
        faiss.omp_set_num_threads(max(1, int(Self.Config.INDEX_CPU_COUNT) if isinstance(Self.Config.INDEX_CPU_COUNT, numpy.uint32) or Self.Config.INDEX_CPU_COUNT > 1 else int(os.cpu_count() * Self.Config.INDEX_CPU_COUNT)))
        Self.日志("log.core.index.generate.start", info_level=0)
        向量数量, 向量维度 = 向量文件.shape
        主 = not 向量重排模式

        if not 模式: 模式 = Self.Config.INDEX_MODE
        模式 = 模式 if 主 else Self.Config.VEC_RERANKER_INDEX_MODE
        量化 = (量化 or Self.Config.INDEX_SQ) if 主 else Self.Config.VEC_RERANKER_INDEX_SQ
        向量索引, 需要训练 = Self.构建索引节点(模式, 向量维度, 向量数量, 主)

        if 需要训练:
            _SAMPLING = Self._取配置(主, "INDEX_SAMPLING", "VEC_RERANKER_INDEX_SAMPLING")
            _SAMPLING_MIN = Self._取配置(主, "INDEX_SAMPLING_MIN", "VEC_RERANKER_INDEX_SAMPLING_MIN")
            采样数量 = int(_SAMPLING) if isinstance(_SAMPLING, numpy.uint32) or _SAMPLING > 1 else int(向量数量 * _SAMPLING)
            采样数量 = max(_SAMPLING_MIN, 采样数量)
            采样数量 = min(采样数量, 向量数量)
            采样向量 = 向量文件[numpy.random.choice(向量数量, 采样数量, replace=False)] if 采样数量 < 向量数量 else 向量文件
            for _ in Self.tqdm(range(1), desc="tqdm.index.train"):
                try:
                    向量索引.train(采样向量)
                except RuntimeError:
                    Self.日志("log.core.index.train.sampling.fallback", info_level=1)
                    向量索引.train(向量文件)
                    

        for _ in Self.tqdm(range(1), desc="tqdm.index.build"):
            向量索引.add(向量文件)

        Self.日志("log.core.index.generate.end", info_level=0)
        return 向量索引
