from TranslatorLib import (os, faiss, numpy, pickle, bm25s,
                           IndexGSQ, GPU_ACC, Config, Mods)

def load(filename: str):
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    index = Mods().索引类(d["模式"])()
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
            "GSQ1": 1,
            "GSTQ1": 1.585,
            "GSQ2": 2,
            "GSQ3": 3,
            "GSQ4": 4,
            "GSQ5": 5,
            "GSQ6": 6,
            "GSQ8": 8,
        }

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

    def 自动处理索引(Self, 向量索引, 主):
        sq = getattr(向量索引, "sq", None)
        if sq is not None:
            try: Self._应用SQ范围统计(sq, 主)
            except Exception: pass
        return 向量索引

    def 设置构建上下文(Self, 规格, 向量维度, 向量数量, 主, 深度=0):
        项 = list(规格) if isinstance(规格, (list, tuple)) else [规格]
        Self.构建维度 = 向量维度
        Self.构建数量 = 向量数量
        Self.构建主 = 主
        Self.当前深度 = 深度
        Self.当前类型 = 项[0] if 项 else None
        Self.当前子规格 = 项[1] if len(项) > 1 else None
        Self.当前量化名 = 项[2] if len(项) > 2 else None

    def 构建索引节点(Self, 规格, 向量维度, 向量数量, 主, 深度=0):
        if 深度 > 64:
            Self.日志("log.index.mode.not", info_level=4)
            return faiss.IndexFlatIP(向量维度), False
        Self.设置构建上下文(规格, 向量维度, 向量数量, 主, 深度)
        构建函数 = Mods().索引构建函数(规格)
        if 构建函数 is not None:
            return 构建函数(Self, numpy.empty((向量数量, 向量维度), dtype=numpy.float32))
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
        Self.设置构建上下文(模式, 向量维度, 向量数量, 主, 0)
        注册构建 = Mods().索引构建函数(模式)
        if 注册构建 is not None:
            向量索引, 需要训练 = Self.自动处理索引(注册构建(Self, 向量文件), 主) # 模组索引
        else:
            向量索引, 需要训练 = Self.构建索引节点(模式, 向量维度, 向量数量, 主) # 未注册类型回落

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

    def 构建索引BM25(Self, 文本文件):
        索引 = bm25s.BM25()
        源文本列表 = [i[0] for i in 文本文件]
        if Self.Config.INDEX_TEXT_LOWER:
            源文本列表 = [i.lower() for i in 源文本列表]
        for _ in Self.tqdm(range(1), desc="tqdm.index.build"):
            索引.index(Mods().BM25分词(Self, Self.Config.INDEX_LANGUAGE, 源文本列表))
        return 索引