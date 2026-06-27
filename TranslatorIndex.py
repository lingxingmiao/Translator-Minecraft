from TranslatorLib import (os, faiss, numpy,
                           IndexGSQ, Log, Locale, RuntimeConfig)

class Index:
    def __init__(Self, Config: dict = None):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.日志 = Log(Config=Config).写入日志
        Self.Locale = Locale(Config=Config)
        Self.Lang = Self.Locale.Lang
        Self.tqdm = Self.Locale.Tqdm
        Self.量化映射 = {
            "Q4": faiss.ScalarQuantizer.QT_4bit,
            "Q6": faiss.ScalarQuantizer.QT_6bit,
            "Q8": faiss.ScalarQuantizer.QT_8bit,
            "F16": faiss.ScalarQuantizer.QT_fp16,
            "BF16": faiss.ScalarQuantizer.QT_bf16,
            "GSQ2": 2,
            "GSQ3": 3,
            "GSQ4": 4,
            "GSQ8": 8,
        }

    def 基础索引(Self, 向量维度, 训练, 模式, 量化类型, 向量重排模式=True):
        if 模式 == "HNSW":
            向量索引 = faiss.IndexHNSWFlat(向量维度, Self.Config.INDEX_HNSW_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_SEARCH
            训练 = True
        elif 模式 == "HNSWSQ":
            向量索引 = faiss.IndexHNSWSQ(向量维度, 量化类型, Self.Config.INDEX_HNSW_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_SEARCH
            训练 = True
        elif 模式 == "HNSWPQ":
            向量索引 = faiss.IndexHNSWPQ(
                向量维度, 
                Self.Config.INDEX_HNSW_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_M, 
                Self.Config.INDEX_HNSW_PQ_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_PQ_M, 
                Self.Config.INDEX_HNSW_NBITS if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_NBITS
            )
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_SEARCH
            训练 = True
        elif 模式 == "L2":
            向量索引 = faiss.IndexFlatL2(向量维度)
        elif 模式 == "IP":
            向量索引 = faiss.IndexFlatIP(向量维度)
        else: 
            Self.日志("log.index.mode.base.not", info_level=4)
        return 向量索引, 训练

    def 构建索引(Self, 向量文件, 模式=None, 基础模式=None, 量化=None, 基础量化=None, 向量重排模式=False):
        if hasattr(向量文件, 'get'): 向量文件 = 向量文件.get()
        faiss.omp_set_num_threads(max(1, int(Self.Config.INDEX_CPU_COUNT) if isinstance(Self.Config.INDEX_CPU_COUNT, numpy.uint32) or Self.Config.INDEX_CPU_COUNT > 1 else int(os.cpu_count() * Self.Config.INDEX_CPU_COUNT)))
        Self.日志("log.core.index.generate.start", info_level=0)
        训练, SQ = False, False
        向量数量, 向量维度 = 向量文件.shape
        
        if not 模式: 模式 = Self.Config.INDEX_MODE
        if not 基础模式: 基础模式 = Self.Config.INDEX_BASE_MODE
        if not 量化: 量化 = Self.Config.INDEX_SQ
        if not 基础量化: 基础量化 = Self.Config.INDEX_BASE_SQ
        
        量化类型 = Self.量化映射[量化]
        基础量化类型 = Self.量化映射[基础量化]
        向量重排模式 = not 向量重排模式
        if 模式 == "GSQ":
            向量索引 = IndexGSQ.IndexGSQKCosine(app=Self, quantization=量化类型)
        elif 模式 == "GSQFast":
            向量索引 = IndexGSQ.IndexGSQKCosineFast(app=Self, quantization=量化类型)
        elif 模式 == "GSQMoE":
            向量索引 = IndexGSQ.IndexGSQKCosineMoE(app=Self, quantization=量化类型)
            训练, SQ = True, False
        elif 模式 == "GSQMoEPlus":
            向量索引 = IndexGSQ.IndexGSQKCosineMoEPlus(app=Self, quantization=量化类型)
            训练, SQ = True, False
        elif 模式 == "HNSWSQ":
            训练, SQ = True, True
            向量索引 = faiss.IndexHNSWSQ(向量维度, 量化类型, Self.Config.INDEX_HNSW_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_SEARCH
        elif 模式 == "HNSWPQ":
            训练, SQ = True, False
            向量索引 = faiss.IndexHNSWPQ(
                向量维度, 
                Self.Config.INDEX_HNSW_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_M, 
                Self.Config.INDEX_HNSW_PQ_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_PQ_M, 
                Self.Config.INDEX_HNSW_NBITS if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_NBITS
            )
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_SEARCH
        elif 模式 == "HNSW":
            训练, SQ = True, False
            向量索引 = faiss.IndexHNSWFlat(向量维度, Self.Config.INDEX_HNSW_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_HNSW_SEARCH
        elif 模式 == "Refine":
            训练, SQ = False, False
            基础索引, 训练 = Self.基础索引(向量维度, 训练, 基础模式, 基础量化类型, 向量重排模式)
            向量索引 = faiss.IndexRefineFlat(基础索引)
            向量索引.k_factor = Self.Config.INDEX_REFINEFLAT_K_FACTOR if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_REFINEFLAT_K_FACTOR
            
        elif 模式 == "IVFSQ":
            训练, SQ = True, True
            基础索引, 训练 = Self.基础索引(向量维度, 训练, 基础模式, 基础量化类型, 向量重排模式)
            向量索引 = faiss.IndexIVFScalarQuantizer(
                基础索引, 
                向量维度, 
                Self.Config.INDEX_IVF_NLIST if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_NLIST, 
                量化类型, 
                by_residual=Self.Config.INDEX_IVF_RQ if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_RQ
            )
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_NPROBE
            
        elif 模式 == "IVFPQ":
            训练, SQ = True, False
            基础索引, 训练 = Self.基础索引(向量维度, 训练, 基础模式, 基础量化类型, 向量重排模式)
            向量索引 = faiss.IndexIVFPQ(
                基础索引, 
                向量维度, 
                Self.Config.INDEX_IVF_NLIST if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_NLIST, 
                Self.Config.INDEX_IVF_PQ_M if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_PQ_M, 
                Self.Config.INDEX_IVF_NLITS if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_NLITS
            )
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_NPROBE
            
        elif 模式 == "IVF":
            训练, SQ = True, False
            基础索引, 训练 = Self.基础索引(向量维度, 训练, 基础模式, 基础量化类型, 向量重排模式)
            向量索引 = faiss.IndexIVFFlat(
                基础索引, 
                向量维度, 
                Self.Config.INDEX_IVF_NLIST if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_NLIST, 
                faiss.METRIC_L2
            )
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_IVF_NPROBE
        elif 模式 == "L2":
            训练, SQ = False, False
            向量索引 = faiss.IndexFlatL2(向量维度)
        elif 模式 == "IP":
            训练, SQ = False, False
            向量索引 = faiss.IndexFlatIP(向量维度)
        else: 
            Self.日志("log.index.mode.not", info_level=4)
        if SQ:
            _RE_MINMAX = Self.Config.INDEX_RE_MINMAX if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_RE_MINMAX
            if _RE_MINMAX:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_minmax
                向量索引.sq.rangestat_arg = _RE_MINMAX
                
            _RE_MEANSTD = Self.Config.INDEX_RE_MEANSTD if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_RE_MEANSTD
            if _RE_MEANSTD:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_meanstd
                向量索引.sq.rangestat_arg = _RE_MEANSTD
                
            _RE_QUANTILES = Self.Config.INDEX_RE_QUANTILES if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_RE_QUANTILES
            if _RE_QUANTILES:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_quantiles
                向量索引.sq.rangestat_arg = _RE_QUANTILES
                
            _RE_OPTIM = Self.Config.INDEX_RE_OPTIM if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_RE_OPTIM
            if _RE_OPTIM:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_optim
                向量索引.sq.rangestat_arg = _RE_OPTIM
        if 训练:
            _SAMPLING = Self.Config.INDEX_SAMPLING if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_SAMPLING
            _SAMPLING_MIN = Self.Config.INDEX_SAMPLING_MIN if 向量重排模式 else Self.Config.VEC_RERANKER_INDEX_SAMPLING_MIN
            
            采样数量 = int(_SAMPLING) if isinstance(_SAMPLING, numpy.uint32) or _SAMPLING > 1 else int(向量数量 * _SAMPLING)
            采样向量 = 向量文件[numpy.random.choice(向量数量, min(向量数量, max(_SAMPLING_MIN, 采样数量)), replace=False)]
            
            for _ in Self.tqdm(range(1), desc="tqdm.index.train"):
                向量索引.train(采样向量)
                
        for _ in Self.tqdm(range(1), desc="tqdm.index.build"):
            向量索引.add(向量文件)
            
        Self.日志("log.core.index.generate.end", info_level=0)
        return 向量索引
