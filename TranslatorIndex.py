from TranslatorLib import (os, faiss, numpy, pickle, bm25s, eb,
                           GPU_ACC, Config, Mods)
def 转多GPU索引(索引库, 向量索引, 设备列表, 分片=True, 资源列表=None): #faiss CPU索引转多GPU索引 失败抛出异常由调用方回退CPU
    资源 = 资源列表 if 资源列表 else [索引库.StandardGpuResources() for _ in 设备列表]
    选项 = 索引库.GpuMultipleClonerOptions()
    选项.shard = bool(分片)
    try: # conda/faiss构建把vector<GpuResourcesProvider*>暴露成GpuResourcesVector 只收Python列表的自编译wheel才走下面那条
        return 索引库.index_cpu_to_gpu_multiple_py(资源, 向量索引, 选项, 设备列表), 资源
    except (AttributeError, TypeError):
        return 索引库.index_cpu_to_gpu_multiple(资源, 设备列表, 向量索引, 选项), 资源
def GPU设备算力(设备列表): # 直接问驱动要每张卡的算力 不依赖 nvidia-smi/CuPy 拿不到就返回空(不拦)
    try:
        import ctypes
        驱动 = None
        for 名称 in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
            try:
                驱动 = ctypes.CDLL(名称)
                break
            except OSError:
                continue
        if 驱动 is None or 驱动.cuInit(0) != 0: return {}
        算力表 = {}
        for 序号 in 设备列表:
            句柄 = ctypes.c_int()
            if 驱动.cuDeviceGet(ctypes.byref(句柄), int(序号)) != 0: continue
            主算力, 次算力 = ctypes.c_int(), ctypes.c_int()
            if 驱动.cuDeviceComputeCapability(ctypes.byref(主算力), ctypes.byref(次算力), 句柄) == 0:
                算力表[int(序号)] = 主算力.value * 10 + 次算力.value
        return 算力表
    except Exception:
        return {}
def GPU自检(实例, 索引库, 设备列表): # 返回可用设备列表(算力初筛 + 最小搜索自检), 空列表表示回退CPU
    最低算力 = int(getattr(实例.Config, "INDEX_GPU_MIN_COMPUTE", 0) or 0)
    缓存 = getattr(实例, "GPU自检缓存", None)
    if 缓存 is None:
        缓存 = {}
        实例.GPU自检缓存 = 缓存
    键 = (tuple(设备列表), 最低算力)
    if 键 not in 缓存:
        可用设备 = list(设备列表)
        if 最低算力: # CUDA 12 的 nvcc 最低只编到 sm_50; 更老的卡(Kepler 3.x)连 cuBLAS 初始化都会
            算力表 = GPU设备算力(可用设备) # FAISS_ASSERT(abort) 崩掉, 那种崩法 try/except 拦不住, 只能提前筛掉
            for 序号 in list(可用设备):
                算力 = 算力表.get(int(序号))
                if 算力 is not None and 算力 < 最低算力:
                    实例.日志("log.core.index.gpu.device.skip", device=序号, cc=算力, min=最低算力, info_level=2)
                    可用设备.remove(序号)
        try:
            if 可用设备:
                探针 = 索引库.IndexFlatIP(8)
                探针.add(numpy.zeros((1, 8), dtype=numpy.float32))
                探针GPU, _ = 转多GPU索引(索引库, 探针, 可用设备, True)
                探针GPU.search(numpy.zeros((1, 8), dtype=numpy.float32), 1)
            缓存[键] = 可用设备
        except Exception:
            实例.日志("log.core.index.gpu.selftest.fallback", e=eb.format_exc(), info_level=2)
            缓存[键] = []
    return 缓存[键]
def 是GPU索引(索引库, 索引, 设备数): # 转换结果是否真的落在GPU上(复合索引看每个子索引)
    try: # faiss对没有GPU实现的类型(NSG/HNSW等)不会报错 而是回一个CPU副本 甚至是被切片的坏对象
        if 索引库.isGpuIndex(索引): return True
    except Exception: pass
    if not hasattr(索引, "at"): return False
    for 序 in range(max(1, int(设备数))):
        try:
            if not 索引库.isGpuIndex(索引.at(序)): return False
        except Exception:
            return False
    return True
def GPU不安全(索引库, 索引, 深度=0): # 会上GPU但被faiss静默改语义的类型, 返回原因字符串; 安全返回None
    if 索引 is None or 深度 > 4: return None
    try: 索引 = faiss.downcast_index(索引) # faiss 把内层索引暴露成基类 Index, 不 downcast 看不到真实类型
    except Exception: pass
    if "IVFPQR" in type(索引).__name__: # ToGpuCloner 转成 GpuIndexIVFPQ 丢掉重排级, 实测CPU/GPU top-12只有约42%重叠
        return "faiss 会把 IndexIVFPQR 转成 GpuIndexIVFPQ 并丢掉重排级, 结果与 CPU 不同, 保持CPU"
    子列表 = []
    for 属性 in ("index", "base_index", "storage"): # IndexPreTransform / IndexRefine / HNSW*SQ,PQ 的内层
        子 = getattr(索引, 属性, None)
        if 子 is not None: 子列表.append(子)
    链 = getattr(索引, "chain", None)
    if 链 is not None:
        for 序 in range(16):
            try: 子列表.append(链.at(序))
            except Exception: break
    if hasattr(索引, "at"): # IndexShards / IndexReplicas
        for 序 in range(16):
            try: 子列表.append(索引.at(序))
            except Exception: break
    for 子 in 子列表:
        if 子 is 索引: continue
        原因 = GPU不安全(索引库, 子, 深度 + 1)
        if 原因: return 原因
    return None
def 从字典加载(d):
    index = Mods().索引类(d["模式"])()
    for k, v in d.items():
        if k == "索引" and v is not None:
            if isinstance(v, dict) and "type" in v and "data" in v:
                if v["type"] == "faiss":
                    v = faiss.deserialize_index(v["data"])
                elif v["type"] == "gsq":
                    v = deserialize_index(v["data"])
            elif isinstance(v, bytes):
                if v.startswith(b'\x80'):
                    v = deserialize_index(v)
                else:
                    v = faiss.deserialize_index(v)
            elif isinstance(v, str):
                v = faiss.read_index(v)
            elif isinstance(v, numpy.ndarray) and v.dtype == numpy.uint8:
                v = faiss.deserialize_index(v)
        setattr(index, k, v)
    return index
def load(filename: str):
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    return 从字典加载(d)
def read_index(filename: str):
    try:
        with open(filename, 'rb') as f:
            魔数 = f.read(4)
    except Exception:
        魔数 = b""
    if 魔数 == b"GGUF":                                  # 新格式: GGUF 存档(自带 mmap)
        from TranslatorPersistence import 向量GGUF读取
        d, _ = 向量GGUF读取(filename)
        return 从字典加载(d)
    return load(filename)
def write_index(index, filename: str):
    index.save(filename)
def index_cpu_to_gpu(index, **kwargs):
    index.gpu()
    return index
def index_gpu_to_cpu(index, **kwargs):
    index.cpu()
    return index
def serialize_index(index):
    return index.serialize()
def deserialize_index(data):
    if isinstance(data, bytes):
        return 从字典加载(pickle.loads(data))
    return 从字典加载(pickle.load(data))

class IndexGSQ:
    load = staticmethod(从字典加载)
    read_index = staticmethod(read_index)
    write_index = staticmethod(write_index)
    index_cpu_to_gpu = staticmethod(index_cpu_to_gpu)
    index_gpu_to_cpu = staticmethod(index_gpu_to_cpu)
    serialize_index = staticmethod(serialize_index)
    deserialize_index = staticmethod(deserialize_index)
    
class Index:
    def __init__(Self, App: Config):
        Self.应用 = App # TranslatorConfig 里 Index 先于 Module 构造(439/440), 所以 Module 只能延迟取
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

    @property
    def Module(Self): # 索引模组里 Self.Module.采样器(...) 的取法; 延迟到调用时再读, 避开 Config 构造顺序
        return getattr(Self.应用, "Module", None)

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

    def GPU资源(Self): # 构建用的常驻GPU资源 复用避免每个索引都新建一遍
        设备 = list(Self.Config.INDEX_GPU_DEVICE_IDS)
        if not 设备: return 设备, []
        if getattr(Self, "GPU资源缓存", None) is None or len(Self.GPU资源缓存) != len(设备):
            Self.GPU资源缓存 = [faiss.StandardGpuResources() for _ in 设备]
        return 设备, Self.GPU资源缓存

    def 转GPU索引(Self, 向量索引): # 构建阶段(GPU训练/添加)把索引搬到GPU 失败回退CPU不阻断构建
        设备, 资源 = Self.GPU资源()
        if not 设备: return None
        可用设备 = GPU自检(Self, faiss, 设备)
        if not 可用设备: return None
        不安全 = GPU不安全(faiss, 向量索引) # 会静默改语义的类型(IVFPQR)先拦掉
        if 不安全:
            Self.日志("log.core.index.gpu.build.fallback", e=不安全, info_level=2)
            return None
        try:
            新索引, _ = 转多GPU索引(faiss, 向量索引, 可用设备, str(Self.Config.INDEX_GPU_SEARCH_MODE).lower() == "indexshards", 资源 if len(资源) == len(可用设备) else None)
            if not 是GPU索引(faiss, 新索引, len(可用设备)): # NSG/HNSW 这类没有GPU实现的类型
                Self.日志("log.core.index.gpu.build.fallback", e="索引类型没有GPU实现, 保持CPU构建", info_level=2)
                return None
            Self.日志("log.core.index.gpu.build", mode=Self.Config.INDEX_GPU_SEARCH_MODE, devices=可用设备, info_level=0)
            return 新索引
        except Exception:
            Self.日志("log.core.index.gpu.build.fallback", e=eb.format_exc(), info_level=2)
            return None

    def 转CPU索引(Self, 向量索引): # 构建阶段的GPU索引搬回CPU 保证索引可写盘
        try:
            return faiss.index_gpu_to_cpu(向量索引)
        except Exception:
            Self.日志("log.core.index.gpu.build.fallback", e=eb.format_exc(), info_level=2)
            return 向量索引

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

        已上GPU = False
        if 需要训练:
            if Self.Config.INDEX_GPU_TRAIN:
                转换后 = Self.转GPU索引(向量索引)
                if 转换后 is not None:
                    向量索引 = 转换后
                    已上GPU = True
            
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
                    
            if 已上GPU and not Self.Config.INDEX_GPU_ADD:
                向量索引 = Self.转CPU索引(向量索引)
                已上GPU = False

        if Self.Config.INDEX_GPU_ADD and not 已上GPU:
            转换后 = Self.转GPU索引(向量索引)
            if 转换后 is not None:
                向量索引 = 转换后
                已上GPU = True

        for _ in Self.tqdm(range(1), desc="tqdm.index.build"):
            向量索引.add(向量文件)
            
        if 已上GPU:
            向量索引 = Self.转CPU索引(向量索引)

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