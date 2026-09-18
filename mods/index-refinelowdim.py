from TranslatorLib import Mods, njit, _prange, numpy, faiss, pickle, IndexGSQ

参数 = Mods().模组配置("index-refinelowdim", {"LowDimMode": "mrl", "LowDimDim": 64, "KFactor": 10.0})
LowDimMode, LowDimDim, KFactor = 参数["LowDimMode"], int(参数["LowDimDim"]), float(参数["KFactor"])


@njit(cache=True, parallel=True, fastmath=True)
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


@Mods().注册索引类("IndexRefineLowDim")
@Mods().注册索引类("RefineLowDim")
class IndexRefineLowDim:
    def __init__(Self, index: faiss.Index = None, low_dim: int = None, reduce_dim_mode: str = "mrl"):
        Self.索引 = index
        Self.低维维度 = low_dim
        Self.高维维度 = None
        Self.降维模式 = reduce_dim_mode.lower() if reduce_dim_mode else "mrl"
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


@Mods().注册索引类型("RefineLowDim", 数据源="Vector", 支持增量=True, 包装型=True)
def 构建(Self, 向量, 索引=None):
    基础索引, 基础训练 = Self.构建索引节点(Self.当前子规格 or "IP", LowDimDim, 向量.shape[0], Self.构建主, Self.当前深度 + 1)
    结果 = IndexRefineLowDim(基础索引, LowDimDim, LowDimMode)
    结果.k_factor = KFactor
    return 结果, 基础训练
