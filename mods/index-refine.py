from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-refine", {"LowDimMode": None, "LowDimDim": 64, "KFactor": 10.0})
LowDimMode, LowDimDim, KFactor = 参数["LowDimMode"], int(参数["LowDimDim"]), float(参数["KFactor"])


@Mods().注册索引类型("Refine", 数据源="Vector", 支持增量=True, 包装型=True)
def 构建(Self, 向量, 索引=None):
    维度, 数量 = 向量.shape[1], 向量.shape[0]
    模式 = LowDimMode.lower() if isinstance(LowDimMode, str) else None
    粗排维度 = LowDimDim if 模式 else 维度
    基础索引, 基础训练 = Self.构建索引节点(Self.当前子规格 or "IP", 粗排维度, 数量, Self.构建主, Self.当前深度 + 1)
    if 模式 == "pca":
        基础索引 = faiss.IndexPreTransform(faiss.PCAMatrix(维度, 粗排维度), 基础索引)
    结果 = faiss.IndexRefineFlat(基础索引)
    结果.k_factor = KFactor
    return 结果, 基础训练
