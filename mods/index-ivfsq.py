# 可用配置(覆盖键 MOD_INDEX_IVFSQ_<键>, 全大写):
#   QUANT        标量量化类型       "Q8"
#   NLIST        聚类数系数         0.25
#   BYRESIDUAL   是否对残差量化     True

from TranslatorLib import Mods, faiss, np

参数 = Mods().模组配置("index-ivfsq", {"QUANT": "Q8", "NLIST": 0.25, "BYRESIDUAL": True})
Quant, Nlist, ByResidual = 参数["QUANT"], float(参数["NLIST"]), bool(参数["BYRESIDUAL"])


@Mods().注册索引类型("IVFSQ", 数据源="Vector", 支持增量=True, 包装型=True)
def 构建(Self, 向量, 索引=None):
    维度, 数量 = 向量.shape[1], 向量.shape[0]
    量化器, _ = Self.构建索引节点(Self.当前子规格 or "IP", 维度, 数量, Self.构建主, Self.当前深度 + 1)
    度量 = 量化器.metric_type
    by_res = ByResidual
    if 度量 == faiss.METRIC_INNER_PRODUCT:
        by_res = False
    nlist = max(1, int(np.sqrt(数量)))
    nprobe = min(nlist, max(1, int(nlist / Nlist)))
    结果 = faiss.IndexIVFScalarQuantizer(量化器, 维度, nlist, Self.量化映射[Quant], 度量, by_res)
    结果.nprobe = nprobe
    return 结果, True
