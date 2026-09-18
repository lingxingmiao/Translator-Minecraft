from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-hnwsq", {"Quant": "Q8", "M": 32, "Efc": 640, "Efs": 240})
Quant, M, Efc, Efs = 参数["Quant"], int(参数["M"]), int(参数["Efc"]), int(参数["Efs"])


@Mods().注册索引类型("HNSWSQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexHNSWSQ(向量.shape[1], Self.量化映射[Quant], M)
    结果.hnsw.efConstruction = Efc
    结果.hnsw.efSearch = Efs
    return 结果, True
