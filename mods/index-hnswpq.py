from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-hnwspq", {"M": 32, "Efc": 640, "Efs": 240, "PqM": 16, "Nbits": 8})
M, Efc, Efs = int(参数["M"]), int(参数["Efc"]), int(参数["Efs"])
PqM, Nbits = int(参数["PqM"]), int(参数["Nbits"])


@Mods().注册索引类型("HNSWPQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexHNSWPQ(向量.shape[1], M, PqM, Nbits)
    结果.hnsw.efConstruction = Efc
    结果.hnsw.efSearch = Efs
    return 结果, True
