from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-hnsw", {"M": 32, "Efc": 640, "Efs": 240})
M, Efc, Efs = int(参数["M"]), int(参数["Efc"]), int(参数["Efs"])


@Mods().注册索引类型("HNSW", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexHNSWFlat(向量.shape[1], M)
    结果.hnsw.efConstruction = Efc
    结果.hnsw.efSearch = Efs
    return 结果, True
