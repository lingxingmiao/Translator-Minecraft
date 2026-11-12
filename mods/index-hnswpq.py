# 可用配置(覆盖键 MOD_INDEX_HNWSPQ_<键>, 全大写):
#   M       每个节点的邻居数   32
#   EFC     构建期搜索广度     640
#   EFS     查询期搜索广度     240
#   PQM     PQ 子空间数        16
#   NBITS   每个子空间的位数   8

from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-hnwspq", {"M": 32, "EFC": 640, "EFS": 240, "PQM": 128, "NBITS": 8})
M, Efc, Efs = int(参数["M"]), int(参数["EFC"]), int(参数["EFS"])
PqM, Nbits = int(参数["PQM"]), int(参数["NBITS"])


@Mods().注册索引类型("HNSWPQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexHNSWPQ(向量.shape[1], M, PqM, Nbits)
    结果.hnsw.efConstruction = Efc
    结果.hnsw.efSearch = Efs
    return 结果, True
