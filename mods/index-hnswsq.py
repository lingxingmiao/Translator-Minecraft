# 可用配置(覆盖键 MOD_INDEX_HNWSQ_<键>, 全大写):
#   QUANT   标量量化类型        "Q8"
#   M       每个节点的邻居数   32
#   EFC     构建期搜索广度     640
#   EFS     查询期搜索广度     240

from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-hnwsq", {"QUANT": "Q8", "M": 32, "EFC": 640, "EFS": 240})
Quant, M, Efc, Efs = 参数["QUANT"], int(参数["M"]), int(参数["EFC"]), int(参数["EFS"])


@Mods().注册索引类型("HNSWSQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexHNSWSQ(向量.shape[1], Self.量化映射[Quant], M)
    结果.hnsw.efConstruction = Efc
    结果.hnsw.efSearch = Efs
    return 结果, True
