# 可用配置(覆盖键 MOD_INDEX_PQ_<键>, 全大写):
#   PQM      PQ 子空间数           16     (需整除向量维度)
#   NBITS    每个子空间的位数       8

from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-pq", {"PQM": 16, "NBITS": 8})
PqM, Nbits = int(参数["PQM"]), int(参数["NBITS"])


@Mods().注册索引类型("PQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexPQ(向量.shape[1], PqM, Nbits)
    return 结果, True
