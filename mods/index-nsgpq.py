# 可用配置(覆盖键 MOD_INDEX_NSGPQ_<键>, 全大写):
#   R         每个节点的邻居数   64
#   L         构建期候选数       144
#   C         裁剪候选数         212
#   SEARCHL   查询期候选数       240
#   PQM       PQ 子空间数        16
#   NBITS     每个子空间的位数   8

from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-nsgpq", {"R": 64, "L": 128, "C": 512, "SEARCHL": 240, "PQM": 128, "NBITS": 8})
R, L, C = int(参数["R"]), int(参数["L"]), int(参数["C"])
SearchL, PqM, Nbits = int(参数["SEARCHL"]), int(参数["PQM"]), int(参数["NBITS"])


@Mods().注册索引类型("NSGPQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexNSGPQ(向量.shape[1], PqM, R, Nbits)
    try: 结果.nsg.L = L
    except Exception: pass
    try: 结果.nsg.C = C
    except Exception: pass
    try: 结果.nsg.search_L = SearchL
    except Exception: pass
    return 结果, True
