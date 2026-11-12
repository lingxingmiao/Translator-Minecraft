# 可用配置(覆盖键 MOD_INDEX_NSG_<键>, 全大写):
#   R         每个节点的邻居数   64
#   L         构建期候选数       144
#   C         裁剪候选数         212
#   SEARCHL   查询期候选数       240

from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-nsg", {"R": 64, "L": 128, "C": 512, "SEARCHL": 240})
R, L, C, SearchL = int(参数["R"]), int(参数["L"]), int(参数["C"]), int(参数["SEARCHL"])


@Mods().注册索引类型("NSG", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexNSGFlat(向量.shape[1], R, faiss.METRIC_L2)
    try: 结果.nsg.L = L
    except Exception: pass
    try: 结果.nsg.C = C
    except Exception: pass
    try: 结果.nsg.search_L = SearchL
    except Exception: pass
    return 结果, False
