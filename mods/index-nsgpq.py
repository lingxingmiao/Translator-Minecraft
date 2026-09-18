from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-nsgpq", {"R": 64, "L": 144, "C": 212, "SearchL": 240, "PqM": 16, "Nbits": 8})
R, L, C = int(参数["R"]), int(参数["L"]), int(参数["C"])
SearchL, PqM, Nbits = int(参数["SearchL"]), int(参数["PqM"]), int(参数["Nbits"])


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
