from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-nsgsq", {"Quant": "Q8", "R": 64, "L": 144, "C": 212, "SearchL": 240})
Quant, R, L, C = 参数["Quant"], int(参数["R"]), int(参数["L"]), int(参数["C"])
SearchL = int(参数["SearchL"])


@Mods().注册索引类型("NSGSQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexNSGSQ(向量.shape[1], Self.量化映射[Quant], R, faiss.METRIC_L2)
    try: 结果.nsg.L = L
    except Exception: pass
    try: 结果.nsg.C = C
    except Exception: pass
    try: 结果.nsg.search_L = SearchL
    except Exception: pass
    return 结果, True
