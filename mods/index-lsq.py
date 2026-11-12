# 可用配置(覆盖键 MOD_INDEX_LSQ_<键>, 全大写):
#   PQM         子码本数(级数)       16    总位宽 = PQM × NBITS
#   NBITS       每级位数             8
#   TRAINITERS  码本训练迭代次数      25
#   ICM         ICM 精调迭代次数      4     每级编码时的局部搜索轮数
#   ENCODEILS   编码 ILS 迭代次数     16    最耗时的旋钮, 调小可显著加速(精度略降)
from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-lsq", {"PQM": 16, "NBITS": 8, "TRAINITERS": 25, "ICM": 4, "ENCODEILS": 16})
PqM, Nbits = int(参数["PQM"]), int(参数["NBITS"])
TrainIters, Icm, EncodeIls = int(参数["TRAINITERS"]), int(参数["ICM"]), int(参数["ENCODEILS"])


@Mods().注册索引类型("LSQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexLocalSearchQuantizer(向量.shape[1], PqM, Nbits)
    try: 结果.lsq.train_iters = TrainIters
    except Exception: pass
    try: 结果.lsq.icm_iters = Icm
    except Exception: pass
    try: 结果.lsq.encode_ils_iters = EncodeIls
    except Exception: pass
    return 结果, True
