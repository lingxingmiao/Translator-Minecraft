# 可用配置(覆盖键 MOD_INDEX_SQ_<键>, 全大写):
#   QUANT     标量量化类型        "Q8"    可选: Q4 Q6 Q8 F16 BF16

from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-sq", {"QUANT": "Q8"})
Quant = str(参数["QUANT"]).upper()


@Mods().注册索引类型("SQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexScalarQuantizer(向量.shape[1], Self.量化映射[Quant])
    return 结果, True
