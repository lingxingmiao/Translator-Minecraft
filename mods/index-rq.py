# 可用配置(覆盖键 MOD_INDEX_RQ_<键>, 全大写):
#   PQM         残差级数(子空间数)   16    总位宽 = PQM × NBITS
#   NBITS       每级位数             8
#   MAXBEAM     残差检索 beam 宽度    5     越大越准越慢
#   SEARCHTYPE  检索类型              ""    留空=默认(decompress); 构建期设定, 训练前生效
#                                           实测本语料只建议 decompress 或 LUT_nonorm:
#                                           decompress R@1 71.4% / 检索 1.27s; LUT_nonorm R@1 61.6% 但检索 0.09s(14x) 且 R@100 更高(86.5%)
#                                           norm_* 系列(norm_float/norm_qint8/norm_rq2x4...) 实测崩坏(R@1 0~37%), 不要用
from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-rq", {"PQM": 16, "NBITS": 8, "MAXBEAM": 5, "SEARCHTYPE": ""})
PqM, Nbits = int(参数["PQM"]), int(参数["NBITS"])
MaxBeam = int(参数["MAXBEAM"])
SearchType = str(参数["SEARCHTYPE"] or "").strip()


@Mods().注册索引类型("RQ", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    结果 = faiss.IndexResidualQuantizer(向量.shape[1], PqM, Nbits)
    try: 结果.rq.max_beam_size = MaxBeam
    except Exception: pass
    if SearchType:
        检索类型 = getattr(faiss.ResidualQuantizer, "ST_" + SearchType, None)
        if 检索类型 is not None:
            try: 结果.rq.search_type = 检索类型
            except Exception: pass
    return 结果, True
