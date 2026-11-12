# 可用配置(覆盖键 MOD_INDEX_OPQ_<键>, 全大写):
#   OPQM      旋转后维度(0=保持原维度)   0
#   ITER      旋转训练迭代次数            50
#   ITERPQ    每轮迭代内的 PQ 训练次数     4      (0 或负数 = 用 faiss 默认)
#
# 包装型: 子规格默认 "PQ"(由 MOD_INDEX_PQ_* 决定其 PQM/NBITS/NBITS),
#         例 ["OPQ", "PQ"] / ["OPQ", "IVFPQ"] / ["OPQ", ["PQ"]] 均可嵌套。

from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-opq", {"OPQM": 0, "ITER": 50, "ITERPQ": 4})
OpqM, Iter, IterPq = int(参数["OPQM"]), int(参数["ITER"]), int(参数["ITERPQ"])

@Mods().注册索引类型("OPQ", 数据源="Vector", 支持增量=True, 包装型=True)
def 构建(Self, 向量, 索引=None):
    维度, 数量 = 向量.shape[1], 向量.shape[0]
    输出维度 = 维度 if OpqM <= 0 else max(1, min(OpqM, 维度))
    子索引, 子训练 = Self.构建索引节点(Self.当前子规格 or "PQ", 输出维度, 数量, Self.构建主, Self.当前深度 + 1)
    旋转 = faiss.OPQMatrix(维度, 输出维度)
    try: 旋转.niter = Iter
    except Exception: pass
    if IterPq > 0:
        try: 旋转.niter_pq = IterPq
        except Exception: pass
    结果 = faiss.IndexPreTransform(旋转, 子索引)
    return 结果, True
