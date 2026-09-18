from TranslatorLib import Mods, faiss

@Mods().注册索引类型("IP", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    return faiss.IndexFlatIP(向量.shape[1]), False
