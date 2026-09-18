from TranslatorLib import Mods, faiss, np

参数 = Mods().模组配置("index-ivfpqr", {"Nlist": 0.25, "PqM": 16, "Nbits": 8, "MRefine": 16, "NbitsRefine": 8})
Nlist = float(参数["Nlist"])
PqM, Nbits = int(参数["PqM"]), int(参数["Nbits"])
MRefine, NbitsRefine = int(参数["MRefine"]), int(参数["NbitsRefine"])


@Mods().注册索引类型("IVFPQR", 数据源="Vector", 支持增量=True, 包装型=True)
def 构建(Self, 向量, 索引=None):
    维度, 数量 = 向量.shape[1], 向量.shape[0]
    量化器, _ = Self.构建索引节点(Self.当前子规格 or "IP", 维度, 数量, Self.构建主, Self.当前深度 + 1)
    nlist = max(1, int(np.sqrt(数量)))
    nprobe = min(nlist, max(1, int(nlist / Nlist)))
    结果 = faiss.IndexIVFPQR(量化器, 维度, nlist, PqM, Nbits, MRefine, NbitsRefine)
    结果.nprobe = nprobe
    return 结果, True
