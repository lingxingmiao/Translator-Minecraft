from TranslatorLib import Mods, np, CleanVRAM, faiss, GPU_ACC

参数 = Mods().模组配置("quant-pq", {"PqM": 128, "PqNbits": 8})
PqM, PqNbits = int(参数["PqM"]), int(参数["PqNbits"])


def PQ编码(Self, 数组):
    M, NBITS = PqM, PqNbits
    数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32)
    总行数, 维度 = 数组.shape
    子维度 = 维度 // M
    码本列表 = []
    编码列表 = []
    for m in range(M):
        子向量 = np.ascontiguousarray(数组[:, m * 子维度:(m + 1) * 子维度])
        聚类 = faiss.Kmeans(d=子维度, k=1 << NBITS, niter=20, gpu=GPU_ACC)
        聚类.train(子向量)
        码本列表.append(聚类.centroids.astype(np.float32))
        _, 分配 = 聚类.index.search(子向量, 1)
        编码列表.append(分配.ravel().astype(np.uint8))
    码本 = np.stack(码本列表, axis=0)
    编码 = np.stack(编码列表, axis=1)
    CleanVRAM()
    return {"Vector": np.ascontiguousarray(编码), "Codebook": 码本, "Shape": (总行数, 维度)}


def PQ解码(Self, 编码, 码本, 形状):
    M = PqM
    总行数, 维度 = 形状
    子维度 = 维度 // M
    输出 = np.empty((总行数, 维度), dtype=np.float32)
    for m in range(M):
        输出[:, m * 子维度:(m + 1) * 子维度] = 码本[m, 编码[:, m].astype(np.int32)]
    return 输出


解码字典 = ["Vector", "Codebook", "Shape"]


@Mods().注册量化类型("PQ", 8)
def 编码(Self, 数组):
    return PQ编码(Self, 数组)


@Mods().注册反量化类型("PQ", 8, 解码字典)
def 解码(Self, 编码, 码本, 形状):
    return PQ解码(Self, 编码, 码本, 形状)
