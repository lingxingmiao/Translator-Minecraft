from TranslatorLib import Mods, np, CleanVRAM, faiss, GPU_ACC

参数 = Mods().模组配置("quant-opq", {"PqM": 128, "PqNbits": 8, "OpqItrs": 25})
PqM, PqNbits = int(参数["PqM"]), int(参数["PqNbits"])
OpqItrs = int(参数["OpqItrs"])


def OPQ训练(Self, 数组):
    M, NBITS = PqM, PqNbits
    迭代次数 = OpqItrs
    总行数, 维度 = 数组.shape
    子维度 = 维度 // M
    均值向量 = np.mean(数组, axis=0, dtype=np.float32)
    中心化数据 = np.ascontiguousarray(数组 - 均值向量, dtype=np.float32)
    旋转矩阵 = np.eye(维度, dtype=np.float32)
    码本 = None
    编码 = None
    for 迭代 in range(迭代次数):
        旋转数据 = np.ascontiguousarray(中心化数据 @ 旋转矩阵, dtype=np.float32)
        码本列表 = []
        编码列表 = []
        for m in range(M):
            子向量 = np.ascontiguousarray(旋转数据[:, m * 子维度:(m + 1) * 子维度])
            聚类 = faiss.Kmeans(d=子维度, k=1 << NBITS, niter=20, gpu=GPU_ACC)
            聚类.train(子向量)
            码本列表.append(聚类.centroids.astype(np.float32))
            _, 分配 = 聚类.index.search(子向量, 1)
            编码列表.append(分配.ravel().astype(np.uint8))
        码本 = np.stack(码本列表, axis=0)
        编码 = np.stack(编码列表, axis=1)
        if 迭代 >= 迭代次数 - 1:
            break
        重建 = np.empty((总行数, 维度), dtype=np.float32)
        for m in range(M):
            重建[:, m * 子维度:(m + 1) * 子维度] = 码本[m, 编码[:, m].astype(np.int32)]
        U, _, Vt = np.linalg.svd(中心化数据.T @ 重建, full_matrices=False)
        旋转矩阵 = np.ascontiguousarray(U @ Vt, dtype=np.float32)
        del 旋转数据, 重建
        CleanVRAM()
    return 编码, 码本, 旋转矩阵, 均值向量


def OPQ解码(Self, 编码, 码本, 旋转矩阵, 均值向量, 形状):
    M = PqM
    总行数, 维度 = 形状
    子维度 = 维度 // M
    重建 = np.empty((总行数, 维度), dtype=np.float32)
    for m in range(M):
        重建[:, m * 子维度:(m + 1) * 子维度] = 码本[m, 编码[:, m].astype(np.int32)]
    return np.ascontiguousarray(重建 @ 旋转矩阵.T + 均值向量, dtype=np.float32)


解码字典 = ["Vector", "Codebook", "RotMatrix", "Mean", "Shape"]


@Mods().注册量化类型("OPQ", 8)
def 编码(Self, 数组):
    数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32)
    总行数, 维度 = 数组.shape
    编码值, 码本, 旋转矩阵, 均值向量 = OPQ训练(Self, 数组)
    CleanVRAM()
    return {"Vector": np.ascontiguousarray(编码值), "Codebook": 码本,
            "RotMatrix": 旋转矩阵, "Mean": 均值向量, "Shape": (总行数, 维度)}


@Mods().注册反量化类型("OPQ", 8, 解码字典)
def 解码(Self, 编码, 码本, 旋转矩阵, 均值向量, 形状):
    return OPQ解码(Self, 编码, 码本, 旋转矩阵, 均值向量, 形状)
