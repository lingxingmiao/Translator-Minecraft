from TranslatorLib import Mods, np, math, CleanVRAM, TranslatorQuantization

参数 = Mods().模组配置("quant-polar", {"BlockSize": 128, "SplLM": 0.05})
BlockSize, SplLM = int(参数["BlockSize"]), float(参数["SplLM"])

# PolarQuant: 块L2归一化 + Walsh-Hadamard旋转 + 在真实样本上拟合 Lloyd-Max 码本
# 参考: PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression

def 变换窗口(块大小):
    长度 = 1
    while 长度 < int(块大小): 长度 <<= 1
    return 长度

def PolarQuant编码(Self, 数组, 聚类数, 采样参数):
    块大小 = 变换窗口(BlockSize)
    数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32)
    形状 = 数组.shape; 行数, 维度 = 形状
    总数 = 行数 * 维度
    填充 = (-总数) % 块大小
    if 填充: 缓冲 = np.zeros(总数 + 填充, dtype=np.float32); 缓冲[:总数] = 数组.ravel()
    else:    缓冲 = 数组.ravel().copy()
    del 数组
    分块 = 缓冲.reshape(-1, 块大小)
    范数 = np.linalg.norm(分块, axis=1).astype(np.float32)
    np.maximum(范数, 1e-8, out=范数)
    分块 /= 范数[:, None]
    Self.Hadamard原地(分块)
    质心 = Self.LloydMax(Self.采样(分块.ravel(), 采样参数), 聚类数)
    边界 = ((质心[:-1] + 质心[1:]) / 2.0).astype(np.float32)
    量化索引 = Self.查表量化(分块, 边界)
    del 分块, 缓冲; CleanVRAM()
    最大范数 = max(float(np.max(范数)), 1e-8)
    return (量化索引.ravel(), Self._编码缩放(范数 / 最大范数), 最大范数,
            质心.astype(np.float32), 形状, 块大小)

def PolarQuant解码(Self, 量化索引_2d, 范数编码, 最大范数, 质心, 形状, 块大小):
    总数 = math.prod(形状)
    块数 = (总数 + 块大小 - 1) // 块大小
    范数 = (Self._解码缩放(范数编码, 块数) * 最大范数)[:块数]
    np.maximum(范数, 1e-8, out=范数)
    重建 = 质心[量化索引_2d]
    Self.Hadamard原地(重建)
    重建 /= float(块大小)
    重建 *= 范数[:, None]
    return 重建.ravel()[:总数].reshape(形状)

def 打包(Self, 索引, 位宽):
    if 位宽 == 1:
        return np.packbits(np.asarray(索引, dtype=np.uint8).ravel())
    if 位宽 == 3:
        return TranslatorQuantization.加速三值打包(np.asarray(索引, dtype=np.uint8).ravel())
    if 位宽 == 2:
        return Self._打包2(索引.ravel())
    if 位宽 == 4:
        return Self._打包3(索引.ravel())
    if 位宽 == 5:
        return Self._打包4(索引.ravel())
    return Self._打包5(索引.ravel())

def 解包(Self, 压缩, 数量, 位宽):
    if 位宽 == 1:
        return np.unpackbits(np.asarray(压缩).astype(np.uint8))[:数量]
    if 位宽 == 3:
        return TranslatorQuantization.加速三值解包(np.asarray(压缩).astype(np.uint8), 数量)
    if 位宽 == 2:
        return Self._解包2(压缩, 数量)
    if 位宽 == 4:
        return Self._解包3(压缩, 数量)
    if 位宽 == 5:
        return Self._解包4(压缩, 数量)
    return Self._解包5(压缩, 数量)

解码字典 = ["PackedVector", "Norms", "MaxNorm", "Centroids", "Shape", "BlockSize"]

档位表 = [
    ("PolarQ1",   2, 1, 1),
    ("PolarTQ1",  3, 3, 1.585),
    ("PolarQ2",   4, 2, 2),
    ("PolarQ3",   8, 4, 3),
    ("PolarQ4",  16, 5, 4),
    ("PolarQ5",  32, 6, 5),
]


def 注册档位(名称, 聚类数, 打包位宽, 声明位数):
    @Mods().注册量化类型(名称, 声明位数)
    def 编码(Self, 数组):
        量化索引, 范数编码, 最大范数, 质心, 形状, 块大小 = PolarQuant编码(Self, 数组, 聚类数, SplLM)
        结果 = {"PackedVector": 打包(Self, 量化索引, 打包位宽), "Norms": 范数编码,
                "MaxNorm": 最大范数, "Centroids": np.asarray(质心, dtype=np.float32),
                "Shape": 形状, "BlockSize": 块大小}
        CleanVRAM()
        return 结果

    @Mods().注册反量化类型(名称, 声明位数, 解码字典)
    def 解码(Self, 压缩, 范数编码, 最大范数, 质心, 形状, 块大小):
        总数 = math.prod(形状); 块数 = (总数 + 块大小 - 1) // 块大小
        量化索引_2d = 解包(Self, 压缩, 块数 * 块大小, 打包位宽).reshape(块数, 块大小)
        return PolarQuant解码(Self, 量化索引_2d, 范数编码, 最大范数, 质心, 形状, 块大小)

    return 编码, 解码


for 名称, 聚类数, 打包位宽, 声明位数 in 档位表:
    注册档位(名称, 聚类数, 打包位宽, 声明位数)
