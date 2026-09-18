from TranslatorLib import Mods, np, math, CleanVRAM, TranslatorQuantization

参数 = Mods().模组配置("quant-q_k_m", {"BlockSize": 128})
BlockSize = int(参数["BlockSize"])

def Qx_K编码(Self, 数组, 最大量级):
    块大小 = BlockSize
    数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32)
    行数, 维度 = 数组.shape
    填充维 = (-维度) % 块大小
    有填充 = 填充维 > 0
    if 有填充:
        缓冲 = np.full((行数, 维度 + 填充维), np.nan, dtype=np.float32)
        缓冲[:, :维度] = 数组
    else:
        缓冲 = 数组.copy()
    del 数组
    组数 = (维度 + 填充维) // 块大小
    分组 = 缓冲.reshape(行数, 组数, 块大小)
    if 有填充:
        最小值 = np.nanmin(分组, axis=2).astype(np.float32)
        最大值 = np.nanmax(分组, axis=2).astype(np.float32)
    else:
        最小值 = np.min(分组, axis=2).astype(np.float32)
        最大值 = np.max(分组, axis=2).astype(np.float32)
    缩放值 = (最大值 - 最小值).astype(np.float32)
    del 最大值
    np.maximum(缩放值, 1e-8, out=缩放值)
    if 有填充:
        np.nan_to_num(分组, nan=0.0, copy=False)
    分组 -= 最小值[:, :, None]
    分组 /= 缩放值[:, :, None]
    np.clip(分组, 0, 1, out=分组)
    分组 *= float(最大量级)
    np.round(分组, out=分组)
    最大最小 = max(float(np.nanmax(np.abs(最小值))), 1e-8)
    最大缩放 = max(float(np.nanmax(np.abs(缩放值))), 1e-8)
    量化值 = 分组.astype(np.uint8).reshape(行数, 组数 * 块大小).ravel()
    if 有填充:
        np.nan_to_num(最小值, nan=0.0, copy=False)
        np.nan_to_num(缩放值, nan=0.0, copy=False)
    del 缓冲, 分组
    CleanVRAM()
    return (量化值, Self._编码缩放(最小值 / 最大最小), 最大最小,
            Self._编码缩放(缩放值 / 最大缩放), 最大缩放, 行数, 维度)


def Qx_K解码(Self, 索引, 最小编码, 最大最小, 缩放编码, 最大缩放, 形状, 最大量级):
    块大小 = BlockSize
    行数, 维度 = 形状
    组数 = (维度 + 块大小 - 1) // 块大小
    最小值 = (Self._解码缩放(最小编码, 行数 * 组数) * 最大最小).reshape(行数, 组数).astype(np.float32)
    缩放值 = (Self._解码缩放(缩放编码, 行数 * 组数) * 最大缩放).reshape(行数, 组数).astype(np.float32)
    np.nan_to_num(最小值, nan=0.0, copy=False)
    np.nan_to_num(缩放值, nan=0.0, copy=False)
    np.maximum(缩放值, 1e-8, out=缩放值)
    索引 = np.asarray(索引).ravel()
    需要 = 行数 * 组数 * 块大小
    if 索引.size < 需要:
        缓冲 = np.zeros(需要, dtype=索引.dtype)
        缓冲[:索引.size] = 索引
        索引 = 缓冲
    索引_3d = 索引[:需要].reshape(行数, 组数, 块大小)
    重建 = (索引_3d.astype(np.float32) / float(最大量级)) * 缩放值[:, :, None] + 最小值[:, :, None]
    return 重建.reshape(行数, 组数 * 块大小)[:, :维度]


def Qx_K_M编码(Self, 数组, 最大量级):
    均值向量 = np.mean(数组, axis=0, dtype=np.float32)
    残差 = np.ascontiguousarray(数组 - 均值向量, dtype=np.float32)
    del 数组
    量化值, Min, MaxMin, Scale, MaxScale, 行数, 维度 = Qx_K编码(Self, 残差, 最大量级)
    del 残差
    CleanVRAM()
    最大均值 = max(float(np.max(np.abs(均值向量))), 1e-8)
    return (量化值, Min, MaxMin, Scale, MaxScale,
            Self._编码缩放(均值向量 / 最大均值), 最大均值, 行数, 维度)


def Qx_K_M解码(Self, 索引, 最小编码, 最大最小, 缩放编码, 最大缩放,
                均值编码, 最大均值, 形状, 最大量级):
    行数, 维度 = 形状
    均值 = (Self._解码缩放(均值编码, 维度) * 最大均值).astype(np.float32)
    残差 = Qx_K解码(Self, 索引, 最小编码, 最大最小, 缩放编码, 最大缩放, 形状, 最大量级)
    return 残差 + 均值


def 打包(Self, 索引, 位宽):
    平 = np.ascontiguousarray(np.asarray(索引, dtype=np.uint8).ravel())
    if 位宽 == 0:
        return 平
    if 位宽 == 1:
        return np.packbits(平)
    if 位宽 == 2:
        return Self._打包2(平)
    if 位宽 == 3:
        return Self._打包3(平)
    if 位宽 == 4:
        return Self._打包4(平)
    if 位宽 == 5:
        return Self._打包5(平)
    if 位宽 == 6:
        return Self._打包6(平)
    return TranslatorQuantization.加速三值打包(平)


def 解包(Self, 压缩, 数量, 位宽):
    if 位宽 == 0:
        return np.asarray(压缩, dtype=np.uint8).ravel()[:数量]
    if 位宽 == 1:
        return np.unpackbits(np.asarray(压缩, dtype=np.uint8).ravel())[:数量]
    if 位宽 == 2:
        return Self._解包2(压缩, 数量)
    if 位宽 == 3:
        return Self._解包3(压缩, 数量)
    if 位宽 == 4:
        return Self._解包4(压缩, 数量)
    if 位宽 == 5:
        return Self._解包5(压缩, 数量)
    if 位宽 == 6:
        return Self._解包6(压缩, 数量)
    return TranslatorQuantization.加速三值解包(np.asarray(压缩).astype(np.uint8), 数量)


解码字典 = ["Vector", "Min", "MaxMin", "Scale", "MaxScale", "Mean", "MaxMean", "Shape"]

档位表 = [
    ("Q8_K_M", 255, 0, 8),
    ("Q6_K_M", 63, 6, 6),
    ("Q5_K_M", 31, 5, 5),
    ("Q4_K_M", 15, 4, 4),
    ("Q3_K_M", 7, 3, 3),
    ("Q2_K_M", 3, 2, 2),
    ("TQ1_K_M", 2, 9, 1.585),
    ("Q1_K_M", 1, 1, 1),
]


def 注册档位(名称, 最大量级, 打包位宽, 声明位数):
    @Mods().注册量化类型(名称, 声明位数, 裁切=True)
    def 编码(Self, 数组):
        量化值, Min, MaxMin, Scale, MaxScale, Mean, MaxMean, 行数, 维度 = Qx_K_M编码(Self, 数组, 最大量级)
        结果 = {"Vector": 打包(Self, 量化值, 打包位宽), "Min": Min, "MaxMin": MaxMin,
                "Scale": Scale, "MaxScale": MaxScale, "Mean": Mean, "MaxMean": MaxMean,
                "Shape": (行数, 维度)}
        CleanVRAM()
        return 结果

    @Mods().注册反量化类型(名称, 声明位数, 解码字典)
    def 解码(Self, 压缩, 最小编码, 最大最小, 缩放编码, 最大缩放, 均值编码, 最大均值, 形状):
        块大小 = BlockSize
        行数, 维度 = 形状
        组数 = (维度 + 块大小 - 1) // 块大小
        索引 = 解包(Self, 压缩, 行数 * 组数 * 块大小, 打包位宽)
        return Qx_K_M解码(Self, 索引, 最小编码, 最大最小, 缩放编码, 最大缩放,
                          均值编码, 最大均值, 形状, 最大量级)

    return 编码, 解码


for 名称, 最大量级, 打包位宽, 声明位数 in 档位表:
    注册档位(名称, 最大量级, 打包位宽, 声明位数)
