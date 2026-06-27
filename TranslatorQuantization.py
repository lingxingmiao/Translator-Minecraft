from TranslatorLib import (np, Path, math, numpy, njit, CPU_ACC, GPU_ACC, CleanVRAM, faiss,
                           RuntimeConfig, Locale, Module)
@njit(cache=True)
def 加速打包2(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 3) >> 2
    输出 = numpy.zeros(输出长度, dtype=numpy.uint8)
    for 索 in range(数量):
        输出[索 >> 2] |= numpy.uint8(量化值[索] << ((索 & 3) << 1))
    return 输出
@njit(cache=True)
def 加速解包2(压缩, 数量):
    输出 = numpy.empty(数量, dtype=numpy.uint8)
    for 索 in range(数量):
        字节索 = 索 >> 2
        if 字节索 < len(压缩):
            输出[索] = numpy.uint8((压缩[字节索] >> ((索 & 3) << 1)) & 3)
        else:
            输出[索] = numpy.uint8(0)
    return 输出
@njit(cache=True)
def 加速打包4(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 1) >> 1
    输出 = numpy.zeros(输出长度, dtype=numpy.uint8)
    for 索 in range(数量):
        if 索 & 1 == 0:
            输出[索 >> 1] = numpy.uint8(量化值[索] << 4)
        else:
            输出[索 >> 1] |= numpy.uint8(量化值[索] & 0xF)
    return 输出
@njit(cache=True)
def 加速解包4(压缩, 数量):
    输出 = numpy.empty(数量, dtype=numpy.uint8)
    for 索 in range(数量):
        字节索 = 索 >> 1
        if 字节索 < len(压缩):
            if 索 & 1 == 0:
                输出[索] = numpy.uint8((压缩[字节索] >> 4) & 0xF)
            else:
                输出[索] = numpy.uint8(压缩[字节索] & 0xF)
        else:
            输出[索] = numpy.uint8(0)
    return 输出
@njit(cache=True)
def 加速线性量化(数据块, 缩放, 零点, 最大值):
    块数 = 数据块.shape[0]; 块大小 = 数据块.shape[1]
    输出 = numpy.empty((块数, 块大小), dtype=numpy.uint8)
    for 行 in range(块数):
        当前缩放 = 缩放[行]; 当前零点 = 零点[行]
        for 列 in range(块大小):
            值 = 数据块[行, 列] / 当前缩放 + 当前零点
            值 = numpy.float32(round(值))
            if 值 < 0.0: 值 = 0.0
            elif 值 > 最大值: 值 = 最大值
            输出[行, 列] = numpy.uint8(值)
    return 输出
@njit(cache=True)
def 加速线性反量化(量化数组, 缩放, 零点):
    块数 = 量化数组.shape[0]; 块大小 = 量化数组.shape[1]
    输出 = numpy.empty((块数, 块大小), dtype=numpy.float32)
    for 行 in range(块数):
        当前缩放 = 缩放[行]; 当前零点 = 零点[行]
        for 列 in range(块大小):
            输出[行, 列] = (numpy.float32(量化数组[行, 列]) - 当前零点) * 当前缩放
    return 输出
@njit(cache=True)
def 加速NF量化(数据块, 边界, 缩放):
    块数 = 数据块.shape[0]; 块大小 = 数据块.shape[1]; 边界数 = len(边界)
    索引 = numpy.empty((块数, 块大小), dtype=numpy.uint8)
    for 行 in range(块数):
        当前缩放 = 缩放[行]
        for 列 in range(块大小):
            值 = 数据块[行, 列] / 当前缩放
            等级 = numpy.uint8(0)
            for 步 in range(边界数):
                if 值 >= 边界[步]: 等级 += numpy.uint8(1)
            索引[行, 列] = 等级
    return 索引
@njit(cache=True)
def 加速FWHT原地(数组):
    块数 = 数组.shape[0]; 长度 = 数组.shape[1]; 步长 = numpy.int64(1)
    while 步长 < 长度:
        跨度 = numpy.int64(2) * 步长
        for 行 in range(块数):
            for 起始 in range(0, 长度, 跨度):
                for 列 in range(起始, 起始 + 步长):
                    左值 = 数组[行, 列]; 右值 = 数组[行, 列 + 步长]
                    数组[行, 列] = 左值 + 右值; 数组[行, 列 + 步长] = 左值 - 右值
        步长 = 跨度
@njit(cache=True)
def 加速查找alpha(样本, alpha列表):
    最佳 = alpha列表[0]; 最佳误差 = numpy.float32(1e30)
    数量 = len(样本); 列表长度 = len(alpha列表)
    for 索 in range(列表长度):
        当前alpha = alpha列表[索]; 误差 = numpy.float32(0.0)
        for 样索 in range(数量):
            值 = 样本[样索]
            if 值 > 当前alpha: 差值 = 值 - numpy.float32(1.0)
            elif 值 < -当前alpha: 差值 = 值 + numpy.float32(1.0)
            else: 差值 = 值
            误差 += 差值 * 差值
        误差 /= numpy.float32(数量)
        if 误差 < 最佳误差: 最佳误差 = 误差; 最佳 = 当前alpha
    return 最佳
@njit(cache=True)
def 加速SVD处理(展平, 边界, 中心点):
    块数 = 展平.shape[0]; 块大小 = 展平.shape[1]; 边界数 = len(边界)
    重建 = numpy.empty((块数, 块大小), dtype=numpy.float32)
    for 行 in range(块数):
        最大值 = numpy.float32(0.0); 最大索 = 0
        for 列 in range(块大小):
            绝对值 = 展平[行, 列]
            if 绝对值 < 0: 绝对值 = -绝对值
            if 绝对值 > 最大值: 最大值 = 绝对值; 最大索 = 列
        异常值 = 展平[行, 最大索]
        求和 = numpy.float32(0.0); 平方和 = numpy.float32(0.0)
        for 列 in range(块大小):
            值 = 展平[行, 列]
            if 列 == 最大索: 值 = numpy.float32(0.0)
            求和 += 值; 平方和 += 值 * 值
        均值 = 求和 / numpy.float32(块大小)
        方差 = 平方和 / numpy.float32(块大小) - 均值 * 均值
        if 方差 < 0.0: 方差 = 0.0
        标准差 = numpy.float32(numpy.sqrt(numpy.float64(方差)))
        if 标准差 < 1e-8: 标准差 = numpy.float32(1e-8)
        for 列 in range(块大小):
            归一值 = 展平[行, 列] / 标准差
            if 列 == 最大索: 归一值 = numpy.float32(0.0)
            等级 = 0
            for 步 in range(边界数):
                if 归一值 >= 边界[步]: 等级 += 1
            重建[行, 列] = 中心点[等级] * 标准差
        重建[行, 最大索] = 异常值
    return 重建
@njit(cache=True)
def 加速三值打包(映射值):
    数量 = len(映射值); 输出长度 = (数量 + 4) // 5
    输出 = numpy.empty(输出长度, dtype=numpy.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 5; 值 = numpy.uint8(0); 倍数 = numpy.uint8(1)
        for 步 in range(5):
            if 基索 + 步 < 数量: 值 += numpy.uint8(映射值[基索 + 步] * 倍数)
            倍数 = numpy.uint8(倍数 * 3)
        输出[块索] = 值
    return 输出
@njit(cache=True)
def 加速三值解包(压缩, 数量):
    输出 = numpy.empty(数量, dtype=numpy.int8)
    for 块索 in range(len(压缩)):
        值 = int(压缩[块索]); 基索 = 块索 * 5
        for 步 in range(5):
            if 基索 + 步 < 数量: 输出[基索 + 步] = numpy.int8(值 % 3)
            值 //= 3
    return 输出
@njit(cache=True)
def 加速打包3(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 7) // 8
    输出 = numpy.zeros(输出长度 * 3, dtype=numpy.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 8; 偏移 = 块索 * 3
        v0 = numpy.uint8(量化值[基索] if 基索 < 数量 else 0)
        v1 = numpy.uint8(量化值[基索+1] if 基索+1 < 数量 else 0)
        v2 = numpy.uint8(量化值[基索+2] if 基索+2 < 数量 else 0)
        v3 = numpy.uint8(量化值[基索+3] if 基索+3 < 数量 else 0)
        v4 = numpy.uint8(量化值[基索+4] if 基索+4 < 数量 else 0)
        v5 = numpy.uint8(量化值[基索+5] if 基索+5 < 数量 else 0)
        v6 = numpy.uint8(量化值[基索+6] if 基索+6 < 数量 else 0)
        v7 = numpy.uint8(量化值[基索+7] if 基索+7 < 数量 else 0)
        输出[偏移]   = numpy.uint8((v0<<5)|(v1<<2)|(v2>>1))
        输出[偏移+1] = numpy.uint8(((v2&1)<<7)|(v3<<4)|(v4<<1)|(v5>>2))
        输出[偏移+2] = numpy.uint8(((v5&3)<<6)|(v6<<3)|v7)
    return 输出

@njit(cache=True)
def 加速解包3(压缩, 数量):
    块数 = len(压缩) // 3; 输出 = numpy.empty(块数 * 8, dtype=numpy.uint8)
    for 块索 in range(块数):
        偏移 = 块索 * 3; 基索 = 块索 * 8
        b0 = 压缩[偏移]; b1 = 压缩[偏移+1]; b2 = 压缩[偏移+2]
        输出[基索]   = numpy.uint8((b0>>5)&7)
        输出[基索+1] = numpy.uint8((b0>>2)&7)
        输出[基索+2] = numpy.uint8(((b0&3)<<1)|(b1>>7))
        输出[基索+3] = numpy.uint8((b1>>4)&7)
        输出[基索+4] = numpy.uint8((b1>>1)&7)
        输出[基索+5] = numpy.uint8(((b1&1)<<2)|(b2>>6))
        输出[基索+6] = numpy.uint8((b2>>3)&7)
        输出[基索+7] = numpy.uint8(b2&7)
    return 输出[:数量]

@njit(cache=True)
def 加速打包6(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 3) // 4
    输出 = numpy.zeros(输出长度 * 3, dtype=numpy.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 4; 偏移 = 块索 * 3
        v0 = numpy.uint8(量化值[基索] if 基索 < 数量 else 0)
        v1 = numpy.uint8(量化值[基索+1] if 基索+1 < 数量 else 0)
        v2 = numpy.uint8(量化值[基索+2] if 基索+2 < 数量 else 0)
        v3 = numpy.uint8(量化值[基索+3] if 基索+3 < 数量 else 0)
        输出[偏移]   = numpy.uint8((v0<<2)|(v1>>4))
        输出[偏移+1] = numpy.uint8(((v1&0xF)<<4)|(v2>>2))
        输出[偏移+2] = numpy.uint8(((v2&3)<<6)|v3)
    return 输出

@njit(cache=True)
def 加速解包6(压缩, 数量):
    块数 = len(压缩) // 3; 输出 = numpy.empty(块数 * 4, dtype=numpy.uint8)
    for 块索 in range(块数):
        偏移 = 块索 * 3; 基索 = 块索 * 4
        b0 = 压缩[偏移]; b1 = 压缩[偏移+1]; b2 = 压缩[偏移+2]
        输出[基索]   = numpy.uint8(b0>>2)
        输出[基索+1] = numpy.uint8(((b0&3)<<4)|(b1>>4))
        输出[基索+2] = numpy.uint8(((b1&0xF)<<2)|(b2>>6))
        输出[基索+3] = numpy.uint8(b2&0x3F)
    return 输出[:数量]
@njit(cache=True)
def 加速打包12(量化值):
    数量 = len(量化值); 输出长度 = ((数量 + 1) >> 1) * 3
    输出 = numpy.zeros(输出长度, dtype=numpy.uint8)
    for 块索 in range(数量 >> 1):
        偏移 = 块索 * 3
        v0 = numpy.uint16(量化值[块索 << 1])
        v1 = numpy.uint16(量化值[(块索 << 1) + 1])
        输出[偏移]   = numpy.uint8(v0 >> 4)
        输出[偏移+1] = numpy.uint8(((v0 & 15) << 4) | (v1 >> 8))
        输出[偏移+2] = numpy.uint8(v1 & 255)
    if 数量 & 1:
        偏移 = (数量 >> 1) * 3
        v0 = numpy.uint16(量化值[数量 - 1])
        输出[偏移]   = numpy.uint8(v0 >> 4)
        输出[偏移+1] = numpy.uint8((v0 & 15) << 4)
    return 输出

@njit(cache=True)
def 加速解包12(压缩, 数量):
    块数 = (数量 + 1) >> 1
    输出 = numpy.empty(块数 * 2, dtype=numpy.uint16)
    for 块索 in range(块数):
        偏移 = 块索 * 3
        b0 = numpy.uint16(压缩[偏移])
        b1 = numpy.uint16(压缩[偏移 + 1])
        b2 = numpy.uint16(压缩[偏移 + 2])
        输出[块索 << 1]         = (b0 << 4) | (b1 >> 4)
        输出[(块索 << 1) + 1]   = ((b1 & 15) << 8) | b2
    return 输出[:数量]
class Quantization:
    def __init__(Self, 配置: dict = None):
        global tqdm
        配置 = 配置 or {}
        Self.Config = RuntimeConfig(**配置)
        Path(Self.Config.LOGS_FILE_PATH).mkdir(parents=True, exist_ok=True)
        Self.Locale = Locale(Config=配置)
        Self.Module = Module()
        tqdm = Self.Locale.Tqdm
        Self.拼接键 = ["Vector"]
        Self.Numba加速 = (not GPU_ACC) and CPU_ACC
        Self._NF边界 = numpy.array([-0.9816, 0.0, 0.9816], dtype=numpy.float32)
        Self._NF中心 = numpy.array([-1.5121, -0.4528, 0.4528, 1.5121], dtype=numpy.float32)
        Self._NF3边界 = numpy.array([
            -1.1503, -0.6745, -0.3186, 0.0, 0.3186, 0.6745, 1.1503
        ], dtype=numpy.float32)
        Self._NF3中心 = numpy.array([
            -1.6471, -0.8946, -0.4916, -0.1582, 0.1582, 0.4916, 0.8946, 1.6471
        ], dtype=numpy.float32)
        if Self.Numba加速:
            try:
                _预热数据 = numpy.zeros((2, 8), dtype=numpy.float32)
                _预热边界 = numpy.array([-1.0, 0.0, 1.0], dtype=numpy.float32)
                _预热中心 = numpy.array([-1.5, -0.5, 0.5, 1.5], dtype=numpy.float32)
                _预热alpha = numpy.linspace(0.1, 1.0, 3, dtype=numpy.float32)
                加速打包2(numpy.zeros(8, dtype=numpy.uint8))
                加速解包2(numpy.zeros(2, dtype=numpy.uint8), 8)
                加速打包4(numpy.zeros(8, dtype=numpy.uint8))
                加速解包4(numpy.zeros(4, dtype=numpy.uint8), 8)
                加速线性量化(_预热数据, numpy.ones(2, dtype=numpy.float32),
                      numpy.zeros(2, dtype=numpy.float32), 15.0)
                加速线性反量化(numpy.zeros((2, 8), dtype=numpy.uint8),
                        numpy.ones(2, numpy.float32), numpy.zeros(2, numpy.float32))
                加速NF量化(_预热数据, _预热边界, numpy.ones(2, dtype=numpy.float32))
                加速FWHT原地(_预热数据.copy())
                加速查找alpha(numpy.zeros(10, dtype=numpy.float32), _预热alpha)
                加速SVD处理(_预热数据.copy(), _预热边界, _预热中心)
                加速三值打包(numpy.zeros(5, dtype=numpy.uint8))
                加速三值解包(numpy.zeros(1, dtype=numpy.uint8), 5)
                加速打包3(numpy.zeros(8, dtype=numpy.uint8))
                加速解包3(numpy.zeros(3, dtype=numpy.uint8), 8)
                加速打包6(numpy.zeros(4, dtype=numpy.uint8))
                加速解包6(numpy.zeros(3, dtype=numpy.uint8), 4)
                加速打包12(numpy.zeros(4, dtype=numpy.uint16))
                加速解包12(numpy.zeros(6, dtype=numpy.uint8), 4)
            except Exception:
                Self.Numba加速 = False
        Self.编码映射 = {
            "Q8_K": lambda Self, 数组: Self.F32编码Q8_K(数组),
            "Q6_K": lambda Self, 数组: Self.F32编码Q6_K(数组),
            "Q4_K": lambda Self, 数组: Self.F32编码Q4_K(数组),
            "Q4_K_H": lambda Self, 数组: Self.F32编码Q4_K_H(数组),
            "Q4_SVD_LM": lambda Self, 数组: Self.F32编码Q4_SVD_LM(数组),
            "GSQ4_0": lambda Self, 数组: Self.F32编码GSQ4_0(数组),
            "Q3_K": lambda Self, 数组: Self.F32编码Q3_K(数组),
            "Q3_K_H": lambda Self, 数组: Self.F32编码Q3_K_H(数组),
            "Q3_SVD_LM": lambda Self, 数组: Self.F32编码Q3_SVD_LM(数组),
            "Q2_K": lambda Self, 数组: Self.F32编码Q2_K(数组),
            "Q2_NF": lambda Self, 数组: Self.F32编码Q2_NF(数组),
            "Q2_E_NF": lambda Self, 数组: Self.F32编码Q2_E_NF(数组),
            "Q2_SVD_LM": lambda Self, 数组: Self.F32编码Q2_SVD_LM(数组),
            "Q2_E_SVD_LM": lambda Self, 数组: Self.F32编码Q2_E_SVD_LM(数组),
            "TQ1_SVD": lambda Self, 数组: Self.F32编码TQ1_SVD(数组),
            "Q1_K_M": lambda Self, 数组: Self.F32编码Q1_K_M(数组),
            "GSQ8_K": lambda Self, 数组: Self.F32编码GSQ8_K(数组),
            "GSQ6_K": lambda Self, 数组: Self.F32编码GSQ6_K(数组),
            "GSQ4_K": lambda Self, 数组: Self.F32编码GSQ4_K(数组),
            "GSQ3_K": lambda Self, 数组: Self.F32编码GSQ3_K(数组),
            "GSQ2_K": lambda Self, 数组: Self.F32编码GSQ2_K(数组),
            "Float32": lambda Self, 数组: {"Vector": np.asarray(数组, dtype=np.float32)},
            "BFloat16": lambda Self, 数组: Self.F32编码BF16(数组),
            "Float16": lambda Self, 数组: {"Vector": np.asarray(数组, dtype=np.float16)},
            "Float8_E4M3": lambda Self, 数组: Self.F32编码FP8_E4M3(数组),
            "Float8_E0M7": lambda Self, 数组: {"Vector": Self.F32编码F8_E0M7(数组)},
            "Float12_E0M11": lambda Self, 数组: Self.F32编码F12_E0M11(数组),
            "Float16_E0M15": lambda Self, 数组: {"Vector": Self.F32编码F16_E0M15(数组)},
        }
        Self.解码映射 = {
            "Q8_K": lambda Self, 数据: Self.Q8_K解码F32(数据["Vector"], 数据["Scale"], 数据["ZeroPoint"], 数据["Shape"]),
            "Q6_K": lambda Self, 数据: Self.Q6_K解码F32(数据["Vector"], 数据["Scale"], 数据["ZeroPoint"], 数据["Shape"]),
            "Q4_K": lambda Self, 数据: Self.Q4_K解码F32(数据["Vector"], 数据["Scale"], 数据["ZeroPoint"], 数据["Shape"]),
            "Q4_K_H": lambda Self, 数据: Self.Q4_K_H解码F32(数据["Vector"], 数据["Min"], 数据["MaxMin"], 数据["Scale"], 数据["MaxScale"], 数据["Shape"]),
            "Q4_SVD_LM": lambda Self, 数据: Self.Q4_SVD_LM解码F32(数据["PackedVector"], 数据["Scale"], 数据["MaxScale"], 数据["RotMatrix"], 数据["Centroids"], 数据["Shape"]),
            "GSQ4_0": lambda Self, 数据: Self.GSQ4_0解码F32(数据["Vector"], 数据["Min"], 数据["Scale"], 数据["Shape"]),
            "Q3_K": lambda Self, 数据: Self.Q3_K解码F32(数据["Vector"], 数据["Scale"], 数据["ZeroPoint"], 数据["Shape"]),
            "Q3_K_H": lambda Self, 数据: Self.Q3_K_H解码F32(数据["Vector"], 数据["Min"], 数据["MaxMin"], 数据["Scale"], 数据["MaxScale"], 数据["Shape"]),
            "Q3_SVD_LM": lambda Self, 数据: Self.Q3_SVD_LM解码F32(数据["PackedVector"], 数据["Scale"], 数据["MaxScale"], 数据["RotMatrix"], 数据["Centroids"], 数据["Shape"]),
            "Q2_K": lambda Self, 数据: Self.Q2_K解码F32(数据["Vector"], 数据["Scale"], 数据["ZeroPoint"], 数据["Shape"]),
            "Q2_NF": lambda Self, 数据: Self.Q2_NF解码F32(数据["Vector"], 数据["Scale"], 数据["MaxScale"], 数据["Shape"]),
            "Q2_E_NF": lambda Self, 数据: Self.Q2_E_NF解码F32(数据["Vector"], 数据["Scale"], 数据["MaxScale"], 数据["Outliers"], 数据["MaxOutlier"], 数据["OutlierIdx"], 数据["Shape"]),
            "Q2_SVD_LM": lambda Self, 数据: Self.Q2_SVD_LM解码F32(数据["PackedVector"], 数据["Scale"], 数据["MaxScale"], 数据["RotMatrix"], 数据["Centroids"], 数据["Shape"]),
            "Q2_E_SVD_LM": lambda Self, 数据: Self.Q2_E_SVD_LM解码F32(数据["Vector"], 数据["Scale"], 数据["MaxScale"], 数据["Outliers"], 数据["MaxOutlier"], 数据["OutlierIdx"], 数据["RotMatrix"], 数据["Centroids"], 数据["Shape"]),
            "TQ1_SVD": lambda Self, 数据: Self.TQ1_SVD解码F32(数据["PackedTernary"], 数据["ScaleINT8"],  数据["MaxScale"], 数据["RotMatrix"], 数据["Shape"]),
            "Q1_K_M": lambda Self, 数据: Self.Q1_K_M解码F32(数据["PackedBinary"], 数据["Scale"], 数据["MaxScale"], 数据["Mean"], 数据["MaxMean"], 数据["Shape"]),
            "GSQ8_K": lambda Self, 数据: Self.GSQ8_K解码F32(数据["Vector"], 数据["Min"], 数据["MaxMin"], 数据["Scale"], 数据["MaxScale"], 数据["Shape"]),
            "GSQ6_K": lambda Self, 数据: Self.GSQ6_K解码F32(数据["Vector"], 数据["Min"], 数据["MaxMin"], 数据["Scale"], 数据["MaxScale"], 数据["Shape"]),
            "GSQ4_K": lambda Self, 数据: Self.GSQ4_K解码F32(数据["Vector"], 数据["Min"], 数据["MaxMin"], 数据["Scale"], 数据["MaxScale"], 数据["Shape"]),
            "GSQ3_K": lambda Self, 数据: Self.GSQ3_K解码F32(数据["Vector"], 数据["Min"], 数据["MaxMin"], 数据["Scale"], 数据["MaxScale"], 数据["Shape"]),
            "GSQ2_K": lambda Self, 数据: Self.GSQ2_K解码F32(数据["Vector"], 数据["Min"], 数据["MaxMin"], 数据["Scale"], 数据["MaxScale"], 数据["Shape"]),
            "Float32": lambda Self, 数据: 数据["Vector"],
            "BFloat16": lambda Self, 数据: Self.BF16解码F32(数据["Vector"]),
            "Float16": lambda Self, 数据: 数据["Vector"],
            "Float8_E4M3": lambda Self, 数据: Self.FP8_E4M3解码F32(数据["Vector"]),
            "Float8_E0M7": lambda Self, 数据: Self.F8_E0M7解码F32(数据["Vector"]),
            "Float12_E0M11": lambda Self, 数据: Self.F12_E0M11解码F32(数据["Vector"], 数据["Shape"]),
            "Float16_E0M15": lambda Self, 数据: Self.F16_E0M15解码F32(数据["Vector"]),
        }
    def _确保H16(Self):
        if not hasattr(Self, 'H16矩阵'):
            H2 = numpy.array([[1., 1.], [1., -1.]], dtype=numpy.float32)
            H4 = numpy.kron(H2, H2); H8 = numpy.kron(H4, H2); H16 = numpy.kron(H8, H2)
            Self.H16矩阵 = np.asarray(H16 / 4.0, dtype=numpy.float32)
    def _分块(Self, 数组, 块大小):
        数组 = np.ascontiguousarray(数组, dtype=np.float32)
        形状 = 数组.shape; 总数 = 数组.size; 填充 = (-总数) % 块大小
        if 填充:
            缓冲 = np.zeros(总数 + 填充, dtype=np.float32); 缓冲[:总数] = 数组.ravel()
        else:
            缓冲 = 数组.ravel().copy()
        del 数组
        return 缓冲.reshape(-1, 块大小), 形状, 总数
    def _百分位(Self, 数据, 百分比, 轴=1, 保持维度=False):
        结果 = np.percentile(数据, [百分比, 100.0 - 百分比], axis=轴, keepdims=保持维度)
        return 结果[0].astype(np.float32), 结果[1].astype(np.float32)
    def _GSQ_K编码(Self, 数组, 最大量级):
        向量块 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32)
        行数, 维度 = 数组.shape; 填充行 = (-行数) % 向量块
        if 填充行:
            缓冲 = np.zeros((行数 + 填充行, 维度), dtype=np.float32); 缓冲[:行数] = 数组
        else: 缓冲 = 数组.copy()
        del 数组; 组数 = (行数 + 填充行) // 向量块; 分组 = 缓冲.reshape(组数, 向量块, 维度)
        最小值 = np.min(分组, axis=1, keepdims=True).astype(np.float32)
        缩放值 = (np.max(分组, axis=1, keepdims=True) - 最小值).astype(np.float32)
        np.maximum(缩放值, 1e-8, out=缩放值)
        分组 -= 最小值; 分组 /= 缩放值; np.clip(分组, 0, 1, out=分组)
        分组 *= float(最大量级); np.round(分组, out=分组)
        最小值存 = 最小值.reshape(组数, 维度); 缩放值存 = 缩放值.reshape(组数, 维度)
        最大最小 = max(float(np.max(np.abs(最小值存))), 1e-8)
        最大缩放 = max(float(np.max(缩放值存)), 1e-8)
        量化值 = 缓冲.ravel()[:行数 * 维度].astype(np.uint8)
        del 缓冲, 分组; CleanVRAM()
        return 量化值, 最小值存, 缩放值存, 最大最小, 最大缩放, 行数, 维度, 向量块
    def _GSQ_K解码(Self, 索引, 最小编码, 最大最小, 缩放编码, 最大缩放, 形状, 最大量级):
        向量块 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        行数, 维度 = 形状; 组数 = (行数 + 向量块 - 1) // 向量块
        数量 = 组数 * 维度
        索引 = 索引[:行数 * 维度].astype(np.float32).reshape(行数, 维度)
        最小值 = (Self._解码缩放(最小编码, 数量) * 最大最小).astype(np.float32).reshape(组数, 维度)
        缩放值 = (Self._解码缩放(缩放编码, 数量) * 最大缩放).astype(np.float32).reshape(组数, 维度)
        组索引 = np.arange(行数) // 向量块
        return (索引 / float(最大量级)) * 缩放值[组索引] + 最小值[组索引]
    def _解码填充(Self, 数据, 类型, 总数):
        缓冲 = np.zeros(总数, dtype=类型); 数量 = min(len(数据), 总数); 缓冲[:数量] = 数据[:数量]
        return 缓冲
    def _打包2(Self, 量化值):
        if Self.Numba加速:
            return 加速打包2(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 4
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        四组 = 量化值.reshape(-1, 4)
        return (四组[:,0]|(四组[:,1]<<2)|(四组[:,2]<<4)|(四组[:,3]<<6)).astype(np.uint8)[:(数量+3)//4]
    def _解包2(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包2(压缩, 数量)
        块数 = (数量 + 3) // 4; 缓冲 = np.zeros(块数, dtype=np.uint8)
        取用 = min(len(压缩), 块数); 缓冲[:取用] = 压缩[:取用]
        输出 = np.empty(块数 * 4, dtype=np.uint8)
        输出[0::4]=缓冲&3; 输出[1::4]=(缓冲>>2)&3; 输出[2::4]=(缓冲>>4)&3; 输出[3::4]=(缓冲>>6)&3
        return 输出[:数量]
    def _打包4(Self, 量化值):
        if Self.Numba加速:
            return 加速打包4(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 2
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        双组 = 量化值.reshape(-1, 2)
        return ((双组[:,0]<<4)|双组[:,1]).astype(np.uint8)[:(数量+1)//2]
    def _解包4(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包4(压缩, 数量)
        块数 = (数量 + 1) // 2; 缓冲 = np.zeros(块数, dtype=np.uint8)
        取用 = min(len(压缩), 块数); 缓冲[:取用] = 压缩[:取用]
        输出 = np.empty(块数 * 2, dtype=np.uint8)
        输出[0::2]=(缓冲>>4)&0xF; 输出[1::2]=缓冲&0xF
        return 输出[:数量]
    def _打包3(Self, 量化值):
        if Self.Numba加速:
            return 加速打包3(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 8
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        八组 = 量化值.reshape(-1, 8)
        b0 = (八组[:,0]<<5)|(八组[:,1]<<2)|(八组[:,2]>>1)
        b1 = ((八组[:,2]&1)<<7)|(八组[:,3]<<4)|(八组[:,4]<<1)|(八组[:,5]>>2)
        b2 = ((八组[:,5]&3)<<6)|(八组[:,6]<<3)|八组[:,7]
        return np.column_stack([b0, b1, b2]).ravel()[:((数量+7)//8)*3].astype(np.uint8)
    def _解包3(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包3(压缩, 数量)
        块数 = (数量 + 7) // 8; 缓冲 = np.zeros(块数 * 3, dtype=np.uint8)
        取用 = min(len(压缩), 块数 * 3); 缓冲[:取用] = 压缩[:取用]
        缓冲 = 缓冲.reshape(块数, 3); 输出 = np.empty((块数, 8), dtype=np.uint8)
        输出[:,0]=(缓冲[:,0]>>5)&7; 输出[:,1]=(缓冲[:,0]>>2)&7
        输出[:,2]=((缓冲[:,0]&3)<<1)|(缓冲[:,1]>>7); 输出[:,3]=(缓冲[:,1]>>4)&7
        输出[:,4]=(缓冲[:,1]>>1)&7; 输出[:,5]=((缓冲[:,1]&1)<<2)|(缓冲[:,2]>>6)
        输出[:,6]=(缓冲[:,2]>>3)&7; 输出[:,7]=缓冲[:,2]&7
        return 输出.ravel()[:数量]
    def _打包6(Self, 量化值):
        if Self.Numba加速:
            return 加速打包6(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 4
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        四组 = 量化值.reshape(-1, 4)
        b0 = (四组[:,0]<<2)|(四组[:,1]>>4)
        b1 = ((四组[:,1]&0xF)<<4)|(四组[:,2]>>2)
        b2 = ((四组[:,2]&3)<<6)|四组[:,3]
        return np.column_stack([b0, b1, b2]).ravel()[:((数量+3)//4)*3].astype(np.uint8)
    def _解包6(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包6(压缩, 数量)
        块数 = (数量 + 3) // 4; 缓冲 = np.zeros(块数 * 3, dtype=np.uint8)
        取用 = min(len(压缩), 块数 * 3); 缓冲[:取用] = 压缩[:取用]
        缓冲 = 缓冲.reshape(块数, 3); 输出 = np.empty((块数, 4), dtype=np.uint8)
        输出[:,0]=缓冲[:,0]>>2; 输出[:,1]=((缓冲[:,0]&3)<<4)|(缓冲[:,1]>>4)
        输出[:,2]=((缓冲[:,1]&0xF)<<2)|(缓冲[:,2]>>6); 输出[:,3]=缓冲[:,2]&0x3F
        return 输出.ravel()[:数量]
    def _打包F12(Self, 量化值):
        if Self.Numba加速:
            return 加速打包12(量化值 if 量化值.dtype == numpy.uint16 else 量化值.astype(numpy.uint16))
        数量 = len(量化值); 填充 = (-数量) % 2
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint16)])
        双组 = 量化值.reshape(-1, 2)
        b0 = (双组[:, 0] >> 4).astype(np.uint8)
        b1 = ((双组[:, 0] & 15) << 4 | (双组[:, 1] >> 8)).astype(np.uint8)
        b2 = (双组[:, 1] & 255).astype(np.uint8)
        return np.column_stack([b0, b1, b2]).ravel()[:((数量 + 1) // 2) * 3]

    def _解包F12(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包12(压缩, 数量)
        块数 = (数量 + 1) // 2; 缓冲 = np.zeros(块数 * 3, dtype=np.uint8)
        取用 = min(len(压缩), 块数 * 3); 缓冲[:取用] = 压缩[:取用]
        缓冲 = 缓冲.reshape(块数, 3); 输出 = np.empty(块数 * 2, dtype=np.uint16)
        输出[0::2] = (缓冲[:, 0].astype(np.uint16) << 4) | (缓冲[:, 1].astype(np.uint16) >> 4)
        输出[1::2] = ((缓冲[:, 1].astype(np.uint16) & 15) << 8) | 缓冲[:, 2].astype(np.uint16)
        return 输出[:数量]
    def _线性编码(Self, 数据块, 最大值, 百分比):
        高位, 低位 = Self._百分位(数据块, 百分比)
        范围 = 高位 - 低位; np.maximum(范围, 1e-8, out=范围)
        缩放 = (范围 / 最大值).astype(np.float32)
        零点 = np.clip(np.round(-低位 / 缩放), 0, 最大值).astype(np.uint8)
        if Self.Numba加速:
            量化向量 = 加速线性量化(数据块, 缩放, 零点.astype(np.float32), numpy.float32(最大值))
        else:
            数据块 /= 缩放[:, None]
            数据块 += 零点.astype(np.float32)[:, None]
            np.round(数据块, out=数据块); np.clip(数据块, 0, 最大值, out=数据块)
            量化向量 = 数据块.astype(np.uint8)
        return 量化向量, 缩放, 零点
    def _线性解码(Self, 量化数组, 缩放编码, 零点, 总数, 形状):
        缩放 = Self._解码缩放(缩放编码, 量化数组.shape[0])
        if Self.Numba加速:
            结果 = 加速线性反量化(量化数组, 缩放, 零点.astype(np.float32))
        else:
            结果 = 量化数组.astype(np.float32)
            结果 -= 零点.astype(np.float32)[:, None]; 结果 *= 缩放[:, None]
        return 结果.ravel()[:总数].reshape(形状)
    def _NF编码(Self, 数据块, 百分比, 边界=None):
        边界 = 边界 if 边界 is not None else Self._NF边界
        if 百分比 < 100.0:
            高位, 低位 = Self._百分位(数据块, 百分比, -1, True)
            np.clip(数据块, 低位, 高位, out=数据块)
        缩放 = np.std(数据块, axis=-1, ddof=0).astype(np.float32)
        np.maximum(缩放, 1e-8, out=缩放)
        if Self.Numba加速:
            原始形状 = 数据块.shape
            索引 = 加速NF量化(
                数据块.reshape(-1, 原始形状[-1]),
                边界,
                缩放.ravel()
            ).reshape(原始形状)
        else:
            数据块 /= 缩放[..., None]
            索引 = np.searchsorted(边界, 数据块).astype(np.uint8)
        return 索引, 缩放
    def _NF异常编码(Self, 数据块, 百分比, 边界=None):
        边界 = 边界 if 边界 is not None else Self._NF边界
        块数 = 数据块.shape[0]; 块索引 = np.arange(块数)
        异常索引 = np.argmax(np.abs(数据块), axis=1)
        异常值 = 数据块[块索引, 异常索引].copy()
        数据块[块索引, 异常索引] = 0.0
        if 百分比 < 100.0:
            高位, 低位 = Self._百分位(数据块, 百分比, 1, True)
            np.clip(数据块, 低位, 高位, out=数据块)
        缩放 = np.std(数据块, axis=1, ddof=0).astype(np.float32)
        np.maximum(缩放, 1e-8, out=缩放)
        if Self.Numba加速:
            索引 = 加速NF量化(数据块, 边界, 缩放)
        else:
            数据块 /= 缩放[:, None]
            索引 = np.searchsorted(边界, 数据块).astype(np.uint8)
        return 索引, 缩放, 异常值, 异常索引
    def _NF解码(Self, 索引, 缩放编码, 最大缩放, 总数, 形状, 块大小=None, 中心=None):
        中心 = 中心 if 中心 is not None else Self._NF中心
        if 块大小 is not None:
            块数 = (总数 + 块大小 - 1) // 块大小; 填充总数 = 块数 * 块大小
            缩放 = (Self._解码缩放(缩放编码, 块数) * 最大缩放).astype(np.float32)  # ← 补 块数
            if len(索引) < 填充总数:
                缓冲 = np.zeros(填充总数, dtype=索引.dtype); 缓冲[:len(索引)] = 索引; 索引 = 缓冲
            索引 = 索引[:填充总数].reshape(块数, 块大小)
            缩放 = 缩放[:块数, None]
        else:
            缩放 = (Self._解码缩放(缩放编码) * 最大缩放).astype(np.float32)        # ← 无块大小时不传数量
            if 缩放.ndim == 1: 缩放 = 缩放[:, None]
        结果 = 中心[索引] * 缩放
        return 结果.ravel()[:总数].reshape(形状)
    def FWHT变换(Self, 数组):
        if Self.Numba加速:
            结果 = 数组.copy()
            if 结果.ndim == 1: 结果 = 结果.reshape(1, -1)
            加速FWHT原地(结果); return 结果.reshape(数组.shape)
        数组 = 数组.copy(); 步长 = 1
        while 步长 < 数组.shape[-1]:
            视图 = 数组.reshape(*数组.shape[:-1], -1, 2, 步长)
            左 = 视图[..., 0, :]; 右 = 视图[..., 1, :]; 和 = 左 + 右
            视图[..., 1, :] = 左 - 右; 视图[..., 0, :] = 和; 步长 *= 2
        return 数组
    def IFWHT逆变换(Self, 数组):
        数组 = Self.FWHT变换(数组); return 数组 / 数组.shape[-1]
    def _SVD学习最优旋转(Self, 数据, 中心, 边界, 迭代次数):
        行数, 维度 = 数据.shape; 块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        旋转矩阵 = np.eye(维度, dtype=np.float32)
        if 维度 % 块大小 == 0:
            return Self._SVD旋转_分块(数据, 中心, 边界, 迭代次数, 旋转矩阵, 块大小)
        return Self._SVD旋转_全量(数据, 中心, 边界, 迭代次数, 旋转矩阵, 块大小)
    def _SVD旋转_分块(Self, 数据, 中心, 边界, 迭代次数, 旋转矩阵, 块大小):
        行数, 维度 = 数据.shape; 分片大小 = max(1, min(行数, 256_000_000 // (维度 * 12)))
        for _ in tqdm(range(迭代次数), desc="tqdm.vectors.svd"):
            协方差 = np.zeros((维度, 维度), dtype=np.float32)
            for 起始 in range(0, 行数, 分片大小):
                结束 = min(起始 + 分片大小, 行数); 数据片 = 数据[起始:结束]
                展平 = np.dot(数据片, 旋转矩阵).reshape(-1, 块大小)
                if Self.Numba加速:
                    重建 = 加速SVD处理(展平, 边界, 中心)
                else:
                    块数 = 展平.shape[0]; 块索引 = np.arange(块数)
                    异常索引 = np.argmax(np.abs(展平), axis=1)
                    异常值 = 展平[块索引, 异常索引].copy()
                    平方和 = np.sum(展平*展平, axis=1) - 异常值*异常值
                    求和 = np.sum(展平, axis=1) - 异常值
                    均值 = 求和 / 块大小; 方差 = 平方和/块大小 - 均值*均值; np.maximum(方差, 0, out=方差)
                    标准差 = np.sqrt(方差).astype(np.float32); np.maximum(标准差, 1e-8, out=标准差)
                    展平 /= 标准差[:, None]; 展平[块索引, 异常索引] = 0.0
                    量化索引 = np.searchsorted(边界, 展平).astype(np.uint8)
                    重建 = 中心[量化索引] * 标准差[:, None]; 重建[块索引, 异常索引] = 异常值
                协方差 += np.dot(数据片.T, 重建.reshape(结束 - 起始, 维度))
                del 展平, 重建, 数据片; CleanVRAM()
            U矩阵, S值, Vt矩阵 = np.linalg.svd(协方差); del 协方差, S值
            新旋转 = np.dot(U矩阵, Vt矩阵).astype(np.float32); del U矩阵, Vt矩阵
            if float(np.max(np.abs(新旋转 - 旋转矩阵))) < 1e-5: 旋转矩阵 = 新旋转; break
            旋转矩阵 = 新旋转
        return 旋转矩阵
    def _SVD旋转_全量(Self, 数据, 中心, 边界, 迭代次数, 旋转矩阵, 块大小):
        行数, 维度 = 数据.shape; 块数 = (行数*维度+块大小-1)//块大小
        for _ in tqdm(range(迭代次数), desc="tqdm.vectors.svd"):
            展平 = np.dot(数据, 旋转矩阵).reshape(-1, 块大小)
            if Self.Numba加速:
                重建 = 加速SVD处理(展平, 边界, 中心)
            else:
                块索引 = np.arange(块数); 异常索引 = np.argmax(np.abs(展平), axis=1)
                异常值 = 展平[块索引, 异常索引].copy()
                平方和 = np.sum(展平*展平, axis=1) - 异常值*异常值
                求和 = np.sum(展平, axis=1) - 异常值; 均值 = 求和/块大小
                方差 = 平方和/块大小 - 均值*均值; np.maximum(方差, 0, out=方差)
                标准差 = np.sqrt(方差).astype(np.float32); np.maximum(标准差, 1e-8, out=标准差)
                展平 /= 标准差[:, None]; 展平[块索引, 异常索引] = 0.0
                量化索引 = np.searchsorted(边界, 展平).astype(np.uint8)
                重建 = 中心[量化索引] * 标准差[:, None]; 重建[块索引, 异常索引] = 异常值
            Y = 重建.reshape(行数, 维度); del 重建, 展平; CleanVRAM()
            协方差 = np.dot(数据.T, Y); del Y; CleanVRAM()
            U矩阵, S值, Vt矩阵 = np.linalg.svd(协方差); del 协方差, S值
            新旋转 = np.dot(U矩阵, Vt矩阵).astype(np.float32); del U矩阵, Vt矩阵
            if float(np.max(np.abs(新旋转 - 旋转矩阵))) < 1e-5: 旋转矩阵 = 新旋转; break
            旋转矩阵 = 新旋转
        return 旋转矩阵
    def _SVD学习最优旋转_3值(Self, 数据, 迭代次数):
        行数, 维度 = 数据.shape; 旋转矩阵 = np.eye(维度, dtype=np.float32)
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        alpha列表 = np.linspace(0.1, 1.5, 30).astype(np.float32)
        最佳alpha = 0.1
        for _ in tqdm(range(迭代次数), desc="tqdm.vectors.svd"):
            展平 = np.dot(数据, 旋转矩阵).reshape(-1, 块大小)
            标准差 = np.maximum(np.std(展平, axis=1, ddof=0, keepdims=True), 1e-8)
            归一化 = 展平 / 标准差; del 展平; CleanVRAM()
            采样数 = min(Self.Config.VEC_QUANTIZATION_SPL_SVD, 归一化.size)
            样本 = 归一化.ravel()[np.random.randint(0, 归一化.size, size=采样数)].astype(np.float32)
            if Self.Numba加速:
                最佳alpha = float(加速查找alpha(样本, alpha列表))
            else:
                量化缓冲 = np.empty(采样数, dtype=np.float32); 最佳误差 = float('inf')
                for 当前alpha in alpha列表:
                    量化缓冲[:] = 0.0; 量化缓冲[样本 > 当前alpha] = 1.0; 量化缓冲[样本 < -当前alpha] = -1.0
                    差值 = 样本 - 量化缓冲; 误差 = float(np.dot(差值, 差值) / 采样数)
                    if 误差 < 最佳误差: 最佳误差 = 误差; 最佳alpha = float(当前alpha)
            del 样本; CleanVRAM()
            三值 = np.zeros(归一化.shape, dtype=np.int8)
            三值 += (归一化 > 最佳alpha); 三值 -= (归一化 < -最佳alpha)
            三值浮点 = 三值.astype(np.float32); del 三值
            掩码 = (三值浮点 != 0).astype(np.float32)
            np.abs(归一化, out=归一化); 归一化 *= 掩码
            非零数 = np.maximum(np.sum(掩码, axis=1, keepdims=True), 1.0)
            行缩放 = np.sum(归一化, axis=1, keepdims=True) / 非零数
            del 掩码, 归一化, 非零数; CleanVRAM()
            三值浮点 *= 行缩放; Y = 三值浮点.reshape(行数, 维度); del 三值浮点, 行缩放
            协方差 = np.dot(数据.T, Y); del Y; CleanVRAM()
            U矩阵, S值, Vt矩阵 = np.linalg.svd(协方差); del 协方差, S值
            新旋转 = np.dot(U矩阵, Vt矩阵).astype(np.float32); del U矩阵, Vt矩阵
            if float(np.max(np.abs(新旋转 - 旋转矩阵))) < 1e-5: 旋转矩阵 = 新旋转; break
            旋转矩阵 = 新旋转
        return 旋转矩阵, 最佳alpha
    def _劳埃德最大化(Self, 数据, 聚类数, 迭代次数, 上限=5_000_000):
        if len(数据) > 上限:
            数据 = 数据[np.random.randint(0, len(数据), size=上限)]
        中心 = np.percentile(数据, np.linspace(0, 100, 聚类数+2)[1:-1]).astype(np.float32)
        for _ in tqdm(range(迭代次数), desc="tqdm.vectors.lm"):
            边界 = (中心[:-1] + 中心[1:]) / 2.0
            量化索引 = np.searchsorted(边界, 数据).astype(np.intp)
            计数 = np.bincount(量化索引, minlength=聚类数); 加权和 = np.bincount(量化索引, weights=数据, minlength=聚类数)
            有效 = 计数 > 0; 新中心 = np.where(有效, 加权和 / np.maximum(计数, 1), 中心).astype(np.float32)
            if float(np.max(np.abs(新中心 - 中心))) < Self.Config.VEC_QUANTIZATION_ES_LM: 中心 = 新中心; break
            中心 = 新中心
        中心 = np.sort(中心); return 中心, (中心[:-1] + 中心[1:]) / 2.0
    def _编码缩放(Self, 数组):
        类型 = Self.Config.VEC_QUANTIZATION_SCALE_TYPE
        数组 = np.ascontiguousarray(数组, dtype=np.float32).ravel()
        if 类型 == "Float32":
            return 数组.copy()
        elif 类型 == "Float16":
            return 数组.astype(np.float16)
        elif 类型 == "BFloat16":
            return Self.F32编码BF16(数组)["Vector"]
        elif 类型 == "Float8_E4M3":
            return Self.F32编码FP8_E4M3(数组)["Vector"]
        elif 类型 == "Float8_E0M7":
            return Self.F32编码F8_E0M7(数组)
        elif 类型 == "Float12_E0M11":
            return Self.F32编码F12_E0M11(数组)["Vector"]
        else:
            return Self.F32编码F16_E0M15(数组)
    def _解码缩放(Self, 数据, 数量=None):
        类型 = Self.Config.VEC_QUANTIZATION_SCALE_TYPE
        if 类型 == "Float32":
            结果 = np.asarray(数据, dtype=np.float32).ravel()
        elif 类型 == "Float16":
            结果 = np.asarray(数据, dtype=np.float16).astype(np.float32).ravel()
        elif 类型 == "BFloat16":
            结果 = Self.BF16解码F32(数据).ravel()
        elif 类型 == "Float8_E4M3":
            结果 = Self.FP8_E4M3解码F32(数据).ravel()
        elif 类型 == "Float8_E0M7":
            结果 = Self.F8_E0M7解码F32(数据).ravel()
        elif 类型 == "Float12_E0M11":
            if 数量 is None: 数量 = (len(np.asarray(数据)) // 3) * 2
            结果 = Self.F12_E0M11解码F32(数据, (数量,))
        else:
            结果 = np.atleast_1d(Self.F16_E0M15解码F32(数据)).astype(np.float32).ravel()
        if 数量 is not None and len(结果) > 数量:
            return 结果[:数量].copy()
        return 结果
#====================================================================================================8Bit====================================================================================================#
    def F32编码Q8_K(Self, 数组):
        块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
        分块数据, 形状, 总数 = Self._分块(数组, 块大小)
        量化向量, 缩放, 零点 = Self._线性编码(分块数据, 255, Self.Config.VEC_QUANTILE * 100)
        del 分块数据; CleanVRAM()
        return {"Vector": 量化向量.ravel()[:总数], "Scale": Self._编码缩放(缩放),
                "ZeroPoint": 零点, "Shape": 形状}
    def Q8_K解码F32(Self, 数据, 缩放编码, 零点, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 填充总数 = ((总数+块大小-1)//块大小)*块大小
        return Self._线性解码(Self._解码填充(数据, np.uint8, 填充总数).reshape(-1, 块大小), 缩放编码, 零点, 总数, 形状)
#====================================================================================================8Bit====================================================================================================#
#====================================================================================================6Bit====================================================================================================#
    def F32编码Q6_K(Self, 数组):
        块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
        分块数据, 形状, 总数 = Self._分块(数组, 块大小)
        量化向量, 缩放, 零点 = Self._线性编码(分块数据, 63, Self.Config.VEC_QUANTILE * 100)
        块数 = 分块数据.shape[0]; 组大小 = 块大小 // 4; 量化组 = 量化向量.reshape(块数, 组大小, 4)
        压缩 = np.empty((块数, 组大小, 3), dtype=np.uint8)
        压缩[:,:,0]=(量化组[:,:,0]<<2)|(量化组[:,:,1]>>4)
        压缩[:,:,1]=((量化组[:,:,1]&0xF)<<4)|(量化组[:,:,2]>>2)
        压缩[:,:,2]=((量化组[:,:,2]&3)<<6)|量化组[:,:,3]
        del 分块数据, 量化向量, 量化组; CleanVRAM()
        return {"Vector": 压缩.ravel()[:((总数+3)//4)*3], "Scale": Self._编码缩放(缩放),
                "ZeroPoint": 零点, "Shape": 形状}
    def Q6_K解码F32(Self, 数据, 缩放编码, 零点, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 填充总数 = ((总数+块大小-1)//块大小)*块大小; 块数 = 填充总数//块大小; 组大小 = 块大小//4
        缓冲 = Self._解码填充(数据, np.uint8, 块数*组大小*3).reshape(块数, 组大小, 3)
        量化组 = np.empty((块数, 组大小, 4), dtype=np.uint8)
        量化组[:,:,0]=缓冲[:,:,0]>>2; 量化组[:,:,1]=((缓冲[:,:,0]&3)<<4)|(缓冲[:,:,1]>>4)
        量化组[:,:,2]=((缓冲[:,:,1]&0xF)<<2)|(缓冲[:,:,2]>>6); 量化组[:,:,3]=缓冲[:,:,2]&0x3F
        return Self._线性解码(量化组.reshape(块数, 块大小), 缩放编码, 零点, 总数, 形状)
#====================================================================================================6Bit====================================================================================================#
#====================================================================================================4Bit====================================================================================================#
    def F32编码Q4_K(Self, 数组):
        块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
        分块数据, 形状, 总数 = Self._分块(数组, 块大小)
        量化向量, 缩放, 零点 = Self._线性编码(分块数据, 15, Self.Config.VEC_QUANTILE * 100)
        压缩 = Self._打包4(量化向量.ravel()); del 分块数据, 量化向量; CleanVRAM()
        return {"Vector": 压缩, "Scale": Self._编码缩放(缩放), "ZeroPoint": 零点, "Shape": 形状}
    def Q4_K解码F32(Self, 数据, 缩放编码, 零点, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 填充总数 = ((总数+块大小-1)//块大小)*块大小
        return Self._线性解码(Self._解包4(数据, 填充总数).reshape(-1, 块大小), 缩放编码, 零点, 总数, 形状)
    def F32编码Q4_K_H(Self, 数组):
        Self._确保H16()
        数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32); 行数, 维度 = 数组.shape
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE); 百分比 = Self.Config.VEC_QUANTILE * 100
        旋转 = np.ascontiguousarray(np.matmul(数组.reshape(行数, 维度//16, 16), Self.H16矩阵.T).reshape(行数, 维度))
        del 数组; 数据块 = 旋转.reshape(行数, 维度//块大小, 块大小)
        if 百分比 < 100.0:
            高位, 低位 = Self._百分位(数据块, 百分比, 轴=2, 保持维度=True)
            数据块 = np.clip(数据块, 低位, 高位)
        最大值 = np.max(数据块, axis=2); 最小值 = np.min(数据块, axis=2)
        缩放值 = np.maximum(最大值 - 最小值, 1e-8)
        归一化 = (数据块 - 最小值[:,:,None]) / 缩放值[:,:,None]
        del 数据块, 旋转; 索引 = np.round(归一化 * 15).astype(np.uint8); del 归一化
        CleanVRAM()
        最小值平坦 = 最小值.ravel().astype(np.float32)
        缩放平坦 = 缩放值.ravel().astype(np.float32)
        最大最小 = max(float(np.max(np.abs(最小值平坦))), 1e-8)
        最大缩放 = max(float(np.max(缩放平坦)), 1e-8)
        return {"Vector": Self._打包4(索引.ravel()),
                "Min": Self._编码缩放(最小值平坦 / 最大最小), "MaxMin": 最大最小,
                "Scale": Self._编码缩放(缩放平坦 / 最大缩放), "MaxScale": 最大缩放,
                "Shape": (行数, 维度)}
    def Q4_K_H解码F32(Self, 数据, 最小值编码, 最大最小, 缩放编码, 最大缩放, 形状):
        Self._确保H16(); 行数, 维度 = 形状; 块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        索引 = Self._解包4(数据, 行数 * 维度)[:行数 * 维度].reshape(行数, 维度//块大小, 块大小)
        块数 = 行数 * 维度 // 块大小
        最小值 = (Self._解码缩放(最小值编码, 块数) * 最大最小).astype(np.float32).reshape(行数, 维度//块大小, 1)
        缩放值 = (Self._解码缩放(缩放编码, 块数) * 最大缩放).astype(np.float32).reshape(行数, 维度//块大小, 1)
        重建 = (索引.astype(np.float32) / 15.0) * 缩放值 + 最小值; del 索引, 最小值, 缩放值
        return np.ascontiguousarray(
            np.matmul(重建.reshape(行数, 维度//16, 16), Self.H16矩阵.T).reshape(行数, 维度))
    def F32编码GSQ4_0(Self, 数组):
        数组 = np.asarray(数组, dtype=np.float32); 行数, 维度 = 数组.shape; 百分比 = Self.Config.VEC_QUANTILE * 100
        最大值, 最小值 = Self._百分位(数组, 百分比, 轴=0)
        缩放值 = np.maximum(最大值 - 最小值, 1e-8)
        归一化 = (np.clip(数组, 最小值, 最大值) - 最小值) / 缩放值
        del 数组; 索引 = np.round(归一化 * 15).astype(np.uint8); del 归一化; CleanVRAM()
        return {"Vector": Self._打包4(索引.ravel()),
                "Min": Self._编码缩放(最小值.ravel()),
                "Scale": Self._编码缩放(缩放值.ravel()),
                "Shape": (行数, 维度)}
    def GSQ4_0解码F32(Self, 数据, 最小编码, 缩放编码, 形状):
        行数, 维度 = 形状; 索引 = Self._解包4(数据, 行数*维度)[:行数*维度].reshape(行数, 维度)
        数量 = 行数 * 维度
        最小值 = Self._解码缩放(最小编码, 数量).astype(np.float32)
        缩放值 = Self._解码缩放(缩放编码, 数量).astype(np.float32)
        return (索引.astype(np.float32) / 15.0) * 缩放值 + 最小值
    def F32编码Q4_SVD_LM(Self, 数组):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        数组 = np.asarray(数组, dtype=np.float32); 形状 = 数组.shape
        旋转矩阵 = Self._SVD学习最优旋转(数组, Self._NF3中心.copy(), Self._NF3边界.copy(),
                                Self.Config.VEC_QUANTIZATION_ITRS_SVD)
        旋转后 = np.dot(数组, 旋转矩阵); del 数组; CleanVRAM()
        展平 = 旋转后.reshape(-1, 块大小)
        高位, 低位 = Self._百分位(展平, Self.Config.VEC_QUANTILE*100, 轴=1, 保持维度=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.std(展平, axis=1, ddof=0).astype(np.float32)
        np.maximum(标准差, 1e-8, out=标准差); 展平 /= 标准差[:, None]
        中心, 边界 = Self._劳埃德最大化(展平.ravel(), 16, Self.Config.VEC_QUANTIZATION_ITRS_LM)
        量化索引 = np.searchsorted(边界, 展平).astype(np.uint8)
        del 展平, 旋转后; CleanVRAM(); 最大缩放 = max(float(np.max(标准差)), 1e-8)
        return {"PackedVector": Self._打包4(量化索引.ravel()),
                "Scale": Self._编码缩放(标准差/最大缩放), "MaxScale": 最大缩放,
                "RotMatrix": Self.F32编码F16_E0M15(旋转矩阵),
                "Centroids": 中心.astype(np.float32), "Shape": 形状}

    def Q4_SVD_LM解码F32(Self, 压缩, 缩放编码, 最大缩放, 旋转编码, 中心, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 行数, 维度 = 形状; 块数 = (总数 + 块大小 - 1) // 块大小
        量化索引 = Self._解包4(压缩, 块数*块大小).reshape(块数, 块大小)
        标准差 = (Self._解码缩放(缩放编码, 块数) * 最大缩放)[:块数, None]
        旋转矩阵 = np.asarray(Self.F16_E0M15解码F32(旋转编码), dtype=np.float32)
        return np.ascontiguousarray(np.dot((中心[量化索引]*标准差).reshape(行数, 维度), 旋转矩阵.T))
#====================================================================================================4Bit====================================================================================================#
#====================================================================================================3Bit====================================================================================================#
    def F32编码Q3_K(Self, 数组):
        块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
        分块数据, 形状, 总数 = Self._分块(数组, 块大小)
        量化向量, 缩放, 零点 = Self._线性编码(分块数据, 7, Self.Config.VEC_QUANTILE * 100)
        块数 = 分块数据.shape[0]; 组大小 = 块大小//8; 量化组 = 量化向量.reshape(块数, 组大小, 8)
        压缩 = np.empty((块数, 组大小, 3), dtype=np.uint8)
        压缩[:,:,0]=(量化组[:,:,0]<<5)|(量化组[:,:,1]<<2)|(量化组[:,:,2]>>1)
        压缩[:,:,1]=((量化组[:,:,2]&1)<<7)|(量化组[:,:,3]<<4)|(量化组[:,:,4]<<1)|(量化组[:,:,5]>>2)
        压缩[:,:,2]=((量化组[:,:,5]&3)<<6)|(量化组[:,:,6]<<3)|量化组[:,:,7]
        del 分块数据, 量化向量, 量化组; CleanVRAM()
        return {"Vector": 压缩.ravel()[:((总数+7)//8)*3], "Scale": Self._编码缩放(缩放),
                "ZeroPoint": 零点, "Shape": 形状}
    def Q3_K解码F32(Self, 数据, 缩放编码, 零点, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 填充总数 = ((总数+块大小-1)//块大小)*块大小; 块数 = 填充总数//块大小; 组大小 = 块大小//8
        缓冲 = Self._解码填充(数据, np.uint8, 块数*组大小*3).reshape(块数, 组大小, 3)
        量化组 = np.empty((块数, 组大小, 8), dtype=np.uint8)
        量化组[:,:,0]=(缓冲[:,:,0]>>5)&7; 量化组[:,:,1]=(缓冲[:,:,0]>>2)&7
        量化组[:,:,2]=((缓冲[:,:,0]&3)<<1)|(缓冲[:,:,1]>>7); 量化组[:,:,3]=(缓冲[:,:,1]>>4)&7
        量化组[:,:,4]=(缓冲[:,:,1]>>1)&7; 量化组[:,:,5]=((缓冲[:,:,1]&1)<<2)|(缓冲[:,:,2]>>6)
        量化组[:,:,6]=(缓冲[:,:,2]>>3)&7; 量化组[:,:,7]=缓冲[:,:,2]&7
        return Self._线性解码(量化组.reshape(块数, 块大小), 缩放编码, 零点, 总数, 形状)
    def F32编码Q3_K_H(Self, 数组):
        Self._确保H16()
        数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32); 行数, 维度 = 数组.shape
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE); 百分比 = Self.Config.VEC_QUANTILE * 100
        旋转 = np.ascontiguousarray(np.matmul(数组.reshape(行数, 维度//16, 16),
                                               Self.H16矩阵.T).reshape(行数, 维度))
        del 数组; 数据块 = 旋转.reshape(行数, 维度//块大小, 块大小)
        if 百分比 < 100.0:
            高位, 低位 = Self._百分位(数据块, 百分比, 轴=2, 保持维度=True)
            数据块 = np.clip(数据块, 低位, 高位)
        最大值 = np.max(数据块, axis=2); 最小值 = np.min(数据块, axis=2)
        缩放值 = np.maximum(最大值 - 最小值, 1e-8)
        归一化 = (数据块 - 最小值[:,:,None]) / 缩放值[:,:,None]
        del 数据块, 旋转; 索引 = np.round(归一化 * 7).astype(np.uint8); del 归一化; CleanVRAM()
        平坦 = 索引.ravel(); 总数 = len(平坦); 组数 = (总数 + 7) // 8
        if 总数 % 8: 平坦 = np.concatenate([平坦, np.zeros(组数*8-总数, dtype=np.uint8)])
        量化组 = 平坦.reshape(组数, 8); 压缩 = np.empty((组数, 3), dtype=np.uint8)
        压缩[:,0] = (量化组[:,0]<<5)|(量化组[:,1]<<2)|(量化组[:,2]>>1)
        压缩[:,1] = ((量化组[:,2]&1)<<7)|(量化组[:,3]<<4)|(量化组[:,4]<<1)|(量化组[:,5]>>2)
        压缩[:,2] = ((量化组[:,5]&3)<<6)|(量化组[:,6]<<3)|量化组[:,7]
        del 量化组, 索引, 平坦; CleanVRAM()
        最小值平坦 = 最小值.ravel().astype(np.float32)
        缩放平坦 = 缩放值.ravel().astype(np.float32)
        最大最小 = max(float(np.max(np.abs(最小值平坦))), 1e-8)
        最大缩放 = max(float(np.max(缩放平坦)), 1e-8)
        return {"Vector": 压缩.ravel(),
                "Min": Self._编码缩放(最小值平坦 / 最大最小), "MaxMin": 最大最小,
                "Scale": Self._编码缩放(缩放平坦 / 最大缩放), "MaxScale": 最大缩放,
                "Shape": (行数, 维度)}
    def Q3_K_H解码F32(Self, 压缩, 最小值编码, 最大最小, 缩放编码, 最大缩放, 形状):
        Self._确保H16(); 行数, 维度 = 形状; 块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = 行数 * 维度; 组数 = (总数 + 7) // 8
        缓冲 = Self._解码填充(压缩, np.uint8, 组数*3).reshape(组数, 3)
        量化组 = np.empty((组数, 8), dtype=np.uint8)
        量化组[:,0] = (缓冲[:,0]>>5)&7; 量化组[:,1] = (缓冲[:,0]>>2)&7
        量化组[:,2] = ((缓冲[:,0]&3)<<1)|(缓冲[:,1]>>7)
        量化组[:,3] = (缓冲[:,1]>>4)&7; 量化组[:,4] = (缓冲[:,1]>>1)&7
        量化组[:,5] = ((缓冲[:,1]&1)<<2)|(缓冲[:,2]>>6)
        量化组[:,6] = (缓冲[:,2]>>3)&7; 量化组[:,7] = 缓冲[:,2]&7
        索引 = 量化组.ravel()[:总数].reshape(行数, 维度//块大小, 块大小)
        块数 = 行数 * 维度 // 块大小
        最小值 = (Self._解码缩放(最小值编码, 块数) * 最大最小).astype(np.float32).reshape(行数, 维度//块大小, 1)
        缩放值 = (Self._解码缩放(缩放编码, 块数) * 最大缩放).astype(np.float32).reshape(行数, 维度//块大小, 1)
        重建 = (索引.astype(np.float32) / 7.0) * 缩放值 + 最小值
        del 索引, 最小值, 缩放值
        return np.ascontiguousarray(
            np.matmul(重建.reshape(行数, 维度//16, 16), Self.H16矩阵.T).reshape(行数, 维度))

    def F32编码Q3_SVD_LM(Self, 数组):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        数组 = np.asarray(数组, dtype=np.float32); 形状 = 数组.shape
        旋转矩阵 = Self._SVD学习最优旋转(数组, Self._NF3中心.copy(), Self._NF3边界.copy(),
                                Self.Config.VEC_QUANTIZATION_ITRS_SVD)
        旋转后 = np.dot(数组, 旋转矩阵); del 数组; CleanVRAM()
        展平 = 旋转后.reshape(-1, 块大小)
        高位, 低位 = Self._百分位(展平, Self.Config.VEC_QUANTILE*100, 轴=1, 保持维度=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.std(展平, axis=1, ddof=0).astype(np.float32)
        np.maximum(标准差, 1e-8, out=标准差); 展平 /= 标准差[:, None]
        中心, 边界 = Self._劳埃德最大化(展平.ravel(), 8, Self.Config.VEC_QUANTIZATION_ITRS_LM)
        量化索引 = np.searchsorted(边界, 展平).astype(np.uint8)
        del 展平, 旋转后; CleanVRAM(); 最大缩放 = max(float(np.max(标准差)), 1e-8)
        平坦 = 量化索引.ravel(); 总数 = len(平坦); 组数 = (总数 + 7) // 8
        if 总数 % 8: 平坦 = np.concatenate([平坦, np.zeros(组数*8-总数, dtype=np.uint8)])
        量化组 = 平坦.reshape(组数, 8); 压缩 = np.empty((组数, 3), dtype=np.uint8)
        压缩[:,0] = (量化组[:,0]<<5)|(量化组[:,1]<<2)|(量化组[:,2]>>1)
        压缩[:,1] = ((量化组[:,2]&1)<<7)|(量化组[:,3]<<4)|(量化组[:,4]<<1)|(量化组[:,5]>>2)
        压缩[:,2] = ((量化组[:,5]&3)<<6)|(量化组[:,6]<<3)|量化组[:,7]
        del 量化组, 量化索引, 平坦; CleanVRAM()
        return {"PackedVector": 压缩.ravel(),
                "Scale": Self._编码缩放(标准差/最大缩放), "MaxScale": 最大缩放,
                "RotMatrix": Self.F32编码F16_E0M15(旋转矩阵),
                "Centroids": 中心.astype(np.float32), "Shape": 形状}

    def Q3_SVD_LM解码F32(Self, 压缩, 缩放编码, 最大缩放, 旋转编码, 中心, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 行数, 维度 = 形状; 块数 = (总数 + 块大小 - 1) // 块大小
        组数 = (块数*块大小 + 7) // 8
        缓冲 = Self._解码填充(压缩, np.uint8, 组数*3).reshape(组数, 3)
        量化组 = np.empty((组数, 8), dtype=np.uint8)
        量化组[:,0] = (缓冲[:,0]>>5)&7; 量化组[:,1] = (缓冲[:,0]>>2)&7
        量化组[:,2] = ((缓冲[:,0]&3)<<1)|(缓冲[:,1]>>7)
        量化组[:,3] = (缓冲[:,1]>>4)&7; 量化组[:,4] = (缓冲[:,1]>>1)&7
        量化组[:,5] = ((缓冲[:,1]&1)<<2)|(缓冲[:,2]>>6)
        量化组[:,6] = (缓冲[:,2]>>3)&7; 量化组[:,7] = 缓冲[:,2]&7
        量化索引 = 量化组.ravel()[:块数*块大小].reshape(块数, 块大小)
        标准差 = (Self._解码缩放(缩放编码, 块数) * 最大缩放)[:块数, None]
        旋转矩阵 = np.asarray(Self.F16_E0M15解码F32(旋转编码), dtype=np.float32)
        return np.ascontiguousarray(np.dot((中心[量化索引]*标准差).reshape(行数, 维度), 旋转矩阵.T))
#====================================================================================================3Bit====================================================================================================#
#====================================================================================================2Bit====================================================================================================#
    def F32编码Q2_K(Self, 数组):
        块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
        分块数据, 形状, 总数 = Self._分块(数组, 块大小)
        量化向量, 缩放, 零点 = Self._线性编码(分块数据, 3, Self.Config.VEC_QUANTILE * 100)
        块数 = 分块数据.shape[0]; 组大小 = 块大小//4; 量化组 = 量化向量.reshape(块数, 组大小, 4)
        压缩 = (量化组[:,:,0]|(量化组[:,:,1]<<2)|(量化组[:,:,2]<<4)|(量化组[:,:,3]<<6)).astype(np.uint8)
        del 分块数据, 量化向量, 量化组; CleanVRAM()
        return {"Vector": 压缩.ravel()[:(总数+3)//4], "Scale": Self._编码缩放(缩放),
                "ZeroPoint": 零点, "Shape": 形状}
    def Q2_K解码F32(Self, 数据, 缩放编码, 零点, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 填充总数 = ((总数+块大小-1)//块大小)*块大小; 块数 = 填充总数//块大小; 组大小 = 块大小//4
        缓冲 = Self._解码填充(数据, np.uint8, 块数*组大小).reshape(块数, 组大小, 1)
        量化组 = np.concatenate([缓冲&3,(缓冲>>2)&3,(缓冲>>4)&3,(缓冲>>6)&3], axis=2)
        return Self._线性解码(量化组.reshape(块数, 块大小), 缩放编码, 零点, 总数, 形状)
    def F32编码Q2_NF(Self, 数组):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        分块数据, 形状, 总数 = Self._分块(数组, 块大小)
        索引, 缩放 = Self._NF编码(分块数据, Self.Config.VEC_QUANTILE * 100.0)
        del 分块数据; CleanVRAM()
        最大缩放 = max(float(np.max(缩放)), 1e-8)
        return {"Vector": Self._打包2(索引.ravel()[:总数]),
                "Scale": Self._编码缩放(缩放 / 最大缩放), "MaxScale": 最大缩放,
                "Shape": 形状, "BlockSize": 块大小}
    def Q2_NF解码F32(Self, 数据, 缩放编码, 最大缩放, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 填充总数 = ((总数+块大小-1)//块大小)*块大小
        return Self._NF解码(Self._解包2(数据, 填充总数), 缩放编码, 最大缩放, 总数, 形状, 块大小=块大小)
    def F32编码Q2_E_NF(Self, 数组):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        分块数据, 形状, 总数 = Self._分块(数组, 块大小)
        索引, 缩放, 异常值, 异常索引 = Self._NF异常编码(分块数据, Self.Config.VEC_QUANTILE * 100)
        del 分块数据; CleanVRAM()
        最大缩放 = max(float(np.max(缩放)), 1e-8); 最大异常 = max(float(np.max(np.abs(异常值))), 1e-8)
        类型 = np.uint8 if 块大小 <= 256 else np.uint16
        return {"Vector": Self._打包2(索引.ravel()[:总数]),
                "Scale": Self._编码缩放(缩放/最大缩放), "MaxScale": 最大缩放,
                "Outliers": Self._编码缩放(异常值/最大异常), "MaxOutlier": 最大异常,
                "OutlierIdx": 异常索引.astype(类型), "Shape": 形状, "BlockSize": 块大小}
    def Q2_E_NF解码F32(Self, 数据, 缩放编码, 最大缩放, 异常编码, 最大异常, 异常索引, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 填充总数 = ((总数+块大小-1)//块大小)*块大小; 块数 = 填充总数//块大小
        索引 = Self._解包2(数据, 填充总数).reshape(块数, 块大小)
        缩放 = (Self._解码缩放(缩放编码, 块数) * 最大缩放).astype(np.float32)[:, None]
        异常值 = (Self._解码缩放(异常编码, 块数) * 最大异常).astype(np.float32)
        结果 = Self._NF中心[索引] * 缩放; del 索引, 缩放
        结果[np.arange(块数), 异常索引[:块数]] = 异常值
        return 结果.ravel()[:总数].reshape(形状)
    def F32编码Q2_SVD_LM(Self, 数组):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        数组 = np.asarray(数组, dtype=np.float32); 形状 = 数组.shape
        旋转矩阵 = Self._SVD学习最优旋转(数组, Self._NF中心.copy(), Self._NF边界.copy(),
                                Self.Config.VEC_QUANTIZATION_ITRS_SVD)
        旋转后 = np.dot(数组, 旋转矩阵); del 数组; CleanVRAM()
        展平 = 旋转后.reshape(-1, 块大小)
        高位, 低位 = Self._百分位(展平, Self.Config.VEC_QUANTILE*100, 轴=1, 保持维度=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.std(展平, axis=1, ddof=0).astype(np.float32)
        np.maximum(标准差, 1e-8, out=标准差); 展平 /= 标准差[:, None]
        中心, 边界 = Self._劳埃德最大化(展平.ravel(), 4, Self.Config.VEC_QUANTIZATION_ITRS_LM)
        量化索引 = np.searchsorted(边界, 展平).astype(np.uint8)
        del 展平, 旋转后; CleanVRAM(); 最大缩放 = max(float(np.max(标准差)), 1e-8)
        return {"PackedVector": Self._打包2(量化索引.ravel()),
                "Scale": Self._编码缩放(标准差/最大缩放), "MaxScale": 最大缩放,
                "RotMatrix": Self.F32编码F16_E0M15(旋转矩阵),
                "Centroids": 中心.astype(np.float32), "Shape": 形状}
    def Q2_SVD_LM解码F32(Self, 压缩, 缩放编码, 最大缩放, 旋转编码, 中心, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 行数, 维度 = 形状; 块数 = (总数+块大小-1)//块大小
        量化索引 = Self._解包2(压缩, 块数*块大小).reshape(块数, 块大小)
        标准差 = (Self._解码缩放(缩放编码, 块数) * 最大缩放)[:块数, None]
        旋转矩阵 = np.asarray(Self.F16_E0M15解码F32(旋转编码), dtype=np.float32)
        return np.ascontiguousarray(np.dot((中心[量化索引]*标准差).reshape(行数, 维度), 旋转矩阵.T))
    def F32编码Q2_E_SVD_LM(Self, 数组):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE); 百分比 = Self.Config.VEC_QUANTILE * 100
        数组 = np.asarray(数组, dtype=np.float32); 形状 = 数组.shape
        旋转矩阵 = Self._SVD学习最优旋转(数组, Self._NF中心.copy(), Self._NF边界.copy(),
                                Self.Config.VEC_QUANTIZATION_ITRS_SVD)
        旋转后 = np.dot(数组, 旋转矩阵); del 数组; CleanVRAM()
        展平 = 旋转后.reshape(-1, 块大小); 块数 = 展平.shape[0]; 块索引 = np.arange(块数)
        异常索引 = np.argmax(np.abs(展平), axis=1); 异常值 = 展平[块索引, 异常索引].copy()
        展平[块索引, 异常索引] = 0.0
        if 百分比 < 100.0:
            高位, 低位 = Self._百分位(展平, 百分比, 轴=1, 保持维度=True)
            np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.std(展平, axis=1, ddof=0).astype(np.float32)
        np.maximum(标准差, 1e-8, out=标准差); 展平 /= 标准差[:, None]
        中心, 边界 = Self._劳埃德最大化(展平.ravel(), 4, Self.Config.VEC_QUANTIZATION_ITRS_LM)
        量化索引 = np.searchsorted(边界, 展平).astype(np.uint8)
        del 展平, 旋转后; CleanVRAM()
        最大缩放 = max(float(np.max(标准差)), 1e-8); 最大异常 = max(float(np.max(np.abs(异常值))), 1e-8)
        return {"Vector": Self._打包2(量化索引.ravel()),
                "Scale": Self._编码缩放(标准差/最大缩放), "MaxScale": 最大缩放,
                "Outliers": Self._编码缩放(异常值/最大异常), "MaxOutlier": 最大异常,
                "OutlierIdx": 异常索引.astype(np.uint8), "RotMatrix": Self.F32编码F16_E0M15(旋转矩阵),
                "Centroids": 中心.astype(np.float32), "Shape": 形状}
    def Q2_E_SVD_LM解码F32(Self, 压缩, 缩放编码, 最大缩放, 异常编码, 最大异常, 异常索引, 旋转编码, 中心, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        总数 = math.prod(形状); 行数, 维度 = 形状; 块数 = (总数+块大小-1)//块大小
        量化索引 = Self._解包2(压缩, 块数*块大小).reshape(块数, 块大小)
        标准差 = (Self._解码缩放(缩放编码, 块数) * 最大缩放)[:块数, None]
        异常值 = (Self._解码缩放(异常编码, 块数) * 最大异常)[:块数]
        旋转矩阵 = np.asarray(Self.F16_E0M15解码F32(旋转编码), dtype=np.float32)
        结果 = 中心[量化索引] * 标准差; del 量化索引, 标准差
        结果[np.arange(块数), 异常索引[:块数]] = 异常值
        return np.ascontiguousarray(np.dot(结果.ravel()[:总数].reshape(行数, 维度), 旋转矩阵.T))
#====================================================================================================2Bit====================================================================================================#
#====================================================================================================1.58Bit====================================================================================================#
    def F32编码TQ1_SVD(Self, 数组):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        数组 = np.asarray(数组, dtype=np.float32); 形状 = 数组.shape
        旋转矩阵, alpha = Self._SVD学习最优旋转_3值(数组, Self.Config.VEC_QUANTIZATION_ITRS_SVD)
        展平 = np.ascontiguousarray(np.dot(数组, 旋转矩阵)).reshape(-1, 块大小)
        del 数组; CleanVRAM()
        标准差 = np.maximum(np.std(展平, axis=1, ddof=0), 1e-8)
        阈值 = alpha * 标准差[:, None]
        三值 = np.zeros(展平.shape, dtype=np.int8)
        三值 += (展平 > 阈值); 三值 -= (展平 < -阈值)
        三值浮点 = 三值.astype(np.float32); del 三值
        掩码 = (三值浮点 != 0); np.abs(展平, out=展平); 展平 *= 掩码
        非零数 = np.maximum(np.sum(掩码, axis=1), 1.0); 行缩放 = np.sum(展平, axis=1) / 非零数
        del 掩码, 展平; CleanVRAM()
        映射值 = (三值浮点 + 1).astype(np.uint8).ravel(); del 三值浮点
        if Self.Numba加速:
            压缩 = 加速三值打包(映射值)
        else:
            填充 = (-len(映射值)) % 5
            if 填充: 映射值 = np.concatenate([映射值, np.zeros(填充, dtype=np.uint8)])
            五组 = 映射值.reshape(-1, 5)
            压缩 = (五组[:,0]+五组[:,1]*3+五组[:,2]*9+五组[:,3]*27+五组[:,4]*81).astype(np.uint8)
        最大缩放 = max(float(np.max(行缩放)), 1e-8)
        return {"PackedTernary": 压缩, "ScaleINT8": np.round(行缩放/最大缩放*255).astype(np.uint8),
                "MaxScale": Self._编码缩放(numpy.array([最大缩放], dtype=numpy.float32)),
                "RotMatrix": Self.F32编码F16_E0M15(旋转矩阵), "Shape": 形状}
    def TQ1_SVD解码F32(Self, 压缩, 缩放INT8, 最大缩放编码, 旋转编码, 形状):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        最大缩放 = float(Self._解码缩放(最大缩放编码, 1)[0]); 旋转矩阵 = np.asarray(Self.F16_E0M15解码F32(旋转编码), dtype=np.float32)
        行数, 维度 = 形状; 总数 = 行数 * 维度
        if Self.Numba加速:
            解码值 = 加速三值解包(压缩.astype(np.uint8), 总数)
        else:
            p = 压缩.astype(np.uint16).copy()
            v0=p%3; p//=3; v1=p%3; p//=3; v2=p%3; p//=3; v3=p%3; p//=3; v4=p%3
            解码值 = np.empty(len(压缩)*5, dtype=np.int8)
            解码值[0::5]=v0; 解码值[1::5]=v1; 解码值[2::5]=v2; 解码值[3::5]=v3; 解码值[4::5]=v4
            解码值 = 解码值[:总数]
        三值浮点 = (解码值 - 1).astype(np.float32).reshape(-1, 块大小)
        缩放 = (缩放INT8.astype(np.float32) / 255.0) * 最大缩放
        return np.ascontiguousarray(np.dot((三值浮点 * 缩放[:, None]).reshape(行数, 维度), 旋转矩阵.T))
#====================================================================================================1.58Bit====================================================================================================#
#====================================================================================================1Bit====================================================================================================#
    def F32编码Q1_K_M(Self, 数组):
        块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        数组 = np.asarray(数组, dtype=np.float32); 行数, 维度 = 数组.shape
        均值向量 = np.mean(数组, axis=0, dtype=np.float32); 中心化 = 数组 - 均值向量; del 数组
        百分99 = np.percentile(np.abs(中心化), Self.Config.VEC_QUANTILE * 100)
        局部缩放 = np.mean(np.abs(np.clip(中心化, -百分99, 百分99).reshape(行数, 维度//块大小, 块大小)), axis=2, dtype=np.float32)
        np.maximum(局部缩放, 1e-8, out=局部缩放); CleanVRAM()
        二进制 = np.packbits((中心化 > 0).ravel()).reshape(行数, 维度//8); del 中心化; CleanVRAM()
        展平缩放 = 局部缩放.ravel().astype(np.float32)
        最大缩放 = max(float(np.max(展平缩放)), 1e-8)
        均值平坦 = 均值向量.ravel().astype(np.float32)
        最大均值 = max(float(np.max(np.abs(均值平坦))), 1e-8)
        return {"PackedBinary": 二进制,
                "Scale": Self._编码缩放(展平缩放 / 最大缩放), "MaxScale": 最大缩放,
                "Mean": Self._编码缩放(均值平坦 / 最大均值), "MaxMean": 最大均值,
                "Shape": (行数, 维度)}
    def Q1_K_M解码F32(Self, 二进制, 缩放编码, 最大缩放, 均值编码, 最大均值, 形状):
        行数, 维度 = 形状; 块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
        符号 = np.unpackbits(二进制.ravel()).reshape(行数, 维度).astype(np.float32) * 2.0 - 1.0
        缩放 = (Self._解码缩放(缩放编码, 行数 * 维度 // 块大小) * 最大缩放).reshape(行数, 维度//块大小, 1)
        均值 = (Self._解码缩放(均值编码, 维度) * 最大均值).reshape(1, 维度)
        return (符号.reshape(行数, 维度//块大小, 块大小) * 缩放).reshape(行数, 维度) + 均值
#====================================================================================================1Bit====================================================================================================#
#====================================================================================================GSQ_K量化====================================================================================================#
    def F32编码GSQ2_K(Self, 数组):
        量化值, 最小值存, 缩放值存, 最大最小, 最大缩放, 行数, 维度, 向量块 = Self._GSQ_K编码(数组, 3)
        return {"Vector": Self._打包2(量化值),
                "Min": Self._编码缩放(最小值存/最大最小), "MaxMin": 最大最小,
                "Scale": Self._编码缩放(缩放值存/最大缩放), "MaxScale": 最大缩放,
                "Shape": (行数, 维度), "VecBlock": 向量块}
    def GSQ2_K解码F32(Self, 压缩, 最小编码, 最大最小, 缩放编码, 最大缩放, 形状):
        return Self._GSQ_K解码(Self._解包2(压缩, math.prod(形状)),
                               最小编码, 最大最小, 缩放编码, 最大缩放, 形状, 3)
    def F32编码GSQ3_K(Self, 数组):
        量化值, 最小值存, 缩放值存, 最大最小, 最大缩放, 行数, 维度, 向量块 = Self._GSQ_K编码(数组, 7)
        return {"Vector": Self._打包3(量化值),
                "Min": Self._编码缩放(最小值存/最大最小), "MaxMin": 最大最小,
                "Scale": Self._编码缩放(缩放值存/最大缩放), "MaxScale": 最大缩放,
                "Shape": (行数, 维度), "VecBlock": 向量块}
    def GSQ3_K解码F32(Self, 压缩, 最小编码, 最大最小, 缩放编码, 最大缩放, 形状):
        return Self._GSQ_K解码(Self._解包3(压缩, math.prod(形状)),
                               最小编码, 最大最小, 缩放编码, 最大缩放, 形状, 7)
    def F32编码GSQ4_K(Self, 数组):
        量化值, 最小值存, 缩放值存, 最大最小, 最大缩放, 行数, 维度, 向量块 = Self._GSQ_K编码(数组, 15)
        return {"Vector": Self._打包4(量化值),
                "Min": Self._编码缩放(最小值存/最大最小), "MaxMin": 最大最小,
                "Scale": Self._编码缩放(缩放值存/最大缩放), "MaxScale": 最大缩放,
                "Shape": (行数, 维度), "VecBlock": 向量块}
    def GSQ4_K解码F32(Self, 压缩, 最小编码, 最大最小, 缩放编码, 最大缩放, 形状):
        return Self._GSQ_K解码(Self._解包4(压缩, math.prod(形状)),
                               最小编码, 最大最小, 缩放编码, 最大缩放, 形状, 15)
    def F32编码GSQ6_K(Self, 数组):
        量化值, 最小值存, 缩放值存, 最大最小, 最大缩放, 行数, 维度, 向量块 = Self._GSQ_K编码(数组, 63)
        return {"Vector": Self._打包6(量化值),
                "Min": Self._编码缩放(最小值存/最大最小), "MaxMin": 最大最小,
                "Scale": Self._编码缩放(缩放值存/最大缩放), "MaxScale": 最大缩放,
                "Shape": (行数, 维度), "VecBlock": 向量块}
    def GSQ6_K解码F32(Self, 压缩, 最小编码, 最大最小, 缩放编码, 最大缩放, 形状):
        return Self._GSQ_K解码(Self._解包6(压缩, math.prod(形状)),
                               最小编码, 最大最小, 缩放编码, 最大缩放, 形状, 63)
    def F32编码GSQ8_K(Self, 数组):
        量化值, 最小值存, 缩放值存, 最大最小, 最大缩放, 行数, 维度, 向量块 = Self._GSQ_K编码(数组, 255)
        return {"Vector": 量化值,
                "Min": Self._编码缩放(最小值存/最大最小), "MaxMin": 最大最小,
                "Scale": Self._编码缩放(缩放值存/最大缩放), "MaxScale": 最大缩放,
                "Shape": (行数, 维度), "VecBlock": 向量块}
    def GSQ8_K解码F32(Self, 数据, 最小编码, 最大最小, 缩放编码, 最大缩放, 形状):
        return Self._GSQ_K解码(数据[:math.prod(形状)],
                               最小编码, 最大最小, 缩放编码, 最大缩放, 形状, 255)
#====================================================================================================GSQ_K量化====================================================================================================#
#====================================================================================================浮点数====================================================================================================#
    def F32编码BF16(Self, 数组):
        u = np.asarray(数组, dtype=np.float32).view(np.uint32)
        压缩 = (u >> 16).astype(np.uint16); 压缩[(u & 0x7FFFFFFF) == 0] = 0
        return {"Vector": 压缩}
    def BF16解码F32(Self, 数据):
        数据 = np.asarray(数据, dtype=np.uint16); 零掩码 = 数据 == 0
        输出 = ((数据.astype(np.uint32) << 16)).view(np.float32); 输出[零掩码] = 0.0
        return 输出
    def F32编码FP8_E4M3(Self, 数组):
        u = np.asarray(数组, dtype=np.float32).view(np.uint32)
        fp8 = (((u>>31)&1).astype(np.uint8)<<7) | \
              (np.clip((u>>23)&0xFF, 120, 135).astype(np.uint8)-120)<<3 | \
              (((u&0x7FFFFF)>>20).astype(np.uint8)&7)
        fp8[np.asarray(数组)==0.0] = 0
        return {"Vector": fp8}
    def FP8_E4M3解码F32(Self, 数据):
        u = np.asarray(数据, dtype=np.uint8)
        f = ((u>>7).astype(np.uint32)<<31) | ((((u>>3)&0xF).astype(np.uint32)+120)<<23) | \
            ((u&7).astype(np.uint32)<<20); f[u==0] = 0
        return f.view(np.float32)
    def F32编码F16_E0M15(Self, 数组):
        return np.round(np.clip(数组, -1.0, 1.0-2**-15)*32768).astype(np.int16).view(np.uint16)
    def F16_E0M15解码F32(Self, 数据):
        return np.asarray(数据).view(np.int16).astype(np.float32) / 32768.0
    def F32编码F8_E0M7(Self, 数组):
        return np.round(np.clip(np.asarray(数组, dtype=np.float32), -1.0, 1.0 - 2**-7) * 128.0).astype(np.int8).view(np.uint8)
    def F8_E0M7解码F32(Self, 数据):
        return np.asarray(数据).view(np.int8).astype(np.float32) / 128.0
    def F32编码F12_E0M11(Self, 数组):
        数组 = np.asarray(数组, dtype=np.float32); 形状 = 数组.shape
        量化浮点 = np.round(np.clip(数组.ravel(), -1.0, 1.0 - 2**-11) * 2048.0)
        量化uint16 = (量化浮点.astype(np.int16).view(np.uint16) & np.uint16(0xFFF))
        return {"Vector": Self._打包F12(量化uint16), "Shape": 形状}
    def F12_E0M11解码F32(Self, 数据, 形状):
        总数 = math.prod(形状)
        无符号 = Self._解包F12(数据, 总数)
        有符号 = ((无符号.astype(np.int32) ^ 2048) - 2048).astype(np.float32)
        return (有符号 / 2048.0).reshape(形状)
#====================================================================================================浮点数====================================================================================================#
#====================================================================================================向量重排====================================================================================================#
    def 向量重排(Self, 数组, 文本列表):
        聚类块大小 = Self.Config.VEC_RERANKER_INDEX_RERANKER_BLOCK_SIZE
        向量数量, 向量维度 = 数组.shape
        if 向量数量 <= 聚类块大小:
            return 数组, 文本列表
        Self.日志("log.quantization.vector.reorder.start", info_level=0)
        数组_归一 = np.ascontiguousarray(数组, dtype=np.float32).copy()
        faiss.normalize_L2(数组_归一)
        向量索引 = Self.Module.构建索引(数组_归一, 向量重排模式=True)
        已访问掩码 = np.zeros(向量数量, dtype=bool)
        重排索引列表 = []
        粗搜数量 = 聚类块大小 * Self.Config.VEC_RERANKER_INDEX_FACTOR
        搜索指针 = 0
        已处理数量 = 0
        with Self.tqdm(total=向量数量, desc="tqdm.vectors.reorder") as 进度条:
            while 已处理数量 < 向量数量:
                while 搜索指针 < 向量数量 and 已访问掩码[搜索指针]: 
                    搜索指针 += 1
                if 搜索指针 >= 向量数量: 
                    break
                查询向量 = 数组_归一[搜索指针:搜索指针+1]
                _, 邻居索引 = 向量索引.search(查询向量, 粗搜数量)
                候选索引 = 邻居索引[0]
                有效掩码 = (候选索引 != -1) & (~已访问掩码[候选索引])
                有效索引 = 候选索引[有效掩码]
                if len(有效索引) == 0:
                    有效索引 = np.array([搜索指针])
                候选向量组 = 数组_归一[有效索引]
                相似度分数 = np.dot(候选向量组, 查询向量.T).flatten()
                截取数量 = min(聚类块大小, len(有效索引))
                if len(有效索引) > 截取数量:
                    顶部索引 = np.argpartition(相似度分数, -截取数量)[-截取数量:]
                    顶部索引 = 顶部索引[np.argsort(相似度分数[顶部索引])[::-1]]
                else:
                    顶部索引 = np.arange(len(有效索引))
                最终索引 = 有效索引[顶部索引]
                重排索引列表.extend(最终索引.tolist())
                已访问掩码[最终索引] = True
                已处理数量 += len(最终索引)
                进度条.update(len(最终索引))
        映射表 = np.array(重排索引列表, dtype=np.int64)
        重排后数组 = 数组[映射表]
        重排后文本 = [文本列表[i] for i in 映射表]
        Self.日志("log.quantization.vector.reorder.end", info_level=0)
        return 重排后数组, 重排后文本
#====================================================================================================向量重排====================================================================================================#
#====================================================================================================类接口====================================================================================================#
    def 解码向量(Self, 向量, 量化格式=None):
        CleanVRAM()
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization.reverse"):
            格式 = 量化格式 or Self.Config.VEC_QUANTIZATION
            输出 = Self.解码映射[格式](Self, 向量)
        CleanVRAM()
        return np.nan_to_num(输出, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    def 编码向量(Self, 向量, 量化格式=None):
        CleanVRAM()
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            格式 = 量化格式 or Self.Config.VEC_QUANTIZATION
            函数 = Self.编码映射[格式]
        CleanVRAM()
        return 函数(Self, 向量)
    def 叠加量化向量(Self, 旧数据, 新数据):
        if "ZeroPoint" in 旧数据:
            Self.模块.写入日志("log.quantization.superposition.error", info_level=2)
            return [新数据, False]
        结果 = {}
        for 键 in Self.拼接键:
            if 键 in 旧数据 and 键 in 新数据:
                结果[键] = np.concatenate((旧数据[键], 新数据[键]), axis=0)
        if "Shape" in 旧数据 and "Shape" in 新数据:
            结果["Shape"] = (旧数据["Shape"][0] + 新数据["Shape"][0],)
        return [结果, True]
#====================================================================================================类接口====================================================================================================#