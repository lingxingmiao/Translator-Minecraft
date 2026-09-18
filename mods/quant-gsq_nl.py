from TranslatorLib import Mods, np, CleanVRAM, TranslatorQuantization

参数 = Mods().模组配置("quant-gsq_nl", {"BlockSize": 128, "SplLM": 0.05})
BlockSize, SplLM = int(参数["BlockSize"]), float(参数["SplLM"])

def GSQ_NL编码(Self, 数组, 电平数):
    分组行数 = BlockSize
    数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32)
    行数, 维度 = 数组.shape
    填充行 = (-行数) % 分组行数
    总行数 = 行数 + 填充行
    组数 = 总行数 // 分组行数
    最小值 = np.empty((组数, 维度), dtype=np.float32)
    最大值 = np.empty((组数, 维度), dtype=np.float32)
    每片组数 = max(1, min(组数, 2_000_000 // max(1, 分组行数 * 维度)))
    for 起 in range(0, 组数, 每片组数):
        止 = min(起 + 每片组数, 组数)
        行起, 行止 = 起 * 分组行数, 止 * 分组行数
        片 = np.zeros((行止 - 行起, 维度), dtype=np.float32)
        真实 = min(行止, 行数) - min(行起, 行数)
        if 真实 > 0:
            片[:真实] = 数组[min(行起, 行数):min(行止, 行数)]
        三维 = 片.reshape(止 - 起, 分组行数, 维度)
        最小值[起:止] = 三维.min(1)
        最大值[起:止] = 三维.max(1)
        del 片, 三维
    CleanVRAM()
    缩放值 = np.maximum(最大值 - 最小值, 1e-8)
    del 最大值
    码本 = Self.LloydMax(Self.采样(采样归一化(Self, 数组, 最小值, 缩放值, 分组行数, 行数), SplLM), 电平数)
    边界 = ((码本[:-1] + 码本[1:]) / 2.0).astype(np.float32)
    索引片列表 = []
    for 起 in range(0, 组数, 每片组数):
        止 = min(起 + 每片组数, 组数)
        行起, 行止 = 起 * 分组行数, 止 * 分组行数
        片 = np.zeros((行止 - 行起, 维度), dtype=np.float32)
        真实起, 真实止 = min(行起, 行数), min(行止, 行数)
        if 真实止 > 真实起:
            片[:真实止 - 真实起] = 数组[真实起:真实止]
        三维 = 片.reshape(止 - 起, 分组行数, 维度)
        三维 -= 最小值[起:止][:, None, :]
        三维 /= 缩放值[起:止][:, None, :]
        np.clip(三维, 0, 1, out=三维)
        索引片列表.append(Self.查表量化(三维, 边界).reshape(行止 - 行起, 维度)[:真实止 - 真实起])
        del 片, 三维
        CleanVRAM()
    量化值 = np.concatenate(索引片列表, axis=0).ravel() if len(索引片列表) > 1 else 索引片列表[0].ravel()
    del 索引片列表
    CleanVRAM()

    最大最小 = max(float(np.max(np.abs(最小值))), 1e-8)
    最大缩放 = max(float(np.max(缩放值)), 1e-8)
    return (量化值, Self._编码缩放(最小值 / 最大最小), 最大最小,
            Self._编码缩放(缩放值 / 最大缩放), 最大缩放, 码本, 行数, 维度, 分组行数)

def 采样归一化(Self, 数组, 最小值, 缩放值, 分组行数, 行数):
    取行 = min(行数, 分组行数 * 256)
    if 取行 <= 0:
        return np.zeros(1, dtype=np.float32)
    片 = np.ascontiguousarray(数组[:取行], dtype=np.float32)
    填充行 = (-取行) % 分组行数
    总行数 = 取行 + 填充行
    if 填充行:
        缓冲 = np.zeros((总行数, 片.shape[1]), dtype=np.float32)
        缓冲[:取行] = 片
    else:
        缓冲 = 片
    组数 = 总行数 // 分组行数
    三维 = 缓冲.reshape(组数, 分组行数, 片.shape[1])
    三维 = (三维 - 最小值[:组数][:, None, :]) / 缩放值[:组数][:, None, :]
    np.clip(三维, 0, 1, out=三维)
    return 三维.ravel()

def GSQ_NL解码(Self, 量化值, 最小编码, 最大最小, 缩放编码, 最大缩放, 码本, 形状):
    分组行数 = BlockSize
    行数, 维度 = 形状
    组数 = (行数 + 分组行数 - 1) // 分组行数
    数量 = 组数 * 维度
    索引 = np.asarray(量化值, dtype=np.uint8).ravel()
    需要 = 行数 * 维度
    if 索引.size < 需要:
        缓冲 = np.zeros(需要, dtype=np.uint8)
        缓冲[:索引.size] = 索引
        索引 = 缓冲
    索引 = 索引[:需要].reshape(行数, 维度)
    最小值 = (Self._解码缩放(最小编码, 数量) * 最大最小).astype(np.float32).reshape(组数, 维度)
    缩放值 = (Self._解码缩放(缩放编码, 数量) * 最大缩放).astype(np.float32).reshape(组数, 维度)
    组索引 = np.arange(行数) // 分组行数
    重建 = np.asarray(码本, dtype=np.float32)[索引]
    return 重建 * 缩放值[组索引] + 最小值[组索引]

def 打包(Self, 索引, 位宽):
    if 位宽 == 1:
        return np.packbits(np.ascontiguousarray(np.asarray(索引, dtype=np.uint8).ravel()))
    if 位宽 == 3:
        return TranslatorQuantization.加速三值打包(np.asarray(索引, dtype=np.uint8).ravel())
    if 位宽 == 2:
        return Self._打包2(索引)
    if 位宽 == 4:
        return Self._打包3(索引)
    if 位宽 == 5:
        return Self._打包4(索引)
    if 位宽 == 6:
        return Self._打包5(索引)
    return Self._打包6(索引)


def 解包(Self, 压缩, 数量, 位宽):
    if 位宽 == 1:
        return np.unpackbits(np.asarray(压缩, dtype=np.uint8).ravel())[:数量]
    if 位宽 == 3:
        return TranslatorQuantization.加速三值解包(np.asarray(压缩).astype(np.uint8), 数量)
    if 位宽 == 2:
        return Self._解包2(压缩, 数量)
    if 位宽 == 4:
        return Self._解包3(压缩, 数量)
    if 位宽 == 5:
        return Self._解包4(压缩, 数量)
    if 位宽 == 6:
        return Self._解包5(压缩, 数量)
    return Self._解包6(压缩, 数量)

解码字典 = ["Vector", "Min", "MaxMin", "Scale", "MaxScale", "Codebook", "Shape"]

档位表 = [
    ("GSTQ1_NL", 3, 3, 1.585),
    ("GSQ1_NL",  2, 1, 1),
    ("GSQ2_NL",  4, 2, 2),
    ("GSQ3_NL",  8, 4, 3),
    ("GSQ4_NL", 16, 5, 4),
    ("GSQ5_NL", 32, 6, 5),
    ("GSQ6_NL", 64, 7, 6),
]


def 注册档位(名称, 电平数, 打包位宽, 声明位数):
    @Mods().注册量化类型(名称, 声明位数)
    def 编码(Self, 数组):
        量化值, Min, MaxMin, Scale, MaxScale, 码本, 行数, 维度, 分组行数 = GSQ_NL编码(Self, 数组, 电平数)
        结果 = {"Vector": 打包(Self, 量化值, 打包位宽), "Min": Min, "MaxMin": MaxMin,
                "Scale": Scale, "MaxScale": MaxScale, "Codebook": np.asarray(码本, dtype=np.float32),
                "Shape": (行数, 维度)}
        CleanVRAM()
        return 结果

    @Mods().注册反量化类型(名称, 声明位数, 解码字典)
    def 解码(Self, 压缩, 最小编码, 最大最小, 缩放编码, 最大缩放, 码本, 形状):
        行数, 维度 = 形状
        索引 = 解包(Self, 压缩, 行数 * 维度, 打包位宽)
        return GSQ_NL解码(Self, 索引, 最小编码, 最大最小, 缩放编码, 最大缩放, 码本, 形状)

    return 编码, 解码


for 名称, 电平数, 打包位宽, 声明位数 in 档位表:
    注册档位(名称, 电平数, 打包位宽, 声明位数)
