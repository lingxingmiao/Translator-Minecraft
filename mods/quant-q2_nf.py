from TranslatorLib import Mods, np, math, CleanVRAM

参数 = Mods().模组配置("quant-q2_nf", {"BlockSize": 128, "Clip": 0.998})
BlockSize, Clip = int(参数["BlockSize"]), 参数["Clip"]

边界 = np.array([-0.9816, 0.0, 0.9816], dtype=np.float32)
中心 = np.array([-1.5121, -0.4528, 0.4528, 1.5121], dtype=np.float32)


def Q2_NF编码(Self, 数组):
    块大小 = BlockSize
    分块数据, 形状, 总数 = Self._分块(数组, 块大小)
    索引, 缩放 = Self._NF编码(分块数据, Clip * 100.0, 边界)
    del 分块数据
    CleanVRAM()
    最大缩放 = max(float(np.max(缩放)), 1e-8)
    return {"Vector": Self._打包2(索引.ravel()[:总数]),
            "Scale": Self._编码缩放(缩放 / 最大缩放), "MaxScale": 最大缩放,
            "Shape": 形状, "BlockSize": 块大小}


def Q2_NF解码(Self, 数据, 缩放编码, 最大缩放, 形状):
    块大小 = BlockSize
    总数 = math.prod(形状)
    填充总数 = ((总数 + 块大小 - 1) // 块大小) * 块大小
    return Self._NF解码(Self._解包2(数据, 填充总数), 缩放编码, 最大缩放, 总数, 形状, 块大小, 中心)


解码字典 = ["Vector", "Scale", "MaxScale", "Shape"]


@Mods().注册量化类型("Q2_NF", 2)
def 编码(Self, 数组):
    return Q2_NF编码(Self, 数组)


@Mods().注册反量化类型("Q2_NF", 2, 解码字典)
def 解码(Self, 数据, 缩放编码, 最大缩放, 形状):
    return Q2_NF解码(Self, 数据, 缩放编码, 最大缩放, 形状)
