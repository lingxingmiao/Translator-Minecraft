from TranslatorLib import Mods, np, math, CleanVRAM

档位表 = [
    ("Int16_Max", 32767.0, 16, ["Vector", "MaxScale"]),
    ("Int12_Max",  2047.0, 12, ["Vector", "MaxScale", "Shape"]),
    ("Int8_Max",    127.0,  8, ["Vector", "MaxScale"]),
    ("Int6_Max",     31.0,  6, ["Vector", "MaxScale", "Shape"]),
    ("Int5_Max",     15.0,  5, ["Vector", "MaxScale", "Shape"]),
    ("Int4_Max",      7.0,  4, ["Vector", "MaxScale", "Shape"]),
    ("Int3_Max",      3.0,  3, ["Vector", "MaxScale", "Shape"]),
]

打包方法 = {3: "_打包3", 4: "_打包4", 5: "_打包5", 6: "_打包6"}
解包方法 = {3: "_解包3", 4: "_解包4", 5: "_解包5", 6: "_解包6"}

def 注册档位(名称, 幅度上限, 位宽, 解码键):
    @Mods().注册量化类型(名称, 位宽)
    def 编码(Self, 数组):
        数组 = np.asarray(数组, dtype=np.float32)
        最大绝对 = max(float(np.max(np.abs(数组))), 1e-8)
        缩放 = 最大绝对 / 幅度上限
        整数 = np.round(数组 / 缩放)
        if 位宽 == 16:
            结果 = {"Vector": 整数.astype(np.int16).view(np.uint16)}
        elif 位宽 == 12:
            结果 = {"Vector": Self._打包F12((整数.astype(np.int16).view(np.uint16) & 0xFFF).ravel()),
                    "Shape": 数组.shape}
        elif 位宽 == 8:
            结果 = {"Vector": 整数.astype(np.int8).view(np.uint8)}
        else:
            补码 = (整数.astype(np.int32) & ((1 << 位宽) - 1)).astype(np.uint8).ravel()
            结果 = {"Vector": getattr(Self, 打包方法[位宽])(补码), "Shape": 数组.shape}
        结果["MaxScale"] = np.array([缩放], dtype=np.float32)
        CleanVRAM()
        return 结果

    @Mods().注册反量化类型(名称, 位宽, 解码键)
    def 解码(Self, 向量, 最大缩放, *形状):
        缩放 = float(np.asarray(最大缩放, dtype=np.float32).ravel()[0])
        if 位宽 == 12:
            整数 = Self._解包F12(向量, math.prod(形状[0]))
            return (((整数.astype(np.int32) ^ 2048) - 2048).astype(np.float32) * 缩放).reshape(形状[0])
        if 位宽 == 16:
            return np.asarray(向量, dtype=np.uint16).view(np.int16).astype(np.float32) * 缩放
        if 位宽 == 8:
            return np.asarray(向量, dtype=np.uint8).view(np.int8).astype(np.float32) * 缩放
        补码 = getattr(Self, 解包方法[位宽])(向量, math.prod(形状[0]))
        半 = 1 << (位宽 - 1)
        return (((补码.astype(np.int32) ^ 半) - 半).astype(np.float32) * 缩放).reshape(形状[0])

    return 编码, 解码


for 名称, 幅度上限, 位宽, 解码键 in 档位表:
    注册档位(名称, 幅度上限, 位宽, 解码键)
