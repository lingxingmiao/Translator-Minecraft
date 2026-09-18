from TranslatorLib import Mods, np, math

# 自定义浮点格式
#   Float32      : 原生 float32 (无损, 也可作缩放类型)
#   Float16      : 原生 float16 (也可作缩放类型)
#   Float16_E0M15: 1 符号 + 15 小数位 (int16 定点, 分辨率 2^-15, 范围 [-1,1))
#   Float8_E0M7  : 1 符号 + 7  小数位 (int8  定点, 分辨率 2^-7,  范围 [-1,1))
#   Float12_E0M11: 12 位打包的定点   (分辨率 2^-11, 范围 [-1,1))
#   Float8_E4M3  : 1 符号 + 4 指数 + 3 尾数 (指数偏置 120)

@Mods().注册量化类型("Float32", 32)
def F32编码Float32(Self, 数组):
    return {"Vector": np.asarray(数组, dtype=np.float32)}

@Mods().注册反量化类型("Float32", 32, ["Vector"])
def Float32解码F32(Self, 向量):
    return np.asarray(向量, dtype=np.float32)


@Mods().注册量化类型("Float16", 16)
def F32编码Float16(Self, 数组):
    return {"Vector": np.asarray(数组, dtype=np.float16)}

@Mods().注册反量化类型("Float16", 16, ["Vector"])
def Float16解码F32(Self, 向量):
    return np.asarray(向量, dtype=np.float16).astype(np.float32)


@Mods().注册量化类型("Float16_E0M15", 16)
def F32编码F16_E0M15(Self, 数组):
    return {"Vector": np.round(np.clip(数组, -1.0, 1.0 - 2**-15) * 32768).astype(np.int16).view(np.uint16)}

@Mods().注册反量化类型("Float16_E0M15", 16, ["Vector"])
def F16_E0M15解码F32(Self, 数据):
    return np.asarray(数据).view(np.int16).astype(np.float32) / 32768.0


@Mods().注册量化类型("Float8_E0M7", 8)
def F32编码F8_E0M7(Self, 数组):
    return {"Vector": np.round(np.clip(np.asarray(数组, dtype=np.float32), -1.0, 1.0 - 2**-7) * 128.0).astype(np.int8).view(np.uint8)}

@Mods().注册反量化类型("Float8_E0M7", 8, ["Vector"])
def F8_E0M7解码F32(Self, 数据):
    return np.asarray(数据).view(np.int8).astype(np.float32) / 128.0


@Mods().注册量化类型("Float12_E0M11", 12)
def F32编码F12_E0M11(Self, 数组):
    数组 = np.asarray(数组, dtype=np.float32)
    量化浮点 = np.round(np.clip(数组.ravel(), -1.0, 1.0 - 2**-11) * 2048.0)
    量化uint16 = 量化浮点.astype(np.int16).view(np.uint16) & np.uint16(0xFFF)
    return {"Vector": Self._打包F12(量化uint16), "Shape": 数组.shape}

@Mods().注册反量化类型("Float12_E0M11", 12, ["Vector", "Shape"])
def F12_E0M11解码F32(Self, 数据, 形状):
    无符号 = Self._解包F12(数据, math.prod(形状))
    有符号 = ((无符号.astype(np.int32) ^ 2048) - 2048).astype(np.float32)
    return (有符号 / 2048.0).reshape(形状)


@Mods().注册量化类型("Float8_E4M3", 8)
def F32编码FP8_E4M3(Self, 数组):
    u = np.asarray(数组, dtype=np.float32).view(np.uint32)
    fp8 = (((u >> 31) & 1).astype(np.uint8) << 7) \
        | ((np.clip((u >> 23) & 0xFF, 120, 135).astype(np.uint8) - 120) << 3) \
        | (((u & 0x7FFFFF) >> 20).astype(np.uint8) & 7)
    fp8[np.asarray(数组) == 0.0] = 0
    return {"Vector": fp8}

@Mods().注册反量化类型("Float8_E4M3", 8, ["Vector"])
def FP8_E4M3解码F32(Self, 数据):
    u = np.asarray(数据, dtype=np.uint8)
    f = ((u >> 7).astype(np.uint32) << 31) \
        | ((((u >> 3) & 0xF).astype(np.uint32) + 120) << 23) \
        | ((u & 7).astype(np.uint32) << 20)
    f[u == 0] = 0
    return f.view(np.float32)
