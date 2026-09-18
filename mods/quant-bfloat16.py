from TranslatorLib import Mods, np

@Mods().注册量化类型("BFloat16", 16)
def F32编码BFloat16(Self, 数组):
    u = np.asarray(数组, dtype=np.float32).view(np.uint32)
    压缩 = (u >> 16).astype(np.uint16)
    压缩[(u & 0x7FFFFFFF) == 0] = 0
    return {"Vector": 压缩}

@Mods().注册反量化类型("BFloat16", 16, ["Vector"])
def BFloat16解码F32(Self, 数据):
    数据 = np.asarray(数据, dtype=np.uint16)
    零掩码 = 数据 == 0
    输出 = (数据.astype(np.uint32) << 16).view(np.float32)
    输出[零掩码] = 0.0
    return 输出
