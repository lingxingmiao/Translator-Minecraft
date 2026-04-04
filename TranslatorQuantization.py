from TranslatorLib import np, Path, math
from TranslatorConfig import RuntimeConfig
from TranslatorModule import Module
from TranslatorLocale import Locale
    
class Quantization:
    def __init__(Self, Config: dict = None):
        global tqdm
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Path(Self.Config.LOGS_FILE_PATH).mkdir(parents=True, exist_ok=True)
        Self.Module = Module(Config=Config)
        Self.Locale = Locale(Config=Config)
        tqdm = Self.Locale.Tqdm
    def F32编码Q8_K_X(Self, 输入数组):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            分位数 = Self.Config.VEC_QUANTILE * 100
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.percentile(块视图, 分位数, axis=1).astype(np.float32)
            块最小值 = np.percentile(块视图, 100 - 分位数, axis=1).astype(np.float32)
            范围 = 块最大值 - 块最小值
            范围 = np.where(范围 < 1e-8, 1.0, 范围)
            缩放因子 = (范围 / 255.0).astype(np.float32)
            零点 = np.round(-块最小值 / 缩放因子).astype(np.uint8)
            零点 = np.clip(零点, 0, 255)
            缩放因子广播 = 缩放因子[:, None]
            零点广播 = 零点[:, None]
            量化值 = np.clip(np.round(块视图 / 缩放因子广播 + 零点广播), 0, 255).astype(np.uint8)
            量化结果 = 量化值.flatten()[:len(扁平化)]
            缩放因子编码 = Self.F32编码F16_E0M15(缩放因子, desc="tqdm.scale.quantization")
        return {"Vector": 量化结果, "Scale": 缩放因子编码, "ZeroPoint": 零点, "Shape": 原始形状}

    def Q8_K_X解码F32(Self, 量化数据, 缩放因子编码, 零点数组, 原始形状):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization.reverse"):
            块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
            量化数据 = np.asarray(量化数据, dtype=np.uint8)
            填充长度 = (-len(量化数据)) % 块大小
            填充后量化 = np.pad(量化数据, (0, 填充长度), mode='constant')
            块数量 = int(len(填充后量化) // 块大小)
            块视图 = 填充后量化.reshape(块数量, 块大小)
            缩放因子 = Self.F16_E0M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子广播 = 缩放因子[:, np.newaxis]
            零点广播 = 零点数组[:, np.newaxis].astype(np.float32)
            反量化块 = (块视图.astype(np.float32) - 零点广播) * 缩放因子广播
            反量化结果 = 反量化块.flatten()[:len(量化数据)]
        return 反量化结果.reshape(原始形状)

    def F32编码Q6_K_X(Self, 输入数组):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            分位数 = Self.Config.VEC_QUANTILE * 100
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.percentile(块视图, 分位数, axis=1).astype(np.float32)
            块最小值 = np.percentile(块视图, 100 - 分位数, axis=1).astype(np.float32)
            范围 = 块最大值 - 块最小值
            范围 = np.where(范围 < 1e-8, 1.0, 范围)
            缩放因子 = (范围 / 63.0).astype(np.float32)
            零点 = np.round(-块最小值 / 缩放因子).astype(np.uint8)
            零点 = np.clip(零点, 0, 63)
            缩放因子广播 = 缩放因子[:, None]
            零点广播 = 零点[:, None]
            量化值 = np.clip(np.round(块视图 / 缩放因子广播 + 零点广播), 0, 63).astype(np.uint8)
            每组4个 = 块大小 // 4
            量化分组 = 量化值.reshape(块数量, 每组4个, 4)
            打包二维 = np.zeros((块数量, 每组4个, 3), dtype=np.uint8)
            打包二维[:, :, 0] = (量化分组[:, :, 0] << 2) | (量化分组[:, :, 1] >> 4)
            打包二维[:, :, 1] = ((量化分组[:, :, 1] & 0x0F) << 4) | (量化分组[:, :, 2] >> 2)
            打包二维[:, :, 2] = ((量化分组[:, :, 2] & 0x03) << 6) | 量化分组[:, :, 3]
            打包数据 = 打包二维.reshape(-1)
            实际元素数 = len(扁平化)
            实际4元组数 = (实际元素数 + 3) // 4
            实际打包长度 = 实际4元组数 * 3
            缩放因子编码 = Self.F32编码F16_E0M15(缩放因子, desc="tqdm.scale.quantization")
        return {"Vector": 打包数据[:实际打包长度], "Scale": 缩放因子编码, "ZeroPoint": 零点, "Shape": 原始形状}

    def Q6_K_X解码F32(Self, 打包数据, 缩放因子编码, 零点数组, 原始形状):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization.reverse"):
            块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
            总元素数 = math.prod(原始形状)
            填充后元素数 = ((总元素数 + 块大小 - 1) // 块大小) * 块大小
            块数量 = int(填充后元素数 // 块大小)
            每组4个 = int(块大小 // 4)
            完整4元组数 = 块数量 * 每组4个
            完整打包长度 = 完整4元组数 * 3
            完整打包 = np.zeros(完整打包长度, dtype=np.uint8)
            完整打包[:len(打包数据)] = 打包数据
            打包三维 = 完整打包.reshape(块数量, 每组4个, 3)
            解包分组 = np.zeros((块数量, 每组4个, 4), dtype=np.uint8)
            解包分组[:, :, 0] = 打包三维[:, :, 0] >> 2
            解包分组[:, :, 1] = ((打包三维[:, :, 0] & 0x03) << 4) | (打包三维[:, :, 1] >> 4)
            解包分组[:, :, 2] = ((打包三维[:, :, 1] & 0x0F) << 2) | (打包三维[:, :, 2] >> 6)
            解包分组[:, :, 3] = 打包三维[:, :, 2] & 0x3F
            解包块 = 解包分组.reshape(块数量, 块大小).astype(np.float32)
            缩放因子 = Self.F16_E0M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子广播 = 缩放因子[:, np.newaxis]
            零点广播 = 零点数组[:, np.newaxis].astype(np.float32)
            反量化块 = (解包块 - 零点广播) * 缩放因子广播
            反量化扁平 = 反量化块.flatten()[:总元素数]
        return 反量化扁平.reshape(原始形状)

    def F32编码Q4_K_X(Self, 输入数组):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            分位数 = Self.Config.VEC_QUANTILE * 100
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.percentile(块视图, 分位数, axis=1).astype(np.float32)
            块最小值 = np.percentile(块视图, 100 - 分位数, axis=1).astype(np.float32)
            范围 = 块最大值 - 块最小值
            范围 = np.where(范围 < 1e-8, 1.0, 范围)
            缩放因子 = (范围 / 15.0).astype(np.float32)
            零点 = np.round(-块最小值 / 缩放因子).astype(np.uint8)
            零点 = np.clip(零点, 0, 15)
            缩放因子广播 = 缩放因子[:, None]
            零点广播 = 零点[:, None]
            量化值 = np.clip(np.round(块视图 / 缩放因子广播 + 零点广播), 0, 15).astype(np.uint8)
            高四位 = 量化值[:, 0::2] << 4
            低四位 = 量化值[:, 1::2]
            打包二维 = 高四位 | 低四位
            打包长度 = (len(扁平化) + 1) // 2
            打包数据 = 打包二维.flatten()[:打包长度]
            缩放因子编码 = Self.F32编码F16_E0M15(缩放因子, desc="tqdm.scale.quantization")
        return {"Vector": 打包数据, "Scale": 缩放因子编码, "ZeroPoint": 零点, "Shape": 原始形状}

    def Q4_K_X解码F32(Self, 打包数据, 缩放因子编码, 零点数组, 原始形状):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization.reverse"):
            块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
            总元素数 = math.prod(原始形状)
            填充后元素数 = ((总元素数 + 块大小 - 1) // 块大小) * 块大小
            块数量 = int(填充后元素数 // 块大小)
            完整打包长度 = (填充后元素数 + 1) // 2
            完整打包 = np.zeros(完整打包长度, dtype=np.uint8)
            完整打包[:len(打包数据)] = 打包数据
            高四位 = (完整打包[:, None] >> 4) & 0xF
            低四位 = 完整打包[:, None] & 0xF
            解包数据 = np.concatenate([高四位, 低四位], axis=1).flatten()[:填充后元素数]
            解包块 = 解包数据.reshape(块数量, 块大小)
            缩放因子 = Self.F16_E0M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子广播 = 缩放因子[:, np.newaxis]
            零点广播 = 零点数组[:, np.newaxis].astype(np.float32)
            反量化块 = (解包块.astype(np.float32) - 零点广播) * 缩放因子广播
            反量化扁平 = 反量化块.flatten()[:总元素数]
        return 反量化扁平.reshape(原始形状)

    def F32编码Q3_K_X(Self, 输入数组):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            分位数 = Self.Config.VEC_QUANTILE * 100
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.percentile(块视图, 分位数, axis=1).astype(np.float32)
            块最小值 = np.percentile(块视图, 100 - 分位数, axis=1).astype(np.float32)
            范围 = 块最大值 - 块最小值
            范围 = np.where(范围 < 1e-8, 1.0, 范围)
            缩放因子 = (范围 / 7.0).astype(np.float32)
            零点 = np.round(-块最小值 / 缩放因子).astype(np.uint8)
            零点 = np.clip(零点, 0, 7)
            缩放因子广播 = 缩放因子[:, None]
            零点广播 = 零点[:, None]
            量化值 = np.clip(np.round(块视图 / 缩放因子广播 + 零点广播), 0, 7).astype(np.uint8)
            每组8个 = 块大小 // 8
            量化分组 = 量化值.reshape(块数量, 每组8个, 8)
            打包二维 = np.zeros((块数量, 每组8个, 3), dtype=np.uint8)
            打包二维[:, :, 0] = (量化分组[:, :, 0] << 5) | (量化分组[:, :, 1] << 2) | (量化分组[:, :, 2] >> 1)
            打包二维[:, :, 1] = ((量化分组[:, :, 2] & 0x01) << 7) | (量化分组[:, :, 3] << 4) | (量化分组[:, :, 4] << 1) | (量化分组[:, :, 5] >> 2)
            打包二维[:, :, 2] = ((量化分组[:, :, 5] & 0x03) << 6) | (量化分组[:, :, 6] << 3) | 量化分组[:, :, 7]
            打包数据 = 打包二维.reshape(-1)
            实际元素数 = len(扁平化)
            实际8元组数 = (实际元素数 + 7) // 8
            实际打包长度 = 实际8元组数 * 3
            缩放因子编码 = Self.F32编码F16_E0M15(缩放因子, desc="tqdm.scale.quantization")
        return {"Vector": 打包数据[:实际打包长度], "Scale": 缩放因子编码, "ZeroPoint": 零点, "Shape": 原始形状}

    def Q3_K_X解码F32(Self, 打包数据, 缩放因子编码, 零点数组, 原始形状):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization.reverse"):
            块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
            总元素数 = math.prod(原始形状)
            填充后元素数 = ((总元素数 + 块大小 - 1) // 块大小) * 块大小
            块数量 = int(填充后元素数 // 块大小)
            每组8个 = int(块大小 // 8)
            完整8元组数 = 块数量 * 每组8个
            完整打包长度 = 完整8元组数 * 3
            完整打包 = np.zeros(完整打包长度, dtype=np.uint8)
            完整打包[:len(打包数据)] = 打包数据
            打包三维 = 完整打包.reshape(块数量, 每组8个, 3)
            解包分组 = np.zeros((块数量, 每组8个, 8), dtype=np.uint8)
            解包分组[:, :, 0] = (打包三维[:, :, 0] >> 5) & 0x07
            解包分组[:, :, 1] = (打包三维[:, :, 0] >> 2) & 0x07
            解包分组[:, :, 2] = ((打包三维[:, :, 0] & 0x03) << 1) | (打包三维[:, :, 1] >> 7)
            解包分组[:, :, 3] = (打包三维[:, :, 1] >> 4) & 0x07
            解包分组[:, :, 4] = (打包三维[:, :, 1] >> 1) & 0x07
            解包分组[:, :, 5] = ((打包三维[:, :, 1] & 0x01) << 2) | (打包三维[:, :, 2] >> 6)
            解包分组[:, :, 6] = (打包三维[:, :, 2] >> 3) & 0x07
            解包分组[:, :, 7] = 打包三维[:, :, 2] & 0x07
            解包块 = 解包分组.reshape(块数量, 块大小).astype(np.float32)
            缩放因子 = Self.F16_E0M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子广播 = 缩放因子[:, np.newaxis]
            零点广播 = 零点数组[:, np.newaxis].astype(np.float32)
            反量化块 = (解包块 - 零点广播) * 缩放因子广播
            反量化扁平 = 反量化块.flatten()[:总元素数]
        return 反量化扁平.reshape(原始形状)

    def F32编码Q2_K_X(Self, 输入数组):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            分位数 = Self.Config.VEC_QUANTILE * 100
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.percentile(块视图, 分位数, axis=1).astype(np.float32)
            块最小值 = np.percentile(块视图, 100 - 分位数, axis=1).astype(np.float32)
            范围 = 块最大值 - 块最小值
            范围 = np.where(范围 < 1e-8, 1.0, 范围)
            缩放因子 = (范围 / 3.0).astype(np.float32)
            零点 = np.round(-块最小值 / 缩放因子).astype(np.uint8)
            零点 = np.clip(零点, 0, 3)
            缩放因子广播 = 缩放因子[:, None]
            零点广播 = 零点[:, None]
            量化值 = np.clip(np.round(块视图 / 缩放因子广播 + 零点广播), 0, 3).astype(np.uint8)
            每组数量 = 块大小 // 4
            量化分组 = 量化值.reshape(块数量, 每组数量, 4)
            打包二维 = (量化分组[:, :, 0] | 
                    (量化分组[:, :, 1] << 2) | 
                    (量化分组[:, :, 2] << 4) | 
                    (量化分组[:, :, 3] << 6)).astype(np.uint8)
            打包数据 = 打包二维.flatten()
            实际元素数 = len(扁平化)
            实际打包长度 = (实际元素数 + 3) // 4
            缩放因子编码 = Self.F32编码F16_E0M15(缩放因子, desc="tqdm.scale.quantization")
        return {"Vector": 打包数据[:实际打包长度], "Scale": 缩放因子编码, "ZeroPoint": 零点, "Shape": 原始形状}

    def Q2_K_X解码F32(Self, 打包数据, 缩放因子编码, 零点数组, 原始形状):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization.reverse"):
            块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
            总元素数 = math.prod(原始形状)
            填充后元素数 = ((总元素数 + 块大小 - 1) // 块大小) * 块大小
            块数量 = int(填充后元素数 // 块大小)
            每组数量 = int(块大小 // 4)
            完整打包长度 = 块数量 * 每组数量
            完整打包 = np.zeros(完整打包长度, dtype=np.uint8)
            完整打包[:len(打包数据)] = 打包数据
            完整打包重塑 = 完整打包.reshape(块数量, 每组数量, 1)
            解包分组 = np.concatenate([
                (完整打包重塑 >> 0) & 0x03,
                (完整打包重塑 >> 2) & 0x03,
                (完整打包重塑 >> 4) & 0x03,
                (完整打包重塑 >> 6) & 0x03,
            ], axis=2).reshape(块数量, 块大小).astype(np.float32)
            缩放因子 = Self.F16_E0M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子广播 = 缩放因子[:, np.newaxis]
            零点广播 = 零点数组[:, np.newaxis].astype(np.float32)
            反量化块 = (解包分组 - 零点广播) * 缩放因子广播
            反量化扁平 = 反量化块.flatten()[:总元素数]
        return 反量化扁平.reshape(原始形状)
    def F32编码BF16(Self, 输入数组, desc="tqdm.vectors.quantization"): #Qwen3-Max编写
        for _ in tqdm(range(1), desc=desc):
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            符号位 = np.signbit(输入数组).astype(np.uint16) << 15
            绝对值数组 = np.abs(输入数组)
            是否为零 = 绝对值数组 < 1e-45
            是否为无穷大 = np.isinf(绝对值数组)
            是否为NaN = np.isnan(输入数组)
            F32数组 = 输入数组.view(np.uint32)
            打包数组 = (F32数组 >> 16).astype(np.uint16)
            打包数组[是否为零] = 0 
            打包数组[是否为无穷大] = 0x7F80 | (符号位[是否为无穷大] >> 15) << 15
            打包数组[是否为NaN] = 0x7FC0 | (符号位[是否为NaN] >> 15) << 15
        return {"Vector": 打包数组.astype(np.uint16)}
    def BF16解码F32(Self, 编码数组, desc="tqdm.vectors.quantization.reverse"): #Qwen3-Max编写
        for _ in tqdm(range(1), desc=desc):
            编码数组 = np.asarray(编码数组, dtype=np.uint16)
            是否为零 = 编码数组 == 0
            是否为无穷大 = (编码数组 & 0x7F80) == 0x7F80
            是否为NaN = (编码数组 & 0x7FC0) == 0x7FC0
            F32编码 = (编码数组.astype(np.uint32) << 16)
            解码值 = F32编码.view(np.float32)
            解码值[是否为零] = 0.0
            解码值[是否为无穷大] = np.inf * np.where((编码数组[是否为无穷大] >> 15) & 0x1, -1.0, 1.0)
            解码值[是否为NaN] = np.nan
        return 解码值.astype(np.float32)
    def F32编码FP8_E4M3(Self, 输入数组, desc="tqdm.vectors.quantization"):
        for _ in tqdm(range(1), desc=desc):
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            F32数组 = 输入数组.view(np.uint32)
            符号位 = (F32数组 >> 31) & 0x1
            指数位 = (F32数组 >> 23) & 0xFF
            尾数位 = F32数组 & 0x7FFFFF
            是否为零 = (输入数组 == 0.0)
            是否为无穷大 = np.isinf(输入数组)
            是否为NaN = np.isnan(输入数组)
            指数偏移 = 127 - 7
            FP8指数 = np.clip(指数位 - 指数偏移, 0, 15).astype(np.uint8)
            FP8尾数 = (尾数位 >> 20).astype(np.uint8) & 0x7
            FP8编码 = (符号位 << 7) | (FP8指数 << 3) | FP8尾数
            FP8编码[是否为零] = 0
            FP8编码[是否为无穷大] = (符号位[是否为无穷大] << 7) | 0x78
            FP8编码[是否为NaN] = (符号位[是否为NaN] << 7) | 0x7C
        return {"Vector": FP8编码.astype(np.uint8)}
    def FP8_E4M3解码F32(Self, 编码数组, desc="tqdm.vectors.quantization.reverse"):
        for _ in tqdm(range(1), desc=desc):
            编码数组 = np.asarray(编码数组, dtype=np.uint8)
            符号位 = (编码数组 >> 7) & 0x1
            指数位 = (编码数组 >> 3) & 0xF
            尾数位 = 编码数组 & 0x7
            是否为零 = (编码数组 == 0)
            是否为无穷大 = (指数位 == 15) & (尾数位 == 0)
            是否为NaN = (指数位 == 15) & (尾数位 != 0)
            指数偏移 = 127 - 7
            F32指数 = (指数位 + 指数偏移).astype(np.uint32)
            F32尾数 = (尾数位.astype(np.uint32) << 20)
            F32编码 = (符号位.astype(np.uint32) << 31) | (F32指数 << 23) | F32尾数
            F32编码[是否为零] = 0
            F32编码[是否为无穷大] = (符号位[是否为无穷大].astype(np.uint32) << 31) | (0xFF << 23)
            F32编码[是否为NaN] = (符号位[是否为NaN].astype(np.uint32) << 31) | (0xFF << 23) | (1 << 22)
            解码值 = F32编码.view(np.float32)
        return 解码值.astype(np.float32)
    def F32编码F16_E0M15(Self, 浮点数组, desc="tqdm.vectors.quantization"): #Qwen3-Max编写
        for _ in tqdm(range(1), desc=desc):
            浮点数组 = np.clip(浮点数组, -1.0, 1.0 - 2**-15)
            整型数值 = np.round(浮点数组 * 32768.0).astype(np.int16)
        return 整型数值.view(np.uint16)
    def F16_E0M15解码F32(Self, 编码数组, desc="tqdm.vectors.quantization.reverse"): #Qwen3-Max编写
        for _ in tqdm(range(1), desc=desc):
            整型数值 = 编码数组.view(np.int16)
        return 整型数值.astype(np.float32) / 32768.0
    def 解码向量(Self, 向量文件, 量化 = None):
        量化格式 = 量化 if 量化 else Self.Config.VEC_QUANTIZATION 
        if 量化格式 == "Q8_K_X":
            向量数组 = Self.Q8_K_X解码F32(向量文件["Vector"], 向量文件["Scale"], 向量文件["ZeroPoint"], 向量文件["Shape"])
        elif 量化格式 == "Q6_K_X":
            向量数组 = Self.Q6_K_X解码F32(向量文件["Vector"], 向量文件["Scale"], 向量文件["ZeroPoint"], 向量文件["Shape"])
        elif 量化格式 == "Q4_K_X":
            向量数组 = Self.Q4_K_X解码F32(向量文件["Vector"], 向量文件["Scale"], 向量文件["ZeroPoint"], 向量文件["Shape"])
        elif 量化格式 == "Q3_K_X":
            向量数组 = Self.Q3_K_X解码F32(向量文件["Vector"], 向量文件["Scale"], 向量文件["ZeroPoint"], 向量文件["Shape"])
        elif 量化格式 == "Q2_K_X":
            向量数组 = Self.Q2_K_X解码F32(向量文件["Vector"], 向量文件["Scale"], 向量文件["ZeroPoint"], 向量文件["Shape"])
        elif 量化格式 == "Float8_E4M3":
            向量数组 = Self.FP8_E4M3解码F32(向量文件["Vector"])
        elif 量化格式 == "Float16":
            向量数组 = 向量文件["Vector"]
        elif 量化格式 == "BFloat16":
            向量数组 = Self.BF16解码F32(向量文件["Vector"])
        elif 量化格式 == "Float16_E0M15":
            向量数组 = Self.F16_E0M15解码F32(向量文件["Vector"])
        elif 量化格式 == "Float32":
            向量数组 = 向量文件["Vector"]
        向量数组 = np.nan_to_num(向量数组, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return 向量数组
    def 编码向量(Self, 向量数组, 量化 = None):
        量化格式 = 量化 if 量化 else Self.Config.VEC_QUANTIZATION 
        if 量化格式 == "Q8_K_X":
            向量文件 = Self.F32编码Q8_K_X(向量数组)
        elif 量化格式 == "Q6_K_X":
            向量文件 = Self.F32编码Q6_K_X(向量数组)
        elif 量化格式 == "Q4_K_X":
            向量文件 = Self.F32编码Q4_K_X(向量数组)
        elif 量化格式 == "Q3_K_X":
            向量文件 = Self.F32编码Q3_K_X(向量数组)
        elif 量化格式 == "Q2_K_X":
            向量文件 = Self.F32编码Q2_K_X(向量数组)
        elif 量化格式 == "Float8_E4M3":
            向量文件 = Self.F32编码FP8_E4M3(向量数组)
        elif 量化格式 == "Float16":
            向量文件 = {"Vector": 向量数组.astype(np.float16)}
        elif 量化格式 == "BFloat16":
            向量文件 = Self.F32编码BF16(向量数组)
        elif 量化格式 == "Float16_E0M15":
            向量文件 = {"Vector": Self.F32编码F16_E0M15(向量数组)}
        elif 量化格式 == "Float32":
            向量文件 = {"Vector": 向量数组.astype(np.float32)}
        return 向量文件
    def 叠加量化向量(Self, 旧字典, 新字典):
        if "ZeroPoint" in [index for index in 旧字典]:
            Self.Module.写入日志("log.quantization.superposition.error", info_level=2)
            结果字典 = 新字典
        else:
            结果字典 = {}
            拼接键 = ["Vector"]
            形状键 = "Shape"
            for key in 拼接键:
                if key in 旧字典 and key in 新字典:
                    结果字典[key] = np.concatenate((旧字典[key], 新字典[key]), axis=0)
            if 形状键 in 旧字典 and 形状键 in 新字典:
                结果字典[形状键] = (旧字典[形状键][0] + 新字典[形状键][0],)
        return 结果字典