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
    def F32编码Q8_K_X(Self, 输入数组): #Qwen3-Max编写
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大绝对值 = np.abs(块视图).max(axis=1)
            缩放因子 = np.where(块最大绝对值 > 1e-8, 块最大绝对值 / 127.0, 1.0).astype(np.float32)
            缩放因子编码 = Self.F32编码F16_S1M15(缩放因子, desc="tqdm.scale.quantization")
            量化块 = np.clip(np.round(块视图 / 缩放因子[:, None]), -127, 127).astype(np.int8)
            量化结果 = 量化块.flatten()[:len(扁平化)]
        return 量化结果.reshape(原始形状), 缩放因子编码, 原始形状
    def Q8_K_X解码F32(Self, 量化数据, 缩放因子编码, 原始形状): #Qwen3-Max编写
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization.reverse"):
            块大小 = int(Self.Config.VEC_QUANTIZATION_BLOCK_SIZE)
            量化数据 = np.asarray(量化数据, dtype=np.int8)
            扁平化量化 = 量化数据.flatten()
            填充长度 = (-len(扁平化量化)) % 块大小
            填充后量化 = np.pad(扁平化量化, (0, 填充长度), mode='constant')
            块数量 = int(len(填充后量化) // 块大小)
            块视图 = 填充后量化.reshape(块数量, 块大小)
            缩放因子 = Self.F16_S1M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            反量化块 = 块视图.astype(np.float32) * 缩放因子[:, None]
            反量化结果 = 反量化块.flatten()[:len(扁平化量化)]
        return 反量化结果.reshape((int(原始形状[0]), int(原始形状[1]),))
    def F32编码Q6_K_X(Self, 输入数组):
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"): #Qwen3-Max编写
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.abs(块视图).max(axis=1)
            缩放因子 = np.where(块最大值 > 1e-8, (2.0 * 块最大值) / 63.0, 2.0 / 63.0).astype(np.float32)
            Module.写入日志(f"[F32编码Q6_K_X] 缩放因子范围: {[缩放因子.min(), 缩放因子.max()]}")
            缩放因子编码 = Self.F32编码F16_S1M15(缩放因子, desc="tqdm.scale.quantization")
            缩放因子广播 = 缩放因子[:, None].astype(np.float32)
            块最大值广播 = 块最大值[:, None]
            量化值 = np.clip(np.round((块视图 + 块最大值广播) / 缩放因子广播),0, 63).astype(np.uint8)
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
        return 打包数据[:实际打包长度], 缩放因子编码, 原始形状
    def Q6_K_X解码F32(Self, 打包数据, 缩放因子编码, 原始形状): #Qwen3-Max编写
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
            缩放因子 = Self.F16_S1M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子_广播 = 缩放因子[:, np.newaxis]
            块最大值_广播 = (缩放因子 * 31.5)[:, np.newaxis]
            反量化块 = 解包块 * 缩放因子_广播 - 块最大值_广播
            反量化扁平 = 反量化块.flatten()[:总元素数]
        return 反量化扁平.reshape((int(原始形状[0]), int(原始形状[1])))
    def F32编码Q4_K_X(Self, 输入数组): #Qwen3-Max编写
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            打包长度 = (len(填充后数组) + 1) // 2
            打包数据 = np.zeros(打包长度, dtype=np.uint8)
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.abs(块视图).max(axis=1)
            缩放因子 = np.where(块最大值 > 1e-8, (2.0 * 块最大值) / 15.0, 2.0 / 15.0).astype(np.float32)
            缩放因子编码 = Self.F32编码F16_S1M15(缩放因子, desc="tqdm.scale.quantization")
            缩放因子广播 = 缩放因子[:, None].astype(np.float32)
            块最大值广播 = 块最大值[:, None]
            量化值 = np.clip(np.round((块视图 + 块最大值广播) / 缩放因子广播),0, 15).astype(np.uint8)
            高四位 = 量化值[:, 0::2] << 4
            低四位 = 量化值[:, 1::2]
            打包二维 = 高四位 | 低四位
            打包数据[:打包二维.size] = 打包二维.flatten()
            实际打包长度 = (len(扁平化) + 1) // 2
        return 打包数据[:实际打包长度], 缩放因子编码, 原始形状
    def Q4_K_X解码F32(Self, 打包数据, 缩放因子编码, 原始形状): #Qwen3-Max编写
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
            缩放因子 = Self.F16_S1M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子_广播 = 缩放因子[:, np.newaxis]
            块最大值_广播 = (缩放因子 * 7.5)[:, np.newaxis]
            反量化块 = 解包块.astype(np.float32) * 缩放因子_广播 - 块最大值_广播
            反量化扁平 = 反量化块.flatten()[:总元素数]
        return 反量化扁平.reshape((int(原始形状[0]), int(原始形状[1]),))
    def F32编码Q3_K_X(Self, 输入数组): #Qwen3-Max编写
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.abs(块视图).max(axis=1)
            缩放因子 = np.where(块最大值 > 1e-8, (2.0 * 块最大值) / 7.0, 2.0 / 7.0).astype(np.float32)
            缩放因子编码 = Self.F32编码F16_S1M15(缩放因子, desc="tqdm.scale.quantization")
            缩放因子广播 = 缩放因子[:, None].astype(np.float32)
            块最大值广播 = 块最大值[:, None]
            量化值 = np.clip(np.round((块视图 + 块最大值广播) / 缩放因子广播),0, 7).astype(np.uint8)
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
        return 打包数据[:实际打包长度], 缩放因子编码, 原始形状
    def Q3_K_X解码F32(Self, 打包数据, 缩放因子编码, 原始形状): #Qwen3-Max编写
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
            缩放因子 = Self.F16_S1M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子_广播 = 缩放因子[:, np.newaxis]
            块最大值_广播 = (缩放因子 * 3.5)[:, np.newaxis]
            反量化块 = 解包块 * 缩放因子_广播 - 块最大值_广播
            反量化扁平 = 反量化块.flatten()[:总元素数]
        return 反量化扁平.reshape((int(原始形状[0]), int(原始形状[1])))
    def F32编码Q2_K_X(Self, 输入数组): #Qwen3-Max编写
        for _ in tqdm(range(1), desc="tqdm.vectors.quantization"):
            块大小 = Self.Config.VEC_QUANTIZATION_BLOCK_SIZE
            输入数组 = np.asarray(输入数组, dtype=np.float32)
            原始形状 = 输入数组.shape
            扁平化 = 输入数组.flatten()
            填充长度 = (-len(扁平化)) % 块大小
            填充后数组 = np.pad(扁平化, (0, 填充长度), mode='constant')
            块数量 = len(填充后数组) // 块大小
            块视图 = 填充后数组.reshape(块数量, 块大小)
            块最大值 = np.abs(块视图).max(axis=1)
            缩放因子 = np.where(块最大值 > 1e-8, (2.0 * 块最大值) / 3.0, 2.0 / 3.0).astype(np.float32)
            缩放因子编码 = Self.F32编码F16_S1M15(缩放因子, desc="tqdm.scale.quantization")
            缩放因子广播 = 缩放因子[:, None].astype(np.float32)
            块最大值广播 = 块最大值[:, None]
            量化值 = np.clip(np.round((块视图 + 块最大值广播) / 缩放因子广播),0, 3).astype(np.uint8)
            每组数量 = 块大小 // 4
            量化分组 = 量化值.reshape(块数量, 每组数量, 4)
            打包二维 = (量化分组[:, :, 0] | 
                       (量化分组[:, :, 1] << 2) | 
                       (量化分组[:, :, 2] << 4) | 
                       (量化分组[:, :, 3] << 6)).astype(np.uint8)
            打包数据 = 打包二维.flatten()
            实际元素数 = len(扁平化)
            实际打包长度 = (实际元素数 + 3) // 4
        return 打包数据[:实际打包长度], 缩放因子编码, 原始形状
    def Q2_K_X解码F32(Self, 打包数据, 缩放因子编码, 原始形状): #Qwen3-Max编写
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
            缩放因子 = Self.F16_S1M15解码F32(缩放因子编码, desc="tqdm.scale.quantization.reverse").astype(np.float32)
            缩放因子_广播 = 缩放因子[:, np.newaxis]
            块最大值_广播 = (缩放因子 * 1.5)[:, np.newaxis]
            反量化块 = 解包分组 * 缩放因子_广播 - 块最大值_广播
            反量化扁平 = 反量化块.flatten()[:总元素数]
        return 反量化扁平.reshape((int(原始形状[0]), int(原始形状[1]),))
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
        return 打包数组.astype(np.uint16)
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
    def F32编码F16_S1M15(Self, 浮点数组, desc="tqdm.vectors.quantization"): #Qwen3-Max编写
        for _ in tqdm(range(1), desc=desc):
            浮点数组 = np.clip(浮点数组, -1.0, 1.0 - 2**-15)
            整型数值 = np.round(浮点数组 * 32768.0).astype(np.int16)
        return 整型数值.view(np.uint16)
    def F16_S1M15解码F32(Self, 编码数组, desc="tqdm.vectors.quantization.reverse"): #Qwen3-Max编写
        for _ in tqdm(range(1), desc=desc):
            整型数值 = 编码数组.view(np.int16)
        return 整型数值.astype(np.float32) / 32768.0
    def 解码向量(Self, 向量文件):
        if Self.Config.VEC_QUANTIZATION == "Q8_K_X":
            向量数组 = Self.Q8_K_X解码F32(向量文件[0], 向量文件[1], 向量文件[2])
        elif Self.Config.VEC_QUANTIZATION == "Q6_K_X":
            向量数组 = Self.Q6_K_X解码F32(向量文件[0], 向量文件[1], 向量文件[2])    
        elif Self.Config.VEC_QUANTIZATION == "Q4_K_X":
            向量数组 = Self.Q4_K_X解码F32(向量文件[0], 向量文件[1], 向量文件[2])
        elif Self.Config.VEC_QUANTIZATION == "Q3_K_X":
            向量数组 = Self.Q3_K_X解码F32(向量文件[0], 向量文件[1], 向量文件[2])
        elif Self.Config.VEC_QUANTIZATION == "Q2_K_X":
            向量数组 = Self.Q2_K_X解码F32(向量文件[0], 向量文件[1], 向量文件[2])
        elif Self.Config.VEC_QUANTIZATION == "Float16":
            向量数组 = 向量文件
        elif Self.Config.VEC_QUANTIZATION == "BFloat16":
            向量数组 = Self.BF16解码F32(向量文件)
        elif Self.Config.VEC_QUANTIZATION == "Float16_S1M15":
            向量数组 = Self.F16_S1M15解码F32(向量文件)
        elif Self.Config.VEC_QUANTIZATION == "Float32":
            向量数组 = 向量文件
        return 向量数组.astype(np.float32)
    def 编码向量(Self, 向量数组):
        if Self.Config.VEC_QUANTIZATION == "Q8_K_X":
            向量文件 = Self.F32编码Q8_K_X(向量数组)
        elif Self.Config.VEC_QUANTIZATION == "Q6_K_X":
            向量文件 = Self.F32编码Q6_K_X(向量数组)    
        elif Self.Config.VEC_QUANTIZATION == "Q4_K_X":
            向量文件 = Self.F32编码Q4_K_X(向量数组)
        elif Self.Config.VEC_QUANTIZATION == "Q3_K_X":
            向量文件 = Self.F32编码Q3_K_X(向量数组)
        elif Self.Config.VEC_QUANTIZATION == "Q2_K_X":
            向量文件 = Self.F32编码Q2_K_X(向量数组)
        elif Self.Config.VEC_QUANTIZATION == "Float16":
            向量文件 = 向量数组.astype(np.float16)
        elif Self.Config.VEC_QUANTIZATION == "BFloat16":
            向量文件 = Self.F32编码BF16(向量数组)
        elif Self.Config.VEC_QUANTIZATION == "Float16_S1M15":
            向量文件 = Self.F32编码F16_S1M15(向量数组)
        elif Self.Config.VEC_QUANTIZATION == "Float32":
            向量文件 = 向量数组.astype(np.float32)
        return 向量文件