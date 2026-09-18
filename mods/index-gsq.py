
from TranslatorLib import Mods, np, njit, _prange, faiss, SimpleNamespace, pickle

参数 = Mods().模组配置("index-gsq", {
    "Quant": "GSQ2",  
    "Nlist": 4.0,  
    "Nprobe": 39, 
    "QueryBlock": 64,
    "MinVectors": 50000, 
    "UsePCA": True,
    "PcaDim": -1,
    "BlockSize": 128,
    "RerankBlockSize": 128,
    "ItrsLM": 200,
    "SplLM": 0.05,
    "EsLM": 1e-6,
})
Quant = 参数["Quant"]
Nlist因子 = float(参数["Nlist"])
Nprobe默认 = int(参数["Nprobe"])
查询块 = int(参数["QueryBlock"])
最小向量数 = int(参数["MinVectors"])
用PCA = bool(参数["UsePCA"])
PcaDim = int(参数["PcaDim"])
块大小 = int(参数["BlockSize"])
重排块大小 = int(参数["RerankBlockSize"])
迭代次数 = int(参数["ItrsLM"])
采样比例 = float(参数["SplLM"])
早停阈值 = float(参数["EsLM"])

@njit(cache=True)
def 加速打包2(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 3) >> 2
    输出 = np.zeros(输出长度, dtype=np.uint8)
    for 索 in range(数量):
        输出[索 >> 2] |= np.uint8(量化值[索] << ((索 & 3) << 1))
    return 输出
@njit(cache=True, fastmath=True)
def 加速解包2(压缩, 数量):
    输出 = np.empty(数量, dtype=np.uint8)
    字节数 = len(压缩)
    for i in range(字节数):
        b = 压缩[i]
        idx = i * 4
        if idx < 数量: 输出[idx] = b & 3
        if idx + 1 < 数量: 输出[idx + 1] = (b >> 2) & 3
        if idx + 2 < 数量: 输出[idx + 2] = (b >> 4) & 3
        if idx + 3 < 数量: 输出[idx + 3] = (b >> 6) & 3
    return 输出
@njit(cache=True)
def 加速打包3(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 7) // 8
    输出 = np.zeros(输出长度 * 3, dtype=np.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 8; 偏移 = 块索 * 3
        v0 = np.uint8(量化值[基索] if 基索 < 数量 else 0)
        v1 = np.uint8(量化值[基索+1] if 基索+1 < 数量 else 0)
        v2 = np.uint8(量化值[基索+2] if 基索+2 < 数量 else 0)
        v3 = np.uint8(量化值[基索+3] if 基索+3 < 数量 else 0)
        v4 = np.uint8(量化值[基索+4] if 基索+4 < 数量 else 0)
        v5 = np.uint8(量化值[基索+5] if 基索+5 < 数量 else 0)
        v6 = np.uint8(量化值[基索+6] if 基索+6 < 数量 else 0)
        v7 = np.uint8(量化值[基索+7] if 基索+7 < 数量 else 0)
        输出[偏移]   = np.uint8((v0<<5)|(v1<<2)|(v2>>1))
        输出[偏移+1] = np.uint8(((v2&1)<<7)|(v3<<4)|(v4<<1)|(v5>>2))
        输出[偏移+2] = np.uint8(((v5&3)<<6)|(v6<<3)|v7)
    return 输出
@njit(cache=True)
def 加速解包3(压缩, 数量):
    块数 = len(压缩) // 3; 输出 = np.empty(块数 * 8, dtype=np.uint8)
    for 块索 in range(块数):
        偏移 = 块索 * 3; 基索 = 块索 * 8
        b0 = 压缩[偏移]; b1 = 压缩[偏移+1]; b2 = 压缩[偏移+2]
        输出[基索]   = np.uint8((b0>>5)&7)
        输出[基索+1] = np.uint8((b0>>2)&7)
        输出[基索+2] = np.uint8(((b0&3)<<1)|(b1>>7))
        输出[基索+3] = np.uint8((b1>>4)&7)
        输出[基索+4] = np.uint8((b1>>1)&7)
        输出[基索+5] = np.uint8(((b1&1)<<2)|(b2>>6))
        输出[基索+6] = np.uint8((b2>>3)&7)
        输出[基索+7] = np.uint8(b2&7)
    return 输出[:数量]
@njit(cache=True)
def 加速打包4(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 1) >> 1
    输出 = np.zeros(输出长度, dtype=np.uint8)
    for 索 in range(数量):
        if 索 & 1 == 0:
            输出[索 >> 1] = np.uint8(量化值[索] << 4)
        else:
            输出[索 >> 1] |= np.uint8(量化值[索] & 0xF)
    return 输出
@njit(cache=True, fastmath=True)
def 加速解包4(压缩, 数量):
    输出 = np.empty(数量, dtype=np.uint8)
    字节数 = len(压缩)
    for i in range(字节数):
        b = 压缩[i]
        idx = i * 2
        if idx < 数量: 输出[idx] = (b >> 4) & 0xF
        if idx + 1 < 数量: 输出[idx + 1] = b & 0xF
    return 输出
@njit(cache=True)
def 加速打包5(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 7) // 8
    输出 = np.zeros(输出长度 * 5, dtype=np.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 8; 偏移 = 块索 * 5
        v0 = np.uint8(量化值[基索] if 基索 < 数量 else 0)
        v1 = np.uint8(量化值[基索+1] if 基索+1 < 数量 else 0)
        v2 = np.uint8(量化值[基索+2] if 基索+2 < 数量 else 0)
        v3 = np.uint8(量化值[基索+3] if 基索+3 < 数量 else 0)
        v4 = np.uint8(量化值[基索+4] if 基索+4 < 数量 else 0)
        v5 = np.uint8(量化值[基索+5] if 基索+5 < 数量 else 0)
        v6 = np.uint8(量化值[基索+6] if 基索+6 < 数量 else 0)
        v7 = np.uint8(量化值[基索+7] if 基索+7 < 数量 else 0)
        输出[偏移]   = np.uint8((v0<<3)|(v1>>2))
        输出[偏移+1] = np.uint8(((v1&3)<<6)|(v2<<1)|(v3>>4))
        输出[偏移+2] = np.uint8(((v3&0xF)<<4)|(v4>>1))
        输出[偏移+3] = np.uint8(((v4&1)<<7)|(v5<<2)|(v6>>3))
        输出[偏移+4] = np.uint8(((v6&7)<<5)|v7)
    return 输出
@njit(cache=True)
def 加速解包5(压缩, 数量):
    块数 = len(压缩) // 5; 输出 = np.empty(块数 * 8, dtype=np.uint8)
    for 块索 in range(块数):
        偏移 = 块索 * 5; 基索 = 块索 * 8
        b0 = 压缩[偏移]; b1 = 压缩[偏移+1]; b2 = 压缩[偏移+2]; b3 = 压缩[偏移+3]; b4 = 压缩[偏移+4]
        输出[基索]   = np.uint8((b0>>3)&0x1F)
        输出[基索+1] = np.uint8(((b0&7)<<2)|(b1>>6))
        输出[基索+2] = np.uint8((b1>>1)&0x1F)
        输出[基索+3] = np.uint8(((b1&1)<<4)|(b2>>4))
        输出[基索+4] = np.uint8(((b2&0xF)<<1)|(b3>>7))
        输出[基索+5] = np.uint8((b3>>2)&0x1F)
        输出[基索+6] = np.uint8(((b3&3)<<3)|(b4>>5))
        输出[基索+7] = np.uint8(b4&0x1F)
    return 输出[:数量]
@njit(cache=True)
def 加速打包6(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 3) // 4
    输出 = np.zeros(输出长度 * 3, dtype=np.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 4; 偏移 = 块索 * 3
        v0 = np.uint8(量化值[基索] if 基索 < 数量 else 0)
        v1 = np.uint8(量化值[基索+1] if 基索+1 < 数量 else 0)
        v2 = np.uint8(量化值[基索+2] if 基索+2 < 数量 else 0)
        v3 = np.uint8(量化值[基索+3] if 基索+3 < 数量 else 0)
        输出[偏移]   = np.uint8((v0<<2)|(v1>>4))
        输出[偏移+1] = np.uint8(((v1&0xF)<<4)|(v2>>2))
        输出[偏移+2] = np.uint8(((v2&3)<<6)|v3)
    return 输出
@njit(cache=True)
def 加速解包6(压缩, 数量):
    块数 = len(压缩) // 3; 输出 = np.empty(块数 * 4, dtype=np.uint8)
    for 块索 in range(块数):
        偏移 = 块索 * 3; 基索 = 块索 * 4
        b0 = 压缩[偏移]; b1 = 压缩[偏移+1]; b2 = 压缩[偏移+2]
        输出[基索]   = np.uint8(b0>>2)
        输出[基索+1] = np.uint8(((b0&3)<<4)|(b1>>4))
        输出[基索+2] = np.uint8(((b1&0xF)<<2)|(b2>>6))
        输出[基索+3] = np.uint8(b2&0x3F)
    return 输出[:数量]
打包表 = {2: 加速打包2, 3: 加速打包3, 4: 加速打包4, 5: 加速打包5, 6: 加速打包6}
解包表 = {2: 加速解包2, 3: 加速解包3, 4: 加速解包4, 5: 加速解包5, 6: 加速解包6}
def _解包(数据包, 位深, 行数, 维度):
    if 位深 == 8: return 数据包["packed"]
    return 解包表[位深](数据包["packed"], 行数 * 维度)


@njit(cache=True, fastmath=True)
def _劳埃德最大化(样本, 电平数, 迭代次数, 早停阈值):
    数量 = len(样本)
    if 数量 == 0:
        return np.full(电平数, np.float32(0.5), dtype=np.float32)
    排序样本 = np.sort(样本)
    中心 = np.empty(电平数, dtype=np.float32)
    for i in range(电平数):
        位置 = (i * 2 + 1) * 数量 // (电平数 * 2)
        if 位置 >= 数量: 位置 = 数量 - 1
        中心[i] = 排序样本[位置]
    边界 = np.empty(电平数 - 1, dtype=np.float32)
    for _ in range(迭代次数):
        for i in range(电平数 - 1):
            边界[i] = (中心[i] + 中心[i + 1]) * np.float32(0.5)
        计数 = np.zeros(电平数, dtype=np.float32)
        累加 = np.zeros(电平数, dtype=np.float32)
        for i in range(数量):
            v = 排序样本[i]
            低 = 0; 高 = 电平数 - 1
            while 低 < 高:
                中 = (低 + 高) >> 1
                if v > 边界[中]: 低 = 中 + 1
                else: 高 = 中
            计数[低] += np.float32(1.0)
            累加[低] += v
        最大变动 = np.float32(0.0)
        for i in range(电平数):
            if 计数[i] > 0.0:
                新中心 = 累加[i] / 计数[i]
                变动 = abs(新中心 - 中心[i])
                if 变动 > 最大变动: 最大变动 = 变动
                中心[i] = 新中心
        if 最大变动 < 早停阈值: break
    return np.sort(中心)


@njit(cache=True, fastmath=True)
def _GSQ_NL码本(数组, 向量块, 电平数, 迭代次数, 采样上限, 早停阈值):
    行数, 维度 = 数组.shape
    组数 = (行数 + 向量块 - 1) // 向量块
    总数 = 组数 * 维度 * 向量块
    步长 = 1
    if 总数 > 采样上限: 步长 = (总数 + 采样上限 - 1) // 采样上限
    缓冲 = np.empty(采样上限, dtype=np.float32)
    计数 = 0
    for g in range(组数):
        for d in range(维度):
            min_val = 1e30; max_val = -1e30
            for r in range(向量块):
                行 = g * 向量块 + r
                val = 数组[行, d] if 行 < 行数 else np.float32(0.0)
                if val < min_val: min_val = val
                if val > max_val: max_val = val
            scale = max_val - min_val
            if scale < 1e-8: scale = 1e-8
            inv_scale = 1.0 / scale
            for r in range(向量块):
                行 = g * 向量块 + r
                if 行 < 行数 and 计数 < 采样上限 and (行 * 维度 + d) % 步长 == 0:
                    norm = (数组[行, d] - min_val) * inv_scale
                    if norm < 0.0: norm = 0.0
                    elif norm > 1.0: norm = 1.0
                    缓冲[计数] = norm
                    计数 += 1
    码本 = _劳埃德最大化(缓冲[:计数], 电平数, 迭代次数, 早停阈值)
    边界 = np.empty(电平数 - 1, dtype=np.float32)
    for i in range(电平数 - 1):
        边界[i] = (码本[i] + 码本[i + 1]) * np.float32(0.5)
    return 码本, 边界


@njit(cache=True, fastmath=True)
def _GSQ_NL编码_Numba(数组, 向量块, 边界):
    行数, 维度 = 数组.shape
    填充行 = (-行数) % 向量块
    总行数 = 行数 + 填充行
    组数 = 总行数 // 向量块
    电平数 = len(边界) + 1
    量化值 = np.zeros(总行数 * 维度, dtype=np.uint8)
    最小值存 = np.zeros((组数, 维度), dtype=np.float32)
    缩放值存 = np.zeros((组数, 维度), dtype=np.float32)
    for g in range(组数):
        for d in range(维度):
            min_val = 1e30; max_val = -1e30
            for r in range(向量块):
                行 = g * 向量块 + r
                val = 数组[行, d] if 行 < 行数 else np.float32(0.0)
                if val < min_val: min_val = val
                if val > max_val: max_val = val
            scale = max_val - min_val
            if scale < 1e-8: scale = 1e-8
            最小值存[g, d] = min_val
            缩放值存[g, d] = scale
            inv_scale = 1.0 / scale
            for r in range(向量块):
                行 = g * 向量块 + r
                if 行 < 行数:
                    norm = (数组[行, d] - min_val) * inv_scale
                    if norm < 0.0: norm = 0.0
                    elif norm > 1.0: norm = 1.0
                    低 = 0; 高 = 电平数 - 1
                    while 低 < 高:
                        中 = (低 + 高) >> 1
                        if norm > 边界[中]: 低 = 中 + 1
                        else: 高 = 中
                    量化值[行 * 维度 + d] = np.uint8(低)
                else:
                    量化值[行 * 维度 + d] = 0
    最大最小 = 1e-8
    最大缩放 = 1e-8
    for g in range(组数):
        for d in range(维度):
            abs_min = abs(最小值存[g, d])
            if abs_min > 最大最小: 最大最小 = abs_min
            if 缩放值存[g, d] > 最大缩放: 最大缩放 = 缩放值存[g, d]
    最小编码 = np.zeros(组数 * 维度, dtype=np.uint16)
    缩放编码 = np.zeros(组数 * 维度, dtype=np.uint16)
    inv_最大最小 = 1.0 / 最大最小
    inv_最大缩放 = 1.0 / 最大缩放
    for i in range(组数 * 维度):
        g = i // 维度
        d = i % 维度
        v_min = 最小值存[g, d] * inv_最大最小
        if v_min < -1.0: v_min = -1.0
        elif v_min > 0.9999695: v_min = 0.9999695
        最小编码[i] = np.uint16(np.int16(round(v_min * 32768.0)))
        v_scale = 缩放值存[g, d] * inv_最大缩放
        if v_scale < -1.0: v_scale = -1.0
        elif v_scale > 0.9999695: v_scale = 0.9999695
        缩放编码[i] = np.uint16(np.int16(round(v_scale * 32768.0)))
    return 量化值[:行数 * 维度], 最小编码, 缩放编码, np.float32(最大最小), np.float32(最大缩放)


def 采样数量(采样, 总数):
    if isinstance(采样, np.uint32) or 采样 > 1:
        return max(1, int(采样))
    return max(1, int(总数 * float(采样)))
def SpecTemp降维(数据, PCA_DIM):
    if PCA_DIM == -1: return 数据, None, None
    目标维度 = int(PCA_DIM)
    if 目标维度 >= 数据.shape[1]: return 数据, None, None
    均值 = np.mean(数据, axis=0, keepdims=True)
    数据中心 = 数据 - 均值
    if 数据.shape[0] > 数据.shape[1]:
        协方差 = 数据中心.T @ 数据中心
        特征值, 特征向量 = np.linalg.eigh(协方差)
        排序 = np.argsort(特征值)[::-1]
        主成分 = 特征向量[:, 排序[:目标维度]].T
    else:
        _, _, Vt = np.linalg.svd(数据中心, full_matrices=False)
        主成分 = Vt[:目标维度, :]
    return 均值, 主成分


@njit(cache=True)
def 预计算范数LUT(量化值_1D, 缩放值, 最小值, 码本, 向量块, 维度):
    组数 = 缩放值.shape[0]
    总行数 = len(量化值_1D) // 维度
    范数 = np.zeros(总行数, dtype=np.float32)
    for g in range(组数):
        起始 = g * 向量块
        结束 = min(起始 + 向量块, 总行数)
        for i in range(起始, 结束):
            s = np.float32(0.0)
            for d in range(维度):
                v = 码本[量化值_1D[i * 维度 + d]] * 缩放值[g, d] + 最小值[g, d]
                s += v * v
            范数[i] = np.sqrt(s) if s > 1e-8 else 1e-8
    return 范数
@njit(cache=True)
def 批量反量化(量化值_1D, 缩放值, 最小值, 码本, 向量块, 维度):
    总行数 = len(量化值_1D) // 维度
    组数 = 缩放值.shape[0]
    结果 = np.empty((总行数, 维度), dtype=np.float32)
    for g in range(组数):
        起始 = g * 向量块
        结束 = min(起始 + 向量块, 总行数)
        for i in range(起始, 结束):
            for d in range(维度):
                结果[i, d] = 码本[量化值_1D[i * 维度 + d]] * 缩放值[g, d] + 最小值[g, d]
    return 结果


class IndexGSQKCosineMoE:
    def __init__(self, app=None, quantization: int = 2, 路由: bool = True):
        self.日志 = getattr(app, "日志", None)
        self.Config = getattr(app, "Config", None)
        self.构建索引 = getattr(app, "构建索引", None)
        self.向量库 = []
        self.映射表 = np.empty(0, dtype=np.uint64)
        self.位深 = quantization
        self.路由 = 路由
        self.模式 = "IndexGSQKCosineFast"
        self.PCA_均值 = None
        self.PCA_主成分 = None
        self.码本 = None
        self.边界 = None
        self.cell中心 = None
        self.cell大小 = None
        self.cell块表 = []
        self.块归属 = []
        self.块起始 = np.empty(0, dtype=np.int64)

    def cpu(self):
        for 属性 in ["映射表", "块起始", "PCA_均值", "PCA_主成分", "cell中心", "cell大小"]:
            v = getattr(self, 属性, None)
            if v is not None and hasattr(v, "get"): setattr(self, 属性, v.get())
        for 包 in self.向量库:
            for k in ["packed", "mins", "scales", "norms", "codebook"]:
                if k in 包 and hasattr(包[k], "get"): 包[k] = 包[k].get()
            for k in ["max_min", "max_scale"]:
                if k in 包 and hasattr(包[k], "get"):
                    v = 包[k].get()
                    包[k] = float(v) if hasattr(v, "item") else v

    def gpu(self):
        for 属性 in ["映射表", "块起始", "PCA_均值", "PCA_主成分", "cell中心", "cell大小"]:
            v = getattr(self, 属性, None)
            if v is not None and not hasattr(v, "get"): setattr(self, 属性, np.asarray(v))
        for 包 in self.向量库:
            for k in ["packed", "mins", "scales", "norms", "codebook"]:
                if k in 包 and not hasattr(包[k], "get"): 包[k] = np.asarray(包[k])
            for k in ["max_min", "max_scale"]:
                if k in 包 and not hasattr(包[k], "get"): 包[k] = np.asarray(包[k])

    def train(self, vectors):
        if 用PCA and PcaDim != -1:
            self.PCA_均值, self.PCA_主成分 = SpecTemp降维(vectors, PcaDim)

    def _降维(self, 数组):
        if self.PCA_主成分 is None:
            return np.ascontiguousarray(数组, dtype=np.float32)
        return np.ascontiguousarray(np.dot(数组 - self.PCA_均值, self.PCA_主成分.T), dtype=np.float32)

    def add(self, vectors):
        self.向量库 = []
        数组 = self._降维(np.atleast_2d(vectors))
        行数, 维度 = 数组.shape
        if self.码本 is None:
            self.码本, self.边界 = _GSQ_NL码本(数组, 块大小, 1 << self.位深,
                                                迭代次数, 采样数量(采样比例, 数组.size), 早停阈值)

        if not self.路由:
            self.cell中心 = None; self.cell大小 = None
            self.cell块表 = []; self.块归属 = []
            self._编码分块(数组, 块大小, 0, None)
            self.映射表 = np.arange(行数, dtype=np.uint64)
            self._刷新块起始()
            return

        归一 = np.ascontiguousarray(数组, dtype=np.float32).copy()
        faiss.normalize_L2(归一)
        nlist = max(1, min(int(Nlist因子 * np.sqrt(max(1, 行数))), 行数 // 8))
        索引 = faiss.IndexIVFFlat(faiss.IndexFlatIP(维度), 维度, nlist, faiss.METRIC_INNER_PRODUCT)
        索引.train(归一)
        索引.add(归一)
        _, 归 = 索引.quantizer.search(归一, 1)
        归属 = 归.ravel().astype(np.int64)
        归属[归属 < 0] = 0
        中心 = 索引.quantizer.reconstruct_n(0, nlist).astype(np.float32)
        范式 = np.linalg.norm(中心, axis=1, keepdims=True)
        范式[范式 < 1e-8] = 1e-8
        self.cell中心 = np.ascontiguousarray(中心 / 范式, dtype=np.float32)
        self.cell大小 = np.bincount(归属, minlength=nlist)
        self.cell块表 = [[] for _ in range(nlist)]
        self.块归属 = []
        self._编码分块(数组, 块大小, 0, 归属)
        self._刷新块起始()

    def _刷新块起始(self):
        长度 = np.array([包["shape"][0] for 包 in self.向量库], dtype=np.int64)
        if 长度.size:
            self.块起始 = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(长度)[:-1]])
        else:
            self.块起始 = np.empty(0, dtype=np.int64)

    def _打包(self, 量化值):
        if self.位深 == 8: return 量化值
        return 打包表[self.位深](量化值)

    def _编码分块(self, 数组, 块大小, 旧行数, 归属):
        组边界 = []
        if 归属 is None:
            总行 = 数组.shape[0]
            映射 = np.arange(旧行数, 旧行数 + 总行, dtype=np.uint64)
            for 起 in range(0, 总行, 重排块大小):
                组边界.append(np.arange(起, min(起 + 重排块大小, 总行)))
        else:
            排序 = np.argsort(归属, kind="stable")
            边界 = np.flatnonzero(np.diff(归属[排序])) + 1
            组边界 = list(np.split(排序, 边界))
            映射 = np.empty(数组.shape[0], dtype=np.uint64)
            已写 = 0
            for 组 in 组边界:
                映射[已写:已写 + 组.size] = 组 + 旧行数
                已写 += 组.size

        for 组 in 组边界:
            if 组.size == 0: continue
            块数组 = np.ascontiguousarray(数组[组])
            量化值, 最小编码, 缩放编码, 最大最小, 最大缩放 = _GSQ_NL编码_Numba(块数组, 块大小, self.边界)
            维度 = 数组.shape[1]
            组数 = (组.size + 块大小 - 1) // 块大小
            缩放值_解码 = (np.asarray(缩放编码).view(np.int16).astype(np.float32) / 32768.0 * 最大缩放).reshape(组数, 维度)
            最小值_解码 = (np.asarray(最小编码).view(np.int16).astype(np.float32) / 32768.0 * 最大最小).reshape(组数, 维度)
            块下标 = len(self.向量库)
            self.向量库.append({
                "packed": self._打包(量化值), "mins": 最小编码, "scales": 缩放编码,
                "max_min": 最大最小, "max_scale": 最大缩放,
                "shape": (组.size, 维度), "bit_depth": self.位深, "vec_block": 块大小,
                "norms": 预计算范数LUT(量化值, 缩放值_解码, 最小值_解码, self.码本, 块大小, 维度),
                "codebook": self.码本,
            })
            if 归属 is not None:
                cid = int(归属[组[0]])
                self.块归属.append(cid)
                self.cell块表[cid].append(块下标)

        if 旧行数 == 0:
            self.映射表 = 映射
        else:
            self.映射表 = np.concatenate([self.映射表, 映射])

    def _解码包(self, 包):
        行数, 维度 = 包["shape"]
        量化值 = _解包(包, 包["bit_depth"], 行数, 维度)
        组数 = (行数 + 包["vec_block"] - 1) // 包["vec_block"]
        缩放值 = (np.asarray(包["scales"]).view(np.int16).astype(np.float32)
                / 32768.0 * 包["max_scale"]).reshape(组数, 维度)
        最小值 = (np.asarray(包["mins"]).view(np.int16).astype(np.float32)
                / 32768.0 * 包["max_min"]).reshape(组数, 维度)
        块矩阵 = 批量反量化(量化值, 缩放值, 最小值, 包["codebook"], 包["vec_block"], 维度)
        范数 = 包.get("norms")
        if 范数 is None:
            范数 = np.linalg.norm(块矩阵, axis=1)
            范数[范数 < 1e-8] = 1e-8
        return 块矩阵, 范数

    def 激活块(self, 查询归一=None, 强制全量=False):
        总块数 = len(self.向量库)
        if (强制全量 or self.cell中心 is None or self.cell中心.shape[0] <= 1
                or sum(包["shape"][0] for 包 in self.向量库) < 最小向量数):
            return list(range(总块数))
        分数 = 查询归一 @ self.cell中心.T
        nprobe = int(min(max(1, Nprobe默认), self.cell中心.shape[0]))
        位置 = np.argpartition(-分数, nprobe - 1, axis=1)[:, :nprobe]
        计数 = {}
        for c in np.unique(位置):
            for b in self.cell块表[int(c)]:
                计数[int(b)] = 计数.get(int(b), 0) + 1
        return sorted(计数)

    def _单块检索(self, 查询归一, 实际_k):
        查询数量 = 查询归一.shape[0]
        块列表 = self.激活块(查询归一)
        总激活行 = int(sum(self.向量库[b]["shape"][0] for b in 块列表))
        全分数 = np.full((查询数量, max(1, 总激活行)), -np.inf, dtype=np.float32)
        全行号 = np.empty(max(1, 总激活行), dtype=np.int64)
        列 = 0
        for b in 块列表:
            包 = self.向量库[b]
            行数 = 包["shape"][0]
            起 = int(self.块起始[b])
            块矩阵, 范数 = self._解码包(包)
            全分数[:, 列:列 + 行数] = (查询归一 @ 块矩阵.T) / 范数[np.newaxis, :]
            全行号[列:列 + 行数] = np.arange(起, 起 + 行数, dtype=np.int64)
            列 += 行数
            del 块矩阵
        取 = min(实际_k, 全分数.shape[1])
        位置 = np.argpartition(-全分数, 取 - 1, axis=1)[:, :取]
        分 = np.take_along_axis(全分数, 位置, axis=1)
        序 = np.argsort(-分, axis=1)
        分 = np.take_along_axis(分, 序, axis=1)
        位置 = np.take_along_axis(位置, 序, axis=1)
        del 全分数
        行号 = np.take(全行号, 位置)
        if 取 < 实际_k:
            补 = 实际_k - 取
            分 = np.hstack([分, np.full((查询数量, 补), -np.inf, dtype=np.float32)])
            行号 = np.hstack([行号, np.full((查询数量, 补), -1, dtype=np.int64)])
        return 分, 行号

    def search(self, query, k):
        if int(self.块起始.shape[0]) != len(self.向量库):
            self._刷新块起始()
            if getattr(self, "块归属", None) is None or len(self.块归属) != len(self.向量库):
                self.块归属 = list(range(len(self.向量库)))
        查询矩阵 = self._降维(np.atleast_2d(query).astype(np.float32))
        查询数量 = 查询矩阵.shape[0]
        查询范数 = np.linalg.norm(查询矩阵, axis=1, keepdims=True)
        查询范数[查询范数 < 1e-8] = 1e-8
        查询归一 = 查询矩阵 / 查询范数

        总目标数 = sum(包["shape"][0] for 包 in self.向量库)
        实际_k = min(k, 总目标数)
        if 实际_k <= 0:
            return np.empty((查询数量, 0), dtype=np.float32), np.empty((查询数量, 0), dtype=np.int64)
        块 = max(1, int(查询块))
        分列表, 行列表 = [], []
        for 起 in range(0, 查询数量, 块):
            止 = min(起 + 块, 查询数量)
            分, 行号 = self._单块检索(查询归一[起:止], 实际_k)
            分列表.append(分); 行列表.append(行号)
        分数 = np.vstack(分列表); 行号 = np.vstack(行列表)
        del 分列表, 行列表

        if 查询数量 > 块:
            序 = np.argsort(-分数, axis=1)[:, :实际_k]
            分数 = np.take_along_axis(分数, 序, axis=1)
            行号 = np.take_along_axis(行号, 序, axis=1)

        合法 = 行号 >= 0
        原下标 = np.full(行号.shape, -1, dtype=np.int64)
        np.copyto(原下标, self.映射表.astype(np.int64)[np.where(合法, 行号, 0)], where=合法)
        np.copyto(分数, -np.inf, where=~合法)

        if k > 总目标数:
            原下标 = np.hstack([原下标, np.full((查询数量, k - 总目标数), -1, dtype=np.int64)])
            分数 = np.hstack([分数, np.full((查询数量, k - 总目标数), -np.inf, dtype=np.float32)])
        return 分数.astype(np.float32), 原下标

    def _状态(self):
        return {
            "模式": self.模式, "向量库": self.向量库, "映射表": self.映射表,
            "位深": self.位深, "路由": self.路由,
            "PCA_均值": self.PCA_均值, "PCA_主成分": self.PCA_主成分,
            "码本": self.码本, "边界": self.边界,
            "cell中心": self.cell中心, "cell大小": self.cell大小,
            "cell块表": self.cell块表, "块归属": self.块归属, "块起始": self.块起始,
        }

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self._状态(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def serialize(self):
        return pickle.dumps(self._状态(), protocol=pickle.HIGHEST_PROTOCOL)


class IndexGSQKCosineMoE_(IndexGSQKCosineMoE):
    def __init__(self, app=None, quantization: int = 2, 路由: bool = True):
        super().__init__(app=app, quantization=quantization, 路由=True)
        self.模式 = "IndexGSQKCosineMoE"


class IndexGSQKCosineFast(IndexGSQKCosineMoE):
    def __init__(self, app=None, quantization: int = 2, 路由: bool = False):
        super().__init__(app=app, quantization=quantization, 路由=False)
        self.模式 = "IndexGSQKCosineFast"


Mods().注册索引类("IndexGSQKCosineMoE")(IndexGSQKCosineMoE_)
Mods().注册索引类("IndexGSQKCosineFast")(IndexGSQKCosineFast)

@Mods().注册索引类型("GSQMoE", 数据源="Vector", 支持增量=True)
def 构建MoE(Self, 向量, 索引=None):
    return IndexGSQKCosineMoE_(app=Self, quantization=Self.量化映射[Quant], 路由=True), True


@Mods().注册索引类型("GSQFast", 数据源="Vector", 支持增量=True)
def 构建Fast(Self, 向量, 索引=None):
    return IndexGSQKCosineFast(app=Self, quantization=Self.量化映射[Quant]), True

@Mods().注册索引部件("GSQBase")
def 部件():
    return SimpleNamespace(
        加速打包2=加速打包2, 加速打包3=加速打包3, 加速打包4=加速打包4, 加速打包5=加速打包5, 加速打包6=加速打包6,
        加速解包2=加速解包2, 加速解包3=加速解包3, 加速解包4=加速解包4, 加速解包5=加速解包5, 加速解包6=加速解包6,
        预计算范数LUT=预计算范数LUT, 批量反量化=批量反量化, _解包=_解包, SpecTemp降维=SpecTemp降维,
        采样数量=采样数量, _GSQ_NL码本=_GSQ_NL码本, _GSQ_NL编码_Numba=_GSQ_NL编码_Numba,
        打包表=打包表, 解包表=解包表,
        IndexGSQKCosineMoE=IndexGSQKCosineMoE_, IndexGSQKCosineFast=IndexGSQKCosineFast,
    )
