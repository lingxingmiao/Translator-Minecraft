from TranslatorLib import njit, pickle, numba, numpy as np, faiss

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
@njit(cache=True, fastmath=True)
def _GSQ_K编码_Numba(数组, 向量块, 最大量级):
    行数, 维度 = 数组.shape
    填充行 = (-行数) % 向量块
    总行数 = 行数 + 填充行
    组数 = 总行数 // 向量块
    量化值 = np.zeros(总行数 * 维度, dtype=np.uint8)
    最小值存 = np.zeros((组数, 维度), dtype=np.float32)
    缩放值存 = np.zeros((组数, 维度), dtype=np.float32)
    for g in range(组数):
        for d in range(维度):
            min_val = 1e30
            max_val = -1e30
            for r in range(向量块):
                row_idx = g * 向量块 + r
                if row_idx < 行数:
                    val = 数组[row_idx, d]
                else:
                    val = 0.0
                if val < min_val: min_val = val
                if val > max_val: max_val = val
            scale = max_val - min_val
            if scale < 1e-8: scale = 1e-8
            最小值存[g, d] = min_val
            缩放值存[g, d] = scale
            inv_scale = 1.0 / scale
            for r in range(向量块):
                row_idx = g * 向量块 + r
                if row_idx < 行数:
                    val = 数组[row_idx, d]
                    norm = (val - min_val) * inv_scale
                    if norm < 0.0: norm = 0.0
                    elif norm > 1.0: norm = 1.0
                    q = round(norm * 最大量级)
                    量化值[row_idx * 维度 + d] = np.uint8(q)
                else:
                    量化值[row_idx * 维度 + d] = 0
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
def 向量重排(数组, 聚类块大小, 重排乘数):
    向量数量, 向量维度 = 数组.shape
    if 向量数量 <= 聚类块大小:
        return 数组, np.arange(向量数量)
    数组_归一 = np.ascontiguousarray(数组, dtype=np.float32).copy()
    faiss.normalize_L2(数组_归一)
    nlist = max(1, int(np.sqrt(向量数量)))
    量化器 = faiss.IndexFlatIP(向量维度)
    向量索引 = faiss.IndexIVFScalarQuantizer(
        量化器, 向量维度, nlist,
        faiss.ScalarQuantizer.QT_4bit,
        faiss.METRIC_INNER_PRODUCT
    )
    向量索引.set_direct_map_type(faiss.DirectMap.Hashtable)
    向量索引.train(数组_归一)
    向量索引.add(数组_归一)
    向量索引.nprobe = min(nlist, max(1, nlist // 4))
    已访问掩码 = np.zeros(向量数量, dtype=bool)
    重排索引列表 = []
    粗搜数量 = 聚类块大小 * 重排乘数
    搜索指针 = 0
    已处理数量 = 0
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
        delete_ids = 最终索引.astype(np.int64)
        向量索引.remove_ids(delete_ids)
    if 向量数量 <= 65535:
        数据类型 = np.uint16
    elif 向量数量 <= 4294967296:
        数据类型 = np.uint32
    else:
        数据类型 = np.uint64
    映射表 = np.array(重排索引列表, dtype=数据类型)
    重排后数组 = 数组[映射表]
    return 重排后数组, 映射表
@njit(cache=True)
def 预计算范数LUT(量化值_1D, 缩放值, 最小值, 最大量级, 向量块, 维度):
    组数 = 缩放值.shape[0]
    总行数 = len(量化值_1D) // 维度
    inv_L = np.float32(1.0 / 最大量级)
    codebook = np.zeros((组数, 维度, 最大量级 + 1), dtype=np.float32)
    for g in range(组数):
        for d in range(维度):
            for v in range(最大量级 + 1):
                codebook[g, d, v] = (np.float32(v) * inv_L) * 缩放值[g, d] + 最小值[g, d]
    范数 = np.zeros(总行数, dtype=np.float32)
    for g in range(组数):
        起始 = g * 向量块
        结束 = min(起始 + 向量块, 总行数)
        for i in range(起始, 结束):
            s = np.float32(0.0)
            for d in range(维度):
                v = codebook[g, d, 量化值_1D[i * 维度 + d]]
                s += v * v
            范数[i] = np.sqrt(s) if s > 1e-8 else 1e-8
    return 范数
@njit(cache=True)
def 批量反量化(量化值_1D, 缩放值, 最小值, 最大量级, 向量块, 维度):
    总行数 = len(量化值_1D) // 维度
    组数 = 缩放值.shape[0]
    inv_L = np.float32(1.0 / 最大量级)
    结果 = np.empty((总行数, 维度), dtype=np.float32)
    for g in range(组数):
        起始 = g * 向量块
        结束 = min(起始 + 向量块, 总行数)
        for i in range(起始, 结束):
            for d in range(维度):
                结果[i, d] = (np.float32(量化值_1D[i * 维度 + d]) * inv_L) * 缩放值[g, d] + 最小值[g, d]
    return 结果
@njit(fastmath=True, parallel=True, cache=True)
def 极速LUT_Cosine检索(查询矩阵_归一, 量化值_1D, 缩放值_块, 最小值_块, 最大量级, 块大小, 维度):
    查询数 = 查询矩阵_归一.shape[0]
    总行数 = 量化值_1D.shape[0] // 维度
    组数 = 缩放值_块.shape[0]
    分数矩阵 = np.zeros((查询数, 总行数), dtype=np.float32)
    inv_L = np.float32(1.0 / 最大量级)
    for g in numba.prange(组数):
        起始行 = g * 块大小
        结束行 = min(起始行 + 块大小, 总行数)
        当前块大小 = 结束行 - 起始行
        块矩阵 = np.empty((当前块大小, 维度), dtype=np.float32)
        范数数组 = np.empty(当前块大小, dtype=np.float32)
        for i in range(当前块大小):
            row_idx = 起始行 + i
            norm_sq = np.float32(0.0)
            for d in range(维度):
                X = (np.float32(量化值_1D[row_idx * 维度 + d]) * inv_L) * 缩放值_块[g, d] + 最小值_块[g, d]
                块矩阵[i, d] = X
                norm_sq += X * X
            范数数组[i] = np.sqrt(norm_sq) if norm_sq > 1e-8 else 1e-8
        点积 = np.dot(块矩阵, 查询矩阵_归一.T)
        for i in range(当前块大小):
            inv_norm = np.float32(1.0 / 范数数组[i])
            for q_idx in range(查询数):
                分数矩阵[q_idx, 起始行 + i] = 点积[i, q_idx] * inv_norm
    return 分数矩阵
def 编码(self, 数组, 位深, 日志):
    重排数组, self.映射表 = 向量重排(数组, self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, self.Config.INDEX_GSQ_RERANKER_FACTOR)
    最大量级 = (1 << 位深) - 1
    总行数 = len(重排数组)
    维度 = 重排数组.shape[1]
    for 起始 in range(0, 总行数, self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE):
        结束 = min(起始 + self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, 总行数)
        块数组 = 重排数组[起始:结束]
        量化值, 最小编码, 缩放编码, 最大最小, 最大缩放 = _GSQ_K编码_Numba(块数组, self.Config.INDEX_GSQ_BLOCK_SIZE, 最大量级)
        if 位深 == 8:
            压缩 = 量化值
        elif 位深 == 6:
            压缩 = 加速打包6(量化值)
        elif 位深 == 4:
            压缩 = 加速打包4(量化值)
        elif 位深 == 3:
            压缩 = 加速打包3(量化值)
        elif 位深 == 2:
            压缩 = 加速打包2(量化值)
        try:
            self.向量库.append({"packed": 压缩, "mins": 最小编码, "scales": 缩放编码, "max_min": 最大最小, "max_scale": 最大缩放, "shape": (结束 - 起始, 维度), "bit_depth": 位深, "vec_block": self.Config.INDEX_GSQ_BLOCK_SIZE})
        except UnboundLocalError:
            日志("log.index.gsq.quantization.err", info_level=3)
def 重排列表(原始列表, 映射表):
    if 原始列表 is None:
        return None
    return [原始列表[i] for i in 映射表]
def _解包(数据包, 位深, 行数, 维度):
    if 位深 == 8:
        return 数据包["packed"]
    elif 位深 == 6:
        return 加速解包6(数据包["packed"], 行数 * 维度)
    elif 位深 == 4:
        return 加速解包4(数据包["packed"], 行数 * 维度)
    elif 位深 == 3:
        return 加速解包3(数据包["packed"], 行数 * 维度)
    elif 位深 == 2:
        return 加速解包2(数据包["packed"], 行数 * 维度)
@njit(fastmath=True, parallel=True, cache=True)
def 极速GSQK余弦检索(查询矩阵, 量化值_uint8, 缩放值, 最小值, 范数数组, 行数, 维度, 组数, 块大小):
    查询数 = 查询矩阵.shape[0]
    分数矩阵 = np.zeros((查询数, 行数), dtype=np.float32)
    for q in numba.prange(查询数):
        for g in range(组数):
            起始行 = g * 块大小
            结束行 = min(起始行 + 块大小, 行数)
            B = np.float32(0.0)
            for d in range(维度):
                B += 查询矩阵[q, d] * 最小值[g, d]
            for i in range(起始行, 结束行):
                dot = B
                for d in range(维度):
                    dot += 查询矩阵[q, d] * 缩放值[g, d] * np.float32(量化值_uint8[i * 维度 + d])
                分数矩阵[q, i] = dot / 范数数组[i]
    return 分数矩阵
class IndexGSQKCosine:
    def __init__(self, app = None, quantization: int = 2): # Module
        if app:
            self.日志 = app.日志
            self.Config = app.Config
        self.向量库 = []
        self.映射表 = []
        self.位深 = quantization
        self.模式 = "IndexGSQKCosine"
    def add(self, vectors):
        编码(self, vectors, self.位深, self.日志)
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                "模式": self.模式,
                "向量库": self.向量库,
                "映射表": self.映射表,
                "位深": self.位深
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    def search(self, query, k):
        if isinstance(self.映射表, list):
            self.映射表 = np.array(self.映射表)
        查询矩阵 = np.atleast_2d(query).astype(np.float32)
        查询数量 = 查询矩阵.shape[0]
        查询范数 = np.linalg.norm(查询矩阵, axis=1, keepdims=True)
        查询范数[查询范数 < 1e-8] = 1e-8
        查询归一 = 查询矩阵 / 查询范数
        总目标数 = sum(数据包["shape"][0] for 数据包 in self.向量库)
        全局相似度 = np.zeros((查询数量, 总目标数), dtype=np.float32)
        当前偏移 = 0
        for 数据包 in self.向量库:
            行数, 维度 = 数据包["shape"]
            块大小 = 数据包["vec_block"]
            位深 = 数据包["bit_depth"]
            最大量级 = (1 << 位深) - 1
            if 位深 == 8: 量化值_1D = 数据包["packed"]
            elif 位深 == 6: 量化值_1D = 加速解包6(数据包["packed"], 行数 * 维度)
            elif 位深 == 4: 量化值_1D = 加速解包4(数据包["packed"], 行数 * 维度)
            elif 位深 == 3: 量化值_1D = 加速解包3(数据包["packed"], 行数 * 维度)
            elif 位深 == 2: 量化值_1D = 加速解包2(数据包["packed"], 行数 * 维度)
            组数 = (行数 + 块大小 - 1) // 块大小
            缩放值_块 = (np.asarray(数据包["scales"]).view(np.int16).astype(np.float32) / 32768.0 * 数据包["max_scale"]).reshape(组数, 维度)
            最小值_块 = (np.asarray(数据包["mins"]).view(np.int16).astype(np.float32) / 32768.0 * 数据包["max_min"]).reshape(组数, 维度)
            分块相似度 = 极速LUT_Cosine检索(查询归一, 量化值_1D, 缩放值_块, 最小值_块, 最大量级, 块大小, 维度)
            全局相似度[:, 当前偏移:当前偏移+行数] = 分块相似度
            当前偏移 += 行数
        实际_k = min(k, 总目标数)
        if 实际_k > 0:
            未排序索引 = np.argpartition(全局相似度, -实际_k, axis=1)[:, -实际_k:]
            顶部分数 = np.take_along_axis(全局相似度, 未排序索引, axis=1)
            排序顺序 = np.argsort(-顶部分数, axis=1)
            重排后TopK索引 = np.take_along_axis(未排序索引, 排序顺序, axis=1)
            TopK分数 = np.take_along_axis(顶部分数, 排序顺序, axis=1)
        else:
            重排后TopK索引 = np.empty((查询数量, 0), dtype=np.int64)
            TopK分数 = np.empty((查询数量, 0), dtype=np.float32)
        原始TopK索引 = np.take(self.映射表, 重排后TopK索引).astype(np.int64)
        if k > 总目标数:
            填充索引 = np.full((查询数量, k - 总目标数), -1, dtype=np.int64)
            填充分数 = np.full((查询数量, k - 总目标数), -np.inf, dtype=np.float32)
            原始TopK索引 = np.hstack([原始TopK索引, 填充索引])
            TopK分数 = np.hstack([TopK分数, 填充分数])
        return TopK分数.astype(np.float32), 原始TopK索引
class IndexGSQKCosineFast:
    def __init__(self, app = None, quantization: int = 2): # Module
        if app:
            self.日志 = app.日志
            self.Config = app.Config
        self.向量库 = []
        self.映射表 = []
        self.位深 = quantization
        self.模式 = "IndexGSQKCosineFast"
    def add(self, vectors):
        编码(self, vectors, self.位深, self.日志)
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                "模式": self.模式,
                "向量库": self.向量库,
                "映射表": self.映射表,
                "位深": self.位深
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    def search(self, query, k):
        查询矩阵 = np.atleast_2d(query).astype(np.float32)
        查询数量 = 查询矩阵.shape[0]
        查询范数 = np.linalg.norm(查询矩阵, axis=1, keepdims=True)
        查询范数[查询范数 < 1e-8] = 1e-8
        查询归一 = 查询矩阵 / 查询范数
        总目标数 = sum(包["shape"][0] for 包 in self.向量库)
        全局分数 = np.zeros((查询数量, 总目标数), dtype=np.float32)
        偏移 = 0
        for 数据包 in self.向量库:
            行数, 维度 = 数据包["shape"]
            缩放值 = (np.asarray(数据包["scales"]).view(np.int16).astype(np.float32)
                    / 32768.0 * 数据包["max_scale"]).reshape(-1, 维度)
            最小值 = (np.asarray(数据包["mins"]).view(np.int16).astype(np.float32)
                    / 32768.0 * 数据包["max_min"]).reshape(-1, 维度)
            位深 = 数据包["bit_depth"]
            量化值 = _解包(数据包, 位深, 行数, 维度)
            块矩阵 = 批量反量化(量化值, 缩放值, 最小值, (1 << 位深) - 1, 数据包["vec_block"], 维度)
            del 量化值
            块范数 = 数据包.get("norms")
            if 块范数 is None:
                块范数 = np.linalg.norm(块矩阵, axis=1)
                块范数[块范数 < 1e-8] = 1e-8
            点积矩阵 = 查询归一 @ 块矩阵.T
            全局分数[:, 偏移:偏移+行数] = 点积矩阵 / 块范数[np.newaxis, :]
            del 块矩阵, 点积矩阵 
            偏移 += 行数
        实际_k = min(k, 总目标数)
        if 实际_k > 0:
            未排序索引 = np.argpartition(全局分数, -实际_k, axis=1)[:, -实际_k:]
            顶部分数 = np.take_along_axis(全局分数, 未排序索引, axis=1)
            del 全局分数 
            排序顺序 = np.argsort(-顶部分数, axis=1)
            重排后TopK索引 = np.take_along_axis(未排序索引, 排序顺序, axis=1)
            TopK分数 = np.take_along_axis(顶部分数, 排序顺序, axis=1)
            del 未排序索引, 顶部分数, 排序顺序
        else:
            del 全局分数
            重排后TopK索引 = np.empty((查询数量, 0), dtype=np.int64)
            TopK分数 = np.empty((查询数量, 0), dtype=np.float32)
        原始TopK索引 = np.take(self.映射表, 重排后TopK索引).astype(np.int64)
        del 重排后TopK索引
        if k > 总目标数:
            填充索引 = np.full((查询数量, k - 总目标数), -1, dtype=np.int64)
            填充分数 = np.full((查询数量, k - 总目标数), -np.inf, dtype=np.float32)
            原始TopK索引 = np.hstack([原始TopK索引, 填充索引])
            TopK分数 = np.hstack([TopK分数, 填充分数])
        return TopK分数.astype(np.float32), 原始TopK索引
class IndexGSQKCosineMoE:
    def __init__(self, app=None, quantization: int = 2):
        if app:
            self.日志 = app.日志
            self.Config = app.Config
        self.向量库 = []
        self.映射表 = []
        self.位深 = quantization
        self.模式 = "IndexGSQKCosineMoE"
        self.路由数据 = None  
        self.exp = 32 
        self.已训练 = False
        self.旋转块矩阵 = None
        self.LM中心 = None      
        self.LM边界 = None
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                "模式": self.模式,
                "向量库": self.向量库,
                "映射表": self.映射表,
                "位深": self.位深,
                "路由数据": self.路由数据,
                "exp": self.exp,
                "已训练": self.已训练,
                "旋转块矩阵": self.旋转块矩阵.astype(np.float16),
                "LM中心": self.LM中心,
                "LM边界": self.LM边界
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    def _应用分块旋转(self, 数组, 旋转块矩阵):
        N, D = 数组.shape
        n块 = D // 16
        分块 = 数组.reshape(N, n块, 16)
        旋转后 = np.einsum('nbi,bij->nbj', 分块, 旋转块矩阵, optimize=True)
        return 旋转后.reshape(N, D)
    def _SVD学习最优旋转_分块(self, 数据, 迭代次数=5, 初始中心=None):
        行数, 维度 = 数据.shape
        n块 = 维度 // 16
        旋转块 = np.tile(np.eye(16, dtype=np.float32), (n块, 1, 1))
        中心 = 初始中心 if 初始中心 is not None else np.linspace(-2.5, 2.5, 16).astype(np.float32)
        边界 = (中心[:-1] + 中心[1:]) / 2.0
        Q块大小 = self.Config.INDEX_GSQ_MOE_BLOCK_SIZE
        for _ in range(迭代次数):
            旋转后 = self._应用分块旋转(数据, 旋转块)
            展平 = 旋转后.reshape(-1, Q块大小)
            高位, 低位 = np.percentile(展平, [99.9, 0.1], axis=1, keepdims=True)
            np.clip(展平, 低位, 高位, out=展平)
            标准差 = np.maximum(np.std(展平, axis=1, ddof=0, keepdims=True), 1e-8)
            归一化 = 展平 / 标准差
            量化索引 = np.searchsorted(边界, 归一化).astype(np.uint8)
            重建 = (中心[量化索引] * 标准差).reshape(行数, 维度)
            重建分块 = 重建.reshape(行数, n块, 16)
            数据分块 = 数据.reshape(行数, n块, 16)
            for b in range(n块):
                协方差 = 数据分块[:, b, :].T @ 重建分块[:, b, :]
                U, _, Vt = np.linalg.svd(协方差)
                旋转块[b] = (U @ Vt).astype(np.float32)
        return 旋转块
    def _劳埃德最大化_路由(self, 数据, 聚类数, 迭代次数):
        中心 = np.percentile(数据, np.linspace(0, 100, 聚类数+2)[1:-1]).astype(np.float32)
        for _ in range(迭代次数):
            边界 = (中心[:-1] + 中心[1:]) / 2.0
            量化索引 = np.searchsorted(边界, 数据).astype(np.intp)
            计数 = np.bincount(量化索引, minlength=聚类数)
            加权和 = np.bincount(量化索引, weights=数据, minlength=聚类数)
            有效 = 计数 > 0
            新中心 = np.where(有效, 加权和 / np.maximum(计数, 1), 中心).astype(np.float32)
            if np.max(np.abs(新中心 - 中心)) < 1e-5:
                中心 = 新中心
                break
            中心 = 新中心
        中心 = np.sort(中心)
        return 中心, (中心[:-1] + 中心[1:]) / 2.0
    def train(self, 训练数据):
        self.旋转块矩阵 = self._SVD学习最优旋转_分块(训练数据, 迭代次数=self.Config.INDEX_GSQ_MOE_SPL_SVD, 初始中心=None)
        旋转后 = self._应用分块旋转(训练数据, self.旋转块矩阵)
        展平 = 旋转后.reshape(-1, 32)
        高位, 低位 = np.percentile(展平, [99.9, 0.1], axis=1, keepdims=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.maximum(np.std(展平, axis=1, ddof=0, keepdims=True), 1e-8)
        归一化 = 展平 / 标准差
        self.LM中心, self.LM边界 = self._劳埃德最大化_路由(归一化.ravel(), self.Config.INDEX_GSQ_MOE_KM_LM, self.Config.INDEX_GSQ_MOE_SPL_LM)
        self.旋转块矩阵 = self._SVD学习最优旋转_分块(训练数据, 迭代次数=self.Config.INDEX_GSQ_MOE_SPL_SVD, 初始中心=self.LM中心)
        旋转后 = self._应用分块旋转(训练数据, self.旋转块矩阵)
        展平 = 旋转后.reshape(-1, 32)
        高位, 低位 = np.percentile(展平, [99.9, 0.1], axis=1, keepdims=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.maximum(np.std(展平, axis=1, ddof=0, keepdims=True), 1e-8)
        归一化 = 展平 / 标准差
        self.LM中心, self.LM边界 = self._劳埃德最大化_路由(归一化.ravel(), self.Config.INDEX_GSQ_MOE_KM_LM, self.Config.INDEX_GSQ_MOE_SPL_LM)
        self.已训练 = True
    def _编码路由数据(self, 均值矩阵):
        行数, 维度 = 均值矩阵.shape
        旋转后 = self._应用分块旋转(均值矩阵, self.旋转块矩阵)
        展平 = 旋转后.reshape(-1, 32)
        高位, 低位 = np.percentile(展平, [99.9, 0.1], axis=1, keepdims=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.maximum(np.std(展平, axis=1, ddof=0), 1e-8).astype(np.float32)
        归一化 = 展平 / 标准差[:, None]
        量化索引 = np.searchsorted(self.LM边界, 归一化).astype(np.uint8)
        最大缩放 = max(float(np.max(标准差)), 1e-8)
        return {
            "PackedVector": 加速打包4(量化索引.ravel()),
            "Scale": (标准差 / 最大缩放).astype(np.float16), 
            "MaxScale": 最大缩放,
            "Shape": (行数, 维度)
        }
    def add(self, vectors):
        if not self.已训练:
            self.train(vectors)
        重排数组, self.映射表 = 向量重排(vectors, self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, self.Config.INDEX_GSQ_RERANKER_FACTOR)
        最大量级 = (1 << self.位深) - 1
        总行数 = len(重排数组)
        维度 = 重排数组.shape[1]
        所有块均值 = []
        for 起始 in range(0, 总行数, self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE):
            结束 = min(起始 + self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, 总行数)
            块数组 = 重排数组[起始:结束]
            所有块均值.append(np.mean(块数组, axis=0))
            量化值, 最小编码, 缩放编码, 最大最小, 最大缩放 = _GSQ_K编码_Numba(块数组, self.Config.INDEX_GSQ_BLOCK_SIZE, 最大量级)
            if self.位深 == 8: 压缩 = 量化值
            elif self.位深 == 6: 压缩 = 加速打包6(量化值)
            elif self.位深 == 4: 压缩 = 加速打包4(量化值)
            elif self.位深 == 3: 压缩 = 加速打包3(量化值)
            elif self.位深 == 2: 压缩 = 加速打包2(量化值)
            self.向量库.append({
                "packed": 压缩, "mins": 最小编码, "scales": 缩放编码, 
                "max_min": 最大最小, "max_scale": 最大缩放, 
                "shape": (结束 - 起始, 维度), "bit_depth": self.位深, 
                "vec_block": self.Config.INDEX_GSQ_BLOCK_SIZE
            })
        if 所有块均值:
            所有块均值矩阵 = np.vstack(所有块均值).astype(np.float32)
            self.路由数据 = self._编码路由数据(所有块均值矩阵)
        self.exp = max(1, int(self.Config.INDEX_GSQ_MOE_EXP) if isinstance(self.Config.INDEX_GSQ_MOE_EXP, np.uint32) or self.Config.INDEX_GSQ_MOE_EXP > 1 else int(len(self.向量库) * self.Config.INDEX_GSQ_MOE_EXP))
        self.日志("log.index.gsq.moe.exp", info_level=0, count=len(self.向量库), exp=self.exp)
    def Q4_SVD_LM反量化_路由(self, 数据字典):
        行数, 维度 = 数据字典["Shape"]
        量化索引 = 加速解包4(数据字典["PackedVector"], 行数 * 维度).reshape(行数, 维度 // 32, 32)
        标准差 = 数据字典["Scale"].astype(np.float32).reshape(行数, 维度 // 32, 1) * 数据字典["MaxScale"]
        重建 = self.LM中心[量化索引] * 标准差
        return 重建.reshape(行数, 维度)
    def search(self, query, k, nprobe=None):
        if nprobe is None: nprobe = self.exp
        if isinstance(self.映射表, list): self.映射表 = np.array(self.映射表)
        查询矩阵 = np.atleast_2d(query).astype(np.float32)
        查询数量 = 查询矩阵.shape[0]
        查询范数 = np.linalg.norm(查询矩阵, axis=1, keepdims=True)
        查询范数[查询范数 < 1e-8] = 1e-8
        查询归一 = 查询矩阵 / 查询范数
        总目标数 = sum(包["shape"][0] for 包 in self.向量库)
        总块数 = len(self.向量库)
        实际_k = min(k, 总目标数)
        if 实际_k <= 0:
            return np.empty((查询数量, 0), dtype=np.float32), np.empty((查询数量, 0), dtype=np.int64)
        查询旋转 = self._应用分块旋转(查询归一, self.旋转块矩阵)
        centroids_rot = self.Q4_SVD_LM反量化_路由(self.路由数据) 
        路由分数 = 查询旋转 @ centroids_rot.T  
        del centroids_rot, 查询旋转
        actual_nprobe = min(nprobe, 总块数)
        激活的块ID矩阵 = np.argpartition(路由分数, -actual_nprobe, axis=1)[:, -actual_nprobe:]
        唯一激活块ID = np.unique(激活的块ID矩阵)
        del 路由分数
        TopK分数 = np.full((查询数量, 实际_k), -np.inf, dtype=np.float32)
        TopK索引 = np.full((查询数量, 实际_k), -1, dtype=np.int64)
        当前全局偏移 = 0
        for 块ID, 数据包 in enumerate(self.向量库):
            行数, 维度 = 数据包["shape"]
            if 块ID not in 唯一激活块ID:
                当前全局偏移 += 行数
                continue 
            块大小 = 数据包["vec_block"]
            组数 = (行数 + 块大小 - 1) // 块大小
            缩放值 = (np.asarray(数据包["scales"]).view(np.int16).astype(np.float32) / 32768.0 * 数据包["max_scale"]).reshape(组数, 维度)
            最小值 = (np.asarray(数据包["mins"]).view(np.int16).astype(np.float32) / 32768.0 * 数据包["max_min"]).reshape(组数, 维度)
            位深 = 数据包["bit_depth"]
            量化值 = _解包(数据包, 位深, 行数, 维度)
            块矩阵 = 批量反量化(量化值, 缩放值, 最小值, (1 << 位深) - 1, 块大小, 维度)
            del 量化值
            块范数 = 数据包.get("norms")
            if 块范数 is None:
                块范数 = np.linalg.norm(块矩阵, axis=1)
                块范数[块范数 < 1e-8] = 1e-8
            点积矩阵 = 查询归一 @ 块矩阵.T
            当前块分数 = 点积矩阵 / 块范数[np.newaxis, :]
            del 块矩阵, 点积矩阵
            if 行数 > 实际_k:
                块内TopK位置 = np.argpartition(当前块分数, -实际_k, axis=1)[:, -实际_k:]
                块内TopK分数 = np.take_along_axis(当前块分数, 块内TopK位置, axis=1)
                块内局部索引 = np.arange(当前全局偏移, 当前全局偏移 + 行数, dtype=np.int64)
                块内TopK索引 = np.take_along_axis(np.tile(块内局部索引, (查询数量, 1)), 块内TopK位置, axis=1)
            else:
                块内TopK分数 = 当前块分数
                块内TopK索引 = np.tile(np.arange(当前全局偏移, 当前全局偏移 + 行数, dtype=np.int64), (查询数量, 1))
            合并分数 = np.hstack([TopK分数, 块内TopK分数])
            合并索引 = np.hstack([TopK索引, 块内TopK索引])
            最终TopK位置 = np.argpartition(合并分数, -实际_k, axis=1)[:, -实际_k:]
            TopK分数 = np.take_along_axis(合并分数, 最终TopK位置, axis=1)
            TopK索引 = np.take_along_axis(合并索引, 最终TopK位置, axis=1)
            当前全局偏移 += 行数
        排序顺序 = np.argsort(-TopK分数, axis=1)
        TopK分数 = np.take_along_axis(TopK分数, 排序顺序, axis=1)
        TopK索引 = np.take_along_axis(TopK索引, 排序顺序, axis=1)
        原始TopK索引 = np.full_like(TopK索引, -1, dtype=np.int64)
        valid_mask = TopK索引 != -1
        原始TopK索引[valid_mask] = self.映射表[TopK索引[valid_mask]]
        if k > 总目标数:
            填充索引 = np.full((查询数量, k - 总目标数), -1, dtype=np.int64)
            填充分数 = np.full((查询数量, k - 总目标数), -np.inf, dtype=np.float32)
            原始TopK索引 = np.hstack([原始TopK索引, 填充索引])
            TopK分数 = np.hstack([TopK分数, 填充分数])
        return TopK分数.astype(np.float32), 原始TopK索引
class IndexGSQKCosineMoEPlus:
    def __init__(self, app=None, quantization: int = 2):
        if app:
            self.日志 = app.日志
            self.Config = app.Config
        self.向量库 = []
        self.映射表 = []
        self.位深 = quantization
        self.模式 = "IndexGSQKCosineMoEPlus"
        self.路由数据 = None  
        self.exp = 32 
        self.已训练 = False
        self.旋转块矩阵 = None
        self.LM中心 = None      
        self.LM边界 = None
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                "模式": self.模式,
                "向量库": self.向量库,
                "映射表": self.映射表,
                "位深": self.位深,
                "路由数据": self.路由数据,
                "exp": self.exp,
                "已训练": self.已训练,
                "旋转块矩阵": self.旋转块矩阵.astype(np.float16),
                "LM中心": self.LM中心,
                "LM边界": self.LM边界
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    def _应用分块旋转(self, 数组, 旋转块矩阵):
        N, D = 数组.shape
        n块 = D // 16
        分块 = 数组.reshape(N, n块, 16)
        旋转后 = np.einsum('nbi,bij->nbj', 分块, 旋转块矩阵, optimize=True)
        return 旋转后.reshape(N, D)
    def _SVD学习最优旋转_分块(self, 数据, 迭代次数=5, 初始中心=None):
        行数, 维度 = 数据.shape
        n块 = 维度 // 16
        旋转块 = np.tile(np.eye(16, dtype=np.float32), (n块, 1, 1))
        中心 = 初始中心 if 初始中心 is not None else np.linspace(-2.5, 2.5, 16).astype(np.float32)
        边界 = (中心[:-1] + 中心[1:]) / 2.0
        Q块大小 = self.Config.INDEX_GSQ_MOE_BLOCK_SIZE
        for _ in range(迭代次数):
            旋转后 = self._应用分块旋转(数据, 旋转块)
            展平 = 旋转后.reshape(-1, Q块大小)
            高位, 低位 = np.percentile(展平, [99.9, 0.1], axis=1, keepdims=True)
            np.clip(展平, 低位, 高位, out=展平)
            标准差 = np.maximum(np.std(展平, axis=1, ddof=0, keepdims=True), 1e-8)
            归一化 = 展平 / 标准差
            量化索引 = np.searchsorted(边界, 归一化).astype(np.uint8)
            重建 = (中心[量化索引] * 标准差).reshape(行数, 维度)
            重建分块 = 重建.reshape(行数, n块, 16)
            数据分块 = 数据.reshape(行数, n块, 16)
            for b in range(n块):
                协方差 = 数据分块[:, b, :].T @ 重建分块[:, b, :]
                U, _, Vt = np.linalg.svd(协方差)
                旋转块[b] = (U @ Vt).astype(np.float32)
        return 旋转块
    def _劳埃德最大化_路由(self, 数据, 聚类数, 迭代次数):
        中心 = np.percentile(数据, np.linspace(0, 100, 聚类数+2)[1:-1]).astype(np.float32)
        for _ in range(迭代次数):
            边界 = (中心[:-1] + 中心[1:]) / 2.0
            量化索引 = np.searchsorted(边界, 数据).astype(np.intp)
            计数 = np.bincount(量化索引, minlength=聚类数)
            加权和 = np.bincount(量化索引, weights=数据, minlength=聚类数)
            有效 = 计数 > 0
            新中心 = np.where(有效, 加权和 / np.maximum(计数, 1), 中心).astype(np.float32)
            if np.max(np.abs(新中心 - 中心)) < 1e-5:
                中心 = 新中心
                break
            中心 = 新中心
        中心 = np.sort(中心)
        return 中心, (中心[:-1] + 中心[1:]) / 2.0
    def train(self, 训练数据):
        self.旋转块矩阵 = self._SVD学习最优旋转_分块(训练数据, 迭代次数=self.Config.INDEX_GSQ_MOE_SPL_SVD, 初始中心=None)
        旋转后 = self._应用分块旋转(训练数据, self.旋转块矩阵)
        展平 = 旋转后.reshape(-1, 32)
        高位, 低位 = np.percentile(展平, [99.9, 0.1], axis=1, keepdims=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.maximum(np.std(展平, axis=1, ddof=0, keepdims=True), 1e-8)
        归一化 = 展平 / 标准差
        self.LM中心, self.LM边界 = self._劳埃德最大化_路由(归一化.ravel(), self.Config.INDEX_GSQ_MOE_KM_LM, self.Config.INDEX_GSQ_MOE_SPL_LM)
        self.旋转块矩阵 = self._SVD学习最优旋转_分块(训练数据, 迭代次数=self.Config.INDEX_GSQ_MOE_SPL_SVD, 初始中心=self.LM中心)
        旋转后 = self._应用分块旋转(训练数据, self.旋转块矩阵)
        展平 = 旋转后.reshape(-1, 32)
        高位, 低位 = np.percentile(展平, [99.9, 0.1], axis=1, keepdims=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.maximum(np.std(展平, axis=1, ddof=0, keepdims=True), 1e-8)
        归一化 = 展平 / 标准差
        self.LM中心, self.LM边界 = self._劳埃德最大化_路由(归一化.ravel(), self.Config.INDEX_GSQ_MOE_KM_LM, self.Config.INDEX_GSQ_MOE_SPL_LM)
        self.已训练 = True
    def _编码路由数据(self, 均值矩阵):
        行数, 维度 = 均值矩阵.shape
        旋转后 = self._应用分块旋转(均值矩阵, self.旋转块矩阵)
        展平 = 旋转后.reshape(-1, 32)
        高位, 低位 = np.percentile(展平, [99.9, 0.1], axis=1, keepdims=True)
        np.clip(展平, 低位, 高位, out=展平)
        标准差 = np.maximum(np.std(展平, axis=1, ddof=0), 1e-8).astype(np.float32)
        归一化 = 展平 / 标准差[:, None]
        量化索引 = np.searchsorted(self.LM边界, 归一化).astype(np.uint8)
        最大缩放 = max(float(np.max(标准差)), 1e-8)
        return {
            "PackedVector": 加速打包4(量化索引.ravel()),
            "Scale": (标准差 / 最大缩放).astype(np.float16), 
            "MaxScale": 最大缩放,
            "Shape": (行数, 维度)
        }
    def add(self, vectors):
        if not self.已训练:
            self.train(vectors)
        重排数组, self.映射表 = 向量重排(vectors, self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, self.Config.INDEX_GSQ_RERANKER_FACTOR)
        最大量级 = (1 << self.位深) - 1
        总行数 = len(重排数组)
        维度 = 重排数组.shape[1]
        所有代表向量 = []
        for 起始 in range(0, 总行数, self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE):
            结束 = min(起始 + self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, 总行数)
            块数组 = 重排数组[起始:结束]
            所有代表向量.append(块数组[0])
            if 块数组.shape[0] > 1:
                所有代表向量.append(块数组[-1])
            else:
                所有代表向量.append(块数组[0])
            量化值, 最小编码, 缩放编码, 最大最小, 最大缩放 = _GSQ_K编码_Numba(块数组, self.Config.INDEX_GSQ_BLOCK_SIZE, 最大量级)
            if self.位深 == 8: 压缩 = 量化值
            elif self.位深 == 6: 压缩 = 加速打包6(量化值)
            elif self.位深 == 4: 压缩 = 加速打包4(量化值)
            elif self.位深 == 3: 压缩 = 加速打包3(量化值)
            elif self.位深 == 2: 压缩 = 加速打包2(量化值)
            self.向量库.append({
                "packed": 压缩, "mins": 最小编码, "scales": 缩放编码, 
                "max_min": 最大最小, "max_scale": 最大缩放, 
                "shape": (结束 - 起始, 维度), "bit_depth": self.位深, 
                "vec_block": self.Config.INDEX_GSQ_BLOCK_SIZE
            })
        if 所有代表向量:
            代表向量矩阵 = np.vstack(所有代表向量).astype(np.float32)
            self.路由数据 = self._编码路由数据(代表向量矩阵)
        self.exp = max(1, int(self.Config.INDEX_GSQ_MOE_EXP) if isinstance(self.Config.INDEX_GSQ_MOE_EXP, np.uint32) or self.Config.INDEX_GSQ_MOE_EXP > 1 else int(len(self.向量库) * self.Config.INDEX_GSQ_MOE_EXP))
        self.日志("log.index.gsq.moe.exp", info_level=0, count=len(self.向量库)*2, exp=self.exp*2)
    def Q4_SVD_LM反量化_路由(self, 数据字典):
        行数, 维度 = 数据字典["Shape"]
        量化索引 = 加速解包4(数据字典["PackedVector"], 行数 * 维度).reshape(行数, 维度 // 32, 32)
        标准差 = 数据字典["Scale"].astype(np.float32).reshape(行数, 维度 // 32, 1) * 数据字典["MaxScale"]
        重建 = self.LM中心[量化索引] * 标准差
        return 重建.reshape(行数, 维度)
    def search(self, query, k, nprobe=None):
        if nprobe is None: nprobe = self.exp
        if isinstance(self.映射表, list): self.映射表 = np.array(self.映射表)
        查询矩阵 = np.atleast_2d(query).astype(np.float32)
        查询数量 = 查询矩阵.shape[0]
        查询范数 = np.linalg.norm(查询矩阵, axis=1, keepdims=True)
        查询范数[查询范数 < 1e-8] = 1e-8
        查询归一 = 查询矩阵 / 查询范数
        总目标数 = sum(包["shape"][0] for 包 in self.向量库)
        总块数 = len(self.向量库)
        实际_k = min(k, 总目标数)
        if 实际_k <= 0:
            return np.empty((查询数量, 0), dtype=np.float32), np.empty((查询数量, 0), dtype=np.int64)
        查询旋转 = self._应用分块旋转(查询归一, self.旋转块矩阵)
        centroids_rot = self.Q4_SVD_LM反量化_路由(self.路由数据) 
        路由分数_平铺 = 查询旋转 @ centroids_rot.T  
        del centroids_rot, 查询旋转
        路由分数_分块 = 路由分数_平铺.reshape(查询数量, 总块数, 2)
        块路由分数 = np.max(路由分数_分块, axis=2)
        del 路由分数_平铺, 路由分数_分块
        actual_nprobe = min(nprobe, 总块数)
        激活的块ID矩阵 = np.argpartition(块路由分数, -actual_nprobe, axis=1)[:, -actual_nprobe:]
        唯一激活块ID = np.unique(激活的块ID矩阵)
        del 块路由分数
        TopK分数 = np.full((查询数量, 实际_k), -np.inf, dtype=np.float32)
        TopK索引 = np.full((查询数量, 实际_k), -1, dtype=np.int64)
        当前全局偏移 = 0
        for 块ID, 数据包 in enumerate(self.向量库):
            行数, 维度 = 数据包["shape"]
            if 块ID not in 唯一激活块ID:
                当前全局偏移 += 行数
                continue 
            块大小 = 数据包["vec_block"]
            组数 = (行数 + 块大小 - 1) // 块大小
            缩放值 = (np.asarray(数据包["scales"]).view(np.int16).astype(np.float32) / 32768.0 * 数据包["max_scale"]).reshape(组数, 维度)
            最小值 = (np.asarray(数据包["mins"]).view(np.int16).astype(np.float32) / 32768.0 * 数据包["max_min"]).reshape(组数, 维度)
            位深 = 数据包["bit_depth"]
            量化值 = _解包(数据包, 位深, 行数, 维度)
            块矩阵 = 批量反量化(量化值, 缩放值, 最小值, (1 << 位深) - 1, 块大小, 维度)
            del 量化值
            块范数 = 数据包.get("norms")
            if 块范数 is None:
                块范数 = np.linalg.norm(块矩阵, axis=1)
                块范数[块范数 < 1e-8] = 1e-8
            点积矩阵 = 查询归一 @ 块矩阵.T
            当前块分数 = 点积矩阵 / 块范数[np.newaxis, :]
            del 块矩阵, 点积矩阵
            if 行数 > 实际_k:
                块内TopK位置 = np.argpartition(当前块分数, -实际_k, axis=1)[:, -实际_k:]
                块内TopK分数 = np.take_along_axis(当前块分数, 块内TopK位置, axis=1)
                块内局部索引 = np.arange(当前全局偏移, 当前全局偏移 + 行数, dtype=np.int64)
                块内TopK索引 = np.take_along_axis(np.tile(块内局部索引, (查询数量, 1)), 块内TopK位置, axis=1)
            else:
                块内TopK分数 = 当前块分数
                块内TopK索引 = np.tile(np.arange(当前全局偏移, 当前全局偏移 + 行数, dtype=np.int64), (查询数量, 1))
            合并分数 = np.hstack([TopK分数, 块内TopK分数])
            合并索引 = np.hstack([TopK索引, 块内TopK索引])
            最终TopK位置 = np.argpartition(合并分数, -实际_k, axis=1)[:, -实际_k:]
            TopK分数 = np.take_along_axis(合并分数, 最终TopK位置, axis=1)
            TopK索引 = np.take_along_axis(合并索引, 最终TopK位置, axis=1)
            当前全局偏移 += 行数
        排序顺序 = np.argsort(-TopK分数, axis=1)
        TopK分数 = np.take_along_axis(TopK分数, 排序顺序, axis=1)
        TopK索引 = np.take_along_axis(TopK索引, 排序顺序, axis=1)
        原始TopK索引 = np.full_like(TopK索引, -1, dtype=np.int64)
        valid_mask = TopK索引 != -1
        原始TopK索引[valid_mask] = self.映射表[TopK索引[valid_mask]]
        if k > 总目标数:
            填充索引 = np.full((查询数量, k - 总目标数), -1, dtype=np.int64)
            填充分数 = np.full((查询数量, k - 总目标数), -np.inf, dtype=np.float32)
            原始TopK索引 = np.hstack([原始TopK索引, 填充索引])
            TopK分数 = np.hstack([TopK分数, 填充分数])
        return TopK分数.astype(np.float32), 原始TopK索引
def load(filename: str):
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    index = globals()[d["模式"]]()
    for k, v in d.items():
        setattr(index, k, v)
    return index
def read_index(filename: str):
    return load(filename)
def write_index(index, filename: str):
    index.save(filename)