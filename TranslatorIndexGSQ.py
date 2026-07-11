from TranslatorLib import njit, pickle, numba, np as np, faiss
_prange = numba.prange if numba is not None else range

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
def 向量重排(数组, 聚类块大小, 重排乘数, 构建索引函数):
    向量数量, 向量维度 = 数组.shape
    if 向量数量 <= 聚类块大小:
        return 数组, np.arange(向量数量)
    数组_归一 = np.ascontiguousarray(数组, dtype=np.float32).copy()
    faiss.normalize_L2(数组_归一)
    向量索引 = 构建索引函数(数组_归一, 向量重排模式=True)
    try:
        向量索引.set_direct_map_type(faiss.DirectMap.Hashtable)
    except (AttributeError, RuntimeError):
        pass
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
        try:
            向量索引.remove_ids(delete_ids)
        except RuntimeError:
            pass
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
    for g in _prange(组数):
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
def 编码(self, 数组, 位深, 日志, 返回重排数组=False):
    重排数组, self.映射表 = 向量重排(数组, self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, self.Config.INDEX_GSQ_RERANKER_FACTOR, self.构建索引)
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
        elif 位深 == 5:
            压缩 = 加速打包5(量化值)
        elif 位深 == 4:
            压缩 = 加速打包4(量化值)
        elif 位深 == 3:
            压缩 = 加速打包3(量化值)
        elif 位深 == 2:
            压缩 = 加速打包2(量化值)
        块行数 = 结束 - 起始
        组数 = (块行数 + self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE - 1) // self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE
        缩放值_解码 = (np.asarray(缩放编码).view(np.int16).astype(np.float32) / 32768.0 * 最大缩放).reshape(组数, 维度)
        最小值_解码 = (np.asarray(最小编码).view(np.int16).astype(np.float32) / 32768.0 * 最大最小).reshape(组数, 维度)
        预计算范数 = 预计算范数LUT(量化值, 缩放值_解码, 最小值_解码, 最大量级, self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, 维度)
        try:
            self.向量库.append({"packed": 压缩, "mins": 最小编码, "scales": 缩放编码, "max_min": 最大最小, "max_scale": 最大缩放, "shape": (块行数, 维度), "bit_depth": 位深, "vec_block": self.Config.INDEX_GSQ_RERANKER_BLOCK_SIZE, "norms": 预计算范数})
        except UnboundLocalError:
            日志("log.index.gsq.quantization.err", info_level=3)
    if 返回重排数组:
        return 重排数组
def 重排列表(原始列表, 映射表):
    if 原始列表 is None:
        return None
    return [原始列表[i] for i in 映射表]
def _解包(数据包, 位深, 行数, 维度):
    if 位深 == 8:
        return 数据包["packed"]
    elif 位深 == 6:
        return 加速解包6(数据包["packed"], 行数 * 维度)
    elif 位深 == 5:
        return 加速解包5(数据包["packed"], 行数 * 维度)
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
    for q in _prange(查询数):
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
def PCA降维(数据, PCA_DIM):
    if PCA_DIM == -1:
        return 数据, None, None
    目标维度 = int(PCA_DIM)
    if 目标维度 >= 数据.shape[1]:
        return 数据, None, None
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
class IndexGSQKCosineFast:
    def __init__(self, app = None, quantization: int = 2): # Index
        if app:
            self.日志 = app.日志
            self.Config = app.Config
            self.构建索引 = app.构建索引
        self.向量库 = []
        self.映射表 = []
        self.位深 = quantization
        self.模式 = "IndexGSQKCosineFast"
        self.PCA_均值 = None
        self.PCA_主成分 = None
    def cpu(self):
        if hasattr(self.映射表, "get"): self.映射表 = self.映射表.get()
        for 包 in self.向量库:
            for k in ["packed", "mins", "scales", "norms"]:
                if k in 包 and hasattr(包[k], "get"): 包[k] = 包[k].get()
            for k in ["max_min", "max_scale"]:
                if k in 包 and hasattr(包[k], "get"):
                    val = 包[k].get()
                    包[k] = float(val) if hasattr(val, "item") else val
            if "mrl" in 包 and hasattr(包["mrl"], "get"): 包["mrl"] = 包["mrl"].get()
        for attr in ["PCA_均值", "PCA_主成分"]:
            val = getattr(self, attr, None)
            if val is not None and hasattr(val, "get"): setattr(self, attr, val.get())
    def gpu(self):
        if self.映射表 is not None and not hasattr(self.映射表, "get"): self.映射表 = np.asarray(self.映射表)
        for 包 in self.向量库:
            for k in ["packed", "mins", "scales", "norms"]:
                if k in 包 and not hasattr(包[k], "get"): 包[k] = np.asarray(包[k])
            for k in ["max_min", "max_scale"]:
                if k in 包 and not hasattr(包[k], "get"): 包[k] = np.asarray(包[k])
            if "mrl" in 包 and not hasattr(包["mrl"], "get"): 包["mrl"] = np.asarray(包["mrl"])
        for attr in ["PCA_均值", "PCA_主成分"]:
            val = getattr(self, attr, None)
            if val is not None and not hasattr(val, "get"): setattr(self, attr, np.asarray(val))
    def train(self, vectors):
        if self.Config.INDEX_GSQ_PCA_DIM != -1:
            self.PCA_均值, self.PCA_主成分 = PCA降维(vectors, self.Config.INDEX_GSQ_PCA_DIM)
    def add(self, vectors):
        if self.PCA_主成分 is not None:
            vectors = np.dot(vectors - self.PCA_均值, self.PCA_主成分.T).astype(np.float32)
        编码(self, vectors, self.位深, self.日志)
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                "模式": self.模式,
                "向量库": self.向量库,
                "映射表": self.映射表,
                "位深": self.位深,
                "PCA_均值": self.PCA_均值,
                "PCA_主成分": self.PCA_主成分,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    def search(self, query, k):
        查询矩阵 = np.atleast_2d(query).astype(np.float32)
        if self.PCA_主成分 is not None:
            查询矩阵 = np.dot(查询矩阵 - self.PCA_均值, self.PCA_主成分.T).astype(np.float32)
        查询数量 = 查询矩阵.shape[0]
        查询范数 = np.linalg.norm(查询矩阵, axis=1, keepdims=True)
        查询范数[查询范数 < 1e-8] = 1e-8
        查询归一 = 查询矩阵 / 查询范数
        总目标数 = sum(包["shape"][0] for 包 in self.向量库)
        实际_k = min(k, 总目标数)
        if 实际_k <= 0:
            return np.empty((查询数量, 0), dtype=np.float32), np.empty((查询数量, 0), dtype=np.int64)
        TopK分数 = np.full((查询数量, 实际_k), -np.inf, dtype=np.float32)
        TopK索引 = np.full((查询数量, 实际_k), -1, dtype=np.int64)
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
            块范数 = 数据包["norms"]
            点积矩阵 = 查询归一 @ 块矩阵.T
            del 块矩阵
            当前块分数 = 点积矩阵 / 块范数[np.newaxis, :]
            del 点积矩阵
            if 行数 > 实际_k:
                块内TopK位置 = np.argpartition(当前块分数, -实际_k, axis=1)[:, -实际_k:]
                块内TopK分数 = np.take_along_axis(当前块分数, 块内TopK位置, axis=1)
                块内局部索引 = np.arange(偏移, 偏移 + 行数, dtype=np.int64)
                块内TopK索引 = np.take_along_axis(np.tile(块内局部索引, (查询数量, 1)), 块内TopK位置, axis=1)
            else:
                块内TopK分数 = 当前块分数
                块内TopK索引 = np.tile(np.arange(偏移, 偏移 + 行数, dtype=np.int64), (查询数量, 1))
            合并分数 = np.hstack([TopK分数, 块内TopK分数])
            合并索引 = np.hstack([TopK索引, 块内TopK索引])
            最终TopK位置 = np.argpartition(合并分数, -实际_k, axis=1)[:, -实际_k:]
            TopK分数 = np.take_along_axis(合并分数, 最终TopK位置, axis=1)
            TopK索引 = np.take_along_axis(合并索引, 最终TopK位置, axis=1)
            偏移 += 行数
        排序顺序 = np.argsort(-TopK分数, axis=1)
        TopK分数 = np.take_along_axis(TopK分数, 排序顺序, axis=1)
        TopK索引 = np.take_along_axis(TopK索引, 排序顺序, axis=1)
        原始TopK索引 = np.take(self.映射表, TopK索引).astype(np.int64)
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
def index_cpu_to_gpu(index, **kwargs):
    index.gpu()
    return index
def index_gpu_to_cpu(index, **kwargs):
    index.cpu()
    return index