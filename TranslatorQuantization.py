import threading
from TranslatorLib import (np, math, numpy, njit, CPU_ACC, GPU_ACC, CleanVRAM, faiss,
                           Config, Mods)
# ↓ 缩放通道(块缩放)的编解码深度: 量化是多线程跑的, 深度必须是线程局部, 不能是全局变量
缩放线程状态 = threading.local()
# ↓ 内层类型(默认 Float16_E0M15)自己也带二级缩放时, 再深一层强制用它, 彻底断掉自递归
缩放终极回退类型 = "Float16_E0M15"
@njit(cache=True)
def 加速打包2(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 3) >> 2
    输出 = numpy.zeros(输出长度, dtype=numpy.uint8)
    for 索 in range(数量):
        输出[索 >> 2] |= numpy.uint8(量化值[索] << ((索 & 3) << 1))
    return 输出
@njit(cache=True)
def 加速解包2(压缩, 数量):
    输出 = numpy.empty(数量, dtype=numpy.uint8)
    for 索 in range(数量):
        字节索 = 索 >> 2
        if 字节索 < len(压缩):
            输出[索] = numpy.uint8((压缩[字节索] >> ((索 & 3) << 1)) & 3)
        else:
            输出[索] = numpy.uint8(0)
    return 输出
@njit(cache=True)
def 加速打包4(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 1) >> 1
    输出 = numpy.zeros(输出长度, dtype=numpy.uint8)
    for 索 in range(数量):
        if 索 & 1 == 0:
            输出[索 >> 1] = numpy.uint8(量化值[索] << 4)
        else:
            输出[索 >> 1] |= numpy.uint8(量化值[索] & 0xF)
    return 输出
@njit(cache=True)
def 加速解包4(压缩, 数量):
    输出 = numpy.empty(数量, dtype=numpy.uint8)
    for 索 in range(数量):
        字节索 = 索 >> 1
        if 字节索 < len(压缩):
            if 索 & 1 == 0:
                输出[索] = numpy.uint8((压缩[字节索] >> 4) & 0xF)
            else:
                输出[索] = numpy.uint8(压缩[字节索] & 0xF)
        else:
            输出[索] = numpy.uint8(0)
    return 输出
@njit(cache=True)
def 加速线性量化(数据块, 缩放, 零点, 最大值):
    块数 = 数据块.shape[0]; 块大小 = 数据块.shape[1]
    输出 = numpy.empty((块数, 块大小), dtype=numpy.uint8)
    for 行 in range(块数):
        当前缩放 = 缩放[行]; 当前零点 = 零点[行]
        for 列 in range(块大小):
            值 = 数据块[行, 列] / 当前缩放 + 当前零点
            值 = numpy.float32(round(值))
            if 值 < 0.0: 值 = 0.0
            elif 值 > 最大值: 值 = 最大值
            输出[行, 列] = numpy.uint8(值)
    return 输出
@njit(cache=True)
def 加速线性反量化(量化数组, 缩放, 零点):
    块数 = 量化数组.shape[0]; 块大小 = 量化数组.shape[1]
    输出 = numpy.empty((块数, 块大小), dtype=numpy.float32)
    for 行 in range(块数):
        当前缩放 = 缩放[行]; 当前零点 = 零点[行]
        for 列 in range(块大小):
            输出[行, 列] = (numpy.float32(量化数组[行, 列]) - 当前零点) * 当前缩放
    return 输出
@njit(cache=True)
def 加速NF量化(数据块, 边界, 缩放):
    块数 = 数据块.shape[0]; 块大小 = 数据块.shape[1]; 边界数 = len(边界)
    索引 = numpy.empty((块数, 块大小), dtype=numpy.uint8)
    for 行 in range(块数):
        当前缩放 = 缩放[行]
        for 列 in range(块大小):
            值 = 数据块[行, 列] / 当前缩放
            等级 = numpy.uint8(0)
            for 步 in range(边界数):
                if 值 >= 边界[步]: 等级 += numpy.uint8(1)
            索引[行, 列] = 等级
    return 索引
@njit(cache=True)
def 加速FWHT原地(数组):
    块数 = 数组.shape[0]; 长度 = 数组.shape[1]; 步长 = numpy.int64(1)
    while 步长 < 长度:
        跨度 = numpy.int64(2) * 步长
        for 行 in range(块数):
            for 起始 in range(0, 长度, 跨度):
                for 列 in range(起始, 起始 + 步长):
                    左值 = 数组[行, 列]; 右值 = 数组[行, 列 + 步长]
                    数组[行, 列] = 左值 + 右值; 数组[行, 列 + 步长] = 左值 - 右值
        步长 = 跨度
@njit(cache=True)
def 加速查找alpha(样本, alpha列表):
    最佳 = alpha列表[0]; 最佳误差 = numpy.float32(1e30)
    数量 = len(样本); 列表长度 = len(alpha列表)
    for 索 in range(列表长度):
        当前alpha = alpha列表[索]; 误差 = numpy.float32(0.0)
        for 样索 in range(数量):
            值 = 样本[样索]
            if 值 > 当前alpha: 差值 = 值 - numpy.float32(1.0)
            elif 值 < -当前alpha: 差值 = 值 + numpy.float32(1.0)
            else: 差值 = 值
            误差 += 差值 * 差值
        误差 /= numpy.float32(数量)
        if 误差 < 最佳误差: 最佳误差 = 误差; 最佳 = 当前alpha
    return 最佳
@njit(cache=True)
def 加速SVD处理(展平, 边界, 中心点):
    块数 = 展平.shape[0]; 块大小 = 展平.shape[1]; 边界数 = len(边界)
    重建 = numpy.empty((块数, 块大小), dtype=numpy.float32)
    for 行 in range(块数):
        最大值 = numpy.float32(0.0); 最大索 = 0
        for 列 in range(块大小):
            绝对值 = 展平[行, 列]
            if 绝对值 < 0: 绝对值 = -绝对值
            if 绝对值 > 最大值: 最大值 = 绝对值; 最大索 = 列
        异常值 = 展平[行, 最大索]
        求和 = numpy.float32(0.0); 平方和 = numpy.float32(0.0)
        for 列 in range(块大小):
            值 = 展平[行, 列]
            if 列 == 最大索: 值 = numpy.float32(0.0)
            求和 += 值; 平方和 += 值 * 值
        均值 = 求和 / numpy.float32(块大小)
        方差 = 平方和 / numpy.float32(块大小) - 均值 * 均值
        if 方差 < 0.0: 方差 = 0.0
        标准差 = numpy.float32(numpy.sqrt(numpy.float64(方差)))
        if 标准差 < 1e-8: 标准差 = numpy.float32(1e-8)
        for 列 in range(块大小):
            归一值 = 展平[行, 列] / 标准差
            if 列 == 最大索: 归一值 = numpy.float32(0.0)
            等级 = 0
            for 步 in range(边界数):
                if 归一值 >= 边界[步]: 等级 += 1
            重建[行, 列] = 中心点[等级] * 标准差
        重建[行, 最大索] = 异常值
    return 重建
@njit(cache=True)
def 加速SVD处理_无异常(展平, 边界, 中心点):
    # 与 加速SVD处理 相同, 但不保留异常值: 每块用绕均值std归一化(不减均值), 全部分量都量化重建
    块数 = 展平.shape[0]; 块大小 = 展平.shape[1]; 边界数 = len(边界)
    重建 = numpy.empty((块数, 块大小), dtype=numpy.float32)
    for 行 in range(块数):
        求和 = numpy.float32(0.0); 平方和 = numpy.float32(0.0)
        for 列 in range(块大小):
            值 = 展平[行, 列]
            求和 += 值; 平方和 += 值 * 值
        均值 = 求和 / numpy.float32(块大小)
        方差 = 平方和 / numpy.float32(块大小) - 均值 * 均值
        if 方差 < 0.0: 方差 = 0.0
        标准差 = numpy.float32(numpy.sqrt(numpy.float64(方差)))
        if 标准差 < 1e-8: 标准差 = numpy.float32(1e-8)
        for 列 in range(块大小):
            归一值 = 展平[行, 列] / 标准差
            等级 = 0
            for 步 in range(边界数):
                if 归一值 >= 边界[步]: 等级 += 1
            重建[行, 列] = 中心点[等级] * 标准差
    return 重建
@njit(cache=True)
def 加速劳埃德统计(数据, 边界, 聚类数):
    # 单趟遍历: 二分查找归属bin(等价 np.searchsorted(边界,值,side='left')) 并同时累加 计数 与 加权和
    计数 = numpy.zeros(聚类数, dtype=numpy.int64)
    加权和 = numpy.zeros(聚类数, dtype=numpy.float64)
    边界数 = len(边界)
    for i in range(数据.shape[0]):
        值 = 数据[i]
        低 = 0; 高 = 边界数
        while 低 < 高:
            中 = (低 + 高) >> 1
            if 边界[中] < 值: 低 = 中 + 1
            else: 高 = 中
        计数[低] += 1
        加权和[低] += 值
    return 计数, 加权和
@njit(cache=True)
def 加速三值打包(映射值):
    数量 = len(映射值); 输出长度 = (数量 + 4) // 5
    输出 = numpy.empty(输出长度, dtype=numpy.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 5; 值 = numpy.uint8(0); 倍数 = numpy.uint8(1)
        for 步 in range(5):
            if 基索 + 步 < 数量: 值 += numpy.uint8(映射值[基索 + 步] * 倍数)
            倍数 = numpy.uint8(倍数 * 3)
        输出[块索] = 值
    return 输出
@njit(cache=True)
def 加速三值解包(压缩, 数量):
    输出 = numpy.empty(数量, dtype=numpy.int8)
    for 块索 in range(len(压缩)):
        值 = int(压缩[块索]); 基索 = 块索 * 5
        for 步 in range(5):
            if 基索 + 步 < 数量: 输出[基索 + 步] = numpy.int8(值 % 3)
            值 //= 3
    return 输出
@njit(cache=True)
def 加速打包3(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 7) // 8
    输出 = numpy.zeros(输出长度 * 3, dtype=numpy.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 8; 偏移 = 块索 * 3
        v0 = numpy.uint8(量化值[基索] if 基索 < 数量 else 0)
        v1 = numpy.uint8(量化值[基索+1] if 基索+1 < 数量 else 0)
        v2 = numpy.uint8(量化值[基索+2] if 基索+2 < 数量 else 0)
        v3 = numpy.uint8(量化值[基索+3] if 基索+3 < 数量 else 0)
        v4 = numpy.uint8(量化值[基索+4] if 基索+4 < 数量 else 0)
        v5 = numpy.uint8(量化值[基索+5] if 基索+5 < 数量 else 0)
        v6 = numpy.uint8(量化值[基索+6] if 基索+6 < 数量 else 0)
        v7 = numpy.uint8(量化值[基索+7] if 基索+7 < 数量 else 0)
        输出[偏移]   = numpy.uint8((v0<<5)|(v1<<2)|(v2>>1))
        输出[偏移+1] = numpy.uint8(((v2&1)<<7)|(v3<<4)|(v4<<1)|(v5>>2))
        输出[偏移+2] = numpy.uint8(((v5&3)<<6)|(v6<<3)|v7)
    return 输出

@njit(cache=True)
def 加速解包3(压缩, 数量):
    块数 = len(压缩) // 3; 输出 = numpy.empty(块数 * 8, dtype=numpy.uint8)
    for 块索 in range(块数):
        偏移 = 块索 * 3; 基索 = 块索 * 8
        b0 = 压缩[偏移]; b1 = 压缩[偏移+1]; b2 = 压缩[偏移+2]
        输出[基索]   = numpy.uint8((b0>>5)&7)
        输出[基索+1] = numpy.uint8((b0>>2)&7)
        输出[基索+2] = numpy.uint8(((b0&3)<<1)|(b1>>7))
        输出[基索+3] = numpy.uint8((b1>>4)&7)
        输出[基索+4] = numpy.uint8((b1>>1)&7)
        输出[基索+5] = numpy.uint8(((b1&1)<<2)|(b2>>6))
        输出[基索+6] = numpy.uint8((b2>>3)&7)
        输出[基索+7] = numpy.uint8(b2&7)
    return 输出[:数量]

@njit(cache=True)
def 加速打包6(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 3) // 4
    输出 = numpy.zeros(输出长度 * 3, dtype=numpy.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 4; 偏移 = 块索 * 3
        v0 = numpy.uint8(量化值[基索] if 基索 < 数量 else 0)
        v1 = numpy.uint8(量化值[基索+1] if 基索+1 < 数量 else 0)
        v2 = numpy.uint8(量化值[基索+2] if 基索+2 < 数量 else 0)
        v3 = numpy.uint8(量化值[基索+3] if 基索+3 < 数量 else 0)
        输出[偏移]   = numpy.uint8((v0<<2)|(v1>>4))
        输出[偏移+1] = numpy.uint8(((v1&0xF)<<4)|(v2>>2))
        输出[偏移+2] = numpy.uint8(((v2&3)<<6)|v3)
    return 输出

@njit(cache=True)
def 加速解包6(压缩, 数量):
    块数 = len(压缩) // 3; 输出 = numpy.empty(块数 * 4, dtype=numpy.uint8)
    for 块索 in range(块数):
        偏移 = 块索 * 3; 基索 = 块索 * 4
        b0 = 压缩[偏移]; b1 = 压缩[偏移+1]; b2 = 压缩[偏移+2]
        输出[基索]   = numpy.uint8(b0>>2)
        输出[基索+1] = numpy.uint8(((b0&3)<<4)|(b1>>4))
        输出[基索+2] = numpy.uint8(((b1&0xF)<<2)|(b2>>6))
        输出[基索+3] = numpy.uint8(b2&0x3F)
    return 输出[:数量]
@njit(cache=True)
def 加速打包5(量化值):
    数量 = len(量化值); 输出长度 = (数量 + 7) // 8
    输出 = numpy.zeros(输出长度 * 5, dtype=numpy.uint8)
    for 块索 in range(输出长度):
        基索 = 块索 * 8; 偏移 = 块索 * 5
        v0 = numpy.uint8(量化值[基索] if 基索 < 数量 else 0)
        v1 = numpy.uint8(量化值[基索+1] if 基索+1 < 数量 else 0)
        v2 = numpy.uint8(量化值[基索+2] if 基索+2 < 数量 else 0)
        v3 = numpy.uint8(量化值[基索+3] if 基索+3 < 数量 else 0)
        v4 = numpy.uint8(量化值[基索+4] if 基索+4 < 数量 else 0)
        v5 = numpy.uint8(量化值[基索+5] if 基索+5 < 数量 else 0)
        v6 = numpy.uint8(量化值[基索+6] if 基索+6 < 数量 else 0)
        v7 = numpy.uint8(量化值[基索+7] if 基索+7 < 数量 else 0)
        输出[偏移]   = numpy.uint8((v0<<3)|(v1>>2))
        输出[偏移+1] = numpy.uint8(((v1&3)<<6)|(v2<<1)|(v3>>4))
        输出[偏移+2] = numpy.uint8(((v3&0xF)<<4)|(v4>>1))
        输出[偏移+3] = numpy.uint8(((v4&1)<<7)|(v5<<2)|(v6>>3))
        输出[偏移+4] = numpy.uint8(((v6&7)<<5)|v7)
    return 输出
@njit(cache=True)
def 加速解包5(压缩, 数量):
    块数 = len(压缩) // 5; 输出 = numpy.empty(块数 * 8, dtype=numpy.uint8)
    for 块索 in range(块数):
        偏移 = 块索 * 5; 基索 = 块索 * 8
        b0 = 压缩[偏移]; b1 = 压缩[偏移+1]; b2 = 压缩[偏移+2]; b3 = 压缩[偏移+3]; b4 = 压缩[偏移+4]
        输出[基索]   = numpy.uint8((b0>>3)&0x1F)
        输出[基索+1] = numpy.uint8(((b0&7)<<2)|(b1>>6))
        输出[基索+2] = numpy.uint8((b1>>1)&0x1F)
        输出[基索+3] = numpy.uint8(((b1&1)<<4)|(b2>>4))
        输出[基索+4] = numpy.uint8(((b2&0xF)<<1)|(b3>>7))
        输出[基索+5] = numpy.uint8((b3>>2)&0x1F)
        输出[基索+6] = numpy.uint8(((b3&3)<<3)|(b4>>5))
        输出[基索+7] = numpy.uint8(b4&0x1F)
    return 输出[:数量]
@njit(cache=True)
def 加速打包12(量化值):
    数量 = len(量化值); 输出长度 = ((数量 + 1) >> 1) * 3
    输出 = numpy.zeros(输出长度, dtype=numpy.uint8)
    for 块索 in range(数量 >> 1):
        偏移 = 块索 * 3
        v0 = numpy.uint16(量化值[块索 << 1])
        v1 = numpy.uint16(量化值[(块索 << 1) + 1])
        输出[偏移]   = numpy.uint8(v0 >> 4)
        输出[偏移+1] = numpy.uint8(((v0 & 15) << 4) | (v1 >> 8))
        输出[偏移+2] = numpy.uint8(v1 & 255)
    if 数量 & 1:
        偏移 = (数量 >> 1) * 3
        v0 = numpy.uint16(量化值[数量 - 1])
        输出[偏移]   = numpy.uint8(v0 >> 4)
        输出[偏移+1] = numpy.uint8((v0 & 15) << 4)
    return 输出

@njit(cache=True)
def 加速解包12(压缩, 数量):
    块数 = (数量 + 1) >> 1
    输出 = numpy.empty(块数 * 2, dtype=numpy.uint16)
    for 块索 in range(块数):
        偏移 = 块索 * 3
        b0 = numpy.uint16(压缩[偏移])
        b1 = numpy.uint16(压缩[偏移 + 1])
        b2 = numpy.uint16(压缩[偏移 + 2])
        输出[块索 << 1]         = (b0 << 4) | (b1 >> 4)
        输出[(块索 << 1) + 1]   = ((b1 & 15) << 8) | b2
    return 输出[:数量]
class Quantization:
    def __init__(Self, App: Config):
        Self.Config = App.Config
        Self.日志 = App.日志
        Self.Lang = App.Lang
        Self.tqdm = App.RichTqdm
        Self.Index = App.Index
        Self.Module = App.Module
        Self.拼接键 = ["Vector"]
        Self.Numba加速 = (not GPU_ACC) and CPU_ACC
        if Self.Numba加速:
            try:
                _预热数据 = numpy.zeros((2, 8), dtype=numpy.float32)
                _预热边界 = numpy.array([-1.0, 0.0, 1.0], dtype=numpy.float32)
                _预热中心 = numpy.array([-1.5, -0.5, 0.5, 1.5], dtype=numpy.float32)
                _预热alpha = numpy.linspace(0.1, 1.0, 3, dtype=numpy.float32)
                加速打包2(numpy.zeros(8, dtype=numpy.uint8))
                加速解包2(numpy.zeros(2, dtype=numpy.uint8), 8)
                加速打包4(numpy.zeros(8, dtype=numpy.uint8))
                加速解包4(numpy.zeros(4, dtype=numpy.uint8), 8)
                加速线性量化(_预热数据, numpy.ones(2, dtype=numpy.float32),
                      numpy.zeros(2, dtype=numpy.float32), 15.0)
                加速线性反量化(numpy.zeros((2, 8), dtype=numpy.uint8),
                        numpy.ones(2, numpy.float32), numpy.zeros(2, numpy.float32))
                加速NF量化(_预热数据, _预热边界, numpy.ones(2, dtype=numpy.float32))
                加速FWHT原地(_预热数据.copy())
                加速查找alpha(numpy.zeros(10, dtype=numpy.float32), _预热alpha)
                加速SVD处理(_预热数据.copy(), _预热边界, _预热中心)
                加速SVD处理_无异常(_预热数据.copy(), _预热边界, _预热中心)
                加速劳埃德统计(numpy.zeros(4, dtype=numpy.float32), _预热边界, 4)
                加速三值打包(numpy.zeros(5, dtype=numpy.uint8))
                加速三值解包(numpy.zeros(1, dtype=numpy.uint8), 5)
                加速打包3(numpy.zeros(8, dtype=numpy.uint8))
                加速解包3(numpy.zeros(3, dtype=numpy.uint8), 8)
                加速打包6(numpy.zeros(4, dtype=numpy.uint8))
                加速解包6(numpy.zeros(3, dtype=numpy.uint8), 4)
                加速打包5(numpy.zeros(8, dtype=numpy.uint8))
                加速解包5(numpy.zeros(5, dtype=numpy.uint8), 8)
                加速打包12(numpy.zeros(4, dtype=numpy.uint16))
                加速解包12(numpy.zeros(6, dtype=numpy.uint8), 4)
            except Exception:
                Self.Numba加速 = False
        Self.编码映射 = {}
        Self.解码映射 = {}
        try:
            Mods().安装量化注册(Self, Self.编码映射, Self.解码映射)
        except Exception: pass
    def _分块(Self, 数组, 块大小):
        数组 = np.ascontiguousarray(数组, dtype=np.float32)
        形状 = 数组.shape; 总数 = 数组.size; 填充 = (-总数) % 块大小
        if 填充:
            缓冲 = np.zeros(总数 + 填充, dtype=np.float32); 缓冲[:总数] = 数组.ravel()
        else:
            缓冲 = 数组.ravel().copy()
        del 数组
        return 缓冲.reshape(-1, 块大小), 形状, 总数
    def _百分位(Self, 数据, 百分比, 轴=1, 保持维度=False):
        结果 = np.percentile(数据, [百分比, 100.0 - 百分比], axis=轴, keepdims=保持维度)
        return 结果[0].astype(np.float32), 结果[1].astype(np.float32)
    def 分位裁切(Self, 数组):
        裁切 = Mods().裁切方法(Self)
        if 裁切 is None:
            return 数组
        return 裁切(Self, 数组)
    def _解码填充(Self, 数据, 类型, 总数):
        缓冲 = np.zeros(总数, dtype=类型); 数量 = min(len(数据), 总数); 缓冲[:数量] = 数据[:数量]
        return 缓冲
    def _打包2(Self, 量化值):
        if Self.Numba加速:
            return 加速打包2(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 4
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        四组 = 量化值.reshape(-1, 4)
        return (四组[:,0]|(四组[:,1]<<2)|(四组[:,2]<<4)|(四组[:,3]<<6)).astype(np.uint8)[:(数量+3)//4]
    def _解包2(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包2(压缩, 数量)
        块数 = (数量 + 3) // 4; 缓冲 = np.zeros(块数, dtype=np.uint8)
        取用 = min(len(压缩), 块数); 缓冲[:取用] = 压缩[:取用]
        输出 = np.empty(块数 * 4, dtype=np.uint8)
        输出[0::4]=缓冲&3; 输出[1::4]=(缓冲>>2)&3; 输出[2::4]=(缓冲>>4)&3; 输出[3::4]=(缓冲>>6)&3
        return 输出[:数量]
    def _打包4(Self, 量化值):
        if Self.Numba加速:
            return 加速打包4(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 2
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        双组 = 量化值.reshape(-1, 2)
        return ((双组[:,0]<<4)|双组[:,1]).astype(np.uint8)[:(数量+1)//2]
    def _解包4(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包4(压缩, 数量)
        块数 = (数量 + 1) // 2; 缓冲 = np.zeros(块数, dtype=np.uint8)
        取用 = min(len(压缩), 块数); 缓冲[:取用] = 压缩[:取用]
        输出 = np.empty(块数 * 2, dtype=np.uint8)
        输出[0::2]=(缓冲>>4)&0xF; 输出[1::2]=缓冲&0xF
        return 输出[:数量]
    def _打包3(Self, 量化值):
        if Self.Numba加速:
            return 加速打包3(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 8
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        八组 = 量化值.reshape(-1, 8)
        b0 = (八组[:,0]<<5)|(八组[:,1]<<2)|(八组[:,2]>>1)
        b1 = ((八组[:,2]&1)<<7)|(八组[:,3]<<4)|(八组[:,4]<<1)|(八组[:,5]>>2)
        b2 = ((八组[:,5]&3)<<6)|(八组[:,6]<<3)|八组[:,7]
        return np.column_stack([b0, b1, b2]).ravel()[:((数量+7)//8)*3].astype(np.uint8)
    def _解包3(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包3(压缩, 数量)
        块数 = (数量 + 7) // 8; 缓冲 = np.zeros(块数 * 3, dtype=np.uint8)
        取用 = min(len(压缩), 块数 * 3); 缓冲[:取用] = 压缩[:取用]
        缓冲 = 缓冲.reshape(块数, 3); 输出 = np.empty((块数, 8), dtype=np.uint8)
        输出[:,0]=(缓冲[:,0]>>5)&7; 输出[:,1]=(缓冲[:,0]>>2)&7
        输出[:,2]=((缓冲[:,0]&3)<<1)|(缓冲[:,1]>>7); 输出[:,3]=(缓冲[:,1]>>4)&7
        输出[:,4]=(缓冲[:,1]>>1)&7; 输出[:,5]=((缓冲[:,1]&1)<<2)|(缓冲[:,2]>>6)
        输出[:,6]=(缓冲[:,2]>>3)&7; 输出[:,7]=缓冲[:,2]&7
        return 输出.ravel()[:数量]
    def _打包5(Self, 量化值):
        if Self.Numba加速:
            return 加速打包5(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 8
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        八组 = 量化值.reshape(-1, 8)
        b0 = (八组[:,0]<<3)|(八组[:,1]>>2)
        b1 = ((八组[:,1]&3)<<6)|(八组[:,2]<<1)|(八组[:,3]>>4)
        b2 = ((八组[:,3]&0xF)<<4)|(八组[:,4]>>1)
        b3 = ((八组[:,4]&1)<<7)|(八组[:,5]<<2)|(八组[:,6]>>3)
        b4 = ((八组[:,6]&7)<<5)|八组[:,7]
        return np.column_stack([b0, b1, b2, b3, b4]).ravel()[:((数量+7)//8)*5].astype(np.uint8)
    def _解包5(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包5(压缩, 数量)
        块数 = (数量 + 7) // 8; 缓冲 = np.zeros(块数 * 5, dtype=np.uint8)
        取用 = min(len(压缩), 块数 * 5); 缓冲[:取用] = 压缩[:取用]
        缓冲 = 缓冲.reshape(块数, 5); 输出 = np.empty((块数, 8), dtype=np.uint8)
        输出[:,0] = (缓冲[:,0]>>3)&0x1F
        输出[:,1] = ((缓冲[:,0]&7)<<2)|(缓冲[:,1]>>6)
        输出[:,2] = (缓冲[:,1]>>1)&0x1F
        输出[:,3] = ((缓冲[:,1]&1)<<4)|(缓冲[:,2]>>4)
        输出[:,4] = ((缓冲[:,2]&0xF)<<1)|(缓冲[:,3]>>7)
        输出[:,5] = (缓冲[:,3]>>2)&0x1F
        输出[:,6] = ((缓冲[:,3]&3)<<3)|(缓冲[:,4]>>5)
        输出[:,7] = 缓冲[:,4]&0x1F
        return 输出.ravel()[:数量]
    def _打包6(Self, 量化值):
        if Self.Numba加速:
            return 加速打包6(量化值 if 量化值.dtype == numpy.uint8 else 量化值.astype(numpy.uint8))
        数量 = len(量化值); 填充 = (-数量) % 4
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint8)])
        四组 = 量化值.reshape(-1, 4)
        b0 = (四组[:,0]<<2)|(四组[:,1]>>4)
        b1 = ((四组[:,1]&0xF)<<4)|(四组[:,2]>>2)
        b2 = ((四组[:,2]&3)<<6)|四组[:,3]
        return np.column_stack([b0, b1, b2]).ravel()[:((数量+3)//4)*3].astype(np.uint8)
    def _解包6(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包6(压缩, 数量)
        块数 = (数量 + 3) // 4; 缓冲 = np.zeros(块数 * 3, dtype=np.uint8)
        取用 = min(len(压缩), 块数 * 3); 缓冲[:取用] = 压缩[:取用]
        缓冲 = 缓冲.reshape(块数, 3); 输出 = np.empty((块数, 4), dtype=np.uint8)
        输出[:,0]=缓冲[:,0]>>2; 输出[:,1]=((缓冲[:,0]&3)<<4)|(缓冲[:,1]>>4)
        输出[:,2]=((缓冲[:,1]&0xF)<<2)|(缓冲[:,2]>>6); 输出[:,3]=缓冲[:,2]&0x3F
        return 输出.ravel()[:数量]
    def _打包F12(Self, 量化值):
        if Self.Numba加速:
            return 加速打包12(量化值 if 量化值.dtype == numpy.uint16 else 量化值.astype(numpy.uint16))
        数量 = len(量化值); 填充 = (-数量) % 2
        if 填充: 量化值 = np.concatenate([量化值, np.zeros(填充, dtype=np.uint16)])
        双组 = 量化值.reshape(-1, 2)
        b0 = (双组[:, 0] >> 4).astype(np.uint8)
        b1 = ((双组[:, 0] & 15) << 4 | (双组[:, 1] >> 8)).astype(np.uint8)
        b2 = (双组[:, 1] & 255).astype(np.uint8)
        return np.column_stack([b0, b1, b2]).ravel()[:((数量 + 1) // 2) * 3]

    def _解包F12(Self, 压缩, 数量):
        if Self.Numba加速: return 加速解包12(压缩, 数量)
        块数 = (数量 + 1) // 2; 缓冲 = np.zeros(块数 * 3, dtype=np.uint8)
        取用 = min(len(压缩), 块数 * 3); 缓冲[:取用] = 压缩[:取用]
        缓冲 = 缓冲.reshape(块数, 3); 输出 = np.empty(块数 * 2, dtype=np.uint16)
        输出[0::2] = (缓冲[:, 0].astype(np.uint16) << 4) | (缓冲[:, 1].astype(np.uint16) >> 4)
        输出[1::2] = ((缓冲[:, 1].astype(np.uint16) & 15) << 8) | 缓冲[:, 2].astype(np.uint16)
        return 输出[:数量]
    def _NF编码(Self, 数据块, 百分比, 边界):
        if 百分比 < 100.0:
            高位, 低位 = Self._百分位(数据块, 百分比, -1, True)
            np.clip(数据块, 低位, 高位, out=数据块)
        缩放 = np.std(数据块, axis=-1, ddof=0).astype(np.float32)
        np.maximum(缩放, 1e-8, out=缩放)
        if Self.Numba加速:
            原始形状 = 数据块.shape
            索引 = 加速NF量化(
                数据块.reshape(-1, 原始形状[-1]),
                边界,
                缩放.ravel()
            ).reshape(原始形状)
        else:
            数据块 /= 缩放[..., None]
            索引 = np.searchsorted(边界, 数据块).astype(np.uint8)
        return 索引, 缩放
    def _NF解码(Self, 索引, 缩放编码, 最大缩放, 总数, 形状, 块大小, 中心):
        if 块大小 is not None:
            块数 = (总数 + 块大小 - 1) // 块大小; 填充总数 = 块数 * 块大小
            缩放 = (Self._解码缩放(缩放编码, 块数) * 最大缩放).astype(np.float32)  # ← 补 块数
            if len(索引) < 填充总数:
                缓冲 = np.zeros(填充总数, dtype=索引.dtype); 缓冲[:len(索引)] = 索引; 索引 = 缓冲
            索引 = 索引[:填充总数].reshape(块数, 块大小)
            缩放 = 缩放[:块数, None]
        else:
            缩放 = (Self._解码缩放(缩放编码) * 最大缩放).astype(np.float32)        # ← 无块大小时不传数量
            if 缩放.ndim == 1: 缩放 = 缩放[:, None]
        结果 = 中心[索引] * 缩放
        return 结果.ravel()[:总数].reshape(形状)
    def FWHT变换(Self, 数组):
        if Self.Numba加速:
            结果 = 数组.copy()
            if 结果.ndim == 1: 结果 = 结果.reshape(1, -1)
            加速FWHT原地(结果); return 结果.reshape(数组.shape)
        数组 = 数组.copy(); 步长 = 1
        while 步长 < 数组.shape[-1]:
            视图 = 数组.reshape(*数组.shape[:-1], -1, 2, 步长)
            左 = 视图[..., 0, :]; 右 = 视图[..., 1, :]; 和 = 左 + 右
            视图[..., 1, :] = 左 - 右; 视图[..., 0, :] = 和; 步长 *= 2
        return 数组
    def IFWHT逆变换(Self, 数组):
        数组 = Self.FWHT变换(数组); return 数组 / 数组.shape[-1]
    def 采样(Self, 数组, 参数):
        平 = np.asarray(数组).ravel()
        总数 = 平.size
        数量 = min(总数, Self.Module.采样器(参数, 总数))
        if 数量 < 总数:
            平 = 平[np.linspace(0, 总数 - 1, 数量).astype(np.intp)]
        return np.ascontiguousarray(平, dtype=np.float32)
    def LloydMax(Self, 样本, 电平数):
        样本 = np.sort(np.asarray(样本, dtype=np.float32).ravel())
        迭代上限 = int(Self.Config.VEC_QUANTIZATION_ITRS_LM)
        早停阈值 = float(Self.Config.VEC_QUANTIZATION_ES_LM)
        中心 = np.percentile(样本, np.linspace(0, 100, 电平数 + 2)[1:-1]).astype(np.float64)
        for 轮 in range(迭代上限):
            边界 = (中心[:-1] + 中心[1:]) / 2.0
            量化索引 = np.searchsorted(边界, 样本)
            计数 = np.bincount(量化索引, minlength=电平数)
            加权和 = np.bincount(量化索引, weights=样本.astype(np.float64), minlength=电平数)
            新中心 = np.where(计数 > 0, 加权和 / np.maximum(计数, 1), 中心)
            if float(np.max(np.abs(新中心 - 中心))) < 早停阈值:
                中心 = 新中心
                break
            中心 = 新中心
        return np.sort(中心).astype(np.float32)
    def 查表量化(Self, 归一值, 边界):
        return np.searchsorted(边界, 归一值).astype(np.uint8)
    def Hadamard原地(Self, 数组):
        if Self.Numba加速:
            加速FWHT原地(数组)
            return
        长度 = 数组.shape[1]; 步长 = 1
        while 步长 < 长度:
            跨度 = 2 * 步长
            for 起始 in range(0, 长度, 跨度):
                左 = 数组[:, 起始:起始+步长].copy()
                右 = 数组[:, 起始+步长:起始+跨度]
                数组[:, 起始:起始+步长] = 左 + 右
                数组[:, 起始+步长:起始+跨度] = 左 - 右
            步长 = 跨度
    def _缩放类型(Self, 深度):
        # ↓ 块缩放通道用哪个类型:
        #     深度0       -> VEC_QUANTIZATION_SCALE_TYPE (用户配置的那个)
        #     深度1       -> VEC_QUANTIZATION_SCALE_INNER_TYPE (默认 Float16_E0M15)
        #     深度>=2     -> 缩放终极回退类型
        #   为什么要有内层: Qx_K_M/GSQ_NL 这类格式内部自己还会调 _编码缩放(mods/quant-q_k_m.py:44,
        #   mods/quant-gsq_nl.py:53), 若内层继续取 SCALE_TYPE 就变成"自己调自己"(自递归)。
        #   内层写死 = llama.cpp 把 k-quant 的 d/dmin 固定打成 fp16 的同一做法。
        if 深度 == 0:
            return Self.Config.VEC_QUANTIZATION_SCALE_TYPE
        if 深度 == 1:
            return getattr(Self.Config, "VEC_QUANTIZATION_SCALE_INNER_TYPE", 缩放终极回退类型)
        return 缩放终极回退类型
    def _编码缩放(Self, 数组):
        # ↓ 整字典透传 + 不再 ravel: 让"带伴随量的定点(Int8_Max)/打包(12bit/6bit)/块量化(Q6_K_M)"
        #   也能当块缩放。返回的是编码器的原始字典(可能含 Vector/MaxScale/Shape/Mean/Codebook...),
        #   由调用方原样存进结果字典(如 mods/quant-gsq_nl.py 的 "Min"/"Scale"), 解码时再原样喂回。
        #   形状也不再压平: 传进来是二维就按二维编(块量化器需要 行数, 维度 = 数组.shape)。
        深度 = getattr(缩放线程状态, "编码深度", 0)
        类型 = Self._缩放类型(深度)
        数组 = np.ascontiguousarray(数组, dtype=np.float32)
        缩放线程状态.编码深度 = 深度 + 1
        try:
            return Self.编码映射[类型](Self, 数组)
        finally:
            缩放线程状态.编码深度 = 深度
    def _解码缩放(Self, 数据, 数量=None):
        # ↓ 与 _编码缩放 对称:
        #     收到字典(新格式)  -> 原样喂回内层解码器(键全都在, 带伴随量/形状的格式也能还原)
        #     收到裸数组(老格式) -> 补 {"Vector": 数据} 走老路径, 兼容历史缓存/已有 GGUF
        #   数量 只在载荷自己没有 Shape 时兜底补一个, 打包类格式(12bit/6bit)解码需要元素数。
        深度 = getattr(缩放线程状态, "解码深度", 0)
        类型 = Self._缩放类型(深度)
        载荷 = 数据 if isinstance(数据, dict) else {"Vector": 数据}
        if 数量 is not None and "Shape" not in 载荷:
            载荷 = dict(载荷); 载荷["Shape"] = (数量,)
        缩放线程状态.解码深度 = 深度 + 1
        try:
            结果 = np.asarray(Self.解码映射[类型](Self, 载荷)).astype(np.float32).ravel()
        finally:
            缩放线程状态.解码深度 = 深度
        if 数量 is not None and len(结果) > 数量:
            return 结果[:数量].copy()
        return 结果
    def _载荷打平(Self, 载荷, 前缀):
        # ↓ 混合量化要把高/低两组的载荷拼进同一个字典, 载荷可能是嵌套字典(块缩放改造后),
        #   所以按 "前缀+键路径" 打平(键里用 / 连接, 与 向量GGUF写入 的嵌套键路径同一套规则)。
        输出 = {}
        def 递归(值, 路径):
            if isinstance(值, dict):
                for 子键, 子值 in 值.items(): 递归(子值, f"{路径}/{子键}")
            else:
                输出[路径] = np.asarray(值)
        # ↓ 顶层只做"前缀+键名"(与改造前的 High_xxx/Low_xxx 完全同名), 嵌套层才用 / 连接
        for 键, 值 in 载荷.items():
            递归(值, f"{前缀}{键}")
        return 输出
    def _载荷还原(Self, 数据, 前缀):
        # ↓ _载荷打平 的逆操作: 前缀+a/b 还原成 {a: {b: ...}}; 没有子路径时还原成裸数组(老格式)。
        输出 = {}
        for 键, 值 in 数据.items():
            if not 键.startswith(前缀): continue
            路径 = 键[len(前缀):]
            if not 路径: continue
            段 = 路径.split("/")
            节点 = 输出
            for 段名 in 段[:-1]: 节点 = 节点.setdefault(段名, {})
            节点[段[-1]] = np.asarray(值)
        return 输出

#====================================================================================================向量重排====================================================================================================#
    def 向量重排(Self, 数组, 文本列表=None):
        聚类块大小 = Self.Config.VEC_RERANKER_INDEX_RERANKER_BLOCK_SIZE
        向量数量, 向量维度 = 数组.shape
        if 向量数量 <= 聚类块大小:
            return 数组, 文本列表 if 文本列表 is not None else 数组
        Self.日志("log.quantization.vector.reorder.start", info_level=0)
        数组_归一 = np.ascontiguousarray(数组, dtype=np.float32).copy()
        faiss.normalize_L2(数组_归一)
        向量索引 = Self.Index.构建索引(数组_归一, 向量重排模式=True)
        已访问掩码 = np.zeros(向量数量, dtype=bool)
        重排索引列表 = []
        粗搜数量 = int(聚类块大小 * Self.Config.VEC_RERANKER_INDEX_FACTOR)
        搜索指针 = 0
        已处理数量 = 0
        with Self.tqdm(total=向量数量, desc="tqdm.vectors.reorder") as 进度条:
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
                进度条.update(len(最终索引))
        映射表 = np.array(重排索引列表, dtype=np.int64)
        重排后数组 = 数组[映射表]
        重排后文本 = [文本列表[i] for i in 映射表] if 文本列表 is not None else None
        Self.日志("log.quantization.vector.reorder.end", info_level=0)
        return 重排后数组, 重排后文本
#====================================================================================================向量重排====================================================================================================#
#====================================================================================================SpectralTempering====================================================================================================#
    def _谱膝点(Self, 曲线):
        曲线 = np.asarray(曲线, dtype=np.float64)
        长度 = 曲线.shape[0]
        if 长度 < 5:
            return 长度 // 2
        x = np.linspace(0.0, 1.0, 长度)
        极差 = float(曲线.max() - 曲线.min())
        y = (曲线 - 曲线.min()) / 极差 if 极差 > 0.0 else np.zeros(长度)
        弦 = y[0] + (y[-1] - y[0]) * x
        偏差 = np.abs(y - 弦)
        偏差[0] = 0.0
        偏差[-1] = 0.0
        return int(np.argmax(偏差))
    def 自适应温度(Self, 特征值, 目标维度, 尾部比例=0.1):
        特征值 = np.asarray(特征值, dtype=np.float64)
        维度 = 特征值.shape[0]
        if 维度 <= 目标维度:
            return 0.0
        尾部起点 = max(1, int(np.floor((1.0 - 尾部比例) * 维度)))
        噪声地板 = float(特征值[尾部起点:].mean())
        if not np.isfinite(噪声地板) or 噪声地板 <= 0.0:
            return 0.0
        SNR = np.maximum(0.0, (特征值 - 噪声地板) / 噪声地板)
        膝点 = Self._谱膝点(SNR)
        参考 = float(SNR[膝点]) if SNR[膝点] > 0.0 else float(SNR[0])
        if not np.isfinite(参考) or 参考 <= 0.0:
            return 0.0
        return float(min(1.0, float(SNR[目标维度 - 1]) / 参考))
    def SpecTemp降维(Self, 数据):
        目标维度 = int(Self.Config.VEC_SPECTEMP_DIM)
        均值 = np.mean(数据, axis=0, keepdims=True)
        数据中心 = 数据 - 均值
        if 数据.shape[0] > 数据.shape[1]:
            协方差 = 数据中心.T @ 数据中心
            特征值, 特征向量 = np.linalg.eigh(协方差)
            排序 = np.argsort(特征值)[::-1]
            特征值 = 特征值[排序]
            投影矩阵 = 特征向量[:, 排序[:目标维度]]
        else:
            _, 奇异值, Vt = np.linalg.svd(数据中心, full_matrices=False)
            投影矩阵 = Vt[:目标维度, :].T
            特征值 = np.zeros(数据.shape[1], dtype=np.float64)
            特征值[:奇异值.shape[0]] = 奇异值.astype(np.float64) ** 2
        实际维度 = 投影矩阵.shape[1]
        if 实际维度 > 0:
            try:
                温度 = float(Self.Config.VEC_SPECTEMP_TEMPER)
            except AttributeError:
                温度 = -1.0
            if 温度 < 0.0:
                温度 = Self.自适应温度(特征值, 实际维度)
            if 温度 > 0.0:
                尺度 = np.maximum(特征值[:实际维度], 0.0) ** (-温度 / 2.0)
                尺度[~np.isfinite(尺度)] = 1.0
                尺度 = 尺度.astype(投影矩阵.dtype)
                投影矩阵 = 投影矩阵 * 尺度[np.newaxis, :]
        数据 = np.dot(数据中心, 投影矩阵)
        return 数据, 均值, 投影矩阵
    def SpecTemp应用(Self, 数据, 均值, 投影矩阵):
        return np.dot(数据 - 均值, 投影矩阵).astype(np.float32)
    def SpecTemp应用懒加载(Self, 向量, 向量文件): #不修改指针
        if 向量文件.PCA_M is not None and 向量文件.PCA_P is not None:
            向量[:] = Self.SpecTemp应用(Self, 向量, 向量文件.PCA_M, 向量文件.PCA_P)
        return 向量
#====================================================================================================SpectralTempering====================================================================================================#
#====================================================================================================TT分解====================================================================================================#
    def TT分解(Self, 数据, TT形状=None, TT秩=4):
        数据 = np.asarray(数据, dtype=np.float32)
        n, d = 数据.shape
        if TT形状 is None:
            k = int(np.floor(np.log2(d)))
            TT形状 = [2] * k
            prod = 2 ** k
            if prod < d:
                TT形状 = [d // prod] + TT形状
                prod = int(np.prod(TT形状))
        else:
            prod = int(np.prod(TT形状))
        if prod != d:
            raise ValueError(Self.Lang("log.core.quantization.tt.shape.mismatch", prod=prod, dim=d))
        均值 = np.mean(数据, axis=0, dtype=np.float32)
        中心数据 = 数据 - 均值
        k = len(TT形状)
        累积核心 = []
        for i in range(k):
            if i == 0:
                累积核心.append(np.zeros((1, TT形状[0], TT秩), dtype=np.float64))
            elif i < k - 1:
                累积核心.append(np.zeros((TT秩, TT形状[i], TT秩), dtype=np.float64))
            else:
                累积核心.append(np.zeros((TT秩, TT形状[k-1], 1), dtype=np.float64))
        采样数 = min(n, 5000)
        采样索引 = np.random.choice(n, 采样数, replace=False) if n > 采样数 else np.arange(n)
        for idx in 采样索引:
            向量 = 中心数据[idx]
            张量 = 向量.reshape(TT形状)
            残差 = 张量.astype(np.float64).copy()
            秩左 = 1
            for i in range(k - 1):
                展平 = 残差.reshape(秩左 * TT形状[i], -1)
                U, S, Vt = np.linalg.svd(展平, full_matrices=False)
                截断秩 = min(TT秩, len(S))
                U = U[:, :截断秩]
                if 累积核心[i].any() and U.size > 0:
                    if np.sum(累积核心[i].ravel()[:min(累积核心[i].size, U.size)] * U.ravel()[:min(累积核心[i].size, U.size)]) < 0:
                        U = -U; Vt = -Vt
                核心 = U.reshape(秩左, TT形状[i], 截断秩)
                累积核心[i] += 核心
                秩左 = 截断秩
                残差 = (np.diag(S[:截断秩]) @ Vt[:截断秩, :]).reshape(截断秩, -1)
            累积核心[k-1] += 残差.reshape(秩左, TT形状[k-1], 1)
        累积核心 = [(核心 / 采样数).astype(np.float32) for 核心 in 累积核心]
        return 累积核心, 均值, TT形状

    def TT压缩(Self, 向量, 核心列表, 均值, TT形状):
        向量 = np.asarray(向量, dtype=np.float32).ravel() - 均值
        张量 = 向量.reshape(TT形状)
        k = len(TT形状)
        系数列表 = []
        残差 = 张量.astype(np.float64).copy()
        秩左 = 1
        for i in range(k - 1):
            核心 = 核心列表[i]
            r_i, d_i, r_ip1 = 核心.shape
            展平 = 残差.reshape(r_i * d_i, -1)
            核心展平 = 核心.reshape(r_i * d_i, r_ip1).astype(np.float64)
            系数 = 核心展平.T @ 展平
            系数列表.append(系数.astype(np.float32).ravel())
            重建 = 核心展平 @ 系数
            残差 = (展平 - 重建).reshape(r_ip1, -1)
        核心展平 = 核心列表[k-1].reshape(核心列表[k-1].shape[0] * 核心列表[k-1].shape[1], 1).astype(np.float64)
        最终系数 = 核心展平.T @ 残差.ravel().reshape(-1, 1)
        系数列表.append(最终系数.astype(np.float32).ravel())
        return np.concatenate(系数列表)

    def TT解压(Self, TT向量, 核心列表, 均值, TT形状):
        k = len(TT形状)
        TT向量 = np.asarray(TT向量, dtype=np.float64)
        当前 = np.array([[1.0]], dtype=np.float64)
        偏移 = 0
        for i in range(k - 1):
            核心 = 核心列表[i]
            r_i, d_i, r_ip1 = 核心.shape
            系数 = TT向量[偏移:偏移 + r_ip1]
            偏移 += r_ip1
            当前 = 当前 @ 核心.reshape(r_i * d_i, r_ip1).astype(np.float64)
            当前 = 当前 * 系数[np.newaxis, :]
            当前 = 当前.reshape(1, -1) 
        最终系数 = TT向量[偏移]
        核心 = 核心列表[k-1]
        r_k, d_k, _ = 核心.shape
        重建向量 = (当前 @ 核心.reshape(r_k * d_k, 1).astype(np.float64)).ravel() * 最终系数
        return (重建向量[:int(np.prod(TT形状))] + 均值).astype(np.float32)

    def TT应用懒加载(Self, 向量, 向量文件):
        if 向量文件.TT_Cores is not None and 向量文件.TT_Mean is not None:
            核心列表 = 向量文件.TT_Cores
            均值 = 向量文件.TT_Mean
            形状 = 向量文件.TT_Shape
            for i in range(len(向量)):
                向量[i] = Self.TT解压(Self, 向量[i], 核心列表, 均值, 形状)
        return 向量
#====================================================================================================TT分解====================================================================================================#
#====================================================================================================类接口====================================================================================================#
    def _解析量化配置(Self, 量化格式=None):
        格式 = 量化格式 if 量化格式 is not None else Self.Config.VEC_QUANTIZATION
        if isinstance(格式, (list, tuple)):
            if len(格式) == 1:
                return [str(格式[0])], False
            return [str(x) for x in 格式[:2]], True
        return [str(格式)], False
    def _维度贡献掩码(Self, 数组, 比例):
        能量 = np.mean(np.asarray(数组, dtype=np.float32) ** 2, axis=0)
        排序 = np.argsort(能量)[::-1]
        k = int(round(len(能量) * float(比例)))
        k = max(1, min(k, len(能量) - 1))
        掩码 = np.zeros(len(能量), dtype=bool); 掩码[排序[:k]] = True
        return 掩码
    def _混合编码(Self, 数组, 格式列表):
        低格式, 高格式 = 格式列表[0], 格式列表[1]
        数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32)
        比例 = float(getattr(Self.Config, 'VEC_QUANTIZATION_MIX_RATIO', 0.2))
        掩码 = Self._维度贡献掩码(数组, 比例)  # 保证高低两组都非空
        高索引 = np.where(掩码)[0]; 低索引 = np.where(~掩码)[0]
        高编码 = Self.编码映射[高格式](Self, 数组[:, 高索引])
        低编码 = Self.编码映射[低格式](Self, 数组[:, 低索引])
        del 数组; CleanVRAM()
        输出 = {"Format": np.packbits(掩码),
                "HighFmt": np.array(高格式), "LowFmt": np.array(低格式),
                "Shape": np.asarray(掩码.shape, dtype=np.int32)}
        输出.update(Self._载荷打平(高编码, "High_"))
        输出.update(Self._载荷打平(低编码, "Low_"))
        del 高编码, 低编码; CleanVRAM()
        return 输出
    def _逐行TopK混合编码(Self, 数组, 格式列表):
        低格式, 高格式 = 格式列表[0], 格式列表[1]
        数组 = np.ascontiguousarray(np.asarray(数组), dtype=np.float32)
        行数, 维度 = 数组.shape
        k = int(getattr(Self.Config, 'VEC_QUANTIZATION_MIX_TOPK', 0) or 0)
        if k <= 0:
            比例 = float(getattr(Self.Config, 'VEC_QUANTIZATION_MIX_RATIO', 0.05))
            k = max(1, int(round(维度 * 比例)))
        k = min(k, 维度 - 1)
        排序 = np.argsort(-np.abs(数组), axis=1)
        高索引 = 排序[:, :k]
        行索引 = np.arange(行数)[:, None]
        高值 = 数组[行索引, 高索引].astype(np.float16)
        低编码 = Self.编码映射[低格式](Self, 数组)
        del 数组; CleanVRAM()
        输出 = {"RowTopK": np.array(k, dtype=np.int32),
                "HighIdx": np.ascontiguousarray(高索引.astype(np.uint16)),
                "HighVal": np.ascontiguousarray(高值.view(np.uint16)),
                "HighFmt": np.array(高格式), "LowFmt": np.array(低格式),
                "Shape": np.asarray((行数, 维度), dtype=np.int32)}
        输出.update(Self._载荷打平(低编码, "Low_"))
        del 低编码, 高索引, 高值; CleanVRAM()
        return 输出
    def _逐行TopK混合解码(Self, 数据):
        高格式 = str(np.asarray(数据["HighFmt"]).item() if np.asarray(数据["HighFmt"]).ndim else np.asarray(数据["HighFmt"]))
        低格式 = str(np.asarray(数据["LowFmt"]).item() if np.asarray(数据["LowFmt"]).ndim else np.asarray(数据["LowFmt"]))
        行数, 维度 = int(np.asarray(数据["Shape"]).ravel()[0]), int(np.asarray(数据["Shape"]).ravel()[1])
        k = int(np.asarray(数据["RowTopK"]).item())
        高索引 = np.asarray(数据["HighIdx"], dtype=np.int32).reshape(行数, k)
        高值 = np.asarray(数据["HighVal"], dtype=np.uint16).view(np.float16).reshape(行数, k).astype(np.float32)
        低编码 = Self._载荷还原(数据, "Low_")
        重建 = Self.解码映射[低格式](Self, 低编码)
        if 重建.ndim == 2 and 重建.shape[0] == 行数 and 重建.shape[1] == 维度:
            重建[np.arange(行数)[:, None], 高索引] = 高值
        del 高索引, 高值, 低编码; CleanVRAM()
        return 重建
    def _混合解码(Self, 数据):
        高格式 = str(np.asarray(数据["HighFmt"]).item() if np.asarray(数据["HighFmt"]).ndim else np.asarray(数据["HighFmt"]))
        低格式 = str(np.asarray(数据["LowFmt"]).item() if np.asarray(数据["LowFmt"]).ndim else np.asarray(数据["LowFmt"]))
        维度 = int(np.asarray(数据["Shape"]).ravel()[0])
        掩码 = np.unpackbits(np.asarray(数据["Format"], dtype=np.uint8))[:维度].astype(bool)
        高索引 = np.where(掩码)[0]; 低索引 = np.where(~掩码)[0]
        高编码 = Self._载荷还原(数据, "High_")
        低编码 = Self._载荷还原(数据, "Low_")
        高重建 = Self.解码映射[高格式](Self, 高编码)
        低重建 = Self.解码映射[低格式](Self, 低编码)
        行数 = max(高重建.shape[0] if 高重建.ndim >= 2 else 0,
                   低重建.shape[0] if 低重建.ndim >= 2 else 0)
        重建 = np.empty((行数, 维度), dtype=np.float32)
        重建[:, 高索引] = 高重建
        重建[:, 低索引] = 低重建
        del 高重建, 低重建; CleanVRAM()
        return 重建
    def 解码向量(Self, 向量, 量化格式=None):
        CleanVRAM()
        if isinstance(向量, dict) and "RowTopK" in 向量:
            for _ in Self.tqdm(range(1), desc="tqdm.vectors.quantization.reverse.rowtopk"):
                向量 = Self._逐行TopK混合解码(向量)
        elif isinstance(向量, dict) and "Format" in 向量 and "HighFmt" in 向量 and "LowFmt" in 向量:
            for _ in Self.tqdm(range(1), desc="tqdm.vectors.quantization.reverse.mix"):
                向量 = Self._混合解码(向量)
        else:
            格式列表, _ = Self._解析量化配置(量化格式)
            for _ in Self.tqdm(range(1), desc="tqdm.vectors.quantization.reverse"):
                向量 = Self.解码映射[格式列表[0]](Self, 向量)
        CleanVRAM()
        return 向量
    def 编码向量(Self, 向量, 量化格式=None):
        CleanVRAM()
        格式列表, 是否混合 = Self._解析量化配置(量化格式)
        if 是否混合:
            for _ in Self.tqdm(range(1), desc="tqdm.vectors.quantization.mix"):
                向量 = Self._逐行TopK混合编码(向量, 格式列表)
        else:
            for _ in Self.tqdm(range(1), desc="tqdm.vectors.quantization"):
                向量 = Self.编码映射[格式列表[0]](Self, 向量)
        CleanVRAM()
        return 向量
    def 叠加量化向量(Self, 旧数据, 新数据, 文本列表=None):
        旧浮点 = Self.解码向量(旧数据)
        浮点 = np.concatenate((旧浮点, 新数据), axis=0)
        if 文本列表 and Self.Config.VEC_RERANKER:
            浮点, 文本列表 = Self.向量重排(浮点, 文本列表)
        return Self.编码向量(浮点), 文本列表
#====================================================================================================类接口====================================================================================================#