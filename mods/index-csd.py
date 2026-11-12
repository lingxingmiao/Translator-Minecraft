# ============================================================================
#  index-csd.py  ——  球面结构 + EM 子单元中心 + N 电平 (对齐 index-gsq.py)
#
#  编码(每个子单元 = 块内一段连续行; S=SUBUNIT 由块内 k-means 切出):
#     c_u 子单元中心(EM 迭代, int8)  ‖r_i‖(uint8)  d_i 用共享 N 电平码本
#     ‖x̂_i‖(uint16)                 路由代表 = 子单元中心(质心) 或 FPS 极值 x REPNUM
#  打包: LEVELS ∈ {2,4,16} -> 字节对齐位平面
#        LEVELS ∈ {3,5,6}  -> **2.3333 bit/符号 位流**(5 电平时 7bit 装 3 符号 = 224 B/行)
#  可选 VQ 残差(VQM>0): PQ + ADC 对基座 top-RERANK 重排
#  打分: ⟨q,x̂_i⟩ = ⟨q,c_u⟩ + ‖r_i‖·Σ_d(q_d·码本[c]), 再除 ‖x̂_i‖
#        —— 表只跟查询有关; 扫行只读码流 + 2 个标量
#
#  配置 MOD_INDEX_CSD_<键>:
#     LEVELS 5  EMITER 3  BLOCKSIZE 128  SUBUNIT 2  CODEC LM  CENTERQ gsqnl  CENTERBITS 5
#     NPROBE 80  SCANVECTORS 0  MINVECTORS 100  NLISTF 4.0
#     VQM 0  VQMODE PQ  RERANK 400  USEPCA False  PCADIM 256
#     GRAPHTYPE NSG  GRAPHR 16  GRAPHK 40  GEF 128
# ============================================================================
from TranslatorLib import Mods, np, njit, _prange, faiss, SimpleNamespace, pickle, importlib, numba as nb, math
import os, sys

# ============================================================================
#  ★ GPU_Accelerator=True 兼容层(最小改动, 见 _work/GPUACC-报告.md)
#  TranslatorLib 在 GPU 模式下把 np 换成了 cupy, 于是本模组里每一处 np.asarray/np.zeros
#  都会产出【cupy 数组】; 而本模组的算力全在 @njit(CPU 并行) / faiss / gguf 存档上,
#  它们只吃 host numpy -> TypingError / TypeError -> 被上层 try/except 吞掉 ->
#  轻则表补推失败, 重则模组注册失败 -> 构建静默回落到 faiss.IndexFlatIP ✗
#  修法: 只在【模组边界】把 np 归一到真 numpy, 内核与既有数值路径完全不动;
#        cupy 句柄留作 cp, 将来要显式 GPU 加速时用它(不靠这个全局开关)。
# ============================================================================
import numpy as _真np

_GPU模式 = (np is not _真np)          # 只有 True 模式才需要边界归一(False 模式全程原路径 ✓)
cp = np if _GPU模式 else _真np        # cupy 句柄(False 模式下等同 numpy, 不会被用到)
np = _真np


def _主(x):
    """边界归一: cupy 数组(及其容器) -> host numpy; 本来就是 numpy 的原样返回(零拷贝)"""
    if x is None: return None
    if isinstance(x, _真np.ndarray): return x
    if type(x).__module__.split(".")[0] == "cupy": return x.get()
    if isinstance(x, dict): return {k: _主(v) for k, v in x.items()}
    if isinstance(x, list): return [_主(v) for v in x]
    if isinstance(x, tuple): return tuple(_主(v) for v in x)
    return x

参数 = Mods().模组配置("index-csd", {
    "LEVELS": 5, "EMITER": 3, "BLOCKSIZE": 128, "SUBUNIT": 2,
    "NPROBE": 80, "SCANVECTORS": 0, "MINVECTORS": 100, "NLISTF": 4.0,
    "VQM": 0, "VQMODE": "PQ", "RERANK": 400, "USEPCA": False, "PCADIM": 256,
    "GRAPHTYPE": "NSG", "GRAPHR": 16, "GRAPHK": 40, "GRAPHALPHA": 1.0, "HNSWM": 8, "GEF": 128,
    "GKNPROBE": 0,          # ★ 建图 kNN: 0=精确(默认 ✓ 召回最好 ✓) ; >0=IVF 近似(快 ✗ 但实测掉召回 ✗ 且没提速 ✗)
    "GEF初始": 32,           # 自适应 ef 的起始值(0=关闭, 即始终用 GEF)
    "JOINT": 0, "REV": 1, "GEDGE": 0.01, "GEDGE系数": 10.0, "YIELD": 0.0, "SUBREP": 0,
    "ROUTEK": 0, "CENTERQ": "gsqnl", "CENTERBITS": 5, "CENTER缓存": 4096,   # ★ 默认 GSQ-NL(均匀骨架 + 共享 32 电平形状表 ✓ 4 配置全胜 ✓ 参数更小 ✓) ; 可选 "gsq" / "fp16"
    # TCQ 码型(CODEC="TCQ" 生效): 12 电平 / 4 子集 / 4 状态 / 分支密度 3-of-4, 每行同为 226 B
    "CODEC": "LM", "qtip周期": 4, "qtip分支": 3, "qtip电平": 12,
    # ★ 稀疏48 实验开关(默认 0 = 关闭; 关闭时全代码零分支 -> 行为逐字节不变 ✓)
    #    1 = 固定丢 d%4∈{2,3}  |  2 = 固定种子随机丢一半  |  3 = 按样本能量丢最小的一半
    #    4 = 与 1 同丢法 + 丢维后【重拟合】中心与码本(走现有 EM/拟合代码 ✓)
    #    只看召回 vs 字节; 存档格式/位宽/每行字节【完全不变】(B/行 仍 244.59, 省字节另算)
    "稀疏48": 0, "稀疏种子": 2024,
    "TCQ缓存": 0})            # TCQ 解码结果缓存的槽数(0=关; 跨查询复用, 只影响速度)
层数 = max(2, int(参数["LEVELS"]))
EM轮数 = max(0, int(参数["EMITER"]))
块大小 = max(16, int(参数["BLOCKSIZE"]))
子单元数 = max(1, int(参数["SUBUNIT"]))
Nprobe默认 = max(1, int(参数["NPROBE"]))
扫描预算配置 = int(参数["SCANVECTORS"])
最小向量数 = int(参数["MINVECTORS"])
Nlist因子 = float(参数["NLISTF"])
VQ字节数 = max(0, int(参数["VQM"]))
VQ方式 = str(参数.get("VQMODE", "PQ")).strip().upper()
用RVQ = (VQ方式 == "RVQ")
图类型 = str(参数.get("GRAPHTYPE", "NSG")).strip().upper()   # NSG(自建剪枝图) / HNSW(faiss建图抽表) / IVF(旧路)
图度上限 = max(2, int(参数.get("GRAPHR", 16)))
kNN候选 = max(图度上限, int(参数.get("GRAPHK", 40)))
剪枝alpha = float(参数.get("GRAPHALPHA", 1.0))
HNSW度 = max(2, int(参数.get("HNSWM", 8)))
图ef = max(8, int(参数.get("GEF", 128)))
图ef初 = int(参数.get("GEF初始", 0))                      # >0 启用每查询自适应 ef
联合归属 = int(参数.get("JOINT", 1)) != 0                     # 量化感知重归属
反向边 = int(参数.get("REV", 1)) != 0                        # 剪枝后加反向边再重剪(NSG/HNSW 标准做法)
图边距 = float(参数.get("GEDGE", 0.01))
自适应系数 = float(参数.get("GEDGE系数", 0.0))          # >0 启用动态终止(按结果分数跨度)                     # 探索容忍边距(打分噪声补偿)
产出权重 = float(参数.get("YIELD", 0.0))                     # 单元打分的产出率修正(0=只用中心余弦)
次代表数 = max(0, int(参数.get("SUBREP", 0)))                # 每单元额外存几个次代表(单元内 2-means 质心)
重排数 = max(0, int(参数["RERANK"]))
用PCA = bool(参数["USEPCA"])
Pca维 = max(8, int(参数["PCADIM"]))
路由投影维 = max(0, int(参数["ROUTEK"]))
码型名 = str(参数.get("CODEC", "LM")).strip().upper()            # LM | QTIP
# ★ 键名: 声明表里的键是 "qtip周期/qtip分支/qtip电平"; MOD_INDEX_CSD_QTIP电平=8 是按大写匹配后
#   写回【声明名】(见 TranslatorMods.模组配置) -> 必须用小写键读, 旧码读 "QTIP电平" 恒取默认 12,
#   这就是"改电平无效"的根因 ✗
qtip周期 = int(参数.get("qtip周期", 参数.get("QTIP周期", 4)))
qtip分支 = int(参数.get("qtip分支", 参数.get("QTIP分支", 3)))
qtip电平 = int(参数.get("qtip电平", 参数.get("QTIP电平", 12)))
qtip每子 = max(1, min(6, qtip电平 // 4)); qtip电平 = qtip每子 * 4     # 4 子集固定 -> 电平数 = 4·每子
import importlib.util as _ilu
try:
    qtip = importlib.import_module("mods.quant-qtip")
except Exception:
    _sp = _ilu.spec_from_file_location("mods.quant-qtip",
                                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant-qtip.py"))
    qtip = _ilu.module_from_spec(_sp); sys.modules["mods.quant-qtip"] = qtip; _sp.loader.exec_module(qtip)
码型I = 1 if 码型名 in ("QTIP", "TCQ") else 0                     # TCQ 为旧名兼容
# ---- QTIP(bitshift 网格): 状态 = 最近 K 个【被传输的】比特 ----
#      因此解码器可直接从比特窗口查表得到子集 -> 无串行依赖(这是相对卷积 TCQ 的关键优势)
QTIP_K = max(2, min(6, int(参数.get("QTIP_K", 4))))
QTIP_G1 = int(参数.get("QTIP_G1", 0b1011)); QTIP_G2 = int(参数.get("QTIP_G2", 0b1110))
QTIP掩 = (1 << QTIP_K) - 1
def _建QTIP(K, g1, g2):
    子 = np.zeros(1 << K, np.int32)
    for r in range(1 << K):
        子[r] = ((bin(r & g1).count("1") & 1) << 1) | (bin(r & g2).count("1") & 1)
    S2 = np.zeros((1 << K, 2), np.int32); SJ = np.zeros((1 << K, 2), np.int32)
    for r in range(1 << K):
        SJ[r, 0] = 子[r]; SJ[r, 1] = 子[r]
        S2[r, 0] = (r << 1) & ((1 << K) - 1); S2[r, 1] = ((r << 1) | 1) & ((1 << K) - 1)
    反 = np.zeros(1 << K, np.int32)
    for r in range(1 << K):
        反[r] = int(format(r, "0%db" % K)[::-1], 2)
    return np.ascontiguousarray(子), np.ascontiguousarray(S2), np.ascontiguousarray(SJ), np.ascontiguousarray(反)
QTIP_子表, QTIP_S2, QTIP_SJ, QTIP_反位 = _建QTIP(QTIP_K, QTIP_G1, QTIP_G2)
# ---- 电平索引流的打包参数(随 QTIP电平 变) ----
#   索流每字节装 QTIP每字节 个符号: L=1->0字节(子集即电平) / L=2->8 / L=3->5 / L=4->4 ✓
QTIP每子 = int(qtip每子)
QTIP每字节 = int(qtip.每字节符号(QTIP每子))
QTIP索表 = np.ascontiguousarray(qtip.建索表(QTIP每子), np.uint8)     # L=3 时与旧 三表 逐元素相同 ✓
QTIP_三表 = QTIP索表                                                 # 旧名(内核参数叫 索表)
# 20 维展开的融合核只对 "5 符号/字节 + 每 20 维 15 个分支位(周期4/分支3)" 成立 -> 其余走通用核
QTIP快路 = (QTIP每子 == 3 and qtip周期 == 4 and qtip分支 == 3)


# ---- numba 磁盘缓存签名(必须!): 上面这些是"编译期常量"(内核里直接当成字面量),
#      但 numba 的 cache=True 只按【源码时间戳】判缓存, 不认全局常量的值 ✗
#      -> 同一份源码先跑 4 电平再跑 12 电平, 会复用 4 电平编译出来的核 -> 静默解错 ✗
#      (已实测: L=1 跑过之后 L=3 复用旧核, 解码与 Python 复刻差 768/768 维)
#      这里把"影响核内常量"的配置写进签名文件, 变了就先清掉 index-csd 的 numba 缓存 ✓
def _清QTIP缓存(签名):
    try:
        目 = (getattr(__import__("numba.core.config", fromlist=["x"]), "CACHE_DIR", "")
               or os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__"))
        os.makedirs(目, exist_ok=True)
        签 = os.path.join(目, "index-csd.qtip签名.txt")
        旧 = ""
        if os.path.exists(签):
            with open(签, "r", encoding="utf-8") as f:
                旧 = f.read().strip()
        if 旧 != 签名:
            for 名 in os.listdir(目):
                if 名.startswith("index-csd.") and 名.endswith((".nbi", ".nbc")):
                    try: os.remove(os.path.join(目, 名))
                    except Exception: pass
            with open(签, "w", encoding="utf-8") as f:
                f.write(签名)
    except Exception:
        pass


中心编码方式 = str(参数["CENTERQ"]).strip()
中心位 = max(1, int(参数.get("CENTERBITS", 4)))              # 中心码每维位数(5=32电平, 4=16电平)
中心电平 = 1 << 中心位
中心用FP16 = (中心编码方式.lower() in ("fp16", "f16", "half"))   # ★ 中心以 float16 直存(近乎无损)
中心用均匀 = (中心编码方式.lower() in ("gsq", "u", "uniform", "uni"))   # ★ 逐维均匀 + 每维 min/步长
中心用形状 = (中心编码方式.lower() in ("gsqnl", "nl", "shape", "vq"))   # ★ GSQ-NL: 均匀骨架 + 共享 32 电平形状表
中心缓存槽 = max(64, int(参数.get("CENTER缓存", 4096)))        # 中心项缓存的槽数(按单元)
图邻接绝对 = (str(参数.get("GEDGE模式", "自动")).strip() != "增量")   # 2字节时按绝对 id 存(库<=65535)
# ★ 编译期常量签名: 内核里以字面量捕获的配置变了 -> 必须清 numba 缓存, 否则静默复用旧核 ✗
_清QTIP缓存(f"v2|{码型I}|{qtip电平}|{qtip周期}|{qtip分支}|{QTIP_K}|{QTIP_G1}|{QTIP_G2}"
            f"|{中心编码方式}|{中心位}|{int(中心用FP16)}|{int(中心用均匀)}|{int(中心用形状)}"
            f"|{int(图邻接绝对)}|{图ef初}|{自适应系数}|{中心缓存槽}")
# ============================================================================
#  ★ 稀疏48 实验开关(默认 0 = 关闭; 关闭时下面所有分支都不进 -> 逐字节不变 ✓)
#  目的: 测【结构化稀疏(每 4 维丢 2 维 = 768->384)】的召回代价, 即"字节减半"值不值。
#  做法: 存档格式 / 位宽 / 每行字节【完全不动】; 只在编码阶段把被丢维的【中心】与【残差】置 0
#        -> 被丢维的电平 = 离 0 最近的那一档(当前索引 6 ≈ 0.005821) -> 重建 ≈ 0
#        -> 打分时这些维自然贡献 ≈ 0(仍然照常算, 不算算力账)
#  因此 B/行 仍是 244.59; 真省多少字节 = 【另算】的定宽核算(见 _work/稀疏48-报告.md)
# ============================================================================
稀疏48 = int(参数.get("稀疏48", 0) or 0)
稀疏种子 = int(参数.get("稀疏种子", 2024) or 2024)
if 稀疏48 not in (0, 1, 2, 3, 4):
    稀疏48 = 0


def _稀疏掩(维度, 数组=None, 种子=None):
    """-> bool 数组(True=保留, 恒保留 维度//2 维) / None(关闭)
       1/4 = 固定丢 d%4∈{2,3}; 2 = 固定种子随机(可复现); 3 = 按样本逐维能量丢最小的一半"""
    if 稀疏48 <= 0: return None
    维度 = int(维度); 保留数 = max(1, 维度 // 2)
    模式 = 1 if 稀疏48 == 4 else 稀疏48            # ★ 4 与 1 同丢法 -> 差异只归因于"重拟合" ✓
    if 模式 == 3 and 数组 is not None and 数组.shape[0] > 0:
        步 = max(1, int(数组.shape[0]) // 20000)   # 最多统计 2 万行(省内存 · 确定性 ✓)
        样 = np.asarray(数组[::步], np.float32)
        能 = np.einsum("ij,ij->j", 样, 样)         # 逐维能量(旋转后空间, 与编码同一坐标系 ✓)
        丢 = np.argsort(能, kind="stable")[:维度 - 保留数]
        掩 = np.ones(维度, bool); 掩[丢] = False
        return 掩
    if 模式 == 2:
        序 = np.random.default_rng(稀疏种子 if 种子 is None else int(种子)).permutation(维度)
        掩 = np.zeros(维度, bool); 掩[序[:保留数]] = True
        return 掩
    掩 = np.ones(维度, bool)
    掩[2::4] = False; 掩[3::4] = False             # 每 4 维丢 2 维 ✓ 恰好一半
    return 掩


@njit(cache=True, fastmath=True)
def _度打包(度, 位):
    """度数 -> 位 比特/个 的位流, 每 8 个挤 位 字节"""
    n = 度.shape[0]; m = (n + 7) // 8; 每 = 位
    出 = np.zeros(m * 每, np.uint8)
    for b in range(m):
        p = b * 每
        for k in range(8):
            i = b * 8 + k
            if i >= n:
                break
            sh = k * 位
            出[p + (sh >> 3)] |= np.uint8((np.uint32(度[i]) << np.uint32(sh & 7)) & np.uint32(0xFF))
            出[p + (sh >> 3) + 1] |= np.uint8((np.uint32(度[i]) >> np.uint32(8 - (sh & 7))) & np.uint32(0xFF))
    return 出


@njit(cache=True, fastmath=True)
def _度解包(流, 行数, 位):
    """位比特/个 的位流 -> uint8 度数"""
    出 = np.zeros(行数, np.uint8)
    每 = 位; 掩 = np.uint32((1 << 位) - 1)
    for i in range(行数):
        sh = (i % 8) * 每; p = (i // 8) * 每 + (sh >> 3); 内 = sh & 7
        acc = np.uint32(0)
        for q in range(4):
            if p + q < 流.size:
                acc |= np.uint32(流[p + q]) << np.uint32(8 * q)
        出[i] = np.uint8((acc >> np.uint32(内)) & 掩)
    return 出


@njit(cache=True, fastmath=True)
def _单元解包(流, 行数, 位):
    """位比特/个 的位流 -> uint16 单元号"""
    出 = np.zeros(行数, np.uint16)
    每 = 位; 掩 = np.uint32((1 << 位) - 1)
    for i in range(行数):
        sh = (i % 8) * 每; p = (i // 8) * 每 + (sh >> 3); 内 = sh & 7
        acc = np.uint32(0)
        for q in range(4):
            if p + q < 流.size:
                acc |= np.uint32(流[p + q]) << np.uint32(8 * q)
        出[i] = np.uint16((acc >> np.uint32(内)) & 掩)
    return 出
用中心码 = 中心用FP16 or 中心用均匀 or 中心用形状        # fp16 直存 / 均匀 / 共享形状(GSQ-NL) ✓

# ---------------- 打包方案 ----------------
# ---- 通用位流打包(不限定电平数) ----
#   每 流组 个符号 -> 一个 流宽 = ceil(流组·log2(层数)) 位的字段
#   约束: 流宽 <= 16 (核内 3 字节不对齐读), 流组 <= 8 (解码表 <= 512KB)
模式 = "位流"
打包位 = 0
流组 = 0; 流宽 = 0; 每块组数 = 0; 每块字节 = 0
for _组 in range(8, 0, -1):                     # 向下搜: 组内符号越多密度越高, 但必须能字节对齐
    _宽 = int(math.ceil(_组 * math.log2(层数)))
    if _宽 < 1 or _宽 > 16 or 层数 ** _组 > (1 << _宽): continue
    _每块 = 64 // _宽
    while _每块 > 1 and (_每块 * _宽) % 8: _每块 -= 1
    if (_每块 * _宽) % 8 or _每块 * _宽 > 64: continue
    流组, 流宽, 每块组数, 每块字节 = _组, _宽, _每块, _每块 * _宽 // 8
    break
if 流组 == 0:
    raise ValueError(f"index-csd: LEVELS={层数} 无法打包")
流表 = np.zeros((1 << 流宽, 流组), np.uint8)
for _v in range(min(层数 ** 流组, 1 << 流宽)):
    _x = _v
    for _j in range(流组):
        流表[_v, _j] = _x % 层数; _x //= 层数
流表 = np.ascontiguousarray(流表)


def _拟合码本(样本, 层数, 轮=40):
    样本 = np.sort(np.asarray(样本, np.float32).ravel())
    取 = np.unique(样本[np.linspace(0, 样本.size - 1, 层数).astype(np.int64)])
    while 取.size < 层数:
        取 = np.sort(np.concatenate([取, 取 + 1e-7]))
    取 = 取[:层数]
    for _ in range(轮):
        边 = (取[:-1] + 取[1:]) / 2
        标 = np.searchsorted(边, 样本)
        新 = np.array([样本[标 == k].mean() if (标 == k).any() else 取[k] for k in range(层数)], np.float32)
        新 = np.sort(新)                                  # 电平可能交叉 -> 必须重排, 否则分箱错乱
        if np.allclose(新, 取, atol=1e-9):
            取 = 新; break
        取 = 新
    return np.ascontiguousarray(取.astype(np.float32))


def _FPS点(X, m):
    n = X.shape[0]
    if n == 0: return np.zeros((m, X.shape[1]), np.float32)
    if n <= m: return np.concatenate([X, np.repeat(X[-1:], m - n, 0)], 0)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    均 = X.mean(0); 均 = 均 / max(float(np.linalg.norm(均)), 1e-8)
    选 = [int(np.argmin(Xn @ 均))]
    距 = 1.0 - Xn @ Xn[选[0]]
    for _ in range(1, m):
        j = int(np.argmax(距)); 选.append(j)
        距 = np.minimum(距, 1.0 - Xn @ Xn[j])
    return X[np.array(选)]


def _二中心(X):
    """单元内 2-means 的两个质心(FPS 初始化 + 3 轮余弦归属)"""
    n = X.shape[0]
    if n < 4: return X[:1].copy()
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    均 = Xn.mean(0); 均 /= max(float(np.linalg.norm(均)), 1e-8)
    a = int(np.argmin(Xn @ 均)); b = int(np.argmax(Xn @ Xn[a]))
    中心 = np.stack([Xn[a], Xn[b]])
    for _ in range(3):
        标 = (Xn @ 中心.T).argmax(1)
        for j in range(2):
            掩 = 标 == j
            if 掩.any(): 中心[j] = Xn[掩].mean(0); 中心[j] /= max(float(np.linalg.norm(中心[j])), 1e-8)
    标 = (Xn @ 中心.T).argmax(1)
    if (标 == 0).sum() < 2 or (标 == 1).sum() < 2: return X[:1].copy()
    return 中心.astype(np.float32)


def _切子单元(X, s):
    n = X.shape[0]
    if s <= 1 or n <= s:
        return X, np.array([n], np.int64), np.arange(n)
    中心 = _FPS点(X, s).copy()
    for _ in range(4):
        标 = (X @ (中心 / np.maximum(np.linalg.norm(中心, axis=1, keepdims=True), 1e-8)).T).argmax(1)
        for j in range(s):
            掩 = 标 == j
            if 掩.any(): 中心[j] = X[掩].mean(0)
    标 = (X @ (中心 / np.maximum(np.linalg.norm(中心, axis=1, keepdims=True), 1e-8)).T).argmax(1)
    置换 = np.argsort(标, kind="stable")
    行数 = np.bincount(标[置换], minlength=s).astype(np.int64)
    return X[置换], 行数[行数 > 0], 置换


def _打包(码):
    """(n,D) 的 0..LEVELS-1 码 -> (n, 每行字节); 位流模式行尾多 1 字节供不对齐 2 字节读"""
    n, D = 码.shape
    if 模式 == "位平面":
        每字节 = 8 // 打包位
        可用 = (D // 每字节) * 每字节
        v = np.ascontiguousarray(码[:, :可用]).reshape(n, -1, 每字节).astype(np.uint16)
        权 = (1 << (打包位 * np.arange(每字节))).astype(np.uint16)
        出 = (v * 权[None, None, :]).sum(2).astype(np.uint8)
        if 可用 < D:
            尾 = np.zeros((n, 1), np.uint8)
            for j in range(D - 可用):
                尾[:, 0] |= (码[:, 可用 + j].astype(np.uint16) & ((1 << 打包位) - 1)).astype(np.uint8) << (打包位 * j)
            出 = np.concatenate([出, 尾], axis=1)
        return np.ascontiguousarray(出)
    # ---- 位流 ----
    G = int(math.ceil(D / 流组))
    补 = G * 流组 - D
    if 补:
        码 = np.concatenate([码, np.zeros((n, 补), np.uint8)], axis=1)
    v = np.ascontiguousarray(码).reshape(n, G, 流组).astype(np.uint64)
    基 = np.array([层数 ** j for j in range(流组)], dtype=np.uint64)
    w = (v * 基[None, None, :]).sum(2)                       # (n,G) 每组值 < 2^流宽
    块数 = int(math.ceil(G / 每块组数))
    需 = 块数 * 每块组数
    if 需 > G:
        w = np.concatenate([w, np.zeros((n, 需 - G), np.uint64)], 1)
    W = w.reshape(n, 块数, 每块组数)
    权 = np.array([1 << (j * 流宽) for j in range(每块组数)], dtype=np.uint64)
    大 = (W * 权[None, None, :]).sum(2)                       # (n,块数) 每块组数*流宽 <= 64
    字节 = np.empty((n, 块数, 每块字节), np.uint8); 剩 = 大.copy()
    for i in range(每块字节):
        字节[:, :, i] = (剩 & np.uint64(0xFF)).astype(np.uint8)
        剩 >>= np.uint64(8)
    出 = 字节.reshape(n, -1)
    return np.ascontiguousarray(np.concatenate([出, np.zeros((n, 2), np.uint8)], axis=1))


def _每行字节(D):
    if 模式 == "位平面":
        每字节 = 8 // 打包位
        return (D + 每字节 - 1) // 每字节
    G = int(math.ceil(D / 流组))
    return int(math.ceil(G / 每块组数)) * 每块字节 + 2


@njit(cache=True, fastmath=True, parallel=True)
def 球面检索(查询集, 单元块, 单元起, 单元行数, 单元行基, 单元常数, 查询单元偏移,
            块P偏移, 块范数偏移, 块残差偏移, 码流, 残差范数, 全流, 残cap, 全cap, 码本,
            维度, 维度流, 层数, 每行字节, 位宽, 流宽, 流组, 流表, k, 出分, 出行):
    nt = nb.get_num_threads()
    表 = np.empty((nt, 维度, 层数), np.float32)
    最佳分 = np.empty((nt, k), np.float32)
    最佳行 = np.empty((nt, k), np.int64)
    掩 = np.uint8((1 << 位宽) - 1) if 位宽 > 0 else np.uint8(0)
    掩2 = np.uint32((1 << 流宽) - 1) if 流宽 > 0 else np.uint32(0)
    for qi in _prange(查询集.shape[0]):
        t = nb.get_thread_id(); q = 查询集[qi]
        for d in range(维度):
            a = q[d]
            for l in range(层数):
                表[t, d, l] = a * 码本[l]
        for j in range(k):
            最佳分[t, j] = np.float32(-1e30); 最佳行[t, j] = -1
        for u in range(查询单元偏移[qi], 查询单元偏移[qi + 1]):
            b = 单元块[u]; 起 = 单元起[u]; L = 单元行数[u]
            行基 = 单元行基[u]; 常 = 单元常数[u]
            p基 = 块P偏移[b]; 范基 = 块范数偏移[b]; 残基 = 块残差偏移[b]
            for i in range(起, 起 + L):
                p = p基 + i * 每行字节
                acc = np.float32(0.0)
                if 码型I == 1:
                    if QTIP快路:
                        acc = _qtip评分(码流, p, 维度, QTIP_反位, QTIP_子表, QTIP_三表, 表[t])
                    else:
                        acc = _qtip评分通用(码流, p, 维度, QTIP_反位, QTIP_子表, QTIP_三表, 表[t])
                elif 位宽 > 0:
                    每字节 = 8 // 位宽
                    for c in range(每行字节):
                        by = 码流[p + c]
                        for j in range(每字节):
                            acc += 表[t, c * 每字节 + j, (by >> (j * 位宽)) & 掩]
                else:
                    for g2 in range(维度流):
                        bo = g2 * 流宽
                        v2 = ((np.uint32(码流[p + (bo >> 3)]) | (np.uint32(码流[p + (bo >> 3) + 1]) << np.uint32(8))
                               | (np.uint32(码流[p + (bo >> 3) + 2]) << np.uint32(16)))
                              >> np.uint32(bo & 7)) & 掩2
                        for j in range(流组):
                            # ★ 与 _节点分 对齐: 维度 不能整除 流组 时(如 LEVELS=3, 768%7≠0)
                            #   最后一个字节会多出 j → 越界读表 → CPU 扫描档结果错 ✗
                            if g2 * 流组 + j < 维度:
                                acc += 表[t, g2 * 流组 + j, 流表[v2, j]]
                残 = np.float32(残差范数[残基 + i]) * 残cap
                gr = 行基 + i                                          # 全局行号
                bo = gr * 12; by = bo >> 3; sh = np.uint32(bo & 7)
                v12 = ((np.uint32(全流[by]) | (np.uint32(全流[by + 1]) << np.uint32(8))
                        | (np.uint32(全流[by + 2]) << np.uint32(16))) >> sh) & np.uint32(4095)
                全 = np.float32(v12) * 全cap
                s = (常 + 残 * acc) / 全
                if s > 最佳分[t, k - 1]:
                    pos = k - 1
                    while pos > 0 and 最佳分[t, pos - 1] < s:
                        最佳分[t, pos] = 最佳分[t, pos - 1]; 最佳行[t, pos] = 最佳行[t, pos - 1]
                        pos -= 1
                    最佳分[t, pos] = s; 最佳行[t, pos] = 行基 + i          # i 已是块内行号(S=1 时起=0 才侥幸等价)
        for j in range(k):
            出分[qi, j] = 最佳分[t, j]; 出行[qi, j] = 最佳行[t, j]


@njit(cache=True, fastmath=True)
def _qtip行(码流, 基, 维度, 出, 反位, 子表, 索表):
    """QTIP 无链解码(写电平数组, 构建期用): 子集 = 子表[最近 K 个传输比特]
       电平 = 索表[索流字节, 槽]*4 + 子集; 每子=1(4 电平)时子集即电平, 无索流 ✓"""
    分支B = (((维度 // qtip周期) * qtip分支 + min(维度 % qtip周期, qtip分支)) + 7) // 8
    索基 = 基 + 分支B
    idx = 0; q5 = 0; r5 = 0
    for tt in range(维度):
        m = idx if idx < QTIP_K else QTIP_K
        if m == 0:
            j = 子表[0]
        else:
            o = idx - m
            w = (np.int32(码流[基 + (o >> 3)]) | (np.int32(码流[基 + (o >> 3) + 1]) << 8)
                 | (np.int32(码流[基 + (o >> 3) + 2]) << 16))
            j = 子表[反位[(w >> (o & 7)) & QTIP掩] >> (QTIP_K - m)]
        if (tt % qtip周期) < qtip分支:
            idx += 1
        if QTIP每子 > 1:
            出[tt] = 索表[码流[索基 + q5], r5] * 4 + j
        else:
            出[tt] = j                                          # 4 电平: 无索流(不读, 省 96 B/行 ✓)
        r5 += 1
        if r5 == QTIP每字节:
            r5 = 0; q5 += 1
    return 0


@njit(cache=True, fastmath=True, inline="always")
def _qtip评分(码流, 基, 维度, 反位, 子表, 索表, 表行):
    """解码+打分融合, 20 维展开, 全静态偏移(寄存器 = 最近 K 个传输比特, 无串行链)
       ★ 仅当 QTIP快路(每字节=5 符号 + 周期4/分支3 -> 每 20 维 15 分支位/4 索字节)成立时使用"""
    分支B = (((维度 // qtip周期) * qtip分支 + min(维度 % qtip周期, qtip分支)) + 7) // 8
    索基 = 基 + 分支B
    a0 = np.float32(0.0); a1 = np.float32(0.0); a2 = np.float32(0.0); a3 = np.float32(0.0)
    块数 = 维度 // 20
    idx = 0; q5 = 0; r5 = 0
    头 = 20 if 块数 else 维度
    for tt in range(头):
        m = idx if idx < QTIP_K else QTIP_K
        if m == 0:
            j = 子表[0]
        else:
            o = idx - m
            w0 = (np.int32(码流[基 + (o >> 3)]) | (np.int32(码流[基 + (o >> 3) + 1]) << 8)
                  | (np.int32(码流[基 + (o >> 3) + 2]) << 16))
            j = 子表[反位[(w0 >> (o & 7)) & QTIP掩] >> (QTIP_K - m)]
        if (tt % qtip周期) < qtip分支:
            idx += 1
        lv = 索表[码流[索基 + q5], r5] * 4 + j
        mm = tt & 3
        if mm == 0:
            a0 += 表行[tt, lv]
        elif mm == 1:
            a1 += 表行[tt, lv]
        elif mm == 2:
            a2 += 表行[tt, lv]
        else:
            a3 += 表行[tt, lv]
        r5 += 1
        if r5 == QTIP每字节:
            r5 = 0; q5 += 1
    for blk in range(1, 块数):
        o20 = blk * 20
        ib = blk * 15
        o = (ib - QTIP_K) >> 3
        sh = (ib - QTIP_K) & 7
        w = (np.int32(码流[基 + o]) | (np.int32(码流[基 + o + 1]) << 8)
             | (np.int32(码流[基 + o + 2]) << 16) | (np.int32(码流[基 + o + 3]) << 24))
        q20 = blk * 4
        j = 子表[反位[(w >> (sh + 0)) & QTIP掩]]
        a0 += 表行[o20 + 0, 索表[码流[索基 + q20 + 0], 0] * 4 + j]
        j = 子表[反位[(w >> (sh + 1)) & QTIP掩]]
        a1 += 表行[o20 + 1, 索表[码流[索基 + q20 + 0], 1] * 4 + j]
        j = 子表[反位[(w >> (sh + 2)) & QTIP掩]]
        a2 += 表行[o20 + 2, 索表[码流[索基 + q20 + 0], 2] * 4 + j]
        j = 子表[反位[(w >> (sh + 3)) & QTIP掩]]
        a3 += 表行[o20 + 3, 索表[码流[索基 + q20 + 0], 3] * 4 + j]
        j = 子表[反位[(w >> (sh + 3)) & QTIP掩]]
        a0 += 表行[o20 + 4, 索表[码流[索基 + q20 + 0], 4] * 4 + j]
        j = 子表[反位[(w >> (sh + 4)) & QTIP掩]]
        a1 += 表行[o20 + 5, 索表[码流[索基 + q20 + 1], 0] * 4 + j]
        j = 子表[反位[(w >> (sh + 5)) & QTIP掩]]
        a2 += 表行[o20 + 6, 索表[码流[索基 + q20 + 1], 1] * 4 + j]
        j = 子表[反位[(w >> (sh + 6)) & QTIP掩]]
        a3 += 表行[o20 + 7, 索表[码流[索基 + q20 + 1], 2] * 4 + j]
        j = 子表[反位[(w >> (sh + 6)) & QTIP掩]]
        a0 += 表行[o20 + 8, 索表[码流[索基 + q20 + 1], 3] * 4 + j]
        j = 子表[反位[(w >> (sh + 7)) & QTIP掩]]
        a1 += 表行[o20 + 9, 索表[码流[索基 + q20 + 1], 4] * 4 + j]
        j = 子表[反位[(w >> (sh + 8)) & QTIP掩]]
        a2 += 表行[o20 + 10, 索表[码流[索基 + q20 + 2], 0] * 4 + j]
        j = 子表[反位[(w >> (sh + 9)) & QTIP掩]]
        a3 += 表行[o20 + 11, 索表[码流[索基 + q20 + 2], 1] * 4 + j]
        j = 子表[反位[(w >> (sh + 9)) & QTIP掩]]
        a0 += 表行[o20 + 12, 索表[码流[索基 + q20 + 2], 2] * 4 + j]
        j = 子表[反位[(w >> (sh + 10)) & QTIP掩]]
        a1 += 表行[o20 + 13, 索表[码流[索基 + q20 + 2], 3] * 4 + j]
        j = 子表[反位[(w >> (sh + 11)) & QTIP掩]]
        a2 += 表行[o20 + 14, 索表[码流[索基 + q20 + 2], 4] * 4 + j]
        j = 子表[反位[(w >> (sh + 12)) & QTIP掩]]
        a3 += 表行[o20 + 15, 索表[码流[索基 + q20 + 3], 0] * 4 + j]
        j = 子表[反位[(w >> (sh + 12)) & QTIP掩]]
        a0 += 表行[o20 + 16, 索表[码流[索基 + q20 + 3], 1] * 4 + j]
        j = 子表[反位[(w >> (sh + 13)) & QTIP掩]]
        a1 += 表行[o20 + 17, 索表[码流[索基 + q20 + 3], 2] * 4 + j]
        j = 子表[反位[(w >> (sh + 14)) & QTIP掩]]
        a2 += 表行[o20 + 18, 索表[码流[索基 + q20 + 3], 3] * 4 + j]
        j = 子表[反位[(w >> (sh + 15)) & QTIP掩]]
        a3 += 表行[o20 + 19, 索表[码流[索基 + q20 + 3], 4] * 4 + j]
    # ---- 尾部: 维度 % 20 的余数维(768 % 20 = 8) ----
    idx = 块数 * 15
    q5 = 块数 * 4
    r5 = 0
    tt = 块数 * 20
    while 块数 > 0 and tt < 维度:            # ★ 块数=0 时头部已走完全部维度 -> 尾部不可再走 ✗
        m = idx if idx < QTIP_K else QTIP_K
        if m == 0:
            j = 子表[0]
        else:
            o = idx - m
            w0 = (np.int32(码流[基 + (o >> 3)]) | (np.int32(码流[基 + (o >> 3) + 1]) << 8)
                  | (np.int32(码流[基 + (o >> 3) + 2]) << 16))
            j = 子表[反位[(w0 >> (o & 7)) & QTIP掩] >> (QTIP_K - m)]
        if (tt % qtip周期) < qtip分支:
            idx += 1
        lv = 索表[码流[索基 + q5], r5] * 4 + j
        mm = tt & 3
        if mm == 0:
            a0 += 表行[tt, lv]
        elif mm == 1:
            a1 += 表行[tt, lv]
        elif mm == 2:
            a2 += 表行[tt, lv]
        else:
            a3 += 表行[tt, lv]
        r5 += 1
        if r5 == QTIP每字节:
            r5 = 0; q5 += 1
        tt += 1
    return (a0 + a1) + (a2 + a3)


@njit(cache=True, fastmath=True, inline="always")
def _qtip评分通用(码流, 基, 维度, 反位, 子表, 索表, 表行):
    """通用解码+打分(不展开): 任意 每字节符号数 / 周期 / 分支 / 每子
       ★ 只有 QTIP快路(=5符号/字节 + 周期4/分支3, 即 12 电平默认档)走 20 维展开核;
         其余(4/8/16 电平 或 改了周期/分支)走这条 -> 正确性优先, 与 _qtip评分 同一电平语义 ✓"""
    分支B = (((维度 // qtip周期) * qtip分支 + min(维度 % qtip周期, qtip分支)) + 7) // 8
    索基 = 基 + 分支B
    a0 = np.float32(0.0); a1 = np.float32(0.0); a2 = np.float32(0.0); a3 = np.float32(0.0)
    idx = 0; q5 = 0; r5 = 0
    for tt in range(维度):
        m = idx if idx < QTIP_K else QTIP_K
        if m == 0:
            j = 子表[0]
        else:
            o = idx - m
            w0 = (np.int32(码流[基 + (o >> 3)]) | (np.int32(码流[基 + (o >> 3) + 1]) << 8)
                  | (np.int32(码流[基 + (o >> 3) + 2]) << 16))
            j = 子表[反位[(w0 >> (o & 7)) & QTIP掩] >> (QTIP_K - m)]
        if (tt % qtip周期) < qtip分支:
            idx += 1
        if QTIP每子 > 1:
            lv = 索表[码流[索基 + q5], r5] * 4 + j
        else:
            lv = j                                          # 4 电平: 子集即电平, 无索流 ✓
        mm = tt & 3
        if mm == 0:
            a0 += 表行[tt, lv]
        elif mm == 1:
            a1 += 表行[tt, lv]
        elif mm == 2:
            a2 += 表行[tt, lv]
        else:
            a3 += 表行[tt, lv]
        r5 += 1
        if r5 == QTIP每字节:
            r5 = 0; q5 += 1
    return (a0 + a1) + (a2 + a3)


@njit(cache=True, fastmath=True, inline="always")
def _取邻(邻接, 邻接字节, j, 本节点):
    """模拟式: 3字节=绝对id; 2字节=有符号差值"""
    基 = j * 邻接字节
    if 邻接字节 == 2:
        v = np.int32(邻接[基]) | (np.int32(邻接[基 + 1]) << np.int32(8))
        if 图邻接绝对:
            return v                                   # 2 字节绝对 id(库 <= 65535)
        return 本节点 + v - np.int32(32768)            # 2 字节有符号增量
    v = (np.int32(邻接[基]) | (np.int32(邻接[基 + 1]) << np.int32(8))
         | (np.int32(邻接[基 + 2]) << np.int32(16)))
    if 邻接字节 >= 4:
        v = v | (np.int32(邻接[基 + 3]) << np.int32(24))
    return v


# ---------------- 无损压缩(存档用) ----------------
# 实测最优(50k 库 ✓): 码流 zlib 0.9910 / 图邻接 lzma 0.8459 / 节点单元压 zlib 0.1104(≈lzma)
#                     全流 bz2 0.5933 / 残差范数 lzma 0.8137 / 映射流 lzma 0.8554 / 图度压 lzma 0.9024
压缩方案 = (("码流", "zlib"), ("图邻接", "lzma"), ("节点单元压", "zlib"), ("全流", "bz2"),
            ("残差范数", "lzma"), ("映射流", "lzma"), ("图度压", "lzma"))
压缩法号 = {"": 0, "zlib": 1, "lzma": 2, "bz2": 3}
压缩法名 = {0: "", 1: "zlib", 2: "lzma", 3: "bz2"}

def 压数组(a, 法):
    """-> (压缩体 uint8, 原长) ; 法为空或压不动就返回 (None, 0)"""
    import zlib as _zl, lzma as _lz, bz2 as _bz
    if not 法: return None, 0
    b = np.ascontiguousarray(a).tobytes()
    try:
        if 法 == "zlib": c = _zl.compress(b, 9)
        elif 法 == "lzma": c = _lz.compress(b, preset=9 | _lz.PRESET_EXTREME)
        elif 法 == "bz2": c = _bz.compress(b, 9)
        else: return None, 0
    except Exception:
        return None, 0
    if len(c) >= len(b): return None, 0                    # 压不动就不压 ✓
    return np.frombuffer(c, np.uint8).copy(), len(b)

def _压存(o, 名):
    """_状态() 用: 返回 {名压: 体, 名原长: 长, 名法: 法号, 名形: 形状, 名型: dtype} 或 {名: 原数组}"""
    a = getattr(o, 名, None)
    if a is None: return {名: None}
    a = np.asarray(a)
    法 = dict(压缩方案).get(名, "")
    体, 长 = (None, 0) if (法 == "" or a.size == 0) else 压数组(a, 法)
    if 体 is None: return {名: a}
    return {名: None, 名 + "压": 体, 名 + "原长": np.int64(长),
            名 + "法": np.int64(压缩法号.get(法, 0)),
            名 + "形": np.asarray(a.shape, np.int64), 名 + "型": str(a.dtype)}

def 解压数组(体, 原长, 法号, dtype, shape):
    import zlib as _zl, lzma as _lz, bz2 as _bz
    法 = 压缩法名.get(int(法号), "")
    b = np.ascontiguousarray(体, np.uint8).tobytes()
    if 法 == "zlib": b = _zl.decompress(b)
    elif 法 == "lzma": b = _lz.decompress(b)
    elif 法 == "bz2": b = _bz.decompress(b)
    return np.frombuffer(b, np.dtype(dtype)).reshape(shape).copy()


import os as _os, time as _time
_PROF = bool(_os.environ.get("CBCT_PROF"))          # ★ CBCT_PROF=1 才计时 ✓ 默认零影响 ✓

def _计时记录(对象, 名, t0):
    if not _PROF: return
    d = getattr(对象, "_prof", None)
    if d is None:
        d = {}; setattr(对象, "_prof", d)
    a = d.get(名)
    if a is None: d[名] = [_time.perf_counter() - t0, 1]
    else: a[0] += _time.perf_counter() - t0; a[1] += 1

@njit(cache=True, fastmath=True, inline="always")
def _节点分(节点, q, 表t, 节点单元, 中心码流, 中心fp16, 中心lo, 中心步, 中心幅, 中心形状, 码流, 残差范数, 全流,
            残cap, 全cap, 维度, 每行字节, 位宽, 流宽, 流组, 流表, 层数,
            缓存签, 缓存分, 缓存槽, 单元码基, 单元全局起, t):
    """与 球面检索 同一套寻址与打分: (⟨q,c_u⟩ + ‖r‖·Σ LUT[码]) / ‖x̂‖"""
    u0 = np.int32(节点单元[节点])
    维基 = 单元码基[u0] + (节点 - 单元全局起[u0]) * 每行字节
    acc = np.float32(0.0)
    if 码型I == 1:
        if QTIP快路:
            acc = _qtip评分(码流, 维基, 维度, QTIP_反位, QTIP_子表, QTIP_三表, 表t)
        else:
            acc = _qtip评分通用(码流, 维基, 维度, QTIP_反位, QTIP_子表, QTIP_三表, 表t)
    elif 位宽 > 0:
        掩 = np.uint8((1 << 位宽) - 1); 每字节 = 8 // 位宽
        for c in range(每行字节):
            by = 码流[维基 + c]
            for j in range(每字节):
                acc += 表t[c * 每字节 + j, (by >> (j * 位宽)) & 掩]
    else:
        掩2 = np.uint32((1 << 流宽) - 1)
        维度流 = (维度 + 流组 - 1) // 流组
        for g2 in range(维度流):
            bo = g2 * 流宽
            v2 = ((np.uint32(码流[维基 + (bo >> 3)]) | (np.uint32(码流[维基 + (bo >> 3) + 1]) << np.uint32(8))
                   | (np.uint32(码流[维基 + (bo >> 3) + 2]) << np.uint32(16))) >> np.uint32(bo & 7)) & 掩2
            for j in range(流组):
                if g2 * 流组 + j < 维度:
                    acc += 表t[g2 * 流组 + j, 流表[v2, j]]
    残 = np.float32(残差范数[节点]) * 残cap
    u = u0
    槽 = u % 缓存槽
    常 = np.float32(0.0)
    if 缓存签[t, 槽] == u:
        常 = 缓存分[t, 槽]
    elif 中心用FP16:                                     # ★ fp16 中心: 直接读, 无位解包/无码本 ✓
        常 = np.float32(0.0)
        for d in range(维度):
            常 += q[d] * np.float32(中心fp16[u, d])
        缓存签[t, 槽] = u; 缓存分[t, 槽] = 常
    elif 中心用形状:                                     # ★ GSQ-NL: lo + 形状表[码] × 幅 ✓ (同一条乘加 ✓)
        常 = np.float32(0.0)
        for d in range(维度):
            bo = (u * 维度 + d) * 中心位; by = bo >> 3
            v5 = ((np.uint32(中心码流[by]) | (np.uint32(中心码流[by + 1]) << np.uint32(8))
                   | (np.uint32(中心码流[by + 2]) << np.uint32(16))) >> np.uint32(bo & 7)) & np.uint32(中心电平 - 1)
            常 += q[d] * (中心lo[d] + 中心形状[v5] * 中心幅[d])   # 参数 ✓
        缓存签[t, 槽] = u; 缓存分[t, 槽] = 常
    elif 中心用均匀:                                     # ★ 均匀中心: 解 5bit 码 -> lo + 码*步 ✓
        常 = np.float32(0.0)
        for d in range(维度):
            bo = (u * 维度 + d) * 中心位; by = bo >> 3
            v5 = ((np.uint32(中心码流[by]) | (np.uint32(中心码流[by + 1]) << np.uint32(8))
                   | (np.uint32(中心码流[by + 2]) << np.uint32(16))) >> np.uint32(bo & 7)) & np.uint32(中心电平 - 1)
            常 += q[d] * (中心lo[d] + np.float32(v5) * 中心步[d])
        缓存签[t, 槽] = u; 缓存分[t, 槽] = 常
    else:                                                # 兜底(用中心码=False 时) ✓
        常 = np.float32(0.0)
        缓存签[t, 槽] = u; 缓存分[t, 槽] = 常
    bo = 节点 * 12; by = bo >> 3
    v12 = ((np.uint32(全流[by]) | (np.uint32(全流[by + 1]) << np.uint32(8))
            | (np.uint32(全流[by + 2]) << np.uint32(16))) >> np.uint32(bo & 7)) & np.uint32(4095)
    全 = np.float32(v12) * 全cap
    return (常 + 残 * acc) / 全


@njit(cache=True, fastmath=True, parallel=True)
def 图检索(查询集, 邻接, 邻接字节, 偏移, 入口, 图层, 图最大层, 每层槽, 累计槽, 行数,
           节点单元, 单元码基, 单元全局起, 中心码流, 码流, 残差范数, 全流, 残cap, 全cap, 码本,
           维度, 层数, 每行字节, 位宽, 流宽, 流组, 流表, k, ef, 边距, 出分, 出行, 中心fp16,
           中心lo, 中心步, 中心幅, 中心形状):
    nt = nb.get_num_threads()
    表 = np.empty((nt, 维度, 层数), np.float32)
    缓存签 = np.full((nt, 中心缓存槽), -1, np.int32); 缓存分 = np.zeros((nt, 中心缓存槽), np.float32)
    访问 = np.zeros((nt, 行数), np.int32); 代 = np.zeros(nt, np.int32)
    候选 = np.empty((nt, ef * 8 + 64), np.int32); 候分 = np.empty((nt, ef * 8 + 64), np.float32)
    结果 = np.empty((nt, ef), np.int32); 结分 = np.empty((nt, ef), np.float32)
    for qi in _prange(查询集.shape[0]):
        t = nb.get_thread_id(); q = 查询集[qi]
        for d in range(维度):
            a = q[d]
            for l in range(层数):
                表[t, d, l] = a * 码本[l]
        for s in range(中心缓存槽):
            缓存签[t, s] = -1
        代[t] += 1
        ef现 = k if k > 图ef初 else 图ef初          # 起始 ef 必须 >= k(否则结果表装不下)
        if ef现 > ef: ef现 = ef                     # ★ 结果表容量 = ef, 绝不能超过(否则越界写 ✗)
        当前 = 入口
        for lv in range(图最大层, 0, -1):                       # HNSW 上层贪心下降(NSG 时循环不执行)
            槽起 = 累计槽[lv]; 槽数 = 每层槽[lv]
            改 = True
            while 改:
                改 = False
                s0 = _节点分(当前, q, 表[t], 节点单元, 中心码流, 中心fp16, 中心lo, 中心步, 中心幅, 中心形状, 码流, 残差范数, 全流,
                             残cap, 全cap, 维度, 每行字节, 位宽, 流宽, 流组, 流表, 层数,
                             缓存签, 缓存分, 中心缓存槽, 单元码基, 单元全局起, t)
                for j in range(偏移[当前] + 槽起, 偏移[当前] + 槽起 + 槽数):
                    邻 = _取邻(邻接, 邻接字节, j, 当前)
                    if 邻 < 0 or 邻 >= 行数: continue
                    s1 = _节点分(邻, q, 表[t], 节点单元, 中心码流, 中心fp16, 中心lo, 中心步, 中心幅, 中心形状, 码流, 残差范数, 全流,
                                 残cap, 全cap, 维度, 每行字节, 位宽, 流宽, 流组, 流表, 层数,
                                 缓存签, 缓存分, 中心缓存槽, 单元码基, 单元全局起, t)
                    if s1 > s0:
                        当前 = 邻; s0 = s1; 改 = True
        s0 = _节点分(当前, q, 表[t], 节点单元, 中心码流, 中心fp16, 中心lo, 中心步, 中心幅, 中心形状, 码流, 残差范数, 全流,
                     残cap, 全cap, 维度, 每行字节, 位宽, 流宽, 流组, 流表, 层数, 缓存签, 缓存分, 中心缓存槽, 单元码基, 单元全局起, t)
        访问[t, 当前] = 代[t]
        候选[t, 0] = 当前; 候分[t, 0] = s0; nc = 1
        结果[t, 0] = 当前; 结分[t, 0] = s0; nr = 1
        while nc > 0:                                          # level0 best-first
            最 = 0
            for j in range(1, nc):
                if 候分[t, j] > 候分[t, 最]: 最 = j
            # ★ 边距改成【相对量纲】: 结分 - 边距*|结分| ✓
            #   原因: 绝对边距在分数尺度不同时失效 ✗ (QTIP 的分数 ≈2.4e-4
            #   而 LM ≈1 → 绝对边距对 QTIP 相对放大 4000 倍 → 判据恒真 →
            #   ef 每次都翻倍到上限 → 遍历量暴涨 → 实测 372k 慢 600 倍 ✗✗)
            #   对 LM(分数 ~1) 行为等价 ✓ 向后兼容 ✓
            if 图ef初 > 0 and nr >= k and ef现 < ef and 候分[t, 最] > 结分[t, nr - 1] - 边距 * abs(结分[t, nr - 1]):
                ef现 = ef现 * 2                       # 队首仍可能改进 -> 加宽(eff 自适应)
                if ef现 > ef: ef现 = ef
            if 自适应系数 > 0.0 and nr > 0:
                # 动态早停: 队首(最好候选)已低于结果下界 - 自适应边距 -> 不可能再改善
                _跨度 = 结分[t, 0] - 结分[t, nr - 1]
                # ★ 边距必须【相对量纲】✓ 否则分数尺度小的编码器(如 QTIP ≈2.4e-4)
                #   这里恒为假 → 早停永不触发 → 遍历暴涨 (实测 372k 慢 600 倍 ✗✗)
                _界 = 结分[t, nr - 1]
                if 候分[t, 最] < _界 - 边距 * (abs(_界) + 自适应系数 * _跨度):
                    nc = 0
                    continue
            现 = 候选[t, 最]; nc -= 1; 候选[t, 最] = 候选[t, nc]; 候分[t, 最] = 候分[t, nc]
            for j in range(偏移[现], 偏移[现 + 1]):
                邻 = _取邻(邻接, 邻接字节, j, 现)
                if 邻 < 0 or 邻 >= 行数 or 访问[t, 邻] == 代[t]: continue
                访问[t, 邻] = 代[t]
                s = _节点分(邻, q, 表[t], 节点单元, 中心码流, 中心fp16, 中心lo, 中心步, 中心幅, 中心形状, 码流, 残差范数, 全流,
                            残cap, 全cap, 维度, 每行字节, 位宽, 流宽, 流组, 流表, 层数,
                            缓存签, 缓存分, 中心缓存槽, 单元码基, 单元全局起, t)
                if nr < ef现 or s > 结分[t, nr - 1] - 边距:
                    if nr < ef现: nr += 1
                    位p = nr - 1
                    while 位p > 0 and 结分[t, 位p - 1] < s:
                        结分[t, 位p] = 结分[t, 位p - 1]; 结果[t, 位p] = 结果[t, 位p - 1]; 位p -= 1
                    结分[t, 位p] = s; 结果[t, 位p] = 邻
                    if nc < 候选.shape[1]:
                        候选[t, nc] = 邻; 候分[t, nc] = s; nc += 1
        for j in range(k):
            if j < nr:
                出分[qi, j] = 结分[t, j]; 出行[qi, j] = np.int64(结果[t, j])
            else:
                出分[qi, j] = np.float32(-1e30); 出行[qi, j] = np.int64(-1)


@njit(cache=True, fastmath=True, parallel=True)
def _反向重剪(邻接, 度, 数组, 度上限, alpha):
    """把反向边并入候选后重剪: 候选 = 出边邻居 ∪ 指向本节点的邻居, 按与自身的相似度降序做多样性剪枝"""
    行数 = 数组.shape[0]; 维 = 数组.shape[1]
    入 = np.zeros(行数, np.int64)
    for i in range(行数):
        for b in range(度[i]):
            j = 邻接[i, b]
            if j >= 0: 入[j] += 1
    起 = np.zeros(行数 + 1, np.int64)
    起[1:] = np.cumsum(入)
    填 = 起[:行数].copy(); 反 = np.empty(int(起[行数]), np.int32)
    for i in range(行数):
        for b in range(度[i]):
            j = 邻接[i, b]
            if j >= 0:
                反[填[j]] = i; 填[j] += 1
    新 = np.full((行数, 度上限), -1, np.int32); 新度 = np.zeros(行数, np.int32)
    for i in _prange(行数):
        n = 0
        候 = np.empty(度[i] + 入[i] + 1, np.int32); 似 = np.empty(度[i] + 入[i] + 1, np.float32)
        for b in range(度[i]):
            if 邻接[i, b] >= 0:
                候[n] = 邻接[i, b]; n += 1
        for b in range(起[i], 起[i + 1]):
            候[n] = 反[b]; n += 1
        for a in range(n):
            ip = np.float32(0.0)
            for d in range(维): ip += 数组[i, d] * 数组[候[a], d]
            似[a] = ip
        for a in range(1, n):                                   # 按相似度降序插入排序
            kv = 似[a]; kc = 候[a]; b2 = a - 1
            while b2 >= 0 and 似[b2] < kv:
                似[b2 + 1] = 似[b2]; 候[b2 + 1] = 候[b2]; b2 -= 1
            似[b2 + 1] = kv; 候[b2 + 1] = kc
        m2 = 0
        for a in range(n):
            c = 候[a]
            if c == i: continue
            重 = False
            for b2 in range(m2):
                if 新[i, b2] == c: 重 = True; break
            if 重: continue
            好 = True
            for b2 in range(m2):
                r = 新[i, b2]
                ip = np.float32(0.0)
                for d in range(维): ip += 数组[c, d] * 数组[r, d]
                if ip > np.float32(1.0) - alpha * (np.float32(1.0) - 似[a]):
                    好 = False; break
            if 好:
                新[i, m2] = c; m2 += 1
                if m2 >= 度上限: break
        新度[i] = m2
    return 新, 新度


def _重归属(数组, 块表, 单元行, 中心表, 码本, 边, 层数):
    """量化感知重归属: 块内每行在候选单元里选【量化重建误差最小】的那个
       重建 = 中心 + ‖x−中心‖·码本[量化方向]; 返回新的 (单元行, 中心表), 并就地重排块内行"""
    每块 = {}
    for ui, u in enumerate(单元行):
        每块.setdefault(u[0], []).append(ui)
    单元行新 = []; 中心表新 = []
    for bi, g in enumerate(块表):
        本 = 每块.get(bi, [])
        if len(本) <= 1:
            for ui in 本:
                单元行新.append(单元行[ui]); 中心表新.append(中心表[ui])
            continue
        误差 = np.empty((len(本), g.size), np.float32)
        for t, ui in enumerate(本):
            c = 中心表[ui]
            r = 数组[g] - c[None, :]
            nr = np.maximum(np.linalg.norm(r, axis=1), 1e-8)
            码q = np.searchsorted(边, r / nr[:, None], side="right").clip(0, 层数 - 1)
            重 = c[None, :] + 码本[码q] * nr[:, None]
            误差[t] = ((数组[g] - 重) ** 2).sum(1)
        选 = 误差.argmin(0)
        新g = np.empty_like(g); 起 = 0
        for t, ui in enumerate(本):
            掩 = 选 == t
            if not 掩.any(): continue
            行 = g[掩]
            新g[起:起 + 行.size] = 行
            单元行新.append((bi, 起, int(行.size), 行))
            中心表新.append(中心表[ui])
            起 += int(行.size)
        块表[bi] = 新g
    return 单元行新, 中心表新


@njit(cache=True, fastmath=True, parallel=True)
def _剪枝图(候选, 相似, 数组, 度上限, alpha):
    """多样性剪枝: 按相似度降序遍历候选 c, 保留 c 当且仅当 c 与已选邻居 r 都满足
       dist(c,r) > alpha * dist(c,i)  (单位向量下 dist = 1 - 内积)"""
    行数 = 数组.shape[0]; 维 = 数组.shape[1]; K = 候选.shape[1]
    邻接 = np.full((行数, 度上限), -1, np.int32)
    度 = np.zeros(行数, np.int32)
    for i in _prange(行数):
        n = 0
        for j in range(K):
            c = 候选[i, j]
            if c < 0 or c == i: continue
            好 = True
            for t in range(n):
                r = 邻接[i, t]
                ip = np.float32(0.0)
                for d in range(维): ip += 数组[c, d] * 数组[r, d]
                if ip > np.float32(1.0) - alpha * (np.float32(1.0) - 相似[i, j]):
                    好 = False; break
            if 好:
                邻接[i, n] = c; n += 1
                if n >= 度上限: break
        度[i] = n
    return 邻接, 度


@njit(cache=True, fastmath=True, parallel=True)
def ADC重排(原行, 基础, 误差, 新范数, 码, LUT, M):
    """VQ 残差 ADC 重排: vq = Σ_m LUT[i, m, 码[原行, m]]; s = (基础 + 误差*vq)/新范数"""
    候选 = 原行.shape[1]
    出 = np.empty((原行.shape[0], 候选), np.float32)
    for i in _prange(原行.shape[0]):
        for j in range(候选):
            u = 原行[i, j]
            if u < 0:
                出[i, j] = np.float32(-1e30)
                continue
            s = np.float32(0.0)
            for m in range(M):
                s += LUT[i, m, 码[u, m]]
            出[i, j] = (基础[i, j] + 误差[i, j] * s) / 新范数[i, j]
    return 出


class IndexCSDMoE:
    """球面 + EM 子单元中心 + N 电平 (+ 可选 VQ 残差重排)"""

    def __init__(self, app=None, 路由: bool = True):
        self.应用 = app; self.路由 = 路由; self.模式 = "IndexCSDMoE"
        self.层数 = 层数; self.打包位 = 打包位; self.流宽 = 流宽; self.流组 = 流组
        self.子单元数 = 子单元数; self.维度 = 0; self.每行字节 = 0
        self.码本 = None; self.边 = None
        self.cell中心 = None; self.cell中心范数 = None
        self.cell块表 = []; self.块归属 = []; self.向量库 = []
        self.块行 = []
        self.映射表 = None; self.块起始 = np.empty(0, np.int64)
        self.映射流 = None; self.映射位宽 = 0; self.映射行数 = 0
        self.块P偏移 = None; self.块范数偏移 = None; self.块残差偏移 = None
        self.码流 = None; self.残差范数 = None; self.全部范数 = None
        self.全流 = None; self.全行数 = 0
        self.残cap = 1.0; self.全cap = 1.0
        self.单元中心 = None; self.单元中心幅 = None; self.单元代表 = None; self.每单元代表数 = 1
        self.单元行数 = []; self.单元块 = []; self.单元起表 = None; self.单元行数表 = None
        self.PQ码本 = None; self.PQ码 = None; self.误差范数 = None; self.新范数 = None
        self.码本列表 = None; self.RVQ = False
        self.图邻接 = None; self.图偏移 = None; self.图入口 = 0
        self.图度 = None; self.图邻接字节 = 3; self.节点单元 = None
        self.单元次代表 = None; self.单元次幅 = None
        self.单元码基 = None; self.单元全局起 = None
        self.图层 = None; self.图类型 = ""
        self.QTIP流 = False                                # 码流是否 QTIP 格式(构建重编码后置 True)
        self.PCA_均值 = None; self.PCA_主成分 = None
        self.中心码流 = None; self.中心行数 = 0
        self.投影中心 = None; self.投影P = None; self.投影均值 = None
        self.投影尺度 = None; self.投影幅 = None
        self._正旋转 = None; self._逆旋转 = None

    def _取旋转(self, 维度):
        if getattr(self, "_正旋转", None) is None:
            self._正旋转, self._逆旋转 = importlib.import_module("mods.quant-gsq_nl").正交旋转(int(维度), 块大小)
        return self._正旋转

    def 正旋转(self, 数组):
        return self._取旋转(数组.shape[1])(np.ascontiguousarray(数组, np.float32))

    def _降维(self, 数组):
        if self.PCA_主成分 is None: return np.ascontiguousarray(数组, np.float32)
        return np.ascontiguousarray((数组 - self.PCA_均值) @ self.PCA_主成分.T, np.float32)

    def train(self, *args, **kwargs):
        return self

    # ---------------- 构建 ----------------
    def add(self, vectors):
        self._归表()                                       # ★ 读档后增量并入: 先把 cupy 表换回 numpy
        _bt = _time.perf_counter() if _PROF else 0.0          # ★ 构建分段计时起点
        _原始 = np.atleast_2d(_主(vectors))                 # ★ 稀疏48=4 判别名(无旋转时 正旋转 原样返回)
        数组 = self.正旋转(_原始)                            # ★ cupy 输入 -> host numpy
        if self.码本 is not None:
            if getattr(self, "PCA_主成分", None) is not None:
                数组 = self._降维(数组)                 # ★ 增量必须走同一套 PCA(否则维度对不上 ✗)
            return self._增量(数组)                     # 增量并入(方向码本固定)
        行数, 维度原 = 数组.shape
        if 用PCA and 维度原 > Pca维:
            # 纯 PCA(无自适应温度): 样本拟合均值 + 前 Pca维 主成分, 再套用全量
            样 = 数组[np.random.choice(行数, min(20000, 行数), replace=False)]
            均 = 样.mean(0).astype(np.float32)
            _, _, Vt = np.linalg.svd(样 - 均, full_matrices=False)
            self.PCA_均值 = 均
            self.PCA_主成分 = np.ascontiguousarray(Vt[:Pca维], np.float32)
            数组 = self._降维(数组)
        维度 = self.维度 = 数组.shape[1]
        # ★ 稀疏48: 掩码按【最终维度】算(True=保留); 模式4 先把输入丢维置 0 -> 下游 IVF/EM/码本自然重拟合 ✓
        _稀疏保 = _稀疏掩(维度, 数组, 稀疏种子)
        if _稀疏保 is not None:
            if 稀疏48 == 4:
                if 数组 is _原始: 数组 = 数组.copy()        # 无旋转时必须拷, 免得改到调用方的数据 ✗
                if not 数组.flags["C_CONTIGUOUS"]: 数组 = np.ascontiguousarray(数组, np.float32)
                数组[:, ~_稀疏保] = 0.0
        self.每行字节 = _每行字节(维度)
        if self.路由 and 行数 >= 最小向量数:
            nlist = max(1, min(int(Nlist因子 * np.sqrt(max(1, 行数))), 行数 // 8))
            归一 = np.ascontiguousarray(数组, np.float32).copy(); faiss.normalize_L2(归一)
            引 = faiss.IndexIVFFlat(faiss.IndexFlatIP(维度), 维度, nlist, faiss.METRIC_INNER_PRODUCT)
            引.train(归一); 引.add(归一)
            _, 归 = 引.quantizer.search(归一, 1)
            归属 = 归.ravel().astype(np.int64); 归属[归属 < 0] = 0
            中心 = 引.quantizer.reconstruct_n(0, nlist).astype(np.float32)
            范式 = np.maximum(np.linalg.norm(中心, axis=1, keepdims=True), 1e-8)
            self.cell中心 = None          # 不在打分路径(路由用单元中心) -> 不存, 省 5 B/行
            self.cell中心范数 = None
            # cell大小 已删: 全代码无读取处(构建期遗留 ✗ 且 = 块行数 ✓ 可推导 ✓)
            self.cell块表 = [[] for _ in range(nlist)]; self.块归属 = []
            del 引, 归一
        else:
            归属 = None
        if 归属 is None:
            组边界 = [np.arange(起, min(起 + 块大小, 行数)) for 起 in range(0, 行数, 块大小)]
        else:
            排序 = np.argsort(归属, kind="stable")
            组边界 = list(np.split(排序, np.flatnonzero(np.diff(归属[排序])) + 1))
        块表 = [g for g in 组边界 if g.size]
        _计时记录(self, "构建①IVF聚类", _bt); _bt = _time.perf_counter() if _PROF else 0.0
        if not 块表: return
        单元行 = []
        for bi, g in enumerate(块表):
            子, 子行数, 置换 = _切子单元(数组[g], 子单元数)
            块表[bi] = g[置换]
            起 = 0
            for L in 子行数:
                L = int(L); 单元行.append((bi, 起, L, 块表[bi][起:起 + L])); 起 += L
        # EM: 子单元中心 <-> 共享码本
        中心表 = [数组[u[3]].mean(0) for u in 单元行]
        _计时记录(self, "构建②切子单元", _bt); _bt = _time.perf_counter() if _PROF else 0.0
        码本 = None; 边 = None
        for _ in range(EM轮数 + 1):
            样 = []
            for u, c in zip(单元行[::5], 中心表[::5]):
                r = 数组[u[3]] - c[None, :]
                nr = np.maximum(np.linalg.norm(r, axis=1), 1e-8)
                样.append((r / nr[:, None]).ravel()[::53])
            码本 = _拟合码本(np.concatenate(样), 层数)
            边 = ((码本[:-1] + 码本[1:]) / 2).astype(np.float32)
            新中心 = []
            for u, c in zip(单元行, 中心表):
                r = 数组[u[3]] - c[None, :]
                nr = np.maximum(np.linalg.norm(r, axis=1), 1e-8)
                码q = np.searchsorted(边, r / nr[:, None], side="right").clip(0, 层数 - 1)
                新中心.append(c + (数组[u[3]] - (c[None, :] + 码本[码q] * nr[:, None])).mean(0))
            中心表 = 新中心
            if 稀疏48 == 4:                                # ★ 模式4: 丢维中心恒 0(否则会被重建误差推着漂 ✓)
                for _c in 中心表:
                    _c[~_稀疏保] = 0.0
            if 联合归属 and _ < EM轮数:                    # 量化感知重归属(最后一轮后不必)
                单元行, 中心表 = _重归属(数组, 块表, 单元行, 中心表, 码本, 边, 层数)
        self.码本 = 码本; self.边 = 边
        # 编码
        映射 = np.empty(行数, np.int64); 已写 = 0
        _计时记录(self, "构建③EM迭代", _bt); _bt = _time.perf_counter() if _PROF else 0.0
        重建基 = np.empty((行数, 维度), np.float32)
        残列表 = []; 全列表 = [];
        self.向量库 = [{"shape": (g.size, 维度), "打包": [], "残差范数": [], "全部范数": [],
                        "中心": [], "中心幅": [], "代表": []} for g in 块表]
        _TCQ源 = []
        _稀疏样本 = []          # ★ 稀疏48∈{1,2,3}: 用【未丢维】方向拟合 QTIP 码本 -> 码本保持基线(不重拟合 ✓)
        for ui, (bi, 起, L, 行) in enumerate(单元行):
            c = 中心表[ui]
            r = 数组[行] - c[None, :]
            if _稀疏保 is not None:
                if 稀疏48 in (1, 2, 3):                    # 先按基线方向采样, 再置 0 ✓
                    _nr0 = np.maximum(np.linalg.norm(r, axis=1), 1e-8)
                    _稀疏样本.append((r / _nr0[:, None]).ravel()[::37])
                c = c.copy(); c[~_稀疏保] = 0.0            # ★ 中心丢维 -> 0
                r[:, ~_稀疏保] = 0.0                        # ★ 残差丢维 -> 0 -> nr/方向只按保留维算 ✓
            nr = np.maximum(np.linalg.norm(r, axis=1), 1e-8)
            _方向块 = r / nr[:, None]
            码q = np.searchsorted(边, _方向块, side="right").clip(0, 层数 - 1).astype(np.uint8)
            if 码型I == 1:
                _TCQ源.append((bi, 起, L, np.ascontiguousarray(_方向块, np.float32)))
            重建 = c[None, :] + 码本[码q] * nr[:, None]
            重建基[行] = 重建
            全 = np.maximum(np.linalg.norm(重建, axis=1), 1e-8)
            幅 = max(float(np.abs(c).max()), 1e-8)
            p = self.向量库[bi]
            p["打包"].append(_打包(码q))
            p["残差范数"].append(nr.astype(np.float32)); p["全部范数"].append(全.astype(np.float32))
            p["中心"].append(np.rint(c / 幅 * 127.0).astype(np.int8)); p["中心幅"].append(幅)
            self.单元行数.append(L); self.单元块.append(bi)
        for bi, g in enumerate(块表):
            映射[已写:已写 + g.size] = g; 已写 += g.size
            self.块行.append(np.asarray(g, np.int32))          # 记录块内行号(增量要重建映射)
            if 归属 is not None:
                cid = int(归属[g[0]]); self.块归属.append(cid); self.cell块表[cid].append(bi)
        _计时记录(self, "构建④重建基", _bt); _bt = _time.perf_counter() if _PROF else 0.0
        self._重建映射()
        self._刷新()
        _计时记录(self, "构建⑤映射刷新", _bt); _bt = _time.perf_counter() if _PROF else 0.0
        self._训PQ(数组, 重建基)
        _计时记录(self, "构建⑥训PQ", _bt); _bt = _time.perf_counter() if _PROF else 0.0
        if 图类型 != "IVF" or 码型I == 1:
            self._建节点单元()                                 # QTIP 的 全流 重算也要它(IVF 档同样需要 ✓)
        if 码型I == 1:                                       # 用 12 电平 TCQ 重编码码流(仍 226 B/行)
            _bt = _time.perf_counter() if _PROF else 0.0
            _样 = (_稀疏样本 if (稀疏48 in (1, 2, 3) and _稀疏样本)
                   else [d.ravel()[::37] for (_, _, _, d) in _TCQ源])   # ★ 1/2/3=基线码本 · 0/4=按本档方向(4 重新拟合 ✓)
            _码本Q = np.asarray(_拟合码本(np.concatenate(_样)[:200000].astype(np.float32), qtip电平), np.float32)
            _长 = np.array([int(_p["shape"][0]) for _p in self.向量库], np.int64)
            _块起 = np.asarray(self.块起始, np.int64) if (self.块起始 is not None and len(self.块起始) == _长.size) \
                    else np.concatenate([[0], np.cumsum(_长)[:-1]])
            _维 = int(self.维度)
            # ★ 行长随电平数变(72/168/226/264 B) -> self.每行字节 必须跟着改, 否则 LM 的 226 会串味 ✗
            _每行, _分支B, _索B = (int(x) for x in qtip.每行字节(_维, qtip周期, qtip分支, qtip每子))
            self.每行字节 = _每行
            _垫 = 8 if _每行 < 112 else 0        # 4 电平(无索流)时分支窗口会越过行尾 -> 尾部垫 8B ✓
            _新流 = np.zeros(int(_长.sum()) * _每行 + _垫, np.uint8)
            for (bi, 起, L, dd) in _TCQ源:
                位流, 索流 = qtip.编码全(dd, _码本Q, QTIP_S2, QTIP_SJ, qtip周期, qtip分支, qtip每子)
                基 = int(_块起[bi]) + 起
                for i in range(L):
                    o = (基 + i) * _每行
                    _新流[o:o + _分支B] = 位流[i]
                    if _索B > 0:
                        _新流[o + _分支B:o + _分支B + _索B] = 索流[i]
                self.向量库[bi]["码字节"] = int(L) * _每行     # 载荷按新行长记账(供 _刷新 复用 ✓)
                self.向量库[bi]["打包"] = None                 # LM 载荷已无用 -> 立刻释放
            self.码流 = _新流; self.码本 = _码本Q; self.边 = (_码本Q[:-1] + _码本Q[1:]) / 2
            self.层数 = qtip电平
            self.QTIP流 = True
            # ★ 码流按新行长重排 -> 块P偏移(= 块起始 × 每行字节, 与 _补推表 同式 ✓) 与 单元码基 必须重算
            self.块P偏移 = np.ascontiguousarray(_块起 * np.int64(_每行))
            self.单元码基 = None
            # 重算 12bit 重建范数(全流): TCQ 码值与 LM 不同 -> ‖x̂‖ 不同, 分数要按新的归一化
            _计时记录(self, "构建⑧TCQ重编码", _bt)
            if self.节点单元 is not None:
                _U = np.asarray(self.节点单元, np.int64)
                _中 = np.asarray(self._单元中心浮点(), np.float32)
                _残v = np.asarray(self.残差范数, np.float32) * np.float32(self.残cap)
                _总行 = int(_长.sum())
                _全v = np.empty(_总行, np.float32); _bf = np.empty(_维, np.int32)
                for (bi, 起, L, dd) in _TCQ源:
                    基 = int(_块起[bi]) + 起
                    for i in range(L):
                        节点 = 基 + i
                        _qtip行(self.码流, 节点 * _每行, _维, _bf, QTIP_反位, QTIP_子表, QTIP_三表)
                        _全v[节点] = np.linalg.norm(_中[int(_U[节点])] + _码本Q[_bf] * _残v[节点])
                _新cap = np.float32(max(float(_全v.max()), 1e-8))
                _量 = np.rint(np.minimum(_全v / _新cap, np.float32(1.0)) * np.float32(4095.0)).astype(np.int64)
                _全流 = np.zeros((_总行 * 12 + 7) // 8 + 2, np.uint8)
                for k in range(_总行):
                    x = int(_量[k]); b = k * 12; i0 = b >> 3; sh = b & 7
                    x <<= sh
                    _全流[i0] |= np.uint8(x & 0xFF)
                    _全流[i0 + 1] |= np.uint8((x >> 8) & 0xFF)
                    _全流[i0 + 2] |= np.uint8((x >> 16) & 0xFF)
                self.全流 = _全流; self.全cap = _新cap
        if 图类型 != "IVF":
            _bt = _time.perf_counter() if _PROF else 0.0
            _流序 = np.empty((行数, int(self.维度)), np.float32)   # 与扫描同一寻址顺序: 按块/单元解码铺平
            _起 = 0
            for _b in range(len(self.向量库)):
                _出, _ = self._解码包(_b); _nb = int(_出.shape[0])
                _流序[_起:_起 + _nb] = _出; _起 += _nb
            _流序 /= np.maximum(np.linalg.norm(_流序, axis=1, keepdims=True), 1e-8)
            _计时记录(self, "构建⑦a节点单元", _bt); _bt = _time.perf_counter() if _PROF else 0.0
            self._建图(_流序)                                    # ★ QTIP 时这里解码的是 QTIP 码流, 图与检索同源 ✓
            _计时记录(self, "构建⑦b建图(NSG)", _bt); _bt = _time.perf_counter() if _PROF else 0.0
            del _流序, _出
            self.块行 = []                              # 图模式不再需要块行列表(省 4B/行运行时)
        self._释放分包载荷()

    def _刷新(self, 增量=False):
        块数 = len(self.向量库)
        长度 = np.array([p["shape"][0] for p in self.向量库], np.int64)
        self.块起始 = np.concatenate([[0], np.cumsum(长度)[:-1]]) if 块数 else np.empty(0, np.int64)
        包行 = np.array([int(p["码字节"]) if p.get("打包") is None else int(sum(x.size for x in p["打包"]))
                         for p in self.向量库], np.int64)   # 字节偏移(不是行数!)
        残行 = np.array([int(sum(x.size for x in p["残差范数"])) for p in self.向量库], np.int64)
        self.块P偏移 = np.concatenate([[0], np.cumsum(包行)[:-1]]) if 块数 else np.empty(0, np.int64)
        self.块范数偏移 = np.concatenate([[0], np.cumsum(残行)[:-1]]) if 块数 else np.empty(0, np.int64)
        self.块残差偏移 = self.块范数偏移.copy()
        if not 增量:
            self.码流 = np.ascontiguousarray(np.concatenate([np.concatenate(p["打包"]).ravel() for p in self.向量库]))
        残 = np.concatenate([np.concatenate(p["残差范数"]) for p in self.向量库])
        全 = np.concatenate([np.concatenate(p["全部范数"]) for p in self.向量库])
        self.残cap = float(max(残.max(), 1e-8)) / 255.0
        self.全cap = float(max(全.max(), 1e-8)) / 4095.0
        self.残差范数 = np.clip(np.rint(残 / self.残cap), 0, 255).astype(np.uint8)
        码全 = np.clip(np.rint(全 / self.全cap), 0, 4095).astype(np.uint16)      # 12bit 位流
        位 = ((码全[:, None] >> np.arange(12, dtype=np.uint16)[None, :]) & np.uint16(1)).astype(np.uint8)
        self.全行数 = int(码全.size)
        self.全流 = np.concatenate([np.packbits(位.reshape(-1), bitorder="little"),
                                    np.zeros(2, np.uint8)])
        if not 增量:
            self.单元中心 = np.ascontiguousarray(np.concatenate([np.stack(p["中心"]) for p in self.向量库]))
        if not 增量:
            self.单元中心幅 = np.array([a for p in self.向量库 for a in p["中心幅"]], np.float32)
        if not 增量:
            self._打包中心()
        if 用中心码 and not 增量:           # b2 码可自行解码 -> 不再保留 int8 中心(否则内存不降)
            self.单元中心 = None; self.单元中心幅 = None
        self.单元代表 = None; self.每单元代表数 = 1              # 代表=质心, 直接复用中心存储
        self.单元行数表 = np.array(self.单元行数, np.int64)
        self.单元块表 = np.array(self.单元块, np.int64)
        self.单元起表 = np.zeros(len(self.单元块), np.int64)
        for i in range(1, len(self.单元块)):
            self.单元起表[i] = self.单元起表[i - 1] + self.单元行数[i - 1] if self.单元块[i] == self.单元块[i - 1] else 0
        self.维度流 = int(math.ceil(self.维度 / 流组)) if 流组 else 0
        if 路由投影维 > 0:
            self._建投影()                     # ★ 使 ROUTEK 真正生效(默认 0 -> 不建, 零影响 ✓)

    def _建投影(self):
        """路由投影(SpecTemp 谱温): 中心 -> k 维, 谱按温度重标定, 常驻 int8
           温度 0=普通PCA, 1=白化(判别优先), <0=偏领头方向; 只用于【排序】"""
        C = self._单元中心浮点() if 路由投影维 > 0 else None
        if C is None or C.shape[0] <= 路由投影维 or C.shape[1] != int(self.维度):
            self.投影中心 = None; self.投影P = None; self.投影均值 = None
            self.投影尺度 = None; self.投影幅 = None
            return
        均 = C.mean(0).astype(np.float32)
        _, S, Vt = np.linalg.svd(C - 均, full_matrices=False)
        k = int(min(路由投影维, S.size))
        尺度 = np.ones(k, np.float32)                    # 温度固定 0 => 纯 PCA(自适应温度已删)
        K = np.ascontiguousarray(((C - 均) @ Vt[:k].T) * 尺度[None, :], np.float32)
        幅 = np.maximum(np.abs(K).max(1, keepdims=True), 1e-8)
        self.投影均值 = 均
        self.投影P = np.ascontiguousarray(Vt[:k], np.float32)
        self.投影尺度 = 尺度
        self.投影幅 = 幅.astype(np.float32)
        self.投影中心 = np.rint(K / 幅 * 127.0).astype(np.int8)

    def _取全(self, 行号):
        """重建范数 12bit 位流 -> float(乘全cap)"""
        if self.全流 is None or 行号.size == 0:
            return np.zeros(np.shape(行号), np.float32)
        r = np.clip(行号, 0, max(0, self.全行数 - 1)).astype(np.int64)
        bo = r * 12; by = bo >> 3; sh = (bo & 7).astype(np.uint32)
        低 = np.zeros(np.shape(行号), np.uint32)
        for c in range(3):
            低 |= self.全流[by + c].astype(np.uint32) << np.uint32(8 * c)
        return (((低 >> sh) & np.uint32(4095)).astype(np.float32)
                * np.float32(self.全cap))

    def _打包映射(self, 映射):
        """行号映射 -> 位流: 每值 ceil(log2 N) bit, 尾部补 4 字节供 5 字节读"""
        n = int(映射.size); self.映射行数 = n
        if n == 0:
            self.映射流 = None; self.映射位宽 = 0; return
        b = max(1, int(np.ceil(np.log2(max(2, n))))); self.映射位宽 = b
        位 = ((映射.astype(np.uint64)[:, None] >> np.arange(b, dtype=np.uint64)[None, :])
              & np.uint64(1)).astype(np.uint8)
        self.映射流 = np.ascontiguousarray(np.packbits(位.reshape(-1), bitorder="little"))
        self.映射流 = np.concatenate([self.映射流, np.zeros(4, np.uint8)])

    def _取映射(self, 行号):
        """只解需要的行(不带负号过滤, 由调用方管); 5 字节小端读 + 移位 + 掩码"""
        if 行号.size == 0 or self.映射流 is None:
            return np.asarray(行号, np.int64)
        b = self.映射位宽
        r = np.clip(行号, 0, max(0, self.映射行数 - 1)).astype(np.int64)
        bo = r * b; byte = bo >> 3; sh = (bo & 7).astype(np.uint32)
        低 = np.zeros(行号.shape, np.uint32)
        for c in range(5):
            低 |= self.映射流[byte + c].astype(np.uint32) << np.uint32(8 * c)
        取 = ((低 >> sh) & np.uint32((1 << b) - 1)).astype(np.int64)
        return np.where(行号 >= 0, 取, np.int64(-1))

    def _重建映射(self):
        """映射统一从 块行 重建(整体构建与增量共用)"""
        if self.块行:
            self._打包映射(np.concatenate(self.块行))
        else:
            self.映射流 = None; self.映射位宽 = 0; self.映射行数 = 0

    def _增量(self, 数组):
        """增量并入: 方向码本(self.码本/边)固定; 新行按 BLOCKSIZE 成块, 块内切子单元后编码"""
        # ★ QTIP/TCQ 的码流是"分支位 + 索流"布局, 与 LM 位流不兼容 -> 先拒绝, 避免静默写坏 ✗
        if 码型I == 1 or bool(getattr(self, "QTIP流", False)):
            raise RuntimeError(
                "index-csd: QTIP/TCQ 档不支持增量并入(码流布局不同); "
                "请重建索引, 或把 CODEC 改为 LM 后再启用增量")
        # ★ 单元行数/单元块 是构建期 list(与 单元行数表/单元块表 逐元素相同 ✓ 不入档 ✓)
        #   只有这里(append)要用 -> 用到时才建 ✓ (载入后它们默认是空 ✓)
        if not getattr(self, "单元行数", None):
            行表3 = getattr(self, "单元行数表", None); 块表3 = getattr(self, "单元块表", None)
            if 行表3 is not None and 块表3 is not None and len(行表3) == len(块表3):
                self.单元行数 = [int(x) for x in np.asarray(行表3).ravel()]
                self.单元块 = [int(x) for x in np.asarray(块表3).ravel()]
            else:
                self.单元行数 = []; self.单元块 = []
        行数, 维度 = 数组.shape
        if int(self.维度) > 0 and 维度 != int(self.维度):
            raise ValueError(f"index-csd: 增量维度 {维度} != 索引维度 {self.维度}")
        self.块行 = self._块行重建()                       # 块行不持久化, 这里还原
        基行 = int(sum(int(x.size) for x in self.块行))
        块偏移 = len(self.向量库)
        _bt = _time.perf_counter() if _PROF else 0.0          # ★ 构建分段计时
        if self.路由 and 行数 >= 最小向量数:
            归一 = np.ascontiguousarray(数组, np.float32).copy(); faiss.normalize_L2(归一)
            nlist新 = max(1, min(int(Nlist因子 * np.sqrt(max(1, 行数))), 行数 // 8))
            引 = faiss.IndexIVFFlat(faiss.IndexFlatIP(维度), 维度, nlist新, faiss.METRIC_INNER_PRODUCT)
            引.train(归一); 引.add(归一)
            _, 归 = 引.quantizer.search(归一, 1)
            归 = 归.ravel().astype(np.int64); 归[归 < 0] = 0
            del 引, 归一
            排序 = np.argsort(归, kind="stable")
            新块 = [g for g in np.split(排序, np.flatnonzero(np.diff(归[排序])) + 1) if g.size]
        else:
            总目标 = 基行 + 行数
            粒度 = max(16, min(块大小, int((总目标 ** 0.5) / max(1.0, Nlist因子))))
            新块 = [g for g in (np.arange(起, min(起 + 粒度, 行数)) for 起 in range(0, 行数, 粒度)) if g.size]
        新单元 = []
        for bi, g in enumerate(新块):
            子, 子行数, 置换 = _切子单元(数组[g], 子单元数)
            新块[bi] = g[置换]
            起 = 0
            for L in 子行数:
                L = int(L); 新单元.append((bi, 起, L, 新块[bi][起:起 + L])); 起 += L
        中心表 = [数组[u[3]].mean(0) for u in 新单元]
        for _ in range(EM轮数):                        # 码本固定, 只把中心推向"量化后无偏"
            for ui, u in enumerate(新单元):
                c = 中心表[ui]
                r = 数组[u[3]] - c[None, :]
                nr = np.maximum(np.linalg.norm(r, axis=1), 1e-8)
                码q = np.searchsorted(self.边, r / nr[:, None], side="right").clip(0, 层数 - 1)
                中心表[ui] = c + (数组[u[3]] - (c[None, :] + self.码本[码q] * nr[:, None])).mean(0)
        for bi, g in enumerate(新块):
            self.向量库.append({"shape": (int(g.size), 维度), "打包": [], "残差范数": [],
                                "全部范数": [], "中心": [], "中心幅": [], "代表": []})
            self.块行.append((基行 + g).astype(np.int32))
        重建新 = np.empty((行数, 维度), np.float32)
        for ui, (bi, 起, L, 行) in enumerate(新单元):
            c = 中心表[ui]
            r = 数组[行] - c[None, :]
            nr = np.maximum(np.linalg.norm(r, axis=1), 1e-8)
            码q = np.searchsorted(self.边, r / nr[:, None], side="right").clip(0, 层数 - 1).astype(np.uint8)
            重建 = c[None, :] + self.码本[码q] * nr[:, None]
            重建新[行] = 重建
            幅 = max(float(np.abs(c).max()), 1e-8)
            p = self.向量库[块偏移 + bi]
            p["打包"].append(_打包(码q))
            p["残差范数"].append(nr.astype(np.float32))
            p["全部范数"].append(np.maximum(np.linalg.norm(重建, axis=1), 1e-8).astype(np.float32))
            p["中心"].append(np.rint(c / 幅 * 127.0).astype(np.int8)); p["中心幅"].append(幅)
            self.单元行数.append(L); self.单元块.append(块偏移 + bi)
        # ---- 追加: 码流 / 中心(int8 或 b2 流) / 投影中心 ----（共享件首建固定, 只追加不重拟）
        新码 = np.concatenate([np.concatenate(self.向量库[块偏移 + bi]["打包"]).ravel() for bi in range(len(新块))])
        self.码流 = np.concatenate([self.码流, 新码.astype(self.码流.dtype)]) if self.码流 is not None and self.码流.size else 新码
        C新 = np.stack(中心表)
        if 中心用FP16:                                     # ★ 增量: fp16 直接追加 ✓
            self.中心fp16 = (np.concatenate([np.asarray(self.中心fp16, np.float16), C新.astype(np.float16)])
                             if getattr(self, "中心fp16", None) is not None else C新.astype(np.float16))
            self.中心行数 = int(getattr(self, "中心行数", 0) or 0) + int(C新.shape[0])
        elif 中心用形状 and self.中心码流 is not None:        # ★ 增量: GSQ-NL 形状码追加 ✓
            幅I = np.maximum(np.asarray(self.中心hi, np.float32) - np.asarray(self.中心lo, np.float32),
                            np.float32(1e-12))
            归I = np.clip((C新 - self.中心lo[None, :]) / 幅I[None, :], 0.0, 1.0)
            表I = np.asarray(self.中心形状, np.float32)
            码I = np.clip(np.searchsorted((表I[:-1] + 表I[1:]) / 2, 归I), 0, 中心电平 - 1).astype(np.uint8)
            位I = ((码I[:, :, None] >> np.arange(中心位, dtype=np.uint8)[None, None, :]) & np.uint8(1))
            新流I = np.packbits(位I.reshape(-1).astype(np.uint8), bitorder="little")
            self.中心码流 = np.concatenate([self.中心码流[:-2], 新流I, np.zeros(2, np.uint8)])
            self.中心行数 = int(self.中心行数) + int(C新.shape[0])
            self._中心形状全表 = None
        elif 中心用均匀 and self.中心码流 is not None:        # ★ 增量: 均匀码追加 ✓
            码c = np.clip(np.rint((C新 - self.中心lo[None, :]) / self.中心步[None, :]),
                          0, 中心电平 - 1).astype(np.uint8)
            位c = ((码c[:, :, None] >> np.arange(中心位, dtype=np.uint8)[None, None, :]) & np.uint8(1))
            新心流 = np.packbits(位c.reshape(-1).astype(np.uint8), bitorder="little")
            self.中心码流 = np.concatenate([self.中心码流[:-2], 新心流, np.zeros(2, np.uint8)])
            self.中心行数 = int(self.中心行数) + int(C新.shape[0])
            self._中心均匀全表 = None                          # 缓存失效 ✓
        self._重建映射()
        self._刷新(增量=True)
        if VQ字节数 > 0 and self.码本列表 is not None:
            e = 数组 - 重建新
            ne = np.maximum(np.linalg.norm(e, axis=1), 1e-8).astype(np.float32)
            ed = np.ascontiguousarray(e / ne[:, None], np.float32)
            码新 = None
            if getattr(self, "RVQ", False):
                码新, 恢 = self._编码RVQ(ed)
            elif self.PQ码本 is not None and 数组.shape[1] % VQ字节数 == 0:
                M = self.PQ码本.shape[0]; 子维 = self.PQ码本.shape[2]
                码新 = np.empty((行数, M), np.uint8)
                for m in range(M):
                    段 = ed[:, m * 子维:(m + 1) * 子维]
                    码新[:, m] = ((段[:, None, :] - self.PQ码本[m][None, :, :]) ** 2).sum(2).argmin(1)
                恢 = np.concatenate([self.PQ码本[m][码新[:, m]] for m in range(M)], axis=1)
            if 码新 is not None:
                self.PQ码 = np.concatenate([self.PQ码, 码新])
                self.误差范数 = np.concatenate([self.误差范数, ne.astype(np.float16)])
                self.新范数 = np.concatenate([self.新范数, np.maximum(
                    np.linalg.norm(重建新 + 恢 * ne[:, None], axis=1), 1e-8).astype(np.float32)])
        if 图类型 != "IVF": self._建图全量()           # 增量后必须按全量重建图
        if 图类型 != "IVF":
            self.块行 = []                                 # 图模式不再需要块行列表(省 4B/行)
        self._释放分包载荷()

    def _释放分包载荷(self):
        for p in self.向量库:
            if p.get("打包") is not None:
                p["码字节"] = int(sum(x.size for x in p["打包"]))      # 偏移重建要用(打包马上释放)
            for k in ("打包", "中心", "代表"):
                p[k] = None                                          # 残差范数/全部范数 保留(增量刷新要重建它们)

    def _图参数(self):
        """返回 (图层, 最大层, 每层槽数, 每层累计槽); NSG 给空数组使下降循环不执行"""
        if self.图层 is not None:
            return (self.图层, int(getattr(self, "图最大层", 0)),
                    np.ascontiguousarray(getattr(self, "图每层", np.zeros(1, np.int32)), np.int32),
                    np.ascontiguousarray(getattr(self, "图累计", np.zeros(1, np.int32)), np.int32))
        return (np.zeros(1, np.int8), 0, np.zeros(1, np.int32), np.zeros(1, np.int32))

    def _补推表(self):
        """★ 从原语(块行数 / 单元行数表 / 单元块表)重算可推导表 —— 存档不再保存它们 ✓ 省 0.43 B/行
           实测严格恒等: 块起始=cumsum(块行数)错位; 块范数偏移=块残差偏移=块起始;
                        块P偏移=块起始×每行字节; 单元起表=块内前缀和(单元行数表); cell大小=块行数 ✓"""
        # ★ 先把压缩存的数组解回来(载入后原名是 None ✓ 只有 <名>压/<名>原长/<名>法 ✓)
        for _名, _ in 压缩方案:
            if getattr(self, _名, None) is None and getattr(self, _名 + "压", None) is not None:
                try:
                    体 = getattr(self, _名 + "压"); 长 = int(getattr(self, _名 + "原长", 0))
                    法 = int(getattr(self, _名 + "法", 0))
                    形 = tuple(int(x) for x in getattr(self, _名 + "形", []))
                    dt = str(getattr(self, _名 + "型", "uint8"))
                    布 = 形 if 形 else (长,)
                    原 = 解压数组(体, 长, 法, dt, 布)
                    if 原.nbytes != 长 and 长:
                        原 = 原.reshape(-1)[:长]
                    setattr(self, _名, 原)
                    for _s in ("压", "原长", "法", "形", "型"):
                        try: setattr(self, _名 + _s, None)
                        except Exception: pass
                except Exception:
                    pass
        # ★ 幂等: 缺什么补什么(载入早期 单元行数表 可能还没恢复 ✓ 所以不能一锁了之 ✗)
        # ★ 注意 块起始 的默认值是 np.empty(0) 而不是 None ✗ -> 必须判"空或 None" ✓
        def _缺(v): return v is None or (isinstance(v, np.ndarray) and v.size == 0)
        if (getattr(self, "_表已补推", False) and not _缺(getattr(self, "块起始", None))
                and not _缺(getattr(self, "单元起表", None))): return
        try:
            块行 = getattr(self, "块行数", None)
            if 块行 is not None and len(块行) and _缺(getattr(self, "块起始", None)):
                起 = np.concatenate([[0], np.cumsum(np.asarray(块行, np.int64))[:-1]]).astype(np.int64)
                self.块起始 = 起
                self.块范数偏移 = 起            # ★ 同一块内存(只读 ✓ 不复制 ✓ 实测三者恒等 ✓)
                self.块残差偏移 = 起
                try: self.块P偏移 = 起 * np.int64(int(self.每行字节))
                except Exception: self.块P偏移 = 起
                # cell大小 已删(无读取处 ✓ = 块行数 ✓)
            行表 = getattr(self, "单元行数表", None); 块表 = getattr(self, "单元块表", None)
            if (行表 is not None and 块表 is not None and len(行表) == len(块表) and len(行表)
                    and _缺(getattr(self, "单元起表", None))):
                行 = np.asarray(行表, np.int64); 块 = np.asarray(块表, np.int64)
                累 = np.cumsum(行) - 行                                    # 每个单元"此前总行数" ✓
                换 = np.empty(行.size, bool); 换[0] = True; 换[1:] = 块[1:] != 块[:-1]
                段首 = np.flatnonzero(换)
                归属 = np.cumsum(换) - 1
                self.单元起表 = (累 - 累[段首][归属]).astype(np.int64)
            # ★ 重建 单元行数/单元块 这两个 list 已移到 _增量(用到才建 ✓ 载入后不必预建 ✓)
            # ★ 关键: 载入后 节点单元/节点单元压 都是 None ✗ 必须在此铺开重建 ✓
            #   否则 search 的前置条件(节点单元 is not None or 节点单元压 is not None)不成立 ✗
            #   -> 图检索整条分支被静默跳过 ✗ 召回错乱 (已实测 ✓)
            if (getattr(self, "节点单元", None) is None
                    and getattr(self, "节点单元压", None) is None):
                try: self._建节点单元()
                except Exception: pass
            self._表已补推 = True
        except Exception:
            pass

    def _归表(self):
        """★ 读档归一: GPU_Accelerator=True 时 向量GGUF读取 直接返回 VRAM(cupy) 数组, 而
           从字典加载 是逐属性 setattr(不经过本模组任何代码) -> 载入后每个表都可能是 cupy,
           一进 @njit 内核就 TypingError。这里在第一次用到时把它们一次性换回 host numpy。
           幂等且只跑一次; False 模式下 _GPU模式=False 直接返回 -> 零开销 ✓"""
        if (not _GPU模式) or self.__dict__.get("_表已归一", False): return self
        for _名, _值 in list(self.__dict__.items()):
            if _值 is None or isinstance(_值, _真np.ndarray): continue
            if type(_值).__module__.split(".")[0] == "cupy":
                setattr(self, _名, _值.get())            # VRAM -> host(逐属性, 不复制已有 numpy ✓)
            elif isinstance(_值, (list, tuple, dict)):
                新 = _主(_值)
                if 新 is not _值: setattr(self, _名, 新)
        self._表已归一 = True
        return self

    def _压缩表(self):
        """裸格式索引表压到够用的最小整型(int64 -> uint16/int32), 不动任何量化"""
        self._归表()                                     # ★ 读档后先把 cupy 表换回 numpy ✓
        self._补推表()                                   # ★ 载入后这些表是 None -> 先补推 ✓
        if getattr(self, "_表已压", False): return
        self._表已压 = True
        try:
            for 名 in ("单元行数表", "单元块表", "单元起表"):
                v = getattr(self, 名, None)
                if v is not None and v.size and v.max() < 65536:
                    setattr(self, 名, np.asarray(v, np.uint16))
            for 名 in ("块起始", "块P偏移", "块范数偏移", "块残差偏移"):
                v = getattr(self, 名, None)
                if v is not None and v.size and v.max() < 2 ** 31:
                    setattr(self, 名, np.asarray(v, np.int32))
        except Exception:
            pass

    def _建次代表懒(self):
        """惰性建次代表: 逐块解码(峰值内存=一个块), 取每单元的行做 2-means, 存第二质心(int8)"""
        if 次代表数 <= 0 or self.单元次代表 is not None or getattr(self, "_次代表已试", False):
            return
        self._次代表已试 = True
        if self.单元行数表 is None or len(self.单元行数表) == 0 or self.码流 is None:
            return
        单元数 = len(self.单元行数表); 维 = int(self.维度)
        块起 = np.asarray(self.块起始, np.int64); 块表 = np.asarray(self.单元块表, np.int64)
        起表 = np.asarray(self.单元起表, np.int64); 行表 = np.asarray(self.单元行数表, np.int64)
        rep = np.zeros((单元数, 维), np.int8); 幅 = np.ones(单元数, np.float32)
        for b in range(len(self.向量库)):
            本 = np.flatnonzero(块表 == b)
            if 本.size == 0: continue
            出, _ = self._解码包(b)
            规 = np.maximum(np.linalg.norm(出, axis=1, keepdims=True), 1e-8)
            出 = 出 / 规
            for ui in 本:
                起 = int(起表[ui]); L = int(行表[ui])
                if L < 4: continue
                c2 = _二中心(出[起:起 + L])
                v = c2[-1] if c2.shape[0] > 1 else c2[0]
                a = max(float(np.abs(v).max()), 1e-8); 幅[ui] = a
                rep[ui] = np.rint(v / a * 127.0).astype(np.int8)
            del 出
        self.单元次代表 = rep; self.单元次幅 = 幅

    def _建图寻址(self):
        """每单元两个小表: 码基(该单元首行的码流字节偏移) 与 全局起(该单元首行的节点号)"""
        if getattr(self, "单元码基", None) is not None and self.单元码基.size == len(self.单元块表):
            return
        单元数 = len(self.单元块表)
        self.单元码基 = np.zeros(单元数, np.int64)
        self.单元全局起 = np.zeros(单元数, np.int64)
        for ui in range(单元数):
            b = int(self.单元块表[ui]); 起 = int(self.单元起表[ui])
            self.单元码基[ui] = int(self.块P偏移[b]) + 起 * int(self.每行字节)
            self.单元全局起[ui] = int(self.块起始[b]) + 起

    def _建节点单元(self):
        """每行所属单元(uint16): 单元在块内连续 -> 由 单元块表/单元起表/单元行数 直接铺开"""
        if self.节点单元 is not None and self.节点单元.size == int(self.映射行数):
            return
        n = int(self.映射行数)
        if getattr(self, "节点单元压", None) is not None:      # 紧凑档
            self.节点单元 = _单元解包(np.ascontiguousarray(self.节点单元压), int(self.节点行数),
                                     int(getattr(self, "节点单元位", 11)))
            return
        self.节点单元 = np.zeros(n, np.uint16)
        for ui in range(len(self.单元块表)):
            b = int(self.单元块表[ui]); 起 = int(self.单元起表[ui]); L = int(self.单元行数表[ui])
            if L: self.节点单元[int(self.块起始[b]) + 起:int(self.块起始[b]) + 起 + L] = ui
        _mx = int(self.节点单元.max()) if self.节点单元.size else 0
        self.节点单元位 = max(9, int(math.ceil(math.log2(_mx + 1))))
        # ★ 不再打包 节点单元压: 它已不入档(可由本函数铺开重算 ✓) -> 打了也只是白占内存 ✗
        #   (旧的 节点单元压 存档仍能读: 上面 _单元解包 那条分支保留 ✓)
        self.节点单元压 = None
        self.节点行数 = int(n)

    def _图就绪(self):
        """加载后按需重建 图偏移(运行时用, 不持久化)"""
        if self.图邻接 is None or self.图偏移 is not None:
            return
        if getattr(self, "图度压", None) is not None:          # 紧凑档(省 0.375 B/行)
            度 = _度解包(np.ascontiguousarray(self.图度压), int(self.图行数),
                         int(getattr(self, "图度位", 5)) or 5).astype(np.int64)
        elif getattr(self, "图度", None) is not None:
            度 = self.图度.astype(np.int64)
        elif self.图层 is not None:                            # HNSW: 槽数 = 2M + 层*M
            度 = 2 * HNSW度 + self.图层.astype(np.int64) * HNSW度
        else:
            return
        self.图偏移 = np.zeros(度.size + 1, np.int64); np.cumsum(度, out=self.图偏移[1:])
        self.图邻接字节 = int(getattr(self, "图邻接字节", 3))

    def _块行重建(self):
        """块行 = 映射流按各块行数切开; 不持久化, 需要时还原"""
        if self.块行:
            return self.块行
        总 = int(self.映射行数)
        映 = self._取映射(np.arange(总))
        起 = 0; 出 = []
        for b in range(len(self.向量库)):
            nb = int(self.向量库[b]["shape"][0]); 出.append(映[起:起 + nb]); 起 += nb
        return 出

    def _建图全量(self):
        """从各块解码出全量库(旋转空间, 归一化)后建图 —— 增量路径用"""
        行数 = int(self.映射行数); 维 = int(self.维度)
        数组 = np.empty((行数, 维), np.float32); 起 = 0
        for b in range(len(self.向量库)):
            出, _ = self._解码包(b); nb = int(出.shape[0])
            数组[起:起 + nb] = 出; 起 += nb
        del 出
        规 = np.linalg.norm(数组, axis=1, keepdims=True)
        数组 /= np.maximum(规, 1e-8)
        self._建图(数组)
        del 数组

    def _建图(self, 数组):
        """建检索图: NSG = 自建 kNN 图 + 多样性剪枝(变长邻接); HNSW = faiss 建图后抽邻接表"""
        self.图类型 = 图类型
        import gc
        行数 = 数组.shape[0]
        if 行数 < 3:
            self.图邻接 = None; self.图偏移 = None; self.图层 = None
            return
        if 图类型 == "HNSW":
            规 = faiss.IndexHNSWFlat(数组.shape[1], HNSW度, faiss.METRIC_INNER_PRODUCT)
            规.hnsw.efConstruction = 200
            规.add(np.ascontiguousarray(数组, np.float32))
            h = 规.hnsw
            邻 = np.asarray(faiss.vector_to_array(h.neighbors), np.int32).copy()
            偏 = np.asarray(faiss.vector_to_array(h.offsets), np.int64).copy()
            层 = np.asarray(faiss.vector_to_array(h.levels), np.int32).copy()
            每层 = np.asarray([int(h.nb_neighbors(int(l))) for l in range(int(h.max_level) + 1)], np.int32)
            累计 = np.asarray([int(h.cum_nb_neighbors(int(l))) for l in range(int(h.max_level) + 1)], np.int32)
            self.图邻接 = 邻.view(np.uint8) if 邻.dtype == np.int32 else 邻
            self.图邻接字节 = 4; self.图偏移 = 偏; self.图层 = 层; self.图度 = None
            self.图入口 = int(h.entry_point)                   # 数组已是节点序 -> faiss 入口即节点号
            self.图每层 = 每层; self.图累计 = 累计
            self.图最大层 = int(h.max_level)
            del 规; gc.collect()
            return
        # NSG: 自己算 kNN 图 + 剪枝
        _knprobe = int(参数.get("GKNPROBE", 0))
        if 行数 <= 120000 and _knprobe > 0:                  # ★ 小库: IVF 近似 kNN(nprobe 可调 ✓ 比精确快几十倍 ✓)
            _归 = np.ascontiguousarray(数组, np.float32)
            _nl = max(1, min(int(Nlist因子 * np.sqrt(max(1, 行数))), 行数 // 8))
            引 = faiss.IndexIVFFlat(faiss.IndexFlatIP(数组.shape[1]), 数组.shape[1], _nl, faiss.METRIC_INNER_PRODUCT)
            引.train(_归); 引.add(_归)
            引.nprobe = max(1, min(_knprobe, _nl))
            _kb = min(kNN候选 + 1, 行数)
            相似, 候 = 引.search(_归, _kb)
        elif 行数 <= 120000:                                 # 精确 kNN(慢 ✗ 仅调试用)
            引 = faiss.IndexFlatIP(数组.shape[1])
            引.add(np.ascontiguousarray(数组, np.float32))
            相似, 候 = 引.search(np.ascontiguousarray(数组, np.float32), min(kNN候选 + 1, 行数))
        else:
            引 = faiss.IndexHNSWFlat(数组.shape[1], 16, faiss.METRIC_INNER_PRODUCT)
            引.hnsw.efConstruction = 200; 引.hnsw.efSearch = max(128, kNN候选 * 4)
            引.add(np.ascontiguousarray(数组, np.float32))
            相似, 候 = 引.search(np.ascontiguousarray(数组, np.float32), min(kNN候选 + 1, 行数))
        del 引; gc.collect()
        候 = np.ascontiguousarray(候[:, 1:], np.int32)          # 去掉自身
        相似 = np.ascontiguousarray(相似[:, 1:], np.float32)
        块, 度 = _剪枝图(候, 相似, np.ascontiguousarray(数组, np.float32), 图度上限, np.float32(剪枝alpha))
        if 反向边:
            块, 度 = _反向重剪(块, 度, np.ascontiguousarray(数组, np.float32), 图度上限, np.float32(剪枝alpha))
        有效 = 块 >= 0
        邻32 = np.ascontiguousarray(块[有效])
        # 2 字节【有符号增量】: 图建在流序数组上 -> 邻居 id 相近, 增量普遍落在 ±32767 内
        # 解码端 _取邻 早已支持(邻接字节==2); 越界则自动退回 3 字节绝对 id
        _源 = np.repeat(np.arange(行数, dtype=np.int64), 块.shape[1])[有效.ravel()]
        _增 = 邻32.astype(np.int64) - _源 + 32768
        if 图邻接绝对 and 行数 <= 65535:
            self.图邻接 = np.empty(邻32.size * 2, np.uint8)      # 2 字节绝对 id(无损, 覆盖到 65535 行)
            self.图邻接[0::2] = (邻32 & 0xFF).astype(np.uint8)
            self.图邻接[1::2] = ((邻32 >> 8) & 0xFF).astype(np.uint8)
            self.图邻接字节 = 2
        elif _增.size and int(_增.min()) >= 0 and int(_增.max()) <= 65535:
            self.图邻接 = np.empty(邻32.size * 2, np.uint8)      # 2 字节有符号增量(无损)
            self.图邻接[0::2] = (_增 & 0xFF).astype(np.uint8)
            self.图邻接[1::2] = ((_增 >> 8) & 0xFF).astype(np.uint8)
            self.图邻接字节 = 2
        elif 行数 < (1 << 24):
            self.图邻接 = np.empty(邻32.size * 3, np.uint8)      # 3 字节绝对 id(无损)
            self.图邻接[0::3] = (邻32 & 0xFF).astype(np.uint8)
            self.图邻接[1::3] = ((邻32 >> 8) & 0xFF).astype(np.uint8)
            self.图邻接[2::3] = ((邻32 >> 16) & 0xFF).astype(np.uint8)
            self.图邻接字节 = 3
        else:
            self.图邻接 = 邻32.view(np.uint8); self.图邻接字节 = 4
        self.图度 = 度.astype(np.uint8)
        _dm = int(self.图度.max()) if self.图度.size else 0
        if _dm <= 31:
            self.图度位 = 5
            self.图度压 = _度打包(self.图度, 5)
            assert bool((_度解包(self.图度压, self.图度.size, 5) == self.图度).all()), "图度打包自检失败"
        else:
            self.图度位 = 0                      # 0 = 不打包, 直接用 uint8 原始数组(保底)
            self.图度压 = None
        self.图行数 = int(行数)
        self.图偏移 = np.zeros(行数 + 1, np.int64); np.cumsum(度, out=self.图偏移[1:])
        self.图层 = None
        # 入口取"与整体最相似"的节点(近似质心)
        样 = 数组[np.random.choice(行数, min(2000, 行数), replace=False)]
        self.图入口 = int(np.argmax(数组 @ 样.sum(0)))          # 数组已是节点序 -> 直接是节点号

    def _编码RVQ(self, ed):
        """多级残差量化编码: 逐级取最近质心, 余量继续; 返回 (码 (n,S) uint8, 重建 (n,D))"""
        S = len(self.码本列表)
        码 = np.empty((ed.shape[0], S), np.uint8)
        恢 = np.zeros_like(ed)
        余 = np.ascontiguousarray(ed, np.float32)
        for s in range(S):
            本 = self.码本列表[s]
            引 = faiss.IndexFlatL2(本.shape[1]); 引.add(np.ascontiguousarray(本, np.float32))
            _, a = 引.search(余, 1)
            a = a.ravel().astype(np.int64); a[a < 0] = 0
            码[:, s] = a
            恢 += 本[a]
            余 = np.ascontiguousarray(余 - 本[a], np.float32)
            del 引
        return 码, np.ascontiguousarray(恢, np.float32)

    def _训PQ(self, 数组, 重建基):
        """残差(方向)编码器训练: PQ 或 RVQ"""
        if VQ字节数 <= 0:
            return
        e = 数组 - 重建基
        ne = np.maximum(np.linalg.norm(e, axis=1), 1e-8).astype(np.float32)
        ed = np.ascontiguousarray(e / ne[:, None], np.float32)
        取 = np.random.choice(ed.shape[0], min(20000, ed.shape[0]), replace=False)
        S = VQ字节数
        if 用RVQ:
            本列 = []
            余 = np.ascontiguousarray(ed[取], np.float32)
            for s in range(S):
                km = faiss.Kmeans(数组.shape[1], 256, niter=25, verbose=False)
                km.train(余)
                本 = np.ascontiguousarray(km.centroids, np.float32)
                本列.append(本)
                引 = faiss.IndexFlatL2(数组.shape[1]); 引.add(本)
                _, a = 引.search(余, 1)
                余 = np.ascontiguousarray(余 - 本[a.ravel()], np.float32)
                del km, 引
            self.码本列表 = 本列; self.PQ码本 = None; self.RVQ = True
            self.PQ码, 恢 = self._编码RVQ(ed)
        else:
            if 数组.shape[1] % S:
                return
            pq = faiss.IndexPQ(数组.shape[1], S, 8); pq.train(np.ascontiguousarray(ed[取]))
            self.PQ码 = np.ascontiguousarray(pq.sa_encode(ed))
            self.PQ码本 = faiss.vector_to_array(pq.pq.centroids).reshape(
                S, 256, 数组.shape[1] // S).astype(np.float32).copy()
            self.码本列表 = [np.ascontiguousarray(self.PQ码本[s]) for s in range(S)]
            self.RVQ = False
            恢 = pq.sa_decode(self.PQ码)
            del pq
        self.误差范数 = ne.astype(np.float16)
        self.新范数 = np.maximum(np.linalg.norm(重建基 + 恢 * ne[:, None], axis=1), 1e-8).astype(np.float32)
        del ed, e, 恢

    def _单元中心浮点(self):
        if 中心用FP16 and getattr(self, "中心fp16", None) is not None:
            return np.ascontiguousarray(self.中心fp16, np.float32)   # ★ 直接读 ✓
        if 用中心码 and self.中心码流 is not None:
            return self._解中心(np.arange(self.中心行数))
        if self.单元中心 is None or self.单元中心.size == 0: return None
        return self.单元中心.astype(np.float32) * (self.单元中心幅[:, None] / 127.0)

    def _打包中心(self):
        """中心编码: gsq(逐维均匀 + 每维 min/步长 ✓ 默认) 或 fp16 直存 ✓"""
        if not 用中心码 or self.单元中心 is None or self.单元中心.shape[0] == 0:
            self.中心码流 = None; self.中心行数 = 0
            return
        C = self.单元中心.astype(np.float32) * (self.单元中心幅[:, None] / 127.0)
        # ★ 调试钩子: 设 CBCT_DUMP_CENTER=<路径> 就把【量化前的真 fp32 中心】存盘 ✓ (默认无影响 ✓)
        try:
            import os as _os
            _钩 = _os.environ.get("CBCT_DUMP_CENTER")
            if _钩:
                np.save(_钩, np.ascontiguousarray(C, np.float32))
        except Exception:
            pass
        if 中心用FP16:                                    # ★ 直接存 float16 ✓ 跳过 Lloyd-Max ✓
            self.中心fp16 = np.ascontiguousarray(C.astype(np.float16))
            self.中心码流 = None
            self.中心行数 = int(C.shape[0])
            return
        if 中心用形状:                                     # ★ GSQ-NL: 均匀骨架 + 共享形状表 ✓
            self.中心lo = np.ascontiguousarray(C.min(0), np.float32)
            self.中心hi = np.ascontiguousarray(C.max(0), np.float32)
            幅 = np.maximum(self.中心hi - self.中心lo, np.float32(1e-12))
            归一 = np.clip((C - self.中心lo[None, :]) / 幅[None, :], 0.0, 1.0)
            表 = np.asarray(_拟合码本(归一.ravel()[::3].astype(np.float32), 中心电平), np.float32)
            表 = np.sort(表); 表 = 表 - 表[0]
            表 = np.clip(表 / max(float(表[-1]), 1e-9), 0.0, 1.0)   # ★ 归一到 [0,1] ✓ 最后一级=1 ✓
            表[0] = 0.0; 表[-1] = 1.0
            self.中心形状 = np.ascontiguousarray(表, np.float32)
            码 = np.clip(np.searchsorted((表[:-1] + 表[1:]) / 2, 归一), 0, 中心电平 - 1).astype(np.uint8)
            self.中心码流 = np.concatenate([np.packbits(
                ((码[:, :, None] >> np.arange(中心位, dtype=np.uint8)[None, None, :]) & np.uint8(1)
                 ).reshape(-1).astype(np.uint8), bitorder="little"), np.zeros(2, np.uint8)])
            self.中心行数 = int(C.shape[0])
            return
        if 中心用均匀:                                     # ★ 逐维均匀: 只存 min/步长(768x2 ✓) 不训码本 ✓
            self.中心lo = np.ascontiguousarray(C.min(0), np.float32)
            self.中心hi = np.ascontiguousarray(C.max(0), np.float32)
            self.中心步 = np.ascontiguousarray(
                (self.中心hi - self.中心lo) / np.float32(中心电平 - 1), np.float32)
            self.中心步 = np.maximum(self.中心步, np.float32(1e-12))
            码 = np.clip(np.rint((C - self.中心lo[None, :]) / self.中心步[None, :]),
                         0, 中心电平 - 1).astype(np.uint8)
            self.中心码流 = np.concatenate([np.packbits(
                ((码[:, :, None] >> np.arange(中心位, dtype=np.uint8)[None, None, :]) & np.uint8(1)
                 ).reshape(-1).astype(np.uint8), bitorder="little"), np.zeros(2, np.uint8)])   # ★ 补 2 字节(读 3 字节 ✓)
            self.中心行数 = int(C.shape[0])
            return

    def _解中心均匀(self, 单元索引):
        """均匀模式: 5bit 位流 -> lo + 码 * 步 (★ 全向量化 + 整表缓存 ✓ 构建时会被反复调用 ✓)"""
        idx = np.asarray(单元索引).ravel().astype(np.int64)
        if self.中心码流 is None or idx.size == 0:
            return np.zeros((idx.size, int(self.维度)), np.float32)
        D = int(self.维度); 位 = int(中心位); 掩 = np.uint32(中心电平 - 1)
        if 中心用形状 and getattr(self, "中心形状", None) is not None:      # ★ GSQ-NL ✓
            表 = np.asarray(self.中心形状, np.float32)
            幅 = np.maximum(np.asarray(self.中心hi, np.float32) - np.asarray(self.中心lo, np.float32),
                            np.float32(1e-12))
            全缓 = getattr(self, "_中心形状全表", None)
            if 全缓 is None or 全缓.shape[0] != int(self.中心行数):
                bo2 = (np.arange(int(self.中心行数), dtype=np.int64)[:, None] * D
                       + np.arange(D, dtype=np.int64)[None, :]) * 位
                by2 = bo2 >> 3; 内2 = (bo2 & 7).astype(np.uint32)
                流2 = self.中心码流
                c0 = 流2[by2].astype(np.uint32); c1 = 流2[by2 + 1].astype(np.uint32); c2 = 流2[by2 + 2].astype(np.uint32)
                v2 = ((c0 | (c1 << np.uint32(8)) | (c2 << np.uint32(16))) >> 内2) & 掩
                全缓 = self.中心lo[None, :] + 表[v2] * 幅[None, :]
                self._中心形状全表 = 全缓
            return np.ascontiguousarray(全缓[idx])
        全缓 = getattr(self, "_中心均匀全表", None)
        if 全缓 is None or 全缓.shape[0] != int(self.中心行数):
            bo = (np.arange(int(self.中心行数), dtype=np.int64)[:, None] * D
                  + np.arange(D, dtype=np.int64)[None, :]) * 位                  # (U, D) ✓ 只算一次 ✓
            by = bo >> 3; 内 = (bo & 7).astype(np.uint32)
            流 = self.中心码流
            b0 = 流[by].astype(np.uint32); b1 = 流[by + 1].astype(np.uint32); b2 = 流[by + 2].astype(np.uint32)
            v = ((b0 | (b1 << np.uint32(8)) | (b2 << np.uint32(16))) >> 内) & 掩
            全缓 = self.中心lo[None, :] + v.astype(np.float32) * self.中心步[None, :]
            self._中心均匀全表 = 全缓
        return np.ascontiguousarray(全缓[idx])

    def _解中心(self, 单元索引):
        """只解指定单元的中心: 5bit 位流 -> (n, D); 2 字节读(5+7<=16, 尾补1字节)"""
        idx = np.asarray(单元索引).ravel().astype(np.int64)
        if self.中心码流 is None or idx.size == 0:
            return np.zeros((idx.size, 0), np.float32)
        return self._解中心均匀(idx)                         # ★ 只剩均匀(gsq) ✓

    def _单元残均(self):
        """每单元的平均残差范数(与打分同量纲) —— 单元在节点序上连续, 用 reduceat 一次算好并缓存"""
        if getattr(self, "_残均缓存", None) is not None:
            return self._残均缓存
        if self.残差范数 is None or self.单元行数表 is None or len(self.单元行数表) == 0:
            self._残均缓存 = np.zeros(0, np.float32); return self._残均缓存
        起 = (np.asarray(self.块起始, np.int64)[np.asarray(self.单元块表, np.int64)]
              + np.asarray(self.单元起表, np.int64))
        序 = np.argsort(起, kind="stable")
        残 = self.残差范数[:int(self.映射行数)].astype(np.float32) * np.float32(self.残cap)
        段 = np.add.reduceat(残, 起[序])
        均 = np.zeros(len(起), np.float32)
        均[序] = 段 / np.maximum(np.asarray(self.单元行数表, np.float32)[序], 1.0)
        self._残均缓存 = 均
        return 均

    def _代表浮点归一(self):
        中 = self._单元中心浮点()
        if 中 is None: return None
        return 中 / np.maximum(np.linalg.norm(中, axis=1, keepdims=True), 1e-8)

    def _cell质心浮点(self):
        if self.cell中心 is None: return None
        return self.cell中心.astype(np.float32) / np.maximum(self.cell中心范数[:, None], 1e-8)

    # ---------------- 选单元 ----------------
    def 选单元(self, 查询归一, 预算=None):
        # ★ 读档后 单元行数/单元块 是空 list(不入档 ✓ 由表重算 ✓)
        #   → 本函数需要它们, 缺了就会静默 0 命中 ✗ (扫描档/GPU档)
        if not getattr(self, "单元行数", None):
            _行表 = getattr(self, "单元行数表", None); _块表 = getattr(self, "单元块表", None)
            if _行表 is not None and _块表 is not None and len(_行表) == len(_块表) and len(_行表):
                self.单元行数 = [int(x) for x in np.asarray(_行表).ravel()]
                self.单元块 = [int(x) for x in np.asarray(_块表).ravel()]
        单元数 = len(self.单元行数)
        if not 单元数:
            return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int32),
                    np.empty(0, np.int64), np.empty(0, np.float32), np.zeros(查询归一.shape[0] + 1, np.int64))
        self._建次代表懒()
        行数 = self.单元行数表
        总行 = int(行数.sum())
        if 预算 is None:
            预算 = 扫描预算配置 if 扫描预算配置 > 0 else int(
                Nprobe默认 * (总行 ** 0.5) / max(1.0, Nlist因子))     # NPROBE 个"目标簇"(√N/Nlist因子 行)
        if getattr(self, "暴力", False):
            预算 = 总行                                               # 暴力档: 全扫所有单元
        if self.投影中心 is not None:                       # 投影(谱温)路由, 只用于排序
            qk = ((查询归一 - self.投影均值) @ self.投影P.T) * self.投影尺度[None, :]
            qk /= np.maximum(np.linalg.norm(qk, axis=1, keepdims=True), 1e-8)
            ck = self.投影中心.astype(np.float32) * (self.投影幅 / 127.0)
            ck /= np.maximum(np.linalg.norm(ck, axis=1, keepdims=True), 1e-8)
            分 = qk @ ck.T
        代表 = self._代表浮点归一()
        if self.投影中心 is None and 代表 is not None:
            分 = 查询归一 @ 代表.T
        elif self.投影中心 is None and self.块归属 and self.cell中心 is not None:
            簇 = self._cell质心浮点()
            分 = (查询归一 @ 簇.T)[:, np.asarray(self.块归属, np.int64)][:, self.单元块表]
        elif self.投影中心 is None:
            分 = np.ones((查询归一.shape[0], 单元数), np.float32)
        中心 = None if 用中心码 else self._单元中心浮点()
        常分 = None if 用中心码 else ((查询归一 @ 中心.T) if 中心 is not None else 分)
        if self.单元次代表 is not None:                  # 次代表: 取两个代表余弦的较大者
            c2 = self.单元次代表.astype(np.float32) * (self.单元次幅[:, None] / 127.0)
            c2 /= np.maximum(np.linalg.norm(c2, axis=1, keepdims=True), 1e-8)
            分 = np.maximum(分, 查询归一 @ c2.T)
        if 产出权重 > 0.0:                              # 产出率修正: 残差大的单元天花板更高
            分 = 分 + np.float32(产出权重) * self._单元残均()[None, :]
        序 = np.argsort(-分, axis=1)
        累积 = np.cumsum(行数[序], axis=1)
        取数 = np.full(查询归一.shape[0], 单元数, np.int64) if 预算 >= 行数.sum() else \
               np.where((累积 >= 预算).any(1), (累积 >= 预算).argmax(1) + 1, 单元数)
        n总 = int(取数.sum())
        U块 = np.empty(n总, np.int32); U起 = np.zeros(n总, np.int32)
        U行 = np.empty(n总, np.int32); U基 = np.empty(n总, np.int64); U常 = np.empty(n总, np.float32)
        偏移 = np.zeros(查询归一.shape[0] + 1, np.int64); 已 = 0
        for i in range(查询归一.shape[0]):
            单元 = 序[i, :取数[i]]
            单元 = 单元[np.lexsort((self.单元起表[单元], self.单元块表[单元]))]      # 码流单调
            n = 单元.size
            U块[已:已 + n] = self.单元块表[单元].astype(np.int32)
            U起[已:已 + n] = self.单元起表[单元].astype(np.int32)
            U行[已:已 + n] = 行数[单元].astype(np.int32)
            U基[已:已 + n] = self.块起始[self.单元块表[单元]]
            U常[已:已 + n] = (查询归一[i] @ self._解中心(单元).T) if 常分 is None else 常分[i, 单元]
            已 += n; 偏移[i + 1] = 已
        return U块, U起, U行, U基, U常, 偏移

    # ---------------- VQ 残差重排(ADC) ----------------
    def _VQ重排(self, 查询归一, 出行, 出分):
        if self.PQ码 is None or 出行.size == 0:
            return 出分, 出行
        可用 = 出行 >= 0
        行号 = np.clip(出行, 0, max(0, self.映射行数 - 1))
        原行 = np.where(可用, self._取映射(出行), np.int64(-1))
        M = self.PQ码.shape[1]
        LUT = np.empty((查询归一.shape[0], M, 256), np.float32)
        if getattr(self, "RVQ", False):                    # 每级一张全维 LUT
            for s in range(M):
                LUT[:, s, :] = 查询归一 @ self.码本列表[s].T
        else:                                              # 每级是子空间, 查询按段切
            子维 = self.PQ码本.shape[2]
            for m in range(M):
                LUT[:, m, :] = 查询归一[:, m * 子维:(m + 1) * 子维] @ self.PQ码本[m].T
        安 = np.clip(原行, 0, self.新范数.size - 1)
        基础 = np.where(可用, 出分, 0.0) * self._取全(行号)
        新分 = ADC重排(原行, np.ascontiguousarray(基础, np.float32),
                      self.误差范数[安].astype(np.float32), self.新范数[安],
                      self.PQ码, np.ascontiguousarray(LUT, np.float32), int(M))
        新分 = np.where(可用, 新分, -np.inf)
        序 = np.argsort(-新分, axis=1)
        return np.take_along_axis(新分, 序, 1), np.take_along_axis(出行, 序, 1)

    # ---------------- 解码接口 ----------------
    def _解码包(self, b):
        p = self.向量库[b]; 行数, 维度 = p["shape"]
        出 = np.empty((行数, 维度), np.float32)
        基 = int(self.块起始[b]); p基 = int(self.块P偏移[b])
        中心全体 = self._单元中心浮点()                 # b2 模式也能解(int8 中心可能已释放)
        for ui in np.flatnonzero(self.单元块表 == b):
            起行 = int(self.单元起表[ui]); L = int(self.单元行数表[ui])
            c = 中心全体[ui]
            nr = self.残差范数[基 + 起行:基 + 起行 + L].astype(np.float32) * self.残cap
            段 = self.码流[p基 + 起行 * self.每行字节: p基 + (起行 + L) * self.每行字节]
            if 码型I == 1 and bool(getattr(self, "QTIP流", False)):
                # ★ QTIP 码流: 逐行走 _qtip行 解电平(不能走 LM 的位流解包表 ✗)
                _码q = np.empty(维度, np.int32)
                for _i in range(L):
                    _qtip行(self.码流, p基 + (起行 + _i) * int(self.每行字节), 维度, _码q,
                            QTIP_反位, QTIP_子表, QTIP_三表)
                    出[起行 + _i] = c + self.码本[_码q] * nr[_i]
            else:
                if self.打包位 > 0:
                    每字节 = 8 // self.打包位; 掩 = (1 << self.打包位) - 1
                    v = 段.reshape(L, self.每行字节).astype(np.uint16)
                    列 = [((v >> (self.打包位 * j)) & 掩).astype(np.uint8) for j in range(每字节)]
                    码 = np.stack(列, 2).reshape(L, -1)[:, :维度]
                else:
                    码 = 解位流(段.reshape(L, -1), self.维度流, self.流宽, self.流组, 流表, 维度)
                出[起行:起行 + L] = c[None, :] + self.码本[码] * nr[:, None]
        return 出, self._取全(np.arange(基, 基 + 行数))

    # ---------------- 检索 ----------------
    def search(self, query, k):
        self._归表()                                      # ★ 读档归一(cupy 表 -> numpy)
        self._物化向量库()
        self._压缩表()
        _t = _time.perf_counter() if _PROF else 0.0        # ①准备
        查询矩阵 = self._降维(self.正旋转(np.atleast_2d(_主(query)).astype(np.float32)))   # ★ cupy 查询 -> host
        if self.码流 is None or self.码流.size == 0:
            return np.empty((查询矩阵.shape[0], 0), np.float32), np.empty((查询矩阵.shape[0], 0), np.int64)
        查询归一 = np.ascontiguousarray(查询矩阵 / np.maximum(np.linalg.norm(查询矩阵, axis=1, keepdims=True), 1e-8))
        _计时记录(self, "①准备", _t)
        if (self.图邻接 is not None
                and (self.节点单元 is not None or getattr(self, "节点单元压", None) is not None)
                and not getattr(self, "暴力", False)):            # ---- 图检索 (CSDFast 暴力档不走路由) ----
            _t = _time.perf_counter() if _PROF else 0.0     # ②图准备
            self._建节点单元()                                   # 读回后由 节点单元压 解包
            self._图就绪()
            self._建图寻址()
            _计时记录(self, "②图准备", _t)
            图行数 = int(self.映射行数)
            图k = min(k, 图行数)
            if 图k <= 0:
                return (np.empty((查询归一.shape[0], 0), np.float32),
                        np.empty((查询归一.shape[0], 0), np.int64))
            出分 = np.full((查询归一.shape[0], 图k), np.float32(-1e30), np.float32)
            出行 = np.full((查询归一.shape[0], 图k), np.int64(-1), np.int64)
            图层, 图最大层v, 图每层v, 图累计v = self._图参数()
            # ★ numba 不支持 float16 ✗ -> 落盘存 fp16(省空间 ✓), 喂内核前展开成 float32 ✓
            _中心fp16 = (np.ascontiguousarray(self.中心fp16, np.float32)
                         if getattr(self, '中心fp16', None) is not None
                         else np.empty((0, 0), np.float32))
            _中心lo = (np.ascontiguousarray(self.中心lo, np.float32)
                       if getattr(self, '中心lo', None) is not None else np.empty(0, np.float32))
            _中心步 = (np.ascontiguousarray(self.中心步, np.float32)
                       if getattr(self, '中心步', None) is not None else np.empty(0, np.float32))
            _中心幅 = (np.ascontiguousarray(np.asarray(self.中心hi, np.float32)
                       - np.asarray(self.中心lo, np.float32), np.float32)
                       if getattr(self, '中心hi', None) is not None else np.empty(0, np.float32))
            _中心形状 = (np.ascontiguousarray(self.中心形状, np.float32)
                         if getattr(self, '中心形状', None) is not None else np.empty(0, np.float32))
            _t = _time.perf_counter() if _PROF else 0.0     # ③图检索内核
            图ef用 = max(int(图ef), int(图k))                # ★ k 可大于 GEF -> 结果表必须能装下 k
            图检索(查询归一, self.图邻接, np.int32(self.图邻接字节), self.图偏移,
                   np.int32(self.图入口), 图层, np.int32(图最大层v), 图每层v, 图累计v,
                   np.int32(图行数), self.节点单元, self.单元码基, self.单元全局起,
                   self.中心码流,
                   self.码流, self.残差范数, self.全流,
                   np.float32(self.残cap), np.float32(self.全cap),
                   np.ascontiguousarray(self.码本, np.float32),
                   int(self.维度), int(self.层数), int(self.每行字节), int(self.打包位),
                   int(self.流宽), int(self.流组), 流表, int(图k), int(图ef用), np.float32(图边距), 出分, 出行,
                     _中心fp16, _中心lo, _中心步, _中心幅, _中心形状)
            _计时记录(self, "③图检索内核", _t)
            _t = _time.perf_counter() if _PROF else 0.0     # ④收尾
            合法 = 出行 >= 0
            原下标 = self._取映射(出行) if (self.映射流 is not None and self.映射流.size) else np.where(合法, 出行, np.int64(-1))
            _计时记录(self, "④收尾", _t)
            return np.where(合法, 出分, -np.inf).astype(np.float32), 原下标
        总行 = int(self.单元行数表.sum())
        实际k = min(k, 总行)
        取数 = int(min(max(实际k, 重排数) if self.PQ码 is not None else 实际k, 总行))
        U块, U起, U行, U基, U常, 偏移 = self.选单元(查询归一)
        出分 = np.full((查询归一.shape[0], 取数), -np.inf, np.float32)
        出行 = np.full((查询归一.shape[0], 取数), -1, np.int64)
        if U块.size:
            _t = _time.perf_counter() if _PROF else 0.0         # 扫描路径
            球面检索(查询归一, U块, U起, U行, U基, U常, 偏移,
                     self.块P偏移, self.块范数偏移, self.块残差偏移, self.码流,
                     self.残差范数, self.全流, np.float32(self.残cap), np.float32(self.全cap),
                     np.ascontiguousarray(self.码本, np.float32),
                     int(self.维度), int(self.维度流), int(self.层数), int(self.每行字节),
                     int(self.打包位), int(self.流宽), int(self.流组), 流表,
                     int(取数), 出分, 出行)
            _计时记录(self, "③扫描内核", _t)
            _t = _time.perf_counter() if _PROF else 0.0
            出分, 出行 = self._VQ重排(查询归一, 出行, 出分)
            _计时记录(self, "③b VQ重排", _t)
            出分 = np.ascontiguousarray(出分[:, :实际k]); 出行 = np.ascontiguousarray(出行[:, :实际k])
        合法 = 出行 >= 0
        if self.映射流 is not None and self.映射流.size:
            原下标 = self._取映射(出行)
        else:
            原下标 = np.where(合法, 出行, np.int64(-1))
        return np.where(合法, 出分, -np.inf).astype(np.float32), 原下标

    def cpu(self): return self
    def gpu(self): return self

    def _状态(self):
        self._压缩表()
        return {"模式": self.模式, "暴力": getattr(self, "暴力", False), "层数": self.层数, "打包位": self.打包位,
                "流宽": self.流宽, "流组": self.流组, "维度": self.维度, "维度流": self.维度流,
                "每行字节": self.每行字节, "子单元数": self.子单元数,
                "码本": self.码本, "边": self.边,
                "块行数": np.array([int(_p["shape"][0]) for _p in (self.向量库 or [])], np.int32),
                **_压存(self, "映射流"), "映射位宽": self.映射位宽, "映射行数": self.映射行数,
                # ★ 不再入档(可重算 ✓): 块起始 / 块P偏移 / 块范数偏移 / 块残差偏移 / cell大小 / 单元起表
                "cell中心": self.cell中心, "cell中心范数": self.cell中心范数,
                "块归属": self.块归属,   # cell块表 不入档(与 块归属 逐元素相同 ✓ 仅构建期用 ✓)
                **_压存(self, "码流"), **_压存(self, "残差范数"), **_压存(self, "全流"), "全行数": self.全行数,
                "残cap": self.残cap, "全cap": self.全cap,
                "单元中心": self.单元中心, "单元中心幅": self.单元中心幅,
                # ★ 不再入档(与 单元行数表/单元块表 逐元素相同 ✓ 载入后重建 ✓ 省 0.57 B/行): 单元行数 / 单元块
                "PQ码本": self.PQ码本, "PQ码": self.PQ码, "误差范数": self.误差范数, "新范数": self.新范数,
                "码本列表": self.码本列表, "RVQ": getattr(self, "RVQ", False),
                # ★ 节点单元压 不再入档: 实测恒等 = repeat(arange(单元数), 单元行数表) ✓
                #   (游程 1785 段 · 严格递增无跳号 ✓) 载入时 _建节点单元 会由表铺开 ✓ 省 1.375 B/行 ✓
                "节点行数": np.int64(getattr(self, "节点行数", 0)),
                "单元次代表": self.单元次代表, "单元次幅": self.单元次幅,
                **_压存(self, "图邻接"), **_压存(self, "图度压"),
                "图行数": np.int64(getattr(self, "图行数", 0)),
                "节点单元位": np.int64(getattr(self, "节点单元位", 11)),
                "图度位": np.int64(getattr(self, "图度位", 5)),
                "图邻接字节": int(getattr(self, "图邻接字节", 3)), "图入口": self.图入口,
                "图层": self.图层, "图类型": self.图类型,
                "图每层": getattr(self, "图每层", None), "图累计": getattr(self, "图累计", None),
                "图最大层": getattr(self, "图最大层", -1),
                "PCA_均值": self.PCA_均值, "PCA_主成分": self.PCA_主成分,
                "投影中心": self.投影中心, "投影P": self.投影P, "投影均值": self.投影均值,
                "投影尺度": self.投影尺度, "投影幅": self.投影幅,
                "中心码流": self.中心码流, "中心行数": self.中心行数,
                "中心fp16": getattr(self, "中心fp16", None),
                "中心lo": getattr(self, "中心lo", None), "中心步": getattr(self, "中心步", None),
                "中心hi": getattr(self, "中心hi", None),
                "中心形状": getattr(self, "中心形状", None),
                "单元行数表": self.单元行数表, "单元块表": self.单元块表,
                "QTIP流": bool(getattr(self, "QTIP流", False)),
                "应用": None, "路由": self.路由,
                "_正旋转": None, "_逆旋转": None}

    def _物化向量库(self):
        """按块行数重建 向量库 骨架(检索只需要 shape; 载荷是构建期脚手架, 不入档)"""
        行数 = getattr(self, "块行数", None)
        if 行数 is None or not len(行数): return
        维 = int(self.维度)
        if not self.向量库 or len(self.向量库) != len(行数):
            self.向量库 = [{"shape": (int(n), 维)} for n in 行数]

    def save(self, filename: str):
        """存档为 GGUF: 整个 _状态() 走项目自带的通用字典写入(tensor 按名对齐 -> 可 mmap)"""
        import importlib
        P = importlib.import_module("TranslatorPersistence")
        状态 = self._状态()
        临时 = f"{filename}.tmp"
        P.向量GGUF写入(临时, 向量字典=dict(状态))
        os.replace(临时, filename)

    def serialize(self):
        return pickle.dumps(self._状态(), protocol=pickle.HIGHEST_PROTOCOL)


def 解位流(段, 维度流, 流宽, 流组, 流表, 维度):
    """(n, 每行字节) 的位流 -> (n, 维度流*流组) 码(超出 维度 的用 0 补)"""
    n = 段.shape[0]
    位 = np.unpackbits(np.ascontiguousarray(段), axis=1, bitorder="little")
    出 = np.zeros((n, 维度流 * 流组), np.uint8)
    for g2 in range(维度流):
        bo = g2 * 流宽
        w = 位[:, bo:bo + 流宽].astype(np.uint32)
        权 = (1 << np.arange(流宽)).astype(np.uint32)
        v = (w * 权[None, :]).sum(1)
        出[:, g2 * 流组:(g2 + 1) * 流组] = 流表[v]
    return np.ascontiguousarray(出[:, :维度])


class IndexCSDFast(IndexCSDMoE):
    """暴力档: 编码与构建和 CSDMoE 完全一致, 仅检索时不做路由、全扫所有单元(召回=码本上限)"""

    def __init__(self, app=None, 路由: bool = True):
        super().__init__(app=app, 路由=路由)
        self.模式 = "IndexCSDFast"
        self.暴力 = True


Mods().注册索引类("IndexCSDMoE")(IndexCSDMoE)
Mods().注册索引类("IndexCSDFast")(IndexCSDFast)


@Mods().注册索引类型("CSDMoE", 数据源="Vector", 支持增量=True)
def 构建CSDMoE(Self, 向量, 索引=None):
    return IndexCSDMoE(app=Self), True


@Mods().注册索引类型("CSDFast", 数据源="Vector", 支持增量=True)
def 构建CSDFast(Self, 向量, 索引=None):
    return IndexCSDFast(app=Self), True


@Mods().注册索引部件("CSDMoEBase")
def 部件():
    return SimpleNamespace(IndexCSDMoE=IndexCSDMoE, 拟合码本=_拟合码本, 打包=_打包, 解位流=解位流,
                           FPS点=_FPS点, 切子单元=_切子单元, 流表=流表, 每行字节=_每行字节)
