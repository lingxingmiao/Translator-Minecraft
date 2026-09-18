import sys
import time
import queue
import threading
from TranslatorLib import TranslatorPersistence, Quantization, Translator, GPU_ACC, CleanVRAM, eb, np, faiss, os, Config, asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 0. 双线程开关
# ==========================================
启用双线程 = True   # False = 退回原来的单线程串行
应用配置   = True   # True = 把每个配置项真正写入 核心配置.Config（原版只打印未生效）
队列深度   = 1      # A 线程最多提前生产 1 个结果，等 B 取走才继续（控制内存峰值）
快速召回   = True   # 向量化召回统计，纯 Python 逐行 intersect1d 会全程持 GIL 导致线程无法并行

# ==========================================
# 1. 初始化与数据加载
# ==========================================
核心配置 = Config(Config={"VEC_FILE_NAME": "FP16_MAX_768", "VEC_QUANTIZATION": "Float16_Max"})
核心模块 = Translator(核心配置)
#核心模块 = Translator()

向量 = asyncio.run(TranslatorPersistence.参考词预处理(核心模块))[0].get()
if 向量.ndim == 1:
    向量 = 向量.reshape(1, -1)
当前向量 = 向量.astype(np.float32)

'''npy_file_path = "reordered_vectors_ivfpq_rerank_k32.npy" 
当前向量 = np.load(npy_file_path)
if 当前向量.ndim == 1:
    当前向量 = 当前向量.reshape(1, -1)
当前向量 = 当前向量.astype(np.float32)'''

N, D = 当前向量.shape
vec_min = float(np.min(当前向量))
vec_max = float(np.max(当前向量))

# ==========================================
# 2. 自动获取量化方案
# ==========================================

所有量化方案 = [
    # 32 位
    "Float32",
    # 16 位
    "Float16",
    "BFloat16",
    "Float16_Max",
    "Float16_E0M15",
    # 12 位
    "Float12_Max",
    "Float12_E0M11",
    # 8 位
    "Q8_K_M",
    "Float8_Max",
    "Float8_E0M7",
    "Float8_E4M3",
    # 6 位
    "Q6_K_M",
    "GSQ6_NL",
    "PolarQ5",
    # 5 位
    "Q5_K_M",
    "GSQ5_NL",
    "PolarQ4",
    # 4 位
    "Q4_K_M",
    "GSQ4_NL",
    "PolarQ3",
    # 3 位
    "Q3_K_M",
    "GSQ3_NL",
    "PolarQ2",
    # 2 位
    "Q2_K_M",
    "GSQ2_NL",
    "PolarQ1",
    "Q2_NF",
    # 1.585 位（三值）
    "TQ1_K_M",
    "GSTQ1_NL",
    "PolarTQ1",
    # 1 位（二值）
    "Q1_K_M",
    "GSQ1_NL",
    "PolarQ1",
    
    "PQ",
    "OPQ",
]
print(f"🎯 自动检测到 {len(所有量化方案)} 种量化方案: {所有量化方案}\n")

# ==========================================
# 3. 测试配置
# ==========================================
配置列表 = [
    {"VEC_QUANTIZATION_PQ_M": 8,   "VEC_QUANTIZATION_PQ_NBITS": 8, "VEC_QUANTIZATION_BLOCK_SIZE": 192, "VEC_QUANTIZATION_CLIP": 1.000},
    {"VEC_QUANTIZATION_PQ_M": 16,  "VEC_QUANTIZATION_PQ_NBITS": 8, "VEC_QUANTIZATION_BLOCK_SIZE": 128, "VEC_QUANTIZATION_CLIP": 1.000},
    {"VEC_QUANTIZATION_PQ_M": 24,  "VEC_QUANTIZATION_PQ_NBITS": 8, "VEC_QUANTIZATION_BLOCK_SIZE": 64 , "VEC_QUANTIZATION_CLIP": 1.000},
    {"VEC_QUANTIZATION_PQ_M": 32,  "VEC_QUANTIZATION_PQ_NBITS": 8, "VEC_QUANTIZATION_BLOCK_SIZE": 32 , "VEC_QUANTIZATION_CLIP": 1.000},
    {"VEC_QUANTIZATION_PQ_M": 48,  "VEC_QUANTIZATION_PQ_NBITS": 8, "VEC_QUANTIZATION_BLOCK_SIZE": 192, "VEC_QUANTIZATION_CLIP": 0.998},
    {"VEC_QUANTIZATION_PQ_M": 64,  "VEC_QUANTIZATION_PQ_NBITS": 8, "VEC_QUANTIZATION_BLOCK_SIZE": 128, "VEC_QUANTIZATION_CLIP": 0.998},
    {"VEC_QUANTIZATION_PQ_M": 96,  "VEC_QUANTIZATION_PQ_NBITS": 8, "VEC_QUANTIZATION_BLOCK_SIZE": 64 , "VEC_QUANTIZATION_CLIP": 0.998},
    {"VEC_QUANTIZATION_PQ_M": 128, "VEC_QUANTIZATION_PQ_NBITS": 8, "VEC_QUANTIZATION_BLOCK_SIZE": 32 , "VEC_QUANTIZATION_CLIP": 0.998},
]

# ==========================================
# 4. 预计算 Ground Truth（只算一次）
# ==========================================
NQ = min(20000, N)
np.random.seed(42)
查询索引 = np.random.choice(N, NQ, replace=False)
xq = 当前向量[查询索引].astype("float32").copy()
faiss.normalize_L2(xq)

原始归一化 = 当前向量.astype("float32").copy()
faiss.normalize_L2(原始归一化)

GT_INDEX = faiss.IndexFlatIP(D)
GT_INDEX.add(原始归一化)
_, I_gt = GT_INDEX.search(xq, 1000)

I_gt_2     = I_gt[:, :2]
I_gt_3     = I_gt[:, :3]
I_gt_5     = I_gt[:, :5]
I_gt_10    = I_gt[:, :10]
I_gt_50    = I_gt[:, :50]
I_gt_100   = I_gt[:, :100]
I_gt_500   = I_gt[:, :500]
I_gt_1000  = I_gt[:, :1000]

del GT_INDEX, 原始归一化
CleanVRAM()

# ==========================================
# 5. 召回率计算（B 线程职责）
# ==========================================
召回档位 = [2, 3, 5, 10, 50, 100, 500, 1000]
GT集合 = {2: I_gt_2, 3: I_gt_3, 5: I_gt_5, 10: I_gt_10,
          50: I_gt_50, 100: I_gt_100, 500: I_gt_500, 1000: I_gt_1000}
_行号键 = np.arange(NQ, dtype=np.int64)[:, None] * (N + 1)   # 行间隔离，保留槽 N 给无效项

def 统计命中(gt, pred):
    """gt/pred 为 (NQ, k) 的索引矩阵，返回逐行交集大小之和。

    原实现逐行 np.intersect1d 是纯 Python 循环（NQ×8 次），会全程持有 GIL，
    导致量化线程无法与之并行；这里改为键化 + 单次 isin，整段在 numpy C 层执行并释放 GIL。
    行内索引唯一，故计数与原逐行 intersect1d 完全等价。
    """
    gt键 = _行号键 + np.asarray(gt, dtype=np.int64)
    pred键 = _行号键 + np.where(np.asarray(pred, dtype=np.int64) < 0, N, pred)
    return int(np.isin(pred键, gt键).sum())

def 计算召回率(恢复数据库):
    # A 线程已优先 .get()，这里保留兼容（幂等，cupy 数组才会触发）
    if hasattr(恢复数据库, "get"):
        恢复数据库 = 恢复数据库.get()

    xb_pred = 恢复数据库.astype("float32").copy()
    faiss.normalize_L2(xb_pred)

    index_pred = faiss.IndexFlatIP(D)
    index_pred.add(xb_pred)
    _, I_pred = index_pred.search(xq, 1000)
    del index_pred, xb_pred

    召回率 = []
    for k in 召回档位:
        if 快速召回:
            命中 = 统计命中(GT集合[k], I_pred[:, :k])
        else:
            gt_s   = np.sort(GT集合[k], axis=1)
            pred_s = np.sort(I_pred[:, :k], axis=1)
            命中 = 0
            for i in range(len(gt_s)):
                命中 += len(np.intersect1d(gt_s[i], pred_s[i]))
        召回率.append(命中 / (NQ * k))
    del I_pred
    return tuple(召回率)

# ==========================================
# 6. 网格扫描（双线程流水线：A=量化/反量化  B=召回率）
# ==========================================
结果集 = {dtype: [None] * len(配置列表) for dtype in 所有量化方案}
大小集 = {dtype: [0] * len(配置列表) for dtype in 所有量化方案}

def _自动单位(b):
    if b >= 1073741824: return f"{b/1073741824:.1f}GB"
    if b >= 1048576: return f"{b/1048576:.1f}MB"
    if b >= 1024: return f"{b>>10}KB"
    return f"{b}B"

def _载荷字节(载荷):
    """载荷字节数: 块缩放改造后 Min/Scale 等键可能是嵌套字典(子载荷), 所以要递归统计"""
    if isinstance(载荷, dict):
        return sum(_载荷字节(值) for 值 in 载荷.values())
    if hasattr(载荷, 'nbytes'):
        return int(载荷.nbytes)
    return len(str(载荷))

输出锁 = threading.Lock()
任务队列 = queue.Queue(maxsize=队列深度)

def A线程量化(投递):
    """A 线程：逐配置 × 逐格式 编码 → 解码 → 投递结果（不参与召回统计）"""
    for cfg_idx, cfg in enumerate(配置列表):
        M = cfg.get("VEC_QUANTIZATION_PQ_M", "?")
        NBITS = cfg.get("VEC_QUANTIZATION_PQ_NBITS", "?")
        with 输出锁:
            print(f"[{cfg_idx+1}/{len(配置列表)}] ⚙️  M={M}, NBITS={NBITS}")
        if 应用配置:
            for 键, 值 in cfg.items():
                setattr(核心配置.Config, 键, 值)
        量化模块 = Quantization(核心配置)

        for dtype in 所有量化方案:
            CleanVRAM()
            try:
                压缩向量 = 量化模块.编码向量(当前向量, dtype)
                压缩后大小 = _载荷字节(压缩向量)
                恢复向量 = 量化模块.解码向量(压缩向量, dtype)
                if GPU_ACC and hasattr(恢复向量, "get"):
                    恢复向量 = 恢复向量.get()   # 跨线程前先转回 numpy，避免 cupy 上下文问题
                del 压缩向量
                原始大小 = 当前向量.nbytes
                with 输出锁:
                    print(f"  {dtype} | 配置 M={M} NBITS={NBITS} | 原始={_自动单位(原始大小)} → 压缩后={_自动单位(压缩后大小)}")
                投递((cfg_idx, dtype, 恢复向量, 压缩后大小, None))
            except Exception:
                投递((cfg_idx, dtype, None, 0, eb.format_exc()))
    投递(None)

def 处理任务(任务):
    """B 线程核心：单条任务的召回率统计与结果落表"""
    cfg_idx, dtype, 恢复向量, 压缩后大小, 错误 = 任务
    if 恢复向量 is None:
        with 输出锁:
            print(f"  ❌ {dtype} 报错:\n{错误}")
        结果集[dtype][cfg_idx] = "Err"
        大小集[dtype][cfg_idx] = 0
        return
    try:
        r2, r3, r5, r10, r50, r100, r500, r1000 = 计算召回率(恢复向量)
        del 恢复向量
        结果集[dtype][cfg_idx] = (f"{r2*100:.1f}/{r3*100:.1f}/{r5*100:.1f}/{r10*100:.1f}/"
                                  f"{r50*100:.1f}/{r100*100:.1f}/{r500*100:.1f}/{r1000*100:.1f}")
        大小集[dtype][cfg_idx] = 压缩后大小
        with 输出锁:
            print(f"      {dtype} R@2={r2*100:.1f}% R@10={r10*100:.1f}% R@100={r100*100:.1f}% R@1000={r1000*100:.1f}%")
    except Exception:
        with 输出锁:
            print(f"  ❌ {dtype} 召回率计算报错:\n{eb.format_exc()}")
        结果集[dtype][cfg_idx] = "Err"
        大小集[dtype][cfg_idx] = 0

def B线程召回():
    """B 线程：从队列取结果 → 计算召回率，直到收到哨兵 None"""
    while True:
        任务 = 任务队列.get()
        try:
            if 任务 is None:
                return
            处理任务(任务)
        finally:
            任务队列.task_done()

开始时间 = time.monotonic()

if 启用双线程:
    A = threading.Thread(target=A线程量化, args=(任务队列.put,), name="A-量化")
    B = threading.Thread(target=B线程召回, name="B-召回")
    A.start(); B.start()
    A.join(); B.join()
else:
    A线程量化(处理任务)

print(f"\n⏱️ 扫描耗时 {time.monotonic() - 开始时间:.1f}s "
      f"({'双线程流水线' if 启用双线程 else '单线程串行'} / 队列深度={队列深度})")

# ==========================================
# 7. 生成 Markdown 表格
# ==========================================
num_configs = len(配置列表)
header_cols = [f"配置 {i+1}" for i in range(num_configs)]

md  = f"| R@2/3/5/10/50/100/500/1000 | {' | '.join(header_cols)} |\n"
md += f"| :--- | {' | '.join([':---:'] * num_configs)} |\n"

md += f"| 范围 | Min.{vec_min:.7f} | Max.{vec_max:.7f} | {' | '.join([''] * (num_configs - 2))} |\n"
# 动态读取配置键生成行
if 配置列表:
    配置键列表 = list(配置列表[0].keys())
    for 键 in 配置键列表:
        行名 = 键.replace("VEC_QUANTIZATION_", "").replace("_", " ").strip()
        md += f"| {行名} | {' | '.join(str(c.get(键, '-')) for c in 配置列表)} |\n"

for dtype in 所有量化方案:
    合并行 = []
    for 召回率, 大小 in zip(结果集[dtype], 大小集[dtype]):
        合并行.append(f"{召回率} / {_自动单位(大小) if 大小 else '-'}")
    md += f"| {dtype} | {' | '.join(合并行)} |\n"

print(f"\n{'=' * 80}")
print("🏆 全量化方案网格扫描最终报告")
print(f"{'=' * 80}")
print(md)
