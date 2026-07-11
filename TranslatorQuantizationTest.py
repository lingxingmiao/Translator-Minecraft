import sys
from TranslatorLib import TranslatorPersistence, Quantization, Translator, GPU_ACC, CleanVRAM, eb, np, faiss, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. 初始化与数据加载
# ==========================================
核心模块 = Translator(Config={"VEC_FILE_NAME": "Vectors2", "VEC_QUANTIZATION": "Float32"})
#核心模块 = Translator()

向量 = TranslatorPersistence.参考词预处理(核心模块)[0].get()
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

#VEC_INT_DTYPE = 核心模块.Config.VEC_INT_DTYPE
VEC_INT_DTYPE = ["Q5_K_M" , "Q5_SVD_LM" , "Q5_K", "GSQ5_K", "Q1_SVD_LM"]
#VEC_FLOAT_DTYPE = 核心模块.Config.VEC_FLOAT_DTYPE
VEC_FLOAT_DTYPE = []

所有量化方案 = VEC_INT_DTYPE + VEC_FLOAT_DTYPE
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

# ==========================================
# 5. 召回率计算
# ==========================================
def 计算召回率(恢复数据库):
    if GPU_ACC:
        恢复数据库 = 恢复数据库.get()

    xb_pred = 恢复数据库.astype("float32").copy()
    faiss.normalize_L2(xb_pred)

    index_pred = faiss.IndexFlatIP(D)
    index_pred.add(xb_pred)
    _, I_pred = index_pred.search(xq, 1000)

    def batch_recall_count(gt, pred):
        gt_s   = np.sort(gt,   axis=1)
        pred_s = np.sort(pred, axis=1)
        counts = np.zeros(len(gt_s), dtype=np.int32)
        for i in range(len(gt_s)):
            counts[i] = len(np.intersect1d(gt_s[i], pred_s[i]))
        return counts.sum()

    r2     = batch_recall_count(I_gt_2,     I_pred[:, :2])     / (NQ * 2)
    r3     = batch_recall_count(I_gt_3,     I_pred[:, :3])     / (NQ * 3)
    r5     = batch_recall_count(I_gt_5,     I_pred[:, :5])     / (NQ * 5)
    r10    = batch_recall_count(I_gt_10,    I_pred[:, :10])    / (NQ * 10)
    r50    = batch_recall_count(I_gt_50,    I_pred[:, :50])    / (NQ * 50)
    r100   = batch_recall_count(I_gt_100,   I_pred[:, :100])   / (NQ * 100)
    r500   = batch_recall_count(I_gt_500,   I_pred[:, :500])   / (NQ * 500)
    r1000 =  batch_recall_count(I_gt_1000,  I_pred[:, :1000])  / (NQ * 1000)
    return r2, r3, r5, r10, r50, r100, r500, r1000

# ==========================================
# 6. 网格扫描
# ==========================================
结果集 = {dtype: [] for dtype in 所有量化方案}
大小集 = {dtype: [] for dtype in 所有量化方案}

def _自动单位(b):
    if b >= 1073741824: return f"{b/1073741824:.1f}GB"
    if b >= 1048576: return f"{b/1048576:.1f}MB"
    if b >= 1024: return f"{b>>10}KB"
    return f"{b}B"

for cfg_idx, cfg in enumerate(配置列表):
    M = cfg.get("VEC_QUANTIZATION_PQ_M", "?")
    NBITS = cfg.get("VEC_QUANTIZATION_PQ_NBITS", "?")
    print(f"[{cfg_idx+1}/{len(配置列表)}] ⚙️  M={M}, NBITS={NBITS}")
    量化模块 = Quantization(Config=cfg)

    for dtype in 所有量化方案:
        CleanVRAM()
        try:
            压缩向量 = 量化模块.编码向量(当前向量, dtype)
            压缩后大小 = sum(v.nbytes if hasattr(v, 'nbytes') else len(str(v)) for v in 压缩向量.values())
            原始大小 = 当前向量.nbytes
            print(f"  {dtype} | 配置 M={M} NBITS={NBITS} | 原始={_自动单位(原始大小)} → 压缩后={_自动单位(压缩后大小)}")
            恢复向量 = 量化模块.解码向量(压缩向量, dtype)
            r2, r3, r5, r10, r50, r100, r500, r1000 = 计算召回率(恢复向量)
            结果集[dtype].append(f"{r2*100:.1f}/{r3*100:.1f}/{r5*100:.1f}/{r10*100:.1f}/{r50*100:.1f}/{r100*100:.1f}/{r500*100:.1f}/{r1000*100:.1f}")
            大小集[dtype].append(压缩后大小)
            print(f"      {dtype} R@2={r2*100:.1f}% R@10={r10*100:.1f}% R@100={r100*100:.1f}% R@1000={r1000*100:.1f}%")
        except Exception:
            print(f"  ❌ {dtype} 报错:\n{eb.format_exc()}")
            结果集[dtype].append("Err")
            大小集[dtype].append(0)

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