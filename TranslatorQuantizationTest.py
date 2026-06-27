import sys
from TranslatorLib import TranslatorPersistence, Quantization, Translator, GPU_ACC, CleanVRAM, eb, np, faiss, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. 初始化与数据加载
# ==========================================
核心模块 = Translator(Config={"VEC_FILE_NAME": "Vectors2", "VEC_QUANTIZATION": "Float32"})
'''
向量 = TranslatorPersistence.参考词预处理(核心模块)[0]
if 向量.ndim == 1:
    向量 = 向量.reshape(1, -1)
当前向量 = 向量.astype(np.float32)'''

npy_file_path = "reordered_vectors_ivfpq_rerank_k32.npy" 
当前向量 = np.load(npy_file_path)
if 当前向量.ndim == 1:
    当前向量 = 当前向量.reshape(1, -1)
当前向量 = 当前向量.astype(np.float32)

N, D = 当前向量.shape
vec_min = float(np.min(当前向量))
vec_max = float(np.max(当前向量))

# ==========================================
# 2. 自动获取量化方案
# ==========================================

#VEC_INT_DTYPE = 核心模块.Config.VEC_INT_DTYPE
VEC_INT_DTYPE = ["GSQ8_K", "GSQ6_K", "GSQ4_K", "GSQ3_K", "GSQ2_K"]
#VEC_FLOAT_DTYPE = 核心模块.Config.VEC_FLOAT_DTYPE
VEC_FLOAT_DTYPE = []

所有量化方案 = VEC_INT_DTYPE + VEC_FLOAT_DTYPE
print(f"🎯 自动检测到 {len(所有量化方案)} 种量化方案: {所有量化方案}\n")

# ==========================================
# 3. 测试配置
# ==========================================
配置列表 = [
    {"VEC_QUANTIZATION_BLOCK_SIZE": 128, "VEC_QUANTILE": 1.000},
    {"VEC_QUANTIZATION_BLOCK_SIZE": 64,  "VEC_QUANTILE": 1.000},
    {"VEC_QUANTIZATION_BLOCK_SIZE": 32,  "VEC_QUANTILE": 1.000},
    #{"VEC_QUANTIZATION_BLOCK_SIZE": 128, "VEC_QUANTILE": 0.998},
    #{"VEC_QUANTIZATION_BLOCK_SIZE": 64,  "VEC_QUANTILE": 0.998},
    #{"VEC_QUANTIZATION_BLOCK_SIZE": 32,  "VEC_QUANTILE": 0.998},
]

# ==========================================
# 4. 预计算 Ground Truth（只算一次）
# ==========================================
K_MAX = 10
NQ = min(500, N)
np.random.seed(42)
查询索引 = np.random.choice(N, NQ, replace=False)
xq = 当前向量[查询索引].astype("float32").copy()
faiss.normalize_L2(xq)

原始归一化 = 当前向量.astype("float32").copy()
faiss.normalize_L2(原始归一化)

GT_INDEX = faiss.IndexFlatIP(D)
GT_INDEX.add(原始归一化)
_, I_gt = GT_INDEX.search(xq, K_MAX)

I_gt_2  = I_gt[:, :2]
I_gt_3  = I_gt[:, :3]
I_gt_5  = I_gt[:, :5]
I_gt_10 = I_gt

# ==========================================
# 5. 召回率计算（R@2 + R@3 + R@5 + R@10 一次出）
# ==========================================
def 计算召回率(恢复数据库):
    if GPU_ACC:
        恢复数据库 = 恢复数据库.get()

    xb_pred = 恢复数据库.astype("float32").copy()
    faiss.normalize_L2(xb_pred)

    index_pred = faiss.IndexFlatIP(D)
    index_pred.add(xb_pred)
    _, I_pred = index_pred.search(xq, K_MAX)

    I_pred_2  = I_pred[:, :2]
    I_pred_3  = I_pred[:, :3]
    I_pred_5  = I_pred[:, :5]
    I_pred_10 = I_pred

    def batch_recall_count(gt, pred):
        gt_s   = np.sort(gt,   axis=1)
        pred_s = np.sort(pred, axis=1)
        counts = np.zeros(len(gt_s), dtype=np.int32)
        for i in range(len(gt_s)):
            counts[i] = len(np.intersect1d(gt_s[i], pred_s[i]))
        return counts.sum()

    r2  = batch_recall_count(I_gt_2,  I_pred_2)  / (NQ * 2)
    r3  = batch_recall_count(I_gt_3,  I_pred_3)  / (NQ * 3)
    r5  = batch_recall_count(I_gt_5,  I_pred_5)  / (NQ * 5)
    r10 = batch_recall_count(I_gt_10, I_pred_10) / (NQ * K_MAX)
    return r2, r3, r5, r10

# ==========================================
# 6. 网格扫描
# ==========================================
结果集 = {dtype: [] for dtype in 所有量化方案}

for cfg_idx, cfg in enumerate(配置列表):
    print(f"[{cfg_idx+1}/{len(配置列表)}] ⚙️ 块大小={cfg['VEC_QUANTIZATION_BLOCK_SIZE']}, 分位数={cfg['VEC_QUANTILE']}")
    量化模块 = Quantization(Config=cfg)

    for dtype in 所有量化方案:
        CleanVRAM()
        try:
            压缩向量 = 量化模块.编码向量(当前向量, dtype)
            恢复向量 = 量化模块.解码向量(压缩向量, dtype)
            r2, r3, r5, r10 = 计算召回率(恢复向量)
            结果集[dtype].append(f"{r2*100:.1f}%/{r3*100:.1f}%/{r5*100:.1f}%/{r10*100:.1f}%")
            print(dtype, "召回率", f"{r2*100:.1f}%/{r3*100:.1f}%/{r5*100:.1f}%/{r10*100:.1f}%")
        except Exception:
            print(f"  ❌ {dtype} 报错:\n{eb.format_exc()}")
            结果集[dtype].append("Err")

# ==========================================
# 7. 生成 Markdown 表格
# ==========================================
num_configs = len(配置列表)
header_cols = [f"配置 {i+1}" for i in range(num_configs)]

md  = f"| R@2/R@3/R@5/R@10 | {' | '.join(header_cols)} |\n"
md += f"| :--- | {' | '.join([':---:'] * num_configs)} |\n"

md += f"| 范围 | Min.{vec_min:.7f} | Max.{vec_max:.7f} | {' | '.join([''] * (num_configs - 2))} |\n"
md += f"| 分位数 | {' | '.join(f'{c['VEC_QUANTILE']:.3f}' for c in 配置列表)} |\n"
md += f"| 块大小 | {' | '.join(str(c['VEC_QUANTIZATION_BLOCK_SIZE']) for c in 配置列表)} |\n"

for dtype in 所有量化方案:
    md += f"| {dtype} | {' | '.join(结果集[dtype])} |\n"

print(f"\n{'=' * 80}")
print("🏆 全量化方案网格扫描最终报告")
print(f"{'=' * 80}")
print(md)