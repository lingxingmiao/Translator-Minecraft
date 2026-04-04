from TranslatorLib import np
from TranslatorQuantization import Quantization
from TranslatorCore import Translator

种子 = 42
维度 = 768
长度 = 1000
量化模块 = Quantization(Config={"VEC_QUANTIZATION": "Float32"})
核心模块 = Translator(Config={"VEC_QUANTIZATION": "Float32"})
范围 = [[-1, 1], [-10, 10], [-20, 20], [-100, 100], [-1000, 1000], [-10000, 10000], [-100000, 100000]]
输出结果 = f"| RMSE/余弦相似度损失 | {' | '.join([str(index) for index in 范围])} |\n"
输出结果 += f"| - | - |{' - |'.join([str("") for _ in range(len(范围))])}\n"
VEC_INT_DTYPE = ["Q8_K_X", "Q6_K_X", "Q4_K_X", "Q3_K_X", "Q2_K_X"]
VEC_FLOAT_DTYPE = ["Float32", "Float16", "Float16_E0M15", "BFloat16", "Float8_E4M3"]
np.random.seed(种子)
#向量文件 = 核心模块.参考词预处理()[0]
#向量 = [量化模块.解码向量(向量文件)]
向量 = np.asarray([np.random.uniform(index[0], index[1], size=(长度, 维度)).astype(np.float32) for index in 范围]).astype(np.float32)

def RMSE计算(向量A, 向量B):
    return float(np.sqrt(np.mean((向量A - 向量B) ** 2)))
def 余弦相似度计算(v1, v2, eps=1e-8):
    norm1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    norm1 = np.maximum(norm1, eps)
    norm2 = np.maximum(norm2, eps)
    cos_sim = np.sum(v1 * v2, axis=-1, keepdims=True) / (norm1 * norm2)
    return float(np.mean(1 - cos_sim.squeeze()))

def 计算相似度(原始向量, 恢复向量):
    return RMSE计算(原始向量, 恢复向量), 余弦相似度计算(原始向量, 恢复向量)

for index in VEC_INT_DTYPE + VEC_FLOAT_DTYPE:
    输出结果列表 = []
    for index1 in 向量:
        压缩向量 = 量化模块.编码向量(index1, index)
        恢复向量 = 量化模块.解码向量(压缩向量, index)
        RMSE, 余弦相似度 = 计算相似度(index1, 恢复向量)
        输出结果列表.append(f"{RMSE:.4f}/{余弦相似度:.4f}")
    输出结果 += f"| {index} | {' | '.join(输出结果列表)} |\n"
print(输出结果)