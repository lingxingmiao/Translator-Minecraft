from TranslatorLib import Mods, np

参数 = Mods().模组配置("quant_clip-sort", {"Clip": 0.998})


@Mods().注册裁切类型("Sort")
def Sort裁切(Self, 数组):
    百分比 = 参数["Clip"]
    if 百分比 is None or not 0.0 < float(百分比) < 1.0:
        return 数组
    数组 = np.array(np.asarray(数组, dtype=np.float32), copy=True)
    if 数组.ndim == 0:
        return 数组
    排序 = np.sort(数组, axis=-1)      # ↓ 最后一维: 一维(块缩放通道的 Min/Scale/均值)与二维(向量)都能用
    数量 = 数组.shape[-1]
    if 数量 <= 1:
        return 数组
    位置 = (1.0 - float(百分比)) * (数量 - 1)
    近序, 远序 = int(np.floor(位置)), int(np.ceil(位置))
    权重 = 位置 - 近序
    低位 = 排序[..., 近序] * (1.0 - 权重) + 排序[..., 远序] * 权重
    高位 = 排序[..., 数量 - 1 - 近序] * (1.0 - 权重) + 排序[..., 数量 - 1 - 远序] * 权重
    np.clip(数组, 低位[..., None], 高位[..., None], out=数组)
    del 排序
    return 数组
