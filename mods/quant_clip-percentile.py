from TranslatorLib import Mods, np

参数 = Mods().模组配置("quant_clip-percentile", {"Clip": 0.998})


@Mods().注册裁切类型("Percentile")
def Percentile裁切(Self, 数组):
    百分比 = 参数["Clip"]
    if 百分比 is None or not 0.0 < float(百分比) < 1.0:
        return 数组
    数组 = np.array(np.asarray(数组, dtype=np.float32), copy=True)
    高位, 低位 = Self._百分位(数组, float(百分比) * 100.0, 轴=-1, 保持维度=True)   # ↓ 最后一维: 一维/二维都能用
    np.clip(数组, 低位, 高位, out=数组)
    return 数组
