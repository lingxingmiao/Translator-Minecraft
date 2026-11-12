# -*- coding: utf-8 -*-
"""GQTIP_NL —— GSQ_NL 的分组逐列仿射 + QTIP 的 4 子集格码（实验原型）

参考 GSQ_NL（mods/quant-gsq_nl.py）的骨架，把它的【逐列 min/scale 归一 + 共享
非线性电平表】搬到 QTIP 的【4 子集 / 状态窗格码】上：

    编码  x --(组内逐列 min/scale)-->  z ∈ [0,1]
          z --(共享非线性电平表 + Viterbi 分支位 + 每子索引)-->  码流
    解码  分支位 --(前向状态机)--> 子集 j
          索流   --(每字节装 每字节 个符号)--> 子集内序号 c
          电平号 = c*4 + j  ->  x̂ = 码本[电平号] * scale + min

与 QTIP 的差别：QTIP 用一张全局表直接量化原值；GQTIP_NL 先用【每 128 行 × 逐列】
的仿射把分布摊到 [0,1]，再走同一套格码。多出来的开销是 min/scale 元数据
（每维每 128 行 2 个值 × 位宽）。

与 GSQ_NL 的差别：GSQ_NL 每维独立打 K 个电平（log2(K) bit/维）；GQTIP_NL 的
电平号受状态机约束，码率 = 分支密度 + log2(每子) bit/维。

声明位数（payload，不含元数据）：
    每子=1 -> 0.75            （4 电平，无索流）
    每子=2 -> 1.75            （8 电平）
    每子=3 -> 2.335           （12 电平，与 QTIP 默认档同）
    每子=4 -> 2.75            （16 电平）
元数据（位宽 b）：2 · b / 128 bit/维 = b/64 bit/维。

本文件是原型：纯 numpy + numba，不依赖 TranslatorLib；不会自动注册量化类型。
需要接入主项目时调用 `注册为量化类型()`（仅在 TranslatorLib 可用时生效）。
"""

import os
import sys
import importlib.util as _ilu

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------- quant-qtip
def _载入_qtip():
    """★ numba 的 cache=True 会按【模块名】回装缓存 -> 动态加载必须先占住 sys.modules，
       否则第二次运行会报 ModuleNotFoundError: No module named '<dynamic>' ✗"""
    for 名 in ("mods.quant-qtip", "quant-qtip"):
        if 名 in sys.modules:
            return sys.modules[名]
    try:
        import mods.quant_qtip as _m          # 包路径可用时
        return _m
    except Exception:
        pass
    路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant-qtip.py")
    名 = "mods.quant-qtip"
    spec = _ilu.spec_from_file_location(名, 路径)
    m = _ilu.module_from_spec(spec)
    sys.modules[名] = m                      # ★ 先注册再 exec
    spec.loader.exec_module(m)
    return m


qtip = _载入_qtip()


# ---------------------------------------------------------------- 格码表
def 建格(状态窗=4, G1=0b1011, G2=0b1110):
    """与 index-csd.py 的 _建QTIP 逐元素相同：状态 = 最近 状态窗 个被传输的比特"""
    K = max(1, int(状态窗))
    掩 = (1 << K) - 1
    子 = np.zeros(1 << K, np.int32)
    for r in range(1 << K):
        子[r] = ((bin(r & G1).count("1") & 1) << 1) | (bin(r & G2).count("1") & 1)
    S2 = np.zeros((1 << K, 2), np.int32)
    SJ = np.zeros((1 << K, 2), np.int32)
    for r in range(1 << K):
        SJ[r, 0] = 子[r]; SJ[r, 1] = 子[r]
        S2[r, 0] = (r << 1) & 掩
        S2[r, 1] = ((r << 1) | 1) & 掩
    反 = np.zeros(1 << K, np.int32)
    for r in range(1 << K):
        反[r] = int(format(r, "0%db" % K)[::-1], 2)
    return (np.ascontiguousarray(子), np.ascontiguousarray(S2),
            np.ascontiguousarray(SJ), np.ascontiguousarray(反))


# ---------------------------------------------------------------- 1D 电平表
def 学习LM(样本, 电平数, 迭代=80, 早停=1e-7, 采样上限=2_000_000, 种子=42):
    """一维 Lloyd-Max（与 AI测试 里既有脚本同口径）"""
    样本 = np.asarray(样本, np.float32).ravel()
    if 样本.size > 采样上限:
        样本 = 样本[np.random.default_rng(种子).integers(0, 样本.size, 采样上限)]
    样本 = np.sort(样本)
    中心 = np.percentile(样本, np.linspace(0, 100, 电平数 + 2)[1:-1]).astype(np.float64)
    for _ in range(迭代):
        边界 = (中心[:-1] + 中心[1:]) / 2.0
        索引 = np.searchsorted(边界, 样本)
        计数 = np.bincount(索引, minlength=电平数).astype(np.float64)
        加权 = np.bincount(索引, weights=样本.astype(np.float64), minlength=电平数)
        新 = np.where(计数 > 0, 加权 / np.maximum(计数, 1e-30), 中心)
        if float(np.max(np.abs(新 - 中心))) < 早停:
            中心 = 新
            break
        中心 = 新
    return np.sort(中心).astype(np.float32)


def 定点(数组, 位):
    """b 位定点（1 符号位 + b-1 小数位），用于 min/scale 的二级量化"""
    if 位 is None or int(位) <= 0:
        return np.asarray(数组, np.float32)
    级 = float((1 << (int(位) - 1)) - 1)
    return (np.round(np.asarray(数组, np.float32) * 级) / 级).astype(np.float32)


def 每行字节(维度, 每子=3, 周期=4, 分支=3, 位宽=16, 块=128):
    """返回 (总B/行, 载荷B, 元数据B, 分支B, 索B)；B/行 含 min/scale 元数据"""
    D = int(维度)
    分支位数 = (D // 周期) * 分支 + min(D % 周期, 分支)
    分支B = (分支位数 + 7) // 8
    每子 = max(1, int(每子))
    if 每子 <= 1:
        索B = 0
    else:
        每字节 = qtip.每字节符号(每子)
        索B = (D + 每字节 - 1) // 每字节
    元B = 0.0 if 位宽 is None else 2.0 * int(位宽) * D / (int(块) * 8.0)
    return 分支B + 索B + 元B, 分支B + 索B, 元B, 分支B, 索B


# ---------------------------------------------------------------- 解码内核
@njit(fastmath=True, parallel=True)
def _解码全部(位流, 索流, 维度, 反位, 子表, 索表, 状态窗, 周期, 分支, 每子, 每字节, 出):
    """前向状态机解码（与 index-csd.py 的 _qtip行 同语义，2D 批量版）
       位流 每行尾部需有 >=2 字节零填充（3 字节不对齐读）"""
    K = 状态窗; 掩 = (1 << K) - 1
    N = 位流.shape[0]
    for i in prange(N):
        idx = 0; q5 = 0; r5 = 0
        for t in range(维度):
            m = idx if idx < K else K
            if m == 0:
                j = 子表[0]
            else:
                o = idx - m
                w = (np.int32(位流[i, o >> 3]) | (np.int32(位流[i, (o >> 3) + 1]) << 8)
                     | (np.int32(位流[i, (o >> 3) + 2]) << 16))
                j = 子表[反位[(w >> (o & 7)) & 掩] >> (K - m)]
            if (t % 周期) < 分支:
                idx += 1
            if 每子 > 1:
                出[i, t] = 索表[索流[i, q5], r5] * 4 + j
            else:
                出[i, t] = j
            r5 += 1
            if r5 == 每字节:
                r5 = 0; q5 += 1
    return 0


# ---------------------------------------------------------------- 编码 / 解码
def 编码(数组, 每子=3, 周期=4, 分支=3, 状态窗=4, G1=0b1011, G2=0b1110,
         位宽=16, 块=128, LM迭代=80, 精炼=0, 采样上限=2_000_000, 种子=42, 码本=None):
    """GQTIP_NL 编码
    参数:
        每子   每子集电平数（总电平 = 4*每子；1/2/3/4 -> 4/8/12/16 电平）
        位宽   min/scale 元数据位宽（None/0 = 不量化）
        精炼   码本重估轮数（0 = 只用 Lloyd-Max 表；1.. = Viterbi 分配后重估均值）
        码本   可选：外部指定电平表（做 NL/均匀 对照时用）；None = 现场拟合
    返回模型 dict（含 位流/索流/码本/min/scale/形状/...）
    """
    原始 = np.ascontiguousarray(数组, np.float32)
    行数, 维度 = 原始.shape
    填充 = (-行数) % 块
    总行 = 行数 + 填充
    if 填充:
        原始 = np.concatenate([原始, np.zeros((填充, 维度), np.float32)], axis=0)
    组数 = 总行 // 块

    三维 = 原始.reshape(组数, 块, 维度)
    最小 = 三维.min(axis=1)                                  # (组数, D)
    最大 = 三维.max(axis=1)
    缩放 = np.maximum(最大 - 最小, 1e-8)
    del 最大

    最大最小 = max(float(np.max(np.abs(最小))), 1e-8)
    最大缩放 = max(float(np.max(缩放)), 1e-8)
    最小q = (定点(最小 / 最大最小, 位宽).astype(np.float32) * 最大最小)
    缩放q = np.maximum(定点(缩放 / 最大缩放, 位宽).astype(np.float32) * 最大缩放, 1e-8)

    z = (三维 - 最小[:, None, :]) / 缩放[:, None, :]
    np.clip(z, 0.0, 1.0, out=z)
    z = np.ascontiguousarray(z.reshape(总行, 维度), np.float32)
    del 三维

    if 码本 is None:
        步 = max(1, z.size // max(1, 采样上限))
        码本 = 学习LM(z.ravel()[::步], 4 * int(每子), LM迭代, 种子=种子)
    else:
        码本 = np.sort(np.asarray(码本, np.float32).ravel())
        if 码本.size != 4 * int(每子):
            raise ValueError(f"GQTIP_NL: 码本长度 {码本.size} != 4*每子 {4 * int(每子)}")

    子表, S2, SJ, 反位 = 建格(状态窗, G1, G2)
    每字节 = qtip.每字节符号(每子) if 每子 > 1 else 0
    索表 = np.ascontiguousarray(qtip.建索表(每子), np.uint8)

    位流 = 索流 = None
    for 轮 in range(int(精炼) + 1):
        位流, 索流 = qtip.编码全(z, 码本, S2, SJ, 周期, 分支, 每子)
        if 轮 < int(精炼):
            电平 = np.empty((总行, 维度), np.uint8)
            # ★ 解码要 3 字节不对齐读 -> 先补 2 字节尾零（否则最后一行越界）✓
            位流补 = np.concatenate([np.asarray(位流, np.uint8),
                                     np.zeros((总行, 2), np.uint8)], axis=1)
            _解码全部(位流补, 索流, 维度, 反位, 子表, 索表,
                      状态窗, 周期, 分支, int(每子), int(每字节), 电平)
            del 位流补
            和 = np.bincount(电平.ravel(), weights=z.ravel().astype(np.float64),
                            minlength=4 * int(每子))
            数 = np.bincount(电平.ravel(), minlength=4 * int(每子)).astype(np.float64)
            新 = np.where(数 > 0, 和 / np.maximum(数, 1.0), 码本)
            码本 = np.sort(新).astype(np.float32)

    位流 = np.ascontiguousarray(np.concatenate(
        [np.asarray(位流, np.uint8), np.zeros((总行, 2), np.uint8)], axis=1), np.uint8)
    索流 = np.ascontiguousarray(np.asarray(索流, np.uint8), np.uint8)
    return {"位流": 位流, "索流": 索流, "码本": np.ascontiguousarray(码本, np.float32),
            "最小": 最小q, "缩放": 缩放q, "行数": int(行数), "总行": int(总行),
            "维度": int(维度), "块": int(块), "每子": int(每子), "周期": int(周期),
            "分支": int(分支), "状态窗": int(状态窗), "G1": int(G1), "G2": int(G2),
            "位宽": 位宽, "每字节": int(每字节), "索表": 索表,
            "子表": 子表, "S2": S2, "SJ": SJ, "反位": 反位}


def 解码(模型):
    """GQTIP_NL 解码 -> (行数, D) float32"""
    总行 = int(模型["总行"]); 维度 = int(模型["维度"]); 块 = int(模型["块"])
    电平 = np.empty((总行, 维度), np.uint8)
    _解码全部(模型["位流"], 模型["索流"], 维度, 模型["反位"], 模型["子表"], 模型["索表"],
              模型["状态窗"], 模型["周期"], 模型["分支"], 模型["每子"],
              int(模型["每字节"]), 电平)
    z = np.asarray(模型["码本"], np.float32)[电平]
    组数 = 总行 // 块
    重建 = z.reshape(组数, 块, 维度) * 模型["缩放"][:, None, :] + 模型["最小"][:, None, :]
    return np.ascontiguousarray(重建.reshape(总行, 维度)[:int(模型["行数"])], np.float32)


# ---------------------------------------------------------------- 可选：接入主项目
def 注册为量化类型():
    """按 GSQ_NL 的方式注册 4 个档位（每子 = 1/2/3/4）。

    ★ 默认不自动调用：本次只验证算法；应用内加载/存档链路尚未实测。
       确认要接入时，在模组入口里调用一次即可。
    """
    try:
        from TranslatorLib import Mods
    except Exception:
        return None

    解码字典 = ["Vector", "Min", "Scale", "Codebook", "Bits", "Shape"]

    def _档位(名称, 每子):
        声明位数 = 0.75 + float(np.log2(max(1, 每子)))

        @Mods().注册量化类型(名称, 声明位数)
        def 编码入口(Self, 数组):
            模型 = 编码(数组, 每子=每子, 精炼=0)
            分支B = int(模型["位流"].shape[1]) - 2
            载荷 = np.concatenate([模型["位流"][:, :分支B].ravel(),
                                  模型["索流"].ravel()]).astype(np.uint8)
            位 = np.array([模型["每子"], 模型["周期"], 模型["分支"], 模型["状态窗"],
                           int(模型["位宽"] or 0), 模型["块"], 模型["G1"], 模型["G2"],
                           分支B], np.int64)
            return {"Vector": 载荷, "Min": 模型["最小"], "Scale": 模型["缩放"],
                    "Codebook": 模型["码本"], "Bits": 位,
                    "Shape": (模型["行数"], 模型["维度"])}

        @Mods().注册反量化类型(名称, 声明位数, 解码字典)
        def 解码入口(Self, Vector, Min, Scale, Codebook, Bits, Shape):
            行数, 维度 = int(Shape[0]), int(Shape[1])
            每子v, 周期v, 分支v, 状态窗v, 位宽v, 块v, G1v, G2v, 分支B = \
                [int(x) for x in np.asarray(Bits).ravel()[:9]]
            总行 = 行数 + ((-行数) % 块v)
            载荷 = np.asarray(Vector, np.uint8).ravel()
            分支总 = 总行 * 分支B
            位流 = 载荷[:分支总].reshape(总行, 分支B)
            索流 = (载荷[分支总:].reshape(总行, -1) if 每子v > 1
                    else np.zeros((总行, 0), np.uint8))
            位流 = np.ascontiguousarray(np.concatenate(
                [位流, np.zeros((总行, 2), np.uint8)], axis=1))
            子表, S2, SJ, 反位 = 建格(状态窗v, G1v, G2v)
            模型 = {"位流": 位流, "索流": 索流, "码本": np.asarray(Codebook, np.float32),
                    "最小": np.asarray(Min, np.float32), "缩放": np.asarray(Scale, np.float32),
                    "行数": 行数, "总行": 总行, "维度": 维度, "块": 块v,
                    "每子": 每子v, "周期": 周期v, "分支": 分支v, "状态窗": 状态窗v,
                    "每字节": int(qtip.每字节符号(每子v)) if 每子v > 1 else 0,
                    "反位": 反位, "子表": 子表,
                    "索表": np.ascontiguousarray(qtip.建索表(每子v), np.uint8)}
            return 解码(模型)

        return 编码入口, 解码入口

    产物 = [_档位(名, 每子) for 名, 每子 in
            (("GQTIP1_NL", 1), ("GQTIP2_NL", 2), ("GQTIP3_NL", 3), ("GQTIP4_NL", 4))]
    return 产物


__all__ = ["建格", "学习LM", "定点", "每行字节", "编码", "解码", "注册为量化类型"]
