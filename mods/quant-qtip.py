# -*- coding: utf-8 -*-
"""QTIP 编解码(时变网格 / 4 子集 / 4 状态 / 【可配置】每子集电平数 L)
   码率 = 分支密度(3/4) + log2(L) bit/维
   电平 = 4 子集 × L 电平; 索引流每个字节能装 每字节 = max{k: L**k <= 256} 个符号
     L=1( 4 电平): 72(分支位 576bit) + 0  (子集已定电平, 不存索引)      =  72 B/行 (768 维)
     L=2( 8 电平): 72 +  96(8 个/字节)                                  = 168 B/行
     L=3(12 电平): 72 + 154(5 个/字节)  ← 默认, 与旧版【逐字节相同】 ✓   = 226 B/行
     L=4(16 电平): 72 + 192(4 个/字节)                                  = 264 B/行
   关键: 分支位是"被传输的信息", 所以
        · 编码端: Viterbi(一次性, 构建期)
        · 解码端: 只需把分支位喂进卷积码【前向】推出子集序列 -> 无 Viterbi, 每维约 4 次运算
"""
import numpy as np
from numba import njit, prange

周期 = 4
分支数 = 3
状态位 = 2
G1 = 0b101
G2 = 0b111
每子集 = 3
电平数 = 每子集 * 4          # 12


def 建表(g1=G1, g2=G2, ν=状态位, 周=周期, 分数=分支数):
    ns = 1 << ν
    S2 = np.zeros((ns, 2), np.int32); SJ = np.zeros((ns, 2), np.int32)
    for s in range(ns):
        for b in (0, 1):
            reg = (b << ν) | s
            c1 = bin(reg & g1).count("1") & 1
            c2 = bin(reg & g2).count("1") & 1
            S2[s, b] = ((s << 1) | b) & (ns - 1)
            SJ[s, b] = (c1 << 1) | c2
    return S2, SJ


S2表, SJ表 = 建表()


def 每字节符号(每子):
    """每个字节最多能装几个电平符号: 最大的 k(<=8) 使 每子**k <= 256"""
    每子 = max(1, int(每子))
    for k in range(8, 0, -1):
        if 每子 ** k <= 256:
            return k
    return 1


def 建索表(每子):
    """索表[字节, j] = 该字节里的第 j 个电平符号(低位在前, 每字节符号 位/字节)
       每子=3 时与旧版 三表 逐元素相同 ✓"""
    每子 = max(1, int(每子)); 每字节 = 每字节符号(每子)
    表 = np.zeros((256, 每字节), np.uint8)
    for b in range(256):
        v = b
        for j in range(每字节):
            表[b, j] = v % 每子; v //= 每子
    return np.ascontiguousarray(表)


# 三进制(默认 L=3)解包查表: 三表[字节, j] = 该字节的第 j 位(低位在前, 5 位/字节)
三表 = 建索表(3)


def 每行字节(维度, 周=周期, 分数=分支数, 每子=每子集):
    """(每行总字节, 分支字节, 索引字节)
       每子=3 / 周=4 / 分数=3 时 = (226, 72, 154) ✓ 与旧版一致"""
    分支位数 = (维度 // 周) * 分数 + min(维度 % 周, 分数)
    分支字节 = (分支位数 + 7) // 8
    if int(每子) <= 1:
        索字节 = 0                                   # 4 电平: 子集就是电平 -> 不存索引
    else:
        每 = 每字节符号(每子)
        索字节 = (维度 + 每 - 1) // 每
    return 分支字节 + 索字节, 分支字节, 索字节


@njit(cache=True, fastmath=True, inline="always")
def _每字节符号(每子):
    for k in range(8, 0, -1):
        if 每子 ** k <= 256:
            return k
    return 1


@njit(cache=True, fastmath=True, inline="always")
def _分支位(位流, 基, 分支字节, t, 周, 分数):
    """第 t 维的分支位(只对 t%周<分数 的维有效)"""
    idx = (t // 周) * 分数 + (t % 周)
    return (np.uint8(位流[基 + (idx >> 3)]) >> np.uint8(idx & 7)) & np.uint8(1)


@njit(cache=True, fastmath=True, inline="always")
def _子集索(索流, 基, t, 每子, 每字节):
    q = t // 每字节; r = t - q * 每字节
    v = np.int32(索流[基 + q])
    for _ in range(r):
        v //= 每子
    return v % 每子


@njit(cache=True, fastmath=True)
def 解码行(位流, 索流, 基, D, S2, SJ, 周, 分数, 每子, 出):
    """前向推出每维绝对电平号(0..4*每子-1) -> 出; 每维约 4 次运算, 无 Viterbi"""
    每字节 = _每字节符号(每子)
    有索 = 每子 > 1
    s = 0
    for t in range(D):
        if (t % 周) < 分数:
            b = _分支位(位流, 基, 0, t, 周, 分数)
            j = SJ[s, b]
            s = S2[s, b]
        else:
            j = SJ[s, 0]
        if 有索:
            k = _子集索(索流, 基, t, 每子, 每字节)
        else:
            k = 0
        出[t] = k * 4 + j
    return 0


@njit(cache=True, fastmath=True, parallel=True)
def 编码全(方向, 码本, S2, SJ, 周, 分数, 每子, 段长=0):
    """方向 (N,D) -> 位流 (N,分支字节), 索流 (N,索字节); Viterbi 每行独立, 行间并行"""
    N, D = 方向.shape
    状态数 = S2.shape[0]
    分支位数 = (D // 周) * 分数 + min(D % 周, 分数)
    分支字节 = (分支位数 + 7) // 8
    每字节 = _每字节符号(每子)
    有索 = 每子 > 1
    索字节 = (D + 每字节 - 1) // 每字节 if 有索 else 0
    位流 = np.zeros((N, 分支字节), np.uint8)
    索流 = np.zeros((N, 索字节), np.uint8)
    子 = np.empty((4, 每子), np.float32)
    for j in range(4):
        for c in range(每子):
            子[j, c] = 码本[c * 4 + j]              # 电平号 = 子集内序号*4 + 子集 ✓
    幂 = np.ones(每字节, np.int32)
    for r in range(1, 每字节):
        幂[r] = 幂[r - 1] * 每子
    for i in prange(N):
        累 = np.full(状态数, np.float32(1e30)); 累[0] = np.float32(0)
        前态 = np.zeros((D, 状态数), np.int32)
        前索 = np.zeros((D, 状态数), np.int32)
        for t in range(D):
            if 段长 > 0 and t > 0 and (t % 段长) == 0:          # 分段: 每段独立, 从状态0起
                累 = np.full(状态数, np.float32(1e30)); 累[0] = np.float32(0)
            x = np.float32(方向[i, t])
            发 = (t % 周) < 分数
            新累 = np.full(状态数, np.float32(1e30))
            for s in range(状态数):
                if 累[s] > np.float32(1e29): continue
                for b in range(2):
                    if (not 发) and b == 1: continue
                    s2 = S2[s, b] if 发 else s
                    j = SJ[s, b]
                    best = np.float32(1e30); bk = 0
                    for k in range(每子):
                        e = x - 子[j, k]; e = e * e
                        if e < best: best = e; bk = k
                    if 累[s] + best < 新累[s2]:
                        新累[s2] = 累[s] + best; 前态[t, s2] = s; 前索[t, s2] = bk
            累 = 新累
        s = 0; best = np.float32(1e30)
        for q in range(状态数):
            if 累[q] < best: best = 累[q]; s = q
        # 回溯: 写回 位流(分支位) 与 索流(每子进制索引)
        for t in range(D - 1, -1, -1):
            k = 前索[t, s]
            if 有索:
                p = t // 每字节; r = t - p * 每字节
                v = np.int32(索流[i, p]); cur = v
                for _ in range(r): cur //= 每子
                cur %= 每子
                索流[i, p] = np.uint8(v + (k - cur) * 幂[r])
            sp = 前态[t, s]
            if (t % 周) < 分数:
                # 反查是哪个输入位把 sp 带到 s
                bb = 1 if S2[sp, 1] == s else 0
                idx = (t // 周) * 分数 + (t % 周)
                位流[i, idx >> 3] |= np.uint8(bb << (idx & 7))
            s = sp
    return 位流, 索流
