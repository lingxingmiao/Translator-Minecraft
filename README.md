# Translator Minecraft
Translator Minecraft 是 [Translator Lang](https://github.com/lingxingmiao/Tools/tree/main/Minecraft%20AI%E7%BF%BB%E8%AF%91%E5%B7%A5%E5%85%B7/ver1.0) 的重构版本，相比原先版本添加了更多功能(其实只是一个框架, 我想加什么就加什么)<br>
此程序提供API服务为以下Minecraft附加组件提供便捷的翻译功能
- 整合包
- 光影包
- 资源包
- 模组
- FTB任务
- 更好的任务
- 核电工艺重制版附加包
- 数据包(含起源)
- 帕秋莉手册
- CraftTweaker(displayName/tooltip)
- 自定义主菜单
- 精致菜单
- 困难任务
- 未知伤亡语言文件(非Minecraft)
- 未知伤亡dll模组(非Minecraft)
- [模组翻译](https://www.mcmod.cn/class/27911.html)

后续计划支持以下组件(可能是大饼)
- KubeJS
- Lavender手册(没什么模组用暂时就不做了)

您还可以导入 [DictMini.json](https://github.com/CFPATools/i18n-dict) 来提高翻译质量，也可以使用这个文件来导出数据集来微调大语言模型<br>
您可以在[工单](https://github.com/lingxingmiao/Translator-Minecraft/issues)中提交您的想法与程序中的问题(别怕, 尽可能表达好就可以)<br>
别看 R1.5 的 Core 文件末尾了, Key我已经删了哈哈哈, 你拿我也没办法哈哈哈哈哈哈哈哈哈。

### 画廊 (双击Ctrl放大)
| <img width="1920" height="1080" alt="9886e91fd0eb8eb3a605e7d2c8e802c5" src="https://github.com/user-attachments/assets/f26b2877-49dd-4a9a-a2f9-612aa563abe6" /> | <img width="1920" height="1080" alt="e30bc972ed445694c1cec45840e46281" src="https://github.com/user-attachments/assets/cced0c18-aa21-426f-bd23-9d1d60784677" /> | <img width="1920" height="1080" alt="ffa09f4b557e0fca9c74f89b8d630853" src="https://github.com/user-attachments/assets/26417c40-398c-4e6c-9500-caa2c349fcaf" /> |
| - | - | - |

### 支持的项目
- [CFPATools/i18n-dict](https://github.com/CFPATools/i18n-dict)
- [NakiriRuri/Minecraft-Shaders-zh_CN-Lang-Files](https://github.com/NakiriRuri/Minecraft-Shaders-zh_CN-Lang-Files)

### 推荐配置
- 中央处理器(程序占用)：CPU-Z多核3000分以上的64位处理器
- 计算加速器(可选)：NVIDIA支持CUDA Toolkit 12.8的Volta以上架构 推荐16GB以上内存
- 内存(程序占用)：最低4GB 推荐8GB

### API
公益API公开测试 https://api.tanslamc.top sk-123456 该API禁止商用/转发, 可自用, 自定义API请求请查看API与Config文件<br>
公益API当前使用模型: 
- [Q1ngMang/Gemma-4-E4B-it-Minecraft-MT-en-zh](https://huggingface.co/Q1ngMang/Gemma-4-E4B-it-Minecraft-MT-en-zh) (条目数大于2500时 20%分配到这里, 不满足该条件全部分配到这里)
- [DeepSeek V4 Flash](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) (条目数大于2500时 80%分配到这里)

### Minecraft Machine Translation Quality Metrics
即将推出... 预计Q2'26-Q4'26<br>
这是一个用于评估翻译模型/大语言模型翻译质量指标的。<br>
从4-12个选项中选择一个

题目分布:
- 模组: 55%
- 任务: 30%
- 光影: 15%

## 推荐模型翻译质量排名
<details>
<summary>点击展开/收起</summary>
警告：WDDM模式下使用LMStudi的CUDA进行并行推理可能会导致模型崩溃！<br>
建议：计算加速器推荐使用TCC模式来获得更快的速度！<br>
"*"表示强制启用推理链, 闭源不记入

### WMT24++ XCOMET-XXL
- [XiaomiResearch/MiLMMT-46-12B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-12B-v0.1) 86.6分
- [Tencent/HY-MT1.5-7B](https://huggingface.co/tencent/HY-MT1.5-7B) 85.7分
- [Google/translategemma-12b-it](https://huggingface.co/google/translategemma-12b-it) 85.5分
- [Tencent/HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B) 85.3分
- [XiaomiResearch/MiLMMT-46-4B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-4B-v0.1) 84.8分
- [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) 78.9分
- [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) 78.3分
- [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) 77.6分
- [XiaomiResearch/MiLMMT-46-1B-v0.1](https://huggingface.co/xiaomi-research/MiLMMT-46-1B-v0.1) 77.3分
- [Google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) 76.6分
- [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) 76.3分
- [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) 75.8分
- [OpenAi/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) 74.4分*
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) 72.6分
- [Qwen/Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) 69.3分*
- [OpenAi/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) 67.8分*
- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 66.6分
- [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) 58.9分*
- [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) 57.4分*
### WMT25 XCOMET-XXL
- [Tencent/Hy-MT2-30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B) 73.6分
- [Tencent/Hy-MT2-7B](https://huggingface.co/tencent/Hy-MT2-7B) 73.0分
- [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) 69.4分
- [DeepSeek/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) 66.5分
- [Google/Gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) 62.0分 当前公益API使用
- [Tencent/HY-MT1.5-7B](https://huggingface.co/tencent/HY-MT1.5-7B) 61.6分
- [Tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) 60.0分
- [Tencent/HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B) 53.1分
- [DeepSeek/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 50.1分
### 未知成绩
- [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) 预计 WMT24++ XCOMET-XXL 54±2分
- [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) 预计 WMT24++ XCOMET-XXL 65±2分
- [Qwen/Qwen2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M)
### 不可用
- [LiquidAI/LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B)
- [LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B)
- [MoonshotAI/Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)
- [Z.ai/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
</details>

## 量化类型
<details>
<summary>点击展开/收起</summary>
该区域/技术由AI管理
RMSE不代表Recall@10
    
### 类型
- _0 表示什么技术都没使用
- _K 表示常规块缩放
- _M 表示中心均值化
- _H 表示使用Hadamard变换，向量过多可能导致召回率下降
- _E 表示极端数值分离
- _NF 表示使用固定[-1.512, -0.453, 0.453, 1.512]质心，无法使用分位数，块大小32出现比特率大幅上升
- _LM 表示使用了Lloyd-Max学习专属质心
- _SVD 表示使用了SVD学习最优旋转，该步骤耗时较长
- TQ1 表示使用三值量化
- GSQ 表示旋转矩阵, 有_0, _K分支

### 存储
| 块大小 与 BPW/S | 128 | 64 | 32 | 自定义 |
| - | - | - | - | - |
| Q8_K | 8.1875 | 8.375 | 8.75 | 8+24/B |
| GSQ8_K | 8.25 | 8.5 | 9 | 8+32/B |
| Q6_K | 6.1875 | 6.375 | 6.75 | 6+24/B |
| GSQ6_K | 6.25 | 6.5 | 7 | 6+32/B |
| Q4_K | 4.1875 | 4.375 | 4.75 | 4+24/B |
| Q4_K_H | 4.25 | 4.5 | 5 | 4+32/B |
| Q4_NF | 4.125 | 4.25 | 4.5 | 4+16/B |
| Q4_NF_H | 4.125 | 4.25 | 4.5 | 4+16/B |
| Q4_SVD_LM | 4.125 | 4.25 | 4.5 | 4+16/B |
| GSQ4_0 | 4+32/N | 4+32/N | 4+32/N | 4+32/N |
| GSQ4_K | 4.25 | 4.5 | 5 | 4+32/B |
| Q3_K | 3.1875 | 3.375 | 3.75 | 3+24/B |
| GSQ3_K | 3.25 | 3.5 | 4 | 3+32/B |
| Q2_K | 2.1875 | 2.375 | 2.75 | 2+24/B |
| Q2_NF | 2.125 | 2.25 | 2.5 | 2+16/B |
| Q2_E_NF | 2.3125 | 2.625 | 3.25 | 2+40/B |
| Q2_NF_H | 2.125 | 2.25 | 2.5 | 2+16/B |
| Q2_E_NF_H | 2.3125 | 2.625 | 3.25 | 2+40/B |
| Q2_SVD_LM | 2.125 | 2.25 | 2.5 | 2+16/B |
| Q2_E_SVD_LM | 2.3125 | 2.625 | 3.25 | 2+40/B |
| GSQ2_K | 2.25 | 2.5 | 3 | 2+32/B |
| TQ1_SVD | 1.6625 | 1.725 | 1.85 | 1.6+8/B |
| Q1_K_M | 1.03125 | 1.0625 | 1.125 | 1+4/B |
| Float32 | 32 | 32 | 32 |
| Float16 | 16 | 16 | 16 |
| BFloat16 | 16 | 16 | 16 |
| Float16_E0M15 | 16 | 16 | 16 |
| Float8_E4M3 | 8 | 8 | 8 |

### 使用 nomic-ai/nomic-embed-text-v1.5
| RMSE/余弦相似度损失 | SentenceTransformer | SentenceTransformer | Llama.cpp | Llama.cpp |
| - | - | - | - | - |
| 范围 | [-6.3716235, 5.829162] | [-6.3716235, 5.829162] | [-0.27004412, 0.24617423] | [-0.27004412, 0.24617423] |
| 分位数 | 1.000 | 0.998 | 1.000 | 0.998 |
| Q8_K_X | 0.0039/0.0000 | 0.0091/0.0001 | 0.0006/0.0001 | 0.0007/0.0002 |
| Q6_K_X | 0.0156/0.0002 | 0.0175/0.0002 | 0.0007/0.0002 | 0.0008/0.0002 |
| Q4_K_X | 0.0657/0.0034 | 0.0653/0.0034 | 0.0030/0.0034 | 0.0030/0.0034 |
| Q3_K_X | 0.1408/0.0153 | 0.1390/0.0150 | 0.0064/0.0153 | 0.0063/0.0150 |
| Q2_K_X | 0.3221/0.0849 | 0.3193/0.0833 | 0.0149/0.0761 | 0.0147/0.0747 |
| Float32 | 0.0000/0.0000 |   | 0.0000/0.0000 |
| Float16 | 0.0002/0.0000 |   | 0.0000/0.0000 |
| Float16_E0M15 | 0.2441/0.0355 |   | 0.0000/0.0000 |
| BFloat16 | 0.0026/0.0000 |   | 0.0001/0.0000 |
| Float8_E4M3 | 0.0405/0.0004 |   | 0.0028/0.0020 |

### 使用 BAAI/bge-large-en
| R@2/R@3/R@5/R@10 | 配置 1 | 配置 2 | 配置 3 | 配置 4 | 配置 5 | 配置 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 范围 | Min.-0.1631309 | Max.0.3503784 |  |  |  |  |
| 分位数 | 1.000 | 1.000 | 1.000 | 0.998 | 0.998 | 0.998 |
| 块大小 | 128 | 64 | 32 | 128 | 64 | 32 |
| Q8_K | 99.6%/99.6%/99.6%/99.4% | 99.6%/99.6%/99.5%/99.4% | 99.7%/99.5%/99.3%/99.3% | 99.3%/99.3%/99.2%/99.1% | 99.4%/99.5%/99.4%/99.3% | 99.5%/99.4%/99.3%/99.2% |
| GSQ8_K | 99.3%/99.4%/99.6%/99.8% | 99.2%/99.5%/99.6%/99.8% | 99.2%/99.5%/99.7%/99.8% | 99.3%/99.4%/99.6%/99.8% | 99.2%/99.5%/99.6%/99.8% | 99.2%/99.5%/99.7%/99.8% |
| Q6_K | 99.4%/99.3%/99.1%/99.0% | 99.5%/99.4%/99.2%/99.2% | 99.6%/99.4%/99.3%/99.3% | 99.3%/99.2%/98.9%/98.9% | 99.3%/99.4%/99.2%/99.0% | 99.5%/99.4%/99.3%/99.2% |
| GSQ6_K | 99.1%/99.2%/99.2%/99.4% | 99.1%/99.2%/99.3%/99.4% | 99.1%/99.2%/99.4%/99.5% | 99.1%/99.2%/99.2%/99.4% | 99.1%/99.2%/99.3%/99.4% | 99.1%/99.2%/99.4%/99.5% |
| Q4_K | 97.8%/97.3%/96.4%/96.3% | 98.1%/97.6%/96.9%/96.8% | 98.2%/97.9%/97.4%/97.3% | 98.1%/97.3%/96.7%/96.5% | 98.2%/97.7%/97.2%/97.0% | 98.3%/98.3%/97.5%/97.3% |
| Q4_K_H | 98.2%/97.4%/96.9%/96.6% | 98.3%/97.9%/97.0%/97.0% | 98.5%/98.2%/97.6%/97.4% | 98.2%/97.7%/96.9%/96.8% | 98.2%/97.8%/97.2%/97.1% | 98.5%/98.1%/97.5%/97.4% |
| Q4_SVD_LM | 98.4%/98.1%/97.3%/97.3% | 98.4%/98.0%/97.4%/97.3% | 98.4%/98.0%/97.4%/97.1% | 98.3%/98.0%/97.4%/97.3% | 98.4%/98.0%/97.4%/97.1% | 98.4%/97.9%/97.3%/97.2% |
| GSQ4_0 | 97.8%/96.9%/96.1%/95.9% | 97.8%/96.9%/96.1%/95.9% | 97.8%/96.9%/96.1%/95.9% | 98.5%/98.0%/97.4%/97.3% | 98.5%/98.0%/97.4%/97.3% | 98.5%/98.0%/97.4%/97.3% |
| GSQ4_K | 98.4%/98.1%/97.7%/97.7% | 98.3%/98.2%/98.1%/98.0% | 98.6%/98.6%/98.3%/98.2% | 98.4%/98.1%/97.7%/97.7% | 98.3%/98.2%/98.1%/98.0% | 98.6%/98.6%/98.3%/98.2% |
| Q3_K | 95.5%/94.4%/92.6%/92.5% | 96.2%/95.2%/93.8%/93.4% | 96.9%/95.8%/94.6%/94.4% | 95.9%/94.9%/93.2%/93.0% | 96.3%/95.3%/93.9%/93.6% | 96.9%/95.8%/94.7%/94.4% |
| GSQ3_K | 96.7%/96.2%/95.2%/95.2% | 97.3%/96.5%/95.8%/95.7% | 97.3%/96.9%/96.5%/96.4% | 96.7%/96.2%/95.2%/95.2% | 97.3%/96.5%/95.8%/95.7% | 97.3%/96.9%/96.5%/96.4% |
| Q2_K | 89.9%/86.9%/84.2%/83.2% | 91.6%/89.2%/86.6%/85.8% | 92.8%/90.6%/88.4%/87.6% | 90.8%/87.7%/85.3%/84.2% | 91.8%/89.3%/87.2%/86.0% | 93.0%/90.7%/88.7%/87.8% |
| Q2_E_NF | 94.4%/92.6%/91.2%/90.7% | 94.8%/93.2%/91.5%/91.0% | 95.2%/93.4%/92.1%/91.5% | 94.5%/92.7%/91.3%/90.7% | 94.8%/93.2%/91.6%/91.0% | 95.2%/93.4%/92.1%/91.6% |
| Q2_NF | 94.3%/92.2%/90.7%/90.1% | 94.5%/92.3%/90.8%/90.1% | 94.3%/92.5%/91.0%/90.3% | 94.3%/92.2%/90.8%/90.1% | 94.6%/92.5%/90.7%/90.1% | 94.5%/92.6%/91.0%/90.3% |
| Q2_SVD_LM | 93.9%/92.2%/90.4%/89.4% | 94.1%/92.4%/91.0%/90.0% | 94.5%/92.7%/91.1%/90.3% | 94.4%/92.6%/90.9%/89.8% | 94.0%/92.5%/91.0%/89.9% | 94.3%/92.7%/91.1%/90.2% |
| Q2_E_SVD_LM | 95.1%/93.7%/92.2%/91.8% | 95.1%/93.9%/92.4%/91.8% | 95.7%/94.1%/92.5%/92.0% | 95.0%/93.7%/92.2%/91.8% | 95.0%/93.8%/92.4%/91.8% | 95.6%/93.9%/92.5%/91.9% |
| GSQ2_K | 92.4%/90.0%/88.5%/87.7% | 92.9%/91.0%/89.5%/89.1% | 93.6%/92.1%/90.7%/90.5% | 92.4%/90.0%/88.5%/87.7% | 92.9%/91.0%/89.5%/89.1% | 93.6%/92.1%/90.7%/90.5% |
| TQ1_SVD | 92.6%/90.0%/87.8%/86.7% | 92.2%/90.0%/87.7%/87.0% | 92.6%/90.3%/87.9%/86.9% | 92.6%/90.0%/87.8%/86.7% | 92.2%/90.0%/87.7%/87.0% | 92.6%/90.3%/87.9%/86.9% |
| Q1_K_M | 92.4%/89.7%/87.7%/86.5% | 92.6%/89.9%/87.7%/86.5% | 92.6%/89.9%/87.8%/86.6% | 92.5%/89.7%/87.6%/86.5% | 92.6%/89.8%/87.7%/86.5% | 92.6%/90.0%/87.8%/86.6% |
| Float32 | 100.0%/100.0%/100.0%/100.0% | 100.0%/100.0%/100.0%/100.0% | 100.0%/100.0%/100.0%/100.0% | 100.0%/100.0%/100.0%/100.0% | 100.0%/100.0%/100.0%/100.0% | 100.0%/100.0%/100.0%/100.0% |
| Float16 | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% |
| BFloat16 | 99.9%/99.9%/99.9%/99.9% | 99.9%/99.9%/99.9%/99.9% | 99.9%/99.9%/99.9%/99.9% | 99.9%/99.9%/99.9%/99.9% | 99.9%/99.9%/99.9%/99.9% | 99.9%/99.9%/99.9%/99.9% |
| Float16_E0M15 | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% | 99.9%/100.0%/100.0%/100.0% |
| Float12_E0M11 | 99.9%/99.9%/99.8%/99.8% | 99.9%/99.9%/99.8%/99.8% | 99.9%/99.9%/99.8%/99.8% | 99.9%/99.9%/99.8%/99.8% | 99.9%/99.9%/99.8%/99.8% | 99.9%/99.9%/99.8%/99.8% |
| Float8_E4M3 | 97.6%/97.1%/96.1%/95.9% | 97.6%/97.1%/96.1%/95.9% | 97.6%/97.1%/96.1%/95.9% | 97.6%/97.1%/96.1%/95.9% | 97.6%/97.1%/96.1%/95.9% | 97.6%/97.1%/96.1%/95.9% |
| Float8_E0M7 | 98.7%/98.4%/97.6%/97.6% | 98.7%/98.4%/97.6%/97.6% | 98.7%/98.4%/97.6%/97.6% | 98.7%/98.4%/97.6%/97.6% | 98.7%/98.4%/97.6%/97.6% | 98.7%/98.4%/97.6%/97.6% |
</details>

## 编译/环境设置
<details>
<summary>点击展开/收起</summary>

```powershell
# 创建环境
conda create -n Translator_Minecraft python=3.12 -y
# 激活环境
conda activate Translator_Minecraft
#          向量处理 向量索引 网络请求 进度显示与艺术 json加速 dll解析
pip install numpy   faiss   requests    rich        ujson   dnfile
# API服务器（可选）
pip install uvicorn fastapi slowapi
# 内置向量生成（可选）
pip install -U "sentence-transformers[onnx]" # 或 pip install -U "sentence-transformers[onnx-gpu]"
pip install einops
pip install uninstall torch
pip install torch==2.9.1 torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cu128
# 向量处理加速（可选）
conda install -c conda-forge cupy cuda-version=12.8 # GPU 要打包别安
pip install numba # CPU
# 打包exe 没有做torch兼容 手动打包cupy不可用
pip install nuitka
nuitka --standalone --jobs=40 --include-package=rich --include-package=uvicorn TranslatorAPI.py
# 退出环境
conda deactivate
# 删除环境
conda env remove -n Translator_Minecraft
```
</details>

## 翻译流程
<details>
<summary>点击展开/收起</summary>

- "*"表示 **开发中** 或者 **未完成**
[![](https://mermaid.ink/img/pako:eNqdll1P2lAYx78KOZcLKm_doFl2ocbshgszrww3ja1ABtR0kL0QErfIiwITE9HMEQ2bG2RTmYtzBEz8Mj1t-RYrPaUe4LQ5jISEPuf3P89zzvN_GrJgQ-QFwILNhPh6I8ZJadfaciTl0j_qfV_rHAy2T9TDtnJUlPu3rrm5Z2ZY61xq7W0UpsBX1hblfh_uNSnYxVVq9PlqmJpdf6HtnCiNCwp0KRzW9huwWqc5GT0Ku3W1VdY-7iq7ZVioUCiU-g2slWAlT8X-UqpXlGzjh3r2Tb47l3vNGTuJS_lEQmk31f4OrrNqpnALiSVahQSSfELiiCYhgSSHkDiiPYhHoeQcjDGOW08wfy33fqo3X-Bd_RHiyWuup1jr_rxX22VaerpnThrkAxTC67e6iTC5V9HuP8PS7QK8PFYvvsvd31rrq3Jag5_auMzq2Wyy0YXPpiJaesbE1jHJ1z3uWEfo4ehO1MirjpBlVEdq5T8h26G2ebk47k_ugaNi9L5zpCZGy5G1KRaFYelYa47bxtiAAsTjaIyPinrrpodlCtQvXOtU7UCsUrvxm049BeLFmPN_eKaUarB2MCjuLyi1hr6GICTAi6IS4DnpBHhJxhgaAwFLhUG-ak3eWB221FhyW8rmtojsRBCB3T25W9brVq-aaq3gyKLDo4xmdiyCIdam5l5TOezZiaABap2_yvUHeN4y3dNuwlPT_8Qlx4NFUsANolKcB2xayghukBSkJDd8BNkhEQHpmJAUIoDVf_Kc9DICIqmcrtniUuuimBzJJDETjQF2k0u80p8yWzyXFpbjXFTiHhAhxQvSkphJpQEb8IWMPQCbBW8Ay3jmGU-QYXxMwB8Ihjx-N3gLWG_gsR5mmKD-9Yc8Qe-TnBu8M7J65736ik-XeLyBoMc7VAh8PC1KYfQ_2Pg7nPsH2-oheg?type=png)](https://mermaid-live.nodejs.cn/edit#pako:eNqdll1P2lAYx78KOZcLKm_doFl2ocbshgszrww3ja1ABtR0kL0QErfIiwITE9HMEQ2bG2RTmYtzBEz8Mj1t-RYrPaUe4LQ5jISEPuf3P89zzvN_GrJgQ-QFwILNhPh6I8ZJadfaciTl0j_qfV_rHAy2T9TDtnJUlPu3rrm5Z2ZY61xq7W0UpsBX1hblfh_uNSnYxVVq9PlqmJpdf6HtnCiNCwp0KRzW9huwWqc5GT0Ku3W1VdY-7iq7ZVioUCiU-g2slWAlT8X-UqpXlGzjh3r2Tb47l3vNGTuJS_lEQmk31f4OrrNqpnALiSVahQSSfELiiCYhgSSHkDiiPYhHoeQcjDGOW08wfy33fqo3X-Bd_RHiyWuup1jr_rxX22VaerpnThrkAxTC67e6iTC5V9HuP8PS7QK8PFYvvsvd31rrq3Jag5_auMzq2Wyy0YXPpiJaesbE1jHJ1z3uWEfo4ehO1MirjpBlVEdq5T8h26G2ebk47k_ugaNi9L5zpCZGy5G1KRaFYelYa47bxtiAAsTjaIyPinrrpodlCtQvXOtU7UCsUrvxm049BeLFmPN_eKaUarB2MCjuLyi1hr6GICTAi6IS4DnpBHhJxhgaAwFLhUG-ak3eWB221FhyW8rmtojsRBCB3T25W9brVq-aaq3gyKLDo4xmdiyCIdam5l5TOezZiaABap2_yvUHeN4y3dNuwlPT_8Qlx4NFUsANolKcB2xayghukBSkJDd8BNkhEQHpmJAUIoDVf_Kc9DICIqmcrtniUuuimBzJJDETjQF2k0u80p8yWzyXFpbjXFTiHhAhxQvSkphJpQEb8IWMPQCbBW8Ay3jmGU-QYXxMwB8Ihjx-N3gLWG_gsR5mmKD-9Yc8Qe-TnBu8M7J65736ik-XeLyBoMc7VAh8PC1KYfQ_2Pg7nPsH2-oheg)
</details>

## 更新日志
版本：我看着水差不多了就发，看起来没毛然后下一个版本大于Bata.2我就会发一个正式版，或者后面要更新大坨的就发。
### Release.1
含1个α版本与2个β版本
- 添加 语言文件 翻译支持
- 添加 IndexFlatL2方法RAG检索
- 添加 资源包 翻译支持 （光影，模组，资源包）
- 添加 导出数据集功能
- 添加 导入参考词功能
- 添加 最大历史上下文
- 添加 文件传入参数
- 添加 额外依赖 numpy faiss-cpu

### Release.1.1
含2个β版本
- 添加 翻译 FTBQ任务 BQ任务
- 添加 思考模型支持（仅为强制思考模型做支持）
- 修改 IndexFlatL2索引 改为 IndexHNSWSQ索引(SQ8)
- 修改 ThreadPoolExecutor索引并发 改为 Faiss并行
- 修复 翻译语言文件 双文件无法正确处理
- 修复 无法传入上下文开关参数
- 添加 额外依赖 ujson pyhocon

### Release.1.2
含2个1版本
- 添加 自动汉化更新 的 I18n词典 导入参考词功能（[Dict-Mini.json](https://github.com/CFPATools/i18n-dict)）
- 添加 向量索引缓存功能（SHA3-256校验 .pkl 与 .npy 文件，生成 .faiss-sha3 与 .faiss 文件）
- 修改 批量翻译 格式
- 修改 向量存储的格式从 .npy 改为 .npz，格式可选:
    - Float32
    - Float16_E0M15
    - Uint8+Float16
    - Uint4+Float16
- 修复 FTBQ 与 BQ 任务翻译无法传入的问题
- 修复 批量翻译 翻译键值映射问题
- 删除 额外依赖 ujson

### Release.1.3 Bata.1
- 大量修改传入方式
- 添加 IndexRefineFlat 方法
- 添加 翻译资源文件 单文件传入键值自动补全
- 添加 模糊匹配语言代码
- 添加 日志功能（目前只有 翻译资源文件、翻译语言文件、生成翻译 有写入日志）
- 更改 向量存储的格式从 Uint4/8 量化的 块缩放 格式从 Float16 改为 Float16_E0M15：
    - Uint8+Float16_S1M15
    - Uint4+Float16_S1M15

### Release.1.3 Bata.2
- 添加 翻译解析/向量生成 错误重试功能
- 添加 CuPy 加速支持
- 添加 SentenceTransformer 自动加载嵌入模型（ONNX、Safetensors）
- 添加 更改 单个词语 翻译为 字符串 而不是 列表
- 添加 I18n词典 导出 数据集 功能（[CFPATools/i18n-dict Dict-Mini.json](https://github.com/CFPATools/i18n-dict)）
- 添加 分离语言文件更新 与 合并语言文件更新，翻译审查文件需要从 翻译语言文件 获取，翻译提示词文件需要从 分离语言文件更新 获取
- 添加 翻译缓存替换
- 添加 导入DictMini参考词 添加 稠密 与 稀疏 模式（默认 稠密，稠密 性能好）
- 添加 向量存储格式:
    - Float16
    - BFloat16
    - Uint6+Float16_S1M15
    - Uint3+Float16_S1M15
    - Uint2+Float16_S1M15
- 修复 Json 语言文件解析错误
- 修复 批量翻译 参考词仅传入一个的问题
- 修复 增加向量 时发生的错误
- 修改 拆分 Config 类与 Quantization 类到两个新的文件
- 修改 Lib 部分函数分离至一个新的文件（Module） 
- 修改 翻译资源文件 合并至 翻译语言文件
- 删除 Lib函数 删除 导出数据集 函数

### Release.1.3
- 大量修改传入方式
- 修改 日志现在不用全塞一个文件了
- 修改 所有（除了SentenceTransformer）需要导入的函数分离至 TranslationLib
- 添加 日志、tqdm 本地化支持（默认Lang文件夹下 Minecraft同格式）
- 添加 MCP调用（需要上占用下文约 4500Tokens）
- 添加 系统变量:
    - FENGMANG_GPU_ACC（是否启用GPU加速）
    - FENGMANG_GPU_DEVICE_ID（GPU加速设备ID）

### Release.1.4 Bata.1
- 添加 调用大语言模型额外参数传入
- 添加 已安装的整合包翻译支持（支持 KJS FTBQ BQ 资源文件夹 模组，支持 I18n翻译剔除）
- 添加 FTB任务与BQ任务翻译自动剔除键与滚木
- 添加 单实例 嵌入模型、向量文件、文本文件、向量索引 持久化（多实例无持久化）
- 添加 翻译任务自动分离 "&§x{key}srt" 混合编码后进行翻译
- 添加 自动汉化更新 的 I18n词典 导入翻译缓存功能（[CFPATools/i18n-dict Dict-Mini.json](https://github.com/CFPATools/i18n-dict)）
- 添加 翻译对照功能，返回示例：想妈妈了(think ma ma le)
- 添加/修改 向量量化存储相关: (神经)
    - 添加 Float8_E4M3
    - 修改添加 非对称量化（Q系列量化）
    - 修改添加 分位数裁剪（Q系列量化）
    - 修改 向量叠加只能叠加浮点数据
- 修改 向量存储的内部格式为类似字典
- 修改 FTB任务文件读取固定4个线程，而不是处理器逻辑处理器的二分之一
- 修复 配置文件相关操作时可能导致的奔溃问题
- 修复 分离语言文件更新 导出路径没有填写的问题
- 修复 翻译语言文件输出语言文件路径不正确导致的一系列问题
- 修复 翻译内容分号哪哪都是导致的解析错误（于 Release.1.4 Bata.3 删除）
- 修复 日志未启用DEBUG导致无视DEBUG选项直接输出INFO
- 修复 翻译任务返回重新排序导致的错误
- 修复 读取FTB任务遇到双重编码字符串导致程序无法正确处理的问题（于 Release.1.5 Bata.1 正确修复）
- 修复 翻译任务传入文件路径与键路径导致的LLM预填充耗时增加的问题

### Release.1.4 Bata.2
- 添加 FastApi（供应商模式，headers:{"Authorization": f"Bearer sk-114514"}）（通用翻译，分离语言文件更新，合并语言文件更新）
- 添加 通用翻译函数
    - 额外支持 未安装整合包（支持 CurseForge Modrinth MultiMC）（仅翻译整合包内已有文件生成补丁）
    - 支持 Zip压缩后的 Minecraft实例 FTB任务 BQ任务 文件夹
- 修改 任务文件读取/写入固定4个线程，而不是处理器逻辑处理器的二分之一
- 修复 Module.从模组文件夹获取模组ID 的缩进块问题
- 修复 整合包翻译没有资源包文件夹导致的错误
- 修复 整合包翻译没有模组文件夹导致的错误
- 优化 MCP服务所有翻译改用 通用翻译 Qwen3.5约4778Token Qwen3约4808Token Gamma4约4361Token 

### Release.1.4 Bata.3
- 添加 API异步轮询（Cloudflare Tunnel有上限太搞了）
- 添加 API返回日志功能
- 添加 API翻译实例工作上限（阻塞）
- 添加 API每IP限速器
- 添加 API SSL文件配置
- 添加 API设置翻译/日志语言
- 添加 全局日志开关
- 添加 详细进程日志
- 添加 模糊匹配日志语言文件名
- 添加 未找到指定语言日志文件退回中文，如果还未找到则退回滚木
- 修改 GPU加速模式默认为false
- 修改 启用上下文默认为false
- 修改 API现在来自所有域名请求都可以处理
- 修改 通用翻译函数遇到问题强制返回错误信息
- 修改 翻译任务读取或写入文件现在可以进行配置
- 修改 缓存路径全部改为uuid库生成
- 修改 翻译缓存改为全局功能
- 修复 通用翻译函数部分不会返回Path对象的问题
- 修复 解析模组ID错误没有在该程序语言文件的问题
- 修复 FTB任务读取subtitle项为列表无法处理的问题
- 修复 FTB任务读取在日志拉屎的问题
- 修复 部分片段代码可读性或可维护性过高的问题
- 修复 向量叠加失败文本会继续叠加的问题
- 修复 LLM与EMB模型请求重试不会等待的问题
- 修复 LLM请求核爆的时候会直接导致线程错误的问题
- 修复 LLM参数导致出现部分回答错误
- 修复 保存Lang文件遇到 \n 时无法正确处理的bug
- 修复 输入遇到引号缓存无法命中的问题（来自 Release.1.4 Bata.1 的修复 翻译内容分号哪哪都是导致的解析错误）

### Release.1.4
- 添加 SentenceTransformer设备选择
- 优化 嵌入模型多实例持久化
- 优化 不同向量文本多实例持久化
- 优化 索引文件多实例持久化
- 修改 GPU加速模式配置更改至config.cfg
- 修改 Tqdm的Tqdm改为Rich的Tqdm
- 修改 mcp-config.cfg 与 api-config.cfg 改为 config-mcp.cfg 与 config-api.cfg

### Release.1.5 Bata.1
- 添加 高质量索引模式(非任务Token消耗平均翻1-3倍 任务翻译峰值超20倍以上)
- 添加 [think]思维链剔除[/think]
- 添加 翻译模型 频率惩罚 存在惩罚 种子 参数传入
- 添加 翻译的时候每个词条写入一次日志
- 优化 额外提示词传入逻辑(Tokens更少)
- 优化 翻译系统提示词(复制 沉浸式翻译 插件)
- 优化 API任务状态持久化
- 修复 FTB任务点击事件翻译完消失的问题
- 修复 通用翻译不会返回值导致的错误跳出
- 修复 模组或资源包内有多个语言文件导致只翻译一个文件的问题
- 修复 翻译语言设置无效的问题(配置文件功能加这么久才发现)
- 修复 BQ任务文件读取 questLines 值为 questDatabase 的问题(666我问Qwen这段代码是不是AI写的时候发现的)
- 修复 翻译遇到错误重试时发生的错误
- 修复 翻译对照功能无法正确处理导致的程序退出
- 修复 翻译返回会带有空格的问题
- 修复 通用翻译不了资源包的问题
- 修改 .translatorlang 文件改为 .translang 文件(翻译审查文件)
- 修改 .translang 文件内部格式,现在更清晰明了
- 修改 翻译语言文件不限制文件类型必须相同
- 删除 额外提示词传入方式, 现在固定系统提示词位置
- 删除 分离语言文件函数的mode参数,固定输出语言文件
- 添加 SnbtLib依赖
- 删除 PyHocon依赖

### Release.1.5 Bata.2
- 添加 API任务状态自动清理
- 添加 翻译/嵌入HTTP请求超时(含系数)
- 添加 导入DictMini缓存使用重排序获取译文相似度最高的文本([CFPATools/i18n-dict Dict-Mini.json](https://github.com/CFPATools/i18n-dict))(默认模型:[Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B))
- 优化 HTTP请求性能
- 优化 正则表达式性能
- 优化 FTB/BQ任务读取/写入性能
- 优化 向量生成变量内存占用
- 优化 FaissSQ量化格式索引
- 优化 日志写入性能
- 优化 if index in []性能
- 优化 翻译缓存内存占用
- 优化 量化/反量化/叠加性能
- 优化 pkl文件去重性能
- 修复 键与值相同会进行翻译的问题(来自公益AP的Imodid:gvcr2翻译请求)
- 修复 单符号会进行翻译的问题(来自公益API的modid:gvcr2翻译请求)
- 修复 [oωo-lib](https://www.mcmod.cn/class/5043.html)模组添加[文本组件](https://zh.minecraft.wiki/w/%E6%96%87%E6%9C%AC%E7%BB%84%E4%BB%B6)格式支持导致的翻译错误的问题(来自公益API的modid:playerex翻译请求)
- 修复 R1.5,B1的 模组或资源包内有多个语言文件导致只翻译一个文件的问题 导致的无索引翻译报错的问题
- 修复 API请求文件带有空格导致的报错问题(来自公益API的"[荧光棒] glow_sticks-fabric-1.21.11-7.3.0.jar"翻译请求)
- 修复 翻译全部命中缓存导致的返回为空错误(来自公益API的"[荧光棒] glow_sticks-fabric-1.21.11-7.3.0.jar"翻译请求)
- 修复 读取日志非UTF-8核爆的问题
- 修复 没有npz向量文件时且有pkl文本文件时pkl文本文件不会写入的问题
- 修复 API会强制设置翻译器语言导致的API Server设置无效的问题
- 修改 LLM_ORIGINAL_REFERENCE为TRANSLATOR_ORIGINAL_REFERENCE
- 修改 LLM/EMB_RETRY_INTERVAL为LLM/EMB_RETRY_TIME
- 修改 LLM_USER_PROMPT为TRANSLATOR_USER_PROMPT
- 修改 LLM_SYSTEM_PROMPT为TRANSLATOR_SYSTEM_PROMPT
- 修改 LLM_CONTEXTS与LLM_CONTEXTS_LENGTH合并为LLM_CONTEXTS
- 修改 API任务状态持久化为SQLite

### Release.1.5
AI给我加了一堆BUG所以不发布
- 添加 导入DictMini提示词可以导入翻译缓存文件
- 添加 语言文件对转换DictMini.json(主要用于转换[NakiriRuri/Minecraft-Shaders-zh_CN-Lang-Files](https://github.com/NakiriRuri/Minecraft-Shaders-zh_CN-Lang-Files))
- 添加 MMTQM机器翻译质量指标第一代(Minecraft Machine Translation Quality Metrics)(依赖翻译列表函数，使用配置注入)
- 添加 API翻译任务请求重复文件直接返回错误(可配置)(老爱有人往我API重复请求)
- 添加 翻译索引开关, 即INDEX_TEXT_K=0
- 添加 同模组已有翻译索引
- 添加 API请求翻译实例设置
- 添加 索引方法 IVPSQ IVPPQ IVPFlat HNSWPQ HNSWFlat FlatL2 FlatIP(文本索引默认HNSWPQ 模组索引默认FlatL2)
- 添加 索引范围统计方法
- 添加 翻译词条数超过x后使用另一个API(单API不变 多API用LLM0, LLM1等数字后缀)
- 添加 导入DictMini参考词 最大长度限制(可配置)
- 添加 自适应翻译上下文顺序
- 添加 相同翻译内容合并为一个请求
- 添加 翻译压缩包内含有contenttweaker文件夹是返回全部内容([NuclearCraft: Overhauled](https://www.mcmod.cn/class/2483.html)兼容)
- 添加 翻译整合包NeoForge模组ID获取
- 优化 向量相关操作内存占用
- 优化 Faiss缓存校验速度
- 修复 翻译读取资源文件lang文件夹内是文件夹的语言文件会报错的问题(不翻译)(来自公益API的modid:sswaystones翻译请求)
- 修复 导入DictMini缓存即使是都是只有一个项也会加载重排序模型的问题
- 修复 多程序日志语言文件无法使用的问题
- 修复 退出的时候日志写一半没写完的问题
- 修复 通用翻译光影的时候即使正常也会炸的问题
- 修复 [oωo-lib](https://www.mcmod.cn/class/5043.html)解析导致翻译任务报错的问题
- 修复 新建翻译实例运行时会重新解码向量导致内存多占一份的问题
- 修复 翻译整合包不会翻译[NuclearCraft: Overhauled](https://www.mcmod.cn/class/2483.html)插件的问题
- 修复 翻译没有获取到结果发送日志的时候会二次报错导致程序退出
- 修复 单文件夹内多个同语言不同扩展名只会翻译一个的问题
- 修改 任务读取/写入从 计算密集 改为 IO密集
- 修改 高质量索引拆分为多个配置
- 修改 json替换为ujson
- 修改 LLM_MAX_BATCH改为TRANSLATOR_BATCH
- 修改 启动标题大标题改为彩色
- 删除 API请求语言设置
- 删除 高质量索引(功能被细化)
- 删除 导入查看词(文件夹版本, 非Dict Mini)
- 停止支持 TranslatorMCPServer.py(太没用了)

### Release.1.6 Bata.1
- 添加 翻译(看到BBSMC有好多功能想搬了)
    - 数据包(含起源)
    - 帕秋莉手册
    - CraftTweaker(displayName/tooltip)
    - 自定义主菜单
    - 精致菜单
    - 困难任务(含1.7.10自定义二进制文件)
    - 未知伤亡语言文件(非Minecraft)
    - 未知伤亡dll模组(非Minecraft)
- 添加 整合包翻译当参考词传入接下来的其他类型翻译提示词
- 添加 翻译迭代精炼
- 添加 导入DictMini参考词剔除重复内容
- 添加 导入DictMini参考词反转模式
- 添加 多翻译模型动态分配
- 优化 翻译极高并发(约256+)导致的API连接超时问题
- 优化 因为单次多次与逐条翻译的识别问题为此我准备了两套说辞
- 优化 翻译上下文管理器性能
- 优化 翻译上下文管理器获取顺序
- 修复 翻译语言列表没有翻译参考列表导致的报错
- 修复 翻译语言列表键为列表导致的报错
- 修复 翻译语言列表没有传入使用模型列表列表导致的报错
- 修复 生成翻译传入列表键导致的报错
- 修复 上下文管理器传入列表键导致的报错问题
- 修复 通用翻译不能翻译非MultiMC实例
- 修复 合并语言文件更新打包格式错误的问题
- 修复 参考词预处理无法生成向量的问题
- 修复 三种模组索引模式结果混乱的问题
- 修复 应用本地化缺少替换参数导致的报错
- 修复 日志再里面拉换行导致我看不清内容的问题
- 修改 Core的Translator类内的部分函数到其他/新的文件
- 修改 snbtlib内置进TranslatorLib
- 修改 BQ/FTB任务读取/写入改为深度优先搜索
- 修改 翻译语言列表启用任务模式时也会使用owolib解析(文本组件)
- 修改 合并已有翻译从翻译语言文件移动至翻译语言列表
- 修改 批量翻译提示词与逐条翻译提示词分离

### Release.1.6 Bata.2 (进行中)
- 添加 翻译
    - [模组翻译](https://www.mcmod.cn/class/27911.html)(Roo Code + DeepSeek V4 Pro太强了)
- 添加 量化方法
    - Q4_K_H Q4_SVD_LM GSQ4_0(Q4_K优化版, 代替Q4_K)
    - Q2_NF Q2_E_NF Q2_SVD_LM Q2_E_SVD_LM(Q2_K优化版, Q3_K代替品)
    - TQ1_SVD(三值高压缩，Q2_K代替品)
    - Q1_K_M(二值极高压缩, Q2_K代替品)
    - GSQ_K系列(GSQ_0改进版, 涵盖2-8bit, 平均为同bit顶尖水平)
- 添加 向量重排(仅QSG_K有效, 索引库[IndexGSQ](https://github.com/lingxingmiao/IndexGSQ/))
- 添加 DictMini转换数据集随机排布
- 添加 DictMini转换数据集 Alpaca 格式加强版(Alpaca-EX)
- 添加 翻译流程支持的类型新增DictMini导出(**警告**:[未知伤亡](https://store.steampowered.com/app/4576490/)含暴力、血腥以及抑郁和自残内容，**若此类内容占比大于5%则严禁投喂AI**)
- 添加 翻译上下文去重
- 添加 翻译模型重复惩罚配置
- 添加 构建索引函数train()随机采样(Float32设置百分比 Int32设置值)
- 添加 基础索引函数用于构建索引函数的IVF与Refine等方法
- 添加 Faiss GPU支持![](https://img.shields.io/badge/状态-等待中-blue)(还要CuPy兼容不是很好整)
- 添加 所有库懒加载(需要Python版本 >= Python 3.15.0a7)
- 添加 Faiss线程设置(Float32设置百分比 Int32设置值)
- 添加 翻译任务夜间进行(部分API 8折)
- 添加 量化器Numba支持
- 添加 量化器_LM计算早停配置添加进配置项
- 添加 翻译请求池
- 添加 [IndexGSQ](https://github.com/lingxingmiao/IndexGSQ/) 超低内存索引(Fast召回率:2bit 90%>,8bit 99.9999%>, 还有速度内存召回率都要的MoE版本)
- 添加 翻译整合包模组并行翻译模组
- 优化 批量翻译我换另一种说法让ast.literal_eval改为ujson.loads
- 优化 模型路由器的 动态执行包装 函数性能
- 优化 上下文管理器的 .get 性能
- 优化 翻译整合包删除缓存性能
- 优化 量化器性能与内存
- 修改 分离TranslatorModule到新的文件
- 修改 删除所有量化方法的"_X" 示例:Q2_K_X -> Q2_K
- 修改 批量翻译核爆太多次退回到逐条翻译
- 修改 翻译思维链剔除改为预编译正则表达式
- 修改 INDEX配置修改多数独立的配置
- 修改 删除了INDEX MODE的Flat文本
- 修改 API顶部的库导入移动到Lib(不安装依旧不会影响Core文件使用)
- 修复 LANG索引第一次生成索引add完再add导致越界索引
- 修复 翻译流程通用参数互相打架导致报错的问题
- 修复 应用dll翻译读取文件路径不对导致报错
- 修复 应用dll翻译可能找不到
- 修复 未知伤亡模组prosthetics V38无法应用翻译的问题
- 修复 保存翻译审查文件出现越界索引的错误
- 修复 命中缓存还会翻译的问题
- 修复 过滤dll文本多处漏过滤导致出现游戏报错的问题
- 修复 批量翻译和逐条翻译提示词写反导致批量翻译无法正常工作的问题
- 修复 通用翻译文件翻译.json文件进入未知伤亡匹配的if的dict in str写反导致的报错
- 修复 DictMini导出Alpaca数据集多出一个字段的问题
- 修复 翻译整合包 pack.metadata 文件没有使用配置里面的值
- 修复 log.core.translator.generate.start使用位置条目数不正确的问题
- 修复 动态分配翻译任务进度条是键的问题
- 修复 使用模型添加null的索引越界报错
- 修复 读取模组ID TQDM核爆的问题
- 修复 反量化后的向量会一直占内存的问题
- 修复 整合包保存语言文件时文件消失的问题(我也不知道是不是这个版本的问题)
- 修复 并行启动翻译函数时会同时并行读取索引文件与向量文件导致内存爆炸的问题
- 修复 翻译JSON语言文件值为列表无法翻译的问题![](https://img.shields.io/badge/状态-进行中-brightgreen)
- 修复 FTBQ写入
    - 写入时索引写成切片的报错
    - 写入时字典当成字符串的报错
- 修复 内置snbtlib的_tokenize函数
    - MULTILINE报错
    - FTBQ注释缩进报错
    - 类型后缀导致的奔溃
- 删除 R1.4 B1 添加的翻译任务自动分离 "&§x{key}srt" 混合编码后进行翻译
- 添加 tqdm 依赖

### 计划
- 添加 翻译耗时预测器
- 修改 自定义tqdm

### 编辑需要
- ![](https://img.shields.io/badge/状态-等待中-blue) ![](https://img.shields.io/badge/状态-进行中-brightgreen) ![](https://img.shields.io/badge/状态-完成-brightgreen) ![](https://img.shields.io/badge/状态-修复极高风险漏洞-FF0000) 
- 预设：`brightgreen` `green` `yellowgreen` `yellow` `orange` `red` `blue` `lightgrey`
