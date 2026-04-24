# Translator Minecraft
Translator Minecraft 是 [Translator Lang](https://github.com/lingxingmiao/Tools/tree/main/Minecraft%20AI%E7%BF%BB%E8%AF%91%E5%B7%A5%E5%85%B7/ver1.0) 的重写版本，相比原先版本添加了更多功能<br>
此软件提供MCP与API服务为Minecraft的整合包、光影包、资源包、模组、FTB任务、更好的任务提供便捷的翻译功能<br>
还可以导入 [DictMini.json](https://github.com/CFPATools/i18n-dict) 来提高翻译质量，可以使用这个文件来导出数据集来微调大语言模型<br>
并且存储的向量还支持多种量化格式，以此来减少存储空间<br>
此外这个程序还可以通过NVIDIA来加速向量处理与程序内直接生成向量（需要手动配置环境）<br>
您可以在[工单](https://github.com/lingxingmiao/Translator-Minecraft/issues)中提交您的想法与程序中的问题<br>

## 演示
### API
公益API公开测试 https://lingxingmiao.github.io 该API禁止商用/转发<br>
[![Image](https://github.com/lingxingmiao/Translator-Minecraft/blob/main/Image/Web.png)](https://lingxingmiao.github.io)

## 推荐配置
- 中央处理器(程序占用)：CPU-Z多核3000分以上的64位处理器
- 计算加速器(可选)：NVIDIA支持CUDA Toolkit 12.0的Maxwell2.0以上架构 推荐16GB以上内存
- 内存(程序占用)：最低4GB 推荐8GB
- 存储：平均4GB（按向量大小）

## 编译/环境设置
```powershell
# 创建环境
conda create -n Translator_Minecraft python=3.12 -y
# 激活环境
conda activate Translator_Minecraft
#          向量处理 向量索引 网络请求 Snbt文件 进度显示与艺术
pip install numpy   faiss  requests snbtlib     rich
# API服务器（可选）
pip install uvicorn fastapi slowapi
# MCP服务器（可选）
pip install fastmcp
# 内置向量生成（可选）
pip install -U "sentence-transformers[onnx]" # 或 pip install -U "sentence-transformers[onnx-gpu]"
pip install einops
pip install uninstall torch
pip install torch==2.9.1 torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cu128
# 向量处理加速（可选 要打包环境别安）
conda install -c conda-forge cupy cuda-version=12.8
# 打包exe 没有做torch兼容 手动打包cupy不可用
pip install nuitka
nuitka --standalone --jobs=40 --include-package=rich --include-package=uvicorn TranslatorMCPServer.py
nuitka --standalone --jobs=40 --include-package=rich --include-package=uvicorn TranslatorAPI.py
# 退出环境
conda deactivate
# 删除环境
conda env remove -n Translator_Minecraft
```
## 推荐模型翻译速度排名 16GRAM
警告：LMStudi使用CUDA进行并行推理可能会导致模型崩溃！
建议：计算加速器推荐使用TCC模式来获得更快的速度！
- [LiquidAI/LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B)（单次多次翻译可能导致输出问）（激活参数过小不可用）
- [LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B)（单次多次翻译可能导致输出问题）（激活参数过小不可用）
- [Google/Gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)（预填充速度稍慢）
- [MoonshotAI/Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)（可能导致无限输出）（预填充速度慢）（激活参数过小不可用）
- [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)（综合最优，推荐！）
- [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)（单次多次翻译可能导致输出问题）（情商高）（综合第二优，推荐！）
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)（单次多次翻译可能导致输出问题）
- [Qwen/Qwen2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M)（稳定性好）
- [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)（高质量）（稳定性好）
- [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)（单次多次翻译可能导致输出问题）（情商高）
- [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)（单次多次翻译可能导致输出问题）（情商高）

## 更新日志
### Release.1 Alpha.1
- 添加 语言文件 翻译支持
- 添加 IndexFlatL2方法RAG检索
- 添加 额外依赖 numpy faiss-cpu

### Release.1 Bata.1
- 添加 资源包 翻译支持 （光影，模组，资源包）
- 添加 导出数据集功能
- 添加 导入参考词功能

### Release.1 Bata.2
- 修复 上下文 system 位置
- 修复 系统提示词为 system 时不会添加 user 的问题
- 添加 最大历史上下文

### Release.1
- 添加 文件传入参数 （如何使用 这一栏）

### Release.1.1 Bata.1
- 添加 FTBQ任务 翻译支持
- 更改 IndexFlatL2索引 改为 IndexHNSWSQ索引(SQ8)
- 更改 ThreadPoolExecutor索引并发 改为 Faiss并行
- 添加 额外依赖 pyhocon

### Release.1.1 Bata.2
- ~~修复了一些已知的问题。~~
- 修复 IndexHNSWSQ索引 没有训练就构建的错误
- 添加 BQ任务 翻译支持
- 添加 思考模型支持（仅为强制思考模型做支持）

### Release.1.1
- 修复 翻译语言文件 双文件无法正确处理
- 修复 无法传入上下文开关参数
- 添加 FTB任务 翻译支持 （选定版本 1.12.2 1.20.1）
- 添加 BQ任务 翻译支持 （选定版本 1.7.10 1.12.2）
- 添加 思考模型支持
- 添加 额外依赖 ujson

### Release.1.2 Bata.1
- 添加 自动汉化更新 的 I18n词典 导入参考词功能（[Dict-Mini.json](https://github.com/CFPATools/i18n-dict)）
- 添加 向量索引缓存功能（SHA3-256校验 .pkl 与 .npy 文件，生成 .faiss-sha3 与 .faiss 文件）
- 更改 向量存储的格式从 .npy 改为 .npz，格式可选:
    - Float32
    - Float16_E0M15
    - Uint8+Float16
    - Uint4+Float16
- 修复 FTBQ 与 BQ 任务翻译无法传入的问题
- 删除 额外依赖 ujson

### Release.1.2 Bata.2
- 修复 单次多词 翻译键值映射问题
- 更改 单次多词 格式

### Release.1.2
- 添加 已安装的整合包翻译支持（失败，Release.1.4 Bata.1完成）

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
- 添加 I18n词典 导出 数据集 功能（[Dict-Mini.json](https://github.com/CFPATools/i18n-dict)）
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
- 修复 单次多词 参考词仅传入一个的问题
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
- 添加 OAuth2.0 格式 API 调用（失败）
- 添加 调用大语言模型额外参数传入
- 添加 已安装的整合包翻译支持（支持 KJS FTBQ BQ 资源文件夹 模组，支持 I18n翻译剔除）
- 添加 FTB任务与BQ任务翻译自动剔除键与滚木
- 添加 单实例 嵌入模型、向量文件、文本文件、向量索引 持久化（多实例无持久化）
- 添加 翻译任务自动分离 "&§x{key}srt" 混合编码后进行翻译
- 添加 自动汉化更新 的 I18n词典 导入翻译缓存功能（[Dict-Mini.json](https://github.com/CFPATools/i18n-dict)）
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
- 修复 通用翻译翻译不了资源包的问题
- 修改 .translatorlang 文件改为 .translang 文件(翻译审查文件)
- 修改 .translang 文件内部格式,现在更清晰明了
- 修改 翻译语言文件不限制文件类型必须相同
- 删除 额外提示词传入方式, 现在固定系统提示词位置
- 删除 分离语言文件函数的mode参数,固定输出语言文件
- 添加 SnbtLib依赖
- 删除 PyHocon依赖

### Release.1.5 Bata.2 (进行中)
- 修复 键与值相同会进行翻译的问题(来自modid:gvcr2翻译请求)
- 修复 单符号会进行翻译的问题(来自modid:gvcr2翻译请求)
