from TranslatorLib import dataclass, replace, re, numpy

@dataclass
class DefaultConfig:
    LANGUAGE_INPUT = "en_us"
    LANGUAGE_OUTPUT = "zh_cn"
    
    ## 可以使用 LLM0 LLM1 LLM2 多模型
    LLM_API_URL            = ""          # 请求地址
    LLM_API_KEY            = ""          # 请求密钥
    LLM_API_KWARGS         = {}          # 请求额外参数
    LLM_MODEL              = ""          # 请求模型
    LLM_TOP_K              = 30
    LLM_TOP_P              = 0.95
    LLM_TEMP               = 0.30
    LLM_RP                 = 1.1
    LLM_PP                 = 0
    LLM_FP                 = 0
    LLM_SEED               = 42
    LLM_MAX_WORKERS        = 24          # 请求最大并发数
    LLM_MIN_COUNT          = 0           # 多模型最低启用翻译条目数
    LLM_RPM                = 0           # 每分钟最大请求数(Requests Per Minute), 0=不限制
    LLM_TPM                = 0           # 每分钟最大Token数(Tokens Per Minute), 0=不限制
    LLM_KEEPALIVE_TIMEOUT  = 20          # keep-alive 连接复用超时(秒)，高并发下调小可减少服务端主动断连(ServerDisconnected)
    LLM_TTL_DNS_CACHE      = 300         # DNS 解析结果缓存时间(秒)
    LLM_ACTIVE_TIME_START  = ""          # 活跃时间开始 (格式 "HH:MM"，如 "00:00"，留空则全天可用)
    LLM_ACTIVE_TIME_END    = ""          # 活跃时间结束 (格式 "HH:MM"，如 "08:00"，支持跨天)
    LLM_TIER_INTERLEAVE    = False       # 多模型混合分布请求API
    LLM_TIER_DYNAMIC       = False       # 动态分配开关
    LLM_TIER_CASCADE       = False       # 级联衰减开关
    LLM_TIER_CASCADE_RATIO = 0.7         # 级联衰减比例
    LLM_TIER_MULTI_OVERLAP = False       # 特定比例开关
    LLM_TIER_MULTI_WEIGHT  = 1.0         # 特定比例权重 LLM0 LLM1 LLM2 这样
    LLM_TIER_OVERLAP       = False       # 双重分配开关
    LLM_TIER_OVERLAP_RATIO = [0.3, 0.7]  # 双重分配比例
    LLM_CONTEXTS           = False
    LLM_MAX_RETRY          = 8
    LLM_TIMEOUT            = 300
    LLM_CONN_TIMEOUT       = 20
    LLM_RETRY_TIME         = 5
    LLM_RETRY_COEF         = 1.2
    LLM_TOKEN_IN           = 0           # 翻译实例.Config.LLM_TOKEN_IN   获取使用了多少Token
    LLM_TOKEN_OUT          = 0           # 翻译实例.Config.LLM_TOKEN_OUT  获取使用了多少Token
    LLM_TOKEN_CACHE_HIT    = 0           # 翻译实例.Config.LLM_TOKEN_HIT  获取使用了多少Token 非OpenAI标准 获取usage.prompt_cache_hit_tokens
    LLM_TOKEN_CACHE_MISS   = 0           # 翻译实例.Config.LLM_TOKEN_MISS 获取使用了多少Token 非OpenAI标准 获取usage.prompt_cache_miss_tokens

    EMB_API_URL             = ""
    EMB_API_KEY             = ""
    EMB_MODEL               = "BAAI/bge-small-en-v1.5"  # string: 嵌入模型/HuggingFace仓库名
    EMB_REASONING_FRAME     = "FastEmbed"               # string: 嵌入模型推理框架 SentenceTransformer FastEmbed
    EMB_MODEL_ACC_MODE      = "ONNX"                    # string: None, ONNX, OpenVINO, float32, float16, bfloat16
    EMB_MODEL_DEVICE        = "cpu"                     # string: cpu 或 cuda:0 ...
    EMB_ENCODE_PROMPT_NAME  = ""                        # string: 预留功能 仅当前程序加载嵌入模型可用
    EMB_MODEL_NORMALIZE     = True                      # bool: 是否归一化 仅当前程序加载嵌入模型可用
    EMB_LOADER_KWARGS       = {}                        # dict: 加载器传参 仅当前程序加载嵌入模型可用
    EMB_LOADER_MODEL_KWARGS = {}                        # dict: 加载器模型传参 仅当前程序加载嵌入模型可用 {"attn_implementation": "flash_attention_2"} HuggingFace推荐安装FlashAttention2
    EMB_MAX_TOKENS          = 512
    EMB_TOKENSTOTEXT_RATIO  = 3.0
    EMB_MAX_WORKERS         = 24
    EMB_MAX_RETRY           = 8
    EMB_TIMEOUT             = 90
    EMB_CONN_TIMEOUT        = 3
    EMB_RETRY_TIME          = 5
    EMB_RETRY_COEF          = 1.2
    
    RERANKER_API_URL      = ""
    RERANKER_API_KEY      = ""
    RERANKER_MODEL        = "Qwen/Qwen3-Reranker-0.6B"
    RERANKER_MODEL_DEVICE = "cpu"
    RERANKER_INSTRUCT     = "Which Chinese translation best matches the meaning of the English source? Consider terminology accuracy and completeness."
    RERANKER_MAX_WORKERS  = 24
    RERANKER_MAX_RETRY    = 8
    RERANKER_TIMEOUT      = 300
    RERANKER_CONN_TIMEOUT = 3
    RERANKER_RETRY_TIME   = 5
    RERANKER_RETRY_COEF   = 1.2

    #GSQ_K请启用向量重排来降低重建误差 小于5w向量请使用非GSQ_K VEC_INT_DTYPE叠加向量误差较大
    VEC_INT_DTYPE =   ["Q8_K_M" ,               "Q8_K", "GSQ8_K",                       #256值 8   比特
                       "Q6_K_M" , "Q6_SVD_LM" , "Q6_K", "GSQ6_K",                       #64值  6   比特
                       "Q5_K_M" , "Q5_SVD_LM" , "Q5_K", "GSQ5_K",                       #32值  5   比特
                       "Q4_K_M" , "Q4_SVD_LM" , "Q4_K", "GSQ4_K",                       #16值  4   比特
                       "Q3_K_M" , "Q3_SVD_LM" , "Q3_K", "GSQ3_K",                       #8值   3   比特
                       "Q2_K_M" , "Q2_SVD_LM" , "Q2_K", "GSQ2_K", "Q2_NF",              #4值   2   比特
                       "TQ1_K_M", "TQ1_SVD_LM",                                         #3值   1.6 比特
                       "Q1_K_M" , "Q1_SVD_LM",                                          #1值   1   比特
                       "PQ"     , "OPQ"                                                 #残差量化
                       ]
    VEC_FLOAT_DTYPE = ["Float32",                                                       #32 比特 Float32原生支持
                       "Float16"       , "BFloat16",    "Float16_E0M15", "Float16_Max", #16 比特 Float16原生支持
                       "Float12_Max",                                                   #12 比特
                       "Float8_E4M3"   , "Float8_E0M7", "Float8_Max"                    #8  比特
                       ]
    VEC_FILE_PATH                = r"./Vectors"         # 向量存储路径
    VEC_FILE_NAME                = "Vectors"            # 向量文件名
    VEC_READ_CACHE               = False                # 读取时缓存解码后的向量到内存 应用到上面的两个配置
    VEC_CACHE_PATH               = r"./Vectors"         # 文本→向量缓存路径
    VEC_CACHE_NAME               = "VectorsCache"       # 文本→向量缓存文件名
    VEC_CACHE_SAVE_INTERVAL      = 30.0                 # 向量缓存定时写盘间隔（秒）；并发生成时改为后台节流批量落盘，避免频繁全量 IO
    VEC_CACHE_DECAY_GRACE        = 256                  # 宽限期（轮），此期限内不计算衰减
    VEC_CACHE_DECAY_THRESHOLD    = 0.05                 # 衰减分数阈值，低于此值淘汰
    VEC_CACHE_MAX_SIZE           = 409600                # 硬上限，超限按衰减分数淘汰最低分
    VEC_DIM_CLIP                 = -1                   # 向量生成时维度裁剪 -1不裁切 仅推荐支持俄罗斯套娃表示学习的模型启用
    VEC_PCA_DIM                  = -1                   # PCA降维维度 -1不降维
    VEC_QUANTIZATION             = "GSQ6_K"             # string: VEC_INT_DTYPE VEC_FLOAT_DTYPE 选其中一个
    VEC_QUANTIZATION_CLIP        = 0.998                # 分位数裁切 GSQ_K系列不受影响
    VEC_QUANTIZATION_ITRS_SVD    = 50                   # _SVD步数
    VEC_QUANTIZATION_SPL_SVD     = numpy.float32(0.05)  # _SVD采样 float百分比采样 uint采样数量 float1时采样100% uint1时采样1条向量 类型32位
    VEC_QUANTIZATION_ITRS_LM     = 200                  # _LM步数
    VEC_QUANTIZATION_SPL_LM      = numpy.float32(0.05)  # _LM采样 float百分比采样 uint采样数量 float1时采样100% uint1时采样1条向量 类型32位
    VEC_QUANTIZATION_ES_LM       = 1e-6                 # _LM早停 两步之间小于该值退出
    VEC_QUANTIZATION_SVD_LM_ITRS = 2                    # SVD_LM 循环迭代次数 (SVD旋转 ↔ LM码本交替优化)
    VEC_QUANTIZATION_BLOCK_SIZE  = 128                  # int: 2的倍数 最小2 最大256 默认32
    VEC_QUANTIZATION_SCALE_TYPE  = "Float16_E0M15"      # string: VEC_FLOAT_DTYPE 选其中一个
    VEC_QUANTIZATION_PQ_M        = 128                  # int: Product Quantization 子向量数 (必须整除维度)
    VEC_QUANTIZATION_PQ_NBITS    = 8                    # int: Product Quantization 每子向量位数 (码本大小=2^NBITS)
    VEC_QUANTIZATION_OPQ_ITRS    = 25                   # int: Optimized Product Quantization 迭代优化次数
    
    # 向量重排 仅支持GSQ_K量化与GSQ索引 不适合向量数小于5w
    VEC_RERANKER                           = True
    VEC_RERANKER_INDEX_RERANKER_BLOCK_SIZE = 128                   # 向量重排块大小(聚类)
    VEC_RERANKER_INDEX_FACTOR              = 8.0                   # 向量搜索乘数
    VEC_RERANKER_INDEX_MODE                = ["Refine", "HNSWPQ"]  # 支持嵌套数组 [类型, 子规格]，例 ["Refine", ["IVFPQ", "IP"]]；叶子: L2 IP；独立: HNSW HNSWSQ HNSWPQ NSGFlat NSGSQ NSGPQ；包装(需子规格): Refine IVFSQ IVFPQ IVFPQR IVF；包装型省略子规格时默认子索引=IP
    VEC_RERANKER_INDEX_BASE_SQ             = "Q8"                  # string: Q4, Q6, Q8, F16, BF16
    VEC_RERANKER_INDEX_SQ                  = "Q8"
    VEC_RERANKER_INDEX_SAMPLING            = numpy.float32(0.05)   # float百分比采样 uint采样数量 float1时采样100% uint1时采样1条向量 类型32位
    VEC_RERANKER_INDEX_SAMPLING_MIN        = 1
    VEC_RERANKER_INDEX_RE_MINMAX           = False
    VEC_RERANKER_INDEX_RE_MEANSTD          = False
    VEC_RERANKER_INDEX_RE_QUANTILES        = False
    VEC_RERANKER_INDEX_RE_OPTIM            = False
    VEC_RERANKER_INDEX_HNSW_M              = 32
    VEC_RERANKER_INDEX_HNSW_CONSTRUCTION   = 640
    VEC_RERANKER_INDEX_HNSW_SEARCH         = 240
    VEC_RERANKER_INDEX_HNSW_NBITS          = 8
    VEC_RERANKER_INDEX_HNSW_PQ_M           = 16
    VEC_RERANKER_INDEX_NSG_R               = 64                    # NSG 图出度 R
    VEC_RERANKER_INDEX_NSG_SEARCH          = 240                   # NSG 检索束宽 search_L (建议>=R)
    VEC_RERANKER_INDEX_NSG_PQ_M            = 16                    # NSGPQ 子量化器数 (需整除向量维度)
    VEC_RERANKER_INDEX_NSG_NBITS           = 8                     # NSGPQ 每子量化位数
    VEC_RERANKER_INDEX_IVF_NLITS           = 8
    VEC_RERANKER_INDEX_IVF_PQ_M            = 16
    VEC_RERANKER_INDEX_IVFPQR_M_REFINE     = 16                    # IVFPQR 精修级PQ子量化器数 (需整除维度)
    VEC_RERANKER_INDEX_IVFPQR_NBITS_REFINE = 8                     # IVFPQR 精修级每子量化位数
    VEC_RERANKER_INDEX_IVF_RQ              = True
    VEC_RERANKER_INDEX_REFINEFLAT_K_FACTOR = 6.0
    
    TRANSLATOR_CACHE_WRITE            = True
    TRANSLATOR_CACHE_READ             = True
    TRANSLATOR_CACHE_PATH             = r"./Translator_Cache"
    TRANSLATOR_CACHE_NAME             = "Translator_Cache"
    TRANSLATOR_CACHE_SAVE_INTERVAL    = 45.0                     # 翻译缓存定时写盘间隔（秒）；改为内存缓存+后台节流落盘，避免并发翻译频繁全量 IO
    TRANSLATOR_REFINE_ROUNDS          = 0                      # 翻译精炼次数 Tokens以乘数+上下文Tokens增长
    TRANSLATOR_BATCH                  = 5                      # 单次请求翻译文本数
    TRANSLATOR_BATCH_RETRY            = 2                      # TRANSLATOR_BATCH不为1重试次数 如果还是失败则退回TRANSLATOR_BATCH=1
    TRANSLATOR_ORIGINAL_REFERENCE     = False                  # 文本对照 Input:UV False:紫外线 True:紫外线(UV)
    TRANSLATOR_MODPACK_MOD_CONCURRENT = 8                      # 翻译整合包时翻译模组并发数
    TRANSLATOR_USER_PROMPT = ["翻译为{LANGUAGE_OUTPUT}(仅输出翻译内容):{text}", "翻译为{LANGUAGE_OUTPUT}(总计{count}个词条,仅输出翻译内容):{text}"]
    TRANSLATOR_SYSTEM_PROMPT = ["""
你是一位专业的 Minecraft 游戏翻译器，需要流畅准确一致地将文本翻译成 {LANGUAGE_OUTPUT} 语言。
## 翻译规则
1. 仅输出翻译内容，不包含解释或额外内容（如“这是译文：”或“如下所示：”）
2. 返回的译文必须与原文保持完全相同的段落数和格式
3. 对于不应翻译的内容（如专有名词，按键操作等），保留原文。
4. 单个符号需要翻译（遇到&或§或%需要保留后一位符号不做翻译）
5. 保留所有HTML标签（如`<br>``<span>``<a href="">`等）和Markdown语法，仅翻译标签/语法内的可读文本内容，不修改标签结构或语法符号本身
## 输出格式：
- 列表输入 → 多个译文严格使用 JSON list 对象格式
## 示例
### 多文本输入：
["原文A", "原文B", "原文C"]
### 多文本输出：
["译文A", "译文B", "译文C"]
## 下文为翻译提示
""", """
你是一位专业的 Minecraft 游戏翻译器，需要流畅准确一致地将文本翻译成 {LANGUAGE_OUTPUT} 语言。
## 翻译规则
1. 仅输出翻译内容，不包含解释或额外内容（如“这是译文：”或“如下所示：”）
2. 返回的译文必须与原文保持完全相同的段落数和格式
3. 对于不应翻译的内容（如专有名词，按键操作等），保留原文。
4. 单个符号需要翻译（遇到&或§或%需要保留后一位符号不做翻译）
5. 保留所有HTML标签（如`<br>``<span>``<a href="">`等）和Markdown语法，仅翻译标签/语法内的可读文本内容，不修改标签结构或语法符号本身
## 输出格式：
- 译文输入 → 译文直接输出译文（无分隔符，无额外文本）
## 示例
### 单文本输入：
原文
### 单文本输出：
译文
## 下文为翻译提示
"""]

    PATH_CACHE           = r"./Cache"
    CACHE_CHECK_INTERVAL = 24          # 缓存清理检测间隔（小时）
    CACHE_TTL_HOURS      = 48          # 缓存保留时间（小时），超时删除
    DEBUG_MODE           = False
    LOGS_FILE_PATH       = r"./Logs"
    LOGS_FILE_NAME       = "logs"
    LOGS_GLOBAL          = False
    LOGS_FLUSH_INTERVAL  = 3           # 日志批量刷盘间隔
    LANG_PATH            = r"./Lang"
    LANGUAGE             = r"zh_CN"
    TQDM_FPS             = 24
    
    #CONCURRENT太高会报错
    QUESTS_READ_MAX_CONCURRENT  = 4
    QUESTS_WRITE_MAX_CONCURRENT = 4
    SCRIPT_READ_MAX_CONCURRENT  = 4
    SCRIPT_WRITE_MAX_CONCURRENT = 4
    MENU_READ_MAX_CONCURRENT    = 4
    MENU_WRITE_MAX_CONCURRENT   = 4
    BOOK_READ_MAX_CONCURRENT    = 4
    BOOK_WRITE_MAX_CONCURRENT   = 4
    DATA_READ_MAX_CONCURRENT    = 4
    DATA_WRITE_MAX_CONCURRENT   = 4
    LANG_READ_MAX_CONCURRENT    = 4
    LANG_WRITE_MAX_CONCURRENT   = 4
    DLL_READ_MAX_CONCURRENT     = 4
    DLL_WRITE_MAX_CONCURRENT    = 4
    SCRIPT_CRT_WRITE_UNICODE    = True
    
    MONO_CECIL_DLL_PATH = r"dll"
    MONO_CECIL_DLL_NAME = "Mono.Cecil.dll"
    
    DATA_COMMAND_PATH = r"./DataPack_Command"
    DATA_COMMAND_FILE = "DataPack_Command.txt"
    
    PACK_META_TEMPLATE_TRANSLATE         = "{name} {lang} 语言资源包\n制作: {author}, 翻译模型：{model}"
    PACK_META_TEMPLATE_MERGE             = "{name} {lang} 语言资源包\n制作: {author}, 工具自动合并"
    PACK_META_TEMPLATE_CASUALTIESUNKNOWN = "{lang} 语言文件\n制作: <color=\"yellow\">{author}</color>, 翻译模型：<color=\"blue\">{model}"  # 未知伤亡
    PACK_AUTHOR                          = ""

    INDEX_TEXT_K                  = 2
    INDEX_WORD_K                  = 2
    INDEX_LANG_K                  = 2
    INDEX_QUESTS_BASIC_WORDS      = []
    INDEX_MODE                    = ["Refine", "HNSWPQ"] # list: 支持嵌套数组 [类型, 子规格]，例 ["Refine", ["IVFPQ", "IP"]]；叶子: L2 IP；独立: HNSW HNSWSQ HNSWPQ NSGFlat NSGSQ NSGPQ；包装(需子规格): Refine IVFSQ IVFPQ IVFPQR IVF；自研低内存: GSQFast；包装型省略子规格时默认子索引=IP
    INDEX_LANG_MODE               = "IP"                 # string: L2 IP
    INDEX_BASE_SQ                 = "Q8"                 # string: faiss: Q4, Q6, Q8, F16, BF16 indexgsq: GSQ2 GSQ3 GSQ4 GSQ6 GSQ8 (GSQ系列必须选)
    INDEX_SQ                      = "Q8"
    INDEX_SAMPLING                = numpy.float32(0.05)  # float百分比采样 uint采样数量 float1时采样100% uint1时采样1条向量 类型32位
    INDEX_SAMPLING_MIN            = 1
    INDEX_RE_MINMAX               = False
    INDEX_RE_MEANSTD              = False
    INDEX_RE_QUANTILES            = False
    INDEX_RE_OPTIM                = False
    INDEX_HNSW_M                  = 32
    INDEX_HNSW_CONSTRUCTION       = 640
    INDEX_HNSW_SEARCH             = 240
    INDEX_HNSW_NBITS              = 8
    INDEX_HNSW_PQ_M               = 16
    INDEX_NSG_R                   = 64                   # NSG 图出度 R
    INDEX_NSG_SEARCH              = 240                  # NSG 检索束宽 search_L (建议>=R)
    INDEX_NSG_PQ_M                = 16                   # NSGPQ 子量化器数 (需整除向量维度)
    INDEX_NSG_NBITS               = 8                    # NSGPQ 每子量化位数
    INDEX_IVF_NLITS               = 8
    INDEX_IVF_PQ_M                = 16
    INDEX_IVF_RQ                  = True
    INDEX_IVFPQR_M_REFINE         = 16                   # IVFPQR 精修级PQ子量化器数 (需整除维度)
    INDEX_IVFPQR_NBITS_REFINE     = 8                    # IVFPQR 精修级每子量化位数
    INDEX_REFINEFLAT_K_FACTOR     = 6.0
    INDEX_GSQ_RERANKER_BLOCK_SIZE = 128                  # 向量重排块大小(聚类)
    INDEX_GSQ_RERANKER_FACTOR     = 8                    # 向量重排检索的向量倍数(向量越多向量重排块缩放越高重排倍数越高耗时越久)
    INDEX_GSQ_BLOCK_SIZE          = 128                  # 量化块大小
    INDEX_GSQ_PCA_DIM             = -1                   # PCA降维维度 -1不降维
    INDEX_CPU_COUNT               = numpy.float32(0.8)   #float百分比线程 uint线程数 float1时线程100% uint1时1线程 类型32位 仅Faiss可用
    INDEX_CONFIG = ["INDEX_MODE", "INDEX_BASE_SQ", "INDEX_SQ",
                    "INDEX_RE_MINMAX", "INDEX_RE_MEANSTD", "INDEX_RE_QUANTILES", "INDEX_RE_OPTIM",
                    "INDEX_HNSW_M", "INDEX_HNSW_CONSTRUCTION", "INDEX_HNSW_SEARCH", "INDEX_HNSW_NBITS", "INDEX_HNSW_PQ_M",
                    "INDEX_NSG_R", "INDEX_NSG_SEARCH", "INDEX_NSG_PQ_M", "INDEX_NSG_NBITS",
                    "INDEX_IVF_NLITS", "INDEX_IVF_PQ_M", "INDEX_IVF_RQ", "INDEX_IVFPQR_M_REFINE", "INDEX_IVFPQR_NBITS_REFINE", "INDEX_REFINEFLAT_K_FACTOR",
                    "INDEX_GSQ_RERANKER_BLOCK_SIZE", "INDEX_GSQ_RERANKER_FACTOR", "INDEX_GSQ_BLOCK_SIZE", "INDEX_GSQ_PCA_DIM"]
    INDEX_CONFIG_NEST = {"Refine", "IVF", "IVFSQ", "IVFPQ"}
    INDEX_CONFIG_TRAIN = {"HNSW", "HNSWSQ", "HNSWPQ", "IVF", "IVFSQ", "IVFPQ"}
    
    API_TRANSLATOR_CORE_CONFIG_WHITE = {r"^LANGUAGE_INPUT$", r"^LANGUAGE_OUTPUT$", r"^LANGUAGE$"}
    API_TRANSLATOR_CORE_CONFIG_BLACK = {}
    API_TRANSLATOR_CORE_CONFIG_RANGE = {r"^LLM\d+_TEMP$": (0.0, 1.0), r"^TRANSLATOR_BATCH$": (1, 1), r"^INDEX_\w+_K$": (0, 5)}
    
    MMTQM_PATH          = "./MMTQM"
    MMTQM_SEED          = 42
    MMTQM_COUNT         = 20
    MMTQM_TEXT_PROMPT   = "原文:{原文} 选项:{选项}"
    MMTQM_USER_PROMPT   = "选项语言{LANGUAGE_OUTPUT}（仅输出选项标号）：{text}"
    MMTQM_SYSTEM_PROMPT = """
你是一位专业的 Minecraft 游戏本地化专家，需要从给定的翻译选项中选择一个最符合 {LANGUAGE_OUTPUT} 语言习惯的最优译文。

## 选择规则
1. 仅输出选中答案标号，严禁包含任何解释、序号、选项标签或其他额外文本。
2. 评估优先级：格式完整性 > Minecraft 专有名词准确度 > 上下文语境契合度 > {LANGUAGE_OUTPUT} 语言流畅度。
3. 格式严格对齐：选出的译文必须与原文保持完全相同的段落数和排版结构。
4. 特殊内容保留：若原文包含按键操作、变量、代码占位符或控制符号（如 &、§、% 及其后紧跟的字符），必须确保选中选项完整保留这些内容，未丢失、错位或篡改。
5. 若所有选项均有瑕疵，请挑选相对最优的一项，切勿自行修改、拼接或重新生成。
6. 保留所有HTML标签（如`<br>``<span>``<a href="">`等）和Markdown语法（如`**粗体**``*斜体*``[链接](url)``# 标题`等），仅翻译标签/语法内的可读文本内容，不修改标签结构或语法符号本身

## 输出格式（含答案标号）：
- **单段落输入** → 输出格式：`选中译文 <大写字母>`，示例：`A`
- **多段落输入** → 输出格式：JSON list，每项为 `译文 <大写选项>`，示例：`["A", "B", "C", "D", "E"]`

## 示例
### 多段落输入：
["原文1\nA:选项A\nB:选项B", "原文2\nA:选项A\nB:选项B"]
### 多段落输出：
["<选项1>", "<选项2>"]

### 单段落输入：
原文\nA:选项A\nB:选项B
### 单段落输出：
<选项>

## 评估提示
- 始终以 Minecraft 官方译名与游戏语境为基准
- 严格核对特殊符号（&/§/%）与变量占位符的完整性
- 答案标号必须与所选选项顺序严格对应（选项1→A/a，选项2→B/b，依此类推）
- 仅输出最终结果，零冗余、零推理过程
"""
DEFAULT_CONFIG = DefaultConfig()
class RuntimeConfig:
    def __init__(self, **kwargs):
        object.__setattr__(self, '_配置', replace(DEFAULT_CONFIG))
        object.__setattr__(self, '_tier_registry', {})
        llm_params = {k: v for k, v in kwargs.items() if re.match(r'^LLM\d+_', k)}
        other_params = {k: v for k, v in kwargs.items() if k not in llm_params}
        for k, v in other_params.items():
            if hasattr(self._配置, k):
                setattr(self._配置, k, v)
            else:
                raise AttributeError(f"未知配置项：{k}")
        self._auto_register_tiers(llm_params)

    def _auto_register_tiers(self, llm_params: dict):
        pattern = re.compile(r'^LLM(\d+)_(.+)$')
        groups = {}
        for k, v in llm_params.items():
            m = pattern.match(k)
            if m:
                num = int(m.group(1))
                field = m.group(2).lower()
                groups.setdefault(num, {})[field] = v
        for num, cfg in groups.items():
            self._tier_registry[num] = self._build_tier(num, cfg)

    def _build_tier(self, num: int, cfg: dict) -> dict:
        """用全局默认值填充一个完整的 tier 配置"""
        return {
            "id": num,
            "url": cfg.get("api_url", self.LLM_API_URL),
            "key": cfg.get("api_key", self.LLM_API_KEY),
            "model": cfg.get("model", self.LLM_MODEL),
            "api_kwargs": cfg.get("api_kwargs", self.LLM_API_KWARGS),
            "temperature": float(cfg.get("temperature", self.LLM_TEMP)),
            "top_p": float(cfg.get("top_p", self.LLM_TOP_P)),
            "top_k": int(cfg.get("top_k", self.LLM_TOP_K)),
            "repeat_penalty": float(cfg.get("repetition_penalty", self.LLM_RP)),
            "presence_penalty": float(cfg.get("presence_penalty", self.LLM_PP)),
            "frequency_penalty": float(cfg.get("frequency_penalty", self.LLM_FP)),
            "seed": int(cfg.get("seed", self.LLM_SEED)),
            "max_retry": int(cfg.get("max_retry", self.LLM_MAX_RETRY)),
            "conn_timeout": float(cfg.get("conn_timeout", self.LLM_CONN_TIMEOUT)),
            "timeout": float(cfg.get("timeout", self.LLM_TIMEOUT)),
            "retry_time": float(cfg.get("retry_time", self.LLM_RETRY_TIME)),
            "retry_coef": float(cfg.get("retry_coef", self.LLM_RETRY_COEF)),
            "max_workers": int(cfg.get("max_workers", self.LLM_MAX_WORKERS)),
            "keepalive_timeout": float(cfg.get("keepalive_timeout", self.LLM_KEEPALIVE_TIMEOUT)),
            "ttl_dns_cache": int(cfg.get("ttl_dns_cache", self.LLM_TTL_DNS_CACHE)),
            "min_count": int(cfg.get("min_count", self.LLM_MIN_COUNT)),
            "weight": float(cfg.get("weight", self.LLM_TIER_MULTI_WEIGHT)),
            "rpm": int(cfg.get("rpm", self.LLM_RPM)),
            "tpm": int(cfg.get("tpm", self.LLM_TPM)),
            "active_time_start": cfg.get("active_time_start", self.LLM_ACTIVE_TIME_START),
            "active_time_end": cfg.get("active_time_end", self.LLM_ACTIVE_TIME_END),
        }

    def get_active_tiers(self):
        """返回按照 min_count 排序的 tier 列表"""
        tiers = list(self._tier_registry.values())
        tiers.sort(key=lambda t: t["min_count"])
        return tiers

    def add_llm_endpoint(self, number: int, **kwargs):
        """动态添加或更新一个 LLM 端点"""
        self._tier_registry[number] = self._build_tier(number, kwargs)

    def remove_llm_endpoint(self, number: int):
        self._tier_registry.pop(number, None)

    def __getattr__(self, 名称):
        try:
            配置 = object.__getattribute__(self, '_配置')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{名称}'")
        
        return getattr(配置, 名称)

    def __setattr__(self, 名称, 值):
        if 名称 == '_配置':
            object.__setattr__(self, 名称, 值)
        else:
            try:
                配置 = object.__getattribute__(self, '_配置')
                setattr(配置, 名称, 值)
            except AttributeError:
                object.__setattr__(self, 名称, 值)