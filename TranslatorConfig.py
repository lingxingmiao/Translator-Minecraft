from TranslatorLib import (dataclass, replace, re, numpy, random, Path,
                           Locale, Log, Index, Builder, Module, Quantization, File, Network, TranslationCache, VectorCache, TokenCalibratorCache, Translator)

@dataclass
class DefaultConfig:
    LANGUAGE_INPUT = "en_us"
    LANGUAGE_OUTPUT = "zh_cn"
    
    # 可以使用 LLM0 LLM1 LLM2 多模型
    LLM_API_URL                = ""                                       # 请求地址
    LLM_API_KEY                = ""                                       # 请求密钥
    LLM_API_KWARGS             = {}                                       # 请求额外参数
    LLM_MODEL                  = ""                                       # 请求模型
    LLM_MODE                   = "Translator"                             # 模型模式  Translator为翻译模式 Summary为翻译前总结模式
    LLM_TOP_K                  = 30
    LLM_TOP_P                  = 0.95
    LLM_TEMP                   = 0.25
    LLM_RP                     = 1.1
    LLM_PP                     = 0
    LLM_FP                     = 0
    LLM_SEED                   = random.getrandbits(32-1)-1
    LLM_RETRY_SEED_RANDOM      = True                                     # 重试是否再次随机种子 未实装
    LLM_MAX_WORKERS            = 24                                       # 请求最大并发数
    LLM_MIN_COUNT              = 0                                        # 多模型最低启用翻译条目数
    LLM_RPM                    = 0                                        # 每分钟最大请求数(Requests Per Minute), 0=不限制
    LLM_TPM                    = 0                                        # 每分钟最大Token数(Tokens Per Minute), 0=不限制
    LLM_TPM_MODE               = "TokenCalibrator"                        # TPM估算模式 TokenCalibrator为实时学习 Max为1个字符一个Token
    LLM_KEEPALIVE_TIMEOUT      = 20                                       # keep-alive 连接复用超时(秒)，高并发下调小可减少服务端主动断连(ServerDisconnected)
    LLM_TTL_DNS_CACHE          = 300                                      # DNS 解析结果缓存时间(秒)
    LLM_ACTIVE_TIME_START      = ""                                       # 活跃时间开始 (格式 "HH:MM"，如 "00:00"，留空则全天可用)
    LLM_ACTIVE_TIME_END        = ""                                       # 活跃时间结束 (格式 "HH:MM"，如 "08:00"，支持跨天)
    LLM_TIER_DYNAMIC           = False                                    # 动态分配开关
    LLM_MAX_RETRY              = 8
    LLM_TIMEOUT                = 300
    LLM_CONN_TIMEOUT           = 20
    LLM_CONN_REUSE             = False                                    # aiohttp.TCPConnector.force_close参数 服务端会中断改成True
    LLM_RETRY_TIME             = 5
    LLM_RETRY_COEF             = 1.2
    LLM_TOKEN_IN               = 0                                        # 翻译实例.Config.LLM_TOKEN_IN   获取使用了多少Token
    LLM_TOKEN_OUT              = 0                                        # 翻译实例.Config.LLM_TOKEN_OUT  获取使用了多少Token
    LLM_TOKEN_CACHE_HIT        = 0                                        # 翻译实例.Config.LLM_TOKEN_HIT  获取使用了多少Token 非OpenAI标准
    LLM_TOKEN_CACHE_HIT_FIELD  = [["usage", "prompt_cache_hit_tokens"]]   # 上配置获取字段 依次尝试
    LLM_TOKEN_CACHE_MISS       = 0                                        # 翻译实例.Config.LLM_TOKEN_MISS 获取使用了多少Token 非OpenAI标准
    LLM_TOKEN_CACHE_MISS_FIELD = [["usage", "prompt_cache_miss_tokens"]]  # 上配置获取字段 依次尝试

    EMB_API_URL             = ""
    EMB_API_KEY             = ""
    EMB_MODEL               = "BAAI/bge-small-en-v1.5"  # string: 嵌入模型/HuggingFace仓库名
    EMB_REASONING_FRAME     = "FastEmbed"               # string: 嵌入模型推理框架 SentenceTransformer FastEmbed
    EMB_MODEL_ACC_MODE      = "ONNX"                    # string: None, ONNX, OpenVINO, float32, float16, bfloat16
    EMB_MODEL_DEVICE        = "cpu"                     # string: list: cpu 或 cuda:0 ...
    EMB_ENCODE_PROMPT_NAME  = ""                        # string: 预留功能 仅当前程序加载嵌入模型可用
    EMB_PROMPT_NAME         = ["{t}", "{t}"]            # string: 提示词前缀 部分模型需要使用 [文档, 搜索] intfloat/multilingual-e5-large需要["passage: {t}", "query: {t}"]
    EMB_MODEL_NORMALIZE     = True                      # bool: 是否归一化 仅当前程序加载嵌入模型可用
    EMB_LOADER_KWARGS       = {}                        # dict: 加载器传参 仅当前程序加载嵌入模型可用 {"file_name": "<file_name>"} 可输入指定ONNX模型
    EMB_LOADER_MODEL_KWARGS = {}                        # dict: 加载器模型传参 仅当前程序加载嵌入模型可用 {"attn_implementation": "flash_attention_2"} HuggingFace推荐安装FlashAttention2
    EMB_API_KWARGS          = {}
    EMB_MAX_TOKENS          = 512
    EMB_TOKENSTOTEXT_RATIO  = 3.0
    EMB_MAX_WORKERS         = 24
    EMB_KEEPALIVE_TIMEOUT   = 20
    EMB_TTL_DNS_CACHE       = 300
    EMB_MAX_RETRY           = 8
    EMB_TIMEOUT             = 90
    EMB_CONN_TIMEOUT        = 3
    EMB_CONN_REUSE          = False
    EMB_RETRY_TIME          = 5
    EMB_RETRY_COEF          = 1.2
    
    RERANKER_API_URL           = ""
    RERANKER_API_KEY           = ""
    RERANKER_MODEL             = "Qwen/Qwen3-Reranker-0.6B"
    RERANKER_API_KWARGS        = {}
    RERANKER_MODEL_DEVICE      = "cpu"
    RERANKER_INSTRUCT          = "Which Chinese translation best matches the meaning of the English source? Consider terminology accuracy and completeness."
    RERANKER_MAX_WORKERS       = 24
    RERANKER_KEEPALIVE_TIMEOUT = 20
    RERANKER_TTL_DNS_CACHE     = 300
    RERANKER_MAX_RETRY         = 8
    RERANKER_TIMEOUT           = 300
    RERANKER_CONN_TIMEOUT      = 3
    RERANKER_CONN_REUSE        = False
    RERANKER_RETRY_TIME        = 5
    RERANKER_RETRY_COEF        = 1.2

    SESSION_CLEAN_INTERVAL = 30 # 清理无任务会话间隔时间 同时管理LLM EMB RERANKER

    #GSQ_K请启用向量重排来降低重建误差 小于5w向量请使用非GSQ_K VEC_INT_DTYPE叠加向量误差较大
    VEC_INT_DTYPE =   ["Q8_K_M" ,               "Q8_K", "GSQ8_K", "PolarQ8"          #256值 8   比特
                       "Q6_K_M" , "Q6_SVD_LM" , "Q6_K", "GSQ6_K", "PolarQ6",         #64值  6   比特
                       "Q5_K_M" , "Q5_SVD_LM" , "Q5_K", "GSQ5_K", "PolarQ5",         #32值  5   比特
                       "Q4_K_M" , "Q4_SVD_LM" , "Q4_K", "GSQ4_K", "PolarQ4",         #16值  4   比特
                       "Q3_K_M" , "Q3_SVD_LM" , "Q3_K", "GSQ3_K", "PolarQ3",         #8值   3   比特
                       "Q2_K_M" , "Q2_SVD_LM" , "Q2_K", "GSQ2_K", "PolarQ2", "Q2_NF",#4值   2   比特
                       "TQ1_K_M", "TQ1_SVD_LM", "PolarTQ1",                           #3值   1.585比特
                       "Q1_K_M" , "Q1_SVD_LM" , "PolarQ1",                           #2值   1   比特
                       "PQ"     , "OPQ"       , "AVQ",                               #残差量化
                       ]
    VEC_FLOAT_DTYPE = ["Float32",                                                    #32 比特 Float32原生支持
                       "Float16"    , "Float16_Max", "BFloat16"  , "Float16_E0M15",  #16 比特 Float16原生支持 Float16_Max不可当作缩放
                       "Float12_Max",                                                #12 比特                全系        不可当作缩放
                       "Float8_E4M3", "Float8_E0M7", "Float8_Max",                   #8  比特                Float8_Max  不可当作缩放
                       ]
    VEC_FILE_PATH                = r"./Vectors"         # 向量存储路径
    VEC_FILE_NAME                = "Vectors"            # 向量文件名
    VEC_READ_CACHE               = False                # 读取时缓存解码后的向量到内存 应用到上面的两个配置
    VEC_CACHE_PATH               = r"./Vectors"         # 文本→向量缓存路径
    VEC_CACHE_NAME               = "VectorsCache"       # 文本→向量缓存文件名
    VEC_CACHE_SAVE_INTERVAL      = 30.0                 # 向量缓存定时写盘间隔（秒）；并发生成时改为后台节流批量落盘，避免频繁全量 IO
    VEC_CACHE_DECAY_GRACE        = 256                  # 宽限期（轮），此期限内不计算衰减
    VEC_CACHE_DECAY_THRESHOLD    = 0.05                 # 衰减分数阈值，低于此值淘汰
    VEC_CACHE_MAX_SIZE           = 409600               # 硬上限，超限按衰减分数淘汰最低分
    VEC_DIM_CLIP                 = -1                   # 向量生成时维度裁剪 -1不裁切 仅推荐支持俄罗斯套娃表示学习的模型启用
    VEC_PCA_DIM                  = -1                   # PCA降维维度 -1不降维
    VEC_TT_SHAPE                 = []                   # list: TT分解各维大小，空=自动拆分为2^?
    VEC_TT_RANK                  = -1                   # int: TT分解截断秩（越大精度越高，越小压缩比越高）
    VEC_QUANTIZATION             = "GSQ6_K"             # string 或 list(2): 单一格式(如 "GSQ6_K") 或 混合格式(如 ["GSQ2_K", "Float12_Max"]) 自动按维度能量贡献分配高低精度
    VEC_QUANTIZATION_MIX_RATIO   = 0.2                  # float: 混合模式下 高精度格式 覆盖的维度占比 (按每维度能量贡献 top% 选择)
    VEC_QUANTIZATION_MIX_TOPK    = 0                    # int: 混合模式下 每行固定取 |值| 最大的 N 个维度用高精度 (0=禁用, 用MIX_RATIO比例)
    VEC_QUANTIZATION_CLIP        = 0.998                # 分位数裁切 GSQ_K系列不受影响
    VEC_QUANTIZATION_ITRS_SVD    = 50                   # _SVD步数
    VEC_QUANTIZATION_SPL_SVD     = numpy.float32(0.05)  # _SVD采样 float百分比采样 uint采样数量 float1时采样100% uint1时采样1条向量 类型32位
    VEC_QUANTIZATION_ITRS_LM     = 200                  # _LM步数
    VEC_QUANTIZATION_SPL_LM      = numpy.float32(0.05)  # _LM采样  float百分比采样 uint采样数量 float1时采样100% uint1时采样1条向量 类型32位
    VEC_QUANTIZATION_ES_LM       = 1e-6                 # _LM早停 两步之间小于该值退出
    VEC_QUANTIZATION_SVD_LM_ITRS = 2                    # SVD_LM 循环迭代次数 (SVD旋转 ↔ LM码本交替优化)
    VEC_QUANTIZATION_BLOCK_SIZE  = 128                  # int: 2的倍数 最小2 最大256 默认32
    VEC_QUANTIZATION_SCALE_TYPE  = "Float16_E0M15"      # string: VEC_FLOAT_DTYPE 选其中一个
    VEC_QUANTIZATION_PQ_M        = 128                  # int: Product Quantization 子向量数 (必须整除维度)
    VEC_QUANTIZATION_PQ_NBITS    = 8                    # int: Product Quantization 每子向量位数 (码本大小=2^NBITS)
    VEC_QUANTIZATION_OPQ_ITRS    = 25                   # int: Optimized Product Quantization 迭代优化次数
    VEC_QUANTIZATION_AVQ_ETA     = 4.125                # float: AVQ 各向异性比值 h_‖/h_⊥ (论文推荐 4.125)
    
      # 向量重排 仅支持GSQ_K量化与GSQ索引
    VEC_RERANKER                           = True
    VEC_RERANKER_INDEX_RERANKER_BLOCK_SIZE = 128                   # 向量重排块大小(聚类)
    VEC_RERANKER_INDEX_FACTOR              = 8.0                   # 向量搜索乘数
    VEC_RERANKER_INDEX_MODE                = ["Refine", "HNSWPQ"]  # 支持嵌套数组 [类型, 子规格]，例 ["Refine", ["IVFPQ", "IP"]]；叶子: L2 IP；独立: HNSW HNSWSQ HNSWPQ NSGFlat NSGSQ NSGPQ；包装(需子规格): Refine RefineLowDim IVFSQ IVFPQ IVFPQR IVF；包装型省略子规格时默认子索引=IP
    VEC_RERANKER_INDEX_BASE_SQ             = "Q8"                  # string: Q4, Q6, Q8, F16, BF16
    VEC_RERANKER_INDEX_SQ                  = "Q8"
    VEC_RERANKER_INDEX_REFINE_LOW_DIM_DIM  = 64                    # int: 粗排维度
    VEC_RERANKER_INDEX_REFINE_LOW_DIM_MODE = None                  # strint: 降维模式 模型支持Matryoshka Representation Learning请使用MRL 否则使用PCA
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
    VEC_RERANKER_INDEX_IVF_NLIST           = 0.25
    VEC_RERANKER_INDEX_IVF_PQ_M            = 16
    VEC_RERANKER_INDEX_IVFPQR_M_REFINE     = 16                    # IVFPQR 精修级PQ子量化器数 (需整除维度)
    VEC_RERANKER_INDEX_IVFPQR_NBITS_REFINE = 8                     # IVFPQR 精修级每子量化位数
    VEC_RERANKER_INDEX_IVF_RQ              = True
    VEC_RERANKER_INDEX_REFINEFLAT_K_FACTOR = 10.0
    
    TOKEN_CALIBRATOR_CACHE_WRITE      = True                     # 是否写入Token估算器缓存
    TOKEN_CALIBRATOR_CACHE_READ       = True                     # 是否读取Token估算器缓存
    TOKEN_CALIBRATOR_CACHE_PATH       = r"./Token_Calibrator"    # Token估算器缓存路径
    TOKEN_CALIBRATOR_CACHE_NAME       = "TokenCalibrator"        # Token估算器缓存文件名
    TOKEN_CALIBRATOR_CACHE_SAVE_INTERVAL = 60.0                  # Token估算器缓存定时写盘间隔（秒）
    
    TRANSLATOR_CACHE_WRITE               = True
    TRANSLATOR_CACHE_READ                = True
    TRANSLATOR_CACHE_PATH                = r"./Translator_Cache"
    TRANSLATOR_CACHE_NAME                = "Translator_Cache"
    TRANSLATOR_CACHE_SAVE_INTERVAL       = 45.0                                   # 翻译缓存定时写盘间隔（秒）
    TRANSLATOR_REFINE_ROUNDS             = 0                                      # 翻译精炼次数
    TRANSLATOR_BATCH                     = 5                                      # 单次请求翻译文本数
    TRANSLATOR_BATCH_RETRY               = 1                                      # 批量翻译重试次数 超过退回单条翻译
    TRANSLATOR_CONTEXTS_MODE             = "space"                                # string: token:最少上下文消耗 space:上下文空间最近 vector:语义空间最近
    TRANSLATOR_CONTEXTS                  = False                                  # bool int: 翻译上下文 False:无 int值:数量(对) True:无上限
    TRANSLATOR_ORIGINAL_REFERENCE        = False                                  # 文本对照 Input:UV False:紫外线 True:紫外线(UV)
    TRANSLATOR_ORIGINAL_REFERENCE_FORMAT = "{t}({o})"                             # 文本对照格式 o为原文 t为译文
    TRANSLATOR_MODPACK_MOD_CONCURRENT    = 8                                      # 翻译整合包时翻译模组并发数
    TRANSLATOR_SUMMARY                   = False                                  # bool: 翻译前总结 用来提升语义一致性
    TRANSLATOR_SUMMARY_MAX_TEXT          = 81920                                  # int: 最大输入大小
    TRANSLATOR_SUMMARY_SYSTEM_PROMPT     = "你是Minecraft模组内容分析器,通读待翻译文本后用{lang}语言输出一段概述(≤200字),概括主题/玩法/核心术语,不翻译具体条目。"
    TRANSLATOR_SUMMARY_USER_PROMPT       = "请总结以下模组文本的内容与核心术语:\n{text}"
    TRANSLATOR_SUMMARY_TEXT              = "\nVIII以下为翻译前总结内容,请参考内容后翻译"
    TRANSLATOR_USER_PROMPT               = "翻译为{lang}语言(仅输出翻译内容):{text}"
    TRANSLATOR_SYSTEM_PROMPT             = """你是minecraft翻译器,将文本准确一致地翻译成{lang}语言
##翻译规则
I返回译文不加解释与废话(如“这是译文”“如下翻译”).
II返回译文必须与段落及格式相同.
III不译内容(如专名、键等),保留原文.
IV遇&或§或%则留后一位不译.
VHTML和Markdown语法翻译可读文本内容,不改标签或符号.
VI<rag-input>为参考内容,不译
VII<context>为上下文内容,不译
"""

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
    TQDM_DIFF            = False       # 详细任务进度条 仅支持翻译请求 实验性
    
    #CONCURRENT太高Windows会报错
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
    
    MONO_CECIL_DLL_PATH = r"./dll"
    MONO_CECIL_DLL_NAME = "Mono.Cecil.dll"
    
    DATA_COMMAND_PATH = r"./DataPack_Command"
    DATA_COMMAND_FILE = "DataPack_Command.txt"
    
    PACK_META_TEMPLATE_TRANSLATE         = "{name} {lang} 语言资源包\n制作: {author}, 翻译模型：{model}"
    PACK_META_TEMPLATE_MERGE             = "{name} {lang} 语言资源包\n制作: {author}, 工具自动合并"
    PACK_META_TEMPLATE_CASUALTIESUNKNOWN = "{lang} 语言文件\n制作: <color=\"yellow\">{author}</color>, 翻译模型：<color=\"blue\">{model}"  # 未知伤亡
    PACK_AUTHOR                          = ""

    INDEX_TEXT_K                  = 2                                            # int: 文本索引数
    INDEX_WORD_K                  = 2                                            # int: 单词索引数
    INDEX_LANG_K                  = 2                                            # int: 索引ID索引数
    INDEX_TEXT_RANGE              = 8                                            # int: 文本索引文本范围 len(文本)超过±该值就不会加入参考文本
    INDEX_WORD_RANGE              = 4                                            # int: 单词索引文本范围 len(文本)超过±该值就不会加入参考文本
    INDEX_LANG_RANGE              = 8                                            # int: 索引ID索引文本范围 len(文本)超过±该值就不会加入参考文本
    INDEX_QUESTS_BASIC_WORDS      = []
    INDEX_MODE                    = ["Refine", "NSGSQ"]                          # strint list: 支持嵌套数组 [类型, 子规格]，例 ["Refine", "IVFPQ", "IP"]；叶子: L2 IP；独立: HNSW HNSWSQ HNSWPQ NSGFlat NSGSQ NSGPQ, GSQFast(下一个索引不能是Faiss)；包装(需子规格): Refine RefineLowDim IVFSQ IVFPQ IVFPQR IVF 默认：IP
    INDEX_LANG_MODE               = "IP"                                         # strint list: 索引ID(整合包索引)索引模式
    INDEX_CONTEXTS_MODE           = "IP"                                         # strint list: 翻译上下文(TRANSLATOR_CONTEXTS)索引模式                                 
    INDEX_SQ                      = "Q8"                                         # string: faiss: Q4, Q6, Q8, F16, BF16 indexgsq: GSQ2 GSQ3 GSQ4 GSQ5 GSQ6 GSQ8 (GSQ系列必须选)
    INDEX_REFINE_LOW_DIM_DIM      = 64                                           # int: 粗排维度
    INDEX_REFINE_LOW_DIM_MODE     = None                                         # strint none: 降维模式 模型支持Matryoshka Representation Learning请使用MRL 否则使用PCA 还有None RefineLowDim支持MRL和PCA Refine仅支持PCA
    INDEX_SAMPLING                = numpy.float32(0.05)                          # float百分比采样 uint采样数量 float1时采样100% uint1时采样1条向量 类型32位
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
    INDEX_NSG_R                   = 64                                           # NSG 图出度 R
    INDEX_NSG_SEARCH              = 240                                          # NSG 检索束宽 search_L (建议>=R)
    INDEX_NSG_PQ_M                = 16                                           # NSGPQ 子量化器数 (需整除向量维度)
    INDEX_NSG_NBITS               = 8                                            # NSGPQ 每子量化位数
    INDEX_IVF_NLITS               = 8
    INDEX_IVF_NLIST               = 0.25                                         # 聚类激活比例
    INDEX_IVF_PQ_M                = 16
    INDEX_IVF_RQ                  = True
    INDEX_IVFPQR_M_REFINE         = 16                                           # IVFPQR 精修级PQ子量化器数 (需整除维度)
    INDEX_IVFPQR_NBITS_REFINE     = 8                                            # IVFPQR 精修级每子量化位数
    INDEX_REFINEFLAT_K_FACTOR     = 10.0
    INDEX_GSQ_RERANKER_BLOCK_SIZE = 128                                          # 向量重排块大小(聚类)
    INDEX_GSQ_RERANKER_FACTOR     = 8                                            # 向量重排检索的向量倍数(向量越多向量重排块缩放越高重排倍数越高耗时越久)
    INDEX_GSQ_BLOCK_SIZE          = 128                                          # 量化块大小
    INDEX_GSQ_PCA_DIM             = -1                                           # PCA降维维度 -1不降维
    INDEX_CPU_COUNT               = numpy.float32(0.8)                           #float百分比线程 uint线程数 float1时线程100% uint1时1线程 类型32位 仅Faiss可用
    INDEX_CONFIG                  = ["INDEX_MODE", "INDEX_SQ",
                    "INDEX_RE_MINMAX", "INDEX_RE_MEANSTD", "INDEX_RE_QUANTILES", "INDEX_RE_OPTIM",
                    "INDEX_HNSW_M", "INDEX_HNSW_CONSTRUCTION", "INDEX_HNSW_SEARCH", "INDEX_HNSW_NBITS", "INDEX_HNSW_PQ_M",
                    "INDEX_NSG_R", "INDEX_NSG_SEARCH", "INDEX_NSG_PQ_M", "INDEX_NSG_NBITS",
                    "INDEX_IVF_NLITS", "INDEX_IVF_PQ_M", "INDEX_IVF_RQ", "INDEX_IVFPQR_M_REFINE", "INDEX_IVFPQR_NBITS_REFINE", "INDEX_REFINEFLAT_K_FACTOR",
                    "INDEX_GSQ_RERANKER_FACTOR", "INDEX_REFINE_LOW_DIM_DIM", "INDEX_REFINE_LOW_DIM_MODE",
                    "INDEX_GSQ_RERANKER_BLOCK_SIZE", "INDEX_GSQ_BLOCK_SIZE", "INDEX_GSQ_PCA_DIM"]
    INDEX_CONFIG_NEST  = {"Refine", "IVF", "IVFSQ", "IVFPQ"}
    INDEX_CONFIG_TRAIN = {"HNSW", "HNSWSQ", "HNSWPQ", "IVF", "IVFSQ", "IVFPQ"}
    
    PATH_CONFIG = ["TOKEN_CALIBRATOR_CACHE_PATH", "TRANSLATOR_CACHE_PATH", "DATA_COMMAND_PATH", "MONO_CECIL_DLL_PATH", "PATH_CACHE", "LOGS_FILE_PATH", "LANG_PATH"]
    LLM_POAT_CONFIG = [[402, 429,], [400, 401, 422, 500, 503]]
    
    API_TRANSLATOR_CORE_CONFIG_WHITE = {r"^LANGUAGE_INPUT$", r"^LANGUAGE_OUTPUT$", r"^LANGUAGE$"}
    API_TRANSLATOR_CORE_CONFIG_BLACK = {}
    API_TRANSLATOR_CORE_CONFIG_RANGE = {r"^LLM\d+_TEMP$": (0.0, 1.0), r"^TRANSLATOR_BATCH$": (1, 1), r"^INDEX_\w+_K$": (0, 5)}
DEFAULT_CONFIG = DefaultConfig()
class RuntimeConfig(DefaultConfig): # 这个括号是继承 RuntimeConfig包含DefaultConfig的功能或值
    class _层级(dict):
        # ↓同时支持 index.键 和 index["键"] 访问方式
        def __getattr__(Self, 名称):
            try: return Self[名称]
            except KeyError: raise AttributeError(f"'{type(Self).__name__}' object has no attribute '{名称}'")

    def __init__(Self, **kwargs):
        # ↓初始化唯一 Config 副本（DefaultConfig），直接访问 Self.Config.xxx
        object.__setattr__(Self, "Config", replace(DEFAULT_CONFIG))
        object.__setattr__(Self, "_层级注册表", {})
        
        # ↓分离层级参数与其他参数 格式: LLM{层级号}_{字段} => {层级号: {字段: 值}, ...}
        层级参数, 其他参数 = {}, {}
        for 键, 值 in kwargs.items():
            if re.match(r'^LLM\d+_', 键): 层级参数[键] = 值
            else: 其他参数[键] = 值
        
        # ↓应用其他参数到配置 network特殊处理存入NETWORK变量
        for 键, 值 in 其他参数.items():
            if 键 == "network":
                object.__setattr__(Self, "NETWORK", 值)
            elif hasattr(Self.Config, 键): setattr(Self.Config, 键, 值)
            else: raise AttributeError(f"Unknown config key: {键}")
        
        # ↓解析层级参数 按层级号分组 格式: {层级号: {字段: 值}, ...}
        匹配模式 = re.compile(r'^LLM(\d+)_(.+)$')
        层级分组 = {}
        for 键, 值 in 层级参数.items():
            匹配结果 = 匹配模式.match(键)
            if 匹配结果:
                层级号, 字段 = int(匹配结果.group(1)), 匹配结果.group(2).lower()
                层级分组.setdefault(层级号, {})[字段] = 值
        
        # ↓没有层级注册一个 防止楼下的for不干活
        if not 层级分组:
            层级分组[0] = {}
        
        # ↓构建层级注册表 未指定字段回退到全局默认值（通过 Self.Config.xxx）
        for 层级号, 层级配置 in 层级分组.items():
            Self._层级注册表[层级号] = {
                "id"               : str(  层级号),
                "url"              : str(  层级配置.get("api_url"           , Self.Config.LLM_API_URL          )),
                "key"              : str(  层级配置.get("api_key"           , Self.Config.LLM_API_KEY          )),
                "model"            : str(  层级配置.get("model"             , Self.Config.LLM_MODEL            )),
                "mode"             : str(  层级配置.get("mode"              , Self.Config.LLM_MODE             )),
                "api_kwargs"       : dict( 层级配置.get("api_kwargs"        , Self.Config.LLM_API_KWARGS       )),
                "temperature"      : float(层级配置.get("temperature"       , Self.Config.LLM_TEMP             )),
                "top_p"            : float(层级配置.get("top_p"             , Self.Config.LLM_TOP_P            )),
                "top_k"            : int(  层级配置.get("top_k"             , Self.Config.LLM_TOP_K            )),
                "repeat_penalty"   : float(层级配置.get("repeat_penalty"    , Self.Config.LLM_RP               )),
                "presence_penalty" : float(层级配置.get("presence_penalty"  , Self.Config.LLM_PP               )),
                "frequency_penalty": float(层级配置.get("frequency_penalty" , Self.Config.LLM_FP               )),
                "seed"             : int(  层级配置.get("seed"              , random.getrandbits(32-1)-1       )),
                "retry_seed_random": bool( 层级配置.get("retry_seed"        , Self.Config.LLM_RETRY_SEED_RANDOM)),
                "max_retry"        : int(  层级配置.get("max_retry"         , Self.Config.LLM_MAX_RETRY        )),
                "conn_timeout"     : float(层级配置.get("conn_timeout"      , Self.Config.LLM_CONN_TIMEOUT     )),
                "conn_reuse"       : bool( 层级配置.get("conn_reuse"        , Self.Config.LLM_CONN_REUSE       )),
                "timeout"          : float(层级配置.get("timeout"           , Self.Config.LLM_TIMEOUT          )),
                "retry_time"       : float(层级配置.get("retry_time"        , Self.Config.LLM_RETRY_TIME       )),
                "retry_coef"       : float(层级配置.get("retry_coef"        , Self.Config.LLM_RETRY_COEF       )),
                "max_workers"      : int(  层级配置.get("max_workers"       , Self.Config.LLM_MAX_WORKERS      )),
                "keepalive_timeout": float(层级配置.get("keepalive_timeout" , Self.Config.LLM_KEEPALIVE_TIMEOUT)),
                "ttl_dns_cache"    : int(  层级配置.get("ttl_dns_cache"     , Self.Config.LLM_TTL_DNS_CACHE    )),
                "min_count"        : int(  层级配置.get("min_count"         , Self.Config.LLM_MIN_COUNT        )),
                "rpm"              : int(  层级配置.get("rpm"               , Self.Config.LLM_RPM              )),
                "tpm"              : int(  层级配置.get("tpm"               , Self.Config.LLM_TPM              )),
                "tpm_mode"         : str(  层级配置.get("tpm_mode"          , Self.Config.LLM_TPM_MODE         )),
                "active_time_start": str(  层级配置.get("active_time_start" , Self.Config.LLM_ACTIVE_TIME_START)),
                "active_time_end"  : str(  层级配置.get("active_time_end"   , Self.Config.LLM_ACTIVE_TIME_END  )),
            }
    
    # get层级 食用方法: for inedx in 返回内容: print(index[键])
    def get(Self):
        层级列表 = list(Self._层级注册表.values())
        层级列表.sort(key=lambda 层级: 层级["min_count"])
        return 层级列表

    def __getattr__(Self, 名称):
        raise AttributeError(f"'{type(Self).__name__}' object has no attribute '{名称}'")

    def __setattr__(Self, 名称, 值):
        if 名称 in ('Config', '_层级注册表'):
            object.__setattr__(Self, 名称, 值)
        elif hasattr(Self.Config, 名称):
            setattr(Self.Config, 名称, 值)
        else:
            raise AttributeError(f"Unknown config key: {名称}")
        
class Config:
    def __init__(Self, Config: dict):
        Self.ConfigDict = Config.copy()
        Config = RuntimeConfig(**(Config or {}))
        Self.Config: DefaultConfig = Config.Config
        Self.Manager: RuntimeConfig = Config
        for index in Self.Config.PATH_CONFIG:
            Path(getattr(Self.Config, index)).mkdir(parents=True, exist_ok=True)
        Self.Locale: Locale = Locale(Self)
        Self.RichTqdm = Self.Locale.RichTqdm # 标准进度条
        Self.TqdmTqdm = Self.Locale.TqdmTqdm # 美化进度条 默认
        Self.DiffTqdm = Self.Locale.DiffTqdm # “扩散模型风格”进度条
        Self.Lang = Self.Locale.Lang
        Self.Log: Log = Log(Self)
        Self.日志 = Self.Log.写入日志
        Self.CacheTranslator: TranslationCache = TranslationCache(Self)
        Self.CacheVector: VectorCache = VectorCache(Self)
        Self.CacheTokenCalibrator: TokenCalibratorCache = TokenCalibratorCache(Self)
        Self.Index: Index = Index(Self)
        Self.Module: Module = Module(Self)
        Self.Network: Network = Network(Self) # 网络管理器核心 并发限制在此
        Self.Builder: Builder = Builder(Self)
        Self.Quantization: Quantization = Quantization(Self)
        Self.File: File = File(Self)
        Self.Translator = None
    def get_translator(Self):
        if Self.Translator is None: Self.Translator = Translator(Self)
        return Self.Translator
    def 过滤用户配置(Self, 用户配置: dict) -> dict:
        规则 = {
            "allow_patterns": Self.Config.API_TRANSLATOR_CORE_CONFIG_WHITE,
            "deny_patterns": Self.Config.API_TRANSLATOR_CORE_CONFIG_BLACK,
            "range_checks": Self.Config.API_TRANSLATOR_CORE_CONFIG_RANGE
        }
        过滤后配置 = {}
        for 键, 值 in 用户配置.items():
            if any(re.match(模式, 键) for 模式 in 规则["deny_patterns"]):
                continue
            if not any(re.match(模式, 键) for 模式 in 规则["allow_patterns"]):
                continue
            for 模式, (最小值, 最大值) in 规则["range_checks"].items():
                if re.match(模式, 键) and isinstance(值, (int, float)):
                    值 = max(最小值, min(值, 最大值))
                    break
            过滤后配置[键] = 值
        return 过滤后配置
    def get_config_temporary(Self, 用户配置: dict = None):
        临时配置 = Config(Self.ConfigDict | (Self.过滤用户配置(用户配置) or {}))
        临时配置.Network = Self.Network
        临时配置.Builder.Network = Self.Network
        临时配置.CacheVector = Self.CacheVector
        临时配置.Builder.CacheVector = Self.CacheVector
        临时配置.CacheTranslator = Self.CacheTranslator
        临时配置.CacheTokenCalibrator = Self.CacheTokenCalibrator
        return 临时配置
    def get_translator_temporary(Self, 用户配置: dict = None):
        临时配置 = Self.get_config_temporary(用户配置)
        return 临时配置.get_translator()