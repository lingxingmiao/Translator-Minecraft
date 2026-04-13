from TranslatorLib import dataclass, replace

@dataclass
class DefaultConfig:
    LANGUAGE_INPUT = "en_us"
    LANGUAGE_OUTPUT = "zh_cn"
    
    LLM_API_URL = ""
    LLM_API_KEY = ""
    LLM_API_KWARGS = {}
    LLM_MODEL = ""
    LLM_TOP_K = 30
    LLM_TOP_P = 0.95
    LLM_TEMP = 0.30
    LLM_PROMPT_LOCATION = "system"
    LLM_CONTEXTS = False
    LLM_CONTEXTS_LENGTH = 65536
    LLM_MAX_WORKERS = 24
    LLM_MAX_BATCH = 1
    LLM_MAX_RETRY = 20
    LLM_RETRY_INTERVAL = 5
    LLM_ORIGINAL_REFERENCE = True
    LLM_SYSTEM_PROMPTEX1 = f"""
【以下是翻译任务提示，严禁输出】
"""
    LLM_SYSTEM_PROMPTEX2 = f"""
- 输入为列表，输出必须严格符合Python标准列表格式，如['翻译1', '翻译2']
"""
    LLM_SYSTEM_PROMPT = f"""/no_thinking
# 【任务内容】
- 仅输出原文翻译内容，不得包含解释、参考、键、额外内容(如“以下是翻译：”或“翻译如下：”等)
- 单个符号需要翻译(遇到&或§或%需要保留后面一个符号)
- 注意不需要翻译上文,也不要额外解释
- 返回的译文数量与输出格式必须一致
- 翻译为{LANGUAGE_OUTPUT}语言
- 翻译领域为Minecraft游戏
- 输入遇到{LLM_SYSTEM_PROMPTEX1}请不要输出这段内容与后面的内容
- 如果遇到 nomifactory.quest.normal.db.12.desc nomifactory.quest.normal.db.884.title 这种格式的键文本(可能带有大中小括号), 请保留原文
"""

    EMB_API_URL = ""
    EMB_API_KEY = ""
    EMB_MODEL = "nomic-ai/nomic-embed-text-v1.5" # string: 嵌入模型/HuggingFace仓库名
    EMB_MODEL_ACC_MODE = "bfloat16" # string: None, ONNX, float64, float32, float16, bfloat16
    EMB_MODEL_DEVICE = "cuda:0"
    EMB_MAX_TOKENS = 2048
    EMB_TOKENSTOTEXT_RATIO = 3.0
    EMB_MAX_WORKERS = 24
    EMB_MAX_RETRY = 20
    EMB_RETRY_INTERVAL = 5

    VEC_INT_DTYPE = ["Q8_K_X", "Q6_K_X", "Q4_K_X", "Q3_K_X", "Q2_K_X"]
    VEC_FLOAT_DTYPE = ["Float32", "Float16", "Float16_E0M15", "BFloat16", "Float8_E4M3"]
    VEC_FILE_PATH = r"./Vectors"
    VEC_FILE_NAME = "Vectors"
    VEC_QUANTIZATION = "Q4_K_X" # string: VEC_INT_DTYPE 与 VEC_FLOAT_DTYPE 选中其中一个
    VEC_QUANTILE = 0.998
    VEC_QUANTIZATION_BLOCK_SIZE = 32 # int: 2的倍数 最小2 最大256 默认32
    
    TRANSLATOR_CACHE_WRITE = True
    TRANSLATOR_CACHE_READ = True
    TRANSLATOR_CACHE_PATH = r"./Translator_Cache"
    TRANSLATOR_CACHE_NAME = "Translator_Cache"

    PATH_CACHE = r"./Cache"
    DEBUG_MODE = False
    LOGS_FILE_PATH = r"./Logs"
    LOGS_FILE_NAME = "logs"
    LOGS_GLOBAL = True
    LANG_PATH = r"./Lang"
    LANGUAGE = r"zh_CN"
    
    QUESTS_FTB_READ_MAX_CONCURRENT = 4
    QUESTS_FTB_WRITE_MAX_CONCURRENT = 4
    QUESTS_BQ_READ_MAX_CONCURRENT = 4
    QUESTS_BQ_WRITE_MAX_CONCURRENT = 4

    INDEX_K = 3
    INDEX_MODE = "RefineFlat" # HNSWSQ RefineFlat
    INDEX_SQ = "Q6" # string: Q4, Q6, Q8, F16, BF16
    INDEX_HNSW_M = 128
    INDEX_HNSW_CONSTRUCTION = 720
    INDEX_HNSW_SEARCH = 480
    INDEX_REFINEFLAT_K_FACTOR = 2.0
DEFAULT_CONFIG = DefaultConfig()
class RuntimeConfig:
    def __init__(self, **参数):
        object.__setattr__(self, '_配置', replace(DEFAULT_CONFIG))
        
        for 键, 值 in 参数.items():
            if hasattr(self._配置, 键):
                setattr(self._配置, 键, 值)
            else:
                raise AttributeError(f"未知配置项：{键}")

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