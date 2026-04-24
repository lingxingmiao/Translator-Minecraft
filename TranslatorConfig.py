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
    LLM_PP = 0
    LLM_FP = 0
    LLM_SEED = 42
    LLM_CONTEXTS = False
    LLM_CONTEXTS_LENGTH = 65536
    LLM_MAX_WORKERS = 24
    LLM_MAX_BATCH = 1
    LLM_MAX_RETRY = 20
    LLM_RETRY_INTERVAL = 5
    LLM_ORIGINAL_REFERENCE = True
    LLM_USER_PROMPT = """
翻译为{LANGUAGE_OUTPUT}（仅输出翻译内容）：{text}
"""
    LLM_SYSTEM_PROMPT = """
你是一位专业的 Minecraft 游戏 {LANGUAGE_OUTPUT} 语母语翻译，需要流畅地将文本翻译成 {LANGUAGE_OUTPUT}。
## 翻译规则
1. 仅输出翻译内容，不包含解释或额外内容（如“这是译文：”或“如下所示：”）
2. 返回的译文必须与原文保持完全相同的段落数和格式
3. 如果文本包含 HTML 标签，请在保持流畅性的同时考虑标签在译文中应放置的位置
4. 对于不应翻译的内容（如专有名词，按键操作等），保留原文。
5. 单个符号需要翻译（遇到&或§或%需要保留后一位符号不做翻译）
## 输出格式：
- **单段落输入** → 直接输出译文（无分隔符，无额外文本）
- **多段落输入** → 多个译文严格使用 Python list 对象格式
## 示例
### 多段落输入：
['原文A', '原文B', '原文C']
### 多段落输出：
['译文A', '译文B', '译文C']
### 单段落输入：
单段落内容
### 单段落输出：
直接翻译，无分隔符
## 翻译提示
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
    VEC_QUANTIZATION = "Q4_K_X" # string: VEC_INT_DTYPE 选其中一个
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
    INDEX_QUESTS_BASIC_WORDS = []
    INDEX_ITEM_HIQ = True
    INDEX_QUESTS_HIQ = True
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