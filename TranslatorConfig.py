from TranslatorLib import dataclass, replace, re

@dataclass
class DefaultConfig:
    LANGUAGE_INPUT = "en_us"
    LANGUAGE_OUTPUT = "zh_cn"
    
    ## 可以使用 LLM0 LLM1 LLM2 多模型
    LLM_API_URL = "" # 请求地址
    LLM_API_KEY = "" # 请求密钥
    LLM_API_KWARGS = {} # 请求额外参数
    LLM_MODEL = "" # 请求模型
    LLM_TOP_K = 30 
    LLM_TOP_P = 0.95
    LLM_TEMP = 0.30
    LLM_PP = 0
    LLM_FP = 0
    LLM_SEED = 42
    LLM_MAX_WORKERS = 24 # 请求最大并发数
    LLM_MIN_COUNT = 0 # 多模型最低启用翻译条目数
    LLM_TIER_INTERLEAVE = False # 多模型混合分布请求API
    LLM_TIER_DYNAMIC = False
    LLM_TIER_CASCADE = False # 级联衰减开关
    LLM_TIER_CASCADE_RATIO = 0.7 # 级联衰减比例
    LLM_TIER_MULTI_OVERLAP = False # 特定比例开关
    LLM_TIER_MULTI_WEIGHT = 1.0 # 特定比例权重 LLM0 LLM1 LLM2 这样
    LLM_TIER_OVERLAP = False # 双重分配开关
    LLM_TIER_OVERLAP_RATIO = [0.3, 0.7] # 双重分配比例
    LLM_CONTEXTS = False
    LLM_MAX_RETRY = 8
    LLM_TIMEOUT = 300
    LLM_CONN_TIMEOUT = 3
    LLM_RETRY_TIME = 5
    LLM_RETRY_COEF = 1.2
    LLM_TOKEN_USAGE = 0 # 翻译实例.Config.LLM_TOKEN_USAGE 获取使用了多少Token

    EMB_API_URL = ""
    EMB_API_KEY = ""
    EMB_MODEL = "nomic-ai/nomic-embed-text-v1.5" # string: 嵌入模型/HuggingFace仓库名
    EMB_MODEL_ACC_MODE = "bfloat16" # string: None, ONNX, float64, float32, float16, bfloat16
    EMB_MODEL_DEVICE = "cuda:0"
    EMB_MAX_TOKENS = 2048
    EMB_TOKENSTOTEXT_RATIO = 3.0
    EMB_MAX_WORKERS = 24
    EMB_MAX_RETRY = 8
    EMB_TIMEOUT = 90
    EMB_CONN_TIMEOUT = 3
    EMB_RETRY_TIME = 5
    EMB_RETRY_COEF = 1.2
    
    RERANKER_API_URL = ""
    RERANKER_API_KEY = ""
    RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
    RERANKER_MODEL_DEVICE = "cuda:0"
    RERANKER_INSTRUCT = "Which Chinese translation best matches the meaning of the English source? Consider terminology accuracy and completeness."
    RERANKER_MAX_WORKERS = 24
    RERANKER_MAX_RETRY = 8
    RERANKER_TIMEOUT = 300
    RERANKER_CONN_TIMEOUT = 3
    RERANKER_RETRY_TIME = 5
    RERANKER_RETRY_COEF = 1.2

    VEC_INT_DTYPE = ["Q8_K_X", "Q6_K_X", "Q4_K_X", "Q3_K_X", "Q2_K_X"]
    VEC_FLOAT_DTYPE = ["Float32", "Float16", "Float16_E0M15", "BFloat16", "Float8_E4M3"]
    VEC_FILE_PATH = r"./Vectors"
    VEC_FILE_NAME = "Vectors"
    VEC_QUANTIZATION = "Q6_K_X" # string: VEC_INT_DTYPE VEC_FLOAT_DTYPE 选其中一个
    VEC_QUANTILE = 0.998
    VEC_QUANTIZATION_BLOCK_SIZE = 32 # int: 2的倍数 最小2 最大256 默认32
    
    TRANSLATOR_CACHE_WRITE = True
    TRANSLATOR_CACHE_READ = True
    TRANSLATOR_CACHE_PATH = r"./Translator_Cache"
    TRANSLATOR_CACHE_NAME = "Translator_Cache"
    TRANSLATOR_REFINE_ROUNDS = 0
    TRANSLATOR_BATCH = 1
    TRANSLATOR_ORIGINAL_REFERENCE = False
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
- 列表输入 → 多个译文严格使用 Python list 对象格式
## 示例
### 多文本输入：
['原文A', '原文B', '原文C']
### 多文本输出：
['译文A', '译文B', '译文C']
## 翻译提示
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
## 翻译提示
"""]

    PATH_CACHE = r"./Cache"
    DEBUG_MODE = False
    LOGS_FILE_PATH = r"./Logs"
    LOGS_FILE_NAME = "logs"
    LOGS_GLOBAL = False
    LANG_PATH = r"./Lang"
    LANGUAGE = r"zh_CN"
    TQDM_FPS = 2
    
    QUESTS_READ_MAX_CONCURRENT = 4
    QUESTS_WRITE_MAX_CONCURRENT = 4
    SCRIPT_READ_MAX_CONCURRENT = 4
    SCRIPT_WRITE_MAX_CONCURRENT = 4
    MENU_READ_MAX_CONCURRENT = 4
    MENU_WRITE_MAX_CONCURRENT = 4
    BOOK_READ_MAX_CONCURRENT = 4
    BOOK_WRITE_MAX_CONCURRENT = 4
    DATA_READ_MAX_CONCURRENT = 4
    DATA_WRITE_MAX_CONCURRENT = 4
    LANG_READ_MAX_CONCURRENT = 4
    LANG_WRITE_MAX_CONCURRENT = 4
    DLL_READ_MAX_CONCURRENT = 4
    DLL_WRITE_MAX_CONCURRENT = 4
    SCRIPT_CRT_WRITE_UNICODE = True
    
    MONO_CECIL_DLL_PATH = r"dll"
    MONO_CECIL_DLL_NAME = "Mono.Cecil.dll"
    
    DATA_COMMAND_PATH = r"./DataPack_Command"
    DATA_COMMAND_FILE = "DataPack_Command.txt"
    
    PACK_META_TEMPLATE_TRANSLATE = "{name} {lang} 语言资源包\n制作: {author}, 翻译模型：{model}"
    PACK_META_TEMPLATE_MERGE = "{name} {lang} 语言资源包\n制作: {author}, 工具自动合并"
    PACK_META_TEMPLATE_CASUALTIESUNKNOWN = "{lang} 语言文件\n制作: <color=\"yellow\">{author}</color>, 翻译模型：<color=\"blue\">{model}" # 未知伤亡
    PACK_AUTHOR = ""

    INDEX_TEXT_K = 2
    INDEX_WORD_K = 2
    INDEX_LANG_K = 2
    INDEX_QUESTS_BASIC_WORDS = []
    INDEX_MODE = "HNSWPQ" # RefineFlat IVFSQ IVFPQ IVFFlat HNSWSQ HNSWPQ HNSWFlat FlatL2 FlatIP # 100%召回选RefineFlat 测试选FlatL2,FlatIP 部署选HNSW,IVF
    INDEX_LANG_MODE = "FlatL2"
    INDEX_SQ = "Q8" # string: Q4, Q6, Q8, F16, BF16
    INDEX_PQ_M = 8
    INDEX_NLIST = 100
    INDEX_NLITS = 8
    INDEX_RE_MINMAX = False
    INDEX_RE_MEANSTD = False
    INDEX_RE_QUANTILES = False
    INDEX_RE_OPTIM = False
    INDEX_HNSW_M = 128
    INDEX_HNSW_CONSTRUCTION = 720
    INDEX_HNSW_SEARCH = 480
    INDEX_REFINEFLAT_K_FACTOR = 2.0
    INDEX_IVF_NPROBE = 10
    
    API_TRANSLATOR_CORE_CONFIG_WHITE = {r"^LANGUAGE_INPUT$", r"^LANGUAGE_OUTPUT$", r"^LANGUAGE$"}
    API_TRANSLATOR_CORE_CONFIG_BLACK = {}
    API_TRANSLATOR_CORE_CONFIG_RANGE = {r"^LLM\d+_TEMP$": (0.0, 1.0), r"^TRANSLATOR_BATCH$": (1, 1), r"^INDEX_\w+_K$": (0, 5)}
    
    MMTQM_PATH = "./MMTQM"
    MMTQM_SEED = 42
    MMTQM_COUNT = 20
    MMTQM_TEXT_PROMPT = "原文:{原文} 选项:{选项}"
    MMTQM_USER_PROMPT = "选项语言{LANGUAGE_OUTPUT}（仅输出选项标号）：{text}"
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
- **多段落输入** → 输出格式：Python list，每项为 `译文 <大写选项>`，示例：`["A", "B", "C", "D", "E"]`

## 示例
### 多段落输入：
['原文1\nA:选项A\nB:选项B', '原文2\nA:选项A\nB:选项B']
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
            "presence_penalty": float(cfg.get("presence_penalty", self.LLM_PP)),
            "frequency_penalty": float(cfg.get("frequency_penalty", self.LLM_FP)),
            "seed": int(cfg.get("seed", self.LLM_SEED)),
            "max_retry": int(cfg.get("max_retry", self.LLM_MAX_RETRY)),
            "conn_timeout": float(cfg.get("conn_timeout", self.LLM_CONN_TIMEOUT)),
            "timeout": float(cfg.get("timeout", self.LLM_TIMEOUT)),
            "retry_time": float(cfg.get("retry_time", self.LLM_RETRY_TIME)),
            "retry_coef": float(cfg.get("retry_coef", self.LLM_RETRY_COEF)),
            "max_workers": int(cfg.get("max_workers", self.LLM_MAX_WORKERS)),
            "min_count": int(cfg.get("min_count", self.LLM_MIN_COUNT)),
            "weight": float(cfg.get("weight", self.LLM_TIER_MULTI_WEIGHT)),
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