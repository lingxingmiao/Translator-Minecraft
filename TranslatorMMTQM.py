from TranslatorLib import random, json, Path
from TranslatorConfig import RuntimeConfig
from TranslatorLocale import Locale
from TranslatorModule import Module
from TranslatorCore import Translator

class MMTQM:
    def __init__(Self, Config: dict = None):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Config["INDEX_K"] = Config.get("INDEX_K", 0)
        Config["TRANSLATOR_CACHE_WRITE"] = False
        Config["TRANSLATOR_CACHE_READ"] = False
        Config["TRANSLATOR_ORIGINAL_REFERENCE"] = False
        Config["TRANSLATOR_BATCH"] = 1
        Config["TRANSLATOR_SYSTEM_PROMPT"] = Self.Config.MMTQM_SYSTEM_PROMPT
        Config["TRANSLATOR_USER_PROMPT"] = Self.Config.MMTQM_USER_PROMPT
        Path(Self.Config.MMTQM_PATH).mkdir(parents=True, exist_ok=True)
        Self.Module = Module(Config=Config)
        Self.Locale = Locale(Config=Config)
        Self.Translator = Translator(Config=Config)
        Self.Lang = Self.Locale.Lang
        Self.日志 = Self.Module.写入日志
        Self.tqdm = Self.Locale.Tqdm
        random.seed(Self.Config.MMTQM_SEED)
        
    def 开始(Self):
        Self.日志("log.mmtqm.start", info_level=0)
        指定文件 = None
        请求内容 = []
        得分 = 0
        所有MMTQM评测文件路径 = [index for index in Path(Self.Config.MMTQM_PATH).rglob("*.json")]
        for index in 所有MMTQM评测文件路径:
            if f"{Self.Config.LANGUAGE_INPUT}-{Self.Config.LANGUAGE_OUTPUT}.json".lower() == index.name.lower():
                指定文件 = index
                break
        if 指定文件:
            for _ in Self.tqdm(range(Self.Config.MMTQM_COUNT), desc="tqdm.mmtqm"):
                with open(指定文件, "r", encoding="utf-8") as f:
                    评测文件 = json.load(f)
                评测文件长度 = len(评测文件)
                for index0, index1 in 评测文件.items():
                    选项列表 = []
                    for index2, index3 in index1.items():
                        选项列表.append([index2, index3])
                    random.shuffle(选项列表)
                    请求内容.append([[None], Self.Config.MMTQM_TEXT_PROMPT.format(原文=index0, 选项='\n'.join(f'{chr(65 + i)}:{s[0]}' for i, s in enumerate(选项列表))), {chr(65 + i): s for i, s in enumerate(选项列表)}])
                请求返回内容 = Self.Translator.翻译语言列表(请求内容)
                print(请求返回内容)
                for _, 选择选项, 选项字典 in 请求返回内容:
                    得分 += 选项字典[选择选项][1] / 评测文件长度
            得分 == 得分 / Self.Config.MMTQM_COUNT
        else:
            raise FileNotFoundError(Self.Lang("log.mmtqm.file.not"))
        Self.日志("log.mmtqm.score", score=得分, info_level=0)
            
测试 = True
if __name__ == "__main__" and 测试:
    参数 = {
        "LLM_API_URL": "http://127.0.0.1:25564/v1/chat/completions",
        "LLM_MODEL": "Qwen3.5-9B",
        "TRANSLATOR_BATCH": 1,
        "LLM_CONTEXTS": 3,
        "EMB_API_URL": "http://127.0.0.1:25564/v1/embeddings",
        "EMB_MODEL": "text-embedding-nomic-embed-text-v1.5",
        "LLM_MAX_WORKERS": 4,
        "TRANSLATOR_ORIGINAL_REFERENCE": False,
        "LANGUAGE": "zh_LLA",
        "TRANSLATOR_CACHE_NAME": "Translator_Cache",
        "VEC_FILE_NAME": "Vectors",
        "EMB_MAX_WORKERS": 2,
        "INDEX_K": 0
    }
    评测 = MMTQM(参数)
    评测.开始()