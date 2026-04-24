from TranslatorLib import json, Path, Dict, tqdm
from TranslatorConfig import RuntimeConfig

class Locale:
    def __init__(Self, Config):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.语言对象: Dict[str, dict] = {}
        Path(Self.Config.LANG_PATH).mkdir(parents=True, exist_ok=True)
        
    def LoadLanguage(Self):
        目标语言 = Self.Config.LANGUAGE
        if 目标语言 in Self.语言对象 and "zh_CN" in Self.语言对象:
            return Self.语言对象[目标语言], Self.语言对象["zh_CN"]
        中文语言文件 = {}
        语言文件 = {}
        def _load_lang_file(lang_name: str):
            file_path = Path(f"{Self.Config.LANG_PATH}/{lang_name}.json").resolve()
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                raise
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON format error in language file {file_path}: {e}")
        try:
            中文语言文件 = _load_lang_file("zh_CN")
            Self.语言对象["zh_CN"] = 中文语言文件
            语言文件 = _load_lang_file(目标语言)
            Self.语言对象[目标语言] = 语言文件
            return 语言文件, 中文语言文件
        except FileNotFoundError:
            Self.语言对象[目标语言] = 中文语言文件
            return 中文语言文件, 中文语言文件
            
    def Lang(Self, key: str = None, **kwargs) -> str:
        语言文件, 中文语言文件 = Self.LoadLanguage()
        指定文本 = 语言文件.get(key, key)
        中文文本 = 中文语言文件.get(key, key)
        if kwargs:
            try:
                return 指定文本.format(**kwargs)
            except KeyError:
                return 中文文本.format(**kwargs)
        return 指定文本
        
    def Tqdm(Self, iterable=None, desc=None, **kwargs):
        return tqdm(iterable=iterable, desc=Self.Lang(desc), **kwargs)