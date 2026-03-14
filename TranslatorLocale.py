from TranslatorLib import json, Path, Dict, tqdm
from TranslatorConfig import RuntimeConfig

class Locale:
    def __init__(Self, Config):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.语言对象: Dict[str, dict] = {}
        Path(Self.Config.LANG_PATH).mkdir(parents=True, exist_ok=True)
    def LoadLanguage(Self) -> dict:
        if Self.Config.LANGUAGE in Self.语言对象:
            return Self.语言对象[Self.Config.LANGUAGE]
        本地化文件路径 = str(Path(f"{Self.Config.LANG_PATH}/{Self.Config.LANGUAGE}.json").resolve())
        try:
            with open(本地化文件路径, "r", encoding="utf-8") as f:
                语言文件 = json.load(f)
            Self.语言对象[Self.Config.LANGUAGE] = 语言文件
            return 语言文件
        except json.JSONDecodeError as e:
            raise ValueError(F"There is a JSON format error in the language file {本地化文件路径} of folder a: {e}")
    def Lang(Self, key: str = None, **kwargs) -> str:
        语言文件 = Self.LoadLanguage()
        文本 = 语言文件.get(key, key)
        if kwargs:
            try:
                文本 = 文本.format(**kwargs)
            except KeyError:
                pass
        return 文本
    def Tqdm(Self, iterable=None, desc=None, **kwargs):
        return tqdm(iterable=iterable, desc=Self.Lang(desc), **kwargs)