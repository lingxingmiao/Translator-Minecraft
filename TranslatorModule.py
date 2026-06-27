from TranslatorLib import (uuid, pickle, Path, threading, SimpleNamespace, re,
                           RuntimeConfig, Log, Locale, Index, Builder)

class Module:
    def __init__(Self, Config: dict = None):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.Locale = Locale(Config=Config)
        Self.日志 = Log(Config=Config).写入日志
        Self.Lang = Self.Locale.Lang
        Self.tqdm = Self.Locale.Tqdm
        Self.线程锁 = threading.Lock()
        Self.Index = Index(Config)
        Self.Builder = Builder(Config)
        Self.正则表达式预编译 = SimpleNamespace()
        Self.正则表达式预编译.ZS模式 = {
            "tooltip": re.compile(r'\.addTooltip\("((?:[^"\\]|\\.)*)"\)'),
            "displayName": re.compile(r'\.displayName\s*=\s*"((?:[^"\\]|\\.)*)"'),
        }
        Self.正则表达式预编译.ZS替换模式 = {
            "tooltip": re.compile(r'(\.addTooltip\(")((?:[^"\\]|\\.)*)("\))'),
            "displayName": re.compile(r'(\.displayName\s*=\s*")((?:[^"\\]|\\.)*)(")'),
        }
    def __enter__(Self):
        return Self
    def 翻译缓存(Self, 输入列表: list = None):
        with Self.线程锁:
            文本文件 = []
            try:
                with open(f"{Self.Config.TRANSLATOR_CACHE_PATH}/{Self.Config.TRANSLATOR_CACHE_NAME}.pkl", "rb+") as f:
                    文本文件 = pickle.load(f)
            except Exception: pass
            if 输入列表:
                文本文件.extend(输入列表)
                文本文件 = list({item[0]: item for item in 文本文件}.values())
                with open(f"{Self.Config.TRANSLATOR_CACHE_PATH}/{Self.Config.TRANSLATOR_CACHE_NAME}.pkl", "wb+") as f:
                    pickle.dump(文本文件, f)
            return {item[0]: item[1] for item in 文本文件}
    def 输出路径处理(Self, path: str):
        if not path:
            path = f"./{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}"
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path).resolve()
    def 列表去重(Self, 列表: list):
        return list(dict.fromkeys(列表))
    def 过滤键文本(Self, 条目):
        return 条目[1] and not (re.match(r'^[a-z0-9._-]+$', 条目[1]) and '.' in 条目[1])
    def 过滤DLL文本(Self, 条目):
        文本 = 条目[1]
        if not 文本 or not 文本.strip():
            return False
        if '_' in 文本: 
            return False    
        if re.match(r'^[A-Z][a-zA-Z0-9]+$', 文本): 
            return False
        if re.match(r'^[a-z]+[A-Z][a-zA-Z0-9]*$', 文本):
            return False
        if 文本.islower() and ' ' not in 文本 and len(文本) < 15: 
            return False 
        if '.' in 文本 and re.match(r'^[a-z0-9.\-]+$', 文本):
            return False
        if 'http://' in 文本 or 'https://' in 文本:
            return False
        if '/' in 文本 and not 文本.startswith('http'): 
            return False
        if 文本.startswith(("org.", "com.", "unityengine.", "system.")): 
            return False
        if re.search(r'\{[0-9]+\}', 文本): 
            return False
        if '("' in 文本 or '")' in 文本:
            return False
        if '!!!' in 文本:
            return False
        if re.search(r'\.[A-Z]', 文本):
            return False
        危险词 = ('initialized', 'postfix', 'prefix', 'warning:', 'error:', 'exception', 'debug', 'patcher', 'log.', 'steamworks', 'failed', 'error unknown')
        if any(kw in 文本.lower() for kw in 危险词): 
            return False
        按键词 = ('left alt', 'right alt', 'left ctrl', 'right ctrl', 'left shift', 'right shift', 'mouse')
        if 文本.lower() in 按键词:
            return False
        if 文本.lower().endswith('.dll'):
            return False
        if '.' in 文本 and 文本.strip().endswith(':'):
            return False
        if '::' in 文本:
            return False
        if re.match(r'^\[\d{4}-\d{2}-\d{2}', 文本) or '-->' in 文本:
            return False
        if re.match(r'^[A-Za-z0-9]+(\.[A-Za-z0-9]+)+$', 文本):
            return False
        if 文本.isupper() and ' ' in 文本 and any(w in 文本 for w in ['ERROR', 'UNKNOWN', 'FAILED', 'EXCEPTION', 'WARNING']):
            return False
        return True
