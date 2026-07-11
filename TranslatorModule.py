from TranslatorLib import (uuid, pickle, Path, threading, time, shutil, SimpleNamespace, re, np,
                           RuntimeConfig, Log, Locale, Index, Builder, TranslatorPersistence)

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
        Self.清理过期缓存()
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
        # 内存缓存模式：读写都在内存中进行，写盘由后台定时线程负责，避免并发生成时频繁全量 IO
        if 输入列表:
            TranslatorPersistence.更新翻译缓存(输入列表)
        return TranslatorPersistence.查询翻译缓存()
    def 输出路径处理(Self, path: str):
        if not path:
            path = f"./{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}"
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path).resolve()
    def 清理过期缓存(Self):
        try:
            缓存根 = Path(Self.Config.PATH_CACHE)
            if not 缓存根.is_dir(): return
            标记文件 = 缓存根 / ".last_cleanup"
            检查间隔 = Self.Config.CACHE_CHECK_INTERVAL * 3600
            if 标记文件.is_file():
                if time.time() - 标记文件.stat().st_mtime < 检查间隔:
                    return
            截止时间 = time.time() - Self.Config.CACHE_TTL_HOURS * 3600
            清理数 = 0
            for 子目录 in 缓存根.iterdir():
                if 子目录.is_dir() and 子目录.name != "__pycache__":
                    try:
                        if 子目录.stat().st_mtime < 截止时间:
                            shutil.rmtree(子目录, ignore_errors=True)
                            清理数 += 1
                    except Exception:
                        pass
            标记文件.write_text(str(int(time.time())))
            if 清理数:
                Self.日志("log.module.cache.clean", count=清理数, info_level=0)
        except Exception:
            pass
    def 归一化向量(Self, 数组):
        数组 = np.ascontiguousarray(数组, dtype=np.float32)
        范数 = np.linalg.norm(数组, axis=1, keepdims=True)
        范数[范数 < 1e-8] = 1e-8
        return 数组 / 范数
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
