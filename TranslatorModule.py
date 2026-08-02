from TranslatorLib import (uuid, Path, threading, time, shutil, SimpleNamespace, re, json,
                           Config)

class Module:
    def __init__(Self, App: Config):
        Self.Config = App.Config
        Self.Locale = App.Locale
        Self.日志 = App.日志
        Self.Lang = App.Lang
        Self.tqdm = App.RichTqdm
        Self.Index = App.Index
        Self.线程锁 = threading.Lock()
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
    def 文本组件深度优先搜索(Self, 组件, 当前路径, 提取记录):
        if isinstance(组件, str):
            提取记录.append((当前路径, 组件))
            return
        if isinstance(组件, dict):
            for k, v in 组件.items():
                if k == "text":
                    提取记录.append((当前路径 + [k], v))
                elif k == "translate":
                    if Self.过滤键文本(["", v]):
                        提取记录.append((当前路径 + [k], v))
                elif k == "extra" and isinstance(v, list):
                    for i, 子组件 in enumerate(v):
                        Self.深度优先搜索(子组件, 当前路径 + [k, i])
                else:
                    Self.深度优先搜索(v, 当前路径 + [k])
            return
        if isinstance(组件, list):
            for i, 项目 in enumerate(组件):
                Self.深度优先搜索(项目, 当前路径 + [i])
            return
    def uuid(Self): # 记不住怎么写(
        return uuid.uuid4().hex
    def 按路径取值(Self, 数据: dict, 路径: list, 默认值):
        if not 路径:
            return 默认值
        if isinstance(路径[0], (list, tuple)):
            for 单条路径 in 路径:
                结果 = Self.按路径取值(数据, 单条路径, None) 
                if 结果 is not None: 
                    return 结果
            return 默认值
        当前 = 数据
        for 键 in 路径:
            if isinstance(当前, dict) and 键 in 当前:
                当前 = 当前[键]
            else:
                return 默认值
        return 当前
            
    async def POST获取错误(Self, 响应体, err):
        try:
            错误原文 = await 响应体.text()
            try:
                错误详情 = json.loads(错误原文)
                错误信息 = 错误详情.get("error") or 错误详情.get("message") or str(错误详情)
            except (json.JSONDecodeError, TypeError): 错误信息 = 错误原文
        except Exception: 错误信息 = str(err)
        return 错误信息