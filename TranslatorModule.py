from TranslatorLib import (
    queue, json, uuid, pickle, zipfile, Path, eb, PurePosixPath, tomllib, snbtlib, ast, logging, RotatingFileHandler, QueueHandler, QueueListener, np, threading, ThreadPoolExecutor, as_completed, time, GPU_ACC, faiss, SimpleNamespace, requests, HTTPAdapter,
    re, fancymenulib, hqmlib, List, shlex, FileHandler, Retry, locale, System, dnfile)
from TranslatorConfig import RuntimeConfig, DEFAULT_CONFIG
from TranslatorLocale import Locale
import TranslatorPersistence

class FlushingFileHandler(RotatingFileHandler):
    def emit(Self, record):
        super().emit(record)
        Self.flush()
class NoRotateHandler(FileHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if self.stream is None:
                self.stream = self._open()
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

class Module:
    def __init__(Self, Config: dict = None):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.Locale = Locale(Config=Config)
        Self.日志 = Self.写入日志
        Self.Lang = Self.Locale.Lang
        Self.tqdm = Self.Locale.Tqdm
        Self.嵌入模型 = None
        Self.重排序模型 = None
        Self.启动日志()
        Self.会话 = SimpleNamespace()
        Self.会话.EMB = Self.高并发会话(Self.Config.EMB_API_KEY, Self.Config.EMB_MAX_WORKERS, Self.Config.EMB_RETRY_COEF, Self.Config.EMB_MAX_RETRY)
        Self.会话.RERANKER = Self.高并发会话(Self.Config.RERANKER_API_KEY, Self.Config.RERANKER_MAX_WORKERS, Self.Config.RERANKER_RETRY_COEF, Self.Config.RERANKER_MAX_RETRY)
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
    def __exit__(Self, *args):
        for 会话名称 in ['EMB', "RERANKER"]:
            getattr(Self.会话, 会话名称).close()
    def 启动日志(Self):
        日志名称 = f"Translator_{id(Self)}"
        Self.日志管理器 = logging.getLogger(日志名称)
        Self.日志管理器.setLevel(logging.DEBUG if Self.Config.DEBUG_MODE else logging.INFO)
        if Self.日志管理器.handlers:
            Self.日志管理器.handlers.clear()
        日志队列 = queue.Queue(-1)
        队列处理器 = QueueHandler(日志队列)
        Self.日志管理器.addHandler(队列处理器)
        日志文件 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log").resolve()
        日志文件.parent.mkdir(parents=True, exist_ok=True)
        格式化器 = logging.Formatter(fmt='[%(asctime)s][%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        文件处理器 = NoRotateHandler(str(日志文件), encoding='utf-8', mode='a')
        文件处理器.setFormatter(格式化器)
        处理器列表 = [文件处理器]
        默认日志 = Path(f"{DEFAULT_CONFIG.LOGS_FILE_PATH}/{DEFAULT_CONFIG.LOGS_FILE_NAME}.log").resolve()
        if Self.Config.LOGS_GLOBAL and 日志文件 != 默认日志:
            默认日志.parent.mkdir(parents=True, exist_ok=True)
            全局处理器 = NoRotateHandler(str(默认日志), encoding='utf-8', mode='a')
            全局处理器.setFormatter(格式化器)
            处理器列表.append(全局处理器)
        Self._队列监听器 = QueueListener(日志队列, *处理器列表, respect_handler_level=True)
        Self._队列监听器.start()
        Self.日志管理器.propagate = False
    def 写入日志(Self, text: str, info_level: int = 0, **kwargs):
        等级映射 = {0: logging.INFO, 1: logging.WARNING, 2: logging.ERROR, 3: logging.CRITICAL, 4: logging.DEBUG}
        等级 = 等级映射.get(info_level, logging.INFO)
        if info_level == 4 and not Self.Config.DEBUG_MODE:
            return
        本地化文本 = Self.Lang(text, **kwargs)
        含堆栈 = bool(re.search(r'traceback|trace back|文件.*行|line \d+', 本地化文本, re.I))
        if not 含堆栈:
            本地化文本 = 本地化文本.replace('\n', ' ').replace('\r', '')
        Self.日志管理器.log(等级, "%s", 本地化文本)
    def 读取日志(Self):
        日志文件名 = Path(Self.Config.LOGS_FILE_PATH) / Self.Config.LOGS_FILE_NAME
        log_path = f"{日志文件名}.log"
        if not Path(log_path).exists():
            return ""
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(log_path, "r", encoding="latin1") as f:
                content = f.read()
            Self.写入日志("log.module.logs.encoding.warning", info_level=1)
            return content
    def 读取Json文件(Self, file):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'utf-16', locale.getpreferredencoding(False)]:
            try:
                with open(file, 'r', encoding=enc) as f:
                    原始对象 = json.load(f)
                return 原始对象
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                continue
    def 读取单个FTBQ_Snbt文件(Self, index: str):
        文本列表 = []
        try:
            SNBT文件 = snbtlib.loads(Path(index).read_text(encoding='utf-8'))
        except Exception:
            Self.写入日志("log.module.quests.load.error", mod="FTB Quests", file=index, e=eb.format_exc(), info_level=2)
            return 文本列表
        目标字段 = {"title", "subtitle", "text", "pages", "Lore", "Name"}
        可能含JSON的字段 = {"description", "text"}
        def 提取(obj, 路径):
            if isinstance(obj, dict):
                for 键, 值 in obj.items():
                    当前路径 = 路径 + [键]
                    if 键 in 目标字段:
                        if isinstance(值, str):
                            文本列表.append([[index, 当前路径], 值])
                        elif isinstance(值, list):
                            for 序号, 元素 in enumerate(值):
                                if isinstance(元素, str):
                                    文本列表.append([[index, 当前路径 + [序号]], 元素])
                                else:
                                    提取(元素, 当前路径 + [序号])
                        else:
                            提取(值, 当前路径)
                    elif 键 in 可能含JSON的字段:
                        if isinstance(值, str):
                            Self._尝试提取内嵌JSON(index, 当前路径, 值, 文本列表)
                        elif isinstance(值, list):
                            for 序号, 元素 in enumerate(值):
                                if isinstance(元素, str):
                                    Self._尝试提取内嵌JSON(index, 当前路径 + [序号], 元素, 文本列表)
                                else:
                                    提取(元素, 当前路径 + [序号])
                        else:
                            提取(值, 当前路径)
                    else:
                        提取(值, 当前路径)
            elif isinstance(obj, list):
                for 序号, 元素 in enumerate(obj):
                    提取(元素, 路径 + [序号])
        提取(SNBT文件, [])
        return 文本列表
    def _尝试提取内嵌JSON(Self, 文件, 路径前缀, 字符串, 输出列表):
        try:
            解析结果 = json.loads(字符串)
        except (json.JSONDecodeError, TypeError):
            输出列表.append([[文件, 路径前缀], 字符串])
            return
        if isinstance(解析结果, dict):
            if "text" in 解析结果:
                输出列表.append([[文件, 路径前缀 + [解析结果, "text"]], 解析结果["text"]])
            elif "translate" in 解析结果:
                输出列表.append([[文件, 路径前缀 + [解析结果, "translate"]], 解析结果["translate"]])
        else:
            输出列表.append([[文件, 路径前缀], 字符串])
    def 应用FTBQ翻译(Self, index: list, mode: str):
        文件映射 = {}
        for 条目 in index:
            文件路径 = 条目[0][0]
            路径段 = 条目[0][1:]
            译文 = 条目[1]
            文件映射.setdefault(文件路径, []).append((路径段, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            SNBT文件 = snbtlib.loads(Path(文件路径).read_text(encoding='utf-8'))
            for 路径, 译文 in 替换列表:
                try:
                    当前对象 = SNBT文件
                    for i, 段 in enumerate(路径):
                        if isinstance(段, (str, int)):
                            if isinstance(当前对象, list):
                                段 = int(段)
                            当前对象 = 当前对象[段]
                        elif isinstance(段, dict):
                            解析对象 = 段
                            剩余路径 = 路径[i+1:]
                            目标 = 解析对象
                            for p in 剩余路径[:-1]:
                                if isinstance(目标, list):
                                    p = int(p)
                                目标 = 目标[p]
                            最后键 = 剩余路径[-1]
                            if isinstance(目标, list):
                                最后键 = int(最后键)
                            目标[最后键] = 译文
                            新字符串 = json.dumps(解析对象, ensure_ascii=False)
                            break
                    else:
                        if isinstance(当前对象, list):
                            路径[-1] = int(路径[-1])
                        当前对象[路径[-1]] = 译文
                        continue
                    当前对象 = SNBT文件
                    for j, 段 in enumerate(路径):
                        if isinstance(段, dict):
                            pass
                            break
                        if isinstance(当前对象, list):
                            段 = int(段)
                        当前对象 = 当前对象[段]
                    父容器 = SNBT文件
                    for j, 段 in enumerate(路径):
                        if isinstance(段, dict):
                            前一段 = 路径[j-1]
                            if isinstance(父容器, list):
                                前一段 = int(前一段)
                            父容器[前一段] = 新字符串
                            break
                        if isinstance(父容器, list):
                            段 = int(段)
                        父容器 = 父容器[段]
                except Exception: Self.写入日志("log.module.quests.write.error", file=文件路径, path=str(路径), e=eb.format_exc(), info_level=2)
            Path(文件路径).write_text(snbtlib.dumps(SNBT文件, compact=False if mode=="H" else True), encoding='utf-8')
    def 读取单个BQ_Json文件(Self, index: str):
        文件列表 = []
        try:
            NBT文件 = Self.读取Json文件(index)
            def dfs(obj, current_path):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key in ("name:8", "desc:8") and isinstance(value, str):
                            文件列表.append([[index, current_path + [key]], value])
                        else:
                            dfs(value, current_path + [key])
                elif isinstance(obj, list):
                    for idx, item in enumerate(obj):
                        dfs(item, current_path + [idx])
            dfs(NBT文件, [])
        except Exception: Self.写入日志("log.module.quests.load.error", mod="BetterQuests", file=index, e=eb.format_exc(), info_level=2)
        return 文件列表
    def 应用BQ翻译(Self, index: list):
        文件映射 = {}
        for 条目 in index:
            位置 = 条目[0]
            译文 = 条目[1]
            文件路径 = 位置[0]
            路径列表 = 位置[1]
            文件映射.setdefault(文件路径, []).append((路径列表, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            try:
                with open(文件路径, "r", encoding="utf-8") as f: 数据 = json.load(f)
            except Exception: Self.写入日志("log.module.quests.load.error", mod="BetterQuests", file=文件路径, e=eb.format_exc(), info_level=2); continue
            for 路径, 译文 in 替换列表:
                try:
                    obj = 数据
                    for key in 路径[:-1]:
                        if isinstance(obj, list):
                            key = int(key)
                        obj = obj[key]
                    最后键 = 路径[-1]
                    if isinstance(obj, list):
                        最后键 = int(最后键)
                    obj[最后键] = 译文
                except Exception: Self.写入日志("log.module.quests.write.error", mod="BetterQuests", file=文件路径, path=str(路径), e=eb.format_exc(), info_level=2)
            with open(文件路径, "w", encoding="utf-8") as f:
                json.dump(数据, f, ensure_ascii=False, indent=4)
    def Unicode转字符串(Self, s: str) -> str:
        return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

    def 字符串转Unicode(Self, s: str) -> str:
        parts = []
        for c in s:
            if c == '"':
                parts.append('\\"')
            elif c == '\\':
                parts.append('\\\\')
            elif ord(c) < 128 and c.isprintable():
                parts.append(c)
            elif ord(c) <= 0xFFFF:
                parts.append(f'\\u{ord(c):04X}')
            else:
                parts.append(c)
        return ''.join(parts)
    def 读取单个ZS文件(Self, 文件路径: str) -> list:
        结果 = []
        try:
            with open(文件路径, "r", encoding="utf-8") as f:
                行列表 = f.readlines()
            for 行号, 行内容 in enumerate(行列表, 1):
                for 类型名, 模式 in Self.正则表达式预编译.ZS模式.items():
                    for 序号, 匹配 in enumerate(模式.finditer(行内容)):
                        key = f"{文件路径}::{行号}::{序号}::{类型名}"
                        decoded = Self.Unicode转字符串(匹配.group(1))
                        结果.append([key, decoded, 文件路径])
        except Exception:
            Self.写入日志("log.module.script.load.error", mod="ZenScript", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果
    def 应用ZS翻译(Self, index: list) -> None:
        if not index:
            return
        文件映射 = {}
        for item in index:
            key_raw = item[0]
            译文 = item[1]
            文件路径 = item[2]
            if isinstance(key_raw, list):
                key_str = key_raw[0]
            else:
                key_str = key_raw
            文件映射.setdefault(文件路径, []).append((key_str, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            路径 = Path(文件路径)
            原内容 = 路径.read_text(encoding="utf-8")
            行列表 = 原内容.splitlines(keepends=True)
            行信息 = {}
            for key_str, 译文 in 替换列表:
                parts = key_str.split("::")
                类型 = parts[-1]
                序号 = int(parts[-2])
                行号 = int(parts[-3])
                行信息.setdefault(行号, []).append((类型, 序号, Self.字符串转Unicode(译文)) if Self.Config.SCRIPT_CRT_WRITE_UNICODE else (类型, 序号, 译文))
            for 行号, 条目列表 in 行信息.items():
                if 行号 < 1 or 行号 > len(行列表):
                    continue
                行内容 = 行列表[行号 - 1]
                待替换 = []
                for 类型, 序号, 译文 in 条目列表:
                    替换模式 = Self.正则表达式预编译.ZS替换模式[类型]
                    matches = list(替换模式.finditer(行内容))
                    if 序号 < len(matches):
                        m = matches[序号]
                        待替换.append((m.start(2), m.end(2), 译文))
                    else:
                        Self.写入日志("log.module.script.match.miss", inedx=key_str, info_level=2)
                待替换.sort(key=lambda x: x[0], reverse=True)
                for start, end, 译文 in 待替换:
                    safe_译文 = 译文.replace('\\', '\\\\').replace('"', '\\"')
                    行内容 = 行内容[:start] + safe_译文 + 行内容[end:]
                行列表[行号 - 1] = 行内容
            新内容 = "".join(行列表)
            路径.write_text(新内容, encoding="utf-8")
    def 读取单个CMM文件(Self, 文件路径: str) -> list:
        结果 = []
        try:
            数据 = Self.读取Json文件(文件路径)
            def 遍历(obj, 父路径):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "text" and isinstance(v, str):
                            结果.append([[文件路径, 父路径, ["text"]], v])
                        else:
                            遍历(v, 父路径 + [k])
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        遍历(item, 父路径 + [i])
            遍历(数据, [])
        except Exception:
            Self.写入日志("log.module.menu.load.error", mod="CustomMainMenu", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果
    def 应用CMM翻译(Self, index: list) -> None:
        if not index:
            return
        文件映射 = {}
        for 项 in index:
            文件路径 = 项[0][0]
            父路径 = 项[0][1]
            字段名 = 项[0][2][0]
            译文 = 项[1]
            文件映射.setdefault(文件路径, []).append((父路径, 字段名, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            数据 = Self.读取Json文件(文件路径)
            for 父路径, 字段名, 译文 in 替换列表:
                obj = 数据
                for key in 父路径:
                    if isinstance(obj, list):
                        key = int(key)
                    obj = obj[key]
                if isinstance(obj, list):
                    obj[int(字段名)] = 译文
                else:
                    obj[字段名] = 译文
            with open(文件路径, "w+", encoding="utf-8") as f:
                json.dump(数据, f, ensure_ascii=False, indent=4)
                
    def 读取单个FM文件(Self, 文件路径: str) -> list:
        结果 = []
        try:
            with open(文件路径, 'r', encoding='utf-8') as f:
                布局数据 = fancymenulib.load(f)
        except Exception:
            Self.写入日志("log.module.menu.load.error", mod="FancyMenu", file=文件路径, e=eb.format_exc(), info_level=2)
            return 结果

        elements = 布局数据.get('element')
        if elements is None:
            return 结果
        if not isinstance(elements, list):
            elements = [elements]
        for el in elements:
            if not isinstance(el, dict):
                continue
            实例ID = el.get('instance_identifier', 'unknown')
            if 'source' in el:
                结果.append([[文件路径, 实例ID, 'source'], el['source']])
            if 'label' in el:
                结果.append([[文件路径, 实例ID, 'label'], el['label']])
            if 'hoverlabel' in el:
                结果.append([[文件路径, 实例ID, 'hoverlabel'], el['hoverlabel']])
        return 结果
    
    def 应用FM翻译(Self, 翻译列表: list) -> None:
        文件映射 = {}
        for 条目 in 翻译列表:
            位置 = 条目[0]
            译文 = 条目[1]
            文件路径 = 位置[0]
            文件映射.setdefault(文件路径, []).append((位置[1], 位置[2], 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            try:
                with open(文件路径, 'r', encoding='utf-8') as f:
                    布局数据 = fancymenulib.load(f)
            except Exception:
                Self.写入日志("log.module.menu.load.error", mod="FancyMenu", file=文件路径, e=eb.format_exc(), info_level=2)
                continue
            elements = 布局数据.get('element')
            if elements is None:
                continue
            is_single = not isinstance(elements, list)
            if is_single:
                elements = [elements]
            el_map = {}
            for i, el in enumerate(elements):
                if isinstance(el, dict) and 'instance_identifier' in el:
                    el_map[el['instance_identifier']] = i
            for 实例ID, 字段名, 译文 in 替换列表:
                if 实例ID in el_map:
                    idx = el_map[实例ID]
                    elements[idx][字段名] = 译文
                else:
                    Self.写入日志("log.module.menu.write.error", mod="FancyMenu", file=文件路径, id=实例ID, field=字段名, info_level=2)
            if is_single:
                布局数据['element'] = elements[0]
            else:
                布局数据['element'] = elements
            try:
                with open(文件路径, 'w', encoding='utf-8') as f:
                    fancymenulib.dump(布局数据, f)
            except Exception:
                Self.写入日志("log.module.menu.write.error", mod="FancyMenu", file=文件路径, e=eb.format_exc(), info_level=2)
    def 读取单个帕秋莉手册文件(Self, 文件路径: str) -> list:
        结果 = []
        try:
            数据 = Self.读取Json文件(文件路径)
            def 遍历(obj, 当前路径):
                if isinstance(obj, dict):
                    for 键, 值 in obj.items():
                        新路径 = 当前路径 + [键]
                        if 键 in {"name", "description", "landing_text", "subtitle", "text", "title", "header", "link_text", "tooltip"} and isinstance(值, str):
                            结果.append([[文件路径, 新路径], 值])
                        else:
                            遍历(值, 新路径)
                elif isinstance(obj, list):
                    for 序号, 元素 in enumerate(obj):
                        遍历(元素, 当前路径 + [序号])
            遍历(数据, [])
        except Exception: Self.写入日志( "log.module.book.load.error", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果
    def 应用帕秋莉手册翻译(Self, 翻译列表: list) -> None:
        文件映射 = {}
        for 条目 in 翻译列表:
            位置 = 条目[0]
            译文 = 条目[1]
            文件路径 = 位置[0]
            路径列表 = 位置[1]
            文件映射.setdefault(文件路径, []).append((路径列表, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            try:
                数据 = Self.读取Json文件(文件路径)
            except Exception: Self.写入日志("log.module.book.load.error", file=文件路径, e=eb.format_exc(), info_level=2); continue
            for 路径, 译文 in 替换列表:
                try:
                    obj = 数据
                    for key in 路径[:-1]:
                        if isinstance(obj, list):
                            key = int(key)
                        obj = obj[key]
                    最后键 = 路径[-1]
                    if isinstance(obj, list):
                        最后键 = int(最后键)
                    obj[最后键] = 译文
                except Exception: Self.写入日志("log.module.book.write.error", file=文件路径, path=str(路径), e=eb.format_exc(), info_level=2)
            with open(文件路径, "w", encoding="utf-8") as f:
                json.dump(数据, f, ensure_ascii=False, indent=4)
    def 读取单个HQM文件(Self, 文件路径: str, mode: str = "H") -> list:
        结果 = []
        try:
            if mode == "H":
                数据 = Self.读取Json文件(文件路径)
            elif mode == "L":
                数据 = hqmlib.load(文件路径, encoding='gbk')
            目标字段 = {"main_description", "name", "description", "long_description", "longDescription"}
            def 递归提取(obj, 当前路径):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        新路径 = 当前路径 + [key]
                        if key in 目标字段 and isinstance(value, str):
                            结果.append([[文件路径, 新路径], value])
                        else:
                            递归提取(value, 新路径)
                elif isinstance(obj, list):
                    for idx, item in enumerate(obj):
                        递归提取(item, 当前路径 + [idx])
            递归提取(数据, [])
        except Exception: Self.写入日志("log.module.quests.load.error", mod="Hardcore Questing Mode", file=文件路径, e=eb.format_exc(),  info_level=2)
        return 结果
    def 应用HQM翻译(Self, index: list, mode: str = "H") -> None:
        文件映射 = {}
        for 条目 in index:
            位置 = 条目[0]
            译文 = 条目[1]
            文件路径 = 位置[0]
            路径列表 = 位置[1]
            文件映射.setdefault(文件路径, []).append((路径列表, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            try:
                if mode == "H":
                    数据 = Self.读取Json文件(文件路径)
                elif mode == "L":
                    数据 = hqmlib.load(文件路径, encoding='gbk')
            except Exception: Self.写入日志("log.module.quests.load.error", mod="Hardcore Questing Mode", file=文件路径, e=eb.format_exc(), info_level=2); continue
            for 路径, 译文 in 替换列表:
                try:
                    obj = 数据
                    for key in 路径[:-1]:
                        if isinstance(obj, list):
                            key = int(key)
                        obj = obj[key]
                    最后键 = 路径[-1]
                    if isinstance(obj, list):
                        最后键 = int(最后键)
                    obj[最后键] = 译文
                except Exception: Self.写入日志( "log.module.quests.write.error", mod="Hardcore Questing Mode",  file=文件路径, path=str(路径), e=eb.format_exc(), info_level=2)
            if mode == "H":
                with open(文件路径, "w", encoding="utf-8") as f:
                    json.dump(数据, f, ensure_ascii=False, indent=4)
            elif mode == "L":
                hqmlib.dump_to_hqm(数据, 文件路径, encoding='gbk')
    def 提取待翻译文本(Self, 命令: str, 规则: str) -> List[str]:
        """
        根据给定的规则行，从命令字符串中提取需要翻译的文本。

        规则格式示例：
            tellraw <targets> <tran:json:message>
            say <tran:plain:message>
            give <targets> <item> [<count>] [<tran:nbt:display.Name:物品名称>]

        参数:
            命令: 实际的命令字符串，如 '/tellraw @a {"text":"你好"}'
            规则: 规则行，如 'tellraw <targets> <tran:json:message>'

        返回:
            提取出的待翻译文本列表
        """
        命令名, 翻译参数列表 = Self._解析规则(规则)
        if not 命令名:
            return []
        命令文本 = 命令.strip()
        if 命令文本.startswith('/'):
            命令文本 = 命令文本[1:].lstrip()
        分割部分 = 命令文本.split(maxsplit=1)
        if not 分割部分 or 分割部分[0] != 命令名:
            return []
        if not 翻译参数列表:
            return []
        for 参数说明 in 翻译参数列表:
            if 参数说明['类型'] == 'plain':
                if len(分割部分) > 1:
                    return [分割部分[1]]
                return []
        try:
            参数组 = shlex.split(命令文本)[1:]
        except ValueError:
            参数组 = 命令文本.split()[1:]
        提取结果 = []
        剩余规则 = list(翻译参数列表)
        for 参数 in 参数组:
            json文本 = Self._尝试提取json文本(参数)
            if json文本 is not None:
                匹配规则 = next((规则项 for 规则项 in 剩余规则 if 规则项['类型'] == 'json'), None)
                if 匹配规则:
                    提取结果.append(json文本)
                    剩余规则.remove(匹配规则)
                    continue
            for 规则项 in list(剩余规则):
                if 规则项['类型'] == 'nbt':
                    路径 = 规则项.get('额外信息', '')
                    nbt文本 = Self._提取nbt文本(参数, 路径)
                    if nbt文本 is not None:
                        提取结果.append(nbt文本)
                        剩余规则.remove(规则项)
                        break
        return 提取结果
    def _解析规则(Self, 规则: str):
        规则 = 规则.strip()
        if not 规则 or 规则.startswith('#'):
            return None, []
        命令名 = 规则.split()[0]
        参数定义列表 = re.findall(r'[<\[]\s*(.*?)\s*[>\]]', 规则)
        翻译参数说明列表 = []
        for 参数定义 in 参数定义列表:
            if 参数定义.startswith('tran:'):
                部分 = 参数定义.split(':')
                说明 = {'类型': 部分[1]}
                if len(部分) > 2:
                    说明['额外信息'] = ':'.join(部分[2:])
                翻译参数说明列表.append(说明)
        return 命令名, 翻译参数说明列表
    def _尝试提取json文本(Self, 参数: str):
        try:
            对象 = json.loads(参数)
            if isinstance(对象, dict) and 'text' in 对象:
                return 对象['text']
        except (json.JSONDecodeError, TypeError):
            pass
        return None
    def _提取nbt文本(Self, nbt参数: str, 路径: str):
        if not (nbt参数.startswith('{') and nbt参数.endswith('}')):
            return None
        键列表 = 路径.split('.')
        当前内容 = nbt参数
        for 键 in 键列表[:-1]:
            模式 = re.escape(键) + r'\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            匹配 = re.search(模式, 当前内容)
            if not 匹配:
                return None
            当前内容 = 匹配.group(1)
        最后键 = 键列表[-1]
        字符串模式 = re.escape(最后键) + r"""\s*:\s*('[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*")"""
        匹配 = re.search(字符串模式, 当前内容)
        if not 匹配:
            return None
        原始值 = 匹配.group(1)
        引号 = 原始值[0]
        值 = 原始值[1:-1]
        if 引号 == "'":
            值 = 值.replace("\\'", "'").replace("\\\\", "\\")
        else:
            try:
                值 = json.loads(原始值)
            except json.JSONDecodeError:
                pass
        try:
            对象 = json.loads(值)
            if isinstance(对象, dict) and 'text' in 对象:
                return 对象['text']
        except (json.JSONDecodeError, TypeError):
            pass
        return None
    def 读取单个数据包文件(Self, 文件路径: str) -> list:
        结果 = []
        if 文件路径.suffix in [".json", ".mcmeta"]:
            try:
                数据 = Self.读取Json文件(文件路径)
                目标字段 = {"description", "title", "name", "subtitle"}
                def 递归提取(obj, 当前路径):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            新路径 = 当前路径 + [key]
                            if key in 目标字段 and (isinstance(value, str) or isinstance(value, dict)):
                                结果.append([[文件路径, 新路径], str(value)])
                            else:
                                递归提取(value, 新路径)
                    elif isinstance(obj, list):
                        for idx, item in enumerate(obj):
                            递归提取(item, 当前路径 + [idx])
                递归提取(数据, [])
            except Exception: Self.写入日志("log.module.data.load.error", file=文件路径, e=eb.format_exc(),  info_level=2)
        elif 文件路径.suffix == ".mcfunction":
            try:
                with open(文件路径, "r", encoding="utf-8") as f:
                    行列表 = f.readlines()
                for 行号, 原始行 in enumerate(行列表, start=1):
                    行 = 原始行.strip()
                    if not 行 or 行.startswith("#"):
                        continue
                    if 行.startswith("/"):
                        行 = 行[1:].lstrip()
                    for 规则 in TranslatorPersistence.缓存数据包指令表(Self):
                        提取到的文本列表 = Self.提取待翻译文本(行, 规则)
                        for 文本 in 提取到的文本列表:
                            结果.append([[文件路径, f"第{行号}行"], 文本])
            except Exception:
                Self.写入日志("log.module.data.load.error", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果
    def 应用数据包翻译(Self, index: list) -> None:
        文件映射 = {}
        for 条目 in index:
            位置 = 条目[0]
            译文 = 条目[1]
            文件路径 = 位置[0]
            if 文件路径.suffix in [".json", ".mcmeta"]:
                路径列表 = 位置[1]
                文件映射.setdefault(文件路径, []).append(("json", 路径列表, 译文))
            elif 文件路径.suffix == ".mcfunction":
                行描述 = 位置[1]
                原文 = 位置[2]
                文件映射.setdefault(文件路径, []).append(("mcfunction", 行描述, 原文, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            if 文件路径.suffix in [".json", ".mcmeta"]:
                try:
                    数据 = Self.读取Json文件(文件路径)
                except Exception: Self.写入日志("log.module.data.load.error", file=文件路径, e=eb.format_exc(), info_level=2); continue
                for 类型, 路径, 译文 in 替换列表:
                    try:
                        obj = 数据
                        for key in 路径[:-1]:
                            if isinstance(obj, list):
                                key = int(key)
                            obj = obj[key]
                        最后键 = 路径[-1]
                        if isinstance(obj, list):
                            最后键 = int(最后键)
                        if isinstance(译文, str):
                            try:
                                译文 = json.loads(译文)
                            except json.JSONDecodeError: pass
                        obj[最后键] = 译文
                    except Exception:
                        Self.写入日志("log.module.data.write.error",
                                    file=文件路径, path=str(路径),
                                    e=eb.format_exc(), info_level=2)
                with open(文件路径, "w", encoding="utf-8") as f:
                    json.dump(数据, f, ensure_ascii=False, indent=4)
            elif 文件路径.suffix == ".mcfunction":
                try:
                    with open(文件路径, "r", encoding="utf-8") as f:
                        所有行 = f.readlines()
                except Exception:
                    Self.写入日志("log.module.data.load.error", file=文件路径, e=eb.format_exc(), info_level=2)
                    continue
                行替换字典 = {}
                for 类型, 行描述, 原文, 译文 in 替换列表:
                    try:
                        行号 = int(行描述[1:-1])
                    except:
                        continue
                    行替换字典.setdefault(行号, []).append((原文, 译文))
                for 行号, 替换对列表 in 行替换字典.items():
                    if 行号 < 1 or 行号 > len(所有行):
                        continue
                    行内容 = 所有行[行号 - 1]
                    for 原文, 译文 in 替换对列表:
                        行内容 = 行内容.replace(原文, 译文)
                    所有行[行号 - 1] = 行内容
                with open(文件路径, "w", encoding="utf-8") as f:
                    f.writelines(所有行)
    def 读取未知伤亡语言文件(Self, file: str):
        提取列表 = []
        try:
            原始对象 = Self.读取Json文件(file)
        except Exception:
            Self.写入日志("log.module.lang.load.error", mod="Casualties: Unknown", file=file, e=eb.format_exc(), info_level=2)
            return []
        def 递归提取(obj, 路径前缀):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("Item2", "Item3"):
                        continue
                    新路径 = f"{路径前缀}.{k}" if 路径前缀 else k
                    递归提取(v, 新路径)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    新路径 = f"{路径前缀}.{idx}" if 路径前缀 else str(idx)
                    递归提取(item, 新路径)
            elif isinstance(obj, str):
                提取列表.append([路径前缀, obj, file])
        for 顶层键, 顶层值 in 原始对象.items():
            if 顶层键 in ("name", "description"):
                continue
            递归提取(顶层值, 顶层键)
        return 提取列表
    def 保存未知伤亡语言文件(Self, 条目列表: list, 使用模型=None):
        if not 条目列表:
            return
        文件路径 = 条目列表[0][2]
        try:
            数据 = Self.读取Json文件(文件路径)
        except Exception:
            Self.写入日志("log.module.lang.load.error", mod="Casualties: Unknown", file=文件路径, e=eb.format_exc(), info_level=2)
            return
        数据["description"] = Self.Config.PACK_META_TEMPLATE_CASUALTIESUNKNOWN.format(lang=Self.Config.LANGUAGE_OUTPUT, model=", ".join(使用模型[0]) or Self.Config.LLM_MODEL or Self.Lang("log.core.package.zip.hit"), author=Self.Config.PACK_AUTHOR or "海盐青茫")
        数据["name"] = Self.Config.LANGUAGE_OUTPUT
        映射 = {}
        for 条目 in 条目列表:
            路径键 = 条目[0]
            译文 = 条目[1]
            映射[路径键] = 译文
        def 设置值(obj, 路径片段列表, 值):
            if not 路径片段列表:
                return
            cur = 路径片段列表[0]
            if len(路径片段列表) == 1:
                if isinstance(obj, list):
                    try:
                        idx = int(cur)
                        if 0 <= idx < len(obj):
                            obj[idx] = 值
                    except (ValueError, IndexError):
                        pass
                elif isinstance(obj, dict):
                    obj[cur] = 值
                return
            if isinstance(obj, list):
                try:
                    idx = int(cur)
                    if 0 <= idx < len(obj):
                        设置值(obj[idx], 路径片段列表[1:], 值)
                except (ValueError, IndexError):
                    pass
            elif isinstance(obj, dict):
                if cur in obj:
                    设置值(obj[cur], 路径片段列表[1:], 值)
        for 路径键, 译文 in 映射.items():
            片段 = 路径键.split('.')
            分类键 = 片段[0]
            if 分类键 in 数据:
                设置值(数据[分类键], 片段[1:], 译文)
        try:
            with open(Path(文件路径).parent / f"{Self.Config.LANGUAGE_OUTPUT}.json", "w", encoding="utf-8") as f:
                json.dump(数据, f, ensure_ascii=False, indent=4)
        except Exception:
            Self.写入日志("log.module.lang.write.error", mod="Casualties: Unknown", file=文件路径, e=eb.format_exc(), info_level=2)
    def 读取单个DLL文件(Self, 文件路径: str) -> list:
        结果 = []
        p_file = Path(文件路径)
        try:
            pe = dnfile.dnPE(str(p_file))
            if not pe.net or not pe.net.user_strings:
                return 结果
            us_data = pe.net.user_strings.__data__
            提取到的字符串 = Self._解析NetUS堆(us_data)
            for s in 提取到的字符串:
                结果.append([["ldstr", s], s, p_file])
        except Exception:
            Self.写入日志("log.module.dll.load.error", file=str(p_file), e=eb.format_exc(), info_level=2)
        finally:
            if pe is not None:
                try:
                    pe.close()
                except Exception:
                    pass
        return 结果

    def _解析NetUS堆(Self, data: bytes) -> list:
        字符串列表 = []
        offset = 1
        while offset < len(data):
            b1 = data[offset]
            if b1 == 0:
                offset += 1
                continue
            if b1 < 0x80: length, offset = b1, offset + 1
            elif b1 < 0xC0: length, offset = ((b1 & 0x3F) << 8) | data[offset+1], offset + 2
            elif b1 < 0xE0: length, offset = ((b1 & 0x1F) << 24) | (data[offset+1] << 16) | (data[offset+2] << 8) | data[offset+3], offset + 4
            else: break
            if length == 0: continue
            str_len = length - 1
            if offset + str_len > len(data) or str_len <= 0:
                offset += length
                continue
            str_bytes = data[offset:offset+str_len]
            offset += length
            try:
                s = str_bytes.decode('utf-16-le')
                if s and len(s.strip()) > 1:
                    字符串列表.append(s)
            except UnicodeDecodeError: 
                pass
        return list(dict.fromkeys(字符串列表))
    def 应用DLL翻译(Self, 翻译列表: list) -> None:
        文件映射 = {}
        for 条目 in 翻译列表:
            位置 = 条目[0]
            译文 = 条目[1]
            文件路径 = 位置[0]
            索引列表 = 位置[1]
            原文 = 索引列表[1]
            文件映射.setdefault(文件路径, {})[原文] = 译文
        for 文件路径, 替换字典 in 文件映射.items():
            if not 替换字典:
                continue
            try:
                Self._使用Cecil回写DLL(文件路径, 替换字典)
            except Exception:
                Self.写入日志("log.module.dll.write.error", file=文件路径, e=eb.format_exc(), info_level=2)

    def _使用Cecil回写DLL(Self, 文件路径: str, 替换字典: dict) -> None:
        p_file = Path(文件路径).resolve()
        cecil_file = Path(f"{Self.Config.MONO_CECIL_DLL_PATH}/{Self.Config.MONO_CECIL_DLL_NAME}").resolve()
        System.Reflection.Assembly.LoadFrom(str(cecil_file))
        from Mono.Cecil import ModuleDefinition, DefaultAssemblyResolver, ReaderParameters  # type: ignore
        from Mono.Cecil.Cil import OpCodes  # type: ignore
        游戏根目录 = p_file.parent.parent.parent 
        managed_dirs = list(游戏根目录.glob("*_Data/Managed"))
        unity_managed = managed_dirs[0] if managed_dirs else None
        resolver = DefaultAssemblyResolver()
        resolver.AddSearchDirectory(str(cecil_file.parent)) 
        bepinex_core = p_file.parent.parent / "core"
        if bepinex_core.exists():
            resolver.AddSearchDirectory(str(bepinex_core))
        if unity_managed and unity_managed.exists():
            resolver.AddSearchDirectory(str(unity_managed))
        reader_params = ReaderParameters()
        reader_params.AssemblyResolver = resolver
        module = ModuleDefinition.ReadModule(str(p_file), reader_params)
        def 处理类型(types):
            for type_def in types:
                for method in type_def.Methods:
                    if not method.HasBody: continue
                    for instr in method.Body.Instructions:
                        if instr.OpCode == OpCodes.Ldstr:
                            if instr.Operand in 替换字典:
                                instr.Operand = 替换字典[instr.Operand]
                if type_def.HasNestedTypes:
                    处理类型(type_def.NestedTypes)
        处理类型(module.Types)
        临时文件 = p_file.with_name(p_file.name + ".translated")
        module.Write(str(临时文件))
        module.Dispose()
        System.GC.Collect()
        System.GC.WaitForPendingFinalizers()
        临时文件.replace(p_file)
    def 读取语言文件(Self, file: str):
        with open(file, "r", encoding="utf-8") as f:
            if Path(file).suffix == ".lang":
                源文件 = f.read().splitlines()
            elif Path(file).suffix == ".json":
                Json文件 = json.load(f)
                源文件 = [f"{index}={Json文件[index]}" for index in Json文件]
            elif Path(file).suffix == ".local":
                源文件 = [re.sub(r'\s*=\s*', '=', line) for line in f.read().splitlines() if line.strip()]
        return [
            (lambda parts: [parts[0], parts[1], file])(line.split('=', 1)) 
            for line in 源文件 
            if (stripped := line.strip()) and stripped and '=' in stripped and not stripped.startswith(('#', '//'))
        ], [源文件, file]
    def 保存语言文件(Self, file: str, 保存列表: list):
        with open(file, "w+", encoding="utf-8") as f:
            if Path(file).suffix == ".lang":
                清理列表 = []
                for line in 保存列表:
                    if not line.strip():
                        清理列表.append(line)
                        continue
                    line = line.replace(r'\n', '<<NEWLINE_PLACEHOLDER>>')
                    line = line.replace(r'\r', '<<CARRIAGE_PLACEHOLDER>>')
                    line = line.replace('\r', ' ').replace('\n', ' ')
                    line = ' '.join(line.split())
                    line = line.replace('<<NEWLINE_PLACEHOLDER>>', r'\n')
                    line = line.replace('<<CARRIAGE_PLACEHOLDER>>', r'\r')
                    清理列表.append(line)
                f.write("\n".join(清理列表))
            elif Path(file).suffix == ".json":
                json文件 = {}
                保存列表 = [line for line in 保存列表 if line.strip() and '=' in line and not line.lstrip().startswith(('//', '#'))]
                for index in 保存列表:
                    k, v = index.split('=', 1)
                    try:
                        v = ast.literal_eval(v)
                    except Exception: pass
                    json文件[k] = v
                f.write(json.dumps(json文件, ensure_ascii=False, indent=4))
            elif Path(file).suffix == ".local":
                f.write("\n".join([re.sub(r'\s*=\s*', ' = ', line) for line in 保存列表]))
    def 读取压缩文件(Self, file_path: str, cache_path: str, original_language: str, target_language: str):
        try:
            for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
                with zipfile.ZipFile(file_path, 'r') as f:
                    文件列表 = f.namelist()
                    缓存路径 = cache_path
                    目标集合 = {"shaders", "contenttweaker"}
                    存在目标 = any(
                        p.rstrip('/').split('/')[-1].lower() in 目标集合
                        for p in 文件列表 if p.rstrip('/').count('/') <= 1
                    )
                    可用文件列表 = [存在目标]
                    if 存在目标:
                        f.extractall(缓存路径)
                    待查语言 = {original_language.lower(), target_language.lower()}
                    for 内部文件 in 文件列表:
                        if f.getinfo(内部文件).is_dir():
                            continue
                        主体 = Path(内部文件).stem.lower()
                        if 主体 in 待查语言:
                            完整路径 = f"{缓存路径}/{内部文件}"
                            if not Path(完整路径).exists():
                                f.extract(内部文件, path=缓存路径)
                            可用文件列表.append([主体, 完整路径])
            return 可用文件列表
        except Exception:
            Self.写入日志("log.module.zip.read.error", e=eb.format_exc(), info_level=3)
            raise FileNotFoundError(eb.format_exc())
    def 读取资源文件(Self, file0: str, file1: str = "", read_error: bool = True):
        Self.写入日志("log.core.file.read.start", file0=file0, file1=file1)
        文件0, 文件0源文件, 文件1 = [], [], []
        压缩路径, 输出扩展名, file2 = "", "", ""
        语言输入 = Self.Config.LANGUAGE_INPUT.lower()
        语言输出 = Self.Config.LANGUAGE_OUTPUT.lower()
        def 解析并追加(路径列表, 目标列表, 源文件列表=None):
            nonlocal 输出扩展名
            for fpath in 路径列表:
                解析数据, 源数据 = Self.读取语言文件(fpath)
                目标列表.extend(解析数据)
                if 源文件列表 is not None:
                    源文件列表.append(源数据)
                输出扩展名 = Path(fpath).suffix

        def 校验并获取路径(zip_ret, 目标语言, 日志键):
            匹配路径 = [item[1] for item in zip_ret[1:] if isinstance(item, list) and item[0] == 目标语言]
            if not 匹配路径 and read_error:
                Self.写入日志(日志键, info_level=3)
                raise FileNotFoundError(Self.Lang(日志键))
            return 匹配路径
        是压缩包0 = Path(file0).suffix.lower() in {".zip", ".jar"}
        是压缩包1 = file1 and Path(file1).suffix.lower() in {".zip", ".jar"}
        if 是压缩包0:
            实际解压目录 = Path(Self.Config.PATH_CACHE) / f"{uuid.uuid4().hex}_{Path(file0).stem}"
            压缩路径 = str(实际解压目录)
            file2 = Self.读取压缩文件(file0, 压缩路径, Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT)
            输入路径 = 校验并获取路径(file2, 语言输入, "log.module.read.file0.not.lang.error")
            解析并追加(输入路径, 文件0, 文件0源文件)
            if 是压缩包1:
                实际解压目录1 = Path(Self.Config.PATH_CACHE) / f"{uuid.uuid4().hex}_{Path(file1).stem}"
                file3 = Self.读取压缩文件(file1, str(实际解压目录1), Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT)
                输出路径 = 校验并获取路径(file3, 语言输出, "log.module.read.file1.not.lang.error")
                解析并追加(输出路径, 文件1)
            elif file1:
                解析并追加([file1], 文件1)
            else:
                输出路径 = [item[1] for item in file2[1:] if isinstance(item, list) and item[0] == 语言输出]
                解析并追加(输出路径, 文件1)
        elif Path(file0).suffix.lower() in {".lang", ".json", ".local"}:
            解析并追加([file0], 文件0, 文件0源文件)
            if file1:
                if 是压缩包1:
                    实际解压目录1 = Path(Self.Config.PATH_CACHE) / f"{uuid.uuid4().hex}_{Path(file1).stem}"
                    压缩路径 = str(实际解压目录1)
                    file3 = Self.读取压缩文件(file1, 压缩路径, Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT)
                    输出路径 = 校验并获取路径(file3, 语言输出, "log.module.read.file1.not.lang.error")
                    解析并追加(输出路径, 文件1)
                else:
                    解析并追加([file1], 文件1)
        Self.写入日志("log.core.file.read.end", file0=文件0, file1=文件1)
        return 文件0, 文件0源文件, 文件1, 压缩路径, 输出扩展名, file2
    def 翻译缓存(Self, 输入列表: list = None):
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
    def 从资源包文件夹获取I18n翻译模组ID(Self, 路径: str):
        Self.写入日志("log.core.modid.get.start", info_level=0)
        try:
            with zipfile.ZipFile(list((Path(路径) / "resourcepacks").glob("Minecraft-Mod-Language-Modpack*")), 'r') as f:
                模组ID集 = set()
                for index in f.namelist():
                    Path路径 = PurePosixPath(index)
                    if Path路径.parts and Path路径.parts[0] == 'assets':
                        if len(Path路径.parts) > 1:
                            模组ID = Path路径.parts[1]
                            模组ID集.add(模组ID)
            返回内容 = list(模组ID集)
        except Exception:
            Self.写入日志("log.module.pack.i18n.no", info_level=3)
            返回内容 = []
        Self.写入日志("log.core.modid.get.end", info_level=0)
        return 返回内容
    def 从模组文件夹获取模组ID(Self, 路径: str):
        Self.写入日志("log.core.modid.get.start", info_level=0)
        所有模组路径 = Path(f"{路径}/{"mods"}").glob('*.jar')
        模组ID列表 = []
        for 模组文件路径 in Self.tqdm(所有模组路径, desc="tqdm.file.modid.get"):
            try:
                with zipfile.ZipFile(模组文件路径, 'r') as 压缩文件:
                    文件列表 = 压缩文件.namelist()
                    模组ID = None
                    if "fabric.mod.json" in 文件列表:
                        try:
                            with 压缩文件.open("fabric.mod.json") as 文件:
                                数据 = json.load(文件)
                            模组ID = 数据.get("id")
                        except Exception:
                            Self.写入日志("log.module.parsemodid.error", e=eb.format_exc())
                    if "META-INF/mods.toml" in 文件列表 and 模组ID is None:
                        try:
                            with 压缩文件.open("META-INF/mods.toml") as 文件:
                                数据 = tomllib.load(文件)
                            模组列表 = 数据.get('mods', [])
                            if 模组列表:
                                模组ID = 模组列表[0].get('modId')
                        except Exception:
                            Self.写入日志("log.module.parsemodid.error", e=eb.format_exc())
                    if "META-INF/neoforge.mods.toml" in 文件列表 and 模组ID is None:
                        try:
                            with 压缩文件.open("META-INF/neoforge.mods.toml") as 文件:
                                数据 = tomllib.load(文件)
                            模组列表 = 数据.get('mods', [])
                            if 模组列表:
                                模组ID = 模组列表[0].get('modId')
                        except Exception:
                            Self.写入日志("log.module.parsemodid.error", e=eb.format_exc())
                    if 模组ID is None:
                        候选文件列表 = ["mcmod.info"]
                        if "mcmod.info" not in 文件列表:
                            候选文件列表 = [文件名 for 文件名 in 文件列表 if 文件名.endswith('.info')]
                        for 信息文件 in 候选文件列表:
                            try:
                                with 压缩文件.open(信息文件) as 文件:
                                    内容 = 文件.read().decode('utf-8-sig')
                                    数据列表 = json.loads(内容)
                                    if isinstance(数据列表, list) and len(数据列表) > 0:
                                        模组ID = 数据列表[0].get('modid')
                                        break
                            except Exception:
                                continue
                    模组ID列表.append([模组ID, 模组文件路径.name])
            except Exception:
                Self.写入日志("log.module.parsemodid.file.error", e=eb.format_exc())
        Self.写入日志("log.core.modid.get.end", info_level=0)
        return 模组ID列表
    def 输出路径处理(Self, path: str):
        if not path:
            path = f"./{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}"
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path).resolve()
    def 列表去重(Self, 列表: list):
        return list(dict.fromkeys(列表))
    def 读取审查文件(Self, file0: str):
        返回列表 = []
        with open(file0, 'r', encoding='utf-8') as 文件对象:
            文件行列表 = 文件对象.read().splitlines()
        for 单行文本 in 文件行列表:
            数据字典 = ast.literal_eval(单行文本)
            for 键名, 键值 in 数据字典.items():
                返回列表.append([键名, 键值[0], 键值[1]])
        return 返回列表
    def 生成向量(Self, text: list, outputs: list = None, outputs1: list = None) -> np.float32:
        重试次数 = 0
        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            with Self.线程锁:
                try:
                    向量列表 = Self.嵌入模型.encode(text)
                    向量列表 = np.asarray(向量列表, dtype=np.float32)
                    return [向量列表, [text, outputs, outputs1]]
                except Exception:
                    Self.日志("log.core.locally.generate.vectors.error", e=eb.format_exc(), info_level=2)
                    return [None, [text, outputs, outputs1]]
        else:
            while 重试次数 < Self.Config.EMB_MAX_RETRY:
                try:
                    请求结果 = Self.会话.EMB.post(url=Self.Config.EMB_API_URL, json={"input": text,"model": Self.Config.EMB_MODEL}, timeout=(Self.Config.EMB_CONN_TIMEOUT, Self.Config.EMB_TIMEOUT))
                    请求结果.raise_for_status()
                    请求结果 = 请求结果.json()
                    向量列表 = []
                    for index in range(len(text)):
                        向量列表.append(请求结果['data'][index]['embedding'])
                    向量列表 = np.asarray(向量列表, dtype=np.float32)
                    return [向量列表, [text, outputs, outputs1]]
                except Exception:
                    重试次数 += 1
                    if 重试次数 >= Self.Config.EMB_MAX_RETRY:
                        Self.日志("log.core.api.generate.vectors.error", e=eb.format_exc(), info_level=3)
                        return [None, [text, outputs, outputs1]]
                    else:
                        Self.日志("log.core.api.generate.vectors.retry", e=eb.format_exc(), info_level=2)
                        time.sleep((Self.Config.EMB_RETRY_COEF ** (重试次数 - 1)) * Self.Config.EMB_RETRY_TIME)
    def 并行生成向量(Self, texts: list) -> list:
        Self.日志("log.core.vector.generate.start", info_level=0)
        if not texts:
            Self.日志("log.core.generated.vector.nan", texts=texts, info_level=3)
            return [np.array([], dtype=np.float32).reshape(0, 0), [[], [], []]]
        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            Self.嵌入模型 = TranslatorPersistence.获取嵌入模型(Self=Self)
        最大字符数 = Self.Config.EMB_MAX_TOKENS * Self.Config.EMB_TOKENSTOTEXT_RATIO
        分组结果, 当前组, 当前总长 = [], [], 0.0
        for index in texts:
            长度 = len(index[0])
            if 当前总长 + 长度 > 最大字符数:
                分组结果.append(当前组)
                当前组, 当前总长 = [], 0.0
            当前组.append(index)
            当前总长 += 长度
        if 当前组: 分组结果.append(当前组)
        待处理文本列表原文 = [[item[0] for item in group] for group in 分组结果]
        待处理文本列表额外输出 = [[item[1] for item in group] for group in 分组结果]
        待处理文本列表额外输出1 = [[item[2] for item in group] for group in 分组结果]
        总条数 = sum(len(g) for g in 分组结果)
        合并向量 = None
        维度 = None
        当前偏移 = 0
        偏移锁 = threading.Lock()
        提前完成缓存 = []
        合并请求文本 = []
        合并额外返回 = []
        合并额外返回1 = []
        with ThreadPoolExecutor(max_workers=Self.Config.EMB_MAX_WORKERS) as 执行器:
            未来任务映射 = {
                执行器.submit(Self.生成向量, 原文组, 额外输出, 额外输出1): 原文组
                for 原文组, 额外输出, 额外输出1 in zip(待处理文本列表原文, 待处理文本列表额外输出, 待处理文本列表额外输出1)
            }
            for 单个任务 in Self.tqdm(as_completed(未来任务映射), total=len(未来任务映射), desc="tqdm.vectors.generate"):
                结果 = 单个任务.result()
                if 结果[0] is None: continue
                块 = 结果[0]
                块长度 = 块.shape[0]
                with 偏移锁:
                    if 维度 is None:
                        维度 = 块.shape[1]
                        合并向量 = np.empty((总条数, 维度), dtype=np.float32)
                        for 缓存块, 缓存元数据 in 提前完成缓存:
                            合并向量[当前偏移:当前偏移+缓存块.shape[0]] = 缓存块
                            当前偏移 += 缓存块.shape[0]
                            合并请求文本.extend(缓存元数据[0])
                            合并额外返回.extend(缓存元数据[1])
                            合并额外返回1.extend(缓存元数据[2])
                        提前完成缓存.clear()
                    elif 块.shape[1] != 维度:
                        continue
                    合并向量[当前偏移:当前偏移+块长度] = 块
                    当前偏移 += 块长度
                合并请求文本.extend(结果[1][0])
                合并额外返回.extend(结果[1][1])
                合并额外返回1.extend(结果[1][2])
        最终向量 = 合并向量[:当前偏移] if 合并向量 is not None else np.array([], dtype=np.float32).reshape(0, 维度 or 0)
        Self.日志("log.core.vector.generate.end", info_level=0)
        return [最终向量, [合并请求文本, 合并额外返回, 合并额外返回1]]
    def 选择相似度最高译文(Self, 请求消息: list):
        请求次数 = 0
        if (not Self.Config.RERANKER_API_URL) and (Self.Config.RERANKER_MODEL):
            with Self.线程锁:
                try:
                    相似度 = Self.重排序模型.predict([(请求消息[0], 候选) for 候选 in 请求消息[1]], show_progress_bar=False)
                    return [请求消息[0], 请求消息[1][相似度.argmax()], 相似度]
                except Exception:
                    Self.日志("log.core.translator.cache.locally.error", e=eb.format_exc(), info_level=2)
                    return [请求消息[0], 请求消息[1][0], [0 for _ in range(len(请求消息[1]))]]
        else:
            请求内容 = {
                "model": Self.Config.RERANKER_MODEL,
                "documents": 请求消息[1],
                "query": 请求消息[0],
                "instruct": Self.Config.RERANKER_INSTRUCT
            }
            while 请求次数 < Self.Config.RERANKER_MAX_RETRY:
                try:
                    相似度 = []
                    请求结果 = Self.会话.RERANKER.post(url=Self.Config.RERANKER_API_URL, json=请求内容, timeout=(Self.Config.RERANKER_CONN_TIMEOUT, Self.Config.RERANKER_TIMEOUT))
                    请求结果.raise_for_status()
                    请求结果 = 请求结果.json()
                    请求结果 = 请求结果["output"]["results"]
                    for index in 请求结果:
                        相似度.append(index["document"]["text"])
                    return [请求消息[0], 请求结果[0]["document"]["text"], 相似度]
                except Exception:
                    Self.日志("log.core.translator.cache.generate.messages.error", messages=请求消息[1], e=eb.format_exc(), info_level=1)
                    请求次数 += 1
                    if 请求次数 >= Self.Config.RERANKER_MAX_RETRY:
                        Self.日志("log.core.translator.cache.generate.error", e=eb.format_exc(), output=请求结果, info_level=2)
                        return [请求消息[0], 请求消息[1][0], [0 for _ in range(len(请求消息[1]))]]
                    else:
                        Self.日志("log.core.translator.cache.generate.retry", e=eb.format_exc(), output=请求结果, info_level=1)
                        time.sleep((Self.Config.RERANKER_RETRY_COEF ** (请求次数 - 1)) * Self.Config.RERANKER_RETRY_TIME)
    def 获取相似度最高译文(Self, 输入字典: dict, 强制重排: bool=False):
        请求列表 = []
        剔除列表 = []
        返回列表 = []
        for index in 输入字典:
            if len(输入字典[index]) == 1 and 强制重排 == False:
                剔除列表.append([index, 输入字典[index][0], [0]])
            else:
                请求列表.append([index, 输入字典[index]])
        if (not Self.重排序模型) and (not Self.Config.RERANKER_API_URL) and (Self.Config.RERANKER_MODEL) and (请求列表):
            Self.重排序模型 = TranslatorPersistence.获取重排模型(Self=Self)
        Self.日志("log.core.translator.cache.generate.start", item=len(请求列表), info_level=0)
        with ThreadPoolExecutor(max_workers=Self.Config.RERANKER_MAX_WORKERS) as 执行器:
            未来任务映射 = {
                执行器.submit(
                    Self.选择相似度最高译文,
                    请求消息 = index,
                ): index
                for index in 请求列表
            }
            for 单个任务 in Self.tqdm(as_completed(未来任务映射), total=len(未来任务映射), desc="tqdm.translator.cache.generate"):
                返回列表.append(单个任务.result())
        Self.日志("log.core.translator.cache.generate.end", info_level=0)
        返回列表 += 剔除列表
        return 返回列表
    def 构建索引(Self, 向量文件, 模式=None):
        Self.日志("log.core.index.generate.start", info_level=0)
        向量文件 = 向量文件.get() if GPU_ACC else 向量文件
        训练, 构建 = False, False
        量化映射 = {
            "Q4": faiss.ScalarQuantizer.QT_4bit,
            "Q6": faiss.ScalarQuantizer.QT_6bit,
            "Q8": faiss.ScalarQuantizer.QT_8bit,
            "F16": faiss.ScalarQuantizer.QT_fp16,
            "BF16": faiss.ScalarQuantizer.QT_bf16,
        }
        量化类型 = 量化映射[Self.Config.INDEX_SQ]
        if not 模式:
            模式 = Self.Config.INDEX_MODE
        if 模式 == "HNSWSQ":
            向量索引 = faiss.IndexHNSWSQ(向量文件.shape[1], 量化类型, Self.Config.INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH
            训练, 构建, SQ = True, True, True
        elif 模式 == "HNSWPQ":
            向量索引 = faiss.IndexHNSWPQ(向量文件.shape[1], Self.Config.INDEX_HNSW_M, Self.Config.INDEX_PQ_M, Self.Config.INDEX_NLITS)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH
            训练, 构建, SQ = True, True, False
        elif 模式 == "HNSWFlat":
            向量索引 = faiss.IndexHNSWFlat(向量文件.shape[1], Self.Config.INDEX_HNSW_M)
            向量索引.hnsw.efConstruction = Self.Config.INDEX_HNSW_CONSTRUCTION
            向量索引.hnsw.efSearch = Self.Config.INDEX_HNSW_SEARCH
            训练, 构建, SQ = True, True, False
        elif 模式 == "RefineFlat":
            向量索引 = faiss.IndexRefineFlat(faiss.IndexFlatL2(向量文件.shape[1]))
            向量索引.k_factor = Self.Config.INDEX_REFINEFLAT_K_FACTOR
            训练, 构建, SQ = False, True, False
        elif 模式 == "IVFSQ":
            向量索引 = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatL2(向量文件.shape[1]), 向量文件.shape[1], Self.Config.INDEX_NLIST, 量化类型)
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE
            训练, 构建, SQ = True, True, True
        elif 模式 == "IVFPQ":
            向量索引 = faiss.IndexIVFPQ(faiss.IndexFlatL2(向量文件.shape[1]), 向量文件.shape[1], Self.Config.INDEX_NLIST, Self.Config.INDEX_PQ_M, Self.Config.INDEX_NLITS)
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE
            训练, 构建, SQ = True, True, False
        elif 模式 == "IVFFlat":
            向量索引 = faiss.IndexIVFFlat(faiss.IndexFlatL2(向量文件.shape[1]), 向量文件.shape[1], Self.Config.INDEX_IVF_NLIST, faiss.METRIC_L2)
            向量索引.nprobe = Self.Config.INDEX_IVF_NPROBE
            训练, 构建, SQ = True, True, False
        elif 模式 == "FlatL2":
            向量索引 = faiss.IndexFlatL2(向量文件.shape[1])
            训练, 构建, SQ = False, True, False
        elif 模式 == "FlatIP":
            向量索引 = faiss.IndexFlatIP(向量文件.shape[1])
            训练, 构建, SQ = False, True, False
        if SQ:
            if Self.Config.INDEX_RE_MINMAX:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_minmax
                向量索引.sq.rangestat_arg = Self.Config.INDEX_RE_MINMAX
            if Self.Config.INDEX_RE_MEANSTD:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_meanstd
                向量索引.sq.rangestat_arg = Self.Config.INDEX_RE_MEANSTD
            if Self.Config.INDEX_RE_QUANTILES:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_quantiles
                向量索引.sq.rangestat_arg = Self.Config.INDEX_RE_QUANTILES
            if Self.Config.INDEX_RE_OPTIM:
                向量索引.sq.rangestat = faiss.ScalarQuantizer.RS_optim
                向量索引.sq.rangestat_arg = Self.Config.INDEX_RE_OPTIM
        if 训练:
            for _ in Self.tqdm(range(1), desc="tqdm.index.train"):
                向量索引.train(向量文件)
        if 构建:
            for _ in Self.tqdm(range(1), desc="tqdm.index.build"):
                向量索引.add(向量文件)
        Self.日志("log.core.index.generate.end", info_level=0)
        return 向量索引
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
        if 文本.islower() and ' ' not in 文本 and len(文本) < 15: 
            return False 
        if '.' in 文本 and re.match(r'^[a-z0-9.\-]+$', 文本):
            return False
        if '/' in 文本 and not 文本.startswith('http'): 
            return False
        if 文本.startswith(("org.", "com.", "unityengine.", "system.")): 
            return False
        if re.search(r'\{[0-9]+\}', 文本): 
            return False
        危险词 = ('initialized', 'postfix', 'prefix', 'warning:', 'error:', 'exception', 'debug', 'patcher', 'log.')
        if any(kw in 文本.lower() for kw in 危险词): 
            return False
        按键词 = ('left alt', 'right alt', 'left ctrl', 'right ctrl', 'left shift', 'right shift', 'mouse')
        if 文本.lower() in 按键词:
            return False
        return True
    def 高并发会话(Self, api_key, 并发数, 避退指数, 重试次数):
        会话 = requests.Session()
        会话.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        adapter = HTTPAdapter(
            pool_connections=并发数,
            pool_maxsize=并发数,
            max_retries=Retry(
                total=重试次数,
                backoff_factor=避退指数,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            ),
            pool_block=False
        )
        会话.mount('https://', adapter)
        会话.mount('http://', adapter)
        return 会话