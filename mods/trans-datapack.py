# 可用配置(覆盖键 MOD_TRANS_DATAPACK_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   ["*.json", "*.mcmeta", "*.mcfunction"]
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (uuid, json, re, ast, Any, List, zipfile, Path, Parser, Mods, eb, TranslatorPersistence)

模组参数 = Mods().模组配置("trans-datapack", {
    "FILE_MATCH": ["*.json", "*.mcmeta", "*.mcfunction"],
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


def 执行(Self, 根, path2=None, **参数):
    解析 = DataPackParser(Self)
    path = Path(根)
    if path.is_file():
        缓存文件夹 = Path(f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
        with zipfile.ZipFile(path, 'r') as zf: zf.extractall(缓存文件夹)
    else: 缓存文件夹 = path
    Self.翻译流程(缓存文件夹, Mods().取值(Self, ("trans-datapack", "FILE_MATCH")), 解析.读取, Self.Module.过滤键文本,
                lambda x: x[0][0], 解析.写回,
                Mods().取值(Self, ("trans-datapack", "READ_CONCURRENT")),
                Mods().取值(Self, ("trans-datapack", "WRITE_CONCURRENT")), "data", path2=path2, **参数)


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
    命令名, 翻译参数列表 = 解析规则(Self, 规则)
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
        参数组 = 分割参数(Self, 命令文本)[1:]
    except Exception:
        参数组 = 命令文本.split()[1:]
    提取结果 = []
    剩余规则 = list(翻译参数列表)
    for 参数 in 参数组:
        json文本列表 = 尝试提取json文本(Self, 参数)
        if json文本列表:
            匹配规则 = next((规则项 for 规则项 in 剩余规则 if 规则项['类型'] == 'json'), None)
            if 匹配规则:
                提取结果.extend(json文本列表)
                剩余规则.remove(匹配规则)
                continue
        for 规则项 in list(剩余规则):
            if 规则项['类型'] == 'nbt':
                路径 = 规则项.get('额外信息', '')
                nbt文本 = 提取nbt文本(Self, 参数, 路径)
                if nbt文本 is not None:
                    提取结果.append(nbt文本)
                    剩余规则.remove(规则项)
                    break
    return 提取结果


def 解析规则(Self, 规则: str):
    规则 = 规则.strip()
    if not 规则 or 规则.startswith('#'):
        return None, []
    命令名 = 规则.split()[0]
    # ↓只挑 tran: 占位符 可选参数写成 [<tran:...>] 也能认出来
    翻译参数说明列表 = []
    for 参数定义 in re.findall(r'tran:([^>\]]+)', 规则):
        部分 = 参数定义.strip().split(':')
        说明 = {'类型': 部分[0]}
        # ↓nbt 规则的下一段是取值路径 再往后只是给人看的名称 不能拼进路径
        if 说明['类型'] == 'nbt' and len(部分) > 1:
            说明['额外信息'] = 部分[1]
        翻译参数说明列表.append(说明)
    return 命令名, 翻译参数说明列表


def 分割参数(Self, 命令文本: str) -> List[str]:
    """按空白切分命令, 但引号与 {} [] 内部保持原样。"""
    参数组, 当前, 深度, 引号, 转义 = [], [], 0, "", False
    for 字符 in 命令文本:
        if 转义:
            当前.append(字符); 转义 = False; continue
        if 引号:
            当前.append(字符)
            if 字符 == "\\":
                转义 = True
            elif 字符 == 引号:
                引号 = ""
            continue
        if 字符 in ("'", '"'):
            引号 = 字符; 当前.append(字符); continue
        if 字符 in ("{", "["):
            深度 += 1
        elif 字符 in ("}", "]"):
            深度 = max(0, 深度 - 1)
        if 字符.isspace() and 深度 == 0:
            if 当前:
                参数组.append("".join(当前)); 当前 = []
            continue
        当前.append(字符)
    if 当前:
        参数组.append("".join(当前))
    return 参数组


def 有可翻字符(文本) -> bool:
    return bool(re.search(r'[A-Za-z\u4e00-\u9fff]', str(文本)))


# ↓这些路径段下面的 name/title 是指向记分板/谓词/皮肤等技术字段的引用, 不是给人看的字
技术路径段 = {'score', 'target', 'objective', 'criteria', 'predicate', 'predicates',
              'custom_data', 'properties', 'profile'}


def 可取展示文本(文本: str) -> bool:
    """数据包 JSON 里的 name/description 多是注册名, 翻了会破坏数据包, 只留像给人看的。"""
    if not 有可翻字符(文本):
        return False
    if re.match(r'^#[A-Za-z0-9_.\-]+$', 文本):                    # 记分板假名 #temp / #search
        return False
    if re.match(r'^[a-z0-9_.\-/]+$', 文本):                        # textures / cold_resist_module
        return False
    if re.match(r'^[A-Za-z0-9_.\-]+:[A-Za-z0-9_./\-]+$', 文本):    # minecraft:jigsaw / 命名空间 ID
        return False
    return True


def 收集组件文本(对象) -> List[str]:
    """递归收集文本组件里真正显示的文字; translate 是语言键引用, 键本身不能翻。"""
    提取 = []
    if isinstance(对象, dict):
        if isinstance(对象.get('text'), str) and 有可翻字符(对象['text']):
            提取.append(对象['text'])
        if 'extra' in 对象:
            提取.extend(收集组件文本(对象['extra']))
    elif isinstance(对象, list):
        for 元素 in 对象:
            if isinstance(元素, str):
                if 有可翻字符(元素):
                    提取.append(元素)
            else:
                提取.extend(收集组件文本(元素))
    elif isinstance(对象, str) and 有可翻字符(对象):
        提取.append(对象)
    return 提取


def 尝试提取json文本(Self, 参数: str) -> List[str]:
    """tellraw/title 的参数可能是单个组件, 也可能是组件数组(内嵌裸字符串), 都收下来。"""
    try:
        对象 = json.loads(参数)
    except (json.JSONDecodeError, TypeError):
        return []
    return 收集组件文本(对象)


def 提取nbt文本(Self, nbt参数: str, 路径: str):
    # ↓give 的 NBT 参数带物品 id 前缀(如 minecraft:stone{...}) 从第一个 { 开始算
    起始 = nbt参数.find('{')
    if 起始 < 0 or not nbt参数.endswith('}'):
        return None
    nbt参数 = nbt参数[起始:]
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


def 收集JSON条目(Self, 结果: list, 文件路径, 路径: list, 值) -> None:
    """JSON 分支的守门人: 只收像给人看的文本, 注册名一类翻了会破坏数据包。

    文本组件(dict)必须原样带结构交给核心, 所以这里只决定收不收, 不改写值。
    """
    if any(str(段).lower() in 技术路径段 for 段 in 路径[:-1]):
        return
    if isinstance(值, dict):
        if 收集组件文本(值):
            结果.append([[文件路径, 路径], str(值)])
        return
    文本 = str(值)
    if 可取展示文本(文本.strip()):
        结果.append([[文件路径, 路径], 文本])


def 还原文本(译文: str):
    """译文多半是 JSON, 但文本组件在提取时是 Python 字面量, 那种写法也认。"""
    try:
        return json.loads(译文)
    except Exception:
        pass
    try:
        对象 = ast.literal_eval(译文)
    except Exception:
        return 译文
    return 对象 if isinstance(对象, (dict, list)) else 译文


def 可直接内联(文本: str) -> bool:
    return not re.search(r'["\\\x00-\x1f]', str(文本))


def 替换文本(行内容: str, 原文: str, 译文: str) -> str:
    候选 = [(原文, 译文)]
    # ↓命令里的 JSON 可能把原文写成转义形式(\" \u00a7 之类) 那就按同样的转义写法替换
    for 转义 in (True, False):
        转义原文 = json.dumps(原文, ensure_ascii=转义)[1:-1]
        if 转义原文 != 原文:
            # ↓译文能直接放进 JSON 就写明文 免得整行变成 \uXXXX
            替换值 = 译文 if 可直接内联(译文) else json.dumps(译文, ensure_ascii=转义)[1:-1]
            候选.append((转义原文, 替换值))
    for 模式, 替换值 in 候选:
        if 模式 in 行内容:
            return 行内容.replace(模式, 替换值)
    return 行内容


class DataPackParser(Parser):

    def 读取(Self, 文件路径: str) -> list:
        结果 = []
        文件路径 = Path(文件路径)
        if 文件路径.suffix in [".json", ".mcmeta"]:
            try:
                数据 = Self.读取Json文件(文件路径)
                Self.递归提取文本(数据, [], {"description", "title", "name", "subtitle"},
                                  lambda 路径, 值: 收集JSON条目(Self, 结果, 文件路径, 路径, 值),
                                  允许值类型=(str, dict))
            except Exception: Self.日志("log.module.data.load.error", file=文件路径, e=eb.format_exc(),  info_level=2)
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
                        提取到的文本列表 = 提取待翻译文本(Self, 行, 规则)
                        for 文本 in 提取到的文本列表:
                            if not 文本:
                                continue
                            # ↓第三位带上原文 写回时才能定位替换 框架会原样透传
                            结果.append([[文件路径, f"第{行号}行"], 文本, 文本])
            except Exception:
                Self.日志("log.module.data.load.error", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果

    def 写回(Self, index: list) -> None:
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
                原文 = 条目[2] if len(条目) > 2 else ""
                if not 原文 or not 译文:
                    continue
                文件映射.setdefault(文件路径, []).append(("mcfunction", 行描述, 原文, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            if 文件路径.suffix in [".json", ".mcmeta"]:
                try:
                    数据 = Self.读取Json文件(文件路径)
                except Exception: Self.日志("log.module.data.load.error", file=文件路径, e=eb.format_exc(), info_level=2); continue
                for 类型, 路径, 译文 in 替换列表:
                    try:
                        if isinstance(译文, str):
                            译文 = 还原文本(译文)
                        Self.设置路径值(数据, 路径, 译文)
                    except Exception:
                        Self.日志("log.module.data.write.error",
                                    file=文件路径, path=str(路径),
                                    e=eb.format_exc(), info_level=2)
                Self.写入Json文件(文件路径, 数据)
            elif 文件路径.suffix == ".mcfunction":
                try:
                    with open(文件路径, "r", encoding="utf-8") as f:
                        所有行 = f.readlines()
                except Exception:
                    Self.日志("log.module.data.load.error", file=文件路径, e=eb.format_exc(), info_level=2)
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
                        行内容 = 替换文本(行内容, 原文, 译文)
                    所有行[行号 - 1] = 行内容
                with open(文件路径, "w", encoding="utf-8", newline="") as f:
                    f.writelines(所有行)


Mods().注册翻译类型("DataPack", 执行, 解析器类=DataPackParser, 显示名="DataPack")


Mods().注册翻译类型到归档("DataPack", ".", 注册路径="/", 打包扩展名={".json", ".mcmeta", ".mcfunction"},
                     容器="数据包")

# 容器探测: 数据包 = pack.mcmeta + data/
Mods().注册容器("数据包", 80, lambda v, s: v.有浅层文件("pack.mcmeta") and v.有浅层前缀("data"))
