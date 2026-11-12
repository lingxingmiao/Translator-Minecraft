# GuideME 指南书(markdown) 翻译
#
# GuideME 是 AE2 团队那套指南框架: 页面是 <指南根>/*.md, 译文放在同级 _<语言>/ 目录下, 例如
#   assets/modern_industrialization/mi_guidebook/io_guide.md          ← 默认语言(原文)
#   assets/modern_industrialization/mi_guidebook/_zh_cn/io_guide.md   ← 中文(游戏语言 zh_cn 时优先加载)
# 所以本模组不覆盖原文, 而是把译文写成 _<目标语言>/ 下的镜像文件(游戏里语言一换就能来回切)。
#
# 指南根怎么认: 先找含 index.md 的目录; 找不到就退回 assets/<命名空间>/<第一层> ——
# 附属模组常常往别的模组的指南里加页面(industrialization_overdrive 就是往 MI 的 mi_guidebook 加),
# 这种包里根本没有 index.md, 只能靠这一层兜底。
#
# 只有"assets/ 下带 YAML front matter"的 md 才当成 GuideME 页面, 所以 LICENSE.md / CHANGELOG.md /
# assets/fancymenu/credits_and_copyright.md 这类不会被误翻。
#
# 翻译时保护这些内容, 免得把文档结构翻坏:
#   ``` 围栏代码块 · `行内代码` · <ItemLink id="..." /> 这类 GuideME/HTML 标签
#   链接与图片的地址 ](url) · 裸链接 · 反斜杠转义
#   front matter 里除 navigation.title 之外的键(icon / parent / position / item_ids 原样保留)
#
# 可用配置(覆盖键 MOD_TRANS_GUIDEME_<键>, 全大写):
#   READ_CONCURRENT   读取并发   4
#   WRITE_CONCURRENT  写入并发   4
#   LANGUAGE_DIR      译文目录名, 默认 "_" + Config.LANGUAGE_OUTPUT(即 _zh_cn)

from TranslatorLib import (zipfile, json, Path, Parser, Mods, eb, re)

模组参数 = Mods().模组配置("trans-guideme", {
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
    "LANGUAGE_DIR": "",
}, 索引指纹=False)


# 语言目录: _zh_cn / _en_us / _pt_br / _es_mx ... GuideME 靠它区分译文
语言目录模式 = re.compile(r'^_[a-z]{2,3}(?:_[a-z0-9]{2,4})?$')

# 行内保护片段: 命中即原样保留, 不送翻译
保护模式 = re.compile(
    r'`[^`]*`'                                          # `行内代码`
    r'|!\[|\[|\]\([^()\s]*(?:\([^()]*\)[^()\s]*)*\)'    # 链接/图片的标记与地址
    r'|<[^<>\n]*>'                                      # <ItemLink id="..." /> 这类标签
    r'|https?://\S+'                                    # 裸链接
    r'|\\.'                                             # 反斜杠转义
)

# 行首标记: # 标题 / - 列表 / 1. 有序 / > 引用 —— 标记本身不送翻译
行首标记模式 = re.compile(r'^[ \t]{0,3}(?:#{1,6}[ \t]+|>[ \t]?|[-*+][ \t]+|\d+[.)][ \t]+)?')

# 命名空间:路径 形式的标识符(industrialization_overdrive:terminal)不翻
标识符模式 = re.compile(r'^[a-z0-9_.\-]+:[a-z0-9_./\-]+$')

压缩包后缀集 = {".jar", ".zip"}


def 是语言目录(段) -> bool:
    return bool(语言目录模式.match(str(段)))


def 在语言目录(路径) -> bool:
    """只看 assets/ 以内的路径段 —— 否则缓存目录 Cache/_tmp 这种名字会被误判成语言目录。"""
    段列表 = [str(段) for 段 in Path(路径).parts]
    小写 = [段.lower() for 段 in 段列表]
    if "assets" in 小写:
        段列表 = 段列表[小写.index("assets") + 1:]
    return any(是语言目录(段) for 段 in 段列表)


def 头部有导航(头部: str) -> bool:
    """头部是 GuideME 页面的特征: 开头 --- 且 front matter 里有 navigation:"""
    行列表 = 头部.split("\n")
    if not 行列表 or 行列表[0].strip() != "---":
        return False
    for 行 in 行列表[1:60]:
        if 行.strip() == "---":
            return False
        if 行.strip().startswith("navigation:"):
            return True
    return False


def 目标语言目录(Self) -> str:
    配置值 = Mods().取值(Self, ("trans-guideme", "LANGUAGE_DIR"))
    if 配置值:
        配置值 = str(配置值)
        return 配置值 if 配置值.startswith("_") else f"_{配置值}"
    return f"_{Self.Config.LANGUAGE_OUTPUT}"


def 找指南根(文件路径) -> Path:
    """md 文件 → 它所属的指南根(译文目录 _<语言> 就插在这一层下面)。"""
    文件路径 = Path(文件路径)
    命中 = []
    当前 = 文件路径.parent
    while 当前.name and 当前.name.lower() != "assets":
        try:
            if (当前 / "index.md").is_file():
                命中.append(当前)
        except OSError:
            pass
        父 = 当前.parent
        if 父 == 当前:
            break
        当前 = 父
    if 命中:
        return 命中[-1]                      # 最靠近 assets/ 的那层
    for 祖先 in 文件路径.parents:            # 兜底: assets/<命名空间>/<第一层>
        if 祖先.name.lower() != "assets":
            continue
        try:
            相对段 = 文件路径.relative_to(祖先).parts
        except ValueError:
            break
        if len(相对段) >= 3:
            return 祖先 / 相对段[0] / 相对段[1]
        if len(相对段) == 2:
            return 祖先 / 相对段[0]
        break
    return 文件路径.parent


def 译文路径(Self, 文件路径) -> Path:
    文件路径 = Path(文件路径)
    指南根 = 找指南根(文件路径)
    return 指南根 / 目标语言目录(Self) / 文件路径.relative_to(指南根)


def 发现指南根(根) -> list:
    """目录(压缩包解压根)里所有指南根 —— 走 找指南根, 它同时管 index.md 和兜底两种情况。"""
    根 = Path(根)
    结果 = []
    for 文件 in sorted(根.rglob("*.md")):
        if 在语言目录(文件):
            continue                       # _<语言>/ 里的是译文, 不是原文
        指南根 = 找指南根(文件)
        if 指南根 not in 结果:
            结果.append(指南根)
    return 结果


def 压缩包指南名列表(压缩包) -> list:
    """压缩包里所有"assets/ 下、不在 _<语言>/ 里、带 front matter"的 md。"""
    命中 = []
    for 名称 in 压缩包.namelist():
        if 名称.endswith("/"):
            continue
        段 = 名称.split("/")
        if not 段 or 段[0].lower() != "assets" or not 名称.lower().endswith(".md"):
            continue
        if any(是语言目录(单段) for 单段 in 段[:-1]):
            continue
        try:
            头部 = 压缩包.read(名称)[:2048].decode("utf-8", errors="ignore")
        except Exception:
            continue
        if 头部有导航(头部):
            命中.append(名称)
    return 命中


def 解压指南(Self, 压缩源, 解压目录) -> int:
    """把压缩包里的 GuideME 页面按原路径解压出来(含 index.md, 指南根判定要用)。"""
    解压目录 = Path(解压目录)
    计数 = 0
    try:
        with zipfile.ZipFile(压缩源, "r") as 压缩包:
            名称列表 = set(压缩包指南名列表(压缩包))
            # index.md 本身可能没有 navigation:, 但指南根判定要它, 一并捞出来
            for 名称 in 压缩包.namelist():
                if 名称.endswith("/") or not 名称.lower().endswith("index.md"):
                    continue
                段 = 名称.split("/")
                if 段 and 段[0].lower() == "assets" and not any(是语言目录(单段) for 单段 in 段[:-1]):
                    名称列表.add(名称)
            for 名称 in sorted(名称列表):
                目标 = 解压目录 / 名称
                if 目标.exists():
                    continue
                目标.parent.mkdir(parents=True, exist_ok=True)
                目标.write_bytes(压缩包.read(名称))
                计数 += 1
    except Exception:
        Self.日志("log.module.book.load.error", mod="GuideME", file=str(压缩源),
                 e=eb.format_exc(), info_level=2)
    return 计数


def 行内可翻片段(行: str, 偏移: int):
    """一行里所有"没被标记包住"的可翻片段 → [(起始, 结束, 文本), ...](偏移是行首标记长度)。"""
    片段列表 = []
    位置 = 偏移
    for 匹配 in 保护模式.finditer(行, 偏移):
        起始, 结束 = 匹配.span()
        if 起始 > 位置:
            片段列表.append((位置, 起始, 行[位置:起始]))
        位置 = max(位置, 结束)
    if 位置 < len(行):
        片段列表.append((位置, len(行), 行[位置:]))
    return 片段列表


def 可翻译(文本: str) -> bool:
    if not 文本 or not 文本.strip():
        return False
    if not re.search(r'[A-Za-z]', 文本):
        return False
    return not 标识符模式.match(文本.strip())


def 过滤片段(条目) -> bool:
    return 可翻译(条目[1])


class GuideMEParser(Parser):

    def 读取(Self, 文件路径: str) -> list:
        结果 = []
        路径 = Path(文件路径)
        if 路径.suffix.lower() != ".md":
            return 结果
        if 在语言目录(路径):
            return 结果                      # 已经在 _<语言>/ 里的, 那是译文, 不当原文
        try:
            with open(路径, "r", encoding="utf-8") as f:
                原文 = f.read()
        except Exception:
            Self.日志("log.module.book.load.error", mod="GuideME", file=文件路径,
                     e=eb.format_exc(), info_level=2)
            return 结果
        行列表 = 原文.split("\n")
        # front matter 区间(首行 --- 到下一个 ---); 没有就是普通 md, 不接
        头信息结束行 = -1
        if 行列表 and 行列表[0].strip() == "---":
            for 行号 in range(1, len(行列表)):
                if 行列表[行号].strip() == "---":
                    头信息结束行 = 行号
                    break
        if 头信息结束行 < 0 or not any(行.strip().startswith("navigation:")
                                        for 行 in 行列表[1:头信息结束行]):
            return 结果
        在代码块 = False
        for 行号, 原始行 in enumerate(行列表):
            行 = 原始行[:-1] if 原始行.endswith("\r") else 原始行
            if 0 <= 行号 <= 头信息结束行:
                if 行号 not in (0, 头信息结束行):
                    Self.收集头信息(结果, 路径, 行号, 行)
                continue
            去空白 = 行.strip()
            if 去空白.startswith("```") or 去空白.startswith("~~~"):
                在代码块 = not 在代码块
                continue
            if 在代码块 or not 去空白:
                continue
            Self.收集行(结果, 路径, 行号, 行)
        return 结果

    def 收集头信息(Self, 结果, 路径, 行号, 行):
        """front matter 只翻 navigation.title, 其余键(icon/parent/position/item_ids)原样。"""
        匹配 = re.match(r'^(\s*title:\s*)(.*)$', 行)
        if not 匹配:
            return
        原始值 = 匹配.group(2)
        值 = 原始值.strip()
        起始 = 匹配.start(2) + (len(原始值) - len(原始值.lstrip()))
        if len(值) >= 2 and 值[0] == 值[-1] and 值[0] in "\"'":
            文本 = 值[1:-1]
            起始 += 1                          # 跳过开引号, 只翻引号里的内容
        else:
            文本 = 值
        if not 可翻译(文本):
            return
        结果.append([[路径, 行号, 起始, 起始 + len(文本)], 文本])

    def 收集行(Self, 结果, 路径, 行号, 行):
        偏移 = 行首标记模式.match(行).end()
        for 起始, 结束, 片段 in 行内可翻片段(行, 偏移):
            if not 可翻译(片段):
                continue
            结果.append([[路径, 行号, 起始, 结束], 片段])

    def 写回(Self, 翻译列表: list) -> None:
        if not 翻译列表:
            return
        文件映射 = {}
        for 条目 in 翻译列表:
            位置, 译文 = 条目[0], 条目[1]
            文件映射.setdefault(str(位置[0]), []).append((位置[1], 位置[2], 位置[3], 译文))
        for 源路径, 替换列表 in 文件映射.items():
            Self.写出译文文件(源路径, 替换列表)

    def 写出译文文件(Self, 源路径, 替换列表):
        源路径 = Path(源路径)
        try:
            with open(源路径, "r", encoding="utf-8") as f:
                原文 = f.read()
        except Exception:
            Self.日志("log.module.book.load.error", mod="GuideME", file=str(源路径),
                     e=eb.format_exc(), info_level=2)
            return
        行列表 = 原文.split("\n")
        行替换 = {}
        for 行号, 起始, 结束, 译文 in 替换列表:
            行替换.setdefault(行号, []).append((起始, 结束, str(译文).strip()))
        for 行号, 片段列表 in 行替换.items():
            if 行号 < 0 or 行号 >= len(行列表):
                continue
            行 = 行列表[行号]
            # 从后往前替换, 免得前面的改动把后面的偏移顶掉
            for 起始, 结束, 译文 in sorted(片段列表, key=lambda 项: -项[0]):
                if 起始 < 0 or 结束 > len(行) or 起始 > 结束:
                    continue
                行 = 行[:起始] + 译文 + 行[结束:]
            行列表[行号] = 行
        目标路径 = 译文路径(Self, 源路径)
        try:
            目标路径.parent.mkdir(parents=True, exist_ok=True)
            with open(目标路径, "w", encoding="utf-8", newline="") as f:
                f.write("\n".join(行列表))
        except Exception:
            Self.日志("log.module.book.write.error", mod="GuideME", file=str(目标路径),
                     e=eb.format_exc(), id="", field="", info_level=2)


def 翻译一个指南根(Self, 指南根, **参数):
    解析 = GuideMEParser(Self)
    Self.翻译流程(指南根, "*.md", 解析.读取, 过滤片段, lambda x: str(x[0][0]), 解析.写回,
                Mods().取值(Self, ("trans-guideme", "READ_CONCURRENT")),
                Mods().取值(Self, ("trans-guideme", "WRITE_CONCURRENT")), "book", **参数)


def 打包语言包与指南(Self, 压缩源, 工作目录, 指南目录, 语言包, 语言):
    """语言文件产物 + 指南 _<语言>/ 译文 → 合成一个资源包 zip。"""
    条目 = {}
    语言包路径 = Path(语言包) if 语言包 else None
    if 语言包路径 and 语言包路径.is_file() and 语言包路径.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(语言包路径) as 包:
                for 名称 in 包.namelist():
                    if not 名称.endswith("/"):
                        条目[名称] = 包.read(名称)
        except Exception:
            Self.日志("log.module.book.load.error", mod="GuideME", file=str(语言包路径),
                     e=eb.format_exc(), info_level=2)
    for 文件 in 指南目录.rglob("*.md"):
        相对段 = 文件.relative_to(指南目录).parts
        if any(是语言目录(段) for 段 in 相对段):
            条目["/".join(相对段)] = 文件.read_bytes()
    if not 条目:
        return None
    if "pack.mcmeta" not in 条目:
        try:
            描述 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(
                name=压缩源.stem, lang=语言, model=Self.Lang("log.core.package.zip.hit"),
                author=Self.Config.PACK_AUTHOR or "海盐青茫")
        except Exception:
            描述 = f"{压缩源.stem} {语言}"
        条目["pack.mcmeta"] = json.dumps({"pack": {
            "description": 描述, "pack_format": 9999, "supported_formats": [0, 9999],
            "min_format": 0, "max_format": 9999}}, ensure_ascii=False, indent=4).encode("utf-8")
    最终 = 工作目录 / f"{压缩源.stem}-{语言}.zip"
    try:
        with zipfile.ZipFile(最终, "w", zipfile.ZIP_DEFLATED) as 包:
            for 名称, 数据 in 条目.items():
                包.writestr(名称, 数据)
    except Exception:
        Self.日志("log.module.book.write.error", mod="GuideME", file=str(最终),
                 e=eb.format_exc(), id="", field="", info_level=2)
        return None
    return str(最终)


def 翻译压缩包(Self, 压缩源, 输出目录, **参数):
    语言 = Self.Config.LANGUAGE_OUTPUT
    工作目录 = Path(输出目录) if 输出目录 else Path(f"{Self.Config.PATH_CACHE}/{Self.Module.uuid()}")
    工作目录.mkdir(parents=True, exist_ok=True)
    指南目录 = 工作目录 / f"guideme_{Self.Module.uuid()}"
    if not 解压指南(Self, 压缩源, 指南目录):
        return None                           # 不是指南包, 交回核心的常规流程
    for 指南根 in 发现指南根(指南目录):
        try:
            翻译一个指南根(Self, 指南根, **参数)
        except Exception:
            Self.日志("log.module.book.write.error", mod="GuideME", file=str(指南根),
                     e=eb.format_exc(), id="", field="", info_level=2)
    # 语言文件照旧翻(这个包整个交给了我们, 不能把原本的 .lang/.json 漏掉)
    语言包 = None
    try:
        透传 = {键: 参数[键] for 键 in ("索引ID", "使用模型", "保存语言") if 键 in 参数}
        语言包 = Self.翻译语言文件(file0=str(压缩源), output_path=str(工作目录),
                               read_error=False, **透传)
    except Exception:
        Self.日志("log.module.book.load.error", mod="GuideME", file=str(压缩源),
                 e=eb.format_exc(), info_level=2)
    return 打包语言包与指南(Self, 压缩源, 工作目录, 指南目录, 语言包, 语言)


def 执行(Self, 根, 输出目录=None, path2=None, **参数):
    根 = Path(根)
    if 根.is_dir():
        产物 = None
        for 指南根 in 发现指南根(根):
            try:
                翻译一个指南根(Self, 指南根, path2=path2, **参数)
            except Exception:
                Self.日志("log.module.book.write.error", mod="GuideME", file=str(指南根),
                         e=eb.format_exc(), id="", field="", info_level=2)
                continue
            产物 = 指南根
        return str(产物) if 产物 else None
    if not 根.is_file():
        return None
    if 根.suffix.lower() == ".md":
        try:
            翻译一个指南根(Self, 根, path2=path2, **参数)
        except Exception:
            Self.日志("log.module.book.write.error", mod="GuideME", file=str(根),
                     e=eb.format_exc(), id="", field="", info_level=2)
            return None
        return str(译文路径(Self, 根))
    if 根.suffix.lower() in 压缩包后缀集:
        return 翻译压缩包(Self, 根, 输出目录, path2=path2, **参数)
    return None


def 探测指南(Self, 路径) -> bool:
    """压缩包看有没有 GuideME 页面; 单独上传的 .md 看它自己是不是 GuideME 页面。"""
    路径 = Path(路径)
    后缀 = 路径.suffix.lower()
    if 后缀 == ".md":
        try:
            with open(路径, "r", encoding="utf-8", errors="ignore") as f:
                return 头部有导航(f.read(2048))
        except Exception:
            return False
    if 后缀 not in 压缩包后缀集:
        return False
    try:
        with zipfile.ZipFile(路径) as 压缩包:
            return bool(压缩包指南名列表(压缩包))
    except Exception:
        return False


Mods().注册翻译类型("GuideME", 执行, 解析器类=GuideMEParser, 显示名="GuideME 指南",
                文件探测=探测指南, 探测分数=85)


Mods().注册翻译类型到模组("GuideME", "assets", 注册路径="/assets", 打包扩展名={".md"}, 最大深度=2)


Mods().注册翻译类型到归档("GuideME", "assets", 注册路径="/assets", 打包扩展名={".md"}, 最大深度=2)
