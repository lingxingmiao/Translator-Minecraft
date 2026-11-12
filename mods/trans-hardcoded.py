# 硬编码扫描 → Vault Patcher 模块
#
# 扫描 mod jar 里 .class 的字节码，只挑「真正被 LDC 加载、且用在界面/提示文本」的字符串，
# 翻译后写成 vaultpatcher/modules/<modid>-hardcoded.json（按方法定位）。
# 不修改原 jar，也不改 class；Vault Patcher 在类加载时按 LDC 精确替换。
#
# 为什么按方法定位：同一个字符串可能既做界面文本又做注册名/存档键/NBT 键，
# 只按类替换会把逻辑用的那一份也换掉，导致模组行为异常。写入 info.method 后，
# Vault Patcher 只替换该方法里的 LDC，逻辑方法不受影响。
#
# 可用配置（覆盖键 MOD_TRANS_HARDCODED_<键>，全大写）：
#   ENABLE              是否启用                   True
#   MAX_STRINGS         每个 jar 最多处理多少条    800
#   MIN_LENGTH          最短原文长度              3
#   MAX_LENGTH          最长原文长度              200
#   INCLUDE_SINGLE_WORDS 是否包含单词标签          True
#   MERGE_MODULES       多模组合并为一个模块       True
#   UI_ONLY             只收界面/提示文本          True
#                       （物品名/工具提示/按钮/提示框；关掉会扫描日志和调试串）
#   TARGET_METHOD       按方法定位替换              True
#   INCLUDE_CLINIT      包含静态初始化字符串       True
#   VALIDATE_FORMAT     校验占位符/颜色码/换行     True
#   BLACKLIST           额外跳过词（子串匹配）     []
#   GUI_ONLY            只翻 GUI/容器/提示类       False
#   CLASS_EXCLUDE       额外跳过类正则（可留空）   ""
#   TRANSLATORVAULT_JAR TranslatorVault 文件名     ""（可选，默认不带）
#   VAULTPATCHER_JAR    VaultPatcher 文件名        vaultpatcher-1.5.4.jar
#
# LIB_PATH 只定义文件夹（默认 ./libs）；VaultPatcher 必带，TranslatorVault 按需。
# VaultPatcher 复制到 <实例>/mods/；配置写入 <实例>/config/vaultpatcher_asm/config.json，
# 并确保 load_all_modules=true，否则 external vaultpatcher/modules 不会被加载。

import json
import os
import re
import shutil
import struct
import zipfile
from collections import Counter
from pathlib import Path

from TranslatorLib import (Loop, Mods, eb)

参数 = Mods().模组配置("trans-hardcoded", {
    "ENABLE": True,
    "MAX_STRINGS": 800,
    "MIN_LENGTH": 3,
    "MAX_LENGTH": 200,
    "INCLUDE_SINGLE_WORDS": True,
    "MERGE_MODULES": True,
    "UI_ONLY": True,
    "TARGET_METHOD": True,
    "INCLUDE_CLINIT": True,
    "VALIDATE_FORMAT": True,
    "BLACKLIST": [],
    "GUI_ONLY": False,
    "CLASS_EXCLUDE": "",
    "TRANSLATORVAULT_JAR": "",
    "VAULTPATCHER_JAR": "vaultpatcher-1.5.4.jar",
}, 索引指纹=False)

_排除后缀 = (
    ".class", ".java", ".json", ".png", ".jpg", ".jpeg", ".gif", ".ogg", ".wav",
    ".nbt", ".toml", ".cfg", ".txt", ".lang", ".properties", ".xml", ".yml",
    ".yaml", ".mf", ".dll", ".so", ".dylib", ".zip", ".jar",
)
_常见类型 = {
    "String", "Object", "Integer", "Long", "Float", "Double", "Boolean", "Byte",
    "Short", "Character", "Void", "Class", "List", "Map", "Set", "Optional",
    "Override", "SuppressWarnings", "Deprecated", "FunctionalInterface",
}
_固定跳过 = {
    "true", "false", "null", "UTF-8", "ISO-8859-1", "US-ASCII", "UTF8",
    "get", "set", "put", "add", "remove", "equals", "hashCode", "toString",
    "Code", "LineNumberTable", "SourceFile", "Exceptions", "InnerClasses",
    "Signature", "StackMapTable", "ConstantValue", "LocalVariableTable",
    "LocalVariableTypeTable", "RuntimeVisibleAnnotations", "RuntimeInvisibleAnnotations",
    "RuntimeVisibleParameterAnnotations", "RuntimeInvisibleParameterAnnotations",
    "AnnotationDefault", "EnclosingMethod", "BootstrapMethods", "MethodParameters",
    "NestHost", "NestMembers", "Record", "PermittedSubclasses",
}
_译文缓存 = {}
_跳过类前缀 = (
    "org/apache/", "com/google/", "net/fabricmc/", "org/spongepowered/",
    "it/unimi/", "com/mojang/", "org/objectweb/", "kotlin/", "scala/",
)
# 全大写常量名(TAG_AMOUNT/MODID/BRANCH)不是界面文本, 每个 jar 一批
_常量名 = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
# 资源 ID(forestry:sapling / minecraft:stone)
_资源ID = re.compile(r"^[a-z0-9_.\-]+:[a-z0-9_./\-]+$")
# Java 标识符式(RSChannel1 / RSCounter / FluidAction): 无空格、内部还有大写
_标识符 = re.compile(r"^(?:[A-Z]{2,}[A-Za-z0-9]*|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)$")
# 小写 snake_case / 点分键(modid, config.key, some_flag)
_小写键 = re.compile(r"^[a-z0-9_]+(?:[.:][a-z0-9_\-]+)+$|^[a-z0-9]+(?:_[a-z0-9]+)+$")
# 纯 ASCII 资源路径式
_路径式 = re.compile(r"^[A-Za-z0-9_.\-/]+$")
# 翻译键/格式占位符
_占位符 = re.compile(r"%(?:\d+\$)?[-#+ 0,(]*\d*(?:\.\d+)?[tT]?[sdfegxXoObBcC%]")
# 技术/底层类：清单压缩、校验、签名、混淆/重映射等，整类都不碰
_技术类 = re.compile(
    r"(?i)manifest|compactor|checksum|digest|hash|signature|security|crypto|"
    r"obfuscat|remap|relocat|packer|repack"
)
# 渲染/着色器链：字符串常被当 shader 名、rendertype、uniform 或注入点匹配用
_渲染类 = re.compile(
    r"(?i)iris|oculus|shader|rendertype|render|postchain|gpu|glsl|optifine|"
    r"flywheel|vulkan|opengl|renderdoc|mesa"
)
# Sodium/Embeddium 只保留 GUI 相关类，渲染/区块类一律跳过
_钠类 = re.compile(r"(?i)sodium|embeddium")
_界面词 = re.compile(
    r"(?i)gui|screen|container|tooltip|menu|button|label|widget|overlay|hud|"
    r"inventory|config|option|extras|setting|preference"
)
# 日志/崩溃/调试类与文本：默认不翻，避免刷屏和误伤
_非展示类 = re.compile(
    r"(?i)crashreport|logger|logging|debug|dump|validator|assert|exception|"
    r"errorhandler|stacktrace|profiler|compatibilitycheck"
)
_诊断文本 = re.compile(
    r"(?i)^(?:failed|could not|couldn't|unable|error|exception|invalid|illegal|"
    r"unexpected|missing|cannot|can't|must |should |warning|warn|debug|trace|"
    r"assert|unknown|unsupported|corrupt|overflow|underflow|attempt|encountered|"
    r"caught|skipping|ignored|loaded|loading|saving|saved|wrote|read|found|"
    r"scanning|registered|created|initializ|not a valid|out of|no longer|"
    r"is missing|is not|was not|does not|- )"
)
_诊断方法词 = (
    "log", "logger", "error", "warn", "debug", "trace", "crash", "dump",
    "report", "validate", "assert", "print", "tostring", "fillcrash", "stacktrace",
)
# JAR 清单/HTTP 头等键：一旦翻译会让清单或协议解析失效
_技术键词 = (
    "Digest", "Manifest-Version", "Created-By", "Main-Class", "Class-Path",
    "Implementation-", "Specification-", "Content-Type", "Content-Length",
    "Multi-Release", "Automatic-Module-Name", "Sealed", "Magic",
    "Last-Modified", "Accept-Ranges", "ETag", "SHA-", "MD5-",
    "Launcher-Agent-Class", "Premain-Class", "Agent-Class", "Can-Redefine",
    "Can-Retransform", "Add-Exports", "Add-Opens", "Enable-Native-Access",
    "Bundle-", "Require-Capability", "Provide-Capability",
)
# 当前 jar 的类简单名集合(SettingsSetup/Serializer/I18nConstants), 由 _收集Jar/_收集目录 填
_类名集合 = set()


def _u1(数据, 位置):
    return 数据[位置]


def _u2(数据, 位置):
    return struct.unpack_from(">H", 数据, 位置)[0]


def _u4(数据, 位置):
    return struct.unpack_from(">I", 数据, 位置)[0]


def _s4(数据, 位置):
    return struct.unpack_from(">i", 数据, 位置)[0]


# 定长指令的字节数（变长/switch/wide 单独处理）
_定长 = {
    0x10: 2, 0x11: 3, 0x12: 2, 0x13: 3, 0x14: 3,
    0x15: 2, 0x16: 2, 0x17: 2, 0x18: 2, 0x19: 2,
    0x36: 2, 0x37: 2, 0x38: 2, 0x39: 2, 0x3a: 2,
    0x84: 3, 0xa9: 2,
    0xb2: 3, 0xb3: 3, 0xb4: 3, 0xb5: 3,
    0xb6: 3, 0xb7: 3, 0xb8: 3, 0xb9: 5, 0xba: 5,
    0xbb: 3, 0xbc: 2, 0xbd: 3, 0xc0: 3, 0xc1: 3,
    0xc5: 4, 0xc6: 3, 0xc7: 3, 0xc8: 5, 0xc9: 5,
}
for _指令 in range(0x99, 0xa9):
    _定长.setdefault(_指令, 3)


def _解析常量池(数据):
    """解析常量池；返回 (结束位置, utf8, 类索引, 字符串索引, 成员引用, 名字类型)。"""
    if len(数据) < 10 or 数据[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("not a class file")
    位置 = 8
    数量 = _u2(数据, 位置)
    位置 += 2
    utf8, 类索引, 字符串索引, 成员引用, 名字类型 = {}, {}, {}, {}, {}
    i = 1
    while i < 数量:
        标签 = _u1(数据, 位置)
        位置 += 1
        if 标签 == 1:  # Utf8
            长度 = _u2(数据, 位置)
            位置 += 2
            utf8[i] = 数据[位置:位置 + 长度].decode("utf-8", "replace")
            位置 += 长度
        elif 标签 == 7:  # Class
            类索引[i] = _u2(数据, 位置)
            位置 += 2
        elif 标签 == 8:  # String
            字符串索引[i] = _u2(数据, 位置)
            位置 += 2
        elif 标签 in (16, 19, 20):  # MethodType / Module / Package
            位置 += 2
        elif 标签 == 15:  # MethodHandle
            位置 += 3
        elif 标签 == 12:  # NameAndType
            名字类型[i] = (_u2(数据, 位置), _u2(数据, 位置 + 2))
            位置 += 4
        elif 标签 in (9, 10, 11):  # Fieldref / Methodref / InterfaceMethodref
            成员引用[i] = (_u2(数据, 位置), _u2(数据, 位置 + 2))
            位置 += 4
        elif 标签 in (3, 4, 17, 18):  # Int / Float / Dynamic / InvokeDynamic
            位置 += 4
        elif 标签 in (5, 6):  # Long / Double 占两个槽
            位置 += 8
            i += 1
        else:
            raise ValueError(f"bad constant tag {标签}")
        i += 1
    return 位置, utf8, 类索引, 字符串索引, 成员引用, 名字类型


def _解析调用(引用索引, utf8, 类索引, 成员引用, 名字类型):
    """把 CP 里的方法引用解析成 (owner, name)。"""
    引用 = 成员引用.get(引用索引)
    if not 引用:
        return None
    类项, 名项 = 引用
    所有者 = utf8.get(类索引.get(类项, -1), "")
    名字 = ""
    名字类型项 = 名字类型.get(名项)
    if 名字类型项:
        名字 = utf8.get(名字类型项[0], "")
    if not 所有者 or not 名字:
        return None
    return 所有者, 名字


def _解析代码(代码, utf8, 字符串索引, 类索引, 成员引用, 名字类型):
    """返回 [(opcode, 参数)]；LDC 参数是字符串，invoke 参数是 (owner, name)。"""
    if len(代码) < 8:
        return []
    代码长度 = _u4(代码, 4)
    if 代码长度 > len(代码) - 8:
        return []
    字节 = 代码[8:8 + 代码长度]
    指令 = []
    位置 = 0
    while 位置 < len(字节):
        操作码 = 字节[位置]
        参数 = None
        try:
            if 操作码 == 0x12:  # ldc
                索引 = 字节[位置 + 1]
                参数 = utf8.get(字符串索引.get(索引, -1), "")
            elif 操作码 in (0x13, 0x14):  # ldc_w / ldc2_w
                索引 = _u2(字节, 位置 + 1)
                if 操作码 == 0x13:
                    参数 = utf8.get(字符串索引.get(索引, -1), "")
            elif 操作码 in (0xb6, 0xb7, 0xb8, 0xb9):  # invoke*
                参数 = _解析调用(_u2(字节, 位置 + 1), utf8, 类索引, 成员引用, 名字类型)
        except Exception:
            参数 = None
        指令.append((操作码, 参数))

        if 操作码 in (0xaa, 0xab):  # tableswitch / lookupswitch
            对齐 = 位置 + 1
            while 对齐 % 4:
                对齐 += 1
            if 操作码 == 0xaa:
                低 = _s4(字节, 对齐 + 4)
                高 = _s4(字节, 对齐 + 8)
                位置 = 对齐 + 12 + 4 * max(0, 高 - 低 + 1)
            else:
                数量 = _s4(字节, 对齐 + 4)
                位置 = 对齐 + 8 + 8 * max(0, 数量)
        elif 操作码 == 0xc4:  # wide
            位置 += 6 if 位置 + 1 < len(字节) and 字节[位置 + 1] == 0x84 else 4
        else:
            位置 += _定长.get(操作码, 1)
    return 指令


def 解析类(数据):
    """返回 (内部类名, {方法名: {"desc", "strings": [(文本, 指令下标)], "calls": []}})"""
    try:
        位置, utf8, 类索引, 字符串索引, 成员引用, 名字类型 = _解析常量池(数据)
        本类索引 = _u2(数据, 位置 + 2)
        类名 = utf8.get(类索引.get(本类索引, -1), "")
        位置 += 6  # access_flags + this_class + super_class
        接口数 = _u2(数据, 位置)
        位置 += 2 + 2 * 接口数

        字段数 = _u2(数据, 位置)
        位置 += 2
        for _ in range(字段数):
            位置 += 6
            属性数 = _u2(数据, 位置)
            位置 += 2
            for _ in range(属性数):
                长度 = _u4(数据, 位置 + 2)
                位置 += 6 + 长度

        方法数 = _u2(数据, 位置)
        位置 += 2
        方法表 = {}
        for _ in range(方法数):
            方法名 = utf8.get(_u2(数据, 位置 + 2), "")
            方法描述 = utf8.get(_u2(数据, 位置 + 4), "")
            位置 += 6
            属性数 = _u2(数据, 位置)
            位置 += 2
            项 = 方法表.setdefault(方法名, {"desc": 方法描述, "strings": [],
                                           "calls": [], "_指令": []})
            for _ in range(属性数):
                属性名 = utf8.get(_u2(数据, 位置), "")
                属性长 = _u4(数据, 位置 + 2)
                起始 = 位置 + 6
                结束 = 起始 + 属性长
                if 属性名 == "Code" and 结束 <= len(数据):
                    指令列表 = _解析代码(数据[起始:结束], utf8, 字符串索引,
                                       类索引, 成员引用, 名字类型)
                    偏移 = len(项["_指令"])
                    for 下标, (操作码, 参数) in enumerate(指令列表):
                        if 操作码 in (0x12, 0x13) and isinstance(参数, str) and 参数:
                            项["strings"].append((参数, 偏移 + 下标))
                        elif isinstance(参数, tuple):
                            项["calls"].append(参数)
                    项["_指令"].extend(指令列表)
                位置 = 结束
        return 类名, 方法表
    except Exception:
        return None, {}


def _是展示调用(所有者, 名字):
    if not 所有者 or not 名字:
        return False
    if 名字 in ("drawString", "drawCenteredString", "drawWordWrap",
               "renderTooltip", "renderComponentTooltip", "renderTooltipFromComponents",
               "showMessageDialog", "showConfirmDialog", "showInputDialog"):
        return True
    if 所有者.endswith(("JOptionPane", "JDialog", "JFrame")):
        return True
    if 所有者.endswith(("TextComponent", "StringTextComponent", "TranslatableComponent",
                        "ChatComponentText", "TextComponent$Serializer")):
        return True
    if 所有者.endswith(("GuiGraphics", "Font", "AbstractWidget", "Button", "Screen")) \
            and 名字.startswith(("draw", "render")):
        return True
    if 所有者.startswith("net/minecraft/network/chat/") \
            and 名字 in ("literal", "translatable", "nullToEmpty", "empty", "append"):
        return True
    if 所有者.endswith(("Component", "MutableComponent")) \
            and 名字.startswith(("m_", "literal", "translatable", "append")):
        return True
    return False


def _是逻辑调用(所有者, 名字):
    if not 所有者:
        return False
    if 所有者 == "java/lang/String" and 名字 in (
            "equals", "equalsIgnoreCase", "hashCode", "startsWith", "endsWith",
            "contains", "matches", "split", "replace", "replaceAll", "indexOf"):
        return True
    if 名字 in ("equals", "hashCode", "compareTo") and 所有者 == "java/util/Objects":
        return True
    if 名字 in ("checkState", "checkArgument", "checkNotNull", "checkElementIndex",
               "checkPositionIndex", "verify", "requireNonNull", "requireNonNullElse"):
        return True
    if "Preconditions" in 所有者 or 所有者.endswith(("Validate", "Assertions")):
        return True
    if 所有者.startswith(("java/security/", "java/nio/charset/")) and 名字 in (
            "getInstance", "forName", "getMessageDigest"):
        return True
    if 所有者.startswith(("java/util/jar/", "java/util/zip/")):
        return True
    if 名字 in ("println", "print", "printf") and 所有者 in (
            "java/io/PrintStream", "java/io/PrintWriter"):
        return True
    if 所有者.startswith(("org/slf4j/", "org/apache/logging/")) or 所有者.endswith(("Logger", "LogManager")):
        return True
    if 名字 in ("get", "getOrDefault", "containsKey", "put", "remove", "computeIfAbsent") \
            and (所有者.endswith(("Map", "HashMap", "ConcurrentHashMap", "JsonObject"))):
        return True
    if 名字 in ("contains", "add", "remove") and 所有者.endswith(("Set", "HashSet", "List")):
        return True
    if 所有者.endswith(("ResourceLocation", "Identifier")) or "ResourceLocation" in 所有者:
        return True
    if 所有者.endswith(("CompoundTag", "ListTag", "StringTag", "NbtIo")) or "nbt" in 所有者.lower():
        return True
    if 所有者.endswith(("CommandSourceStack", "Commands", "CommandDispatcher")):
        return True
    if "Config" in 所有者 and 名字 in (
            "get", "getBoolean", "getInt", "getString", "getValue", "contains", "has",
            "set", "define", "defineInRange", "comment", "translation", "push", "pop"):
        return True
    if "Command" in 所有者 and 名字 in ("literal", "argument", "getString", "getEntityOrException"):
        return True
    if 所有者.endswith(("Exception", "Error")) or 名字 == "printStackTrace":
        return True
    return False


_展示方法名 = {
    "getName", "getDisplayName", "getDescription", "getDescriptionText", "getTitle",
    "getText", "getMessage", "getTooltip", "getTooltipLines", "appendHoverText",
    "addInformation", "addTooltip", "buildTooltip", "appendTooltip",
    "getNameComponent", "getHoverName", "getLocalizedName", "displayClientMessage",
    "sendSystemMessage", "sendMessage", "getLabel", "getDisplay", "getSubtitle",
    "getFormattedName",
}
_展示方法前缀 = (
    "render", "draw", "tooltip", "getName", "getDisplay", "getDesc", "getTitle",
    "getText", "getMessage", "appendHover", "addInfo", "getLabel", "getLocalized",
)
_逻辑方法名 = {
    "equals", "hashCode", "compareTo", "read", "write", "load", "save",
    "serialize", "deserialize", "parse", "register", "tick", "update",
}


def _方法可展示(方法名):
    return 方法名 in _展示方法名 or 方法名.startswith(_展示方法前缀)


def _类像界面(类名):
    低 = "/" + 类名.lower() + "/"
    return any(词 in 低 for 词 in (
        "gui", "screen", "widget", "tooltip", "hud", "overlay", "menu",
        "button", "label", "provider", "chat", "options", "config", "extras",
        "setting", "preference",
    ))


def _命中黑名单(文本):
    规则 = _参数("BLACKLIST")
    if not 规则:
        return False
    if isinstance(规则, str):
        规则 = [规则]
    try:
        return any(isinstance(项, str) and 项 and 项 in 文本 for 项 in 规则)
    except Exception:
        return False


def 像展示文本(文本):
    if not isinstance(文本, str):
        return False
    文本 = 文本.strip()
    if not (_参数("MIN_LENGTH") <= len(文本) <= _参数("MAX_LENGTH")):
        return False
    if _命中黑名单(文本):
        return False
    if 文本 in _固定跳过 or "\ufffd" in 文本:
        return False
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", 文本):
        return False
    if re.search(r"[\u4e00-\u9fff]", 文本):  # 已经是中文，不重复翻
        return False
    if not re.search(r"[A-Za-z]", 文本):  # 至少要有拉丁字母
        return False
    if _常量名.fullmatch(文本) or _资源ID.fullmatch(文本) or _标识符.fullmatch(文本):
        return False
    if _小写键.fullmatch(文本):
        return False
    if " " not in 文本 and "%" in 文本 and re.fullmatch(r"[A-Za-z0-9_.%\-]+", 文本):
        return False  # 带占位符的翻译键，如 modmenu.nameTranslation.%s
    if 文本 in _类名集合:
        return False
    if any(字符 in 文本 for 字符 in "\\;{}<>^$|"):
        return False
    if re.search(r"\[[^\]]*[0-9A-Za-z]\\?-[^\]]*\]", 文本):  # 正则字符组，如 [0-9A-F]
        return False
    if re.search(r"\(\?[a-zA-Z]*[:=!<]", 文本):  # 正则组标记，如 (?i) / (?=...)
        return False
    if re.search(r"\\[dDsSwWbB]", 文本):  # 正则转义，如 \d \s
        return False
    if any(字符 in 文本 for 字符 in ("=", "->", "::", "\t")):
        return False
    if "://" in 文本 or 文本.startswith(("http", "assets/", "data/", "META-INF/",
                                       "net/", "com/", "org/", "java/", "javax/", "/", "#", "@")):
        return False
    if 文本.startswith(("#version", "uniform ", "varying ", "attribute ", "precision ")):
        return False  # GLSL 片段
    if re.search(r"(?i)\b(?:gl_[a-z0-9_]+|sampler[a-z0-9_]*|mat[234]|vec[234])\b", 文本):
        return False  # 着色器类型/内置变量名
    if 文本.lower().endswith(_排除后缀):
        return False
    if re.fullmatch(r"[a-z0-9_.$:\-]+", 文本):  # 全小写的 id / 路径
        return False
    if _路径式.fullmatch(文本) and ("/" in 文本 or "." in 文本) and " " not in 文本:
        return False
    if len(re.findall(r"[A-Za-z]", 文本)) < 2:
        return False
    if " " not in 文本 and re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", 文本):
        段 = 文本.split("-")
        if all(部分.isdigit() or 部分.isupper() for 部分 in 段):
            return False  # SHA-512 / UTF-8 这类全大写编号
        if any(词.lower() in 文本.lower() for 词 in _技术键词):
            return False  # Manifest-Version / SHA-256-Digest 这类清单键
    if 文本.count("%") and not re.sub(_占位符, "", 文本).strip(" %sdfegxXoObBcC0123456789$.-+"):
        return False

    有空格 = " " in 文本
    有句读 = bool(re.search(r"[.!?…:;,。！？：；，]", 文本))
    if not (有空格 or 有句读):
        if not _参数("INCLUDE_SINGLE_WORDS"):
            return False
        if len(文本) < 4 or not 文本[0].isupper() or 文本 in _常见类型:
            return False
        if "_" in 文本 or "." in 文本:
            return False
    return True


def _参数(键):
    return 参数.get(键)


def _近邻命中(指令, 下标, 判定, 距离):
    for i in range(下标 + 1, min(len(指令), 下标 + 1 + 距离)):
        操作码, 参数 = 指令[i]
        if isinstance(参数, tuple) and 判定(参数[0], 参数[1]):
            return True
    return False


def _收集类文本(类名, 方法表):
    """按方法筛出界面文本，返回 {方法名: [文本...]}（已排序去重）。"""
    简单名 = 类名.rsplit("/", 1)[-1].split("$")[0]
    if _技术类.search(简单名) or _渲染类.search(类名) or _非展示类.search(简单名):
        return {}
    if _钠类.search(类名) and not _界面词.search(类名):
        return {}
    if _参数("GUI_ONLY") and not _界面词.search(类名):
        return {}
    排除正则 = str(_参数("CLASS_EXCLUDE") or "").strip()
    if 排除正则:
        try:
            if re.search(排除正则, 类名):
                return {}
        except re.error:
            pass
    结果 = {}
    界面类 = _类像界面(类名)
    仅界面 = bool(_参数("UI_ONLY"))
    for 方法名, 信息 in 方法表.items():
        if 方法名 == "<clinit>" and not _参数("INCLUDE_CLINIT"):
            continue
        if 方法名.startswith("__vp"):
            continue
        指令 = 信息.get("_指令")
        if 指令 is None:
            # 解析类时没保存完整指令表，用字符串位次近似：只做方法名/类名判定
            指令 = []
        可展示方法 = _方法可展示(方法名)
        类可展示 = 界面类 and 方法名 not in _逻辑方法名
        诊断方法 = any(词 in 方法名.lower() for 词 in _诊断方法词)
        选中 = []
        for 文本, 下标 in 信息["strings"]:
            if not 像展示文本(文本):
                continue
            if 指令 and _近邻命中(指令, 下标, _是逻辑调用, 6):
                continue
            if 指令 and _近邻命中(指令, 下标, _是展示调用, 12):
                选中.append(文本)
            elif 可展示方法:
                选中.append(文本)
            elif 类可展示:
                选中.append(文本)
            elif not 仅界面 and not 诊断方法 and not _诊断文本.match(文本):
                选中.append(文本)
        if 选中:
            结果[方法名] = sorted(set(选中))
    return 结果


def _模组ID(压缩包, 默认):
    try:
        名称 = set(压缩包.namelist())
        if "fabric.mod.json" in 名称:
            with 压缩包.open("fabric.mod.json") as 文件:
                return str(json.load(文件).get("id") or 默认)
        for 元数据 in ("META-INF/neoforge.mods.toml", "META-INF/mods.toml"):
            if 元数据 in 名称:
                with 压缩包.open(元数据) as 文件:
                    文本 = 文件.read().decode("utf-8", "replace")
                命中 = re.search(r'(?m)^\s*modId\s*=\s*"([^"]+)"', 文本)
                if 命中:
                    return 命中.group(1)
        if "mcmod.info" in 名称:
            with 压缩包.open("mcmod.info") as 文件:
                数据 = json.load(文件)
            if isinstance(数据, list) and 数据 and isinstance(数据[0], dict):
                return str(数据[0].get("modid") or 默认)
    except Exception:
        pass
    return 默认


def _安全名(名称):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", str(名称 or "")).strip() or "hardcoded"


def _类名集(条目列表):
    """从 .class 条目名收集类简单名(内部类拆开), 用来排除「类名被当界面文本」。"""
    名字 = set()
    for 条目 in 条目列表:
        简单 = 条目[:-6].rsplit("/", 1)[-1]
        名字.add(简单)
        for 段 in 简单.split("$"):
            if 段:
                名字.add(段)
    return 名字


def _收集Jar(路径):
    global _类名集合
    类映射 = {}
    try:
        with zipfile.ZipFile(路径) as 压缩包:
            模组ID = _安全名(_模组ID(压缩包, Path(路径).stem))
            类条目 = [条目 for 条目 in 压缩包.namelist()
                      if 条目.endswith(".class")
                      and not 条目.startswith("META-INF/versions/")
                      and not 条目.startswith(_跳过类前缀)]
            _类名集合 = _类名集(类条目)
            for 条目 in 类条目:
                try:
                    类名, 方法表 = 解析类(压缩包.read(条目))
                except Exception:
                    continue
                if 类名:
                    文本 = _收集类文本(类名, 方法表)
                    if 文本:
                        类映射[类名] = 文本
            return 模组ID, 类映射
    except Exception:
        return _安全名(Path(路径).stem), {}


def _收集目录(根):
    global _类名集合
    类映射 = {}
    根 = Path(根)
    条目列表 = [文件.relative_to(根).as_posix() for 文件 in 根.rglob("*.class")
                if not 文件.relative_to(根).as_posix().startswith(_跳过类前缀)]
    _类名集合 = _类名集(条目列表)
    for 相对 in 条目列表:
        try:
            类名, 方法表 = 解析类((根 / 相对).read_bytes())
        except Exception:
            continue
        if 类名:
            文本 = _收集类文本(类名, 方法表)
            if 文本:
                类映射[类名] = 文本
    return _安全名(根.name), 类映射


def _校验译文(原文, 译文):
    """占位符/颜色码/换行不一致就丢弃，宁可保留英文也不弄坏格式化文本。"""
    if not isinstance(译文, str) or not 译文.strip() or 译文 == 原文:
        return None
    if 译文.startswith("@"):  # Vault Patcher 把 @ 开头当半匹配，必须避开
        return None
    if "\ufffd" in 译文:
        return None
    if _参数("VALIDATE_FORMAT"):
        if Counter(_占位符.findall(原文)) != Counter(_占位符.findall(译文)):
            return None
        if re.findall(r"§.", 原文) != re.findall(r"§.", 译文):
            return None
        if 原文.count("\n") != 译文.count("\n"):
            return None
        if re.findall(r"\{[^{}]*\}", 原文) != re.findall(r"\{[^{}]*\}", 译文):
            return None
        if re.findall(r"\\[nrt]", 原文) != re.findall(r"\\[nrt]", 译文):
            return None
    # 保留原文首尾空白，避免拼接后少空格/多空格
    前导 = 原文[:len(原文) - len(原文.lstrip())]
    尾部 = 原文[len(原文.rstrip()):]
    译文 = 译文.strip()
    if not 译文:
        return None
    return 前导 + 译文 + 尾部


def _翻译(Self, 原文列表):
    """批量翻译候选文本；返回 原文 -> 译文。"""
    原文列表 = list(dict.fromkeys(文本 for 文本 in 原文列表 if 文本))
    if not 原文列表:
        return {}
    Self.本次读取条目数 = getattr(Self, "本次读取条目数", 0) + len(原文列表)
    语言 = str(getattr(getattr(Self, "Config", None), "LANGUAGE_OUTPUT", "") or "")

    映射 = {}
    for 文本 in 原文列表:
        键 = (语言, 文本)
        if 键 in _译文缓存:
            映射[文本] = _译文缓存[键]
    待翻译原文 = [文本 for 文本 in 原文列表 if 文本 not in 映射]
    待翻译 = [[f"hc:{i}", 文本, ""] for i, 文本 in enumerate(待翻译原文)]
    if 待翻译:
        try:
            结果 = Loop.运行(Self.翻译语言列表(待翻译, 使用模型=set()))
        except Exception:
            Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=2)
            结果 = []
        for 条目 in 结果:
            try:
                键, 译文 = str(条目[0]), str(条目[1])
                if 键.startswith("hc:"):
                    原文 = 待翻译原文[int(键[3:])]
                    译文 = _校验译文(原文, 译文)
                    if 译文:
                        映射[原文] = 译文
                        _译文缓存[(语言, 原文)] = 译文
            except Exception:
                continue
    return 映射


def _写模块(输出, 模块):
    """原子写 + 语法校验 + 旧文件备份；坏 JSON 不如不写。"""
    文本 = json.dumps(模块, ensure_ascii=False, indent=2)
    json.loads(文本)
    输出 = Path(输出)
    输出.parent.mkdir(parents=True, exist_ok=True)
    if 输出.is_file():
        try:
            if 输出.read_text(encoding="utf-8") != 文本:
                shutil.copy2(输出, 输出.with_name(输出.name + ".bak"))
        except Exception:
            pass
    临时 = 输出.with_name(输出.name + ".tmp")
    临时.write_text(文本, encoding="utf-8")
    os.replace(临时, 输出)


def _模块条目(模组ID, 条目列表, 合并说明=None):
    return [
        {"name": f"{模组ID} Hardcoded Translation",
         "desc": 合并说明 or "Generated by TranslatorMinecraft",
         "mods": 模组ID,
         "authors": "TranslatorMinecraft",
         "dynamic": False,
         "i18n": False},
    ] + 条目列表


def _确保VP配置(Self, 根):
    """外部模块要被 VP 加载，必须 load_all_modules=true；已有配置只补键不覆盖。"""
    配置 = Path(根) / "config" / "vaultpatcher_asm" / "config.json"
    try:
        数据 = {}
        if 配置.is_file():
            try:
                旧 = json.loads(配置.read_text(encoding="utf-8"))
                if isinstance(旧, dict):
                    数据 = 旧
            except Exception:
                数据 = {}
        if 数据.get("load_all_modules") is True and 配置.is_file():
            return
        数据["load_all_modules"] = True
        数据.setdefault("default_language",
                       str(getattr(Self.Config, "LANGUAGE_OUTPUT", "zh_cn") or "zh_cn"))
        _写模块(配置, 数据)
        Self.日志("log.core.translator.succeed", path=str(配置), info_level=0)
    except Exception:
        Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=1)


def 执行(Self, 根, path2=None, **参数):
    if not _参数("ENABLE"):
        return 根
    根 = Path(根)
    if not 根.exists():
        return 根

    任务 = []  # (base_dir, modid, class_map)
    if 根.is_file() and 根.suffix.lower() == ".jar":
        模组ID, 类映射 = _收集Jar(根)
        if 类映射:
            基目录 = 根.parent
            if 基目录.name.lower() == "mods":  # 直接丢 jar 时别写进 mods 里
                基目录 = 基目录.parent
            任务.append((基目录, 模组ID, 类映射))
    elif (根 / "mods").is_dir():
        for jar in sorted((根 / "mods").glob("*.jar")):
            模组ID, 类映射 = _收集Jar(jar)
            if 类映射:
                任务.append((根, 模组ID, 类映射))
    else:
        模组ID, 类映射 = _收集目录(根)
        if 类映射:
            任务.append((根, 模组ID, 类映射))

    if not 任务:
        return 根

    # 每个 jar 独立限流，避免一个大模组吃光所有条目
    全部原文 = []
    已见 = set()
    单jar上限 = max(1, int(_参数("MAX_STRINGS")))
    总上限 = 单jar上限 * max(1, min(len(任务), 20))
    for _, _, 类映射 in 任务:
        本任务数量 = 0
        for 类名 in sorted(类映射):
            for 方法名 in sorted(类映射[类名]):
                for 文本 in 类映射[类名][方法名]:
                    if len(全部原文) >= 总上限 or 本任务数量 >= 单jar上限:
                        break
                    if 文本 not in 已见:
                        已见.add(文本)
                        全部原文.append(文本)
                        本任务数量 += 1
                if len(全部原文) >= 总上限 or 本任务数量 >= 单jar上限:
                    break
            if len(全部原文) >= 总上限 or 本任务数量 >= 单jar上限:
                break
    if not 全部原文:
        return 根

    映射 = _翻译(Self, 全部原文)
    if not 映射:
        return 根

    模块任务 = []
    for 基目录, 模组ID, 类映射 in 任务:
        条目列表 = []
        for 类名 in sorted(类映射):
            目标 = 类名.replace("/", ".")
            类级映射 = {}
            for 方法名 in sorted(类映射[类名]):
                对 = [{"k": 文本, "v": 映射[文本]}
                      for 文本 in 类映射[类名][方法名] if 文本 in 映射]
                if not 对:
                    continue
                if _参数("TARGET_METHOD"):
                    条目列表.append({"target_class": [目标],
                                     "info": {"method": 方法名},
                                     "pairs": 对})
                else:
                    类级映射.update({项["k"]: 项["v"] for 项 in 对})
            if 类级映射:
                条目列表.append({"target_class": [目标],
                                 "pairs": [{"k": k, "v": v} for k, v in 类级映射.items()]})
        if 条目列表:
            模块任务.append((基目录, 模组ID, 条目列表))

    if _参数("MERGE_MODULES") and len(模块任务) > 1:
        合并映射 = {}
        模组列表 = []
        for _, 模组ID, 条目列表 in 模块任务:
            if 模组ID not in 模组列表:
                模组列表.append(模组ID)
            for 条目 in 条目列表:
                目标 = 条目["target_class"][0]
                方法 = (条目.get("info") or {}).get("method", "")
                对映射 = 合并映射.setdefault((目标, 方法), {})
                for 对 in 条目["pairs"]:
                    对映射[对["k"]] = 对["v"]
        合并条目 = []
        for (目标, 方法), 对映射 in sorted(合并映射.items()):
            条目 = {"target_class": [目标],
                     "pairs": [{"k": k, "v": v} for k, v in sorted(对映射.items())]}
            if 方法:
                条目 = {"target_class": [目标], "info": {"method": 方法},
                         "pairs": 条目["pairs"]}
            合并条目.append(条目)
        模块 = _模块条目("TranslatorVault", 合并条目,
                         "Generated by TranslatorMinecraft (merged: " + ",".join(模组列表) + ")")
        模块[0]["mods"] = ",".join(模组列表)
        输出目录 = Path(模块任务[0][0]) / "vaultpatcher" / "modules"
        输出 = 输出目录 / "translatorvault-hardcoded.json"
        try:
            _写模块(输出, 模块)
            Self.日志("log.core.translator.succeed", path=str(输出), info_level=0)
        except Exception:
            Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=1)
    else:
        for 基目录, 模组ID, 条目列表 in 模块任务:
            输出目录 = Path(基目录) / "vaultpatcher" / "modules"
            输出 = 输出目录 / f"{模组ID}-hardcoded.json"
            try:
                _写模块(输出, _模块条目(模组ID, 条目列表))
                Self.日志("log.core.translator.succeed", path=str(输出), info_level=0)
            except Exception:
                Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=1)

    if 模块任务:
        输出根 = 根 if 根.is_dir() else 任务[0][0]
        _确保VP配置(Self, 输出根)
        if (Path(根) / "mods").is_dir():
            _复制运行时(Self, 根)
    return 根


def _复制运行时(Self, 根):
    """按配置复制运行时 jar；TranslatorVault 默认不复制，留空即跳过。"""
    try:
        库目录 = Path(getattr(Self.Config, "LIB_PATH", "./libs"))
        if not 库目录.is_absolute():
            库目录 = (Path(__file__).resolve().parent.parent / 库目录).resolve()
        if not 库目录.is_dir():
            Self.日志("log.core.translator.general.error.unknown",
                      e=f"LIB_PATH not found: {库目录}", info_level=1)
            return
        目标表 = {
            "TRANSLATORVAULT_JAR": Path(根) / "vaultpatcher" / "plugins",
            "VAULTPATCHER_JAR": Path(根) / "mods",
        }
        清单 = {"mods": [], "plugins": []}
        for 配置键, 目标目录 in 目标表.items():
            文件名 = str(_参数(配置键) or "").strip()
            if not 文件名:
                continue
            源 = 库目录 / 文件名
            if not 源.is_file():
                Self.日志("log.core.translator.general.error.unknown",
                          e=f"{配置键} not found: {源}", info_level=1)
                continue
            目标目录.mkdir(parents=True, exist_ok=True)
            目标 = 目标目录 / 文件名
            if not 目标.exists():
                shutil.copy2(源, 目标)
                Self.日志("log.core.translator.succeed", path=str(目标), info_level=0)
            if 配置键 == "TRANSLATORVAULT_JAR":
                清单["plugins"].append(文件名)
            else:
                清单["mods"].append(文件名)
        清单路径 = Path(根) / "vaultpatcher" / "translatorvault_runtime.json"
        清单路径.parent.mkdir(parents=True, exist_ok=True)
        清单路径.write_text(json.dumps(清单, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=1)


def 探测类(Self, 路径):
    return Path(路径).suffix.lower() == ".class"


def 执行单个类(Self, 根, path2=None, **参数):
    """翻译单个 .class：生成同名 *-hardcoded.json，可直接放进 vaultpatcher/modules。"""
    global _类名集合
    if not _参数("ENABLE"):
        return 根
    文件 = Path(根)
    if not 文件.is_file() or 文件.suffix.lower() != ".class":
        return 根
    _类名集合 = {文件.stem} | {段 for 段 in 文件.stem.split("$") if 段}
    try:
        类名, 方法表 = 解析类(文件.read_bytes())
    except Exception:
        return 根
    if not 类名:
        return 根
    类文本 = _收集类文本(类名, 方法表)
    候选 = [文本 for 方法 in 类文本.values() for 文本 in 方法]
    if not 候选:
        return 根

    映射 = _翻译(Self, sorted(set(候选)))
    if not 映射:
        return 根

    条目列表 = []
    for 方法名 in sorted(类文本):
        对 = [{"k": 文本, "v": 映射[文本]} for 文本 in 类文本[方法名] if 文本 in 映射]
        if 对:
            条目列表.append({"target_class": [类名.replace("/", ".")],
                             "info": {"method": 方法名}, "pairs": 对})
    if not 条目列表:
        return 根
    输出 = 文件.with_name(f"{文件.stem}-hardcoded.json")
    _写模块(输出, _模块条目(文件.stem, 条目列表))
    Self.日志("log.core.translator.succeed", path=str(输出), info_level=0)
    return 输出


Mods().注册翻译类型("Hardcoded", 执行, 显示名="Hardcoded")
Mods().注册翻译类型("HardcodedClass", 执行单个类, 显示名="Hardcoded Class",
                    文件探测=探测类, 探测分数=90)

Mods().注册翻译类型到模组("Hardcoded", ".", 注册路径="/", 打包扩展名={".json"})
Mods().注册翻译类型到整合包("Hardcoded", ".", 注册路径="/", 打包扩展名={".json"})
Mods().注册翻译类型到归档("Hardcoded", ".", 注册路径="/", 打包扩展名={".json"})
