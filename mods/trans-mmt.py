# 可用配置(覆盖键 MOD_TRANS_MMT_<键>, 全大写):
#   READ_CONCURRENT   读取并发   4
#   WRITE_CONCURRENT  写入并发   4

from TranslatorLib import (json, re, Path, Parser, Mods, eb)

模组参数 = Mods().模组配置("trans-mmt", {
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


def 探测JSON(Self, 路径):
    if Path(路径).suffix.lower() != '.json':
        return False
    try:
        with open(路径, 'r', encoding='utf-8-sig') as f:
            数据 = json.load(f)
    except Exception:
        return False
    return isinstance(数据, dict) and 'header' in 数据 and 'content' in 数据


def 执行JSON(Self, 根, path2=None, **参数):
    解析 = MMTParser(Self)
    Self.翻译流程(根, "*.json", 解析.读取, Self.Module.过滤键文本, lambda x: x[0][0],
                解析.写回, Mods().取值(Self, ("trans-mmt", "READ_CONCURRENT")),
                Mods().取值(Self, ("trans-mmt", "WRITE_CONCURRENT")), "mmt", path2=path2, **参数)


def 探测TXT(Self, 路径):
    if Path(路径).suffix.lower() != '.txt':
        return False
    try:
        with open(路径, 'r', encoding='utf-8') as f:
            头 = f.read(200)
    except Exception:
        return False
    return '### FILE_NAME ###' in 头 or '### AI_PROMPT ###' in 头


def 执行TXT(Self, 根, path2=None, **参数):
    解析TXT = MMTTXTParser(Self)
    Self.翻译流程(根, "*.txt", 解析TXT.读取, lambda x: x[1] and x[1].strip(),
                lambda x: x[0][0], 解析TXT.写回,
                Mods().取值(Self, ("trans-mmt", "READ_CONCURRENT")),
                Mods().取值(Self, ("trans-mmt", "WRITE_CONCURRENT")), "mmt", path2=path2, **参数)


class MMTParser(Parser):

    def 读取(Self, 文件路径: str) -> list:
        结果 = []
        try:
            数据 = Self.读取Json文件(文件路径)
            if not isinstance(数据, dict) or "content" not in 数据:
                return 结果
            for 模组ID, 模组数据 in 数据["content"].items():
                if not isinstance(模组数据, dict) or "translations" not in 模组数据:
                    continue
                for 键, 值 in 模组数据["translations"].items():
                    if isinstance(值, str) and 值.strip():
                        结果.append([[文件路径, 模组ID, 键], 值])
        except Exception:
            Self.日志("log.module.mmt.load.error", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果

    def 写回(Self, index: list) -> None:
        if not index:
            return
        文件映射 = {}
        for 项 in index:
            文件路径 = 项[0][0]
            模组ID = 项[0][1]
            键 = 项[0][2]
            译文 = 项[1]
            文件映射.setdefault(文件路径, []).append((模组ID, 键, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            try:
                数据 = Self.读取Json文件(文件路径)
                if "content" not in 数据:
                    continue
                for 模组ID, 键, 译文 in 替换列表:
                    if 模组ID in 数据["content"] and 键 in 数据["content"][模组ID].get("translations", {}):
                        数据["content"][模组ID]["translations"][键] = 译文
                数据["header"]["translated"] = True
                Self.写入Json文件(文件路径, 数据, 缩进=2)
            except Exception:
                Self.日志("log.module.mmt.write.error", file=文件路径, e=eb.format_exc(), info_level=2)
    # ========== 数据包 ==========


class MMTTXTParser(Parser):

    def 读取(Self, 文件路径: str) -> list:
        结果 = []
        try:
            with open(文件路径, "r", encoding="utf-8") as f:
                内容 = f.read()
        except Exception:
            Self.日志("log.module.mmt.load.error", file=文件路径, e=eb.format_exc(), info_level=2)
            return 结果
        行列表 = 内容.splitlines()
        当前模组ID = None
        当前源语言 = None
        在提示中 = False
        在代码块 = False
        for 行 in 行列表:
            去空格 = 行.strip()
            if not 去空格:
                continue
            if re.match(r'^\s*```', 去空格):
                在代码块 = not 在代码块
                continue
            if 在代码块:
                continue
            if re.match(r'^\s*#', 去空格) and not 去空格.startswith('###'):
                continue
            if 在提示中:
                if 去空格.startswith('###'):
                    在提示中 = False
                else:
                    continue
            if re.match(r'###\s*FILE_NAME\s*###', 去空格):
                continue
            if re.match(r'###\s*AI_PROMPT\s*###', 去空格):
                在提示中 = True
                continue
            if re.match(r'###\s*TRANSLATION_STATUS\s*###', 去空格):
                continue
            模组匹配 = re.match(r'###\s*\[?([^|\]]+)\]?\s*\|\s*\[?([^|\]#]+)\]?\s*###', 去空格)
            if 模组匹配:
                当前模组ID = 模组匹配.group(1).strip()
                当前源语言 = 模组匹配.group(2).strip()
                continue
            if 当前模组ID is None:
                continue
            分隔匹配 = re.match(r'\s*([^=:→]+)\s*[=:→]\s*(.*)', 行)
            if 分隔匹配:
                键 = 分隔匹配.group(1).strip()
                值 = 分隔匹配.group(2).strip()
                if 键:
                    结果.append([[文件路径, 当前模组ID, 键], 值])
        return 结果

    def 写回(Self, 翻译列表: list) -> None:
        if not 翻译列表:
            return
        文件映射 = {}
        for 项 in 翻译列表:
            文件路径 = 项[0][0]
            模组ID = 项[0][1]
            键 = 项[0][2]
            译文 = 项[1]
            文件映射.setdefault(文件路径, {}).setdefault(模组ID, {})[键] = 译文
        for 文件路径, 模组条目 in 文件映射.items():
            try:
                with open(文件路径, "r", encoding="utf-8") as f:
                    原始内容 = f.read()
            except Exception:
                Self.日志("log.module.mmt.write.error", file=文件路径, e=eb.format_exc(), info_level=2)
                continue
            行列表 = 原始内容.splitlines()
            新行列表 = []
            当前模组ID = None
            在提示中 = False
            在代码块 = False
            跳过状态行 = False
            for 行 in 行列表:
                去空格 = 行.strip()
                if re.match(r'^\s*```', 去空格):
                    在代码块 = not 在代码块
                    新行列表.append(行)
                    continue
                if 在代码块:
                    新行列表.append(行)
                    continue
                if 在提示中:
                    if 去空格.startswith('###'):
                        在提示中 = False
                    else:
                        新行列表.append(行)
                        continue
                if re.match(r'###\s*AI_PROMPT\s*###', 去空格):
                    在提示中 = True
                    新行列表.append(行)
                    continue
                if re.match(r'###\s*TRANSLATION_STATUS\s*###', 去空格):
                    新行列表.append(行)
                    新行列表.append("TRANSLATED")
                    跳过状态行 = True
                    continue
                if 跳过状态行:
                    跳过状态行 = False
                    continue
                模组匹配 = re.match(r'###\s*\[?([^|\]]+)\]?\s*\|\s*\[?([^|\]#]+)\]?\s*###', 去空格)
                if 模组匹配:
                    当前模组ID = 模组匹配.group(1).strip()
                    新行列表.append(行)
                    continue
                if 当前模组ID and 当前模组ID in 模组条目:
                    分隔匹配 = re.match(r'\s*([^=:→]+)\s*[=:→]\s*.*', 行)
                    if 分隔匹配:
                        键 = 分隔匹配.group(1).strip()
                        if 键 in 模组条目[当前模组ID]:
                            译文 = 模组条目[当前模组ID][键]
                            行 = re.sub(r'(\s*[^=:→]+\s*[=:→]\s*).*', lambda _m: _m.group(1) + 译文,行)
                新行列表.append(行)
            try:
                with open(文件路径, "w", encoding="utf-8", newline="") as f:
                    f.write("\n".join(新行列表))
            except Exception:
                Self.日志("log.module.mmt.write.error", file=文件路径, e=eb.format_exc(), info_level=2)


Mods().注册翻译类型("MMT JSON", 执行JSON, 显示名="Mine Mod Translator Mod Json", 解析器类=MMTParser,
                文件探测=探测JSON, 探测分数=90, 产物="源文件")


Mods().注册翻译类型("MMT TXT", 执行TXT, 显示名="Mine Mod Translator Mod TXT", 解析器类=MMTTXTParser,
                文件探测=探测TXT, 探测分数=90, 产物="源文件")
