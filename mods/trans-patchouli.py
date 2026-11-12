# 可用配置(覆盖键 MOD_TRANS_PATCHOULI_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   "*.json"
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (shutil, Path, Parser, Mods, eb)

模组参数 = Mods().模组配置("trans-patchouli", {
    "FILE_MATCH": "*.json",
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


def 执行(Self, 根, path2=None, **参数):
    解析 = PatchouliParser(Self)
    Self.翻译流程(根, Mods().取值(Self, ("trans-patchouli", "FILE_MATCH")), 解析.读取, Self.Module.过滤键文本,
                lambda x: str(x[2]), 解析.写回,
                Mods().取值(Self, ("trans-patchouli", "READ_CONCURRENT")),
                Mods().取值(Self, ("trans-patchouli", "WRITE_CONCURRENT")), "book", path2=path2, **参数)


def 语言版本执行(Self, 根, 索引ID=None, **参数):
    """把 <书>/<源语言>/ 复制成 <书>/<目标语言>/ 再翻(压缩包内帕秋莉手册用)"""
    根 = Path(根)
    if not 根.is_dir():
        return
    源语言 = Self.Config.LANGUAGE_INPUT
    目标语言 = Self.Config.LANGUAGE_OUTPUT
    if 源语言 == 目标语言:
        return
    额外参数 = {}
    if 索引ID is not None: 额外参数["索引ID"] = 索引ID
    for 书籍目录 in 根.iterdir():
        if not 书籍目录.is_dir():
            continue
        源目录 = next((d for d in 书籍目录.iterdir() if d.is_dir() and d.name.lower() == 源语言), None)
        if 源目录 is None:
            continue
        目标目录 = 书籍目录 / 目标语言
        try:
            if 目标目录.exists():
                shutil.rmtree(目标目录, ignore_errors=True)
            shutil.copytree(源目录, 目标目录)
        except Exception:
            Self.日志("log.module.book.load.error", file=str(源目录), e=eb.format_exc(), info_level=1)
            continue
        try:
            执行(Self, str(目标目录), **额外参数)
        except Exception:
            Self.日志("log.module.book.write.error", file=str(目标目录), path="", e=eb.format_exc(), info_level=1)


class PatchouliParser(Parser):

    def 读取(Self, 文件路径: str) -> list:
        结果 = []
        try:
            数据 = Self.读取Json文件(文件路径)
            Self.递归提取文本(数据, [], {"name", "description", "landing_text", "subtitle", "text", "title", "header", "link_text", "tooltip"}, lambda 路径, 值: 结果.append([[文件路径, 路径], 值]))
        except Exception: Self.日志("log.module.book.load.error", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果

    def 写回(Self, 翻译列表: list) -> None:
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
            except Exception: Self.日志("log.module.book.load.error", file=文件路径, e=eb.format_exc(), info_level=2); continue
            for 路径, 译文 in 替换列表:
                try:
                    Self.设置路径值(数据, 路径, 译文)
                except Exception: Self.日志("log.module.book.write.error", file=文件路径, path=str(路径), e=eb.format_exc(), info_level=2)
            Self.写入Json文件(文件路径, 数据)
    # ========== Hardcore Questing Mode ==========


Mods().注册翻译类型("Patchouli", 执行, 解析器类=PatchouliParser, 显示名="Patchouli")


Mods().注册翻译类型("PatchouliLanguage", 语言版本执行, 显示名="Patchouli")


Mods().注册翻译类型到整合包("Patchouli", "patchouli_books", 注册路径="/patchouli_books",
                     打包扩展名={".json"}, 最大深度=2)


Mods().注册翻译类型到归档("Patchouli", "patchouli_books", 注册路径="/patchouli_books",
                     打包扩展名={".json"}, 最大深度=2)
