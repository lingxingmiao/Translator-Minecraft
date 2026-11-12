# 可用配置(覆盖键 MOD_TRANS_BETTERQUESTING_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   "*.json"
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (Path, json, Parser, Mods, eb)

模组参数 = Mods().模组配置("trans-betterquesting", {
    "FILE_MATCH": "*.json",
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


def 执行(Self, 根, path2=None, **参数):
    解析 = BQParser(Self)
    Self.翻译流程(根, Mods().取值(Self, ("trans-betterquesting", "FILE_MATCH")), 解析.读取, Self.Module.过滤键文本,
                lambda x: x[0][0], 解析.写回,
                Mods().取值(Self, ("trans-betterquesting", "READ_CONCURRENT")),
                Mods().取值(Self, ("trans-betterquesting", "WRITE_CONCURRENT")), "quests", path2=path2, **参数)
    资源路径 = Path(f"{根}/resources")
    if 资源路径.is_dir():
        for 命名空间目录 in 资源路径.iterdir():
            if not 命名空间目录.is_dir(): continue
            语言目录 = 命名空间目录 / "lang"
            if not 语言目录.is_dir(): continue
            for 语言文件 in 语言目录.iterdir():
                if 语言文件.is_file() and 语言文件.name.lower() == f"{Self.Config.LANGUAGE_INPUT}.lang".lower():
                    Self.翻译语言文件(file0=str(语言文件), output_path=str(语言目录), **参数)
                    break


class BQParser(Parser):

    def 读取(Self, index: str):
        文件列表 = []
        try:
            NBT文件 = Self.读取Json文件(index)
            Self.递归提取文本(NBT文件, [], {"name:8", "desc:8"}, lambda 路径, 值: 文件列表.append([[index, 路径], 值]))
        except Exception: Self.日志("log.module.quests.load.error", mod="BetterQuests", file=index, e=eb.format_exc(), info_level=2)
        return 文件列表

    def 写回(Self, index: list):
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
            except Exception: Self.日志("log.module.quests.load.error", mod="BetterQuests", file=文件路径, e=eb.format_exc(), info_level=2); continue
            for 路径, 译文 in 替换列表:
                try:
                    Self.设置路径值(数据, 路径, 译文)
                except Exception: Self.日志("log.module.quests.write.error", mod="BetterQuests", file=文件路径, path=str(路径), e=eb.format_exc(), info_level=2)
            Self.写入Json文件(文件路径, 数据)


Mods().注册翻译类型("BetterQuesting", 执行, 解析器类=BQParser, 显示名="BetterQuesting")


Mods().注册翻译类型到整合包("BetterQuesting", "config/betterquesting", 注册路径="/config/betterquesting",
                     打包扩展名={".json", ".lang"})


Mods().注册翻译类型到归档("BetterQuesting", "betterquesting", 注册路径="/betterquesting",
                     打包扩展名={".json", ".lang"}, 最大深度=2)
