# 可用配置(覆盖键 MOD_TRANS_CMM_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   "*.json"
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (Parser, Mods, eb)

模组参数 = Mods().模组配置("trans-cmm", {
    "FILE_MATCH": "*.json",
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


class CMMParser(Parser):

    def 读取(Self, 文件路径: str) -> list:
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
            Self.日志("log.module.menu.load.error", mod="CustomMainMenu", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果

    def 写回(Self, index: list) -> None:
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
                Self.设置路径值(数据, 父路径 + [字段名], 译文)
            Self.写入Json文件(文件路径, 数据)


Mods().注册翻译类型(
    "CustomMainMenu", "翻译流程", 文件匹配=("trans-cmm", "FILE_MATCH"), 解析器类=CMMParser,
    过滤="Module.过滤键文本", 分组键="文件",
    读并发=("trans-cmm", "READ_CONCURRENT"), 写并发=("trans-cmm", "WRITE_CONCURRENT"),
    日志类型="menu", 显示名="Custom Main Menu")


Mods().注册翻译类型到整合包("CustomMainMenu", "config/CustomMainMenu", 注册路径="/config/CustomMainMenu",
                     打包扩展名={".json"})
