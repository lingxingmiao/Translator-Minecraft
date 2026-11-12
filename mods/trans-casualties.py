# 可用配置(覆盖键 MOD_TRANS_CASUALTIES_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   ["*.json"]
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (json, Path, Parser, Mods, eb)

模组参数 = Mods().模组配置("trans-casualties", {
    "FILE_MATCH": ["*.json"],
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


def 探测语言文件(Self, 路径):
    if Path(路径).suffix.lower() != '.json':
        return False
    try:
        with open(路径, 'r', encoding='utf-8-sig') as f:
            数据 = json.load(f)
    except Exception:
        return False
    return isinstance(数据, dict) and 'name' in 数据


def 执行语言文件(Self, 根, path2=None, **参数):
    解析 = CasualtiesParser(Self)
    Self.翻译流程(根, Mods().取值(Self, ("trans-casualties", "FILE_MATCH")), 解析.读取, Self.Module.过滤键文本, lambda x: x[2],
                解析.写回, Mods().取值(Self, ("trans-casualties", "READ_CONCURRENT")),
                Mods().取值(Self, ("trans-casualties", "WRITE_CONCURRENT")), "lang",
                输出方法=lambda p: p.parent / f"{Self.Config.LANGUAGE_OUTPUT}.json",
                传入使用模型=True, path2=path2, **参数)


class CasualtiesParser(Parser):

    def 读取(Self, file: str):
        提取列表 = []
        try:
            原始对象 = Self.读取Json文件(file)
        except Exception:
            Self.日志("log.module.lang.load.error", mod="Casualties: Unknown", file=file, e=eb.format_exc(), info_level=2)
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

    def 写回(Self, 条目列表: list, 使用模型=None):
        if not 条目列表:
            return
        文件路径 = 条目列表[0][2]
        try:
            数据 = Self.读取Json文件(文件路径)
        except Exception:
            Self.日志("log.module.lang.load.error", mod="Casualties: Unknown", file=文件路径, e=eb.format_exc(), info_level=2)
            return
        数据["description"] = Self.Config.PACK_META_TEMPLATE_CASUALTIESUNKNOWN.format(lang=Self.Config.LANGUAGE_OUTPUT, model=", ".join(m for m in (使用模型 or []) if m and m != "null") or Self.Config.LLM_MODEL or Self.Lang("log.core.package.zip.hit"), author=Self.Config.PACK_AUTHOR or "海盐青茫")
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
            Self.写入Json文件(Path(文件路径).parent / f"{Self.Config.LANGUAGE_OUTPUT}.json", 数据)
        except Exception:
            Self.日志("log.module.lang.write.error", mod="Casualties: Unknown", file=文件路径, e=eb.format_exc(), info_level=2)


Mods().注册翻译类型("CasualtiesUnknown", 执行语言文件, 解析器类=CasualtiesParser,
                显示名="Casualties: Unknown Language File", 文件探测=探测语言文件, 探测分数=80,
                产物="输出语言文件")
