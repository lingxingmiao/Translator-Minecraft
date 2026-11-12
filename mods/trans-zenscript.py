# 可用配置(覆盖键 MOD_TRANS_ZENSCRIPT_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   "*.zs"
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4
#   UNICODE_ESCAPE    译文里的非 ASCII 是否转成 \uXXXX 转义   True

from TranslatorLib import (Path, Parser, Mods, eb)

模组参数 = Mods().模组配置("trans-zenscript", {
    "FILE_MATCH": "*.zs",
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
    "UNICODE_ESCAPE": True,
}, 索引指纹=False)


def 执行(Self, 根, path2=None, **参数):
    解析 = ZSParser(Self)
    Self.翻译流程(
        根, Mods().取值(Self, ("trans-zenscript", "FILE_MATCH")), 解析.读取,
        lambda 条目: 条目[1] and not Self.正则表达式预编译.翻译剔除方法.match(条目[1]),
        lambda x: x[0][0], 解析.写回,
        Mods().取值(Self, ("trans-zenscript", "READ_CONCURRENT")),
        Mods().取值(Self, ("trans-zenscript", "WRITE_CONCURRENT")),
        "script", path2=path2, **参数)


def 探测脚本(Self, 路径):
    return Path(路径).suffix.lower() == '.zs'


class ZSParser(Parser):

    def 读取(Self, 文件路径: str) -> list:
        结果 = []
        try:
            with open(文件路径, "r", encoding="utf-8") as f:
                行列表 = f.readlines()
            for 行号, 行内容 in enumerate(行列表, 1):
                for 类型名, 模式 in Self.正则表达式预编译.ZS模式.items():
                    for 序号, 匹配 in enumerate(模式.finditer(行内容)):
                        key = f"{文件路径}::{行号}::{序号}::{类型名}"
                        decoded = Self.File.Unicode转字符串(匹配.group(1))
                        结果.append([key, decoded, 文件路径])
        except Exception:
            Self.日志("log.module.script.load.error", mod="ZenScript", file=文件路径, e=eb.format_exc(), info_level=2)
        return 结果

    def 写回(Self, index: list) -> None:
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
                行信息.setdefault(行号, []).append((类型, 序号, Self.File.字符串转Unicode(译文)) if Mods().取值(Self, ("trans-zenscript", "UNICODE_ESCAPE")) else (类型, 序号, 译文))
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
                        Self.日志("log.module.script.match.miss", inedx=key_str, info_level=2)
                待替换.sort(key=lambda x: x[0], reverse=True)
                for start, end, 译文 in 待替换:
                    safe_译文 = 译文.replace('\\', '\\\\').replace('"', '\\"')
                    行内容 = 行内容[:start] + safe_译文 + 行内容[end:]
                行列表[行号 - 1] = 行内容
            新内容 = "".join(行列表)
            路径.write_text(新内容, encoding="utf-8", newline="")
    # ========== Custom Main Menu ==========


Mods().注册翻译类型("ZenScript", 执行, 解析器类=ZSParser, 显示名="ZenScripts", 文件探测=探测脚本,
                探测分数=70, 产物="源文件")


Mods().注册翻译类型到整合包("ZenScript", ("script", "scripts"), 注册路径="/script", 打包扩展名={".zs"})


Mods().注册翻译类型到归档("ZenScript", "scripts", 注册路径="/scripts", 打包扩展名={".zs"}, 最大深度=2)
