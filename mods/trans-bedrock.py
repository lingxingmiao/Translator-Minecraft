# 可用配置(覆盖键 MOD_TRANS_BEDROCK_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   ["*.lang"]
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (Path, Parser, Mods, eb)

模组参数 = Mods().模组配置("trans-bedrock", {
    "FILE_MATCH": ["*.lang"],
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


def 转义(文本):
    return str(文本).replace("\r\n", "\n").replace("\n", "\\n")


class BedrockAddonParser(Parser):

    def 读取(Self, 文件路径):
        结果 = []
        路径 = Path(文件路径)
        try:
            行列表 = 路径.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except Exception:
            Self.日志("log.module.lang.load.error", mod="Bedrock Addon", file=str(路径), e=eb.format_exc(), info_level=2)
            return 结果
        for 行 in 行列表:
            文本 = 行.strip()
            if not 文本 or 文本.startswith("#"):
                continue
            键, 分隔, 值 = 文本.partition("=")
            if not 分隔:
                continue
            键 = 键.strip()
            值 = 值.strip()
            if not 键 or not 值:
                continue
            结果.append([[str(路径), 键], 值, 键])
        return 结果

    def 写回(Self, 翻译列表):
        文件映射 = {}
        for 条目 in 翻译列表:
            文件路径 = 条目[0][0]
            键 = 条目[0][1]
            译文 = 条目[1]
            if 译文 is None or 译文 == "":
                continue
            文件映射.setdefault(文件路径, {})[键] = 转义(译文)
        for 文件路径, 替换字典 in 文件映射.items():
            路径 = Path(文件路径)
            try:
                行列表 = 路径.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            except Exception:
                Self.日志("log.module.lang.load.error", mod="Bedrock Addon", file=str(路径), e=eb.format_exc(), info_level=2)
                continue
            输出 = []
            for 行 in 行列表:
                文本 = 行.strip()
                if 文本 and not 文本.startswith("#") and "=" in 文本:
                    键 = 文本.partition("=")[0].strip()
                    if 键 in 替换字典:
                        输出.append(f"{键}={替换字典[键]}")
                        continue
                输出.append(行)
            try:
                路径.write_text("\n".join(输出) + "\n", encoding="utf-8", newline="")
            except Exception:
                Self.日志("log.module.lang.write.error", mod="Bedrock Addon", file=str(路径), e=eb.format_exc(), info_level=2)


def 执行(Self, 根, path2=None, **参数):
    解析 = BedrockAddonParser(Self)
    Self.翻译流程(
        根, Mods().取值(Self, ("trans-bedrock", "FILE_MATCH")), 解析.读取, Self.Module.过滤键文本,
        lambda x: x[0][0], 解析.写回,
        Mods().取值(Self, ("trans-bedrock", "READ_CONCURRENT")),
        Mods().取值(Self, ("trans-bedrock", "WRITE_CONCURRENT")),
        "lang", path2=path2, **参数)


Mods().注册翻译类型("BedrockAddon", 执行, 解析器类=BedrockAddonParser, 显示名="Bedrock Addon")

Mods().注册翻译类型到归档("BedrockAddon", "texts", 注册路径="/texts",
                     打包扩展名={".lang", ".json"}, 容器="附加包", 最大深度=2)

# 容器探测: 附加包 = manifest.json + texts/languages.json
Mods().注册容器("附加包", 100, lambda v, s: v.有浅层文件("manifest.json", 最大深度=3)
              and v.有浅层文件("texts/languages.json", 最大深度=3))
