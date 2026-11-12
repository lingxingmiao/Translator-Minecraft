from TranslatorLib import (Path, Mods)


def 执行(Self, 根, path2=None, **参数):
    TXLoader根 = Path(根)
    if not TXLoader根.is_dir(): return
    forceload目录 = TXLoader根 / "forceload"
    if forceload目录.is_dir():
        TXLoader根 = forceload目录
    for 模组目录 in TXLoader根.iterdir():
        if not 模组目录.is_dir(): continue
        语言目录 = 模组目录 / "lang"
        if not 语言目录.is_dir(): continue
        源语言文件 = next((f for f in 语言目录.iterdir() if f.is_file() and f.name.lower() == f"{Self.Config.LANGUAGE_INPUT}.lang".lower()), None)
        if 源语言文件 is None: continue
        参考文件 = next((f for f in 语言目录.iterdir() if f.is_file() and f.name.lower() == f"{Self.Config.LANGUAGE_OUTPUT}.lang".lower()), None)
        Self.翻译语言文件(file0=str(源语言文件), file1=str(参考文件) if 参考文件 else "", output_path=str(语言目录), **参数)


Mods().注册翻译类型("TXLoader", 执行, 显示名="TXLoader")


Mods().注册翻译类型到整合包("TXLoader", ("config/txloader/forceload", "config/txloader"),
                     注册路径="/config/txloader/forceload", 打包扩展名={".lang"})


# 单独传一份 txloader 配置(压缩包或目录)也认
Mods().注册翻译类型到归档("TXLoader", ("config/txloader/forceload", "config/txloader"),
                     注册路径="/config/txloader/forceload", 打包扩展名={".lang"}, 最大深度=2)
