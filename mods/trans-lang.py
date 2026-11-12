from TranslatorLib import (Path, Mods)


def 探测语言文件(Self, 路径):
    return Path(路径).suffix.lower() in {".lang", ".json", ".local", ".loca"}


def 探测模组(Self, 路径):
    return Path(路径).suffix.lower() == '.jar'


Mods().注册翻译类型("LanguageFile", "翻译语言文件", 显示名="Language File", 文件探测=探测语言文件,
                探测分数=10)


Mods().注册翻译类型("Mod", "翻译语言文件", 显示名="Mod", 文件探测=探测模组,
                探测分数=60)


Mods().注册翻译类型到模组("LanguageFile", ".", 注册路径="/")


Mods().注册翻译类型到归档("LanguageFile", "shaders", 注册路径="/shaders", 打包扩展名={".lang", ".json"},
                     容器="光影包", 最大深度=2)


Mods().注册翻译类型到归档("LanguageFile", ".", 注册路径="/", 打包扩展名={".lang", ".json"},
                     容器="资源包")


# 容器探测: 模组(mods.toml / fabric.mod.json / .jar) 资源包 光影包
Mods().注册容器("模组", 95, lambda v, s: v.有浅层文件("META-INF/mods.toml"))
Mods().注册容器("模组", 95, lambda v, s: v.有浅层文件("fabric.mod.json"))
Mods().注册容器("模组", 90, lambda v, s: s.lower() == ".jar")
Mods().注册容器("资源包", 80, lambda v, s: v.有浅层文件("pack.mcmeta") and v.有浅层前缀("assets"))
Mods().注册容器("光影包", 70, lambda v, s: bool(v.查找("shaders", 最大深度=2, 任意深度=True)))
