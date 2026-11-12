# 容器探测: 整合包(启动器标记 / 实例结构) —— 探测规则归本文件
#
# 分数越高越先采用: 启动器标记 100 / 实例结构 60。

from TranslatorLib import Mods

Mods().注册容器("整合包", 100, lambda v, s: v.有浅层文件("manifest.json")
              and not v.有浅层文件("texts/languages.json", 最大深度=3))
Mods().注册容器("整合包", 100, lambda v, s: v.有浅层文件("modrinth.index.json"))
Mods().注册容器("整合包", 100, lambda v, s: v.有浅层文件("instance.cfg") or v.有浅层文件("mmc-pack.json"))
Mods().注册容器("整合包", 60, lambda v, s: v.有浅层前缀("mods")
              and any(v.有浅层前缀(名) for 名 in ("config", "scripts", "kubejs", "resourcepacks")))
