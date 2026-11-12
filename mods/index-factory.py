# 索引工厂: 工厂串写在索引模式里, 例 ["Factory", "IVF256,PQ16x8"] / ["Factory", ["PCAR64", "IVF100", "SQ8"]]
# 可用配置(覆盖键 MOD_INDEX_FACTORY_<键>, 全大写):
#   INDEX   工厂串(索引模式没给子规格时用它)      ""
#   PARAMS  索引参数表 {"字段": 值}               {}
#           字段支持点路径, 例 {"nprobe": 32} / {"hnsw.efSearch": 240} / {"nsg.search_L": 240}
#           带变换的工厂串(PCAR/OPQ...)结果是 IndexPreTransform, 参数要走到里层: {"index.nprobe": 32}
# 参数表也能直接夹在子规格里: ["Factory", ["IVF256,PQ16x8", {"nprobe": 32}]]

from TranslatorLib import Mods, faiss

参数 = Mods().模组配置("index-factory", {"INDEX": "", "PARAMS": {}})


def _取设置(Self):
    子规格 = Self.当前子规格
    if isinstance(子规格, dict):
        return str(参数["INDEX"] or "").strip(), 子规格
    if isinstance(子规格, (list, tuple)):
        参数表 = next((项 for 项 in 子规格 if isinstance(项, dict)), None)
        工厂串 = ",".join(str(项) for 项 in 子规格 if not isinstance(项, dict))
        return (工厂串 or str(参数["INDEX"] or "")).strip(), (参数["PARAMS"] if 参数表 is None else 参数表)
    return str(子规格 or 参数["INDEX"] or "").strip(), 参数["PARAMS"]


def _定位(索引, 路径):
    目标, 名 = 索引, str(路径).strip()
    if "." in 名:
        前缀, 名 = 名.rsplit(".", 1)
        for 段 in 前缀.split("."):
            目标 = getattr(目标, 段, None)
            if 目标 is None:
                break
            try: 目标 = faiss.downcast_index(目标)
            except Exception: pass
    return 目标, 名


def _套参数(索引, 参数表):
    for 路径, 值 in (参数表 or {}).items():
        目标, 名 = _定位(索引, 路径)
        if 目标 is None:
            continue
        for 转换 in (None, int, float):
            try:
                setattr(目标, 名, 值 if 转换 is None else 转换(值))
                break
            except Exception:
                continue


@Mods().注册索引类型("faiss_factory", 数据源="Vector", 支持增量=True)
def 构建(Self, 向量, 索引=None):
    工厂串, 参数表 = _取设置(Self)
    结果 = faiss.index_factory(向量.shape[1], 工厂串)
    _套参数(结果, 参数表)
    return 结果, not bool(结果.is_trained)
