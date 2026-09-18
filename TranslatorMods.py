from TranslatorLib import *

# BM25注册======================================================== #
BM25分词器注册表: Dict[str, Type] = {}

def 注册分词器(语言: str):
    def decorator(cls):
        BM25分词器注册表[语言] = cls
        return cls
    return decorator

def BM25分词器(Self, 语言: str):
    分词器 = BM25分词器注册表.get(语言)
    if not 分词器:
        Self.日志("log.mods.bm25.not.lang", lang=语言, info_level=2)
    return 分词器

def BM25分词(Self, 语言: str, 文本列表: list):
    分词器 = BM25分词器(Self, 语言)
    if not 分词器:
        return None
    结果 = 分词器(文本列表)
    if hasattr(结果, "ids"):
        return 结果
    if not 结果:
        return 结果
    首个 = 结果[0]
    if isinstance(首个, (list, tuple, set)) and 首个:
        try:
            if isinstance(next(iter(首个)), int):
                return [[str(编号) for 编号 in 文档] for 文档 in 结果]
        except Exception:
            return 结果
    return 结果
# ================================================================ #

# 量化类型注册================================================================ #
量化编码注册表: Dict[str, Any] = {}
量化解码注册表: Dict[str, Any] = {}
量化位数注册表: Dict[str, float] = {}
量化解码键注册表: Dict[str, list] = {}
量化裁切注册表: Dict[str, bool] = {}

def 注册量化类型(量化类型: str, 位数: float, 裁切: bool = False):
    def decorator(编码函数):
        量化编码注册表[量化类型] = 编码函数
        量化位数注册表[量化类型] = 位数
        量化裁切注册表[量化类型] = 裁切
        return 编码函数
    return decorator

def 注册反量化类型(量化类型: str, 位数: float, 键列表):
    def decorator(解码函数):
        量化解码注册表[量化类型] = 解码函数
        量化位数注册表[量化类型] = 位数
        量化解码键注册表[量化类型] = list(键列表)
        return 解码函数
    return decorator

def 量化位数(量化类型: str):
    return 量化位数注册表.get(量化类型)

def 量化类型列表():
    return sorted(set(量化编码注册表) | set(量化解码注册表))

def 绑定解码(解码函数, 键列表):
    def 解码(Self, 数据):
        return 解码函数(Self, *(数据[键] for 键 in 键列表))
    return 解码

def 裁切包装(编码函数):
    def 编码(Self, 数组):
        return 编码函数(Self, Self.分位裁切(数组))
    return 编码

def 安装量化注册(Self, 编码映射: dict, 解码映射: dict):
    目标类 = type(Self)
    for 名称, 编码函数 in 量化编码注册表.items():
        setattr(目标类, f"F32编码{名称}", 编码函数)
        编码映射[名称] = 裁切包装(编码函数) if 量化裁切注册表.get(名称) else 编码函数
    for 名称, 解码函数 in 量化解码注册表.items():
        setattr(目标类, f"{名称}解码F32", 解码函数)
        解码映射[名称] = 绑定解码(解码函数, 量化解码键注册表[名称])
    return 编码映射, 解码映射

@注册量化类型("Float32", 32)
def F32编码Float32(Self, 数组):
    return {"Vector": np.asarray(数组, dtype=np.float32).copy()}

@注册反量化类型("Float32", 32, ["Vector"])
def Float32解码F32(Self, 向量):
    return np.asarray(向量, dtype=np.float32)

@注册量化类型("Float16", 16)
def F32编码Float16(Self, 数组):
    return {"Vector": np.asarray(数组, dtype=np.float16)}

@注册反量化类型("Float16", 16, ["Vector"])
def Float16解码F32(Self, 向量):
    return np.asarray(向量, dtype=np.float16).astype(np.float32)
# ============================================================================= #

# 分位裁切注册====================================================== #
裁切方法注册表: Dict[str, Any] = {}

def 注册裁切类型(裁切类型: str):
    def decorator(裁切函数):
        裁切方法注册表[裁切类型] = 裁切函数
        return 裁切函数
    return decorator

def 裁切方法(Self):
    return 裁切方法注册表.get(Self.Config.VEC_QUANTIZATION_CLIP_TYPE)

def 裁切类型列表():
    return sorted(裁切方法注册表)
# ================================================================= #

# 向量搜索注册================================================================ #
索引类型注册表: Dict[str, Any] = {}
索引部件注册表: Dict[str, Any] = {}

def 注册索引类型(名称, 数据源="Vector", 支持增量=False, 包装型=False):
    名称列表 = [名称] if isinstance(名称, str) else list(名称)
    def decorator(构建函数):
        for 单个名称 in 名称列表:
            索引类型注册表[单个名称] = SimpleNamespace(
                名称=单个名称, 构建函数=构建函数,
                数据源 =数据源, 支持增量=支持增量, 包装型=包装型)
        return 构建函数
    return decorator

def 索引类型(名称):
    return 索引类型注册表.get(名称)

def 索引类型列表():
    return sorted(索引类型注册表)

def 索引模式名(模式):
    if isinstance(模式, (list, tuple)):
        return 模式[0] if 模式 else None
    return 模式

def 索引数据源(模式, 默认="Vector"):
    条目 = 索引类型(索引模式名(模式))
    return 条目.数据源 if 条目 else 默认

def 索引支持增量(模式, 默认=True):
    条目 = 索引类型(索引模式名(模式))
    return 条目.支持增量 if 条目 else 默认

def 索引包装型(模式):
    条目 = 索引类型(索引模式名(模式))
    return 条目.包装型 if 条目 else False

def 索引构建函数(模式):
    条目 = 索引类型(索引模式名(模式))
    return 条目.构建函数 if 条目 else None

def 注册索引部件(名称: str):
    def decorator(部件):
        索引部件注册表[名称] = 部件
        return 部件
    return decorator

def 索引部件(名称: str):
    return 索引部件注册表.get(名称)

def 索引部件列表():
    return sorted(索引部件注册表)

索引类注册表: Dict[str, Any] = {}

def 注册索引类(类名: str):
    def decorator(类):
        索引类注册表[类名] = 类
        return 类
    return decorator

def 索引类(名称: str):
    return 索引类注册表.get(名称)

def 索引类列表():
    return sorted(索引类注册表)
# ================================================================= #

# 模组配置(cfg)====================================================== #
# 统一管理 mods/ 下 *.cfg, 格式:
#     #mod:index-hnsw        <- 段头(模组名)
#     #HNSW 邻居数           <- 注释
#     M=32
模组配置表: Dict[str, Dict[str, Any]] = {}
模组配置声明表: Dict[str, dict] = {}
模组配置行缓存: Dict[str, list] = {}
模组配置已载入 = False
模组配置路径 = None

def _配置路径(路径=None):
    global 模组配置路径
    if 路径 is None and 模组配置路径 is not None:
        return 模组配置路径
    路径 = Path(路径)
    模组配置路径 = 路径 if 路径.is_absolute() else Path(__file__).resolve().parent / 路径
    return 模组配置路径

def 设置模组配置路径(路径):
    return _配置路径(路径)

def _模组目录():
    return _配置路径().parent

def _解析配置值(文本):
    文本 = 文本.strip()
    if 文本.lower() in ("true", "false"):
        return 文本.lower() == "true"
    try:
        return json.loads(文本)
    except Exception:
        return 文本

def _格式化配置值(值):
    if isinstance(值, bool):
        return "true" if 值 else "false"
    if 值 is None:
        return "null"
    if isinstance(值, (int, float)):
        return str(值)
    return json.dumps(值, ensure_ascii=False)

def _段头名(文本: str):
    for 前缀 in ("#mod:", "//mod:"):
        if 文本.startswith(前缀):
            return 文本[len(前缀):].strip()
    return None

def 载入模组配置(目录=None, 强制: bool = False):
    global 模组配置已载入
    if 模组配置已载入 and not 强制:
        return 模组配置表
    配置目录 = Path(目录) if 目录 else _模组目录()
    模组配置表.clear(); 模组配置行缓存.clear()
    当前段 = None
    for 文件 in sorted(配置目录.glob("*.cfg")):
        try:
            行列表 = 文件.read_text(encoding="utf-8-sig").splitlines()
        except Exception:
            continue
        模组配置行缓存[str(文件)] = 行列表
        for 行 in 行列表:
            文本 = 行.strip()
            段名 = _段头名(文本)
            if 段名 is not None:
                当前段 = 段名
                模组配置表.setdefault(当前段, {})
                continue
            if not 文本 or 文本.startswith("#") or 文本.startswith("//"):
                continue
            if "=" not in 文本:
                continue
            if 当前段 is None:
                当前段 = 文件.stem
                模组配置表.setdefault(当前段, {})
            键, _, 值 = 文本.partition("=")
            模组配置表[当前段][键.strip()] = _解析配置值(值)
    模组配置已载入 = True
    return 模组配置表

def 声明模组配置(名称: str, 默认: dict = None):
    声明 = dict(默认 or {})
    if 模组配置声明表.get(名称):
        声明 = {**模组配置声明表[名称], **声明}
    模组配置声明表[名称] = 声明
    模组配置表.setdefault(名称, {})
    return 声明

def 模组配置(名称: str, 默认: dict = None):
    载入模组配置()
    声明 = 声明模组配置(名称, 默认)
    结果 = dict(声明)
    结果.update(模组配置表.get(名称, {}))
    return 结果

def 配置值(名称: str, 键: str, 默认=None):
    载入模组配置()
    return 模组配置表.get(名称, {}).get(键, 默认)

def 配置段列表():
    载入模组配置()
    return sorted(模组配置表)

def _切分配置段(行列表):
    前导, 段列表, 当前 = [], [], None
    for 行 in 行列表:
        段名 = _段头名(行.strip())
        if 段名 is not None:
            当前 = [段名, 行, []]
            段列表.append(当前)
            continue
        if 当前 is None: 前导.append(行)
        else: 当前[2].append(行)
    return 前导, 段列表

def _段内键(段体):
    已有 = set()
    for 行 in 段体:
        文本 = 行.strip()
        if 文本 and not 文本.startswith("#") and not 文本.startswith("//") and "=" in 文本:
            已有.add(文本.partition("=")[0].strip())
    return 已有

def 同步模组配置(保存: bool = True, 目录=None, 路径=None):
    目标路径 = _配置路径(路径)
    载入模组配置(目录, 强制=True)
    结果 = {"新建": [], "删除": [], "补键": []}
    配置目录 = Path(目录) if 目录 else 目标路径.parent
    新建文件 = str(Path(目录) / "mods.cfg") if 目录 else str(目标路径)
    已有段 = set()
    for 路径, 行列表 in list(模组配置行缓存.items()):
        前导, 段列表 = _切分配置段(行列表)
        新行 = list(前导)
        for 段名, 段头, 段体 in 段列表:
            if 段名 not in 模组配置声明表:
                结果["删除"].append(段名) 
                continue
            已有段.add(段名)
            缺失 = {键: 值 for 键, 值 in 模组配置声明表[段名].items() if 键 not in _段内键(段体)}
            if 缺失:
                结果["补键"].append(段名)
                段体 = list(段体)
                if 段体 and 段体[-1].strip(): 段体.append("")
                for 键, 值 in 缺失.items():
                    段体.append(f"#{键}")
                    段体.append(f"{键}={_格式化配置值(值)}")
            新行.append(段头); 新行.extend(段体)
        if 新行 != 行列表:
            模组配置行缓存[路径] = 新行
    缺段 = [名 for 名 in 模组配置声明表 if 名 not in 已有段]
    if 缺段:
        新行 = list(模组配置行缓存.get(新建文件, []))
        for 名 in sorted(缺段):
            if 新行 and 新行[-1].strip(): 新行.append("")
            新行.append(f"#mod:{名}")
            for 键, 值 in 模组配置声明表[名].items():
                新行.append(f"#{键}")
                新行.append(f"{键}={_格式化配置值(值)}")
            结果["新建"].append(名)
        模组配置行缓存[新建文件] = 新行
    if 保存:
        for 路径, 行列表 in 模组配置行缓存.items():
            原行 = None
            try: 原行 = Path(路径).read_text(encoding="utf-8-sig").splitlines()
            except Exception: pass
            if 原行 == 行列表:
                continue
            try:
                Path(路径).parent.mkdir(parents=True, exist_ok=True)
                Path(路径).write_text("\n".join(行列表) + "\n", encoding="utf-8")
            except Exception:
                pass
    载入模组配置(目录, 强制=True)
    return 结果

def 索引配置快照():
    载入模组配置()
    快照 = []
    for 段 in sorted(模组配置表):
        条目 = 模组配置表[段]
        for 键 in sorted(条目):
            快照.append(f"{段}.{键}={条目[键]}")
    return 快照
# ================================================================= #
