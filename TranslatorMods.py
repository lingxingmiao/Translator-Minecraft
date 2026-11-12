from TranslatorLib import *

# BM25注册======================================================== #
BM25分词器注册表: Dict[str, Type] = {}

def 注册分词器(语言: str):
    # 参数:
    #     语言:   语言代码(如 "zh_cn" / "en_us") 决定这份分词器归谁
    # 返回:
    #     callable: 装饰器 把分词器类登记进 BM25分词器注册表

    def decorator(cls):
        # 参数:
        #     cls:  分词器类
        # 返回:
        #     Type: 原类(登记后原样返回)

        BM25分词器注册表[语言] = cls
        return cls
    return decorator

def BM25分词器(Self, 语言: str):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     语言: 语言代码 用于查 BM25分词器注册表
    # 返回:
    #     Type | None: 该语言的分词器类; 没注册就记日志并返回 None

    分词器 = BM25分词器注册表.get(语言)
    if not 分词器:
        Self.日志("log.mods.bm25.not.lang", lang=语言, info_level=2)
    return 分词器

def BM25分词(Self, 语言: str, 文本列表: list):
    # 参数:
    #     Self:     Translator实例 隐式传入
    #     语言:     语言代码 决定用哪个分词器
    #     文本列表: 待分词的文本 格式: [文本, ...]
    # 返回:
    #     list | None: 分词结果 格式: [[词, ...], ...](int 编号统一转成 str);
    #                  没有分词器返回 None, 分词器自带 ids 时原样返回该对象

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
    # 参数:
    #     量化类型: 类型名(如 "Float32")
    #     位数:     该量化占多少比特 写进 量化位数注册表
    #     裁切:     是否套 分位裁切(见 裁切包装) 默认 False
    # 返回:
    #     callable: 装饰器 把编码函数登记进 量化编码注册表

    def decorator(编码函数):
        # 参数:
        #     编码函数: (Self, 数组) -> dict
        # 返回:
        #     callable: 原函数(登记后原样返回)

        量化编码注册表[量化类型] = 编码函数
        量化位数注册表[量化类型] = 位数
        量化裁切注册表[量化类型] = 裁切
        return 编码函数
    return decorator

def 注册反量化类型(量化类型: str, 位数: float, 键列表):
    # 参数:
    #     量化类型: 类型名(与编码侧同名)
    #     位数:     该量化占多少比特 写进 量化位数注册表
    #     键列表:   解码时从数据字典里取哪些键 按序传给解码函数
    # 返回:
    #     callable: 装饰器 把解码函数登记进 量化解码注册表

    def decorator(解码函数):
        # 参数:
        #     解码函数: (Self, 键1, 键2, ...) -> 向量
        # 返回:
        #     callable: 原函数(登记后原样返回)

        量化解码注册表[量化类型] = 解码函数
        量化位数注册表[量化类型] = 位数
        量化解码键注册表[量化类型] = list(键列表)
        return 解码函数
    return decorator

def 量化位数(量化类型: str):
    # 参数:
    #     量化类型: 类型名
    # 返回:
    #     float | None: 该量化类型的比特数; 未注册则 None

    return 量化位数注册表.get(量化类型)

def 量化类型列表():
    # 参数:
    #     无
    # 返回:
    #     list: [类型名, ...] 编码表与解码表的并集 已排序

    return sorted(set(量化编码注册表) | set(量化解码注册表))

def 绑定解码(解码函数, 键列表):
    # 参数:
    #     解码函数: (Self, 键1, 键2, ...) 形式的解码实现
    #     键列表:   从数据字典里按键序取出实参
    # 返回:
    #     callable: 解码(Self, 数据) 供量化解码注册表用

    def 解码(Self, 数据):
        # 参数:
        #     Self: Translator实例 隐式传入
        #     数据: 量化数据字典
        # 返回:
        #     Any: 解码函数(数据[键1], 数据[键2], ...) 的结果

        return 解码函数(Self, *(数据[键] for 键 in 键列表))
    return 解码

def 裁切包装(编码函数):
    # 参数:
    #     编码函数: (Self, 数组) 形式的编码实现
    # 返回:
    #     callable: 编码(Self, 数组) 编码前先过 Self.分位裁切

    def 编码(Self, 数组):
        # 参数:
        #     Self: Translator实例 隐式传入
        #     数组: 待量化向量
        # 返回:
        #     dict: 编码函数(Self.分位裁切(数组)) 的结果

        return 编码函数(Self, Self.分位裁切(数组))
    return 编码

def 安装量化注册(Self, 编码映射: dict, 解码映射: dict):
    # 参数:
    #     Self:     Translator实例 隐式传入
    #     编码映射: 输出参数 会被填成 {类型名: 编码函数}
    #     解码映射: 输出参数 会被填成 {类型名: 解码(Self, 数据)}
    # 返回:
    #     tuple: (编码映射, 解码映射); 同时把 F32编码<名> / <名>解码F32 挂到 Self 的类上

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
    # 参数:
    #     Self: Translator实例 隐式传入
    #     数组: 待量化的向量 np.ndarray
    # 返回:
    #     dict: {"Vector": float32 副本}

    return {"Vector": np.asarray(数组, dtype=np.float32).copy()}

@注册反量化类型("Float32", 32, ["Vector"])
def Float32解码F32(Self, 向量):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     向量: 量化后的向量
    # 返回:
    #     np.ndarray: float32 向量

    return np.asarray(向量, dtype=np.float32)

@注册量化类型("Float16", 16)
def F32编码Float16(Self, 数组):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     数组: 待量化的向量 np.ndarray
    # 返回:
    #     dict: {"Vector": float16 向量}

    return {"Vector": np.asarray(数组, dtype=np.float16)}

@注册反量化类型("Float16", 16, ["Vector"])
def Float16解码F32(Self, 向量):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     向量: 量化后的 float16 向量
    # 返回:
    #     np.ndarray: float32 向量

    return np.asarray(向量, dtype=np.float16).astype(np.float32)
# ============================================================================= #

# 分位裁切注册====================================================== #
裁切方法注册表: Dict[str, Any] = {}

def 注册裁切类型(裁切类型: str):
    # 参数:
    #     裁切类型: 类型名 对应 Config.VEC_QUANTIZATION_CLIP_TYPE
    # 返回:
    #     callable: 装饰器 把裁切函数登记进 裁切方法注册表

    def decorator(裁切函数):
        # 参数:
        #     裁切函数: (Self, 数组) -> 裁切后的数组
        # 返回:
        #     callable: 原函数(登记后原样返回)

        裁切方法注册表[裁切类型] = 裁切函数
        return 裁切函数
    return decorator

def 裁切方法(Self):
    # 参数:
    #     Self: Translator实例 隐式传入 用 Self.Config.VEC_QUANTIZATION_CLIP_TYPE 选
    # 返回:
    #     callable | None: 裁切函数; 未注册则 None

    return 裁切方法注册表.get(Self.Config.VEC_QUANTIZATION_CLIP_TYPE)

def 裁切类型列表():
    # 参数:
    #     无
    # 返回:
    #     list: [裁切类型名, ...] 已排序

    return sorted(裁切方法注册表)
# ================================================================= #

# 向量搜索注册================================================================ #
索引类型注册表: Dict[str, Any] = {}
索引部件注册表: Dict[str, Any] = {}

def 注册索引类型(名称, 数据源="Vector", 支持增量=False, 包装型=False):
    # 参数:
    #     名称:     索引类型名 可以是 str 或 str 列表(一次登记多个别名)
    #     数据源:   该索引取哪种数据 默认 "Vector"
    #     支持增量: 是否支持增量添加 默认 False
    #     包装型:   是否是包装其它索引的类型 默认 False
    # 返回:
    #     callable: 装饰器 把构建函数登记进 索引类型注册表

    名称列表 = [名称] if isinstance(名称, str) else list(名称)
    def decorator(构建函数):
        # 参数:
        #     构建函数: (Self, ...) -> 索引对象
        # 返回:
        #     callable: 原函数(登记后原样返回)

        for 单个名称 in 名称列表:
            索引类型注册表[单个名称] = SimpleNamespace(
                名称=单个名称, 构建函数=构建函数,
                数据源 =数据源, 支持增量=支持增量, 包装型=包装型)
        return 构建函数
    return decorator

def 索引类型(名称):
    # 参数:
    #     名称: 索引类型名
    # 返回:
    #     SimpleNamespace | None: {名称, 构建函数, 数据源, 支持增量, 包装型}

    return 索引类型注册表.get(名称)

def 索引类型列表():
    # 参数:
    #     无
    # 返回:
    #     list: [索引类型名, ...] 已排序

    return sorted(索引类型注册表)

def 索引模式名(模式):
    # 参数:
    #     模式: 索引模式 可能是单值 或 [主名, 参数...] 列表
    # 返回:
    #     Any | None: 列表取首元素(主名); 否则原样返回

    if isinstance(模式, (list, tuple)):
        return 模式[0] if 模式 else None
    return 模式

def 索引数据源(模式, 默认="Vector"):
    # 参数:
    #     模式: 索引模式(单值或列表)
    #     默认: 查不到时的返回值 默认 "Vector"
    # 返回:
    #     Any: 该索引类型的数据源

    条目 = 索引类型(索引模式名(模式))
    return 条目.数据源 if 条目 else 默认

def 索引支持增量(模式, 默认=True):
    # 参数:
    #     模式: 索引模式(单值或列表)
    #     默认: 查不到时的返回值 默认 True
    # 返回:
    #     bool: 该索引类型是否支持增量

    条目 = 索引类型(索引模式名(模式))
    return 条目.支持增量 if 条目 else 默认

def 索引包装型(模式):
    # 参数:
    #     模式: 索引模式(单值或列表)
    # 返回:
    #     bool: 该索引类型是否包装型; 查不到 False

    条目 = 索引类型(索引模式名(模式))
    return 条目.包装型 if 条目 else False

def 索引构建函数(模式):
    # 参数:
    #     模式: 索引模式(单值或列表)
    # 返回:
    #     callable | None: 该索引类型的构建函数

    条目 = 索引类型(索引模式名(模式))
    return 条目.构建函数 if 条目 else None

def 注册索引部件(名称: str):
    # 参数:
    #     名称: 部件名
    # 返回:
    #     callable: 装饰器 把部件登记进 索引部件注册表

    def decorator(部件):
        # 参数:
        #     部件: 部件实现
        # 返回:
        #     Any: 原部件(登记后原样返回)

        索引部件注册表[名称] = 部件
        return 部件
    return decorator

def 索引部件(名称: str):
    # 参数:
    #     名称: 部件名
    # 返回:
    #     Any | None: 部件

    return 索引部件注册表.get(名称)

def 索引部件列表():
    # 参数:
    #     无
    # 返回:
    #     list: [部件名, ...] 已排序

    return sorted(索引部件注册表)

索引类注册表: Dict[str, Any] = {}

def 注册索引类(类名: str):
    # 参数:
    #     类名: 索引类名
    # 返回:
    #     callable: 装饰器 把类登记进 索引类注册表

    def decorator(类):
        # 参数:
        #     类: 索引类
        # 返回:
        #     Type: 原类(登记后原样返回)

        索引类注册表[类名] = 类
        return 类
    return decorator

def 索引类(名称: str):
    # 参数:
    #     名称: 索引类名
    # 返回:
    #     Type | None: 索引类

    return 索引类注册表.get(名称)

def 索引类列表():
    # 参数:
    #     无
    # 返回:
    #     list: [索引类名, ...] 已排序

    return sorted(索引类注册表)

模组配置声明表: Dict[str, dict] = {}
模组配置覆盖表: Dict[str, Any] = {}
当前配置 = None
当前实例 = None
索引指纹排除: set = set()
模组配置键前缀 = "MOD_"
模组配置字段名 = "MOD_CONFIG"

def 是模组配置键(键):
    # 参数:
    #     键: 配置键名
    # 返回:
    #     bool: 是否形如 MOD_XXX(排除 MOD_CONFIG 本身)

    return (isinstance(键, str) and 键 != 模组配置字段名
            and 键.startswith(模组配置键前缀) and len(键) > len(模组配置键前缀))

def 归一化模组配置(配置=None):
    # 参数:
    #     配置: 原始配置字典 可以是 None
    # 返回:
    #     dict: {键(大写, 已去 MOD_ 前缀): 值} 只留 MOD_ 开头的键

    if not isinstance(配置, dict):
        return {}
    return {str(键).upper(): 值 for 键, 值 in 配置.items() if 是模组配置键(键)}

def 绑定配置(配置):
    # 参数:
    #     配置: Config 实例
    # 返回:
    #     Any: 绑定后的 当前配置(模块全局) 核心基础设施靠它取配置项

    global 当前配置
    当前配置 = 配置
    return 当前配置


def 绑定模组配置(配置=None):
    # 参数:
    #     配置: 原始配置字典 可以是 None
    # 返回:
    #     dict: 归一化后的 模组配置覆盖表(模块全局)

    global 模组配置覆盖表
    模组配置覆盖表 = 归一化模组配置(配置)
    return 模组配置覆盖表


def 绑定实例(实例=None):
    # 参数:
    #     实例: 带 Lang/Log 的实例(Config); 加载模组时由 加载模组 传进来, 可为 None
    # 返回:
    #     Any: 绑定后的 当前实例(模块全局) —— 模组 import 期注册报错靠它出本地化文案

    global 当前实例
    当前实例 = 实例
    return 当前实例


def 注册期实例():
    # 参数:
    #     无
    # 返回:
    #     Any: 注册期能出 Lang/Log 的实例 —— 当前实例 优先, 退到 当前配置;
    #          两者都拿不到就 None(调用方退回中文兜底文案)

    return 当前实例 if 当前实例 is not None else 当前配置

def 声明模组配置(名称: str, 默认: dict = None):
    # 参数:
    #     名称: 模组名(如 "trans-ftbquests")
    #     默认: 默认配置字典
    # 返回:
    #     dict: 合并后的声明(同名再声明时 新值覆盖旧值)

    声明 = dict(默认 or {})
    if 模组配置声明表.get(名称):
        声明 = {**模组配置声明表[名称], **声明}
    模组配置声明表[名称] = 声明
    return 声明

def 模组配置前缀(名称: str):
    # 参数:
    #     名称: 模组名
    # 返回:
    #     str: "MOD_<名称大写, '-'换'_'>_" 覆盖键前缀

    return f"{模组配置键前缀}{str(名称).strip().upper().replace('-', '_')}_"

def 模组配置覆盖(名称: str):
    # 参数:
    #     名称: 模组名
    # 返回:
    #     dict: {键: 值} 该模组被 MOD_<前缀>_<键> 覆盖的项(键已去前缀)

    前缀 = 模组配置前缀(名称)
    return {键[len(前缀):]: 值 for 键, 值 in 模组配置覆盖表.items() if 键.startswith(前缀)}

def 模组配置(名称: str, 默认: dict = None, 索引指纹: bool = True):
    # 参数:
    #     名称:     模组名
    #     默认:     默认配置字典
    #     索引指纹: 默认 True 参与索引缓存指纹;
    #               False 则该模组配置不进指纹(翻译类模组用 免得改个并发数就重建索引)
    # 返回:
    #     dict: 默认值 与 MOD_ 覆盖 合并后的配置(键名大小写不敏感)

    if not 索引指纹:
        索引指纹排除.add(名称)
    结果 = dict(声明模组配置(名称, 默认))
    字段索引 = {str(键).upper(): 键 for 键 in 结果}
    for 键, 值 in 模组配置覆盖(名称).items():
        结果[字段索引.get(键, 键)] = 值
    return 结果

def 配置值(名称: str, 键: str, 默认=None):
    # 参数:
    #     名称: 模组名
    #     键:   配置键
    #     默认: 都没命中时的返回值
    # 返回:
    #     Any: 覆盖值 > 声明值 > 默认

    覆盖 = 模组配置覆盖(名称)
    字段 = str(键).upper()
    if 字段 in 覆盖:
        return 覆盖[字段]
    return 模组配置声明表.get(名称, {}).get(键, 默认)

def 配置段列表():
    # 参数:
    #     无
    # 返回:
    #     list: [已声明过的模组名, ...] 已排序

    return sorted(模组配置声明表)

def 索引配置快照():
    # 参数:
    #     无
    # 返回:
    #     list: ["MOD_<模组>_<键>=值", ...] 已排序;
    #           跳过 索引指纹排除 里的模组(它们不影响索引缓存)

    快照 = []
    for 名称 in sorted(模组配置声明表):
        if 名称 in 索引指纹排除:
            continue
        前缀 = 模组配置前缀(名称)
        for 键, 值 in sorted(模组配置(名称).items(), key=lambda 项: str(项[0])):
            快照.append(f"{前缀}{str(键).upper()}={值}")
    return 快照


翻译类型注册表: Dict[str, Any] = {}
模组翻译挂载表: List[Any] = []
整合包翻译挂载表: List[Any] = []
归档翻译挂载表: List[Any] = []
容器探测表: List[Any] = []


def 注册容器(名称: str, 分数: int, 判定):
    # 参数:
    #     名称: 容器名(整合包/模组/资源包/数据包/光影包/归档/附加包...)
    #     分数: 判定优先级 越高越先采用(同分按注册先后)
    #     判定: callable(视图, 后缀) -> bool
    # 返回:
    #     SimpleNamespace: 追加进 容器探测表 的条目 {容器, 分数, 判定}

    条目 = SimpleNamespace(容器=名称, 分数=int(分数), 判定=判定)
    容器探测表.append(条目)
    return 条目


def 注册翻译类型(名称: str, 执行器: str, 文件匹配=None, 读取=None, 过滤=None, 分组键=None,
                应用=None, 读并发=None, 写并发=None, 日志类型=None, 输出方法=None,
                变体参数: str = None, 显示名: str = None,
                文件探测=None, 探测分数: int = 50, 产物: str = None,
                解析器类=None):
    # 参数:
    #     名称:       类型名(如 "FTBQuests")
    #     执行器:     Self 上的方法名 / "对象.方法" 点路径 / 模组自带 callable(Self, 根, ...);
    #                 == "翻译流程" 时才用 文件匹配/读取/过滤/分组键/应用 这套接线
    #     文件匹配:   该类型认哪些文件 glob(str 或 (模组名, 键) 引用)
    #     读取:       读取方法 字符串=Self 上的方法; 有 解析器类 时忽略
    #     过滤:       过滤方法 条目 -> bool
    #     分组键:     "文件"(按来源文件) / "上下文"(按条目第三项) / 可调用
    #     应用:       写回方法 字符串=Self 上的方法; 有 解析器类 时忽略
    #     读并发:     读取并发 int 或 Config 键名
    #     写并发:     写入并发 int 或 Config 键名
    #     日志类型:   日志键里的类型段(如 "quests") 默认取 名称
    #     输出方法:   收尾上报路径用 callable(路径) 可 None
    #     变体参数:   形如 "mode"; 挂载里的 变体 会以该名字 partial 进执行器
    #     显示名:     产物包名/日志名 默认取 名称
    #     文件探测:   callable(Self, 路径) -> bool 单文件指纹识别 可 None
    #     探测分数:   文件探测命中时的分数 默认 50
    #     产物:       单文件入口的返回规则 "源文件"/"输出语言文件"/None(用执行器返回值)
    #     解析器类:   内容级解析器类(Parser) 有则用它自己的 读取/写回
    # 返回:
    #     SimpleNamespace: 登记进 翻译类型注册表 的条目

    条目 = SimpleNamespace(
        名称=名称, 执行器=执行器, 文件匹配=文件匹配, 读取=读取, 过滤=过滤, 分组键=分组键,
        应用=应用, 读并发=读并发, 写并发=写并发, 日志类型=日志类型 or 名称,
        输出方法=输出方法, 变体参数=变体参数, 显示名=显示名 or 名称,
        文件探测=文件探测, 探测分数=探测分数, 产物=产物, 解析器类=解析器类)
    翻译类型注册表[名称] = 条目
    return 条目


def _挂载(Self, 目标: str, 类型名: str, 路径模式, 根=None, 变体=None, 注册路径=None,
         打包扩展名=None, 最大深度: int = 3, 具体度: int = None, 容器=None):
    # 参数:
    #     Self:       出本地化文案用的实例 —— 注册期由 注册期实例() 给(Config 已带 Lang/Log);
    #                 实在拿不到就退回中文兜底文案
    #     目标:       挂载表名 "模组" / "整合包" / "归档"
    #     类型名:     已注册的翻译类型名 不存在直接抛 LookupError
    #     路径模式:   路径 glob str 或 tuple(多模式)
    #     根:         模式命中后从哪一层开始翻 默认整包
    #     变体:       变体值 执行时以 类型.变体参数 的名字注入
    #     注册路径:   写进语言包/资源包时的目标路径
    #     打包扩展名: 该挂载会产出哪些后缀 供打包判断
    #     最大深度:   查找模式的最大深度 默认 3
    #     具体度:     模式具体程度 默认按模式段数算 越大越优先
    #     容器:       限定只在某些容器下生效 None=不限
    # 返回:
    #     SimpleNamespace: 追加进对应挂载表的条目

    if 类型名 not in 翻译类型注册表:
        raise LookupError(Self.Lang("log.mods.transl.type.not", type=类型名))
    模式 = (路径模式,) if isinstance(路径模式, str) else tuple(路径模式)
    if 具体度 is None:
        具体度 = max(len([段 for 段 in 单个.split("/") if 段]) for 单个 in 模式)
    条目 = SimpleNamespace(目标=目标, 类型名=类型名, 路径模式=模式, 根=根, 变体=变体,
                           注册路径=注册路径, 打包扩展名=打包扩展名, 最大深度=最大深度,
                           具体度=具体度, 容器=(容器,) if isinstance(容器, str) else 容器)
    {"模组": 模组翻译挂载表, "整合包": 整合包翻译挂载表, "归档": 归档翻译挂载表}[目标].append(条目)
    return 条目


def 注册翻译类型到模组(类型名: str, 路径模式, **参数):
    # 参数:
    #     类型名:   已注册的翻译类型名
    #     路径模式: 路径 glob str 或 tuple
    #     **参数:   透传给 _挂载(根/变体/注册路径/打包扩展名/最大深度/具体度/容器)
    # 返回:
    #     SimpleNamespace: 追加进 模组翻译挂载表 的条目

    return _挂载(注册期实例(), "模组", 类型名, 路径模式, **参数)


def 注册翻译类型到整合包(类型名: str, 路径模式, **参数):
    # 参数:
    #     类型名:   已注册的翻译类型名
    #     路径模式: 路径 glob str 或 tuple
    #     **参数:   透传给 _挂载(根/变体/注册路径/打包扩展名/最大深度/具体度/容器)
    # 返回:
    #     SimpleNamespace: 追加进 整合包翻译挂载表 的条目

    return _挂载(注册期实例(), "整合包", 类型名, 路径模式, **参数)


def 注册翻译类型到归档(类型名: str, 路径模式, **参数):
    # 参数:
    #     类型名:   已注册的翻译类型名
    #     路径模式: 路径 glob str 或 tuple
    #     **参数:   透传给 _挂载(根/变体/注册路径/打包扩展名/最大深度/具体度/容器)
    # 返回:
    #     SimpleNamespace: 追加进 归档翻译挂载表 的条目(内部用 也留给社区)

    return _挂载(注册期实例(), "归档", 类型名, 路径模式, **参数)


def 文件探测(Self, 路径) -> List[Tuple[str, int]]:
    # 参数:
    #     Self: Translator实例 隐式传入
    #     路径: 待识别的单个文件
    # 返回:
    #     list: [(类型名, 分数), ...] 按分数从高到低 同分按类型名排序

    命中 = []
    for 名称, 类型 in 翻译类型注册表.items():
        探测 = getattr(类型, "文件探测", None)
        if 探测 is None:
            continue
        try:
            命中_ = bool(探测(Self, 路径))
        except Exception:
            命中_ = False
        if 命中_:
            命中.append((名称, 类型.探测分数))
    命中.sort(key=lambda 项: (-项[1], 项[0]))
    return 命中


def 取解析器实例(Self, 类型名: str):
    # 参数:
    #     Self:   Translator实例 隐式传入
    #     类型名: 翻译类型名
    # 返回:
    #     Parser | None: 该类型的解析器实例(解析器类(Self)); 没注册则 None

    类型 = 翻译类型注册表.get(类型名)
    类 = getattr(类型, "解析器类", None) if 类型 else None
    return 类(Self) if 类 is not None else None


def 匹配挂载(视图, 挂载表: List[Any], 跳过根模式: bool = False) -> List[Tuple[Any, str]]:
    # 参数:
    #     视图:       归档/目录视图 提供 查找()
    #     挂载表:     模组/整合包/归档 挂载表
    #     跳过根模式: 默认 False; True 时忽略 "." / "" 整包模式(容器探测用)
    # 返回:
    #     list: [(挂载, 实际根路径), ...] 同一挂载命中多处(同一层)全都算

    命中 = []
    for 挂载 in 挂载表:
        for 模式 in 挂载.路径模式:
            if 跳过根模式 and 模式.replace("\\", "/").strip("/") in ("", "."):
                continue
            结果 = 视图.查找(模式, 最大深度=挂载.最大深度, 任意深度=True)
            if not 结果 and getattr(视图, "自身匹配", None) and 视图.自身匹配(模式):
                命中.append((挂载, ""))       # 上传的就是这个内容目录本身
                break
            if 结果:
                # 多个附加包/多个同名内容目录都要翻, 但嵌套在更深处的重复就不管
                最浅层数 = min(路径.count("/") for 路径 in 结果)
                命中.extend((挂载, 路径) for 路径 in 结果 if 路径.count("/") == 最浅层数)
                break
    return 命中


def _绑定(Self, 值):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     值:   注册项 None / 字符串("对象.方法") / callable / 已绑定方法
    # 返回:
    #     callable | None: 可直接调用的方法(模组函数会先塞 Self)

    if 值 is None or isinstance(值, str):
        return _解析可调用(Self, 值)
    if getattr(值, "__self__", None) is not None:
        return 值
    from functools import partial
    return partial(值, Self)


def 取值(Self, 值, 默认=None):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     值:   (模组名, 键) 引用 / Config 键名 / 字面量
    #     默认: 引用取不到时的返回值
    # 返回:
    #     Any: 运行时取到的值((模组名, 键) 每跑一次现查 覆盖即时生效)

    if isinstance(值, tuple) and len(值) == 2 and isinstance(值[0], str):
        候选 = 模组配置(值[0]).get(值[1], None)
        return 取配置值(Self, 候选, 默认) if 候选 is not None else 默认
    return 取配置值(Self, 值, 默认)


def 取配置值(Self, 值, 默认=None):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     值:   字符串=Config 键名(取不到原样返回) 其它原样返回
    #     默认: 结果为 None 时的返回值
    # 返回:
    #     Any: 并发数这类配置项的最终值

    结果 = _取配置值(Self, 值)
    return 默认 if 结果 is None else 结果


def _取配置值(Self, 值):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     值:   字符串=Config 键名 其它原样
    # 返回:
    #     Any: getattr(Self.Config, 值, 值)

    if isinstance(值, str) and 值:
        return getattr(Self.Config, 值, 值)
    return 值


def _解析可调用(Self, 值):
    # 参数:
    #     Self: Translator实例 隐式传入
    #     值:   callable / None / "对象.方法" 点路径字符串
    # 返回:
    #     callable | None: 解析出的可调用(None 原样返回)

    if callable(值) or 值 is None:
        return 值
    对象名, _, 方法名 = str(值).rpartition(".")
    对象 = getattr(Self, 对象名) if 对象名 else Self
    return getattr(对象, 方法名)


def 生成计划(视图, 目标: str) -> List[Any]:
    # 参数:
    #     视图: 归档/目录视图
    #     目标: 容器名 "整合包" / "模组" / 其它(走归档表)
    # 返回:
    #     list: [计划, ...] 形如 {类型名, 根, 变体, 注册路径, 打包扩展名, 具体度}
    #           按 具体度 降序; 多命中多计划

    挂载表 = {"整合包": 整合包翻译挂载表, "模组": 模组翻译挂载表}.get(目标, 归档翻译挂载表)
    计划列表 = []
    for 挂载, 实际根 in 匹配挂载(视图, 挂载表):
        if 挂载.容器 and 目标 not in 挂载.容器:
            continue
        计划列表.append(SimpleNamespace(
            类型名=挂载.类型名, 根=实际根, 变体=挂载.变体, 注册路径=挂载.注册路径,
            打包扩展名=挂载.打包扩展名, 具体度=挂载.具体度))
    计划列表.sort(key=lambda 计划: (-计划.具体度, 计划.类型名))
    return 计划列表


def 保守计划() -> List[Any]:
    # 参数:
    #     无
    # 返回:
    #     list: 只有一个 LanguageFile 计划的兜底列表(判不出类型时用 绝不套整合包流程)

    return [SimpleNamespace(类型名="LanguageFile", 根="", 变体=None, 注册路径=None,
                            打包扩展名={".lang", ".json"}, 具体度=0)]


def 执行计划(Self, 计划, 原始路径=None, 输出目录=None, **总参数):
    # 参数:
    #     Self:     Translator实例 隐式传入
    #     计划:     生成计划() 产出的计划条目
    #     原始路径: 归档/目录本身(语言文件流程要整个包 而不是子目录)
    #     输出目录: 写回目标(通常是缓存目录)
    #     **总参数: 透传给流程(如 path2=参考包目录) 解压/缓存由核心负责
    # 返回:
    #     Any: "翻译流程" 返回 计划.根(目录); 其余为执行器产物

    类型 = 翻译类型注册表.get(计划.类型名)
    if 类型 is None:
        raise LookupError(Self.Lang("log.mods.transl.reg.not", type=计划.类型名))
    变体参数 = {类型.变体参数: 计划.变体} if (类型.变体参数 and 计划.变体 is not None) else {}
    根 = 计划.根
    if 类型.执行器 == "翻译流程":
        return 执行翻译流程(Self, 计划.类型名, 根, 变体=计划.变体, **总参数)
    if 类型.执行器 == "翻译语言文件":
        参数 = {"file0": str(原始路径 or 根)}
        if 输出目录:
            参数["output_path"] = str(输出目录)
        return Self.翻译语言文件(**参数, **总参数)
    执行器 = _解析可调用(Self, 类型.执行器)
    if not isinstance(类型.执行器, str):        # 模组自带的完整流程(接收 Self)
        return 执行器(Self, 根, **变体参数, **总参数)
    return 执行器(根, **变体参数, **总参数)


def 取执行器(Self, 类型名: str):
    # 参数:
    #     类型名: 翻译类型名
    # 返回:
    #     Any: 执行器 字符串=Self 上的方法名 或模组自带 callable

    类型 = 翻译类型注册表.get(类型名)
    if 类型 is None:
        raise LookupError(Self.Lang("log.mods.transl.reg.not", type=类型名))
    return 类型.执行器

def 执行翻译流程(Self, 类型名: str, 根, 变体=None, **总参数):
    # 参数:
    #     Self:     Translator实例 隐式传入
    #     类型名:   翻译类型名(要求执行器 == "翻译流程")
    #     根:       待翻目录 或 单个文件
    #     变体:     变体值 以 类型.变体参数 的名字 partial 进 读取/应用
    #     **总参数: 透传给 翻译流程(如 path2=参考包目录)
    # 返回:
    #     Any: 根(原样返回 方便调用方登记产物)

    类型 = 翻译类型注册表.get(类型名)
    if 类型 is None:
        raise LookupError(Self.Lang("log.mods.transl.reg.not", type=类型名))
    变体参数 = {类型.变体参数: 变体} if (类型.变体参数 and 变体 is not None) else {}
    实例 = 取解析器实例(Self, 类型名)
    if 实例 is not None:
        读取, 应用 = 实例.读取, 实例.写回
        过滤 = _解析可调用(Self, 类型.过滤) if 类型.过滤 else None
    else:
        读取 = _绑定(Self, 类型.读取)
        过滤 = _绑定(Self, 类型.过滤)
        应用 = _绑定(Self, 类型.应用)
    if 变体参数:
        读取 = _偏(读取, 变体参数)
        应用 = _偏(应用, 变体参数)
    Self.翻译流程(
        根, 取值(Self, 类型.文件匹配), 读取, 过滤, _分组键(Self, 类型.分组键), 应用,
        取值(Self, 类型.读并发), 取值(Self, 类型.写并发), 类型.日志类型,
        输出方法=_解析可调用(Self, 类型.输出方法), **总参数)
    return 根

def 执行(Self, 类型名: str, 根, **总参数):
    # 参数:
    #     Self:     Translator实例 隐式传入
    #     类型名:   翻译类型名
    #     根:       待翻目录 或 单个文件
    #     **总参数: 透传给流程(如 path2=参考包目录)
    # 返回:
    #     Any: 执行器/流程的产物

    执行器 = 取执行器(Self, 类型名)
    if 执行器 == "翻译流程":
        return 执行翻译流程(Self, 类型名, 根, **总参数)
    if isinstance(执行器, str):
        return _解析可调用(Self, 执行器)(根, **总参数)
    return 执行器(Self, 根, **总参数)


def _偏(可调用, 变体参数: Dict[str, Any]):
    # 参数:
    #     可调用:   读取/应用 方法 可为 None
    #     变体参数: {变体参数名: 变体值}
    # 返回:
    #     callable | None: 绑好变体的 partial; 传入 None 时返回 None

    if 可调用 is None:
        return None
    return partial(可调用, **变体参数)


def _分组键(Self, 值):
    # 参数:
    #     Self: Translator实例 隐式传入(只用来出本地化报错文案)
    #     值: "文件" / "上下文" / "位置" / 可调用 / None
    # 返回:
    #     callable | None: 分组键函数 项目 -> 组名; 未知取值抛 KeyError

    if callable(值) or 值 is None:
        return 值
    if 值 == "文件":
        return lambda 项目: 项目[0][0]
    if 值 == "上下文":
        return lambda 项目: str(项目[2])
    if 值 == "位置":
        return lambda 项目: str(项目[0])
    raise KeyError(Self.Lang("log.mods.group.key.unk", type=值))