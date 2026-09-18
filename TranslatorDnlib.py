from TranslatorLib import Path, eb, clr, Dnlib加载

_依赖提示 = "pythonnet" if clr is None else "dnlib"

class Dnlib:
    def __init__(Self, App):
        Self.Config = App.Config
        Self.日志 = App.日志
        Self._App = App
        Self._已尝试加载 = False
        Self._可加载 = False
        Self.失败原因 = ""
        Self.失败键 = ""
        Self.失败参数 = {}
        Self.OpCodes = None
        Self.WriterOptions = None
        Self.ModuleDefMD = None
    def 确保加载(Self) -> bool:
        if Self._已尝试加载:
            return Self._可加载
        Self._已尝试加载 = True
        if clr is None:
            Self.标记不可用(_依赖提示)
            return False
        try:
            Self.加载()
            Self._可加载 = True
        except Exception as e:
            Self.失败原因 = eb.format_exc()
            Self.失败键 = "log.module.dll.dnlib.load.error"
            Self.失败参数 = {"e": str(e)}
            Self.日志("log.module.dll.dnlib.load.error", e=str(e), info_level=2)
        return Self._可加载
    @property
    def 可用(Self) -> bool:
        return Self.确保加载()
    def 文本(Self, 键: str, **kwargs) -> str:
        Lang = getattr(Self._App, "Lang", None)
        if callable(Lang):
            try:
                return Lang(键, **kwargs)
            except Exception:
                pass
        return 键
    def 标记不可用(Self, e):
        Self.失败原因 = e
        Self.失败键 = "log.module.dll.dnlib.unavailable"
        Self.失败参数 = {"e": e}
    def 异常(Self, 键: str, **kwargs) -> RuntimeError:
        return RuntimeError(Self.文本(键, **kwargs))
    def 加载(Self):
        dll路径 = Path(f"{Self.Config.DNLIB_DLL_PATH}/{Self.Config.DNLIB_DLL_NAME}").resolve()
        if not dll路径.is_file():
            raise FileNotFoundError(Self.文本("log.module.dll.dnlib.not.found", path=str(dll路径)))
        模块 = Dnlib加载(dll路径)
        if 模块 is None:
            raise ImportError("pythonnet unavailable")
        Self.ModuleDefMD = 模块.DotNet.ModuleDefMD
        Self.OpCodes = 模块.DotNet.Emit.OpCodes
        Self.WriterOptions = 模块.DotNet.Writer.ModuleWriterOptions
    @staticmethod
    def 是否插件(mod) -> bool:
        try:
            for 类型 in mod.GetTypes():
                for 特性 in 类型.CustomAttributes:
                    if "BepInPlugin" in str(特性.TypeFullName):
                        return True
        except Exception:
            return True
        return False
    def 读取字符串(Self, 文件路径) -> list:
        if not Self.可用:
            Self.日志(Self.失败键 or "log.module.dll.dnlib.unavailable", info_level=2, **Self.失败参数)
            return []
        路径 = Path(文件路径)
        mod = None
        try:
            mod = Self.ModuleDefMD.Load(str(路径))
        except Exception as e:
            Self.日志("log.module.dll.native.skip", file=str(路径), e=type(e).__name__, info_level=3)
            return []
        try:
            if not Self.是否插件(mod):
                程序集 = ""
                try:
                    程序集 = str(mod.Assembly.FullName) if mod.Assembly else mod.Name
                except Exception:
                    程序集 = str(路径)
                Self.日志("log.module.dll.not.plugin", file=str(路径), assembly=程序集, info_level=3)
                return []
            结果 = []
            已见 = set()
            for 类型 in mod.GetTypes():
                for 方法 in 类型.Methods:
                    if not 方法.HasBody:
                        continue
                    for 指令 in 方法.Body.Instructions:
                        if 指令.OpCode != Self.OpCodes.Ldstr:
                            continue
                        v = 指令.Operand
                        if isinstance(v, str) and v and v not in 已见:
                            已见.add(v)
                            结果.append(v)
            return 结果
        finally:
            if mod is not None:
                try:
                    mod.Dispose()
                except Exception:
                    pass
    @staticmethod
    def 占位符集合(文本: str) -> frozenset:
        import re
        结果 = set()
        for m in re.finditer(r'\{[^{}]*\}', 文本):
            结果.add(m.group(0))
        for m in re.finditer(r'%(?:\d+\$)?[-+ 0#]*\d*(?:\.\d+)?[sdfoxXeEgGcCbB]', 文本):
            结果.add(m.group(0))
        return frozenset(结果)
    def 回写(Self, 文件路径, 替换字典: dict) -> dict:
        统计 = {"文件": str(文件路径), "请求": len(替换字典), "应用": 0,
                "占位符跳过": 0, "校验失败": 0, "回滚": False,
                "错误键": "", "错误参数": {}}
        if not Self.可用:
            键 = Self.失败键 or "log.module.dll.dnlib.unavailable"
            参数 = Self.失败参数 or {"e": Self.失败原因}
            Self.日志(键, info_level=2, **参数)
            统计.update(错误键=键, 错误参数=参数, 回滚=True)
            raise Self.异常(键, **参数)
        if not 替换字典:
            return 统计
        源 = Path(文件路径).resolve()
        临时 = 源.with_name(源.name + ".dnlibtmp")
        安全字典 = {}
        for 原文, 译文 in 替换字典.items():
            if not isinstance(译文, str) or not 译文:
                continue
            if Self.占位符集合(原文) != Self.占位符集合(译文):
                统计["占位符跳过"] += 1
                continue
            安全字典[原文] = 译文
        if not 安全字典:
            return 统计
        mod = None
        已替换 = set()
        try:
            mod = Self.ModuleDefMD.Load(str(源))
            期望序列 = []
            实际替换 = 0
            for 类型 in mod.GetTypes():
                for 方法 in 类型.Methods:
                    if not 方法.HasBody:
                        continue
                    for 指令 in 方法.Body.Instructions:
                        if 指令.OpCode != Self.OpCodes.Ldstr:
                            continue
                        v = 指令.Operand
                        if not isinstance(v, str):
                            continue
                        if v in 安全字典:
                            译文 = 安全字典[v]
                            指令.Operand = 译文
                            已替换.add(v)
                            if 译文 != v:
                                实际替换 += 1
                            v = 译文
                        期望序列.append(v)
            选项 = Self.WriterOptions(mod)
            mod.Write(str(临时), 选项)
            mod.Dispose()
            mod = None
            回读 = Self.ModuleDefMD.Load(str(临时))
            回读序列 = []
            try:
                for 类型 in 回读.GetTypes():
                    for 方法 in 类型.Methods:
                        if not 方法.HasBody:
                            continue
                        for 指令 in 方法.Body.Instructions:
                            if 指令.OpCode == Self.OpCodes.Ldstr and isinstance(指令.Operand, str):
                                回读序列.append(指令.Operand)
            finally:
                回读.Dispose()
            失败键 = ""
            失败参数 = {}
            if len(回读序列) != len(期望序列):
                统计["校验失败"] = abs(len(回读序列) - len(期望序列))
                失败键 = "log.module.dll.count.mismatch"
                失败参数 = {"file": str(源), "expect": len(期望序列), "actual": len(回读序列)}
            else:
                错位 = sum(1 for a, b in zip(期望序列, 回读序列) if a != b)
                if 错位:
                    统计["校验失败"] = 错位
                    失败键 = "log.module.dll.content.mismatch"
                    失败参数 = {"file": str(源), "count": 错位, "total": len(期望序列)}
            if 失败键:
                Self.日志(失败键, info_level=2, **失败参数)
                统计.update(错误键=失败键, 错误参数=失败参数, 回滚=True)
                raise Self.异常(失败键, **失败参数)
            临时.replace(源)
            统计["应用"] = 实际替换
        except Exception as e:
            统计["回滚"] = True
            if 临时.exists():
                try:
                    临时.unlink()
                except Exception:
                    pass
            if isinstance(e, RuntimeError) and 统计.get("错误键"):
                raise
            键 = "log.module.dll.write.rollback"
            参数 = {"file": str(源), "e": eb.format_exc()}
            Self.日志(键, info_level=2, **参数)
            统计.update(错误键=键, 错误参数=参数)
            raise Self.异常(键, **参数) from e
        finally:
            if mod is not None:
                try:
                    mod.Dispose()
                except Exception:
                    pass
        return 统计
