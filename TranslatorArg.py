# VibeCoding 后面改
from dataclasses import field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.live import Live
from rich.rule import Rule

from TranslatorLib import (Config, DEFAULT_CONFIG, json, re, numpy, Path, shutil, dataclass,
                           threading, time, Log)

console = Console(force_terminal=True, color_system="auto")

# ============================================================
# 持久化配置（与 GUI 共享 GUISettings.json）
# ============================================================
持久化路径 = Path("GUISettings.json").resolve()

def 加载持久化配置() -> str:
    """从 GUISettings.json 加载上次保存的 JSON 配置字符串"""
    try:
        if 持久化路径.is_file():
            with open(持久化路径, "r", encoding="utf-8") as f:
                数据 = json.load(f)
            if isinstance(数据, dict):
                return 数据.get("settings_json", "{}")
    except Exception:
        pass
    return "{}"

def 保存持久化配置(内容: str):
    """把 JSON 配置字符串保存到 GUISettings.json"""
    try:
        with open(持久化路径, "w", encoding="utf-8") as f:
            json.dump({"settings_json": 内容}, f, ensure_ascii=False, indent=4)
    except Exception:
        pass

def 解析配置字典(文本: str) -> dict:
    """把 JSON 文本解析为配置字典（自动还原 f/i 前缀为 numpy 类型）"""
    配置 = json.loads(文本)

    def _还原(值):
        if isinstance(值, str):
            if re.match(r"^f-?\d+(\.\d+)?$", 值):
                return numpy.float32(float(值[1:]))
            if re.match(r"^i-?\d+$", 值):
                return numpy.int32(int(值[1:]))
            return 值
        if isinstance(值, dict):
            return {k: _还原(v) for k, v in 值.items()}
        if isinstance(值, list):
            return [_还原(v) for v in 值]
        return 值

    return _还原(配置)

def 获取配置简介映射() -> dict:
    """从 ConfigDescriptions.json 读取配置键的简介与介绍"""
    简介映射 = {}
    try:
        文件路径 = Path("ConfigDescriptions.json").resolve()
        if 文件路径.is_file():
            with open(文件路径, "r", encoding="utf-8") as f:
                原始数据 = json.load(f)
            for 键, 值 in 原始数据.items():
                if isinstance(值, list) and len(值) >= 2:
                    简介映射[键] = {"简介": 值[0], "介绍": 值[1]}
                elif isinstance(值, list) and len(值) == 1:
                    简介映射[键] = {"简介": 值[0], "介绍": ""}
                elif isinstance(值, str):
                    简介映射[键] = {"简介": 值, "介绍": ""}
    except Exception:
        pass
    return 简介映射

def 转JSON可序列化(值):
    """递归把 numpy 标量转换为 JSON 可序列化的 Python 类型"""
    if isinstance(值, numpy.integer):
        return int(值)
    if isinstance(值, numpy.floating):
        return float(值)
    if isinstance(值, dict):
        return {k: 转JSON可序列化(v) for k, v in 值.items()}
    if isinstance(值, (list, tuple)):
        return [转JSON可序列化(v) for v in 值]
    return 值

def 格式化默认值(默认值):
    """把默认值格式化为显示文本"""
    if isinstance(默认值, numpy.floating):
        return f"f{float(默认值)}"
    if isinstance(默认值, numpy.integer):
        return f"i{int(默认值)}"
    return json.dumps(转JSON可序列化(默认值), ensure_ascii=False)

def 解析值文本(值文本: str):
    """解析值文本：i数字 → numpy.int32，f数字 → numpy.float32，其他 → JSON"""
    文本 = str(值文本).strip()
    if re.match(r"^i-?\d+$", 文本):
        return numpy.int32(int(文本[1:]))
    if re.match(r"^f-?\d+(\.\d+)?$", 文本):
        return numpy.float32(float(文本[1:]))
    try:
        return json.loads(文本)
    except Exception:
        return 值文本

# ============================================================
# 任务数据
# ============================================================
@dataclass
class 任务数据:
    ID: str
    名称: str
    文件路径: str = ""
    日志: str = ""
    翻译中: bool = False
    翻译参数: dict = field(default_factory=lambda: {"file1": "", "all_mode": False})
    日志文件名: str = ""
    日志文件路径: str = ""
    日志缓冲: list = field(default_factory=list)
    日志锁: object = None
    刷新已调度: bool = False
    翻译完成: bool = False
    结果路径: str = ""

# ============================================================
# 主界面类
# ============================================================
class TranslatorArgApp:
    def __init__(Self):
        Self.任务列表 = []
        Self.当前任务 = None
        Self.任务序号 = 0
        Self.全局配置 = None
        Self.设置JSON = 加载持久化配置()
        Self.配置简介映射 = 获取配置简介映射()

    # ---------- 启动入口 ----------
    def run(Self):
        while True:
            try:
                Self._显示主菜单()
                选择 = Prompt.ask("\n[bold cyan]请选择操作[/]", default="1").lower()
                if 选择 == "1":
                    Self._翻译页面()
                elif 选择 == "2":
                    Self._设置页面()
                elif 选择 == "q":
                    Self._退出()
                    break
            except KeyboardInterrupt:
                console.print("\n  [dim]Ctrl+C 已忽略，请输入 q 退出[/]")
                continue

    # ---------- 主菜单 ----------
    def _显示主菜单(Self):
        运行状态 = "[green]● 运行中[/]" if Self.全局配置 else "[red]○ 未运行[/]"
        任务数 = len(Self.任务列表)
        console.print(Rule("[bold]主菜单[/]", style="blue"))
        console.print(f"  配置状态: {运行状态}    任务数: {任务数}\n")
        console.print("  [bold cyan]1[/]  翻译文件")
        console.print("  [bold cyan]2[/]  设置")
        console.print("  [bold cyan]q[/]  退出")

    # ========== 翻译页面 ==========
    def _翻译页面(Self):
        while True:
            console.print(Rule("[bold]翻译[/]", style="blue"))
            # 显示任务列表
            if Self.任务列表:
                表格 = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
                表格.add_column("#", width=4)
                表格.add_column("名称", min_width=20)
                表格.add_column("文件路径", min_width=30)
                表格.add_column("状态", width=12)
                for i, t in enumerate(Self.任务列表, 1):
                    当前标记 = " [bold yellow]◄[/]" if t is Self.当前任务 else ""
                    if t.翻译中:
                        状态 = "[yellow]翻译中[/]"
                    elif t.结果路径:
                        状态 = "[green]✓ 完成[/]"
                    elif t.文件路径:
                        状态 = "[cyan]就绪[/]"
                    else:
                        状态 = "[dim]待配置[/]"
                    表格.add_row(
                        str(i),
                        f"{t.名称}{当前标记}",
                        t.文件路径 if t.文件路径 else "[dim]无[/]",
                        状态
                    )
                console.print(表格)
            else:
                console.print("  [dim]暂无任务[/]")

            console.print()
            操作列表 = {
                "c": "选择任务",
                "n": "新建任务",
                "s": "选择文件",
                "p": "设置参数",
                "r": "开始翻译",
                "a": "开始全部",
                "m": "监控翻译",
                "l": "查看日志",
                "d": "保存结果",
                "b": "返回",
            }
            操作提示 = " | ".join(f"[bold]{k}[/]:{v}" for k, v in 操作列表.items())
            console.print(f"  {操作提示}")
            try:
                操作 = Prompt.ask("\n[bold cyan]操作[/]", default="b").lower()
            except (KeyboardInterrupt, EOFError):
                console.print("\n  [dim]Ctrl+C 已忽略，输入 b 返回主菜单[/]")
                continue

            if 操作 == "b":
                break
            elif 操作 == "c":
                Self._选择当前任务()
            elif 操作 == "n":
                Self._新建任务()
            elif 操作 == "s":
                Self._选择文件()
            elif 操作 == "p":
                Self._设置任务参数()
            elif 操作 == "r":
                Self._开始翻译()
            elif 操作 == "a":
                Self._开始全部()
            elif 操作 == "m":
                Self._监控翻译()
            elif 操作 == "l":
                Self._查看日志()
            elif 操作 == "d":
                Self._保存结果()

    def _新建任务(Self):
        Self.任务序号 += 1
        ID = f"任务_{Self.任务序号}"
        任务 = 任务数据(ID=ID, 名称=f"任务 {Self.任务序号}")
        Self.任务列表.append(任务)
        Self.当前任务 = 任务
        console.print(f"  [green]✓ 已创建：[/]{任务.名称}")

    def _选择当前任务(Self):
        """输入编号切换当前选中任务"""
        if not Self.任务列表:
            console.print("  [dim]暂无任务[/]")
            return
        for i, t in enumerate(Self.任务列表, 1):
            标记 = " ◄" if t is Self.当前任务 else ""
            状态 = "[yellow]翻译中[/]" if t.翻译中 else ("[green]✓[/]" if t.翻译完成 else "[dim]就绪[/]")
            console.print(f"    {i}. {t.名称}{标记} ({状态})")
        idx = IntPrompt.ask("  输入任务编号", default=str(Self.任务列表.index(Self.当前任务) + 1 if Self.当前任务 else 1))
        if 1 <= idx <= len(Self.任务列表):
            Self.当前任务 = Self.任务列表[idx - 1]
            console.print(f"  [green]✓ 已选中：[/]{Self.当前任务.名称}")
        else:
            console.print("  [red]无效编号[/]")

    def _确保有当前任务(Self) -> bool:
        if Self.当前任务 is None:
            if Self.任务列表:
                Self.当前任务 = Self.任务列表[-1]
            else:
                Self._新建任务()
        return True

    def _选择文件(Self):
        if not Self.任务列表:
            Self._新建任务()
        # 多任务时先选择目标任务
        if len(Self.任务列表) > 1:
            console.print("  选择目标任务:")
            for i, t in enumerate(Self.任务列表, 1):
                标记 = " ◄" if t is Self.当前任务 else ""
                console.print(f"    {i}. {t.名称}{标记}")
            idx = IntPrompt.ask("  任务编号", default=str(Self.任务列表.index(Self.当前任务) + 1 if Self.当前任务 else 1))
            if 1 <= idx <= len(Self.任务列表):
                Self.当前任务 = Self.任务列表[idx - 1]
        路径 = Prompt.ask("  请输入文件路径")
        路径 = 路径.strip().strip('"').strip("'")
        if not 路径:
            return
        Self.当前任务.文件路径 = str(Path(路径).resolve())
        Self.当前任务.名称 = Path(路径).name
        console.print(f"  [green]✓ 文件已选择：[/]{Self.当前任务.文件路径}")

    def _设置任务参数(Self):
        if not Self.当前任务:
            if not Self.任务列表:
                console.print("  [red]请先创建任务[/]")
                return
            # 选择任务
            for i, t in enumerate(Self.任务列表, 1):
                console.print(f"    {i}. {t.名称}")
            idx = IntPrompt.ask("  选择任务编号", default="1")
            if 1 <= idx <= len(Self.任务列表):
                Self.当前任务 = Self.任务列表[idx - 1]
            else:
                return

        任务 = Self.当前任务
        console.print(f"\n  [bold]任务参数 — {任务.名称}[/]")
        console.print(f"  当前 file1: [dim]{任务.翻译参数.get('file1', '') or '无'}[/]")
        console.print(f"  当前 all_mode: [dim]{任务.翻译参数.get('all_mode', False)}[/]")

        file1 = Prompt.ask("  参考文件 (file1，留空跳过)", default=任务.翻译参数.get("file1", ""))
        if file1:
            任务.翻译参数["file1"] = str(Path(file1.strip().strip('"').strip("'")).resolve())

        all_mode_str = Prompt.ask("  全量翻译 (all_mode) [y/n]",
                                   default="y" if 任务.翻译参数.get("all_mode", False) else "n")
        任务.翻译参数["all_mode"] = all_mode_str.lower() in ("y", "yes", "true", "1")

        console.print("  [green]✓ 参数已更新[/]")

    def _启动任务翻译(Self, 任务: 任务数据):
        """非阻塞启动单个任务的翻译（后台线程）"""
        if 任务.翻译中:
            console.print(f"  [yellow]{任务.名称} 正在翻译中[/]")
            return False
        if not 任务.文件路径:
            console.print(f"  [red]{任务.名称} 请先选择文件[/]")
            return False
        if Self.全局配置 is None:
            console.print("  [yellow]配置未运行，正在自动启动...[/]")
            if not Self._运行配置():
                return False

        任务.翻译中 = True
        任务.翻译完成 = False
        任务.日志 = ""
        任务.结果路径 = ""
        任务.日志文件名 = f"task_{任务.ID}_{int(time.time() * 1000)}"
        任务.日志缓冲 = []
        任务.日志锁 = threading.Lock()
        任务.刷新已调度 = False

        # 清理旧日志
        try:
            for 旧文件 in Path(Self.全局配置.Config.LOGS_FILE_PATH).glob(f"task_{任务.ID}_*.log"):
                try: 旧文件.unlink()
                except Exception: pass
        except Exception: pass

        def _执行():
            try:
                临时配置 = Self.全局配置.get_config_temporary({})
                临时配置.Config.LOGS_FILE_NAME = 任务.日志文件名
                临时配置.Config.LOGS_GLOBAL = False
                临时配置.Config.LOGS_FLUSH_INTERVAL = 0.01
                任务.日志文件路径 = str(Path(临时配置.Config.LOGS_FILE_PATH) / f"{任务.日志文件名}.log")
                try: 临时配置.Log.关闭()
                except Exception: pass
                临时配置.Log = Log(临时配置)
                临时配置.日志 = 临时配置.Log.写入日志
                翻译器 = 临时配置.get_translator()
                if not hasattr(翻译器, "Log"): 翻译器.Log = 临时配置.Log
                参数 = 任务.翻译参数
                结果 = 翻译器.翻译通用文件(
                    file0=任务.文件路径,
                    file1=参数.get("file1", ""),
                    all_mode=参数.get("all_mode", False),
                    export_inspection=False,
                )
                任务.结果路径 = str(结果) if 结果 else ""
            except Exception as e:
                任务.结果路径 = ""
                console.print(f"  [bold red]✗ {任务.名称} 翻译出错：[/]{e}")
            finally:
                任务.翻译中 = False
                任务.翻译完成 = True

        threading.Thread(target=_执行, daemon=True).start()
        console.print(f"  [bold green]▶ 已启动翻译：[/]{任务.名称}")
        return True

    def _开始翻译(Self):
        """启动当前选中任务的翻译（非阻塞）"""
        if not Self.当前任务:
            if not Self.任务列表:
                console.print("  [red]请先创建任务并选择文件[/]")
                return
            Self.当前任务 = Self.任务列表[-1]
        Self._启动任务翻译(Self.当前任务)

    def _开始全部(Self):
        """启动所有就绪任务的翻译"""
        就绪任务 = [t for t in Self.任务列表 if t.文件路径 and not t.翻译中 and not t.翻译完成]
        if not 就绪任务:
            console.print("  [dim]没有就绪的任务[/]")
            return
        启动数 = 0
        for 任务 in 就绪任务:
            if Self._启动任务翻译(任务):
                启动数 += 1
        console.print(f"  [green]已启动 {启动数} 个任务[/]")

    def _监控翻译(Self):
        """实时监控正在翻译的任务日志（阻塞直到全部完成或按 Ctrl+C 退出）"""
        运行中 = [t for t in Self.任务列表 if t.翻译中]
        if not 运行中:
            console.print("  [dim]没有正在翻译的任务[/]")
            return
        if len(运行中) == 1:
            Self._实时显示日志(运行中[0])
            return
        # 多任务：选择监控哪个
        console.print("  正在翻译的任务:")
        for i, t in enumerate(运行中, 1):
            console.print(f"    {i}. {t.名称}")
        console.print(f"    0. 全部")
        idx = IntPrompt.ask("  选择监控编号", default="0")
        if idx == 0:
            Self._实时显示多任务日志(运行中)
        elif 1 <= idx <= len(运行中):
            Self._实时显示日志(运行中[idx - 1])

    def _实时显示日志(Self, 任务: 任务数据):
        """实时显示单个任务的日志（阻塞直到完成或按 Ctrl+C 退出）"""
        console.print("  [dim]监控中... 按 Ctrl+C 退出监控（翻译不会停止）[/]")
        日志行数 = 0
        最近日志 = []
        try:
            with Live(console=console, refresh_per_second=4, vertical_overflow="visible") as live:
                while not 任务.翻译完成:
                    if 任务.日志文件路径 and Path(任务.日志文件路径).is_file():
                        try:
                            with open(任务.日志文件路径, "r", encoding="utf-8", errors="ignore") as f:
                                所有行 = f.read().strip().split('\n')
                            if len(所有行) > 日志行数:
                                最近日志.extend(l for l in 所有行[日志行数:] if l.strip())
                                日志行数 = len(所有行)
                                if len(最近日志) > 30: 最近日志 = 最近日志[-30:]
                        except Exception: pass
                    显示行 = "\n".join(最近日志[-25:]) if 最近日志 else "  [dim]等待日志输出...[/]"
                    live.update(Panel(显示行, title=f"[bold]翻译日志 — {任务.名称}[/] [dim](Ctrl+C 退出)[/]", border_style="blue", width=120))
                    time.sleep(0.25)
                # 最终读取
                if 任务.日志文件路径 and Path(任务.日志文件路径).is_file():
                    try:
                        with open(任务.日志文件路径, "r", encoding="utf-8", errors="ignore") as f:
                            所有行 = f.read().strip().split('\n')
                        最近日志.extend(l for l in 所有行[日志行数:] if l.strip())
                    except Exception: pass
                显示行 = "\n".join(最近日志[-25:]) if 最近日志 else ""
                live.update(Panel(显示行, title=f"[bold]翻译日志 — {任务.名称}[/]", border_style="green", width=120))
        except KeyboardInterrupt:
            console.print("\n  [dim]已退出监控（翻译在后台继续）[/]")
            return
        if 任务.结果路径:
            console.print(f"  [green]✓ {任务.名称} 完成[/] 输出: {任务.结果路径}")

    def _实时显示多任务日志(Self, 任务列表_):
        """同时监控多个任务的日志（Ctrl+C 退出监控）"""
        console.print("  [dim]监控中... 按 Ctrl+C 退出监控（翻译不会停止）[/]")
        日志状态 = {t.ID: {"行数": 0, "最近": []} for t in 任务列表_}
        try:
            with Live(console=console, refresh_per_second=2, vertical_overflow="visible") as live:
                while any(not t.翻译完成 for t in 任务列表_):
                    面板列表 = []
                    for 任务 in 任务列表_:
                        状态 = 日志状态[任务.ID]
                        if 任务.日志文件路径 and Path(任务.日志文件路径).is_file():
                            try:
                                with open(任务.日志文件路径, "r", encoding="utf-8", errors="ignore") as f:
                                    所有行 = f.read().strip().split('\n')
                                if len(所有行) > 状态["行数"]:
                                    状态["最近"].extend(l for l in 所有行[状态["行数"]:] if l.strip())
                                    状态["行数"] = len(所有行)
                                    if len(状态["最近"]) > 10: 状态["最近"] = 状态["最近"][-10:]
                            except Exception: pass
                        标记 = "[green]✓[/]" if 任务.翻译完成 else ("[yellow]翻译中[/]" if 任务.翻译中 else "[dim]待启动[/]")
                        内容 = "\n".join(状态["最近"][-8:]) if 状态["最近"] else "  [dim]等待...[/]"
                        面板列表.append(Panel(内容, title=f"{任务.名称} {标记} [dim](Ctrl+C 退出)[/]", border_style="blue", width=60))
                    from rich.columns import Columns as _Columns
                    live.update(_Columns(面板列表, padding=1))
                    time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\n  [dim]已退出监控（翻译在后台继续）[/]")
            return
        完成数 = sum(1 for t in 任务列表_ if t.翻译完成 and t.结果路径)
        console.print(f"  [green]{完成数}/{len(任务列表_)} 个任务已完成[/]")

    def _查看日志(Self):
        if not Self.当前任务:
            console.print("  [dim]无选中任务[/]")
            return
        任务 = Self.当前任务
        if 任务.日志文件路径 and Path(任务.日志文件路径).is_file():
            try:
                with open(任务.日志文件路径, "r", encoding="utf-8", errors="ignore") as f:
                    内容 = f.read()
                if 内容:
                    console.print(Panel(内容[-4000:], title=f"[bold]日志 — {任务.名称}[/]", border_style="blue"))
                else:
                    console.print("  [dim]日志为空[/]")
            except Exception as e:
                console.print(f"  [red]读取日志失败：{e}[/]")
        else:
            console.print("  [dim]暂无日志文件[/]")

    def _保存结果(Self):
        if not Self.当前任务:
            console.print("  [dim]无选中任务[/]")
            return
        任务 = Self.当前任务
        if not 任务.结果路径 or not Path(任务.结果路径).exists():
            console.print("  [yellow]暂无可保存的翻译结果[/]")
            return
        目标 = Prompt.ask("  保存到路径", default=str(Path(任务.结果路径).name))
        try:
            shutil.copy2(任务.结果路径, 目标)
            console.print(f"  [green]✓ 已保存到：{目标}[/]")
        except Exception as e:
            console.print(f"  [red]保存失败：{e}[/]")

    # ========== 设置页面 ==========
    def _设置页面(Self):
        while True:
            运行状态 = "[green]● 运行中[/]" if Self.全局配置 else "[red]○ 未运行[/]"
            console.print(Rule("[bold]设置[/]", style="blue"))
            console.print(f"  配置状态: {运行状态}\n")

            操作列表 = {
                "r": "运行配置",
                "x": "停止配置",
                "t": "重启配置",
                "v": "查看当前配置",
                "e": "编辑配置 (JSON)",
                "a": "添加配置项",
                "m": "快捷修改已有配置",
                "b": "返回",
            }
            操作提示 = " | ".join(f"[bold]{k}[/]:{v}" for k, v in 操作列表.items())
            console.print(f"  {操作提示}")
            try:
                操作 = Prompt.ask("\n[bold cyan]操作[/]", default="b").lower()
            except (KeyboardInterrupt, EOFError):
                console.print("\n  [dim]Ctrl+C 已忽略，输入 b 返回[/]")
                continue

            if 操作 == "b":
                break
            elif 操作 == "r":
                Self._运行配置()
            elif 操作 == "x":
                Self._停止配置()
            elif 操作 == "t":
                Self._停止配置()
                Self._运行配置()
            elif 操作 == "v":
                Self._查看配置()
            elif 操作 == "e":
                Self._编辑配置()
            elif 操作 == "a":
                Self._添加配置项()
            elif 操作 == "m":
                Self._快捷修改配置()

    def _运行配置(Self) -> bool:
        if Self.全局配置 is not None:
            console.print("  [yellow]配置已在运行中[/]")
            return True
        try:
            配置字典 = 解析配置字典(Self.设置JSON)
        except Exception as e:
            console.print(f"  [red]JSON 解析错误：{e}[/]")
            return False

        console.print("  [yellow]启动配置管理器中...[/]")

        def _创建():
            try:
                Self.全局配置 = Config(配置字典)
            except Exception as e:
                console.print(f"  [red]启动失败：{e}[/]")

        线程 = threading.Thread(target=_创建, daemon=True)
        线程.start()
        线程.join(timeout=120)

        if Self.全局配置 is not None:
            console.print("  [green]✓ 配置管理器已启动[/]")
            return True
        else:
            console.print("  [red]✗ 配置管理器启动失败[/]")
            return False

    def _停止配置(Self):
        if Self.全局配置 is None:
            console.print("  [dim]配置未运行[/]")
            return
        try:
            Self.全局配置.关闭()
        except Exception:
            pass
        Self.全局配置 = None
        console.print("  [dim]配置已停止[/]")

    def _查看配置(Self):
        if not Self.设置JSON or Self.设置JSON.strip() == "{}":
            console.print("  [dim]当前配置为空（使用默认配置）[/]")
            return
        try:
            数据 = json.loads(Self.设置JSON)
            格式化 = json.dumps(数据, ensure_ascii=False, indent=4)
        except Exception:
            格式化 = Self.设置JSON
        console.print(格式化)

    def _编辑配置(Self):
        console.print("  [dim]当前配置 JSON:[/]")
        console.print(Self.设置JSON if Self.设置JSON.strip() != "{}" else "{}")
        console.print("\n  [dim]输入新的 JSON 配置（输入空行结束）：[/]")
        行列表 = []
        while True:
            行 = input("  ")
            if 行.strip() == "":
                break
            行列表.append(行)
        if 行列表:
            新JSON = "\n".join(行列表)
            try:
                json.loads(新JSON)  # 验证 JSON
                Self.设置JSON = 新JSON
                保存持久化配置(Self.设置JSON)
                console.print("  [green]✓ 配置已保存[/]")
            except json.JSONDecodeError as e:
                console.print(f"  [red]JSON 格式错误：{e}[/]")

    def _快捷修改配置(Self):
        """快捷修改单个配置项（列表选择）"""
        while True:
            try:
                当前数据 = json.loads(Self.设置JSON) if Self.设置JSON.strip() else {}
            except Exception:
                当前数据 = {}

            键列表 = list(当前数据.keys())
            
            if not 键列表:
                console.print("  [yellow]当前没有已自定义的配置项。[/]")
                操作 = Prompt.ask("  是否前往添加配置项？(y/n)", default="n").lower()
                if 操作 in ("y", "yes"):
                    Self._添加配置项()
                return

            console.print(Rule("[bold]快捷修改已有配置[/]", style="blue"))
            表格 = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
            表格.add_column("#", width=4)
            表格.add_column("配置键", min_width=30)
            表格.add_column("当前值", min_width=20)
            表格.add_column("简介", min_width=30)

            for i, 键 in enumerate(键列表, 1):
                原始值 = 当前数据[键]
                # 格式化当前值用于显示
                if isinstance(原始值, str):
                    显示值 = 原始值
                else:
                    显示值 = json.dumps(转JSON可序列化(原始值), ensure_ascii=False)
                if len(显示值) > 40:
                    显示值 = 显示值[:37] + "..."
                
                简介信息 = Self.配置简介映射.get(键, {"简介": "", "介绍": ""})
                简介 = 简介信息.get("简介", "")
                
                表格.add_row(str(i), 键, 显示值, 简介)

            console.print(表格)
            console.print(f"\n  [dim]共 {len(键列表)} 个已自定义配置项[/]")
            
            选择 = Prompt.ask("  [bold cyan]输入序号修改 / a添加新项 / d删除项 / q返回[/]", default="q").strip().lower()
            
            if 选择 == "q":
                break
            elif 选择 == "a":
                Self._添加配置项()
                continue
            elif 选择 == "d":
                删序号 = Prompt.ask("  输入要删除的序号", default="")
                try:
                    idx = int(删序号)
                    if 1 <= idx <= len(键列表):
                        删键 = 键列表[idx - 1]
                        del 当前数据[删键]
                        Self.设置JSON = json.dumps(当前数据, ensure_ascii=False, indent=4)
                        保存持久化配置(Self.设置JSON)
                        console.print(f"  [green]✓ 已删除：{删键}[/]")
                    else:
                        console.print("  [red]无效序号[/]")
                except ValueError:
                    console.print("  [red]请输入数字[/]")
                continue
            else:
                try:
                    idx = int(选择)
                    if 1 <= idx <= len(键列表):
                        键名 = 键列表[idx - 1]
                        Self._修改指定配置(键名, 当前数据)
                    else:
                        console.print("  [red]无效序号[/]")
                except ValueError:
                    console.print("  [red]请输入数字序号或命令[/]")

    def _修改指定配置(Self, 键名: str, 当前数据: dict):
        简介信息 = Self.配置简介映射.get(键名, {"简介": "", "介绍": ""})
        console.print(f"\n  [cyan]配置项:[/] [bold]{键名}[/]")
        if 简介信息["简介"]:
            console.print(f"  [cyan]简介:[/] {简介信息['简介']}")
        if 简介信息["介绍"]:
            console.print(f"  [dim]{简介信息['介绍']}[/]")
        
        原始值 = 当前数据.get(键名)
        if isinstance(原始值, str):
            提示默认值 = 原始值
        else:
            提示默认值 = json.dumps(转JSON可序列化(原始值), ensure_ascii=False)
            
        新值文本 = Prompt.ask(f"  输入新值 (留空删除该项恢复默认)", default=提示默认值)
        
        if not 新值文本.strip():
            if 键名 in 当前数据:
                del 当前数据[键名]
                Self.设置JSON = json.dumps(当前数据, ensure_ascii=False, indent=4)
                保存持久化配置(Self.设置JSON)
                console.print(f"  [green]✓ 已删除：{键名} (将使用默认值)[/]")
            return
            
        值 = 解析值文本(新值文本)
        if isinstance(值, numpy.floating):
            当前数据[键名] = f"f{float(值)}"
        elif isinstance(值, numpy.integer):
            当前数据[键名] = f"i{int(值)}"
        else:
            当前数据[键名] = 转JSON可序列化(值)
            
        Self.设置JSON = json.dumps(当前数据, ensure_ascii=False, indent=4)
        保存持久化配置(Self.设置JSON)
        console.print(f"  [green]✓ 已更新：{键名}[/]")

    def _添加配置项(Self):
        """从默认配置中选择并添加配置项"""
        允许类型 = (str, int, float, bool, list, dict, type(None), tuple)
        允许类型 += (numpy.integer, numpy.floating)
        全部项 = []
        for 键 in dir(DEFAULT_CONFIG):
            if 键.startswith("_") or 键.upper().endswith("CONFIG"):
                continue
            try:
                默认值 = getattr(DEFAULT_CONFIG, 键)
            except Exception:
                continue
            if isinstance(默认值, 允许类型):
                简介信息 = Self.配置简介映射.get(键, {"简介": "", "介绍": ""})
                全部项.append({"键": 键, "简介": 简介信息["简介"], "介绍": 简介信息["介绍"], "默认值": 默认值})

        # 搜索过滤
        while True:
            关键词 = Prompt.ask("\n  [bold]搜索配置键名[/] (留空显示全部, q退出)", default="")
            if 关键词.lower() == "q":
                break

            if 关键词:
                过滤项 = [i for i in 全部项 if 关键词.lower() in i["键"].lower() or 关键词.lower() in i["简介"].lower()]
            else:
                过滤项 = 全部项

            if not 过滤项:
                console.print("  [dim]无匹配项[/]")
                continue

            # 分页显示
            每页 = 20
            总页 = max(1, (len(过滤项) + 每页 - 1) // 每页)
            页码 = 1

            while True:
                起始 = (页码 - 1) * 每页
                结束 = min(起始 + 每页, len(过滤项))

                表格 = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1), show_lines=False)
                表格.add_column("#", width=4)
                表格.add_column("配置键", min_width=35)
                表格.add_column("简介", min_width=20)
                表格.add_column("默认值", min_width=20)

                for i, 项 in enumerate(过滤项[起始:结束], 起始 + 1):
                    默认显示 = 格式化默认值(项["默认值"])
                    if len(默认显示) > 40:
                        默认显示 = 默认显示[:37] + "..."
                    表格.add_row(str(i), 项["键"], 项["简介"], 默认显示)

                console.print(表格)
                console.print(f"  [dim]显示 {起始+1}-{结束} / {len(过滤项)} 项 (第{页码}/{总页}页)[/]")

                操作 = Prompt.ask("  [bold]输入行号添加 / n下页 / p上页 / q返回[/]", default="q")
                if 操作.lower() == "q":
                    break
                elif 操作.lower() == "n" and 页码 < 总页:
                    页码 += 1
                elif 操作.lower() == "p" and 页码 > 1:
                    页码 -= 1
                else:
                    try:
                        行号 = int(操作)
                        if 1 <= 行号 <= len(过滤项):
                            项 = 过滤项[行号 - 1]
                            默认文本 = 格式化默认值(项["默认值"])
                            console.print(f"  [bold]{项['键']}[/]: {项['简介']}")
                            if 项["介绍"]:
                                console.print(f"  [dim]{项['介绍']}[/]")
                            值文本 = Prompt.ask(f"  输入值", default=默认文本)
                            Self._合并配置项(项["键"], 值文本)
                    except ValueError:
                        pass

    def _合并配置项(Self, 键: str, 值文本: str):
        """把单个配置项合并进设置 JSON"""
        if not 键:
            return
        值 = 解析值文本(值文本)
        try:
            数据 = json.loads(Self.设置JSON) if Self.设置JSON.strip() else {}
        except Exception:
            数据 = {}

        if isinstance(值, numpy.floating):
            数据[键] = f"f{float(值)}"
        elif isinstance(值, numpy.integer):
            数据[键] = f"i{int(值)}"
        else:
            数据[键] = 转JSON可序列化(值)

        Self.设置JSON = json.dumps(数据, ensure_ascii=False, indent=4)
        保存持久化配置(Self.设置JSON)
        console.print(f"  [green]✓ 已添加/修改：{键}[/]")

    # ========== 退出 ==========
    def _退出(Self):
        保存持久化配置(Self.设置JSON)
        if Self.全局配置 is not None:
            try:
                Self.全局配置.关闭()
            except Exception:
                pass
        console.print("[dim]已退出[/]")


if __name__ == "__main__":
    app = TranslatorArgApp()
    app.run()