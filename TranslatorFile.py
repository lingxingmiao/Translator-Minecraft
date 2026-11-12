from TranslatorLib import (shutil, json, uuid, tarfile, zipfile, Path, eb, PurePosixPath, tomllib, libarchive, re,
                           List, Any, Callable, Optional, locale, np, Tuple,
                           Config, DEFAULT_CONFIG, Mods)

class File:
    def __init__(Self, App: Config):
        Self.Config = App.Config
        Self.Module = App.Module
        Self.Locale = App.Locale
        Self.日志 = App.日志
        Self.Lang = App.Lang
        Self.tqdm = App.RichTqdm
        Self.ttqdm = App.TqdmTqdm

    # ========== 通用工具 ==========
    def 设置路径值(Self, 数据, 路径, 译文):
        """沿路径钻取到目标容器并赋值，列表路径自动转数字索引"""
        obj = 数据
        for key in 路径[:-1]:
            if isinstance(obj, list):
                key = int(key)
            obj = obj[key]
        最后键 = 路径[-1]
        if isinstance(obj, list):
            最后键 = int(最后键)
        obj[最后键] = 译文

    def 写入Json文件(Self, 文件路径, 数据, 缩进=4):
        """以 UTF-8 无转义缩进格式写回 JSON 文件"""
        with open(文件路径, "w", encoding="utf-8", newline="") as f:
            json.dump(数据, f, ensure_ascii=False, indent=缩进)

    def 递归提取文本(Self, obj, 当前路径, 目标字段, 提取方法, 允许值类型=str):
        """深度遍历 dict/list，命中目标字段时调用 提取方法(路径, 值)"""
        if isinstance(obj, dict):
            for 键, 值 in obj.items():
                新路径 = 当前路径 + [键]
                if 键 in 目标字段 and isinstance(值, 允许值类型):
                    提取方法(新路径, 值)
                else:
                    Self.递归提取文本(值, 新路径, 目标字段, 提取方法, 允许值类型)
        elif isinstance(obj, list):
            for 序号, 元素 in enumerate(obj):
                Self.递归提取文本(元素, 当前路径 + [序号], 目标字段, 提取方法, 允许值类型)

    def 读取Json文件(Self, file):
        # 先按 BOM 判定编码: 带 UTF-8 BOM 的文件用 utf-8 解析时 ujson/json 会把 BOM 当非法字节
        # 直接报 "Expected object or value"(文件本身内容完整), 所以 BOM 文件显式走 utf-8-sig 剥离 BOM
        try:
            with open(file, 'rb') as f:
                文件头 = f.read(4)
        except OSError:
            文件头 = b''
        优先编码 = []
        if 文件头.startswith(b'\xef\xbb\xbf'):                                    # UTF-8 BOM
            优先编码.append('utf-8-sig')
        elif 文件头.startswith((b'\xff\xfe\x00\x00', b'\x00\x00\xfe\xff')):        # UTF-32 BOM(前缀与UTF-16相同, 需先判)
            优先编码.append('utf-32')
        elif 文件头.startswith((b'\xff\xfe', b'\xfe\xff')):                        # UTF-16 BOM
            优先编码.append('utf-16')
        for enc in 优先编码 + ['utf-8-sig', 'utf-8', 'gbk', 'utf-16', locale.getpreferredencoding(False)]:
            try:
                with open(file, 'r', encoding=enc) as f:
                    原始对象 = json.load(f)
                return 原始对象
            except (UnicodeDecodeError, ValueError):  # json.JSONDecodeError/ujson.JSONDecodeError 均为 ValueError 子类
                continue
        return None

    def Unicode转字符串(Self, s: str) -> str:
        return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

    def 字符串转Unicode(Self, s: str) -> str:
        parts = []
        for c in s:
            if c == '"':
                parts.append('\\"')
            elif c == '\\':
                parts.append('\\\\')
            elif ord(c) < 128 and c.isprintable():
                parts.append(c)
            elif ord(c) <= 0xFFFF:
                parts.append(f'\\u{ord(c):04X}')
            else:
                parts.append(c)
        return ''.join(parts)

    # ========== FTB Quests ==========
    def 读取语言文件(Self, file: str):
        后缀 = Path(file).suffix.lower()
        源文件 = []
        if 后缀 == ".json":
            # 统一走 读取Json文件: 已处理 UTF-8 BOM 与多编码回退
            Json文件 = Self.读取Json文件(file)
            if Json文件 is None:
                Self.日志("log.module.lang.load.error", mod="通用语言文件", file=file, e="JSON 解析失败(文件为空或编码不受支持)", info_level=2)
            else:
                源文件 = [f"{index}={Json文件[index]}" for index in Json文件]
        else:
            # .lang/.local 同样用 utf-8-sig: 部分模组文件带 UTF-8 BOM
            with open(file, "r", encoding="utf-8-sig") as f:
                if 后缀 == ".lang":
                    源文件 = f.read().splitlines()
                elif 后缀 == ".local":
                    源文件 = [re.sub(r'\s*=\s*', '=', line) for line in f.read().splitlines() if line.strip()]
        return [
            (lambda parts: [parts[0], parts[1], file])(line.split('=', 1))
            for line in 源文件
            if (stripped := line.strip()) and stripped and '=' in stripped and not stripped.startswith(('#', '//'))
        ], [源文件, file]

    def 保存语言文件(Self, file: str, 保存列表: list):
        # 原文件带 BOM 就继续用 utf-8-sig 写(保住 BOM); newline="" 关掉 Windows 的 \n -> \r\n 翻译
        try:
            with open(file, "rb") as 探测:
                编码 = "utf-8-sig" if 探测.read(3) == b"\xef\xbb\xbf" else "utf-8"
        except Exception:
            编码 = "utf-8"
        with open(file, "w+", encoding=编码, newline="") as f:
            if Path(file).suffix == ".lang":
                清理列表 = []
                for line in 保存列表:
                    if not line.strip():
                        清理列表.append(line)
                        continue
                    line = line.replace(r'\n', '<<NEWLINE_PLACEHOLDER>>')
                    line = line.replace(r'\r', '<<CARRIAGE_PLACEHOLDER>>')
                    line = line.replace('\r', ' ').replace('\n', ' ')
                    line = ' '.join(line.split())
                    line = line.replace('<<NEWLINE_PLACEHOLDER>>', r'\n')
                    line = line.replace('<<CARRIAGE_PLACEHOLDER>>', r'\r')
                    清理列表.append(line)
                f.write("\n".join(清理列表))
            elif Path(file).suffix == ".json":
                json文件 = {}
                保存列表 = [line for line in 保存列表 if line.strip() and '=' in line and not line.lstrip().startswith(('//', '#'))]
                for index in 保存列表:
                    k, v = index.split('=', 1)
                    try:
                        v = json.loads(v)
                    except Exception: pass
                    json文件[k] = v
                def 清洗数据(obj):
                    if isinstance(obj, dict):
                        return {kk: 清洗数据(vv) for kk, vv in obj.items()}
                    elif isinstance(obj, (list, tuple, set, frozenset)):
                        return [清洗数据(item) for item in obj]
                    elif hasattr(obj, 'isnan') and callable(getattr(obj, 'isnan')):
                        try:
                            if float(obj) != float(obj):
                                return None
                        except (TypeError, ValueError):
                            pass
                    elif isinstance(obj, (np.floating,)):
                        return None if obj != obj else float(obj)
                    elif isinstance(obj, (np.integer,)):
                        return int(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                清理后 = 清洗数据(json文件)
                f.write(json.dumps(清理后, ensure_ascii=False, indent=4))
            elif Path(file).suffix == ".local":
                f.write("\n".join([re.sub(r'\s*=\s*', ' = ', line) for line in 保存列表]))
        if Self.Config.LANGUAGE_VARIANT and Path(file).suffix.lower() == ".lang" and Path(file).stem == Path(file).stem.lower():
            语言变体 = Self.Module.语言代码大小写互补(Path(file).stem)
            if 语言变体 and 语言变体 != Path(file).stem:
                Self.保存语言文件(str(Path(file).with_name(f"{语言变体}{Path(file).suffix}")), 保存列表)

    # ========== 压缩包与资源文件 ==========
    def 读取压缩文件(Self, file_path: str, cache_path: str, original_language: str, target_language: str):
        try:
            for _ in Self.tqdm(range(1), desc="tqdm.file.read"):
                with zipfile.ZipFile(file_path, 'r') as f:
                    文件列表 = f.namelist()
                    缓存路径 = cache_path
                    目标集合 = {"shaders", "contenttweaker"}
                    存在目标 = any(
                        p.rstrip('/').split('/')[-1].lower() in 目标集合
                        for p in 文件列表 if p.rstrip('/').count('/') <= 1
                    )
                    可用文件列表 = [存在目标]
                    if 存在目标:
                        f.extractall(缓存路径)
                    待查语言 = {original_language.lower(), target_language.lower()}
                    for 内部文件 in 文件列表:
                        if f.getinfo(内部文件).is_dir():
                            continue
                        主体 = Path(内部文件).stem.lower()
                        if 主体 in 待查语言:
                            完整路径 = f"{缓存路径}/{内部文件}"
                            if not Path(完整路径).exists():
                                f.extract(内部文件, path=缓存路径)
                            可用文件列表.append([主体, 完整路径])
            return 可用文件列表
        except Exception:
            Self.日志("log.module.zip.read.error", e=eb.format_exc(), info_level=3)
            raise FileNotFoundError(Self.Lang("log.module.zip.read.error", e=eb.format_exc()))

    def 读取资源文件(Self, file0: str, file1: str = "", read_error: bool = True):
        Self.日志("log.core.file.read.start", file0=file0, file1=file1)
        文件0, 文件0源文件, 文件1 = [], [], []
        压缩路径, 输出扩展名, file2 = "", "", ""
        语言输入 = Self.Config.LANGUAGE_INPUT.lower()
        语言输出 = Self.Config.LANGUAGE_OUTPUT.lower()
        def 解析并追加(路径列表, 目标列表, 源文件列表=None):
            nonlocal 输出扩展名
            for fpath in 路径列表:
                解析数据, 源数据 = Self.读取语言文件(fpath)
                目标列表.extend(解析数据)
                if 源文件列表 is not None:
                    源文件列表.append(源数据)
                输出扩展名 = Path(fpath).suffix

        def 校验并获取路径(zip_ret, 目标语言, 日志键):
            匹配路径 = [item[1] for item in zip_ret[1:] if isinstance(item, list) and item[0] == 目标语言]
            if not 匹配路径 and read_error:
                Self.日志(日志键, info_level=3)
                raise FileNotFoundError(Self.Lang(日志键))
            return 匹配路径
        是压缩包0 = Path(file0).suffix.lower() in 归档可写后缀集()
        是压缩包1 = file1 and Path(file1).suffix.lower() in 归档可写后缀集()
        if 是压缩包0:
            实际解压目录 = Path(Self.Config.PATH_CACHE) / f"{uuid.uuid4().hex}_{Path(file0).stem}"
            压缩路径 = str(实际解压目录)
            file2 = Self.读取压缩文件(file0, 压缩路径, Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT)
            输入路径 = 校验并获取路径(file2, 语言输入, "log.module.read.file0.not.lang.error")
            解析并追加(输入路径, 文件0, 文件0源文件)
            if 是压缩包1:
                实际解压目录1 = Path(Self.Config.PATH_CACHE) / f"{uuid.uuid4().hex}_{Path(file1).stem}"
                file3 = Self.读取压缩文件(file1, str(实际解压目录1), Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT)
                输出路径 = 校验并获取路径(file3, 语言输出, "log.module.read.file1.not.lang.error")
                解析并追加(输出路径, 文件1)
            elif file1:
                解析并追加([file1], 文件1)
            else:
                输出路径 = [item[1] for item in file2[1:] if isinstance(item, list) and item[0] == 语言输出]
                解析并追加(输出路径, 文件1)
        elif Path(file0).suffix.lower() in {".lang", ".json", ".local"}:
            解析并追加([file0], 文件0, 文件0源文件)
            if file1:
                if 是压缩包1:
                    实际解压目录1 = Path(Self.Config.PATH_CACHE) / f"{uuid.uuid4().hex}_{Path(file1).stem}"
                    压缩路径 = str(实际解压目录1)
                    file3 = Self.读取压缩文件(file1, 压缩路径, Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT)
                    输出路径 = 校验并获取路径(file3, 语言输出, "log.module.read.file1.not.lang.error")
                    解析并追加(输出路径, 文件1)
                else:
                    解析并追加([file1], 文件1)
        Self.日志("log.core.file.read.end", file0=文件0, file1=文件1)
        return 文件0, 文件0源文件, 文件1, 压缩路径, 输出扩展名, file2

    # ========== 模组 ID ==========
    def 从资源包文件夹获取I18n翻译模组ID(Self, 路径: str):
        Self.日志("log.core.modid.get.start", info_level=0)
        try:
            with zipfile.ZipFile(list((Path(路径) / "resourcepacks").glob("Minecraft-Mod-Language-Modpack*")), 'r') as f:
                模组ID集 = set()
                for index in f.namelist():
                    Path路径 = PurePosixPath(index)
                    if Path路径.parts and Path路径.parts[0] == 'assets':
                        if len(Path路径.parts) > 1:
                            模组ID = Path路径.parts[1]
                            模组ID集.add(模组ID)
            返回内容 = list(模组ID集)
        except Exception:
            Self.日志("log.module.pack.i18n.no", info_level=3)
            返回内容 = []
        Self.日志("log.core.modid.get.end", info_level=0)
        return 返回内容

    def 从模组文件夹获取模组ID(Self, 路径: str):
        Self.日志("log.core.modid.get.start", info_level=0)
        所有模组路径 = Path(f"{路径}/{"mods"}").glob('*.jar')
        模组ID列表 = []
        for 模组文件路径 in Self.ttqdm(所有模组路径, desc="tqdm.file.modid.get"):
            try:
                with zipfile.ZipFile(模组文件路径, 'r') as 压缩文件:
                    文件列表 = 压缩文件.namelist()
                    模组ID = None
                    if "fabric.mod.json" in 文件列表:
                        try:
                            with 压缩文件.open("fabric.mod.json") as 文件:
                                数据 = json.load(文件)
                            模组ID = 数据.get("id")
                        except Exception:
                            Self.日志("log.module.parsemodid.error", e=eb.format_exc())
                    if "META-INF/mods.toml" in 文件列表 and 模组ID is None:
                        try:
                            with 压缩文件.open("META-INF/mods.toml") as 文件:
                                数据 = tomllib.load(文件)
                            模组列表 = 数据.get('mods', [])
                            if 模组列表:
                                模组ID = 模组列表[0].get('modId')
                        except Exception:
                            Self.日志("log.module.parsemodid.error", e=eb.format_exc())
                    if "META-INF/neoforge.mods.toml" in 文件列表 and 模组ID is None:
                        try:
                            with 压缩文件.open("META-INF/neoforge.mods.toml") as 文件:
                                数据 = tomllib.load(文件)
                            模组列表 = 数据.get('mods', [])
                            if 模组列表:
                                模组ID = 模组列表[0].get('modId')
                        except Exception:
                            Self.日志("log.module.parsemodid.error", e=eb.format_exc())
                    if 模组ID is None:
                        候选文件列表 = ["mcmod.info"]
                        if "mcmod.info" not in 文件列表:
                            候选文件列表 = [文件名 for 文件名 in 文件列表 if 文件名.endswith('.info')]
                        for 信息文件 in 候选文件列表:
                            try:
                                with 压缩文件.open(信息文件) as 文件:
                                    内容 = 文件.read().decode('utf-8-sig')
                                    数据列表 = json.loads(内容)
                                    if isinstance(数据列表, list) and len(数据列表) > 0:
                                        模组ID = 数据列表[0].get('modid')
                                        break
                            except Exception:
                                continue
                    模组ID列表.append([模组ID, 模组文件路径.name])
            except Exception:
                Self.日志("log.module.parsemodid.file.error", e=eb.format_exc())
        Self.日志("log.core.modid.get.end", info_level=0)
        return 模组ID列表

    def 从模组读取游戏版本(Self, 模组文件: str, 模组ID: str = ""):
        try:
            with zipfile.ZipFile(模组文件, 'r') as 压缩文件:
                文件列表 = 压缩文件.namelist()
                for 元数据文件 in ("META-INF/mods.toml", "META-INF/neoforge.mods.toml"):
                    if 元数据文件 not in 文件列表:
                        continue
                    try:
                        with 压缩文件.open(元数据文件) as 文件:
                            数据 = tomllib.load(文件)
                        if 模组ID:
                            依赖列表 = 数据.get('dependencies', {}).get(模组ID, [])
                            for 依赖 in 依赖列表:
                                if 依赖.get('modId') == "minecraft" and 依赖.get('versionRange'):
                                    版本 = Self._提取版本号(str(依赖['versionRange']))
                                    if 版本: return 版本
                        for 段依赖 in 数据.get('dependencies', {}).values():
                            for 依赖 in 段依赖:
                                if 依赖.get('modId') == "minecraft" and 依赖.get('versionRange'):
                                    版本 = Self._提取版本号(str(依赖['versionRange']))
                                    if 版本: return 版本
                    except Exception:
                        continue
                if "fabric.mod.json" in 文件列表:
                    try:
                        with 压缩文件.open("fabric.mod.json") as 文件:
                            数据 = json.load(文件)
                        for 来源 in ("depends", "recommends", "suggests"):
                            约束 = 数据.get(来源, {}).get("minecraft")
                            if isinstance(约束, str) and 约束:
                                版本 = Self._提取版本号(约束)
                                if 版本: return 版本
                            elif isinstance(约束, dict) and 约束.get("version"):
                                版本 = Self._提取版本号(str(约束["version"]))
                                if 版本: return 版本
                    except Exception:
                        pass
                候选文件列表 = ["mcmod.info"] if "mcmod.info" in 文件列表 else [文件名 for 文件名 in 文件列表 if 文件名.endswith('.info')]
                for 信息文件 in 候选文件列表:
                    try:
                        with 压缩文件.open(信息文件) as 文件:
                            内容 = 文件.read().decode('utf-8-sig')
                            数据列表 = json.loads(内容)
                            if isinstance(数据列表, list) and len(数据列表) > 0:
                                版本 = Self._提取版本号(str(数据列表[0].get('mcversion') or 数据列表[0].get('mcVersion') or ""))
                                if 版本: return 版本
                    except Exception:
                        continue
        except Exception:
            pass
        匹配 = re.search(r"(?:^|[-_])(1\.\d{1,2}(?:\.\d{1,2})?)", Path(str(模组文件)).stem)
        if 匹配:
            return 匹配.group(1)
        return None

    def _提取版本号(Self, 约束: str):
        匹配 = re.search(r"(1\.\d{1,2}(?:\.\d{1,2})?)", 约束)
        return 匹配.group(1) if 匹配 else None

    def 从模组文件夹获取游戏版本(Self, 路径: str) -> bool:
        版本计数 = {}
        for 模组文件路径 in Self.ttqdm(Path(f"{路径}/{"mods"}").glob('*.jar'), desc="tqdm.file.modver.get"):
            try:
                版本 = Self.从模组读取游戏版本(str(模组文件路径))
                if 版本:
                    版本计数[版本] = 版本计数.get(版本, 0) + 1
            except Exception:
                continue
        if not 版本计数:
            return False
        最高版本 = max(版本计数.items(), key=lambda 项: (项[1], len(项[0].split('.')), tuple(int(段) for 段 in 项[0].split('.'))))[0]
        版本段 = 最高版本.split('.')
        主, 次 = (int(版本段[0]), int(版本段[1])) if len(版本段) >= 2 else (int(版本段[0]), 0)
        return (主, 次) >= (1, 6) and (主, 次) <= (1, 10)

    # ========== 审查文件 ==========
    def 读取审查文件(Self, file0: str):
        返回列表 = []
        with open(file0, 'r', encoding='utf-8') as 文件对象:
            数据字典 = json.load(文件对象)
        for 键名, 键值 in 数据字典.items():
            返回列表.append([键名, 键值[0], 键值[1]])
        return 返回列表

    # ========== 归档流程 ==========
    def 实例根定位(Self, 缓存文件夹):
        """解压根 → 真实实例根(启动器套壳/单层包装), 并把 overrides 之类覆盖目录摊平到根。"""
        根 = Path(缓存文件夹)
        候选 = [根] + [p for p in 根.iterdir() if p.is_dir()]
        for 目录 in 候选:
            if any((目录 / 名).is_file() for 名 in ("manifest.json", "modrinth.index.json", "instance.cfg", "mmc-pack.json")):
                根 = 目录
                break
        else:
            子目录 = [p for p in 根.iterdir() if p.is_dir()]
            if len(子目录) == 1 and not any((根 / 名).exists() for 名 in ("mods", "config", "scripts", "kubejs")):
                根 = 子目录[0]
        for 覆盖 in ("overrides", "client-overrides", "minecraft", ".minecraft"):
            源 = 根 / 覆盖
            if 源.is_dir():
                shutil.copytree(源, 根, dirs_exist_ok=True)
                shutil.rmtree(源, ignore_errors=True)
        return 根

    def 归档执行一个计划(Self, Translator, file0, file1, 实例根, 计划, 缓存文件夹, 缓存文件夹2, export_inspection=False):
        """执行归档里的一个翻译计划。

        Translator: 核心实例——计划的执行器(翻译语言文件/翻译流程)与 本次读取条目数 都挂在核心上,
        所以这里显式接收而不是用 File 自己的 Self。
        """
        类型 = Mods().翻译类型注册表.get(计划.类型名)
        显示名 = (getattr(类型, "显示名", None) or 计划.类型名) if 类型 else 计划.类型名
        Translator.本次读取条目数 = 0
        if 类型 is not None and 类型.执行器 == "翻译语言文件":
            Translator.日志("log.core.translator.general.model", model=显示名, info_level=0)
            产出 = Translator.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
            if not Translator.本次读取条目数:
                Translator.日志("log.core.translator.general.nothing.read", path=f"{计划.类型名}@{计划.根 or '.'}", info_level=2)
                return None
            return Path(产出)
        子根 = 计划.根
        计划.根 = str(Path(实例根) / 子根) if 子根 else str(实例根)
        Translator.日志("log.core.translator.general.model", model=显示名, info_level=0)
        Mods().执行计划(Translator, 计划, 原始路径=计划.根, 输出目录=缓存文件夹, path2=缓存文件夹2)
        if not Translator.本次读取条目数:
            Translator.日志("log.core.translator.general.nothing.read", path=f"{计划.类型名}@{计划.根 or '.'}", info_level=2)
            return None
        return True

    def 模组内翻译产物(Self, Translator, 模组路径, 收集目录, **总参数):
        """mods/*.jar 里"挂在模组档、但不是语言文件"的翻译计划 → 解压、原地跑 → 产出收进收集目录。

        整合包流程对每个 jar 只跑语言文件流程(见 TranslatorCore.翻译整合包), 所以 GuideME 这种
        把译文写成 <指南根>/_<语言>/ 的类型在整合包模式下会整包漏掉, 这里补上。
        只收"新增的、或与原包内容不同"的文件, 按 jar 内原路径写进收集目录(整合包资源包缓存);
        语言文件相关的计划跳过(那边已经处理过)。读不了原包(非 zip 归档)时只收新增文件。
        这一步不看 I18n 资源包 —— 那里面只有语言文件, 和指南页无关。
        解压前先用"挂载声明的打包扩展名"和"类型自己的文件探测"过滤, 普通 mod 不会被白解压一遍。
        """
        模组路径 = Path(模组路径)
        if not 模组路径.is_file():
            return 0
        try:
            视图 = 归档视图(模组路径)
        except Exception:
            return 0
        计划列表, 后缀集 = [], set()
        for 计划 in Mods().生成计划(视图, "模组"):
            类型 = Mods().翻译类型注册表.get(计划.类型名)
            if 类型 is None or 类型.执行器 == "翻译语言文件":
                continue
            计划列表.append(计划)
            if 计划.打包扩展名:
                后缀集 |= {str(后缀).lower() for 后缀 in 计划.打包扩展名}
        if not 计划列表:
            return 0
        if 后缀集 and not any(Path(名).suffix.lower() in 后缀集
                              for 名 in 视图.namelist if not 名.endswith("/")):
            return 0                          # 包里连这类文件都没有, 别白解压一遍
        需要处理 = False
        for 计划 in 计划列表:
            类型 = Mods().翻译类型注册表.get(计划.类型名)
            探测 = getattr(类型, "文件探测", None) if 类型 is not None else None
            if 探测 is None:
                需要处理 = True               # 没有探测函数: 只能解压看
                break
            try:
                if 探测(Translator, 模组路径):
                    需要处理 = True
                    break
            except Exception:
                continue
        if not 需要处理:
            return 0                          # 有 .md 但不是指南页(例如 lang/README.md): 不解压
        临时 = Path(f"{Translator.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
        try:
            归档解压(模组路径, 临时, 条目上限=Translator.Config.ARCHIVE_MAX_ENTRIES,
                     字节上限=Translator.Config.ARCHIVE_MAX_BYTES)
        except Exception:
            shutil.rmtree(临时, ignore_errors=True)
            return 0
        try:
            原有文件 = {文件.relative_to(临时).as_posix() for 文件 in 临时.rglob("*") if 文件.is_file()}
            for 计划 in 计划列表:
                子根 = 计划.根
                计划.根 = str(临时 / 子根) if 子根 else str(临时)
                try:
                    Mods().执行计划(Translator, 计划, 原始路径=计划.根, 输出目录=临时, **总参数)
                except Exception:
                    Translator.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=2)
            候选 = [文件 for 文件 in 临时.rglob("*") if 文件.is_file()
                    and (not 后缀集 or 文件.suffix.lower() in 后缀集)]
            原包数据, 可比较 = {}, True
            try:
                with zipfile.ZipFile(模组路径) as 原包:
                    原包名 = set(原包.namelist())
                    for 文件 in 候选:
                        相对 = 文件.relative_to(临时).as_posix()
                        if 相对 in 原包名:
                            try:
                                原包数据[相对] = 原包.read(相对)
                            except Exception:
                                pass
            except Exception:
                可比较 = False
            收集 = 0
            for 文件 in 候选:
                相对 = 文件.relative_to(临时).as_posix()
                try:
                    数据 = 文件.read_bytes()
                except Exception:
                    continue
                if 可比较:
                    if 原包数据.get(相对) == 数据:
                        continue              # 和原包里一模一样 = 没翻出东西
                elif 相对 in 原有文件:
                    continue                  # 读不了原包又不是新文件: 判不了就不收
                目标 = Path(收集目录) / 相对
                目标.parent.mkdir(parents=True, exist_ok=True)
                目标.write_bytes(数据)
                收集 += 1
            return 收集
        finally:
            shutil.rmtree(临时, ignore_errors=True)

    def 打包硬编码包(Self, 原文件, 解压根, 语言包, Translator):
        """单模组硬编码产物打包：只带运行时 + 替换文本 + 语言资源包，
        不再分发原模组或翻译后的模组 jar。
        """
        原文件 = Path(原文件)
        解压根 = Path(解压根)
        语言包 = Path(语言包) if 语言包 else None
        语言 = str(getattr(Translator.Config, "LANGUAGE_OUTPUT", "zh_cn") or "zh_cn")
        输出 = 解压根 / f"{原文件.stem}-Hardcoded-{语言}.zip"
        模块目录 = 解压根 / "vaultpatcher" / "modules"
        运行时 = []
        库目录 = Path(getattr(Translator.Config, "LIB_PATH", "./libs"))
        if not 库目录.is_absolute():
            库目录 = (Path(__file__).resolve().parent / 库目录).resolve()
        清单路径 = 解压根 / "vaultpatcher" / "translatorvault_runtime.json"
        清单 = None
        if 清单路径.is_file():
            try:
                清单 = json.loads(清单路径.read_text(encoding="utf-8"))
            except Exception:
                清单 = None
        if isinstance(清单, dict):
            for name in 清单.get("mods", []):
                源 = 库目录 / name
                if 源.is_file():
                    运行时.append((源, f"mods/{name}"))
            for name in 清单.get("plugins", []):
                源 = 库目录 / name
                if 源.is_file():
                    运行时.append((源, f"vaultpatcher/plugins/{name}"))
        else:
            候选 = []
            if 库目录.is_dir():
                for 模式 in ("vaultpatcher*.jar",):
                    候选.extend(sorted(库目录.glob(模式)))
            for jar in 候选:
                if not jar.is_file():
                    continue
                运行时.append((jar.resolve(), f"mods/{jar.name}"))

        with zipfile.ZipFile(输出, "w", zipfile.ZIP_DEFLATED) as 包:
            for jar, arc in 运行时:
                包.write(jar, arcname=arc)
            if 模块目录.is_dir():
                for 模块 in sorted(模块目录.glob("*.json")):
                    包.write(模块, arcname=f"vaultpatcher/modules/{模块.name}")
            配置目录 = 解压根 / "config" / "vaultpatcher_asm"
            if 配置目录.is_dir():
                for 配置 in sorted(配置目录.glob("*.json")):
                    包.write(配置, arcname=f"config/vaultpatcher_asm/{配置.name}")
            if 语言包 is not None and 语言包.is_file() and 语言包.suffix.lower() == ".zip":
                包.write(语言包, arcname=f"resourcepacks/{原文件.stem}-lang-{语言}.zip")
        Translator.日志("log.core.translator.succeed", path=str(输出), info_level=0)
        return 输出

    def 打包整个包(Self, file0, 解压根):
        """就地翻译过的解压根 → 重新压成一个归档; 能原样写回就沿用原后缀, 否则补默认后缀。"""
        名字 = Path(file0).name
        if Path(file0).suffix.lower() not in 归档可写后缀集():
            名字 = 归档后缀模式().sub("", 名字) + 归档默认后缀()
        输出 = Path(解压根) / 名字
        with zipfile.ZipFile(输出, 'w', zipfile.ZIP_DEFLATED) as 包:
            for 文件 in Path(解压根).rglob('*'):
                if 文件.is_file() and 文件 != 输出:
                    包.write(文件, arcname=文件.relative_to(解压根).as_posix())
        return 输出

    def 打包整合包(Self, Translator, 实例根, 缓存文件夹, all_mode: bool = False):
        """翻译整个整合包实例, 再按 压缩路径映射 把产物收进 ModPack-Translation-Addion.zip。

        Translator: 核心实例——翻译整合包 挂在核心上, 所以显式接收。
        """
        压缩路径映射 = Translator.翻译整合包(Path(实例根), all_mode=all_mode)
        输出Zip路径 = Path(缓存文件夹) / "ModPack-Translation-Addion.zip"
        with zipfile.ZipFile(输出Zip路径, 'w', zipfile.ZIP_DEFLATED) as 包:
            for 相对路径, 类型列表 in 压缩路径映射.items():
                类型 = 类型列表[0] if 类型列表 else ""
                打包扩展名 = 类型列表[1] if len(类型列表) > 1 else None
                真实文件路径 = Path(实例根) / 相对路径.lstrip('/')
                if 类型 == "file":
                    包.write(真实文件路径, arcname=相对路径.lstrip('/'))
                elif 类型 == "path":
                    for 文件完整路径 in 真实文件路径.rglob('*'):
                        if 文件完整路径.is_file() and (打包扩展名 is None or 文件完整路径.suffix.lower() in 打包扩展名):
                            包.write(文件完整路径, arcname=文件完整路径.relative_to(实例根).as_posix())
        return 输出Zip路径


def 配置项(名称, 默认):
    """按 当前配置 > 默认配置 > 传入默认值 取配置项; 配置未就绪时安全回退。"""
    try:
        配置 = getattr(Mods(), "当前配置", None)
    except Exception:
        配置 = None
    值 = getattr(配置, 名称, None)
    if 值 is None:
        值 = getattr(DEFAULT_CONFIG, 名称, None)
    return 默认 if 值 is None else 值


def 归档配置项(名称, 默认):
    """归档相关配置项; 等价于 配置项(), 保留名称以表明归档语义。"""
    return 配置项(名称, 默认)


def 归档后缀表():
    """归档后缀注册表 Config.ARCHIVE_SUFFIXES 的容错读取; 缺失时回退到默认值。"""
    return tuple((str(后缀).lower(), str(后端)) for 后缀, 后端 in 归档配置项("ARCHIVE_SUFFIXES", ()))


def 归档后缀集() -> set:
    """全部归档后缀(上传识别范围), 由配置注册表展开, 不允许各调用点各自硬编码。"""
    return {后缀 for 后缀, _ in 归档后缀表()}


def 归档可写后缀集() -> set:
    """能原样写回的后缀: 注册表里归 zipfile 后端(可读可写)的项; tarfile/libarchive 专属只读。"""
    return {后缀 for 后缀, 后端 in 归档后缀表() if 后端 == "zipfile"} | {".zip"}


def 归档输出后缀集() -> set:
    """写归档时唯一可用的后缀; 由配置注册表与 zipfile 能力交集派生。"""
    可写 = {str(后缀).lower() for 后缀 in 归档配置项("ARCHIVE_WRITABLE_SUFFIXES", (".zip",))}
    return {后缀 for 后缀 in 归档可写后缀集() if 后缀 in 可写}


def 归档默认后缀() -> str:
    """输出名不带可写归档后缀时补的后缀。"""
    return sorted(归档输出后缀集())[0]


def 归档后缀模式() -> re.Pattern:
    """按配置注册表编译"剥离归档后缀"正则; 长后缀优先, 配置改动后自动跟随。"""
    后缀列表 = sorted(归档后缀集(), key=len, reverse=True)
    if not 后缀列表:
        return re.compile(r"(?!)")
    return re.compile(r"(?i)(" + "|".join(re.escape(后缀) for 后缀 in 后缀列表) + r")$")


def 归档后端列表() -> List[str]:
    名 = []
    if not libarchive is None:
        名.append("libarchive")
    名.append("zipfile")
    名.append("tarfile")
    return 名


def 归档有libarchive() -> bool:
    return not libarchive is None


def _归档解码(名) -> str:
    if isinstance(名, bytes):
        for 编码 in ("utf-8", "gbk", "cp437"):
            try:
                return 名.decode(编码)
            except Exception:
                continue
        return 名.decode("utf-8", "replace")
    return str(名)


def _归档安全检查(名: str) -> Optional[str]:
    """返回 None 表示安全; 否则返回拒绝原因。"""
    文本 = _归档解码(名).replace("\\", "/")
    if not 文本 or 文本.startswith("/"):
        return "绝对路径"
    if len(文本) > 1 and 文本[1] == ":":
        return "盘符路径"
    if any(段 == ".." for 段 in 文本.split("/")):
        return "路径穿越"
    return None


def 归档列出(路径, 后端: str = None) -> List[str]:
    """列出归档条目路径; 目录以 '/' 结尾(与 zipfile.namelist 语义一致)。失败抛异常。"""
    路径 = str(路径)
    for 后端名 in ([后端] if 后端 else 归档后端列表()):
        try:
            if 后端名 == "libarchive" and not libarchive is None:
                return _归档libarchive列出(路径)
            if 后端名 == "zipfile" and zipfile.is_zipfile(路径):
                return _归档zip列出(路径)
            if 后端名 == "tarfile" and tarfile.is_tarfile(路径):
                return _归档tar列出(路径)
        except Exception:
            continue
    raise ValueError(f"无法识别的归档格式: {路径}")


def _归档libarchive列出(路径: str) -> List[str]:
    with libarchive.file_reader(路径) as 归档:
        return [_归档解码(条目.pathname) + ("/" if 条目.isdir else "") for 条目 in 归档]


def _归档zip列出(路径: str) -> List[str]:
    with zipfile.ZipFile(路径) as 包:
        return [信息.filename for 信息 in 包.infolist()]


def _归档tar列出(路径: str) -> List[str]:
    with tarfile.open(路径) as 包:
        return [成员.name + ("/" if 成员.isdir() else "") for 成员 in 包.getmembers()]


def 归档限制(条目上限=None, 字节上限=None):
    """归档上限: 显式传参 > 当前配置(ARCHIVE_MAX_ENTRIES/ARCHIVE_MAX_BYTES) > 不限制。"""
    配置 = getattr(Mods(), "当前配置", None)
    条目 = 条目上限 if 条目上限 is not None else getattr(配置, "ARCHIVE_MAX_ENTRIES", None)
    字节 = 字节上限 if 字节上限 is not None else getattr(配置, "ARCHIVE_MAX_BYTES", None)
    return 条目, 字节


def 归档解压(路径, 目标目录, 过滤: Callable[[str], bool] = None,
        条目上限=None, 字节上限=None) -> Tuple[int, List[str]]:
    """解压到目标目录, 返回 (写出的文件数, 被跳过的条目说明)。上限默认取当前配置。"""
    路径 = str(路径)
    条目上限, 字节上限 = 归档限制(条目上限, 字节上限)
    目标 = Path(目标目录)
    目标.mkdir(parents=True, exist_ok=True)
    for 后端名 in 归档后端列表():
        try:
            if 后端名 == "libarchive" and not libarchive is None:
                return _归档libarchive解压(路径, 目标, 过滤, 条目上限, 字节上限)
            if 后端名 == "zipfile" and zipfile.is_zipfile(路径):
                return _归档zip解压(路径, 目标, 过滤, 条目上限, 字节上限)
            if 后端名 == "tarfile" and tarfile.is_tarfile(路径):
                return _归档tar解压(路径, 目标, 过滤, 条目上限, 字节上限)
        except Exception:
            continue
    raise ValueError(f"无法解压的归档格式: {路径}")


def _归档落盘(目标: Path, 名: str, 块迭代) -> int:
    完整 = 目标 / 名.replace("\\", "/")
    完整.parent.mkdir(parents=True, exist_ok=True)
    写入 = 0
    with open(完整, "wb") as f:
        for 块 in 块迭代:
            if not 块:
                continue
            f.write(块)
            写入 += len(块)
    return 写入


def _归档libarchive解压(路径, 目标, 过滤, 条目上限, 字节上限):
    写出, 跳过, 总计, 条目数 = 0, [], 0, 0
    with libarchive.file_reader(路径) as 归档:
        for 条目 in 归档:
            条目数 += 1
            if 条目上限 is not None and 条目数 > 条目上限:
                跳过.append(f"条目数超过上限 {条目上限}")
                break
            名 = _归档解码(条目.pathname)
            原因 = _归档安全检查(名)
            if 原因:
                跳过.append(f"{名}({原因})")
                continue
            if 条目.isdir:
                (目标 / 名.replace("\\", "/")).mkdir(parents=True, exist_ok=True)
                continue
            if not 条目.isreg:
                跳过.append(f"{名}(非普通文件)")
                continue
            if 过滤 and not 过滤(名):
                continue
            if 字节上限 is not None and 总计 > 字节上限:
                跳过.append(f"解压体积超过上限 {字节上限}")
                break
            总计 += _归档落盘(目标, 名, 条目.get_blocks())
            写出 += 1
    return 写出, 跳过


def _归档zip解压(路径, 目标, 过滤, 条目上限, 字节上限):
    写出, 跳过, 总计, 条目数 = 0, [], 0, 0
    with zipfile.ZipFile(路径) as 包:
        for 信息 in 包.infolist():
            条目数 += 1
            if 条目上限 is not None and 条目数 > 条目上限:
                跳过.append(f"条目数超过上限 {条目上限}")
                break
            名 = 信息.filename
            原因 = _归档安全检查(名)
            if 原因:
                跳过.append(f"{名}({原因})")
                continue
            if 信息.is_dir():
                (目标 / 名.replace("\\", "/")).mkdir(parents=True, exist_ok=True)
                continue
            if 过滤 and not 过滤(名):
                continue
            if 字节上限 is not None and 总计 > 字节上限:
                跳过.append(f"解压体积超过上限 {字节上限}")
                break
            with 包.open(信息) as 源:
                总计 += _归档落盘(目标, 名, iter(lambda: 源.read(1 << 20), b""))
            写出 += 1
    return 写出, 跳过


def _归档tar解压(路径, 目标, 过滤, 条目上限, 字节上限):
    写出, 跳过, 总计, 条目数 = 0, [], 0, 0
    with tarfile.open(路径) as 包:
        for 成员 in 包.getmembers():
            条目数 += 1
            if 条目上限 is not None and 条目数 > 条目上限:
                跳过.append(f"条目数超过上限 {条目上限}")
                break
            名 = 成员.name
            原因 = _归档安全检查(名)
            if 原因:
                跳过.append(f"{名}({原因})")
                continue
            if 成员.isdir():
                (目标 / 名.replace("\\", "/")).mkdir(parents=True, exist_ok=True)
                continue
            if not 成员.isfile():
                跳过.append(f"{名}(非普通文件)")
                continue
            if 过滤 and not 过滤(名):
                continue
            if 字节上限 is not None and 总计 > 字节上限:
                跳过.append(f"解压体积超过上限 {字节上限}")
                break
            源 = 包.extractfile(成员)
            if 源 is None:
                跳过.append(f"{名}(读取失败)")
                continue
            with 源:
                总计 += _归档落盘(目标, 名, iter(lambda: 源.read(1 << 20), b""))
            写出 += 1
    return 写出, 跳过


def 归档可打开(路径) -> bool:
    try:
        归档列出(路径)
        return True
    except Exception:
        return False


def 归档后端名(路径) -> str:
    """返回实际能打开它的后端名; 打不开返回空串。"""
    路径 = str(路径)
    if not libarchive is None:
        try:
            _归档libarchive列出(路径)
            return "libarchive"
        except Exception:
            pass
    if zipfile.is_zipfile(路径):
        return "zipfile"
    if tarfile.is_tarfile(路径):
        return "tarfile"
    return ""


class 归档视图:
    """压缩包用路径清单建索引; 目录用懒查询。查询一律精确锚定。"""

    def __init__(self, 路径=None, namelist=None, 最大深度: int = 3):
        self.路径 = Path(路径) if 路径 else None
        self.最大深度 = 最大深度
        self.namelist: List[str] = []
        self.后端 = ""
        self.是单文件 = bool(self.路径 and self.路径.is_file() and namelist is None and not 归档可打开(self.路径))
        self.是压缩包 = namelist is not None or (self.路径 is not None and self.路径.is_file() and not self.是单文件)
        self._目录集 = set()
        if self.是压缩包:
            if namelist is None:
                self.后端 = 归档后端名(self.路径)
                namelist = 归档列出(self.路径)
            self.namelist = [名.replace("\\", "/") for 名 in namelist]
            for 名 in self.namelist:
                if 名.endswith("/"):
                    self._目录集.add(名.rstrip("/"))
                    continue
                父 = 名.rsplit("/", 1)[0] if "/" in 名 else ""
                while 父:
                    self._目录集.add(父)
                    父 = 父.rsplit("/", 1)[0] if "/" in 父 else ""

    # ---------- 基本查询 ----------
    def 有文件(self, 相对路径: str) -> bool:
        if self.是单文件:
            return bool(self.路径) and self.路径.name == 相对路径
        if self.是压缩包:
            return 相对路径 in {名 for 名 in self.namelist if not 名.endswith("/")}
        return bool(self.路径) and (self.路径 / 相对路径).is_file()

    def 有目录(self, 相对路径: str) -> bool:
        if self.是压缩包:
            return 相对路径.rstrip("/") in self._目录集
        return bool(self.路径) and (self.路径 / 相对路径).is_dir()

    def 有浅层前缀(self, 名称: str, 最大深度: int = 2) -> bool:
        """名称出现在 根级 或 单层包装目录内 (容忍"内容直压"和"套一层文件夹")。"""
        return bool(self.查找(名称, 最大深度=最大深度, 任意深度=False))

    def 有浅层文件(self, 名称: str, 最大深度: int = 2) -> bool:
        """标记文件出现在 根级 或 包装目录内 —— 启动器标记常被套一层文件夹。

        名称可以是多段(如 "texts/languages.json"), 按尾部段匹配。
        """
        目标段 = [段 for 段 in 名称.replace("\\", "/").strip("/").split("/") if 段]
        if not 目标段:
            return False
        if self.是单文件:
            return bool(self.路径) and self.路径.name == 目标段[-1]
        if self.是压缩包:
            for 名 in self.namelist:
                if 名.endswith("/"):
                    continue
                段 = [段 for 段 in 名.split("/") if 段]
                if len(段) <= 最大深度 and 段[-len(目标段):] == 目标段:
                    return True
            return False
        if not self.路径 or not self.路径.is_dir():
            return False
        候选 = [self.路径 / 名称]
        for 层数 in range(1, max(1, 最大深度)):
            候选.extend(self.路径.glob(f"{'*/' * 层数}{名称}"))
        return any(p.is_file() for p in 候选)

    def 匹配根(self, 名称: str) -> List[str]:
        """任意一层根里存在该目录(光影包常被套一层文件夹)。"""
        return self.查找(名称, 最大深度=self.最大深度, 任意深度=True)

    def 自身匹配(self, 模式: str) -> bool:
        """目录视图的"根本身"就是这个内容目录(用户直接上传 ftbquests/ 文件夹)。"""
        if self.是压缩包 or self.路径 is None or not self.路径.is_dir():
            return False
        末段 = [段 for 段 in 模式.replace("\\", "/").strip("/").split("/") if 段]
        return bool(末段) and self.路径.name.lower() == 末段[-1].lower()

    def 全部文件(self) -> List[str]:
        if self.是单文件:
            return [self.路径.name] if self.路径 else []
        if self.是压缩包:
            return [名 for 名 in self.namelist if not 名.endswith("/")]
        if not self.路径:
            return []
        根 = self.路径
        return [str(p.relative_to(根)).replace("\\", "/") for p in 根.rglob("*") if p.is_file()]

    # ---------- 路径模式匹配 ----------
    def 查找(self, 模式: str, 最大深度: int = None, 任意深度: bool = True) -> List[str]:
        """返回所有"以该模式结尾"的目录路径(由浅到深); 模式段可用 * 通配单段。

        模式 "." 表示"整包根"(返回 [""]), 用于"对整包执行"的类型(数据包/资源包)。
        """
        最大深度 = self.最大深度 if 最大深度 is None else 最大深度
        模式 = 模式.replace("\\", "/").strip("/")
        if 模式 in ("", "."):
            return [""]
        段 = [段 for 段 in 模式.split("/") if 段]
        if not 段:
            return []
        命中 = []
        for 目录 in self._候选目录(段, 最大深度, 任意深度):
            实际 = 目录.split("/")
            if len(实际) < len(段):
                continue
            if not 任意深度 and len(实际) > len(段) + 1:
                continue
            if all(模式段 == "*" or 模式段 == 实段 for 模式段, 实段 in zip(段, 实际[-len(段):])):
                命中.append(目录)
        return sorted(命中, key=lambda p: (p.count("/"), p))

    def _候选目录(self, 段: List[str], 最大深度: int, 任意深度: bool):
        if self.是压缩包:
            return [目录 for 目录 in self._目录集 if 目录]
        根 = self.路径
        if 根 is None:
            return []
        候选 = []
        尾 = 段[-1]
        if "*" in 尾:
            候选.extend(根.rglob(尾))
            if 任意深度:
                候选.extend(根.rglob(f"*/{尾}"))
        else:
            候选.append(根 / 段[-1])
            if 任意深度:
                候选.extend(p for p in 根.glob(f"*/{段[-1]}") if p.is_dir())
                候选.extend(p for p in 根.glob(f"*/*/{段[-1]}") if p.is_dir())
            else:
                候选.extend(p for p in 根.glob(f"*/{段[-1]}") if p.is_dir())
        return [str(p.relative_to(根)).replace("\\", "/") for p in 候选 if p.is_dir()]


def 探测容器(视图: 归档视图, 后缀: str = "") -> str:
    """判容器名; 判不出来返回 "未知"。规则全部来自模组注册(Mods().容器探测表)。"""
    候选 = []
    for 规则 in Mods().容器探测表:
        try:
            命中 = bool(规则.判定(视图, 后缀))
        except Exception:
            命中 = False
        if 命中:
            候选.append((规则.容器, 规则.分数))
    if not 候选:
        return "未知"
    候选.sort(key=lambda 项: -项[1])
    return 候选[0][0]


def 路由(视图: 归档视图, 后缀: str = "") -> Tuple[str, List[Any]]:
    """视图 → (容器, 计划列表); 计划和注册表都在 TranslatorMods 管。"""
    容器 = 探测容器(视图, 后缀)
    if 容器 == "未知":
        return 容器, Mods().保守计划()
    计划列表 = Mods().生成计划(视图, 容器)
    if not 计划列表:
        计划列表 = Mods().保守计划()
    return 容器, 计划列表
