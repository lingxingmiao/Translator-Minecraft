from TranslatorLib import queue, json, uuid, pickle, zipfile, Path, eb, PurePosixPath, tomllib, snbtlib, ast, logging, RotatingFileHandler, QueueHandler, QueueListener
from TranslatorConfig import RuntimeConfig, DEFAULT_CONFIG
from TranslatorLocale import Locale

class FlushingFileHandler(RotatingFileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

class Module:
    def __init__(Self, Config: dict = None):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.Locale = Locale(Config=Config)
        Self.Lang = Self.Locale.Lang
        Self.tqdm = Self.Locale.Tqdm
        Self.启动日志()
    def 启动日志(Self):
        日志名称 = f"Translator_{id(Self)}"
        Self.日志管理器 = logging.getLogger(日志名称)
        Self.日志管理器.setLevel(logging.DEBUG if Self.Config.DEBUG_MODE else logging.INFO)
        if Self.日志管理器.handlers:
            Self.日志管理器.handlers.clear()
        日志队列 = queue.Queue(-1)
        队列处理器 = QueueHandler(日志队列)
        Self.日志管理器.addHandler(队列处理器)
        日志文件 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log").resolve()
        日志文件.parent.mkdir(parents=True, exist_ok=True)
        格式化器 = logging.Formatter(fmt='[%(asctime)s][%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        文件处理器 = FlushingFileHandler(str(日志文件), encoding='utf-8', maxBytes=10*1024*1024, backupCount=5)
        文件处理器.setFormatter(格式化器)
        处理器列表 = [文件处理器]
        默认日志 = Path(f"{DEFAULT_CONFIG.LOGS_FILE_PATH}/{DEFAULT_CONFIG.LOGS_FILE_NAME}.log").resolve()
        if Self.Config.LOGS_GLOBAL and 日志文件 != 默认日志:
            默认日志.parent.mkdir(parents=True, exist_ok=True)
            全局处理器 = FlushingFileHandler(str(默认日志), encoding='utf-8', maxBytes=10*1024*1024, backupCount=5)
            全局处理器.setFormatter(格式化器)
            处理器列表.append(全局处理器)
        Self._队列监听器 = QueueListener(日志队列, *处理器列表, respect_handler_level=True)
        Self._队列监听器.start()
        Self.日志管理器.propagate = False
    def 写入日志(Self, text: str, info_level: int = 0, **kwargs):
        等级映射 = {0: logging.INFO, 1: logging.WARNING, 2: logging.ERROR, 3: logging.CRITICAL, 4: logging.DEBUG}
        等级 = 等级映射.get(info_level, logging.INFO)
        if info_level == 4 and not Self.Config.DEBUG_MODE:
            return
        本地化文本 = Self.Lang(text, **kwargs)
        Self.日志管理器.log(等级, 本地化文本)

    def 读取日志(Self):
        日志文件名 = Path(Self.Config.LOGS_FILE_PATH) / Self.Config.LOGS_FILE_NAME
        log_path = f"{日志文件名}.log"
        if not Path(log_path).exists():
            return ""
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(log_path, "r", encoding="latin1") as f:
                content = f.read()
            Self.写入日志("log.module.logs.encoding.warning", info_level=1)
            return content
    def 读取单个FTBQ_Snbt文件(Self, index: str):
        文本列表 = []
        SNBT文件 = snbtlib.loads(Path(index).read_text(encoding='utf-8'))
        try:
            if "description" in SNBT文件:
                if isinstance(SNBT文件["description"], str):
                    文本列表.append([[index, ["description"]], SNBT文件["description"]])
                elif isinstance(SNBT文件["description"], list):
                    for index2q, index2 in enumerate(SNBT文件["description"]):
                        文本列表.append([[index, ["description", index2q]], index2])
            if "rewards" in SNBT文件:
                for index2q, index2 in enumerate(SNBT文件["rewards"]):
                    if "title" in index2:
                        文本列表.append([[index, ["rewards", index2q], ["title"]], index2["title"]])
            if "subtitle" in SNBT文件:
                if isinstance(SNBT文件["subtitle"], str):
                    文本列表.append([[index, ["subtitle"]], SNBT文件["subtitle"]])
                if isinstance(SNBT文件["subtitle"], list):
                    for index2q, index2 in enumerate(SNBT文件["subtitle"]):
                        文本列表.append([[index, ["subtitle", index2q]], index2])
            if "title" in SNBT文件:
                if isinstance(SNBT文件["title"], str):
                    文本列表.append([[index, ["title"]], SNBT文件["title"]])
                if isinstance(SNBT文件["title"], list):
                    for index2q, index2 in enumerate(SNBT文件["title"]):
                        文本列表.append([[index, ["title", index2q]], index2])
            if "tasks" in SNBT文件:
                for index1q, index1 in enumerate(SNBT文件["tasks"]):
                    if "items" in index1:
                        for index2q, index2 in enumerate(index1["items"]):
                            if "tag" in index2:
                                if "pages" in index2["tag"]:
                                    for index3q, index3 in enumerate(index2["tag"]["pages"]):
                                        文本列表.append([[index, ["tasks", index1q], ["items", index2q], ["tag"], ["pages", index3q]], index3])
                                if "title" in index2["tag"]:
                                    文本列表.append([[index, ["tasks", index1q], ["items", index2q], ["tag"], ["title"]], index3])
                                if "display" in index2["tag"]:
                                    if "Lore" in index2["tag"]["display"]:
                                        for index3q, index3 in enumerate(index2["tag"]["display"]["Lore"]):
                                            文本列表.append([[index, ["tasks", index1q], ["items", index2q], ["tag", "display"], ["Lore", index3q]], index3])
                                    if "Name" in index2["tag"]["display"]:
                                        文本列表.append([[index, ["tasks", index1q], ["items", index2q], ["tag", "display"], ["Name"]], index2["tag"]["display"]["Name"]])
                    if "title" in index1:
                        文本列表.append([[index, ["tasks", index1q], ["title"]], index1["title"]])
            if "rewards" in SNBT文件:
                for index1q, index1 in enumerate(SNBT文件["rewards"]):
                    if "item" in index1:
                        if isinstance(index1["item"], dict) and "tag" in index1["item"]:
                            if "pages" in index1["item"]["tag"]:
                                for index3q, index3 in enumerate(index1["item"]["tag"]["pages"]):
                                    文本列表.append([[index, ["rewards", index1q], ["item", "tag"], ["pages", index3q]], index3])
                            if "title" in index1["item"]["tag"]:
                                文本列表.append([[index, ["rewards", index1q], ["item", "tag"], ["title"]], index3])
                            if "display" in index1["item"]["tag"]:
                                if "Lore" in index1["item"]["tag"]["display"]:
                                    for index3q, index3 in enumerate(index1["item"]["tag"]["display"]["Lore"]):
                                        文本列表.append([[index, ["rewards", index1q], ["item", "tag", "display"], ["Lore", index3q]], index3])
                                if "Name" in index1["item"]["tag"]["display"]:
                                    文本列表.append([[index, ["rewards", index1q], ["item", "tag", "display"], ["Name"]], index1["item"]["tag"]["display"]["Name"]])
                    if "title" in index1:
                        文本列表.append([[index, ["rewards", index1q], ["title"]], index1["title"]])
            if "text" in SNBT文件:
                for index1q, index1 in enumerate(SNBT文件["text"]):
                    文本列表.append([[index, ["text", index1q]], index1])
            if "quests" in SNBT文件:
                for index1q, index1 in enumerate(SNBT文件["quests"]):
                    if "description" in index1:
                        for index2q, index2 in enumerate(index1["description"]):
                            try:
                                返回内容 = json.loads(index2.replace(r'\"', '"'))
                                if "text" in 返回内容:
                                    文本列表.append([[index, ["quests", index1q], ["description", index2q, 返回内容, "text"]], 返回内容["text"]])
                                elif "translate" in 返回内容[1]:
                                    文本列表.append([[index, ["quests", index1q], ["description", index2q, 返回内容, 1, "translate"]], 返回内容[1]["translate"]])
                            except Exception as e:
                                文本列表.append([[index, ["quests", index1q], ["description", index2q]], index2])
                            
                    if "rewards" in index1:
                        for index2q, index2 in enumerate(index1["rewards"]):
                            if "title" in index2:
                                文本列表.append([[index, ["quests", index1q], ["rewards", index2q], ["title"]], index2["title"]])
                    if "subtitle" in index1:
                        文本列表.append([[index, ["quests", index1q], ["subtitle"]], index1["subtitle"]])
                    if "title" in index1:
                        文本列表.append([[index, ["quests", index1q], ["title"]], index1["title"]])
                    if "tasks" in index1:
                        for index2q, index2 in enumerate(index1["tasks"]):
                            if "title" in index2:
                                文本列表.append([[index, ["quests", index1q], ["tasks", index2q], ["title"]], index2["title"]])
            if "chapter_groups" in SNBT文件:
                for index1q, index1 in enumerate(SNBT文件["chapter_groups"]):
                    if "title" in index1:
                        文本列表.append([[index, ["chapter_groups", index1q]], index1["title"]])
        except Exception:
            Self.写入日志("log.module.quests.load.error", mod="FTB Quests", file=index, e=eb.format_exc(), info_level=2)
        return 文本列表
    def 应用FTBQ翻译(Self, index: list, mode: str):
        位置 = "None"
        层数 = "None"
        SNBT文件 = snbtlib.loads(Path(index[0][0][0]).read_text(encoding='utf-8'))
        for index1p, index1 in enumerate(index):
            try:
                层数 = len(index[index1p][0][1:])
                位置 = index1[0][1:]
                if 层数 == 1:
                    if len(位置[0]) == 1:
                        SNBT文件[位置[0][0]] = index1[1]
                    else:
                        列表 = SNBT文件[位置[0][0]]
                        列表[位置[0][1]] = index1[1]
                        SNBT文件[位置[0][0]] = 列表
                elif 层数 == 2:
                    if len(位置[1]) == 1:
                        SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]] = index1[1]
                    elif len(位置[1]) == 2:
                        列表 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]]
                        列表[位置[1][1]] = index1[1]
                        SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]] = 列表
                    elif len(位置[1]) == 4:
                        字典 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]]
                        字典值 = 位置[1][2]
                        字典值[位置[1][3]] = index1[1]
                        字典[位置[1][1]] = f'"{json.dumps(字典值, ensure_ascii=False, separators=(', ', ': ')).replace('"', '\\"')}"'
                        SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]] = 字典
                    elif len(位置[1]) == 5:
                        字典 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]]
                        字典值 = 位置[1][2]
                        字典值[位置[1][3]][位置[1][4]] = index1[1]
                        字典[位置[1][1]] = f'"{json.dumps(字典值, ensure_ascii=False, separators=(', ', ': ')).replace('"', '\\"')}"'
                        SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]] = 字典
                elif 层数 == 3:
                    if not isinstance(位置[1][1], str):
                        if len(位置[2]) == 1:
                            SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]] = index1[1]
                        else:
                            列表 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]]
                            列表[位置[2][1]] = index1[1]
                            SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]] = 列表
                    else:
                        if len(位置[1]) == 2:
                            if len(位置[2]) == 1:
                                SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]] = index1[1]
                            else:
                                列表 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]]
                                列表[位置[2][1]] = index1[1]
                                SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]] = 列表
                        elif len(位置[1]) == 3:
                            if len(位置[2]) == 1:
                                SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[1][2]][位置[2][0]] = index1[1]
                            else:
                                列表 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[1][2]][位置[2][0]]
                                列表[位置[2][1]] = index1[1]
                                SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[1][2]][位置[2][0]] = 列表
                elif 层数 == 4:
                    if len(位置[2]) == 1:
                        if len(位置[3]) == 1:
                            SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]][位置[3][0]] = index1[1]
                        elif len(位置[3]) == 2:
                            列表 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]][位置[3][0]]
                            列表[位置[3][1]] = index1[1]
                            SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]][位置[3][0]] = 列表
                    elif len(位置[2]) == 2:
                        if len(位置[3]) == 1:
                            SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]][位置[2][1]][位置[3][0]] = index1[1]
                        elif len(位置[3]) == 2:
                            列表 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]][位置[2][1]][位置[3][0]]
                            列表[位置[3][1]] = index1[1]
                            SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]][位置[1][1]][位置[2][0]][位置[2][1]][位置[3][0]] = 列表
            except Exception:
                Self.写入日志("log.module.quests.write.error", index=index[0][0][0], item=index1, level=层数, position=位置, e=eb.format_exc(), info_level=2)
        Path(index[0][0][0]).write_text(snbtlib.dumps(SNBT文件, compact=False if mode=="H" else True), encoding='utf-8')
    def 读取单个BQ_Json文件(Self, index: str):
        文件列表 = []
        with open(index, "r", encoding="utf-8") as f:
            NBT文件 = json.load(f)
        if "properties:10" in NBT文件:
            if "betterquesting:10" in NBT文件["properties:10"]:
                if "name:8" in NBT文件["properties:10"]["betterquesting:10"]:
                    文件列表.append([[index, ["properties:10", "betterquesting:10"], ["name:8"]], NBT文件["properties:10"]["betterquesting:10"]["name:8"]])
                if "desc:8" in NBT文件["properties:10"]["betterquesting:10"]:
                    文件列表.append([[index, ["properties:10", "betterquesting:10"], ["desc:8"]], NBT文件["properties:10"]["betterquesting:10"]["desc:8"]])
        if "questDatabase:9" in NBT文件:
            for index1 in NBT文件["questDatabase:9"]:
                if "properties:10" in NBT文件["questDatabase:9"][index1]:
                    if "betterquesting:10" in NBT文件["questDatabase:9"][index1]["properties:10"]:
                        if "name:8" in NBT文件["questDatabase:9"][index1]["properties:10"]["betterquesting:10"]:
                            文件列表.append([[index, ["questDatabase:9", index1, "properties:10", "betterquesting:10"], ["name:8"]], NBT文件["questDatabase:9"][index1]["properties:10"]["betterquesting:10"]["name:8"]])
                        if "desc:8" in NBT文件["questDatabase:9"][index1]["properties:10"]["betterquesting:10"]:
                            文件列表.append([[index, ["questDatabase:9", index1, "properties:10", "betterquesting:10"], ["desc:8"]], NBT文件["questDatabase:9"][index1]["properties:10"]["betterquesting:10"]["desc:8"]])
        if "questLines:9" in NBT文件:
            for index1 in NBT文件["questLines:9"]:
                if "properties:10" in NBT文件["questLines:9"][index1]:
                    if "betterquesting:10" in NBT文件["questLines:9"][index1]["properties:10"]:
                        if "name:8" in NBT文件["questLines:9"][index1]["properties:10"]["betterquesting:10"]:
                            文件列表.append([[index, ["questLines:9", index1, "properties:10", "betterquesting:10"], ["name:8"]], NBT文件["questLines:9"][index1]["properties:10"]["betterquesting:10"]["name:8"]])
                        if "desc:8" in NBT文件["questLines:9"][index1]["properties:10"]["betterquesting:10"]:
                            文件列表.append([[index, ["questLines:9", index1, "properties:10", "betterquesting:10"], ["desc:8"]], NBT文件["questLines:9"][index1]["properties:10"]["betterquesting:10"]["desc:8"]])
        return 文件列表
    def 应用BQ翻译(Self, index: list):
        位置 = "None"
        with open(index[0][0][0], "r", encoding="utf-8") as f:
            NBT文件 = json.load(f)
        for index1 in index:
            try:
                位置 = index1[0][1:]
                if len(位置[0]) == 2:
                    NBT文件[位置[0][0]][位置[0][1]][位置[1][0]] = index1[1]
                if len(位置[0]) == 4:
                    NBT文件[位置[0][0]][位置[0][1]][位置[0][2]][位置[0][3]][位置[1][0]] = index1[1]
            except Exception:
                Self.写入日志("log.module.quests.write.error", index=index[0][0][0], item=index1, level="None", position=位置, e=eb.format_exc(), info_level=2)
        with open(index[0][0][0], "w+", encoding="utf-8") as f:
            json.dump(NBT文件, f, ensure_ascii=False, indent=2)
    def 读取语言文件(Self, file: str):
        with open(file, "r", encoding="utf-8") as f:
            if Path(file).suffix == ".lang":
                源文件 = f.read().splitlines()
            elif Path(file).suffix == ".json":
                Json文件 = json.load(f)
                源文件 = [f"{index}={Json文件[index]}" for index in Json文件]
        return [
            (lambda parts: [parts[0], parts[1], file])(line.split('=', 1)) 
            for line in 源文件 
            if (stripped := line.strip()) and stripped and '=' in stripped and not stripped.startswith(('#', '//'))
        ], [源文件, file]
    def 保存语言文件(Self, file: str, 保存列表: list):
        with open(file, "w+", encoding="utf-8") as f:
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
                        v = ast.literal_eval(v)
                    except Exception: pass
                    json文件[k] = v
                f.write(json.dumps(json文件, ensure_ascii=False, indent=4))
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
            Self.写入日志("log.module.zip.read.error", e=eb.format_exc(), info_level=3)
            raise FileNotFoundError(eb.format_exc())
    def 读取资源文件(Self, file0: str, file1: str = "", read_error: bool = True):
        Self.写入日志("log.core.file.read.start", file0=file0, file1=file1)
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
                Self.写入日志(日志键, info_level=3)
                raise FileNotFoundError(Self.Lang(日志键))
            return 匹配路径
        是压缩包0 = Path(file0).suffix.lower() in {".zip", ".jar"}
        是压缩包1 = file1 and Path(file1).suffix.lower() in {".zip", ".jar"}
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
        elif Path(file0).suffix.lower() in {".lang", ".json"}:
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
        Self.写入日志("log.core.file.read.end", file0=文件0, file1=文件1)
        return 文件0, 文件0源文件, 文件1, 压缩路径, 输出扩展名, file2
    def 翻译缓存(Self, 输入列表: list = None):
        文本文件 = []
        try:
            with open(f"{Self.Config.TRANSLATOR_CACHE_PATH}/{Self.Config.TRANSLATOR_CACHE_NAME}.pkl", "rb+") as f:
                文本文件 = pickle.load(f)
        except Exception: pass
        if 输入列表:
            文本文件.extend(输入列表)
            文本文件 = list({item[0]: item for item in 文本文件}.values())
            with open(f"{Self.Config.TRANSLATOR_CACHE_PATH}/{Self.Config.TRANSLATOR_CACHE_NAME}.pkl", "wb+") as f:
                pickle.dump(文本文件, f)
        return {item[0]: item[1] for item in 文本文件}
    def 从资源包文件夹获取I18n翻译模组ID(Self, 路径: str):
        Self.写入日志("log.core.modid.get.start", info_level=0)
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
            Self.写入日志("log.module.pack.i18n.no", info_level=3)
            返回内容 = []
        Self.写入日志("log.core.modid.get.end", info_level=0)
        return 返回内容
    def 从模组文件夹获取模组ID(Self, 路径: str):
        Self.写入日志("log.core.modid.get.start", info_level=0)
        所有模组路径 = Path(f"{路径}/{"mods"}").glob('*.jar')
        模组ID列表 = []
        for 模组文件路径 in Self.tqdm(所有模组路径, desc="tqdm.file.modid.get"):
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
                            Self.写入日志("log.module.parsemodid.error", e=eb.format_exc())
                    if "META-INF/mods.toml" in 文件列表 and 模组ID is None:
                        try:
                            with 压缩文件.open("META-INF/mods.toml") as 文件:
                                数据 = tomllib.load(文件)
                            模组列表 = 数据.get('mods', [])
                            if 模组列表:
                                模组ID = 模组列表[0].get('modId')
                        except Exception:
                            Self.写入日志("log.module.parsemodid.error", e=eb.format_exc())
                    if "META-INF/neoforge.mods.toml" in 文件列表 and 模组ID is None:
                        try:
                            with 压缩文件.open("META-INF/neoforge.mods.toml") as 文件:
                                数据 = tomllib.load(文件)
                            模组列表 = 数据.get('mods', [])
                            if 模组列表:
                                模组ID = 模组列表[0].get('modId')
                        except Exception:
                            Self.写入日志("log.module.parsemodid.error", e=eb.format_exc())
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
                Self.写入日志("log.module.parsemodid.file.error", e=eb.format_exc())
        Self.写入日志("log.core.modid.get.end", info_level=0)
        return 模组ID列表
    def 输出路径处理(Self, path: str):
        if not path:
            path = f"./{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}"
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path).resolve()
    def 列表去重(Self, 列表: list):
        return list(dict.fromkeys(列表))
    def 读取审查文件(Self, file0: str):
        返回列表 = []
        with open(file0, 'r', encoding='utf-8') as 文件对象:
            文件行列表 = 文件对象.read().splitlines()
        for 单行文本 in 文件行列表:
            数据字典 = ast.literal_eval(单行文本)
            for 键名, 键值 in 数据字典.items():
                返回列表.append([键名, 键值[0], 键值[1]])
        return 返回列表