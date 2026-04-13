from TranslatorLib import re, os, time, json, uuid, pickle, zipfile, Path, ConfigFactory, HOCONConverter, eb, PurePosixPath, glob, tomllib
from TranslatorConfig import RuntimeConfig, DEFAULT_CONFIG
from TranslatorLocale import Locale

class Module:
    def __init__(Self, Config: dict = None):
        global tqdm
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.Locale = Locale(Config=Config)
        Self.Lang = Self.Locale.Lang
        tqdm = Self.Locale.Tqdm
        Self.LOGS_FILE_PATH = Self.Config.LOGS_FILE_PATH
        Self.LOGS_FILE_NAME = Self.Config.LOGS_FILE_NAME
    def 写入日志(Self, text: str, info_level: int = 0, **kwargs):
        text = Self.Lang(text, **kwargs)
        日志文件名 = Path(Self.LOGS_FILE_PATH) / Self.LOGS_FILE_NAME
        时间 = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
        日志等级 = "[INFO]"
        if info_level == 1:
            日志等级 = "[WARNING]"
        elif info_level == 2:
            日志等级 = "[ERROR]"
        elif info_level == 3:
            日志等级 = "[FATAL]"
        elif info_level == 4 and Self.Config.DEBUG_MODE:
            日志等级 = "[DEBUG]"
        elif info_level == 4 and (not Self.Config.DEBUG_MODE):
            return
        写入内容 = f"{时间}{日志等级}{text}\n"
        if (Path(日志文件名).resolve() != Path(f"{DEFAULT_CONFIG.LOGS_FILE_PATH}/{DEFAULT_CONFIG.LOGS_FILE_NAME}.log").resolve()) and Self.Config.LOGS_GLOBAL:
            Self.写入全局日志(写入内容)
        with open(f"{日志文件名}.log", "a+", encoding="utf-8") as f:
            f.write(写入内容)
    def 写入全局日志(Self, text: str):
        with open(f"{DEFAULT_CONFIG.LOGS_FILE_PATH}/{DEFAULT_CONFIG.LOGS_FILE_NAME}.log", "a+", encoding="utf-8") as f:
            f.write(text)
        
    def 读取日志(Self):
        日志文件名 = Path(Self.LOGS_FILE_PATH) / Self.LOGS_FILE_NAME
        with open(f"{日志文件名}.log", "r", encoding="utf-8") as f:
            return f.read()
    def 读取单个FTBQ_Snbt文件(Self, index: str):
        文本列表 = []
        SNBT文件 = ConfigFactory.parse_file(index)
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
                            返回内容 = index2
                            if isinstance(index2, str):
                                双重字符串 = index2.strip()
                                if (双重字符串.startswith('[') and 双重字符串.endswith(']')) or (双重字符串.startswith('{') and 双重字符串.endswith('}')):
                                    try:
                                        parsed = json.loads(双重字符串)
                                        if isinstance(parsed, list) and len(parsed) > 1:
                                            返回内容 = parsed[1]
                                            if isinstance(返回内容, dict) and "translate" in 返回内容:
                                                返回内容 = 返回内容["translate"]
                                        else:
                                            返回内容 = index2 
                                    except json.JSONDecodeError as e:
                                        返回内容 = index2
                                    except (TypeError, IndexError, KeyError) as e:
                                        返回内容 = index2
                            文本列表.append([[index, ["quests", index1q], ["description", index2q]], 返回内容])
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
        SNBT文件 = ConfigFactory.parse_file(index[0][0][0])
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
                    else:
                        列表 = SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]]
                        列表[位置[1][1]] = index1[1]
                        SNBT文件[位置[0][0]][位置[0][1]][位置[1][0]] = 列表
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
 
        with open(index[0][0][0], "w", encoding="utf-8") as f:
            json文件 = HOCONConverter.to_json(SNBT文件, indent=4).splitlines()
            if mode ==  "H":
                去逗号json文件 = []
                for index in json文件:
                    if index.endswith(","):
                        index = index[:-1]
                    去逗号json文件.append(index)
                json文件 = 去逗号json文件
            for index1p, index1 in enumerate(json文件):
                index1 = re.sub(r'":\s*"([+-]?\d+\.?\d*[bdL])"',r'": \1',index1)
                #index1 = re.sub(r'"([^"]+)":', r'\1:', index1)
                index1 = re.sub(r'"([a-zA-Z_][a-zA-Z0-9_]*)":', r'\1:', index1)
                json文件[index1p] = index1
            f.write("\n".join(json文件))
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
                            文件列表.append([[index, ["questLines:9", index1, "properties:10", "betterquesting:10"], ["name:8"]], NBT文件["questDatabase:9"][index1]["properties:10"]["betterquesting:10"]["name:8"]])
                        if "desc:8" in NBT文件["questLines:9"][index1]["properties:10"]["betterquesting:10"]:
                            文件列表.append([[index, ["questLines:9", index1, "properties:10", "betterquesting:10"], ["desc:8"]], NBT文件["questDatabase:9"][index1]["properties:10"]["betterquesting:10"]["desc:8"]])
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
            return [line.split('=', 1)   for line in 源文件   if (stripped := line.strip()) and not stripped.startswith('#') and not stripped.startswith('//')], 源文件
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
                保存列表 = [line for line in 保存列表 if line.strip() and '=' in line and not line.lstrip().startswith(('//', '#'))]
                json文件 = {line.split('=', 1)[0]: line.split('=', 1)[1] for line in 保存列表}
                f.write(json.dumps(json文件, ensure_ascii=False, indent=4))
    def 读取压缩文件(Self, file_path: str, cache_path: str, original_language: str, target_language: str):
        try:
            for _ in tqdm(range(1), desc="tqdm.file.read"): 
                with zipfile.ZipFile(file_path, 'r') as f:
                    可用文件列表 = [False]
                    文件列表 = f.namelist()
                    文件路径 = f"{cache_path}/{uuid.uuid4().hex}_{Path(file_path).stem}"
                    if any(name.startswith("shaders") for name in 文件列表):
                        可用文件列表 = [True]
                        f.extractall(文件路径)
                    for index in [original_language, target_language]:
                        for index1 in 文件列表:
                            index1文件名 = os.path.splitext(index1.split('/')[-1])[0]
                            if index1文件名.lower() == index.lower():
                                f.extract(index1, path=文件路径)
                                可用文件列表.append([index, f"{文件路径}/{index1}"])
        except Exception:
            Self.写入日志("log.module.zip.read.error", e=eb.format_exc(), info_level=3)
            raise FileNotFoundError(eb.format_exc())
        return 可用文件列表
    def 读取资源文件(Self, file0: str, file1: str = "", read_error: bool = True):
        Self.写入日志("log.core.file.read.start", file0=file0, file1=file1)
        压缩路径 = ""
        文件1 = ""
        file2 = ""
        if Path(file0).suffix in [".zip", ".jar"]:
            缓存路径 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}"
            file2 = Self.读取压缩文件(file0, 缓存路径, Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT)
            file3 = Self.读取压缩文件(file1, 缓存路径, Self.Config.LANGUAGE_INPUT, Self.Config.LANGUAGE_OUTPUT) if file1 else []
            if not any(isinstance(item, list) and len(item) > 0 and item[0] == Self.Config.LANGUAGE_INPUT.lower() for item in file2[1:]):
                if read_error:
                    Self.写入日志("log.module.read.file0.not.lang.error", info_level=3)
                raise FileNotFoundError(Self.Lang("log.module.read.file0.not.lang.error"))
            if file1 and not any(isinstance(item, list) and len(item) > 0 and item[0] == Self.Config.LANGUAGE_OUTPUT.lower() for item in file3[1:]):
                if read_error:
                    Self.写入日志("log.module.read.file1.not.lang.error", info_level=3)
                raise FileNotFoundError(Self.Lang("log.module.read.file1.not.lang.error"))
            for index in file2[1:]:
                if index[0] == Self.Config.LANGUAGE_INPUT.lower():
                    文件0 = index[1]
                    压缩路径 = index[1]
            if file3:
                for index in file3[1:]:
                    if index[0] == Self.Config.LANGUAGE_OUTPUT.lower():
                        文件1 = index[1]
            else:
                for index in file2[1:]:
                    if index[0] == Self.Config.LANGUAGE_OUTPUT.lower():
                        文件1 = index[1]
            输出扩展名 = Path(文件0).suffix
            文件0, 文件0源文件 = Self.读取语言文件(文件0)
            if 文件1:
                文件1, _ = Self.读取语言文件(文件1)
        elif Path(file0).suffix in [".lang", ".json"]:
            文件0, 文件0源文件 = Self.读取语言文件(file0)
            if file1:
                文件1, _ = Self.读取语言文件(file1)
            输出扩展名 = Path(file0).suffix
        Self.写入日志("log.core.file.read.end", file0=file0, file1=file1)
        return 文件0, 文件0源文件, 文件1, 压缩路径, 输出扩展名, file2
    def 翻译缓存(Self, 输入列表: list = []):
        文本文件 = []
        try:
            with open(f"{Self.Config.TRANSLATOR_CACHE_PATH}/{Self.Config.TRANSLATOR_CACHE_NAME}.pkl", "rb+") as f:
                文本文件 = pickle.load(f)
        except Exception: pass
        if 输入列表:
            文本文件.extend(输入列表)
            with open(f"{Self.Config.TRANSLATOR_CACHE_PATH}/{Self.Config.TRANSLATOR_CACHE_NAME}.pkl", "wb+") as f:
                pickle.dump(文本文件, f)
        return {item[0]: item[1] for item in 文本文件}, {item[0] for item in 文本文件}
    def 从资源包文件夹获取I18n翻译模组ID(Self, 路径: str):
        Self.写入日志("log.core.modid.get.start", info_level=0)
        try:
            with zipfile.ZipFile(glob.glob(os.path.join(f"{路径}/resourcepacks/", "Minecraft-Mod-Language-Modpack*"))[0], 'r') as f:
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
        for 模组文件路径 in 所有模组路径:
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