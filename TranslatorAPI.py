from TranslatorLib import (APIConfig, Dict, uvicorn, Path, time, json, asyncio, shutil, threading, eb, Dict, Any, quote, hashlib, re, PDict, secrets, uuid, List, zstd, asynccontextmanager, mimetypes,
                            Config, TranslatorFile,
                            FastAPI, UploadFile, HTTPException, status, Depends, Security, Form, _rate_limit_exceeded_handler, get_remote_address,
                            HTTPBearer, HTTPAuthorizationCredentials, CORSMiddleware, Limiter, RateLimitExceeded, GZipMiddleware, Request, BackgroundTasks, FileResponse, PlainTextResponse,)

管理器实例 = Config(APIConfig)

@asynccontextmanager
async def API生命周期(_: FastAPI):
    任务状态字典.启动()
    try: yield
    finally: await 任务状态字典.停止()

API实例 = FastAPI(title="TranslationMinecraft V1.7 Bata.1", lifespan=API生命周期)
API实例.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def API配置值(键: str, 默认值: Any = None) -> Any:
    return getattr(管理器实例.Config, 键, 默认值)

API实例.add_middleware(GZipMiddleware, minimum_size=512, compresslevel=API配置值("API_UPLOAD_COMPLVL"))
同时最大线程数 = Limiter(key_func=lambda request: request.headers.get("Authorization", get_remote_address(request)))
API实例.state.limiter = 同时最大线程数
API实例.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
任务并行锁 = asyncio.Semaphore(max(1, int(API配置值("API_WORKERS", 5) or 1)))
限流配置值 = f"{API配置值("API_RPM")}/minute"
限流速率 = 限流配置值 if isinstance(限流配置值, str) else f"{int(限流配置值)}/minute"

去重文件线程锁 = threading.Lock()
去重文件哈希值 = {}

后台任务集合: set = set()
def 提交后台任务(协程):
    任务 = asyncio.create_task(协程)
    后台任务集合.add(任务)
    任务.add_done_callback(后台任务集合.discard)
    return 任务

任务状态字典 = PDict(
    文件路径=Path(API配置值("API_TASK_PATH")) / f"{API配置值('API_TASK_FILE')}.pkl",
    保存间隔=API配置值("API_TASK_FTD"),
    清理间隔=API配置值("API_TASK_VALIDITY_INSP"),
    留存间隔=API配置值("API_TASK_VALIDITY"),
    豁免键=["key", "admin_key"]
)

#基础模块
def 日志行时间(行: str) -> float:
    m = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\]", 行)
    if not m: return -1.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))

产物媒体类型注册表 = dict(TranslatorFile.配置项("API_PRODUCT_MIME", ()))
二进制产物后缀 = {str(后缀).lower() for 后缀 in TranslatorFile.配置项("API_BINARY_SUFFIXES", ())}
产物分组指纹 = None
产物分组缓存 = ({}, {})


def 产物分组(媒体注册表):
    已压缩集 = TranslatorFile.归档后缀集() | 二进制产物后缀
    可压缩, 已压缩 = {}, {}
    for 后缀, 媒体类型 in 媒体注册表.items():
        分组 = 已压缩 if 后缀 in 已压缩集 else 可压缩
        分组[后缀] = 媒体类型
    return 可压缩, 已压缩


def 产物类型分组():
    global 产物分组指纹, 产物分组缓存
    指纹 = (tuple(产物媒体类型注册表.items()), tuple(sorted(TranslatorFile.归档后缀集() | 二进制产物后缀)))
    if 产物分组指纹 != 指纹:
        产物分组指纹, 产物分组缓存 = 指纹, 产物分组(产物媒体类型注册表)
    return 产物分组缓存


def 产物媒体类型(文件名: str) -> str:
    可压缩产物类型, 已压缩产物类型 = 产物类型分组()
    后缀 = Path(文件名).suffix.lower()
    if 后缀 in 可压缩产物类型:
        return 可压缩产物类型[后缀]
    if 后缀 in 已压缩产物类型:
        return 已压缩产物类型[后缀]
    return mimetypes.guess_type(文件名)[0] or "application/octet-stream"


可压缩产物类型, 已压缩产物类型 = 产物类型分组()


def 解析字符串字典(Config: str = Form(None, alias="Config"), 备用字段: str = Form(None, alias="字符串")) -> dict:
    # FastAPI 拿「参数名 / alias」当 multipart 字段名。这里原来只声明了中文参数名 字符串，
    # 而网页与文档发的字段就叫 Config → Config 永远解析成空字典，请求里的 LANGUAGE_INPUT /
    # LANGUAGE_OUTPUT / LANGUAGE_VARIANT / LANGUAGE 全部被丢掉，内核只能回退到服务端默认的
    # zh_cn（于是翻译缓存永远命中 zh_cn 那一桶，文言之类设置完全不生效）。
    原文 = Config if Config is not None else 备用字段   # 备用字段兼容旧调用方
    if 原文 is None: return {}
    try: return json.loads(原文)
    except json.JSONDecodeError: raise HTTPException(400, "Invalid JSON in Config")
def 流式解压保存(流, 目标路径: Path) -> str:
    临时路径 = 目标路径.with_name(目标路径.name + ".part")
    目标路径.parent.mkdir(parents=True, exist_ok=True)
    摘要, 总数 = hashlib.sha3_256(), 0
    try:
        with zstd.open(流, "rb") as 读流, open(临时路径, "wb") as 写流:
            while 块 := 读流.read(管理器实例.Config.API_ZSTD_BLKSIZE):
                总数 += len(块)
                if 总数 > 管理器实例.Config.API_ZSTD_MDFS: raise HTTPException(413, "Decompressed payload too large")
                摘要.update(块); 写流.write(块)
        临时路径.replace(目标路径)
    except BaseException as 异常:
        临时路径.unlink(missing_ok=True)
        if isinstance(异常, (zstd.ZstdError, EOFError)):
            raise HTTPException(400, f"Invalid zstd payload: {异常}") from 异常
        raise
    return 摘要.hexdigest()
def 剔除任务中重复文件(任务ID: str, 哈希值: str):
    with 去重文件线程锁:
        if 哈希值 in 去重文件哈希值.values():
            return True
        else:
            去重文件哈希值[任务ID] = 哈希值
            return False

#Key接口========================================================================================================================================#
配置管理员密钥 = {f"SystemConfig-{序号}": 密钥 for 序号, 密钥 in enumerate([k for k in (API配置值("API_ADMIN_KEY") or []) if isinstance(k, str) and k.strip()], 1)}
for 键 in ("key", "admin_key"):
    if not isinstance(任务状态字典.get(键), dict): 任务状态字典[键] = {}

安全 = HTTPBearer(auto_error=False)
async def 验证密钥抽象(凭证: HTTPAuthorizationCredentials | None = Security(安全), 有效密钥: Dict[str, str] | List[str] | None = None) -> str:
    if 凭证 is None or 凭证.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    密钥 = 凭证.credentials
    候选密钥 = frozenset(k for k in (有效密钥.values() if isinstance(有效密钥, dict) else 有效密钥 or []) if isinstance(k, str) and k.strip())
    if not 候选密钥:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: No valid API keys found")
    if not any(secrets.compare_digest(密钥, k) for k in 候选密钥):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    return 密钥
async def 验证用户密钥(凭证: HTTPAuthorizationCredentials | None = Security(安全)) -> str:
    return await 验证密钥抽象(凭证, 用户密钥字典())
async def 验证管理员密钥(凭证: HTTPAuthorizationCredentials | None = Security(安全)) -> str:
    return await 验证密钥抽象(凭证, 管理员密钥字典())
async def 验证通用密钥(凭证: HTTPAuthorizationCredentials | None = Security(安全)) -> str:
    return await 验证密钥抽象(凭证, 通用密钥字典())

def 用户密钥字典() -> Dict[str, str]:
    密钥字典 = 任务状态字典.get("key")
    if not isinstance(密钥字典, dict):
        密钥字典 = {}
        任务状态字典["key"] = 密钥字典
    return 密钥字典

async def 落盘用户密钥(密钥字典: Dict[str, str]) -> None:
    任务状态字典["key"] = 密钥字典
    await 任务状态字典.保存()

def 拆分密钥(原始值: str | None) -> List[str]:
    return [k.strip() for k in re.split(r"[\s,;]+", 原始值 or "") if k.strip()]

def 生成密钥(uuid4=uuid.uuid4, 前缀: str = "sk-") -> str:
    return f"{前缀}{uuid4().hex}{uuid4().hex}"

def 生成有效密钥(已存在) -> str:
    已有密钥 = set(已存在)
    候选密钥 = 生成密钥()
    while 候选密钥 in 已有密钥:
        候选密钥 = 生成密钥()
    return 候选密钥

def 运行时管理员密钥字典() -> Dict[str, str]:
    密钥字典 = 任务状态字典.get("admin_key")
    if not isinstance(密钥字典, dict):
        密钥字典 = {}
        任务状态字典["admin_key"] = 密钥字典
    return 密钥字典

def 管理员密钥字典() -> Dict[str, str]:
    合并密钥 = dict(配置管理员密钥)
    已有密钥值 = set(合并密钥.values())
    for 备注, 密钥 in 运行时管理员密钥字典().items():
        if 密钥 in 已有密钥值: continue
        唯一备注 = str(备注)
        while 唯一备注 in 合并密钥: 唯一备注 = f"{唯一备注}_"
        合并密钥[唯一备注] = 密钥
        已有密钥值.add(密钥)
    return 合并密钥

def 通用密钥字典() -> Dict[str, str]:
    合并密钥 = {}
    已有密钥值 = set()
    for 前缀, 来源 in (("Admin", 管理员密钥字典()), ("User", 用户密钥字典())):
        for 备注, 密钥 in 来源.items():
            if not isinstance(密钥, str) or not 密钥.strip(): continue
            if 密钥 in 已有密钥值: continue
            唯一备注 = f"{前缀}-{备注}"
            while 唯一备注 in 合并密钥: 唯一备注 = f"{唯一备注}_"
            合并密钥[唯一备注] = 密钥
            已有密钥值.add(密钥)
    return 合并密钥


# API
@API实例.post("/key/user/add", dependencies=[Depends(验证管理员密钥)])
@同时最大线程数.limit(限流速率)
async def 添加用户密钥(request: Request,
    备注: str = Form(..., alias="note")
) -> Dict[str, Any]:
    备注 = 备注.strip()
    if not 备注: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide note")
    密钥字典 = 用户密钥字典()
    if 备注 in 密钥字典: raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Note already exists")
    密钥字典[备注] = 生成有效密钥(密钥字典.values())
    await 落盘用户密钥(密钥字典)
    return {"status": "success", "keys": {备注: 密钥字典[备注]}, "total": len(密钥字典)}

@API实例.post("/key/user/rm", dependencies=[Depends(验证管理员密钥)])
@同时最大线程数.limit(限流速率)
async def 删除用户密钥(request: Request,
    密钥: str | None = Form(None, alias="keys"),
    全部: bool = Form(False, alias="all")
) -> Dict[str, Any]:
    if not 密钥 and not 全部: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide keys, or set all=true")
    密钥字典 = 用户密钥字典()
    待删除 = set() if 全部 else set(拆分密钥(密钥))
    已删除 = dict(密钥字典) if 全部 else {备注: k for 备注, k in 密钥字典.items() if 备注 in 待删除 or k in 待删除}
    for 备注 in 已删除: 密钥字典.pop(备注, None)
    await 落盘用户密钥(密钥字典)
    命中 = set(已删除) | set(已删除.values())
    return {"status": "success", "removed": 已删除, "missing": sorted(待删除 - 命中), "total": len(密钥字典)}
@API实例.post("/key/user/list", dependencies=[Depends(验证管理员密钥)])
@同时最大线程数.limit(限流速率)
async def 获取用户密钥(request: Request) -> Dict[str, Any]:
    密钥字典 = 用户密钥字典()
    return {"status": "success", "keys": dict(密钥字典), "total": len(密钥字典)}

@API实例.post("/key/admin/add", dependencies=[Depends(验证管理员密钥)])
@同时最大线程数.limit(限流速率)
async def 添加管理员密钥(request: Request,
    备注: str = Form(..., alias="note")
) -> Dict[str, Any]:
    备注 = 备注.strip()
    if not 备注: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide note")
    运行时密钥 = 运行时管理员密钥字典()
    if 备注 in 管理员密钥字典(): raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Note already exists")
    运行时密钥[备注] = 生成有效密钥(运行时密钥.values())
    任务状态字典["admin_key"] = 运行时密钥
    await 任务状态字典.保存()
    return {"status": "success", "keys": {备注: 运行时密钥[备注]}, "total": len(管理员密钥字典())}
@API实例.post("/key/admin/rm")
@同时最大线程数.limit(限流速率)
async def 删除管理员密钥(request: Request,
    当前密钥: str = Depends(验证管理员密钥),
    密钥: str | None = Form(None, alias="keys"),
    全部: bool = Form(False, alias="all")
) -> Dict[str, Any]:
    if not 密钥 and not 全部: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide keys, or set all=true")
    运行时密钥 = 运行时管理员密钥字典()
    待删除 = set() if 全部 else set(拆分密钥(密钥))
    配置密钥值 = set(配置管理员密钥.values())
    候选 = {备注: k for 备注, k in 运行时密钥.items() if 全部 or 备注 in 待删除 or k in 待删除}
    保护 = {备注: k for 备注, k in 候选.items() if k in 配置密钥值}
    可删除 = {备注: k for 备注, k in 候选.items() if 备注 not in 保护}
    自身备注 = next((备注 for 备注, k in 可删除.items() if k == 当前密钥), None)
    if 自身备注 is not None: 保护[自身备注] = 可删除.pop(自身备注)
    if not set(管理员密钥字典().values()) - set(可删除.values()):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Refuse to remove the last admin key")
    for 备注 in 可删除: 运行时密钥.pop(备注, None)
    任务状态字典["admin_key"] = 运行时密钥
    await 任务状态字典.保存()
    命中 = set(可删除) | set(可删除.values())
    return {"status": "success", "removed": 可删除, "protected": sorted(保护.values()), "missing": sorted(待删除 - 命中), "total": len(管理员密钥字典())}
@API实例.post("/key/admin/list")
@同时最大线程数.limit(限流速率)
async def 获取管理员密钥(request: Request,
    当前密钥: str = Depends(验证管理员密钥)
) -> Dict[str, Any]:
    密钥字典 = 管理员密钥字典()
    return {"status": "success", "keys": 密钥字典, "total": len(密钥字典)}
#===============================================================================================================================================#

#Core接口=======================================================================================================================================#
def 捕获任务日志(翻译器实例, 任务ID: str, 轮询间隔: float):
    已捕获行数 = [0]
    def 日志轮询():
        开始时间 = time.time()
        while 任务ID in 任务状态字典 and 任务状态字典[任务ID]["status"] not in ["completed", "failed"]:
            if time.time() - 开始时间 > 7200: break
            try:
                日志文本 = 翻译器实例.Log.读取日志()
                if 日志文本 and 任务ID in 任务状态字典:
                    全部日志行 = 日志文本.strip().split('\n')
                    新增日志行 = 全部日志行[已捕获行数[0]:]
                    if 新增日志行:
                        有效日志行 = [行 for 行 in 新增日志行 if 行.strip()]
                        任务状态字典[任务ID]["logs"].extend(有效日志行[-100:])
                        已捕获行数[0] = len(全部日志行)
                        if len(任务状态字典[任务ID]["logs"]) > 500:
                            任务状态字典[任务ID]["logs"] = 任务状态字典[任务ID]["logs"][-500:]
            except Exception:
                print(翻译器实例.Lang("log.api.logs.get.error", e=eb.format_exc()))
            time.sleep(轮询间隔)
    threading.Thread(target=日志轮询, daemon=True).start()

进度阶段区间 = {"item": (30.0, 85.0), "mod": (30.0, 55.0), "mod2": (55.0, 68.0), "plan": (68.0, 85.0)}
进度阶段文案 = {"item": "翻译中", "mod": "翻译模组", "mod2": "解析模组内容", "plan": "解析整合包"}

def 挂进度上报(实例, 任务ID: str):
    上次时间, 上次百分比, 当前阶段 = [0.0], [0.0], [None]
    def 上报(已完成, 总数, 阶段: str = "item"):
        现在 = time.time()
        if 现在 - 上次时间[0] < 0.4: return
        上次时间[0] = 现在
        状态 = 任务状态字典.get(任务ID)
        if not isinstance(状态, dict): return
        if 状态.get("status") in ("completed", "failed"): return
        if 阶段 == "item" and 当前阶段[0] in ("mod", "mod2", "plan"): return
        if 阶段 in ("mod", "mod2", "plan"): 当前阶段[0] = 阶段
        总数 = int(总数 or 0)
        已完成 = max(0, int(已完成 or 0))
        起点, 终点 = 进度阶段区间.get(阶段, 进度阶段区间["item"])
        百分比 = round(起点 + (终点 - 起点) * (min(1.0, 已完成 / 总数) if 总数 else 0.0), 1)
        if 百分比 <= 上次百分比[0]: return
        上次百分比[0] = 百分比
        状态["done"] = 已完成
        状态["total"] = 总数
        状态["phase"] = 进度阶段文案.get(阶段, "翻译中")
        状态["progress"] = 百分比
    try: 实例.进度上报 = 上报
    except Exception: pass
    return 上报

async def 执行任务(任务ID, 实例, 函数, **参数):
    任务状态字典[任务ID]["_开始时间"] = 日志行时间(实例.Log.格式化("log.api.task.initialization.start"))
    任务状态字典[任务ID]["logs"] = [实例.Log.格式化("log.api.task.initialization.start")]
    捕获任务日志(实例, 任务ID, 2.5)
    任务状态字典[任务ID]["progress"] = 10
    任务状态字典[任务ID]["phase"] = "初始化"
    任务状态字典[任务ID]["logs"].append(实例.Log.格式化("log.api.task.enqueued"))
    async with 任务并行锁:
        任务状态字典[任务ID]["progress"] = 20
        任务状态字典[任务ID]["phase"] = "等待资源"
        任务状态字典[任务ID]["logs"].append(实例.Log.格式化("log.api.task.acquiring.resources.success"))
        try:
            任务状态字典[任务ID]["progress"] = 30
            任务状态字典[任务ID]["phase"] = "翻译中"
            任务状态字典[任务ID]["logs"].append(实例.Log.格式化("log.api.task.processing"))
            挂进度上报(实例, 任务ID)
            结果 = await asyncio.to_thread(函数, **参数)
            任务状态字典[任务ID]["progress"] = 90
            任务状态字典[任务ID]["phase"] = "收尾"
            任务状态字典[任务ID]["logs"].append(实例.Log.格式化("log.api.task.completed"))
            if 结果 is None:
                任务状态字典[任务ID].update({
                    "status": "failed",
                    "phase": "失败",
                    "progress": 100,
                    "error": 实例.Log.格式化("log.api.task.error.none_result"),
                    "logs": 任务状态字典[任务ID].get("logs", []) + [实例.Log.格式化("log.api.task.error.none_result")]})
                去重文件哈希值.pop(任务ID, None)
            else:
                刷新 = getattr(实例.Log, "刷新日志", None)
                if callable(刷新):
                    try: 刷新()
                    except Exception: pass
                输出日志 = 实例.Log.读取日志()
                if 输出日志:
                    输出日志行 = [i for i in 输出日志.strip().split('\n') if i.strip()]
                    起始时间 = 任务状态字典[任务ID].get("_开始时间", 0)
                    本次日志行 = [行 for 行 in 输出日志行 if 日志行时间(行) >= 起始时间] if 起始时间 else []
                    本次日志行 = 本次日志行 or 输出日志行[-30:]
                    已有日志 = 任务状态字典[任务ID].get("logs", [])
                    已有集合, 已有去重 = set(), []
                    for 行 in 已有日志:
                        if 行 and 行 not in 已有集合:
                            已有集合.add(行)
                            已有去重.append(行)
                    新增行 = [行 for 行 in 本次日志行 if 行 not in 已有集合]
                    任务状态字典[任务ID]["logs"] = (已有去重 + 新增行)[-200:]
                任务状态字典[任务ID].update({
                    "status": "completed",
                    "phase": "已完成",
                    "progress": 100,
                    "result_path": str(结果),
                    "filename": Path(结果).name,
                    "logs": 任务状态字典[任务ID].get("logs", []) + [实例.Log.格式化("log.api.task.completed")]})
                去重文件哈希值.pop(任务ID, None)
        except Exception as e:
            任务状态字典[任务ID].update({
                "status": "failed",
                "phase": "失败",
                "progress": 100,
                "error": str(e),
                "logs": 任务状态字典[任务ID].get("logs", []) + [实例.Log.格式化("log.api.error", e=eb.format_exc(), info_level=2)]
            })
            去重文件哈希值.pop(任务ID, None)
        finally:
            try: 实例.Log.关闭()
            except Exception: pass
async def 双路径流式处理(File0, FileName0, Config, File1=None, FileName1=None, Mode="Translator"):
    File1目录 = None
    任务ID = 管理器实例.Module.uuid()
    缓存目录 = Path(管理器实例.Config.PATH_CACHE) / 任务ID
    翻译器实例 = 管理器实例.get_translator_temporary(Config) if Mode == "Translator" else 管理器实例.get_tool_temporary(Config)
    try:
        File0目录 = 缓存目录 / FileName0
        File0哈希 = 流式解压保存(File0.file, File0目录)
    except:
        shutil.rmtree(缓存目录, ignore_errors=True)
        try: 翻译器实例.Log.关闭()
        except Exception: pass
        raise
    if File1 and FileName1:
        File1目录 = 缓存目录 / FileName1
        流式解压保存(File1.file, File1目录)
    if 管理器实例.Config.API_TASK_DEDUP and 剔除任务中重复文件(任务ID, File0哈希):
        shutil.rmtree(缓存目录, ignore_errors=True)
        try: 翻译器实例.Log.关闭()
        except Exception: pass
        任务状态字典[任务ID] = {"status": "success", "progress": 0, "result_path": None, "error": 翻译器实例.Lang("log.api.task.error.file.exists"), "logs": []}
        return (False, {"task_id": 任务ID, "status": "success", "message": 翻译器实例.Lang("log.api.task.error.file.exists")})
    return (True, (任务ID, 翻译器实例, File0目录, File1目录))
async def 双路径本地处理(File0, Config, File1=None, Mode="Translator"):
    任务ID = 管理器实例.Module.uuid()
    翻译器实例 = 管理器实例.get_translator_temporary(Config) if Mode == "Translator" else 管理器实例.get_tool_temporary(Config)
    return 任务ID, 翻译器实例, File0, File1

@API实例.post("/network/transl", dependencies=[Depends(验证通用密钥)])
@同时最大线程数.limit(限流速率)
async def 翻译网络文件(request: Request, BackgroundTasks: BackgroundTasks,
                      File0: UploadFile, FileName0: str = Form(...),
                      Config: dict = Depends(解析字符串字典),
                      File1: UploadFile = None, FileName1: str = Form(None),
                      AllMode: bool = Form(False), ExportInspection: bool = Form(False)) -> dict:
    现在进行时, 任务信息 = await 双路径流式处理(File0, FileName0, Config, File1, FileName1)
    if 现在进行时: 任务ID, 应用实例, File0目录, File1目录 = 任务信息
    else: return 任务信息
    任务状态字典[任务ID] = {"status": "queued", "progress": 0, "phase": "排队中", "result_path": None, "error": None, "logs": []}
    提交后台任务(执行任务(任务ID, 应用实例, 应用实例.翻译通用文件,
                                file0=File0目录, file1=File1目录, all_mode=AllMode, export_inspection=ExportInspection))
    return {"task_id": 任务ID, "status": "queued", "message": 应用实例.Lang("log.api.task.submitted")}
@API实例.post("/local/transl", dependencies=[Depends(验证管理员密钥)])
async def 翻译本地文件(request: Request, BackgroundTasks: BackgroundTasks,
                      File0: str,
                      Config: dict = Depends(解析字符串字典),
                      File1: str = Form(None),
                      AllMode: bool = Form(False), ExportInspection: bool = Form(False)) -> dict:
    任务ID, 应用实例, File0目录, File1目录 = await 双路径本地处理(File0, Config, File1)
    任务状态字典[任务ID] = {"status": "queued", "progress": 0, "phase": "排队中", "result_path": None, "error": None, "logs": []}
    提交后台任务(执行任务(任务ID, 应用实例, 应用实例.翻译通用文件,
                                file0=File0目录, file1=File1目录, all_mode=AllMode, export_inspection=ExportInspection))
    return {"task_id": 任务ID, "status": "queued", "message": 应用实例.Lang("log.api.task.submitted")}
#===============================================================================================================================================#

#基础接口========================================================================================================================================#
@API实例.get("/", include_in_schema=False)
async def 服务探活():
    return {"service": "TranslationMinecraft API", "status": "ok"}

def 清理任务缓存(任务ID: str):
    if 任务ID in 任务状态字典:
        缓存目录 = Path(管理器实例.Config.PATH_CACHE) / 任务ID
        if 缓存目录.exists(): shutil.rmtree(缓存目录, ignore_errors=True)
        del 任务状态字典[任务ID]

@API实例.get("/task/status/{task_id}", dependencies=[Depends(验证通用密钥)])
async def 查询任务状态(task_id: str):
    if task_id not in 任务状态字典: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return 任务状态字典[task_id]
@API实例.get("/task/download/{task_id}", dependencies=[Depends(验证通用密钥)])
async def 下载任务结果(task_id: str, 后台任务: BackgroundTasks):
    if task_id not in 任务状态字典: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    状态 = 任务状态字典[task_id]
    if 状态["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Task not ready. Current status: {状态['status']}")
    文件路径 = 状态["result_path"]
    文件名 = 状态.get("filename", Path(文件路径).name)
    后台任务.add_task(清理任务缓存, task_id)
    编码文件名 = quote(文件名, safe='')
    return FileResponse(
        path=文件路径,
        filename=文件名,
        media_type=产物媒体类型(文件名),
        headers={"Content-Disposition": f'attachment; filename="{文件名.encode("ascii", errors="ignore").decode() or "download"}"; filename*=UTF-8\'\'{编码文件名}' })
@API实例.get("/task/logs/{task_id}", response_class=PlainTextResponse, dependencies=[Depends(验证通用密钥)])
async def 获取任务日志(task_id: str, last_lines: int = 100):
    if task_id not in 任务状态字典:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    日志 = 任务状态字典[task_id].get("logs", [])
    显示日志 = 日志[-last_lines:] if len(日志) > last_lines else 日志
    return "\n".join(显示日志) 

@API实例.get("/task/logs/{task_id}/json", dependencies=[Depends(验证通用密钥)])
async def 获取任务日志Json(task_id: str, last_lines: int = 100):
    if task_id not in 任务状态字典:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    日志 = 任务状态字典[task_id].get("logs", [])
    显示日志 = 日志[-last_lines:] if len(日志) > last_lines else 日志
    return {
        "task_id": task_id,
        "status": 任务状态字典[task_id].get("status", "unknown"),
        "progress": 任务状态字典[task_id].get("progress", 0),
        "logs": 显示日志,
        "error": 任务状态字典[task_id].get("error"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
#===============================================================================================================================================#
if __name__ == "__main__":
    传入参数 = {
        "host": API配置值("API_HOST", "127.0.0.1"),
        "port": int(API配置值("API_PORT", 49180)),
    }
    if API配置值("API_SSL_KEY_FILE"): 传入参数["ssl_keyfile"] = API配置值("API_SSL_KEY_FILE")
    if API配置值("API_SSL_CERT_FILE"): 传入参数["ssl_certfile"] = API配置值("API_SSL_CERT_FILE")
    uvicorn.run(API实例, **传入参数)