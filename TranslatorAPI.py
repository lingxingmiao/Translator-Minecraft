from TranslatorLib import (APIConfig, Dict, uvicorn, Path as pt, time, json, asyncio, uuid, shutil, threading, eb, Dict, Any, atexit, sqlite3, quote, hashlib, re,
                           Translator, Tool,
                           FastAPI, UploadFile, HTTPException, status, Depends, Security, Form, Request, BackgroundTasks, FileResponse, PlainTextResponse, HTTPBearer, HTTPAuthorizationCredentials, CORSMiddleware, Limiter, _rate_limit_exceeded_handler, get_remote_address, RateLimitExceeded, asynccontextmanager)

@asynccontextmanager
async def 应用生命周期(app):
    yield
    from TranslatorPersistence import 关闭所有异步会话
    await 关闭所有异步会话()

FastAPI = FastAPI(title="TranslationMinecraft", lifespan=应用生命周期)
FastAPI.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
限流器 = Limiter(key_func=lambda request: request.headers.get("Authorization", get_remote_address(request)))
FastAPI.state.limiter = 限流器
FastAPI.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
请求并行数 = asyncio.Semaphore(APIConfig["api"]["max_concurrent"])
API翻译器实例 = Translator(APIConfig["server"])

class 持久化状态字典(Dict[str, Any]):
    def __init__(Self, db_path: str = "task_states.db", save_interval: float = 5.0, cleanup_hours: float = 24.0, cleanup_interval: float = 300.0):
        super().__init__()
        Self._db_path = pt(db_path)
        Self._save_interval = save_interval
        Self._cleanup_hours = cleanup_hours
        Self._cleanup_interval = cleanup_interval
        Self._lock = threading.Lock()
        Self._stop_event = threading.Event()
        Self._创建时间: Dict[str, float] = {}
        Self._db_path.parent.mkdir(parents=True, exist_ok=True)
        Self._conn = sqlite3.connect(str(Self._db_path), check_same_thread=False)
        Self._conn.execute("PRAGMA journal_mode=WAL;")
        Self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        Self._conn.commit()
        Self._加载数据库()
        threading.Thread(target=Self._轮询同步, daemon=True).start()
        threading.Thread(target=Self._轮询清理, daemon=True).start()
        atexit.register(Self._优雅退出)
    def _加载数据库(Self):
        with Self._lock:
            try:
                for 任务ID, 状态JSON, 创建时间戳 in Self._conn.execute("SELECT task_id, state, created_at FROM tasks"):
                    状态数据 = json.loads(状态JSON)
                    super().__setitem__(任务ID, 状态数据)
                    Self._创建时间[任务ID] = 创建时间戳
            except Exception as e:
                print(API翻译器实例.Lang("log.api.task.load.error", e=eb.format_exc()))
    def _执行同步(Self):
        with Self._lock:
            try:
                批量数据 = []
                for 任务ID, 状态数据 in Self.items():
                    时间戳 = Self._创建时间.get(任务ID, time.time())
                    安全快照 = dict(状态数据)
                    批量数据.append((任务ID, json.dumps(安全快照, ensure_ascii=False), 时间戳))
                
                if 批量数据:
                    Self._conn.executemany(
                        "INSERT OR REPLACE INTO tasks (task_id, state, created_at) VALUES (?, ?, ?)",
                        批量数据
                    )
                    Self._conn.commit()
            except Exception as e:
                print(API翻译器实例.Lang("log.api.task.sync.error", e=eb.format_exc()))
    def _执行清理(Self):
        截止时间 = time.time() - (Self._cleanup_hours * 3600)
        with Self._lock:
            try:
                Self._conn.execute("DELETE FROM tasks WHERE created_at < ?", (截止时间,))
                Self._conn.commit()
                过期任务 = [tid for tid, t in Self._创建时间.items() if t < 截止时间]
                for tid in 过期任务:
                    Self._创建时间.pop(tid, None)
                    if super().__contains__(tid):
                        super().__delitem__(tid)
            except Exception as e:
                print(API翻译器实例.Lang("log.api.task.clean.error", e=eb.format_exc()))
    def _轮询同步(Self):
        while not Self._stop_event.wait(Self._save_interval):
            Self._执行同步()
    def _轮询清理(Self):
        while not Self._stop_event.wait(Self._cleanup_interval):
            Self._执行清理()
    def _优雅退出(Self):
        Self._stop_event.set()
        Self._执行同步()
        Self._conn.close()
    def __setitem__(Self, key, value):
        with Self._lock:
            if key not in Self._创建时间:
                Self._创建时间[key] = time.time()
            super().__setitem__(key, value)
    def __delitem__(Self, key):
        with Self._lock:
            if super().__contains__(key):
                Self._conn.execute("DELETE FROM tasks WHERE task_id = ?", (key,))
                Self._conn.commit()
            Self._创建时间.pop(key, None)
            super().__delitem__(key)
    def clear(Self):
        with Self._lock:
            Self._conn.execute("DELETE FROM tasks")
            Self._conn.commit()
            Self._创建时间.clear()
            super().clear()
任务状态字典 = 持久化状态字典(
    db_path=APIConfig["api"]["task_states_file"],
    save_interval=APIConfig["api"]["task_states_save_interval"],
    cleanup_hours=APIConfig["api"]["task_states_cleanup_hours"],
    cleanup_interval=APIConfig["api"]["task_states_cleanup_interval"]
)
文件哈希值 = {}

def 设置时间():
    时间 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + f"{int((time.time() % 1) * 10000):04d}"
    APIConfig["server"]['LOGS_FILE_NAME'] = f"logs-{时间}"

安全 = HTTPBearer(auto_error=False)
async def 验证Key(凭证: HTTPAuthorizationCredentials = Security(安全)):
    if 凭证 is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    密钥 = 凭证.credentials
    with open("config-api.cfg", 'r', encoding='utf-8') as f:
        data = json.load(f)
    有效密钥 = data.get("keys", [])
    if not 有效密钥:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: No valid API keys found")
    if 密钥 not in 有效密钥:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    return 密钥
def _捕获任务日志(翻译器实例: Translator, task_id: str, interval: float = 5.0):
    """定期捕获翻译器日志并追加到任务日志池（增量模式）"""
    已捕获行数 = [0]
    def _日志轮询():
        while task_id in 任务状态字典 and 任务状态字典[task_id]["status"] not in ["completed", "failed"]:
            try:
                logs_text = 翻译器实例.Log.读取日志()
                if logs_text and task_id in 任务状态字典:
                    all_lines = logs_text.strip().split('\n')
                    new_lines = all_lines[已捕获行数[0]:]
                    if new_lines:
                        timestamped_lines = [line for line in new_lines if line.strip()]
                        任务状态字典[task_id]["logs"].extend(timestamped_lines[-100:])
                        已捕获行数[0] = len(all_lines)
                        if len(任务状态字典[task_id]["logs"]) > 500:
                            任务状态字典[task_id]["logs"] = 任务状态字典[task_id]["logs"][-500:]
            except Exception:
                print(翻译器实例.Lang("log.api.logs.get.error", e=eb.format_exc()))
            time.sleep(interval)
    threading.Thread(target=_日志轮询, daemon=True).start()

async def _执行核心任务(task_id: str, 处理函数, 翻译器实例: Translator, 缓存目录: pt, **任务参数):
    """通用后台任务执行逻辑，包含并发控制、进度更新、日志捕获与异常捕获"""
    任务状态字典[task_id]["logs"] = [f"[INFO] {翻译器实例.Lang("log.api.task.initialization.start")}"]
    _捕获任务日志(翻译器实例, task_id)
    任务状态字典[task_id]["progress"] = 10
    任务状态字典[task_id]["logs"].append(f"[INFO] {翻译器实例.Lang("log.api.task.enqueued")}")
    async with 请求并行数:
        任务状态字典[task_id]["progress"] = 20
        任务状态字典[task_id]["logs"].append(f"[INFO] {翻译器实例.Lang("log.api.task.acquiring.resources.success")}")
        try:
            任务状态字典[task_id]["progress"] = 30
            任务状态字典[task_id]["logs"].append(f"[INFO] {翻译器实例.Lang("log.api.task.processing")}")
            结果 = await asyncio.to_thread(处理函数, **任务参数)
            任务状态字典[task_id]["progress"] = 90
            任务状态字典[task_id]["logs"].append(f"[INFO] {翻译器实例.Lang("log.api.task.completed")}")
            
            if 结果 is None:
                任务状态字典[task_id].update({
                    "status": "failed",
                    "progress": 100,
                    "error": 翻译器实例.Lang("log.api.task.error.none_result"),
                    "logs": 任务状态字典[task_id].get("logs", []) + [f"[ERROR] {翻译器实例.Lang("log.api.task.error.none_result")}"]
                })
                文件哈希值.pop(task_id, None)
            else:
                final_logs = 翻译器实例.Log.读取日志()
                if final_logs:
                    任务状态字典[task_id]["logs"].extend(
                        [line for line in final_logs.strip().split('\n') if line.strip()][-20:]
                    )
                
                任务状态字典[task_id].update({
                    "status": "completed",
                    "progress": 100,
                    "result_path": str(结果),
                    "filename": pt(结果).name,
                    "logs": 任务状态字典[task_id].get("logs", []) + [f"[INFO] {翻译器实例.Lang("log.api.task.completed")}"]
                })
                文件哈希值.pop(task_id, None)
        except Exception as e:
            任务状态字典[task_id].update({
                "status": "failed",
                "progress": 100,
                "error": str(e),
                "logs": 任务状态字典[task_id].get("logs", []) + [f"[ERROR] {翻译器实例.Lang("log.api.error", e=eb.format_exc())}"]
            })
            文件哈希值.pop(task_id, None)
def 清理任务缓存(task_id: str):
    """下载完成后或超时后清理缓存目录"""
    if task_id in 任务状态字典:
        缓存目录 = pt(APIConfig["server"].get("PATH_CACHE", "cache")) / task_id
        if 缓存目录.exists():
            shutil.rmtree(缓存目录, ignore_errors=True)
        del 任务状态字典[task_id]
        
def 剔除任务中重复文件(task_id: str, sha: str):
    if sha in 文件哈希值.values():
        return True
    else:
        文件哈希值[task_id] = sha
        return False
def 解析字符串字典(字符串: str = Form(None)) -> dict:
    if 字符串 is None:
        return {}
    try:
        return json.loads(字符串)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in translatorcore")

def 过滤用户配置(用户配置: dict) -> dict:
    规则 = {
        "allow_patterns": API翻译器实例.Config.API_TRANSLATOR_CORE_CONFIG_WHITE,
        "deny_patterns": API翻译器实例.Config.API_TRANSLATOR_CORE_CONFIG_BLACK,
        "range_checks": API翻译器实例.Config.API_TRANSLATOR_CORE_CONFIG_RANGE
    }
    过滤后配置 = {}
    for 键, 值 in 用户配置.items():
        if any(re.match(模式, 键) for 模式 in 规则["deny_patterns"]):
            continue
        if not any(re.match(模式, 键) for 模式 in 规则["allow_patterns"]):
            continue
        for 模式, (最小值, 最大值) in 规则["range_checks"].items():
            if re.match(模式, 键) and isinstance(值, (int, float)):
                值 = max(最小值, min(值, 最大值))
                break
        过滤后配置[键] = 值
    return 过滤后配置
@FastAPI.post("/translate", dependencies=[Depends(验证Key)])
@限流器.limit(APIConfig["api"]["current-limiting"])
async def 提交翻译任务(
    request: Request, background_tasks: BackgroundTasks,
    file0: UploadFile, file_name0: str = Form(...),
    translatorcore: dict = Depends(解析字符串字典),
    file1: UploadFile = None, file_name1: str = Form(None),
    all_mode: bool = Form(False), export_inspection: bool = Form(False)
) -> Dict:
    设置时间()
    task_id = uuid.uuid4().hex
    file0_content = await file0.read()
    file0_hash = hashlib.sha3_256(file0_content).hexdigest()
    翻译器实例 = Translator(APIConfig["server"] | 过滤用户配置(translatorcore))
    if APIConfig["api"]["transalator_file_exists_del"] and 剔除任务中重复文件(task_id, file0_hash):
        任务状态字典[task_id] = {"status": "success", "progress": 0, "result_path": None, "error": 翻译器实例.Lang("log.api.task.error.file.exists"), "logs": []}
        return {"task_id": task_id, "status": "success", "message": 翻译器实例.Lang("log.api.task.error.file.exists")}
    缓存目录 = pt(f"{翻译器实例.Config.PATH_CACHE}/{task_id}/")
    缓存目录.mkdir(parents=True, exist_ok=True)
    file0_path = 缓存目录 / file_name0
    with open(file0_path, "wb") as f:
        f.write(file0_content)
    file1_path = ""
    if file1 and file_name1:
        file1_path = 缓存目录 / file_name1
        with open(file1_path, "wb") as f:
            f.write(await file1.read())
    任务状态字典[task_id] = {"status": "queued", "progress": 0, "result_path": None, "error": None, "logs": []}
    asyncio.create_task(
        _执行核心任务(
            task_id, 
            翻译器实例.翻译通用文件, 
            翻译器实例, 
            缓存目录, 
            file0=str(file0_path), 
            file1=str(file1_path), 
            all_mode=all_mode, 
            export_inspection=export_inspection
        )
    )
    return {"task_id": task_id, "status": "queued", "message": 翻译器实例.Lang("log.api.task.submitted")}

@FastAPI.post("/separatelangupdate", dependencies=[Depends(验证Key)])
@限流器.limit(APIConfig["api"]["current-limiting"])
async def 提交分离语言更新任务(
    request: Request, background_tasks: BackgroundTasks,
    file0: UploadFile, file_name0: str = Form(...),
    translatorcore: dict = Depends(解析字符串字典),
    file1: UploadFile = None, file_name1: str = Form(None)
) -> Dict:
    设置时间()
    task_id = uuid.uuid4().hex
    翻译器实例 = Tool(APIConfig["server"] | 过滤用户配置(translatorcore))
    缓存目录 = pt(f"{翻译器实例.Config.PATH_CACHE}/{task_id}/")
    缓存目录.mkdir(parents=True, exist_ok=True)
    file0_path = 缓存目录 / file_name0
    with open(file0_path, "wb") as f:
        f.write(await file0.read())
    file1_path = ""
    if file1 and file_name1:
        file1_path = 缓存目录 / file_name1
        with open(file1_path, "wb") as f:
            f.write(await file1.read())

    任务状态字典[task_id] = {"status": "queued", "progress": 0, "result_path": None, "error": None, "logs": []}
    asyncio.create_task(
        _执行核心任务(
            task_id,
            翻译器实例.分离语言文件更新,
            翻译器实例,
            缓存目录,
            file0=str(file0_path),
            file1=str(file1_path),
            output_path=str(缓存目录 / "output"),
        )
    )
    return {"task_id": task_id, "status": "queued", "message": 翻译器实例.Lang("log.api.task.submitted")}

@FastAPI.post("/mergelangupdate", dependencies=[Depends(验证Key)])
@限流器.limit(APIConfig["api"]["current-limiting"])
async def 提交合并语言更新任务(
    request: Request, background_tasks: BackgroundTasks,
    file0: UploadFile, notlang_file: UploadFile,
    translatorcore: dict = Depends(解析字符串字典),
    file_name0: str = Form(...), nolang_file_name: str = Form(...),
    file1: UploadFile = None, file_name1: str = Form(None)
) -> Dict:
    设置时间()
    task_id = uuid.uuid4().hex
    翻译器实例 = Tool(APIConfig["server"] | 过滤用户配置(translatorcore))
    缓存目录 = pt(f"{翻译器实例.Config.PATH_CACHE}/{task_id}/")
    缓存目录.mkdir(parents=True, exist_ok=True)
    file0_path = 缓存目录 / file_name0
    with open(file0_path, "wb") as f:
        f.write(await file0.read())
    notlang_path = 缓存目录 / nolang_file_name
    with open(notlang_path, "wb") as f:
        f.write(await notlang_file.read())
    file1_path = ""
    if file1 and file_name1:
        file1_path = 缓存目录 / file_name1
        with open(file1_path, "wb") as f:
            f.write(await file1.read())
    任务状态字典[task_id] = {"status": "queued", "progress": 0, "result_path": None, "error": None, "logs": []}
    asyncio.create_task(
        _执行核心任务(
            task_id,
            翻译器实例.合并语言文件更新,
            翻译器实例,
            缓存目录,
            file0=str(file0_path),
            file1=str(file1_path),
            output_path=str(缓存目录 / "output"),
            notlang_file=str(notlang_path)
        )
    )
    return {"task_id": task_id, "status": "queued", "message": 翻译器实例.Lang("log.api.task.submitted")}
@FastAPI.get("/task/status/{task_id}")
async def 查询任务状态(task_id: str):
    if task_id not in 任务状态字典:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return 任务状态字典[task_id]
@FastAPI.get("/task/download/{task_id}")
async def 下载任务结果(task_id: str, background_tasks: BackgroundTasks):
    if task_id not in 任务状态字典:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    状态 = 任务状态字典[task_id]
    if 状态["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Task not ready. Current status: {状态['status']}"
        )
    文件路径 = 状态["result_path"]
    文件名 = 状态.get("filename", pt(文件路径).name)
    background_tasks.add_task(清理任务缓存, task_id)
    encoded_filename = quote(文件名, safe='')
    return FileResponse(
        path=文件路径,
        filename=文件名,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{文件名.encode("ascii", errors="ignore").decode() or "download"}"; filename*=UTF-8\'\'{encoded_filename}'
        }
    )
@FastAPI.get("/task/logs/{task_id}", response_class=PlainTextResponse)
async def 获取任务日志(task_id: str, last_lines: int = 100):
    """
    获取指定任务的实时日志
    
    参数:
        task_id: 任务ID
        last_lines: 返回最后N行日志（默认100行）
    
    返回:
        纯文本格式的日志内容
    """
    if task_id not in 任务状态字典:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    logs = 任务状态字典[task_id].get("logs", [])
    display_logs = logs[-last_lines:] if len(logs) > last_lines else logs
    status_summary = f"\n\n===== 任务状态摘要 =====\n"
    status_summary += f"任务ID: {task_id}\n"
    status_summary += f"当前状态: {任务状态字典[task_id].get('status', 'unknown')}\n"
    status_summary += f"进度: {任务状态字典[task_id].get('progress', 0)}%\n"
    if 任务状态字典[task_id].get("error"):
        status_summary += f"错误: {任务状态字典[task_id].get('error')}\n"
    
    return "\n".join(display_logs) + status_summary

@FastAPI.get("/task/logs/{task_id}/json")
async def 获取任务日志_json(task_id: str, last_lines: int = 100):
    """
    获取指定任务的日志（JSON格式）
    
    返回:
        {
            "task_id": "xxx",
            "status": "processing",
            "progress": 45,
            "logs": ["[INFO] ...", "..."],
            "timestamp": "2026-04-08T12:34:56Z"
        }
    """
    if task_id not in 任务状态字典:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    logs = 任务状态字典[task_id].get("logs", [])
    display_logs = logs[-last_lines:] if len(logs) > last_lines else logs
    return {
        "task_id": task_id,
        "status": 任务状态字典[task_id].get("status", "unknown"),
        "progress": 任务状态字典[task_id].get("progress", 0),
        "logs": display_logs,
        "error": 任务状态字典[task_id].get("error"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
if __name__ == "__main__":
    传入参数 = {}
    if APIConfig["api"]["host"]:
        传入参数["host"] = APIConfig["api"]["host"]
    if APIConfig["api"]["port"]:
        传入参数["port"] = APIConfig["api"]["port"]
    if APIConfig["api"]["ssl_keyfile"]:
        传入参数["ssl_keyfile"] = APIConfig["api"]["ssl_keyfile"]
    if APIConfig["api"]["ssl_certfile"]:
        传入参数["ssl_certfile"] = APIConfig["api"]["ssl_certfile"]
    uvicorn.run(FastAPI, **传入参数)