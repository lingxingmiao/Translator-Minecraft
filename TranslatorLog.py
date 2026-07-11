from TranslatorLib import (queue, Path, logging, QueueHandler, QueueListener, re, FileHandler,
                           threading, time, atexit,
                           RuntimeConfig, DEFAULT_CONFIG, Locale)
class NoRotateHandler(FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False, flush_interval=5.0):
        super().__init__(filename, mode, encoding, delay)
        self._flush_interval = flush_interval
        self._buffer = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._定时刷新, daemon=True)
        self._flush_thread.start()
        atexit.register(self._退出刷新)
    def emit(self, record):
        try:
            msg = self.format(record)
            with self._lock:
                self._buffer.append(msg + self.terminator)
                if time.time() - self._last_flush >= self._flush_interval:
                    self._执行刷新()
        except Exception:
            self.handleError(record)
    def _执行刷新(self):
        if self._buffer:
            try:
                if self.stream is None:
                    self.stream = self._open()
                self.stream.writelines(self._buffer)
                self.stream.flush()
            except Exception:
                pass
            self._buffer.clear()
        self._last_flush = time.time()
    def _定时刷新(self):
        while not self._stop_event.wait(self._flush_interval):
            with self._lock:
                self._执行刷新()
    def _退出刷新(self):
        self._stop_event.set()
        with self._lock:
            self._执行刷新()
    def close(self):
        self._退出刷新()
        super().close()
class Log:
    def __init__(Self, Config: dict = None):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.Lang = Locale(Config=Config).Lang
        Self.启动日志()
    def __enter__(Self):
        return Self
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
        刷新间隔 = getattr(Self.Config, 'LOGS_FLUSH_INTERVAL', 5.0)
        文件处理器 = NoRotateHandler(str(日志文件), encoding='utf-8', mode='a', flush_interval=刷新间隔)
        文件处理器.setFormatter(格式化器)
        处理器列表 = [文件处理器]
        默认日志 = Path(f"{DEFAULT_CONFIG.LOGS_FILE_PATH}/{DEFAULT_CONFIG.LOGS_FILE_NAME}.log").resolve()
        if Self.Config.LOGS_GLOBAL and 日志文件 != 默认日志:
            默认日志.parent.mkdir(parents=True, exist_ok=True)
            全局处理器 = NoRotateHandler(str(默认日志), encoding='utf-8', mode='a', flush_interval=刷新间隔)
            全局处理器.setFormatter(格式化器)
            处理器列表.append(全局处理器)
        Self._队列监听器 = QueueListener(日志队列, *处理器列表, respect_handler_level=True)
        Self._队列监听器.start()
        Self.日志管理器.propagate = False
    def 写入日志(Self, text: str, info_level: int = 0, **kwargs):
        等级映射 = {0: logging.INFO, 1: logging.WARNING, 2: logging.ERROR, 3: logging.FATAL, 4: logging.DEBUG}
        等级 = 等级映射.get(info_level, logging.INFO)
        if info_level == 4 and not Self.Config.DEBUG_MODE:
            return
        本地化文本 = Self.Lang(text, **kwargs)
        含堆栈 = bool(re.search(r'traceback|trace back|文件.*行|line \d+', 本地化文本, re.I))
        if not 含堆栈:
            本地化文本 = 本地化文本.replace('\n', ' ').replace('\r', '')
        Self.日志管理器.log(等级, "%s", 本地化文本)
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