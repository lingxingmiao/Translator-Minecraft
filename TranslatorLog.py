from TranslatorLib import (queue, Path, logging, QueueHandler, QueueListener, re, FileHandler, threading, time, atexit,
                           DEFAULT_CONFIG, Config)
logging.addLevelName(logging.CRITICAL, 'FATAL')
logging.addLevelName(logging.WARNING, 'WARN')
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
        try: atexit.unregister(self._退出刷新)
        except Exception: pass
        super().close()
class Log:
    def __init__(Self, App: Config):
        Self.Config = App.Config
        Self.Lang = App.Lang
        Self.等级映射 = {0: logging.INFO, 1: logging.WARNING, 2: logging.ERROR, 3: logging.FATAL, 4: logging.DEBUG}
        Self.启动日志()
    def __enter__(Self):
        return Self
    def 启动日志(Self):
        旧监听器 = getattr(Self, "_队列监听器", None)
        if 旧监听器 is not None:
            try: 旧监听器.stop()
            except Exception: pass
            Self._队列监听器 = None
        Self.日志名称 = f"Translator_{id(Self)}"
        Self.日志管理器 = logging.getLogger(Self.日志名称)
        Self.日志管理器.setLevel(logging.DEBUG if Self.Config.DEBUG_MODE else logging.INFO)
        if Self.日志管理器.handlers:
            for 处理器 in list(Self.日志管理器.handlers):
                Self.日志管理器.removeHandler(处理器)
                try: 处理器.close()
                except Exception: pass
        日志队列 = queue.Queue(-1)
        Self._日志队列 = 日志队列
        队列处理器 = QueueHandler(日志队列)
        Self.日志管理器.addHandler(队列处理器)
        日志文件 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log").resolve()
        日志文件.parent.mkdir(parents=True, exist_ok=True)
        格式化器 = logging.Formatter(fmt='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
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
        等级 = Self.等级映射.get(info_level, logging.INFO)
        if info_level == 4 and not Self.Config.DEBUG_MODE:
            return
        本地化文本 = Self.Lang(text, **kwargs)
        含堆栈 = bool(re.search(r'traceback|trace back|文件.*行|line \d+', 本地化文本, re.I))
        if not 含堆栈:
            本地化文本 = 本地化文本.replace('\n', ' ').replace('\r', '')
        Self.日志管理器.log(等级, "%s", 本地化文本)
    def 格式化(Self, text: str, info_level: int = 0, **kwargs):
        等级 = Self.等级映射.get(info_level, logging.INFO)
        if info_level == 4 and not Self.Config.DEBUG_MODE: return ""
        本地化文本 = str(Self.Lang(text, **kwargs))
        含堆栈 = bool(re.search(r'traceback|trace back|文件.*行|line \d+', 本地化文本, re.I))
        if not 含堆栈: 本地化文本 = 本地化文本.replace('\n', ' ').replace('\r', '')
        记录 = logging.LogRecord(name=getattr(Self, "日志名称", "Translator"), level=等级,
                                pathname="", lineno=0, msg=本地化文本, args=(), exc_info=None)
        格式化器 = logging.Formatter(fmt='[%(asctime)s] [%(levelname)s]: %(message)s',
                                    datefmt='%H:%M:%S')
        return 格式化器.format(记录)
    def 读取日志(Self):
        Self.刷新日志()   # 先落盘再读, 否则短任务读到的永远是空的或残缺的
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
    def 刷新日志(Self):
        """把还没落盘的日志立刻刷到磁盘, 保证紧接着的 读取日志 能看到刚写的行。

        链路是 写入日志 -> (QueueHandler) 队列 -> QueueListener 线程 -> 缓冲 FileHandler -> 磁盘,
        所以要比 读取日志 领先两步:
          1) 等队列排空, 让监听线程把记录交给处理器;
          2) 刷新处理器的内存缓冲(LOGS_FLUSH_INTERVAL 默认 3 秒, 短任务根本等不到)。
        队列排空靠连续读两遍队列长度确认(单次空可能是监听线程正在取件的瞬间)。"""
        监听器 = getattr(Self, "_队列监听器", None)
        if 监听器 is None:
            return
        for _ in range(4):
            空次数 = 0
            for _ in range(50):
                try:
                    剩余 = Self._日志队列.qsize()
                except Exception:
                    剩余 = 0
                if 剩余 == 0:
                    空次数 += 1
                    if 空次数 >= 2:
                        break
                else:
                    空次数 = 0
                time.sleep(0.01)
            for 处理器 in getattr(监听器, "handlers", ()):
                执行刷新 = getattr(处理器, "_执行刷新", None)
                if 执行刷新 is None:
                    continue
                try:
                    with 处理器._lock:
                        执行刷新()
                except Exception:
                    pass
            try:
                if Self._日志队列.qsize() == 0:
                    break
            except Exception:
                break
    def 关闭(Self):
        监听器 = getattr(Self, "_队列监听器", None)
        if 监听器 is not None:
            try:
                监听器.stop()
            except Exception:
                pass
            for 处理器 in getattr(监听器, "handlers", ()):
                try:
                    处理器.close()
                except Exception:
                    pass
            Self._队列监听器 = None
        if getattr(Self, "日志管理器", None) is not None:
            for 处理器 in list(Self.日志管理器.handlers):
                Self.日志管理器.removeHandler(处理器)
                try:
                    处理器.close()
                except Exception:
                    pass
        try:
            logging.Logger.manager.loggerDict.pop(getattr(Self, "日志名称", None), None)
        except Exception:
            pass
    def __del__(Self):
        try:
            Self.关闭()
        except Exception:
            pass