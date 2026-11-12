from TranslatorLib import asyncio, threading
from concurrent.futures import CancelledError as FutureCancelled

class TranslatorLoop:
    def __init__(Self):
        Self.锁      = threading.Lock()
        Self._循环   = None
        Self._线程   = None
        Self.就绪    = threading.Event()
        Self.归属线程 = None
        Self.启动超时 = 5.0
        Self.语言     = None
        Self.日志     = None

    def 绑定语言(Self, 语言=None, 日志=None):
        if 语言 is not None: Self.语言 = 语言
        if 日志 is not None: Self.日志 = 日志
        return Self

    def 文本(Self, 键, **参数):
        if Self.语言 is None:
            return 键
        try:
            return Self.语言(键, **参数)
        except Exception:
            return 键

    def 报错(Self, 键, 等级=2, **参数):
        文本 = Self.文本(键, **参数)
        if Self.日志 is not None:
            try: Self.日志(键, info_level=等级, **参数)
            except Exception: pass
        return RuntimeError(文本)

    @property
    def 循环(Self):
        return Self.启动()

    def 启动(Self):
        with Self.锁:
            if Self._循环 is not None:
                return Self._循环
            Self.就绪.clear()
            Self._线程 = threading.Thread(target=Self.循环线程, name="TranslatorLoop", daemon=True)
            Self._线程.start()
            Self.就绪.wait(Self.启动超时)
            if Self._循环 is None:
                raise Self.报错("log.core.loop.start.error", 超时=Self.启动超时)
            return Self._循环

    def 循环线程(Self):
        循环 = asyncio.new_event_loop()
        Self._循环 = 循环
        asyncio.set_event_loop(循环)
        Self.归属线程 = threading.get_ident()
        Self.就绪.set()
        try:
            循环.run_forever()
        finally:
            try:
                循环.run_until_complete(循环.shutdown_asyncgens())
            finally:
                try:
                    循环.close()
                finally:
                    asyncio.set_event_loop(None)

    def 绑定当前循环(Self):
        with Self.锁:
            Self._循环 = asyncio.get_running_loop()
            Self.归属线程 = threading.get_ident()
            Self.就绪.set()
            return Self._循环

    def 已在服务循环内(Self):
        try:
            return asyncio.get_running_loop() is Self._循环
        except RuntimeError:
            return False

    def 运行(Self, 协程, 超时=None):
        if Self.已在服务循环内():
            Self.丢弃(协程)
            raise Self.报错("log.core.loop.sync.wait.error", 协程=type(协程).__name__, 循环线程=Self.归属线程, 当前线程=threading.get_ident())
        if 超时 is not None:
            协程 = Self.限时(协程, 超时)
        try:
            任务 = asyncio.run_coroutine_threadsafe(协程, Self.循环)
        except Exception:
            Self.丢弃(协程)
            raise
        try:
            return 任务.result()
        except FutureCancelled:
            raise Self.报错("log.core.loop.wait.cancelled", 线程=Self.归属线程) from None

    @staticmethod
    async def 限时(协程, 超时):
        try:
            return await asyncio.wait_for(协程, 超时)
        except asyncio.TimeoutError:
            raise TimeoutError() from None

    @staticmethod
    def 丢弃(协程):
        关闭 = getattr(协程, "close", None)
        if callable(关闭):
            try: 关闭()
            except Exception: pass

    async def 异步运行(Self, 协程):
        if Self.已在服务循环内():
            return await 协程
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(协程, Self.循环))

    def 关闭(Self):
        with Self.锁:
            if Self._循环 is None:
                return
            循环, 线程 = Self._循环, Self._线程
            Self._循环 = None
            Self._线程 = None
        循环.call_soon_threadsafe(循环.stop)
        if 线程 is not None and 线程 is not threading.current_thread():
            线程.join(5.0)

Loop = TranslatorLoop()
