from TranslatorLib import (aiohttp, asyncio, SimpleNamespace, threading, defaultdict, time, datetime, token_calibrator,
                           Config)

class Network: # 当前版本(R1.6 B.4)工作内容: VibeCoding内容重新组合 # 内容: 网络连接 请求管理
    def __init__(Self, App: Config):
        Self.Config = App.Config
        Self.Module = App.Module
        Self.CacheTokenCalibrator = App.CacheTokenCalibrator
        Self.日志 = App.日志
        Self.层级列表 = App.Manager.get()
        Self.ˀꜝ锁锁ꜝˀ = SimpleNamespace()
        Self.ˀꜝ锁锁ꜝˀ.异步会话 = threading.Lock()
        Self.ˀꜝ锁锁ꜝˀ.翻译层级并发 = threading.Lock()
        Self.ˀꜝ锁锁ꜝˀ.嵌入层级并发 = threading.Lock()
        Self.ˀꜝ锁锁ꜝˀ.重排层级并发 = threading.Lock()
        Self.ˀꜝ缓存ꜝˀ = SimpleNamespace()
        Self.ˀꜝ缓存ꜝˀ.异步会话 = {}
        Self.ˀꜝ缓存ꜝˀ.翻译层级并发 = defaultdict(lambda: [0, [], [], []]) # {"层级ID": [工作活跃数, RPM, TPM, 工作ID]}
        Self.ˀꜝ缓存ꜝˀ.嵌入层级并发 = defaultdict(lambda: [0, [], []]) # {"层级ID": [工作活跃数, RPM(无效果), 工作ID]}
        Self.ˀꜝ缓存ꜝˀ.重排层级并发 = defaultdict(lambda: [0, [], []]) # {"层级ID": [工作活跃数, RPM(无效果), 工作ID]}
        Self.停止事件 = asyncio.Event()
        Self.定时任务 = {}
        Self.嵌入层级 = {
            "url": Self.Config.EMB_API_URL,
            "key": Self.Config.EMB_API_KEY,
            "model": Self.Config.EMB_MODEL,
            "api_kwargs": Self.Config.EMB_API_KWARGS,
            "timeout": Self.Config.EMB_TIMEOUT,
            "conn_timeout": Self.Config.EMB_CONN_TIMEOUT,
            "max_workers": Self.Config.EMB_MAX_WORKERS,
            "max_retry": Self.Config.EMB_MAX_RETRY,
            "ttl_dns_cache": Self.Config.EMB_TTL_DNS_CACHE,
            "conn_reuse": Self.Config.EMB_CONN_REUSE,
            "retry_coef": Self.Config.EMB_RETRY_COEF,
            "retry_time": Self.Config.EMB_RETRY_TIME,
            "keepalive_timeout": Self.Config.EMB_KEEPALIVE_TIMEOUT,
        }
        Self.重排层级 = {
            "url": Self.Config.RERANKER_API_URL,
            "key": Self.Config.RERANKER_API_KEY,
            "model": Self.Config.RERANKER_MODEL,
            "api_kwargs": Self.Config.RERANKER_API_KWARGS,
            "timeout": Self.Config.RERANKER_TIMEOUT,
            "conn_timeout": Self.Config.RERANKER_CONN_TIMEOUT,
            "max_workers": Self.Config.RERANKER_MAX_WORKERS,
            "max_retry": Self.Config.RERANKER_MAX_RETRY,
            "ttl_dns_cache": Self.Config.RERANKER_TTL_DNS_CACHE,
            "conn_reuse": Self.Config.RERANKER_CONN_REUSE,
            "retry_coef": Self.Config.RERANKER_RETRY_COEF,
            "retry_time": Self.Config.RERANKER_RETRY_TIME,
            "keepalive_timeout": Self.Config.RERANKER_KEEPALIVE_TIMEOUT,
        }
#====================================================================================================↓限速与超时↓====================================================================================================#
    def 限速滑动窗口抽象(Self, 时间戳列表, Token数列表):
        while 时间戳列表 and 时间戳列表[0] <= time.monotonic() - 60:
            时间戳列表.pop(0)
            Token数列表.pop(0)
            
    def TranslatorRPM(Self, 层级):
        if 层级["rpm"] <= 0: return True # 不限速直接回家
        Self.限速滑动窗口抽象(Self.缓存.翻译层级并发[层级["id"]][1], Self.缓存.翻译层级并发[层级["id"]][2])
        时间戳列表 = Self.缓存.翻译层级并发[层级["id"]][1]
        return len(时间戳列表) < 层级["rpm"]
    def TranslatorTPM(Self, 层级):
        if 层级["tpm"] <= 0: return True # 不限速直接回家
        Self.限速滑动窗口抽象(Self.缓存.翻译层级并发[层级["id"]][1], Self.缓存.翻译层级并发[层级["id"]][2])
        Token数列表 = Self.缓存.翻译层级并发[层级["id"]][2]
        return sum(Token数列表) < 层级["tpm"]
    def 任务滑动窗口抽象(Self, 并发列表, 超时, 日志模型名):
        if len(并发列表[1]) == 0:
            return
        过期任务索引 = []
        for index, timestamp in enumerate(并发列表[1]):
            if (time.monotonic() - timestamp) > 超时:
                过期任务索引.append(index)
        if 过期任务索引:
            for index in reversed(过期任务索引):
                并发列表[0] -= 1
                并发列表[1].pop(index)
                if len(并发列表) > 2: 并发列表[2].pop(index)
                if len(并发列表) > 3: 并发列表[3].pop(index)
            Self.日志("log.network.tier.cleanup.info", info_level=1, model=日志模型名, count=len(过期任务索引))
    def 翻译任务滑动窗口(Self):
        for index in Self.层级列表:
            Self.任务滑动窗口抽象(Self.ˀꜝ缓存ꜝˀ.翻译层级并发[index["id"]], index["timeout"], "LLM")

    def 嵌入任务滑动窗口(Self):
        Self.任务滑动窗口抽象(Self.ˀꜝ缓存ꜝˀ.嵌入层级并发[Self.嵌入层级["model"]], Self.嵌入层级["timeout"], "EMB")

    def 重排任务滑动窗口(Self):
        Self.任务滑动窗口抽象(Self.ˀꜝ缓存ꜝˀ.重排层级并发[Self.重排层级["model"]], Self.重排层级["timeout"], "RERANKER")

#====================================================================================================↑限速与超时↑====================================================================================================#
#====================================================================================================↓公开方法↓====================================================================================================#
    def get_llm(Self, 总条目数: int, 上下文: any, 总结=False):
        任务ID = Self.Module.uuid()
        
        # ↓先清理所有已激活层级中超过超时时间的任务，防止紫砂的任务占用并发槽位
        with Self.ˀꜝ锁锁ꜝˀ.翻译层级并发: # 需要编辑 Self.ˀꜝ缓存ꜝˀ.翻译层级并发
            Self.翻译任务滑动窗口()
            
        # ↓获取当前时间，计算当前分钟数用于时间段过滤
        当前时间 = datetime.datetime.now()
        当前分钟数 = 当前时间.hour * 60 + 当前时间.minute

        # ↓筛选有效层级：满足最小条目数且在活跃时间段内
        已激活 = []
        for 层级 in Self.层级列表:
            # 格式: 层级 = {"min_count": 100, "active_time_start": "08:00", "active_time_end": "22:00", "weight": 1.0, ...}
            if 总条目数 < 层级["min_count"]:
                continue
            if ("translator" if 总结 else "summary") == 层级["mode"].lower():
                continue

            # ↓检查活跃时间段是否在当前时间内
            起始时间字符串 = 层级.get("active_time_start", "")
            结束时间字符串 = 层级.get("active_time_end", "")
            if 起始时间字符串 and 结束时间字符串:
                try:
                    起始小时, 起始分钟 = map(int, 起始时间字符串.split(":"))
                    结束小时, 结束分钟 = map(int, 结束时间字符串.split(":"))
                    起始分钟数 = 起始小时 * 60 + 起始分钟
                    结束分钟数 = 结束小时 * 60 + 结束分钟
                    # 处理正常时间范围 (如 08:00 ~ 22:00)
                    if 起始分钟数 <= 结束分钟数:
                        if not (起始分钟数 <= 当前分钟数 <= 结束分钟数):
                            continue
                    # 处理跨午夜时间范围 (如 22:00 ~ 08:00)
                    else:
                        if not (当前分钟数 >= 起始分钟数 or 当前分钟数 <= 结束分钟数):
                            continue
                except ValueError:
                    pass
            已激活.append(层级)
            
        if not 已激活:
            # ↓没有的化选择min_count最小的层级 Config内排序过了
            选中层级 = Self.层级列表[0]
            Self.日志("log.network.no.tier.warn", info_level=1, model=选中层级["model"])
        
        with Self.ˀꜝ锁锁ꜝˀ.翻译层级并发: # 需要编辑 Self.ˀꜝ缓存ꜝˀ.翻译层级并发
            # ↓选中层级 按照并发数依次选中
            for index in 已激活: # 发扬index艺术
                if (Self.ˀꜝ缓存ꜝˀ.翻译层级并发[index["id"]][0] < index["max_workers"] # 看一下该层级活跃数满了没
                    and Self.TranslatorRPM(index)   # 一分钟内请求数限制
                    and Self.TranslatorTPM(index)): # 一分钟内Token数限制
                    if index["tpm_mode"] == "TokenCalibrator":
                        Token数 = Self.CacheTokenCalibrator.估算Token(index["model"], str(上下文))
                    elif index["tpm_mode"] == "Max":
                        Token数 = len(str(上下文))
                    
                    选中层级 = index
                    Self.ˀꜝ缓存ꜝˀ.翻译层级并发[index["id"]][0] += 1 # 活跃数加1
                    Self.ˀꜝ缓存ꜝˀ.翻译层级并发[index["id"]][1].append(time.monotonic()) # PRM位置加入一个时的间
                    Self.ˀꜝ缓存ꜝˀ.翻译层级并发[index["id"]][2].append(Token数) # TPM位置加入一个Token数
                    Self.ˀꜝ缓存ꜝˀ.翻译层级并发[index["id"]][3].append(任务ID) # 加入任务ID
                    break # 选中了直接滚 因为break所以不会执行else
            else:
                return (None, None, None) # 没选中跳出翻译函数
        
        return (Self.获取异步会话(选中层级), 选中层级, 任务ID)
    
    def get_emb(Self):
        任务ID = Self.Module.uuid()
        
        with Self.ˀꜝ锁锁ꜝˀ.嵌入层级并发: # 需要编辑 Self.ˀꜝ缓存ꜝˀ.嵌入层级
            if Self.ˀꜝ缓存ꜝˀ.嵌入层级并发[Self.嵌入层级["model"]][0] < Self.嵌入层级["max_workers"]:
                Self.ˀꜝ缓存ꜝˀ.嵌入层级并发[Self.嵌入层级["model"]][0] += 1 # 活跃数加1
                Self.ˀꜝ缓存ꜝˀ.嵌入层级并发[Self.嵌入层级["model"]][1].append(time.monotonic())
                Self.ˀꜝ缓存ꜝˀ.嵌入层级并发[Self.嵌入层级["model"]][2].append(任务ID) # 加入任务ID
            else:
                return (None, None, None) # 没选中跳出翻译函数
        
        return (Self.获取异步会话(Self.嵌入层级), Self.嵌入层级, 任务ID)
    def reranker_emb(Self):
        任务ID = Self.Module.uuid()
        
        with Self.ˀꜝ锁锁ꜝˀ.重排层级并发: # 需要编辑 Self.ˀꜝ缓存ꜝˀ.重排层级
            if Self.ˀꜝ缓存ꜝˀ.重排层级并发[Self.重排层级["model"]][0] < Self.嵌入层级["max_workers"]:
                Self.ˀꜝ缓存ꜝˀ.重排层级并发[Self.重排层级["model"]][0] += 1 # 活跃数加1
                Self.ˀꜝ缓存ꜝˀ.重排层级并发[Self.重排层级["model"]][1].append(time.monotonic())
                Self.ˀꜝ缓存ꜝˀ.重排层级并发[Self.重排层级["model"]][2].append(任务ID) # 加入任务ID
            else:
                return (None, None, None) # 没选中跳出翻译函数
        
        return (Self.获取异步会话(Self.重排层级), Self.重排层级, 任务ID)
    def close抽象(Self, 并发字典, 锁, 任务ID, id索引, 删除列表):
        with 锁:
            for 层级id, 缓存数据 in list(并发字典.items()):
                if 任务ID not in 缓存数据[id索引]:
                    continue
                索引 = 缓存数据[id索引].index(任务ID)
                缓存数据[0] -= 1 # 活跃计数减1
                for index in 删除列表:
                    缓存数据[index].pop(索引) # 移除对应位置的其他数据
                break   # 找到并处理后立即退出，避免重复操作
    def close_llm(Self, 任务ID):
        Self.close抽象(Self.ˀꜝ缓存ꜝˀ.翻译层级并发, Self.ˀꜝ锁锁ꜝˀ.翻译层级并发, 任务ID, 3, [1, 2, 3])
    def close_emb(Self, 任务ID):
        Self.close抽象(Self.ˀꜝ缓存ꜝˀ.嵌入层级并发, Self.ˀꜝ锁锁ꜝˀ.嵌入层级并发, 任务ID, 2, [1, 2])
    def close_reranker(Self, 任务ID):
        Self.close抽象(Self.ˀꜝ缓存ꜝˀ.重排层级并发, Self.ˀꜝ锁锁ꜝˀ.重排层级并发, 任务ID, 2, [1, 2])
#====================================================================================================↑公开方法↑====================================================================================================#
#====================================================================================================↓会话自动管理↓====================================================================================================#
    def 获取当前事件循环ID(Self):
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return None
    def 获取异步会话(Self, 层级) -> aiohttp.ClientSession:
        循环id = Self.获取当前事件循环ID()
        缓存键 = (层级["url"], 层级["key"], 层级["model"])
        with Self.ˀꜝ锁锁ꜝˀ.异步会话:
            本循环会话 = Self.ˀꜝ缓存ꜝˀ.异步会话.get(循环id)
            if 本循环会话 is not None:
                已有会话 = 本循环会话.get(缓存键)
                if 已有会话 is not None and not 已有会话.closed:
                    return 已有会话
            超时配置 = aiohttp.ClientTimeout(
                total=层级["timeout"],
                connect=层级["conn_timeout"],
                sock_read=层级["timeout"],
                sock_connect=层级["conn_timeout"],
            )
            安全并发 = max(1, int(层级["max_workers"]))
            连接配置 = aiohttp.TCPConnector(
                limit=安全并发,
                limit_per_host=安全并发,
                ttl_dns_cache=层级["ttl_dns_cache"],
                force_close=层级["conn_reuse"],
                enable_cleanup_closed=True,
                keepalive_timeout=层级["keepalive_timeout"],
            )
            请求头 = {
                "Authorization": f"Bearer {层级["key"]}",
                "Content-Type": "application/json",
            }
            会话 = aiohttp.ClientSession(
                timeout=超时配置,
                connector=连接配置,
                headers=请求头,
            )
            Self.ˀꜝ缓存ꜝˀ.异步会话.setdefault(循环id, {})[缓存键] = 会话
            return 会话
    async def 关闭异步会话(Self, url=None, key=None, model=None):
        缓存键 = (url, key, model)
        with Self.ˀꜝ锁锁ꜝˀ.异步会话:
            for 循环id, 层级会话 in list(Self.ˀꜝ缓存ꜝˀ.异步会话.items()):
                if 缓存键 in 层级会话:
                    会话 = 层级会话[缓存键]
                    if not 会话.closed:
                        await 会话.close()
                        del 层级会话[缓存键]
                        
    async def 自动关闭会话(Self):
        while not Self.停止事件.is_set(): # 上班到下班
            for 层级id, 缓存数据 in list(Self.ˀꜝ缓存ꜝˀ.翻译层级并发.items()):
                if 缓存数据[0] == 0: # 没有活跃任务
                    层级 = next((t for t in Self.层级列表 if t["id"] == 层级id), None)
                    if 层级:
                        await Self.关闭异步会话(层级["url"], 层级["key"], 层级["model"])
            for 层级id, 缓存数据 in list(Self.ˀꜝ缓存ꜝˀ.嵌入层级并发.items()):
                if 缓存数据[0] == 0: # 没有活跃任务
                    await Self.关闭异步会话(Self.嵌入层级["url"], Self.嵌入层级["key"], Self.嵌入层级["model"])
            for 层级id, 缓存数据 in list(Self.ˀꜝ缓存ꜝˀ.重排层级并发.items()):
                if 缓存数据[0] == 0: # 没有活跃任务
                    await Self.关闭异步会话(Self.重排层级["url"], Self.重排层级["key"], Self.重排层级["model"])
            try:
                await asyncio.wait_for(Self.停止事件.wait(), timeout=Self.Config.SESSION_CLEAN_INTERVAL)
            except asyncio.TimeoutError:
                pass
    async def __aenter__(Self): # 启动自动调用
        Self.定时任务["清理无活跃会话"] = asyncio.create_task(Self.自动关闭会话())
        return Self
    async def __aexit__(Self, exc_type, exc_val, exc_tb): # 关闭自动调用
        Self.停止事件.set()
        任务 = list(Self.定时任务.values())
        for index in 任务:
            if not index.done():
                index.cancel()
        if 任务:
            await asyncio.gather(*任务, return_exceptions=True)
#====================================================================================================↑会话自动管理↑====================================================================================================#