from TranslatorLib import (HARDWARE_INFO, np, threading, zipfile, json, ast, eb, re, partial, defaultdict, Path, ThreadPoolExecutor, queue, Callable, GPU_ACC, time, uuid, bisect, SimpleNamespace, Any, random, heapq, datetime,
                           RuntimeConfig, Quantization, Locale, Module, File, TranslatorPersistence, Log, Builder, NOT_IMPORT)

class 翻译上下文管理器:
    __slots__ = ('数据字典', '已翻译索引列表', '已翻译键集合', '索引到键列表', '线程锁', '键索引缓存', 'switch')
    def __init__(Self, 初始字典=None):
        Self.数据字典 = {}
        Self.已翻译索引列表 = []
        Self.已翻译键集合 = set()
        Self.索引到键列表 = []
        Self.线程锁 = threading.Lock()
        Self.键索引缓存 = {}
        Self.switch = True
        
        if 初始字典:
            for 索引, (键, 值) in enumerate(初始字典.items()):
                键 = str(键)
                Self.数据字典[键] = 值
                Self.键索引缓存[键] = 索引
                Self.索引到键列表.append(键)
                if len(值) > 1:
                    译文 = 值[1]
                    if 译文 is not None and 译文 != "":
                        Self.已翻译索引列表.append(索引)
                        Self.已翻译键集合.add(键)

    def add(Self, 键, 译文):
        with Self.线程锁:
            键 = str(键)
            if 键 in Self.数据字典:
                Self.数据字典[键][1] = 译文
                if 译文 and 译文 != "" and 键 not in Self.已翻译键集合:
                    Self.已翻译键集合.add(键)
                    bisect.insort(Self.已翻译索引列表, Self.键索引缓存[键])
            else:
                Self.数据字典[键] = [译文]
                索引 = len(Self.索引到键列表)
                Self.键索引缓存[键] = 索引
                Self.索引到键列表.append(键)
    def get(Self, 当前键, 数量):
        if not Self.switch: return []
        with Self.线程锁:
            当前键str = str(当前键)
            if 当前键str not in Self.数据字典:
                return []
            当前全局索引 = Self.键索引缓存[当前键str]
            pos = bisect.bisect_left(Self.已翻译索引列表, 当前全局索引)
            start = max(0, pos - 数量)
            选取索引列表 = Self.已翻译索引列表[start:pos]
        结果列表 = []
        for 索引 in 选取索引列表:
            遍历键 = Self.索引到键列表[索引]
            结果列表.append({"role": "assistant", "content": Self.数据字典[遍历键][1]})
            结果列表.append({"role": "user", "content": Self.数据字典[遍历键][0]})
        return 结果列表[::-1]
    def clean(Self, 消息列表: list):
        if not 消息列表:
            return []
        结果列表 = []
        已见集合 = set()
        索引 = 0
        总长度 = len(消息列表)
        while 索引 < 总长度:
            当前消息 = 消息列表[索引]
            if 当前消息.get("role") == "system":
                结果列表.append(当前消息)
                索引 += 1
                continue
            用户消息 = 当前消息
            机器消息 = 消息列表[索引 + 1]
            对话特征 = (
                json.dumps(用户消息, sort_keys=True, ensure_ascii=False),
                json.dumps(机器消息, sort_keys=True, ensure_ascii=False)
            )
            if 对话特征 not in 已见集合:
                已见集合.add(对话特征)
                结果列表.append(用户消息)
                结果列表.append(机器消息)
            索引 += 2
        return 结果列表

class 请求池:
    """串行RAG请求池 - 串行执行RAG检索后提交到池中并发处理API请求"""
    __slots__ = ('队列', '工作线程列表', '是否运行', '日志', '翻译方法', '上下文管理器',
                 '返回列表', '返回锁', '进度条', '进度锁', '配置', '最大并发', 'tqdm',
                 '层级列表', '层级并发状态', '层级堆', '状态锁', '状态条件')
    def __init__(Self, 最大并发: int, 日志: Callable, 翻译方法: Callable, 上下文管理器, 配置, tqdm):
        Self.队列 = queue.Queue()
        Self.工作线程列表 = []
        Self.是否运行 = False
        Self.日志 = 日志
        Self.翻译方法 = 翻译方法
        Self.上下文管理器 = 上下文管理器
        Self.配置 = 配置
        Self.返回列表 = []
        Self.返回锁 = threading.Lock()
        Self.进度条 = None
        Self.进度锁 = threading.Lock()
        Self.最大并发 = 最大并发
        Self.tqdm = tqdm
        Self.层级列表 = []
        Self.层级并发状态 = {}
        Self.层级堆 = []
        Self.状态锁 = threading.Lock()
        Self.状态条件 = threading.Condition(Self.状态锁)
    def 注册层级(Self, 层级列表: list):
        Self.层级列表 = 层级列表
        Self.层级并发状态 = {}
        Self.层级堆 = []
        for cfg in 层级列表:
            tier_id = cfg.get("id", id(cfg)) if cfg else -1
            max_conn = cfg.get("max_connections", cfg.get("max_workers", Self.配置.LLM_MAX_WORKERS)) if cfg else Self.配置.LLM_MAX_WORKERS
            weight = cfg.get("weight", 1.0) if cfg else 1.0
            Self.层级并发状态[tier_id] = {"cfg": cfg, "current": 0, "max": max_conn, "weight": weight}
            if max_conn > 0:
                heapq.heappush(Self.层级堆, (-weight, 0, tier_id))
    def _选择层级(Self):
        with Self.状态条件:
            while True:
                while Self.层级堆:
                    neg_w, curr, tid = Self.层级堆[0]
                    info = Self.层级并发状态[tid]
                    if curr == info["current"] and info["current"] < info["max"]:
                        heapq.heappop(Self.层级堆)
                        info["current"] += 1
                        if info["current"] < info["max"]:
                            heapq.heappush(Self.层级堆, (-info["weight"], info["current"], tid))
                        return info["cfg"]
                    else:
                        heapq.heappop(Self.层级堆)
                Self.状态条件.wait()
    def _释放层级(Self, cfg):
        if cfg is None: return
        with Self.状态条件:
            tier_id = cfg.get("id", id(cfg))
            info = Self.层级并发状态.get(tier_id)
            if info:
                info["current"] -= 1
                if info["current"] < info["max"]:
                    heapq.heappush(Self.层级堆, (-info["weight"], info["current"], tier_id))
                Self.状态条件.notify_all()
    def 启动(Self, 总条目数: int = 0):
        Self.是否运行 = True
        if 总条目数 > 0:
            Self.进度条 = Self.tqdm(total=总条目数, desc="tqdm.translator.generate")
        for _ in range(Self.最大并发):
            线程 = threading.Thread(target=Self.工作线程循环, daemon=True)
            线程.start()
            Self.工作线程列表.append(线程)
    def 停止(Self):
        Self.是否运行 = False
        for _ in Self.工作线程列表:
            Self.队列.put(None)
        for 线程 in Self.工作线程列表:
            线程.join(timeout=2)
        if Self.进度条:
            Self.进度条.close()
    def 提交(Self, 翻译内容列表: list, 其他内容列表: list, 层级配置):
        Self.队列.put((翻译内容列表, 其他内容列表, 层级配置))
    def 工作线程循环(Self):
        while Self.是否运行:
            try:
                任务 = Self.队列.get(timeout=0.5)
            except queue.Empty:
                continue
            if 任务 is None:
                Self.队列.task_done()
                break
            翻译内容列表, 其他内容列表, 层级配置 = 任务
            try:
                if 层级配置 is None and Self.层级列表:
                    层级配置 = Self._选择层级()
                结果 = Self.翻译方法(翻译内容=翻译内容列表, 其他内容=其他内容列表, 层级配置=层级配置, context=Self.上下文管理器)
                with Self.返回锁:
                    Self.返回列表.extend(结果)
                with Self.进度锁:
                    if Self.进度条:
                        Self.进度条.update(len(结果))
            except Exception:
                Self.日志("log.core.request.pool.error", e=eb.format_exc(), info_level=2)
            finally:
                if 层级配置 and Self.层级列表:
                    Self._释放层级(层级配置)
                Self.队列.task_done()
    def 获取结果(Self):
        Self.队列.join()
        return Self.返回列表

class Translator:
    def __init__(Self, Config: dict = None):
        Config, Self.增量索引缓存, Self.owolib解析缓存 = Config or {}, {}, {}
        Self.Config = RuntimeConfig(**Config)
        Path(Self.Config.LOGS_FILE_PATH).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.VEC_FILE_PATH).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.PATH_CACHE).mkdir(parents=True, exist_ok=True)
        Path(Self.Config.TRANSLATOR_CACHE_PATH).mkdir(parents=True, exist_ok=True)
        Self.Quantization = Quantization(Config=Config)
        Self.Module = Module(Config=Config)
        Self.Locale = Locale(Config=Config)
        Self.File = File(Config=Config)
        Self.Lang = Self.Locale.Lang
        Self.Log = Log(Config=Config)
        Self.日志 = Self.Log.写入日志
        Self.tqdm = Self.Locale.Tqdm
        Self.Builder = Builder(Config=Config)
        if GPU_ACC: Self.日志("log.core.numpy.gpu", type=HARDWARE_INFO['type'], acc_type=HARDWARE_INFO['acc_type'], version=HARDWARE_INFO['version'], acc_version=HARDWARE_INFO['acc_version'], deviceid=HARDWARE_INFO['device_id'], count=HARDWARE_INFO["device_count"], info_level=0)
        else: Self.日志("log.core.numpy.cpu", type=HARDWARE_INFO['type'], acc_type=HARDWARE_INFO['acc_type'], version=HARDWARE_INFO['version'], acc_version=HARDWARE_INFO['acc_version'], e=HARDWARE_INFO['error'], info_level=0)
        for index in NOT_IMPORT: Self.日志("log.core.not_import", index=index, info_level=1)
        Self.线程锁 = threading.Lock()
        Self.Index = Self.Module.Index
        Self.正则表达式预编译 = SimpleNamespace()
        Self.正则表达式预编译.翻译剔除方法 = re.compile(r'^\{[^}]*\}$|^[^\w\u4e00-\u9fa5]{1,2}$')
        Self.正则表达式预编译.模型思维链剔除方法 = re.compile(r'(?s)(?:<think>.*?</think>|\[think\].*?\[/think\])\s*')
    def __enter__(Self):
        return Self
    def 生成翻译(Self, 翻译内容: list, 其他内容: str, 层级配置: dict = None, context: 翻译上下文管理器 = None):
            请求文本, 返回结果 = [], []
            消息结果, 请求结果, 附加内容 = "", "", "",
            是否重试 = False
            请求错误次数, 批量翻译错误次数 = 0, 0
            层级配置 = 层级配置 or {}
            请求地址 = 层级配置.get("url", Self.Config.LLM_API_URL)
            请求密钥 = 层级配置.get("key", Self.Config.LLM_API_KEY)
            模型名称 = 层级配置.get("model", Self.Config.LLM_MODEL)
            模型核采样 = 层级配置.get("top_p", Self.Config.LLM_TOP_P)
            模型前K个候选词 = 层级配置.get("top_k", Self.Config.LLM_TOP_K)
            模型温度 = 层级配置.get("temperature", Self.Config.LLM_TEMP)
            模型重复惩罚 = 层级配置.get("repetition_penalty", Self.Config.LLM_RP)
            模型存在惩罚 = 层级配置.get("presence_penalty", Self.Config.LLM_PP)
            模型频率惩罚 = 层级配置.get("frequency_penalty", Self.Config.LLM_FP)
            模型种子 = 层级配置.get("seed", Self.Config.LLM_SEED)
            重试次数 = 层级配置.get("max_retry", Self.Config.LLM_MAX_RETRY)
            超时时间 = (层级配置.get("conn_timeout", Self.Config.LLM_CONN_TIMEOUT), 层级配置.get("timeout", Self.Config.LLM_TIMEOUT))
            重试时间 = 层级配置.get("retry_time", Self.Config.LLM_RETRY_TIME)
            重试系数 = 层级配置.get("retry_coef", Self.Config.LLM_RETRY_COEF)
            请求额外参数 = 层级配置.get("api_kwargs", Self.Config.LLM_API_KWARGS)
            请求会话 = TranslatorPersistence.获取会话(请求地址, 请求密钥, 模型名称, 层级配置.get("max_workers", Self.Config.LLM_MAX_WORKERS), 重试系数, 重试次数)
            原始翻译内容 = 翻译内容.copy()
            翻译内容 = [翻译内容]
            原始其他内容 = 其他内容.copy()
            其他内容 = [其他内容]
            while True:
                try:
                    合并返回的请求结果 = []
                    for 文本, 其他 in zip(翻译内容, 其他内容):
                        请求文本长度 = len(文本)
                        文本, 其他 = ([文本], [其他]) if isinstance(文本, str) else (文本, 其他)
                        键提示词 = []
                        额外内容 = [[index[1][0], index[1][1]] for index in 其他]
                        额外提示词 = [f"{index1[0]}={index1[1]}" for index in 额外内容 for index1 in index[1]]
                        额外提示词 = Self.Module.列表去重(额外提示词)
                        额外提示词 = "- " + "\n- ".join(额外提示词) if 额外提示词 else ""
                        for idx in 其他:
                            所有键 = idx[1][0] if (len(idx) > 1 and len(idx[1]) > 0) else (idx[0] if len(idx) > 0 else [])
                            if not isinstance(所有键, list): 所有键 = [所有键]
                            键提示词.append("Key:" + ",".join(所有键))
                        键提示词 = "- " + "\n- ".join(键提示词) if 键提示词 else ""
                        附加内容 = f"\n{键提示词}\n{额外提示词}"
                        messages = [{"role": "system", "content": Self.Config.TRANSLATOR_SYSTEM_PROMPT[请求文本长度 == 1].format(LANGUAGE_OUTPUT=Self.Config.LANGUAGE_OUTPUT) + 附加内容}]
                        if Self.Config.LLM_CONTEXTS != False:
                            for index in 其他:
                                messages.extend(context.get(str(index[0]), Self.Config.LLM_CONTEXTS))
                        messages = context.clean(messages)
                        请求文本 = 文本[0] if 请求文本长度 == 1 else json.dumps(文本)
                        请求文本 = Self.Config.TRANSLATOR_USER_PROMPT[请求文本长度 == 1].format(text=请求文本, LANGUAGE_OUTPUT=Self.Config.LANGUAGE_OUTPUT, count=len(请求文本))
                        messages.append({"role": "user", "content": 请求文本})
                        请求结果 = 请求会话.post(url=请求地址, json={"model": 模型名称, "messages": messages, "top_p": 模型核采样, "top_k": 模型前K个候选词, "temperature": 模型温度, "seed": 模型种子, "repeat_penalty": 模型重复惩罚, "presence_penalty": 模型存在惩罚, "frequency_penalty": 模型频率惩罚, "stream": False,} | 请求额外参数, timeout=超时时间)
                        请求结果.raise_for_status()
                        请求结果 = 请求结果.json()
                        Token结果 = 请求结果.get("usage", {})
                        消息结果 = 请求结果["choices"][0]["message"]["content"]
                        Self.Config.LLM_TOKEN_USAGE += Token结果.get("total_tokens", 0)
                        Self.日志("log.core.translator.generate.request.outputs.debug", messages=翻译内容, item=消息结果, promptex=附加内容, info_level=4)
                        消息结果 = Self.正则表达式预编译.模型思维链剔除方法.sub("", 消息结果)
                        try:
                            消息结果 = json.loads(消息结果)
                            返回的请求结果 = [消息结果[0]] if 请求文本长度 == 1 else 消息结果
                        except Exception: 返回的请求结果 = [消息结果]
                        合并返回的请求结果.extend(返回的请求结果)
                    if len(合并返回的请求结果) != len(原始翻译内容): raise IndexError
                    for index in range(len(原始翻译内容)):
                        返回结果.append([原始其他内容[index][0], 原始翻译内容[index], 合并返回的请求结果[index], 原始其他内容[index][2]])
                        if Self.Config.LLM_CONTEXTS != False:
                            context.add(原始其他内容[index][0], 合并返回的请求结果[index])
                        Self.日志("log.core.translator.generate", input=原始翻译内容[index], output=合并返回的请求结果[index])
                    return 返回结果
                except IndexError:
                    批量翻译错误次数 += 1
                    if 批量翻译错误次数 >= Self.Config.TRANSLATOR_BATCH_RETRY and 是否重试 == False:
                        Self.日志("log.core.translator.generate.batch.error", info_level=1)
                        是否重试 = True
                        翻译内容 = 翻译内容[0]
                        其他内容 = 其他内容[0]
                    else:
                        Self.日志("log.core.translator.generate.batch.retry", info_level=1)
                except Exception:
                    Self.日志("log.core.translator.generate.messages.error", promptex=附加内容, messages=翻译内容, e=eb.format_exc(), info_level=1)
                    返回结果 = [[其他内容[index][0], 翻译内容[index], 翻译内容[index], 其他内容[index][2]] for index in range(len(翻译内容))]
                    请求错误次数 += 1
                    if 请求错误次数 >= 重试次数:
                        Self.日志("log.core.translator.generate.error", e=eb.format_exc(), output=请求结果, info_level=2)
                        return 返回结果
                    else:
                        Self.日志("log.core.translator.generate.retry", e=eb.format_exc(), output=请求结果, info_level=1)
                        基础等待 = (重试系数 ** (请求错误次数 - 1)) * 重试时间
                        time.sleep(基础等待 + random.uniform(0, 基础等待 * 0.3))
    def 任务分配器(Self, 总数: int):
        层级列表 = Self.Config.get_active_tiers()
        if not 层级列表:
            return []
        now = datetime.datetime.now()
        current_mins = now.hour * 60 + now.minute
        有效层级 = []
        for t in 层级列表:
            if 总数 < t.get("min_count", 0):
                continue
            start_str = t.get("active_time_start", "")
            end_str = t.get("active_time_end", "")
            if start_str and end_str:
                try:
                    sh, sm = map(int, start_str.split(":"))
                    eh, em = map(int, end_str.split(":"))
                    start_mins = sh * 60 + sm
                    end_mins = eh * 60 + em
                    if start_mins <= end_mins:
                        if not (start_mins <= current_mins <= end_mins):
                            continue
                    else:
                        if not (current_mins >= start_mins or current_mins <= end_mins):
                            continue
                except ValueError:
                    pass
            有效层级.append(t)
        if not 有效层级:
            return []
        if Self.Config.LLM_TIER_DYNAMIC:
            权重列表 = [t.get("weight", 1.0) for t in 有效层级]
            总权重 = sum(权重列表)
            return [(t, w / 总权重) for t, w in zip(有效层级, 权重列表)]
        if Self.Config.LLM_TIER_MULTI_OVERLAP:
            权重列表 = [t.get("weight", 1.0) for t in 有效层级]
            总权重 = sum(权重列表)
            return [(t, w / 总权重) for t, w in zip(有效层级, 权重列表)]
        if Self.Config.LLM_TIER_CASCADE:
            r = Self.Config.LLM_TIER_CASCADE_RATIO
            分配结果 = []
            剩余比例 = 1.0
            for i in range(len(有效层级)-1, 0, -1):
                层级 = 有效层级[i]
                if i == len(有效层级)-1:
                    比例 = r
                else:
                    比例 = 剩余比例 * r
                分配结果.append((层级, 比例))
                剩余比例 -= 比例
            分配结果.append((有效层级[0], 剩余比例))
            分配结果.reverse()
            return 分配结果
        选定层级 = 有效层级[-1]
        if not Self.Config.LLM_TIER_OVERLAP:
            return [(选定层级, 1.0)]
        索引值 = 层级列表.index(选定层级)
        if 索引值 == 0:
            return [(选定层级, 1.0)]
        上一级 = 层级列表[索引值 - 1]
        低比例, 高比例 = Self.Config.LLM_TIER_OVERLAP_RATIO
        return [(上一级, 低比例), (选定层级, 高比例)]

    def 翻译语言列表(Self, texts: list, 参考列表: list=None, 使用模型: list=None, 索引ID: list=uuid.uuid4().hex, 获取参考文本: bool = False) -> list:
        输入列表, 返回列表, 命中缓存, 翻译缓存输入, 返回请求内容, 返回其他内容, 完整返回列表, 去翻译列表, 翻译参考列表, 未翻译列表, 任务列表 = [], [], [], [], [], [], [], [], [], [], []
        参考字典 = {}
        if 使用模型 == None: 使用模型 = []
        if texts == []: return []
        try: QuestsMode = True if isinstance(texts[0][0], list) else False
        except: QuestsMode = False
        texts = [texts[index] for index in range(len(texts)) if not bool(Self.正则表达式预编译.翻译剔除方法.match(texts[index][1]))] if QuestsMode else texts
        texts = [index for index in texts if not f"{index[0]}" == f"{index[1]}"]
        
        if 参考列表 != None:
            for item in 参考列表:
                try:
                    参考字典[str(item[0])] = item[1]
                except Exception:
                    Self.日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
            for index in texts:
                key = str(index[0])
                if key in 参考字典:
                    翻译参考列表.append([index[1], key, 参考字典[key]])
        参考向量索引, 参考键文本, 参考原文文本, 参考译文文本 = TranslatorPersistence.增量索引(Self, 翻译参考列表, 索引ID)
        if 参考键文本:
            参考字典 = dict(zip(参考键文本, 参考译文文本))
            for index in texts:
                if index[0] in 参考字典:
                    去翻译列表.append([index[0], index[1], 参考字典[index[0]], index[2]])
                else:
                    未翻译列表.append(index)
        else:
            未翻译列表 = texts.copy()
            
        输入复制 = texts.copy()
        if Self.Config.TRANSLATOR_CACHE_READ:
            翻译缓存 = Self.Module.翻译缓存()
            原始长度 = len(未翻译列表)
            待翻译 = []
            for item in Self.tqdm(未翻译列表, desc="tqdm.translator.cache.use"):
                if item[1] in 翻译缓存:
                    命中缓存.append([item[0], item[1], 翻译缓存[item[1]], item[2]])
                else:
                    待翻译.append(item)
            未翻译列表[:] = 待翻译
            成功缓存 = len(命中缓存)
            命中率 = (成功缓存 / 原始长度) if 原始长度 > 0 else 0.0
            Self.日志("log.core.translator.cache.hit", hit=f"{命中率:.4%}", info_level=0)
        try:
            if 未翻译列表:
                for index in 未翻译列表:
                    try:
                        输入列表.append([index[1], index[0], index[2]])
                    except Exception:
                        Self.日志("log.core.parsing.parameters.error", e=eb.format_exc(), index=index, info_level=2)
                        pass
                向量文件, 文本文件 = TranslatorPersistence.参考词预处理(Self=Self)
                索引聚合缓存 = {}
                单词结果暂存 = {}
                if 文本文件:
                    if Self.Config.INDEX_TEXT_K + Self.Config.INDEX_WORD_K + Self.Config.INDEX_LANG_K != 0 and Self.Config.INDEX_TEXT_K != 0:
                        向量索引 = TranslatorPersistence.缓存索引(Self=Self, 向量文件=向量文件, 文本文件=文本文件)
                        if Self.Config.INDEX_TEXT_K != 0:
                            Self.日志("log.core.index.search.start", info_level=0)
                            输入列表 = Self.Builder.并行生成向量(输入列表)
                            向量列表 = np.asarray(输入列表[0], dtype=np.float32)
                            向量列表 = 向量列表.get() if GPU_ACC else 向量列表
                            for _ in Self.tqdm(range(1), desc="tqdm.index.search"):
                                索引结果矩阵 = 向量索引.search(向量列表, Self.Config.INDEX_TEXT_K)[1]
                            if Self.Config.INDEX_LANG_K != 0 and 翻译参考列表:
                                for _ in Self.tqdm(range(1), desc="tqdm.index.search"):
                                    参考索引结果矩阵 = 参考向量索引.search(向量列表, Self.Config.INDEX_TEXT_K)[1]
                            for index in range(len(向量列表)):
                                键 = 输入列表[1][1][index]
                                文件路径 = 输入列表[1][2][index]
                                其他内容 = [键, ["None", []], 文件路径] if QuestsMode else [键, [键, []], 文件路径]
                                for index2 in 索引结果矩阵[index]:
                                    if index2 < 0: continue
                                    其他内容[1][1].append(文本文件[index2])
                                if Self.Config.INDEX_LANG_K != 0 and 翻译参考列表:
                                    for index2 in 参考索引结果矩阵[index]:
                                        if index2 < 0: continue
                                        其他内容[1][1].append([参考原文文本[index2], 参考译文文本[index2]])
                                键 = str(键)
                                if 键 not in 索引聚合缓存:
                                    索引聚合缓存[键] = {"请求": 输入列表[1][0][index], "数据": 其他内容}
                                else:
                                    索引聚合缓存[键]["数据"][1][1].extend(其他内容[1][1])
                            Self.日志("log.core.index.search.end", info_level=0)
                        if Self.Config.INDEX_WORD_K != 0:
                            Self.日志("log.core.index.search.start", info_level=0)
                            单词列表 = []
                            for index in 未翻译列表:
                                单词列表 += index[1].split()
                            单词列表 = Self.Module.列表去重(单词列表)
                            单词列表 = [w for w in 单词列表 if len(w.strip()) > 1]
                            单词列表 = [w for w in 单词列表 if w.lower() not in {w.lower() for w in Self.Config.INDEX_QUESTS_BASIC_WORDS}]
                            单词列表 = [[index, "", ""] for index in 单词列表]
                            单词输入列表 = Self.Builder.并行生成向量(单词列表)
                            单词向量列表 = np.asarray(单词输入列表[0], dtype=np.float32)
                            单词向量列表 = 单词向量列表.get() if GPU_ACC else 单词向量列表
                            for _ in Self.tqdm(range(1), desc="tqdm.index.search"):
                                单词索引结果矩阵 = 向量索引.search(单词向量列表, Self.Config.INDEX_WORD_K)[1]
                            for index in range(len(单词向量列表)):
                                请求内容 = 单词输入列表[1][0][index]
                                其他内容 = [请求内容, ["None", []], "WordIndex"] if QuestsMode else [请求内容, [请求内容, []], "WordIndex"]
                                for index2 in 单词索引结果矩阵[index]:
                                    if index2 < 0: continue
                                    其他内容[1][1].append(文本文件[index2])
                                单词结果暂存[请求内容] = 其他内容[1][1]
                            if 单词结果暂存 and 索引聚合缓存:
                                for 缓存键, 缓存数据 in 索引聚合缓存.items():
                                    查询文本 = 缓存数据["请求"]
                                    for 分词 in 查询文本.split():
                                        if 分词 in 单词结果暂存:
                                            缓存数据["数据"][1][1].extend(单词结果暂存[分词])
                            Self.日志("log.core.index.search.end", info_level=0)
                        for 缓存项 in 索引聚合缓存.values():
                            返回请求内容.append(缓存项["请求"])
                            返回其他内容.append(缓存项["数据"])
                    else:
                        返回请求内容 = [row[1] for row in 未翻译列表]
                        返回其他内容 = [[row[0], ["None", []], row[2]] for row in 未翻译列表] if QuestsMode else [[row[0], [row[0], []], row[2]] for row in 未翻译列表]
                else:
                    返回请求内容 = [row[1] for row in 未翻译列表]
                    返回其他内容 = [[row[0], ["None", []], row[2]] for row in 未翻译列表] if QuestsMode else [[row[0], [row[0], []], row[2]] for row in 未翻译列表]
                if 获取参考文本:
                    return 返回请求内容, 返回其他内容
                处理后的请求内容 = []
                处理后的其他内容 = []
                额外列表 = defaultdict(list)
                def 深度优先搜索(组件, 当前路径):
                    if isinstance(组件, str):
                        提取记录.append((当前路径, 组件))
                        return
                    if isinstance(组件, dict):
                        for k, v in 组件.items():
                            if k == "text":
                                提取记录.append((当前路径 + [k], v))
                            elif k == "translate":
                                if Self.Module.过滤键文本(["", v]):
                                    提取记录.append((当前路径 + [k], v))
                            elif k == "extra" and isinstance(v, list):
                                for i, 子组件 in enumerate(v):
                                    深度优先搜索(子组件, 当前路径 + [k, i])
                            else:
                                深度优先搜索(v, 当前路径 + [k])
                        return
                    if isinstance(组件, list):
                        for i, 项目 in enumerate(组件):
                            深度优先搜索(项目, 当前路径 + [i])
                        return
                for index0, index1 in zip(返回请求内容, 返回其他内容):
                    try:
                        解析数据 = ast.literal_eval(index0)
                        if isinstance(解析数据, (dict, list)):
                            Self.owolib解析缓存[index0] = 解析数据
                            提取记录 = []
                            深度优先搜索(解析数据, [])
                            if 提取记录:
                                额外列表[f"{index1[0]}{index0}"] = []
                                for 路径, 文本 in 提取记录:
                                    处理后的请求内容.append(文本)
                                    其他内容值 = index1.copy()
                                    路径键 = "|".join(str(p) for p in 路径)
                                    其他内容值[0] = f"{index1[0]}{路径键}"
                                    处理后的其他内容.append(其他内容值)
                                    额外列表[f"{index1[0]}{index0}"].append((路径, 文本))
                        else:
                            处理后的请求内容.append(index0)
                            处理后的其他内容.append(index1)
                    except Exception:
                        处理后的请求内容.append(index0)
                        处理后的其他内容.append(index1)
                原始请求内容 = 处理后的请求内容
                原始其他内容 = 处理后的其他内容
                原文到索引映射 = defaultdict(list)
                原文到键列表映射 = defaultdict(list)
                for idx, 原文 in enumerate(原始请求内容):
                    标识符 = 原始其他内容[idx][0]
                    原文到索引映射[原文].append(idx)
                    原文到键列表映射[原文].append(标识符)
                唯一请求内容 = []
                唯一其他内容 = []
                for 原文, 索引列表 in 原文到索引映射.items():
                    唯一请求内容.append(原文)
                    基础条目 = 原始其他内容[索引列表[0]]
                    所有键列表 = 原文到键列表映射[原文]
                    唯一其他内容.append(基础条目 + [所有键列表])
                处理后的请求内容 = 唯一请求内容
                处理后的其他内容 = 唯一其他内容
                上下文初始化字典 = defaultdict(list)
                for index0, index1 in zip(处理后的请求内容, 处理后的其他内容):
                    上下文初始化字典[str(index1[0])] = [index0, ""]
                上下文管理器 = 翻译上下文管理器(上下文初始化字典)
                总条目数 = len(处理后的请求内容)
                分配结果 = Self.任务分配器(总条目数)
                本次新增模型 = list(dict.fromkeys(层级配置.get('model', "") for 层级配置, 比例 in 分配结果 if 比例 > 0))
                if 使用模型: 使用模型[0].extend([m for m in 本次新增模型 if m not in set(使用模型[0])])
                else: 使用模型.append(本次新增模型)
                Self.日志("log.core.translator.generate.start", item=总条目数, info_level=0)
                for 轮次 in range(Self.Config.TRANSLATOR_REFINE_ROUNDS + 1):
                    if 轮次 == 0 and Self.Config.TRANSLATOR_REFINE_ROUNDS > 0: 上下文管理器.switch = False
                    if 轮次 == 1: 上下文管理器.switch = True
                    请求池实例 = 请求池(
                        最大并发=Self.Config.LLM_MAX_WORKERS,
                        日志=Self.日志,
                        翻译方法=Self.生成翻译,
                        上下文管理器=上下文管理器,
                        配置=Self.Config,
                        tqdm=Self.tqdm
                    )
                    if getattr(Self.Config, "LLM_TIER_DYNAMIC", False):
                        请求池实例.注册层级([cfg for cfg, _ in 分配结果 if cfg is not None])
                    请求池实例.启动(总条目数=总条目数)
                    if getattr(Self.Config, "LLM_TIER_DYNAMIC", False):
                        for 步进 in range(0, 总条目数, Self.Config.TRANSLATOR_BATCH):
                            子请求 = 处理后的请求内容[步进:步进 + Self.Config.TRANSLATOR_BATCH]
                            子其他 = 处理后的其他内容[步进:步进 + Self.Config.TRANSLATOR_BATCH]
                            请求池实例.提交(子请求, 子其他, None)
                    elif Self.Config.LLM_TIER_INTERLEAVE:
                        条目计数 = []
                        剩余 = 总条目数
                        for 索引, (层级配置, 比例) in enumerate(分配结果):
                            if 索引 == len(分配结果) - 1:
                                条目计数.append(剩余)
                            else:
                                条数 = int(总条目数 * 比例)
                                条目计数.append(条数)
                                剩余 -= 条数
                        权重 = [w for w in 条目计数]
                        总权重 = sum(权重)
                        误差 = [0.0] * len(权重)
                        条目归属 = [0] * 总条目数
                        for i in range(总条目数):
                            最佳 = 0
                            for j in range(1, len(权重)):
                                if 误差[j] < 误差[最佳]:
                                    最佳 = j
                            条目归属[i] = 最佳
                            误差[最佳] += 权重[最佳] / 总权重
                        索引 = 0
                        while 索引 < 总条目数:
                            当前层 = 条目归属[索引]
                            当前层级配置 = 分配结果[当前层][0]
                            批次请求 = []
                            批次其他 = []
                            while 索引 < 总条目数 and len(批次请求) < Self.Config.TRANSLATOR_BATCH and 条目归属[索引] == 当前层:
                                批次请求.append(处理后的请求内容[索引])
                                批次其他.append(处理后的其他内容[索引])
                                索引 += 1
                            if 批次请求:
                                请求池实例.提交(批次请求, 批次其他, 当前层级配置)
                    else:
                        起始索引 = 0
                        for 索引, (层级配置, 比例) in enumerate(分配结果):
                            if 索引 == len(分配结果) - 1:
                                条目数量 = 总条目数 - 起始索引
                            else:
                                条目数量 = int(总条目数 * 比例)
                            结束索引 = 起始索引 + 条目数量
                            子请求 = 处理后的请求内容[起始索引:结束索引]
                            子其他 = 处理后的其他内容[起始索引:结束索引]
                            for 步进 in range(0, len(子请求), Self.Config.TRANSLATOR_BATCH):
                                请求池实例.提交(子请求[步进:步进 + Self.Config.TRANSLATOR_BATCH], 子其他[步进:步进 + Self.Config.TRANSLATOR_BATCH], 层级配置)
                            起始索引 = 结束索引
                    返回列表 = 请求池实例.获取结果()
                    请求池实例.停止()
                唯一结果映射 = {}
                for a, b, c, d in 返回列表:
                    if isinstance(b, (list, dict)):
                        b = str(b)
                    唯一结果映射[b] = (c, d)
                展开返回列表 = [None] * len(原始请求内容)
                for idx in range(len(原始请求内容)):
                    原文 = 原始请求内容[idx]
                    if 原文 in 唯一结果映射:
                        译文, 其他 = 唯一结果映射[原文]
                        新条目 = [原始其他内容[idx][0], 原文, 译文, 原始其他内容[idx][2]]
                    else:
                        新条目 = [原始其他内容[idx][0], 原文, 原文, 原始其他内容[idx][2]]
                    展开返回列表[idx] = 新条目
                返回列表 = 展开返回列表
            返回列表.extend(去翻译列表)
            返回列表.extend(命中缓存)
            if not 使用模型: 使用模型.append(["null"])
            返回列表 = {str(a): [b, c, d] for a, b, c, d in 返回列表}
            完整返回列表 = []
            翻译缓存输入 = []
            参考字典 = {str(item[0]) for item in 输入复制} if QuestsMode else None
            for 原始条目 in 输入复制:
                if 参考字典 and str(原始条目[0]) not in 参考字典: continue
                try:
                    解析数据 = Self.owolib解析缓存[原始条目[1]]
                    记录列表 = 额外列表.get(f"{原始条目[0]}{原始条目[1]}", [])
                    
                    for 路径, _ in 记录列表:
                        路径键 = "|".join(str(p) for p in 路径)
                        完整键 = f"{原始条目[0]}{路径键}"
                        翻译项 = 返回列表.get(完整键)
                        
                        if 翻译项 and 翻译项[1]:
                            目标对象 = 解析数据
                            for 键 in 路径[:-1]:
                                目标对象 = 目标对象[键]
                            目标对象[路径[-1]] = f"{翻译项[1]}({翻译项[0]})" if Self.Config.TRANSLATOR_ORIGINAL_REFERENCE else 翻译项[1]
                    完整返回列表.append([原始条目[0], json.dumps(解析数据, ensure_ascii=False), 原始条目[2]])
                except Exception:
                    基础键 = 原始条目[0]
                    翻译项 = 返回列表.get(str(基础键))
                    if 翻译项:
                        翻译缓存输入.append([翻译项[0], 翻译项[1]])
                        if Self.Config.TRANSLATOR_ORIGINAL_REFERENCE:
                            完整返回列表.append([基础键, f"{翻译项[1]}({翻译项[0]})", 原始条目[2]])
                        else:
                            完整返回列表.append([基础键, 翻译项[1], 原始条目[2]])
            if Self.Config.TRANSLATOR_CACHE_WRITE:
                Self.Module.翻译缓存(翻译缓存输入)
        except Exception:
            Self.日志("log.core.translator.error", e=eb.format_exc(), texts=输入复制, info_level=3)
            raise
        return 完整返回列表
    def 翻译语言文件(Self, file0: str,  file1: str="", 索引ID: str=uuid.uuid4().hex, output_path: str = "", export_inspection: bool = False, output_lang_str: bool = False, read_error: bool = True, 使用模型: list = []):
        output_path = Self.Module.输出路径处理(output_path)
        输出列表 = []
        if 使用模型 == None: 使用模型 = []
        可翻译源文件, 源文件, 参考文件, 压缩路径, 输出扩展名, file2 = Self.File.读取资源文件(file0, file1, read_error)
        翻译列表 = Self.翻译语言列表(可翻译源文件, 参考文件, 使用模型, 索引ID) #翻译核心
        if export_inspection:
            for index in Self.tqdm(翻译列表, desc="tqdm.progress.encoding"):
                行数据 = {index[0]: index[1]}
                输出列表.append(repr(行数据))
            with open(str(Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translang")), 'w+', encoding='utf-8') as f:
                f.write("\n".join(输出列表))
            Self.日志("log.core.translator.succeed", path=Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translang").resolve(), info_level=0)
            return Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}.translang")
        else:
            分组 = defaultdict(list)
            for a, b, c in 翻译列表:
                分组[c].append([a, b])
            翻译列表 = dict(分组)
            输出列表 = []
            翻译输出列表 = []
            for index in 源文件:
                翻译输出列表 = []
                for index1 in index[0]:
                    if index1.strip().startswith(('#', '//')):
                        翻译输出列表.append(index1)
                    else:
                        索引成功 = False
                        for index2 in 翻译列表[index[1]]:
                            if index1.split('=', 1)[0] == index2[0]:
                                翻译输出列表.append(f"{index2[0]}={index2[1]}")
                                索引成功 = True
                                break
                        if not 索引成功:
                            翻译输出列表.append(index1)
                if 翻译输出列表:
                    输出列表.append([index[1], 翻译输出列表])
            if 压缩路径 and (not output_lang_str):
                for index in 输出列表:
                    Self.File.保存语言文件(f"{Path(index[0]).parent}/{Self.Config.LANGUAGE_OUTPUT}{Path(index[0]).suffix}", index[1])
                压缩文件夹Path = Path(压缩路径)
                if file2[0] == False:
                    文档内容 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(name=Path(file0).stem, lang=Self.Config.LANGUAGE_OUTPUT, model=", ".join(使用模型[0]) or Self.Config.LLM_MODEL or Self.Lang("log.core.package.zip.hit"), author=Self.Config.PACK_AUTHOR or "海盐青茫")
                    with open(压缩文件夹Path/"pack.mcmeta", "w+", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "pack": {
                                "description": 文档内容,
                                "pack_format": 9999,
                                "supported_formats": [0, 9999],
                                "min_format": 0,
                                "max_format": 9999
                            }
                        }, ensure_ascii=False, indent=4))
                with zipfile.ZipFile(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                    for 压缩文件 in 压缩文件夹Path.rglob('*'):
                        if 压缩文件.is_file():
                            f.write(压缩文件, arcname=压缩文件.relative_to(压缩文件夹Path))
                Self.日志("log.core.translator.succeed", path=Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve(), info_level=0)
                return Path(f"{output_path}/{Path(file0).stem}-{Self.Config.LANGUAGE_OUTPUT}.zip")
            else:
                if not Path(output_path).suffix:
                    output_path = str(Path(f"{output_path}/{Self.Config.LANGUAGE_OUTPUT}{输出扩展名}"))
                Self.File.保存语言文件(output_path, 翻译输出列表)
                Self.日志("log.core.translator.succeed", path=Path(output_path).resolve(), info_level=0)
                return Path(f"{output_path}")
    def 翻译流程(Self, path1, 文件匹配, 读取方法, 过滤方法, 分组键方法, 应用方法, 读取并发, 写入并发, 日志类型, 输出方法 = None, path2 = None, **总参数):
        文件列表, 翻译列表, 参考列表, 参考文件列表 = [], [], [], []
        path1 = Path(path1)
        if isinstance(文件匹配, str):
            文件匹配 = [文件匹配]
        for index in 文件匹配:
            文件列表.extend([p for p in path1.rglob(index)] if Path(path1).is_dir() else [path1])
        Self.日志(f"log.core.file.{日志类型}.read.start", info_level=0)
        with ThreadPoolExecutor(max_workers=读取并发) as 执行器:
            结果集 = 执行器.map(读取方法, 文件列表)
            for 结果 in Self.tqdm(结果集, total=len(文件列表), desc="tqdm.file.read"):
                翻译列表.extend(结果)
        if path2:
            path2 = Path(path2)
            for index in 文件匹配:
                参考文件列表.extend([p for p in path2.rglob(index)] if Path(path2).is_dir() else [path2])
            Self.日志(f"log.core.file.{日志类型}.read.start", info_level=0)
            with ThreadPoolExecutor(max_workers=读取并发) as 执行器:
                结果集 = 执行器.map(读取方法, 参考文件列表)
                for 结果 in Self.tqdm(结果集, total=len(参考文件列表), desc="tqdm.file.read"):
                    参考列表.extend(结果)
        Self.日志(f"log.core.file.{日志类型}.read.end", info_level=0)
        过滤后 = []
        try:
            for 条目 in 翻译列表:
                if 过滤方法(条目):
                    过滤后.append(条目)
        except Exception:
            Self.日志(f"log.module.{日志类型}.clean.error", index=条目, e=eb.format_exc(), info_level=2)
        待翻译 = [[条目[0], 条目[1], 条目[2] if len(条目) > 2 else ""] for 条目 in 过滤后]
        使用模型 = []
        翻译函数黑名单 = {"传入使用模型"}
        翻译参数 = {k: v for k, v in 总参数.items() if not k in 翻译函数黑名单}
        保存参数 = {k: v for k, v in 总参数.items() if k in 翻译函数黑名单}
        翻译结果 = Self.翻译语言列表(待翻译, 参考列表, 使用模型=使用模型, **翻译参数)
        分组 = defaultdict(list)
        for 项目 in 翻译结果:
            分组[分组键方法(项目)].append(项目)
        with ThreadPoolExecutor(max_workers=写入并发) as 执行器:
            任务 = 执行器.map(lambda x: 应用方法(x, **保存参数), 分组.values())
            for _ in Self.tqdm(任务, total=len(分组), desc="tqdm.translator.use"):
                pass
        Self.日志("log.core.translator.succeed", path=输出方法(path1) if 输出方法 else path1.resolve(), info_level=0)
    def 翻译FTB任务(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.snbt", Self.File.读取单个FTBQ_Snbt文件, Self.Module.过滤键文本, lambda x: x[0][0], partial(Self.File.应用FTBQ翻译, mode="H" if (Path(path) / "quests").is_dir() else "L"), Self.Config.QUESTS_READ_MAX_CONCURRENT, Self.Config.QUESTS_WRITE_MAX_CONCURRENT, "quests", path2=path2, **参数)
    def 翻译BQ任务(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.json", Self.File.读取单个BQ_Json文件, Self.Module.过滤键文本, lambda x: x[0][0], Self.File.应用BQ翻译, Self.Config.QUESTS_READ_MAX_CONCURRENT, Self.Config.QUESTS_WRITE_MAX_CONCURRENT, "quests", path2=path2, **参数)
    def 翻译HQM任务(Self, path, path2=None, **参数):
        Self.翻译流程(path, ["*.hqm", "*.json"], partial(Self.File.读取单个HQM文件, mode="L" if any(Path(path).rglob("*.hqm")) else "H"), Self.Module.过滤键文本, lambda x: x[0][0], partial(Self.File.应用HQM翻译, mode="L" if any(Path(path).rglob("*.hqm")) else "H"), Self.Config.QUESTS_READ_MAX_CONCURRENT, Self.Config.QUESTS_WRITE_MAX_CONCURRENT, "quests", path2=path2, **参数)
    def 翻译ZS脚本(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.zs", Self.File.读取单个ZS文件, lambda 条目: 条目[1] and not Self.正则表达式预编译.翻译剔除方法.match(条目[1]), lambda x: x[0][0], Self.File.应用ZS翻译, Self.Config.SCRIPT_READ_MAX_CONCURRENT, Self.Config.SCRIPT_WRITE_MAX_CONCURRENT, "script", path2=path2, **参数)
    def 翻译CMM菜单(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.json", Self.File.读取单个CMM文件, Self.Module.过滤键文本, lambda x: x[0][0], Self.File.应用CMM翻译, Self.Config.MENU_READ_MAX_CONCURRENT, Self.Config.MENU_WRITE_MAX_CONCURRENT, "menu", path2=path2, **参数)
    def 翻译FM菜单(Self, path, path2=None, **参数):
        if Path(f"{path}/customization").is_dir(): Self.翻译流程(f"{path}/customization", "*.txt", Self.File.读取单个FM文件, Self.Module.过滤键文本, lambda x: x[0][0], Self.File.应用FM翻译, Self.Config.MENU_READ_MAX_CONCURRENT, Self.Config.MENU_WRITE_MAX_CONCURRENT, "menu", path2=path2, **参数)
        if Path(f"{path}/locals").is_dir():
            翻译语言文件参数 = {"file0": f"{path}/locals/{Self.Config.LANGUAGE_INPUT}.local", "output_path": f"{path}/locals", "output_lang_str": True}
            翻译语言文件参数["file1"] = f"{path}/locals/{Self.Config.LANGUAGE_OUTPUT}.local" if any(Self.Config.LANGUAGE_OUTPUT.lower() in p.name.lower() for p in Path(f"{path}/locals").iterdir() if p.is_file() and p.suffix == '.local') else ""
            Self.翻译语言文件(**翻译语言文件参数, **参数)
    def 翻译帕秋莉手册(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.json", Self.File.读取单个帕秋莉手册文件, Self.Module.过滤键文本, lambda x: str(x[2]), Self.File.应用帕秋莉手册翻译, Self.Config.BOOK_READ_MAX_CONCURRENT, Self.Config.BOOK_WRITE_MAX_CONCURRENT, "book", path2=path2, **参数)
    def 翻译数据包(Self, path, path2=None, **参数):
        path = Path(path)
        if path.is_file():
            缓存文件夹 = Path(f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
            with zipfile.ZipFile(path, 'r') as zf: zf.extractall(缓存文件夹)
        else: 缓存文件夹 = path
        Self.翻译流程(缓存文件夹, ["*.json", "*.mcmeta", "*.mcfunction"], Self.File.读取单个数据包文件, Self.Module.过滤键文本, lambda x: x[0][0], Self.File.应用数据包翻译, Self.Config.DATA_READ_MAX_CONCURRENT, Self.Config.DATA_WRITE_MAX_CONCURRENT, "data", path2=path2, **参数)
    def 翻译未知伤亡语言文件(Self, path, path2=None, **参数):
        Self.翻译流程(path, ["*.json"], Self.File.读取未知伤亡语言文件, Self.Module.过滤键文本, lambda x: x[0][0], Self.File.保存未知伤亡语言文件, Self.Config.LANG_READ_MAX_CONCURRENT, Self.Config.LANG_WRITE_MAX_CONCURRENT, "lang", 输出方法=lambda p: p.parent / f"{Self.Config.LANGUAGE_OUTPUT}.json", 传入使用模型=True, path2=path2, **参数)
    def 翻译未知伤亡dll模组(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.dll", Self.File.读取单个DLL文件, Self.Module.过滤DLL文本, lambda x: x[0][0], Self.File.应用DLL翻译, Self.Config.DLL_READ_MAX_CONCURRENT, Self.Config.DLL_WRITE_MAX_CONCURRENT, "dll", path2=path2, **参数)
    def 翻译MMT文件(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.json", Self.File.读取单个MMT文件, Self.Module.过滤键文本, lambda x: x[0][0], Self.File.应用MMT翻译, Self.Config.LANG_READ_MAX_CONCURRENT, Self.Config.LANG_WRITE_MAX_CONCURRENT, "mmt", path2=path2, **参数)
    def 翻译整合包(Self, path: str, all_mode: bool = False):
        翻译列表路径 = {}
        使用模型 = []
        索引ID = uuid.uuid4().hex
        if Path(f"{path}/mods").is_dir():
            I18n模组ID = [] if all_mode else Self.File.从资源包文件夹获取I18n翻译模组ID(path)
            模组ID = Self.File.从模组文件夹获取模组ID(path)
            模组ID字典 = {item[0]: item[1] for item in 模组ID}
            I18n缺失模组ID = []
            for index in 模组ID字典:
                if index not in I18n模组ID:
                    I18n缺失模组ID.append([index, 模组ID字典[index]])
            缓存路径 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}/"
            Path(缓存路径).mkdir(parents=True, exist_ok=True)
            def 翻译单个模组(模组信息):
                模组ID, 模组文件 = 模组信息
                try:
                    保存路径 = Path(f"{缓存路径}/assets/{模组ID}/lang/")
                    保存路径.mkdir(parents=True, exist_ok=True)
                    Self.翻译语言文件(file0=f"{path}/mods/{模组文件}", file1="", output_path=保存路径, output_lang_str=True, read_error=False, 索引ID=索引ID, 使用模型=使用模型)
                except FileNotFoundError:
                    Self.日志("log.core.translator.modpack.error.mod", e="", mod=模组ID, info_level=0)
                except Exception:
                    Self.日志("log.core.translator.modpack.error.mod", e=eb.format_exc(), mod=模组ID, info_level=1)
            with ThreadPoolExecutor(max_workers=Self.Config.TRANSLATOR_MODPACK_MOD_CONCURRENT) as 执行器:
                for _ in Self.tqdm(执行器.map(翻译单个模组, I18n缺失模组ID), total=len(I18n缺失模组ID), desc="tqdm.translator.mod"):
                    pass
            with open(f"{str(缓存路径)}/pack.mcmeta", "w+", encoding="utf-8") as f:
                文档内容 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(name="", lang=Self.Config.LANGUAGE_OUTPUT, model=", ".join(使用模型[0]) or Self.Config.LLM_MODEL or Self.Lang("log.core.package.zip.hit"), author=Self.Config.PACK_AUTHOR or "海盐青茫")
                f.write(json.dumps({
                    "pack": {
                        "description": 文档内容,
                        "pack_format": 9999,
                        "supported_formats": [0, 9999],
                        "min_format": 0,
                        "max_format": 9999
                    }
                }, ensure_ascii=False, indent=4))
            Path(f"{path}/resourcepacks/").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(f"{path}/resourcepacks/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                for 压缩文件 in Path(缓存路径).rglob('*'):
                    if 压缩文件.is_file():
                        f.write(压缩文件, arcname=压缩文件.relative_to(缓存路径))
            翻译列表路径[f"/resourcepacks/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}.zip"] = ["file"]
        if Path(f"{path}/script").is_dir():
            Self.翻译ZS脚本(f"{path}/script", 索引ID=索引ID)
            翻译列表路径[f"/script"] = ["path"]
        if Path(f"{path}/CustomMainMenu").is_dir():
            Self.翻译CMM菜单(f"{path}/config/CustomMainMenu", 索引ID=索引ID)
            翻译列表路径[f"/config/CustomMainMenu"] = ["path"]
        if Path(f"{path}/config/fancymenu").is_dir():
            Self.翻译FM菜单(f"{path}/config/fancymenu", 索引ID=索引ID)
            翻译列表路径[f"/config/fancymenu"] = ["path"]
        if Path(f"{path}/config/ftbquests").is_dir():
            Self.翻译FTB任务(f"{path}/config/ftbquests", 索引ID=索引ID)
            翻译列表路径[f"/config/ftbquests"] = ["path"]
        if Path(f"{path}/config/betterquesting").is_dir():
            Self.翻译BQ任务(f"{path}/config/betterquesting", 索引ID=索引ID)
            翻译列表路径[f"/config/betterquesting"] = ["path"]
        if Path(f"{path}/config/hqm").is_dir():
            Self.翻译HQM任务(path=f"{path}/config/hqm", 索引ID=索引ID)
            翻译列表路径[f"/config/hqm"] = ["path"]
        if Path(f"{path}/patchouli_books").is_dir():
            Self.翻译帕秋莉手册(path=f"{path}/patchouli_books", 索引ID=索引ID)
            翻译列表路径[f"/patchouli_books"] = ["path"]
        for index in frozenset(["resources", "kubejs/assets"]):
            文件夹路径 = f"{path}/{index}"
            if Path(文件夹路径).is_dir():
                所有文件夹 = [p.name for p in Path(文件夹路径).iterdir() if p.is_dir()]
                if "nuclearcraft" in frozenset(所有文件夹) and Path(f"{文件夹路径}/nuclearcraft/addons/").is_dir():
                    for index2 in Self.tqdm(Path(f"{文件夹路径}/nuclearcraft/addons/").glob("*.zip"), desc="tqdm.translator.nuclearcraftaddonspack"):
                        Self.翻译语言文件(file0=index2, output_path=f"{文件夹路径}/nuclearcraft/addons/", 索引ID=索引ID)
                for 文件夹 in Self.tqdm(所有文件夹, desc="tqdm.translator.resource"):
                    lang_dir = Path(f"{文件夹路径}/{文件夹}/lang")
                    if lang_dir.is_dir():
                        for f in lang_dir.iterdir():
                            if f.is_file() and f.name.lower() == f"{Self.Config.LANGUAGE_INPUT}.lang".lower():
                                Self.翻译语言文件(file0=f, output_path=str(lang_dir), 索引ID=索引ID)
                                break
                        for f in lang_dir.iterdir():
                            if f.is_file() and f.name.lower() == f"{Self.Config.LANGUAGE_INPUT}.json".lower():
                                Self.翻译语言文件(file0=f, output_path=str(lang_dir), 索引ID=索引ID)
                                break
            翻译列表路径[f"/{index}"] = ["path"]
        Self.日志("log.core.translator.succeed", path=Path(f"{path}/resourcepacks/ModPack_Translation-{Self.Config.LANGUAGE_OUTPUT}.zip").resolve(), info_level=0)
        return 翻译列表路径
    def 翻译通用文件(Self, file0, file1 = None, all_mode: bool = False, export_inspection = False):
        缓存文件夹2 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/"
        file0 = Path(file0).resolve()
        if file1:
            file1 = Path(file1).resolve()
        Self.日志("log.core.translator.general.generate.file.input", file0=file0, file1=file1, info_level=0)
        缓存文件夹 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/"
        Path(缓存文件夹).mkdir(parents=True, exist_ok=True)
        Self.日志("log.core.translator.general.generate.start", info_level=0)
        返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
        try:
            if file1 is not None and Path(file1).is_file():
                文件1扩展名 = Path(file1).suffix
                if 文件1扩展名 == ".zip":
                    with zipfile.ZipFile(file0, 'r') as zf:
                        zf.extractall(缓存文件夹2)
            if Path(file0).is_file():
                文件0扩展名 = Path(file0).suffix
                if 文件0扩展名 in [".lang", ".json", ".jar"]:
                   if 文件0扩展名 == ".json":
                       文件数据 = Self.File.读取Json文件(file0)
                       if isinstance(文件数据, dict) and "header" in 文件数据 and "content" in 文件数据:
                           Self.日志("log.core.translator.general.model", model="Mine Mod Translator Mod Json", info_level=0)
                           Self.翻译MMT文件(path=file0, path2=file1)
                           返回内容 = Path(file0)
                       elif "name" in 文件数据:
                           Self.日志("log.core.translator.general.model", model="Casualties: Unknown Language File", info_level=0)
                           Self.翻译未知伤亡语言文件(path=file0, path2=file1)
                           返回内容 = Path(file0)
                       else:
                           Self.日志("log.core.translator.general.model", model="Language File", info_level=0)
                           返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                           返回内容 = Path(返回路径)
                   else:
                       Self.日志("log.core.translator.general.model", model="Mod" if 文件0扩展名 == ".jar" else "Language File", info_level=0)
                       返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                       返回内容 = Path(返回路径)
                   Self.日志("log.core.translator.succeed", path=返回内容.resolve(), info_level=0)
                elif 文件0扩展名 in [".zs"]:
                    Self.日志("log.core.translator.general.model", model="CraftTweaker ZenScripts", info_level=0)
                    Self.翻译ZS脚本(file0=file0, output_path=缓存文件夹, path2=file1)
                    Self.日志("log.core.translator.succeed", path=Path(file0).resolve(), info_level=0)
                    返回内容 = Path(file0)
                elif 文件0扩展名 in [".dll"]:
                    Self.日志("log.core.translator.general.model", model="Casualties: Unknown dll Mod", info_level=0)
                    Self.翻译未知伤亡dll模组(path=file0)
                    Self.日志("log.core.translator.succeed", path=Path(file0).resolve(), info_level=0)
                    返回内容 = Path(file0)
                elif 文件0扩展名 in [".zip", ".mrpack"]:
                    with zipfile.ZipFile(file0, 'r') as zf:
                        namelist = zf.namelist()
                        def has_dir(prefix: str) -> bool:
                            return any(name.startswith(prefix + '/') or name == prefix for name in namelist)
                        def has_path(target: str) -> bool:
                            target = target.strip('/')
                            if not target: return False
                            return any(f"/{target}/" in f"/{name}/" for name in namelist)
                        def 是否仅含指定根文件夹(目标文件夹名: str) -> bool:
                            根目录集合 = {f.split('/', 1)[0] + '/' for f in namelist if '/' in f}
                            if len(根目录集合) != 1:
                                return False
                            根前缀 = 根目录集合.pop()
                            目标完整前缀 = 根前缀 + 目标文件夹名.rstrip('/') + '/'
                            return any(路径.startswith(目标完整前缀) for 路径 in namelist)
                        def 翻译语言文件匹配(显示名称: str):
                            Self.日志("log.core.translator.general.model", model=显示名称, info_level=0)
                            返回路径 = Self.翻译语言文件(file0=file0, file1=file1, output_path=缓存文件夹, export_inspection=export_inspection)
                            Self.日志("log.core.translator.succeed", path=Path(返回路径).resolve(), info_level=0)
                            return Path(返回路径)
                        def 翻译流程匹配(文件夹名称: str, 显示名称: str, 流程名称: Callable[..., Any]):
                            Self.日志("log.core.translator.general.model", model=文件夹名称, info_level=0)
                            zf.extractall(缓存文件夹)
                            流程名称(f"{缓存文件夹}/{文件夹名称}", path2=缓存文件夹2)
                            with zipfile.ZipFile(f"{缓存文件夹}/{f"{显示名称}-Translation" if 文件夹名称 else file0.stem}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                                for 压缩文件 in Path(f"{缓存文件夹}/{文件夹名称}").rglob('*'):
                                    if 压缩文件.is_file():
                                        f.write(压缩文件, arcname=压缩文件.relative_to(str(缓存文件夹)))
                            return Path(f"{缓存文件夹}/{f"{显示名称}-Translation" if 文件夹名称 else file0.stem}.zip")
                        匹配规则 = [
                            ("shaders", "Shaders", 翻译语言文件匹配, None, has_dir("shaders")),
                            ("ftbquests", "FTBQuests", 翻译流程匹配, Self.翻译FTB任务, 是否仅含指定根文件夹("ftbquests")),
                            ("betterquesting", "BetterQuesting", 翻译流程匹配, Self.翻译BQ任务, 是否仅含指定根文件夹('betterquesting')),
                            ("contenttweaker", "NuclearCraft: Overhauled Addons Pack", 翻译语言文件匹配, None, has_dir("contenttweaker")),
                            ("scripts", "ZenScripts", 翻译流程匹配, Self.翻译ZS脚本, 是否仅含指定根文件夹("scripts")),
                            ("CustomMainMenu", "CustomMainMenu", 翻译流程匹配, Self.翻译ZS脚本, 是否仅含指定根文件夹("CustomMainMenu")), 
                            ("fancymenu", "FancyMenu", 翻译流程匹配, Self.翻译FM菜单, 是否仅含指定根文件夹("fancymenu")),
                            ("hqm", "HardcoreQuestingMode", 翻译流程匹配, Self.翻译HQM任务, 是否仅含指定根文件夹("hqm")),
                            ("patchouli_books", "Patchouli", 翻译流程匹配, Self.翻译帕秋莉手册, 是否仅含指定根文件夹("patchouli_books")),
                            ("", "DataPack", 翻译流程匹配, Self.翻译数据包, (has_path(f"data") and has_path("pack.mcmeta"))) # BUG: 压缩文件里会出现一个无内容压缩文件
                        ]
                        返回内容 = None
                        for 文件夹, 显示名, 处理函数, 额外参数, 匹配方法 in 匹配规则:
                            if 匹配方法:
                                if 额外参数 is None:
                                    返回内容 = 处理函数(显示名)
                                else:
                                    返回内容 = 处理函数(文件夹, 显示名, 额外参数)
                                break
                        else:
                            roots = {n.split('/')[0] for n in namelist if not n.startswith('__MACOSX/')}
                            整合包模式 = "General ModPack"
                            if has_dir('overrides'):
                                roots = ["overrides"]
                                整合包模式 = "CurseForge/Modrint/General ModPack"
                            if has_dir('minecraft'):
                                roots = ["minecraft"]
                                整合包模式 = "MultiMC/General ModPack"
                            if len(roots) == 1:
                                root = roots.pop()
                                if has_dir(f'{root}/mods') or has_dir(f'{root}/config') or has_dir(f'{root}/kubejs') or has_dir(f'{root}/resources'):
                                    Self.日志("log.core.translator.general.model", model=整合包模式, info_level=0)
                                    zf.extractall(缓存文件夹)
                                    解压根目录完整路径 = Path(f"{缓存文件夹}/{root}")
                                    压缩路径映射 = Self.翻译整合包(解压根目录完整路径, all_mode=all_mode)
                                    输出Zip路径 = f"{缓存文件夹}/ModPack-Translation-Addion.zip"
                                    with zipfile.ZipFile(输出Zip路径, 'w', zipfile.ZIP_DEFLATED) as modpackzf:
                                        for 相对路径, 类型列表 in 压缩路径映射.items():
                                            类型 = 类型列表[0] if 类型列表 else ""
                                            清理后的相对路径 = 相对路径.lstrip('/')
                                            真实文件路径 = 解压根目录完整路径 / 清理后的相对路径
                                            if 类型 == "file":
                                                modpackzf.write(真实文件路径, arcname=相对路径.lstrip('/'))
                                            elif 类型 == "path":
                                                for 文件完整路径 in 真实文件路径.rglob('*'):
                                                    if 文件完整路径.is_file():
                                                        arcname = 文件完整路径.relative_to(解压根目录完整路径).as_posix()
                                                        modpackzf.write(文件完整路径, arcname=arcname)
                                    Self.日志("log.core.translator.succeed", path=Path(输出Zip路径).resolve(), info_level=0)
                                    返回内容 = Path(输出Zip路径)
                                
                                else:
                                    Self.日志("log.core.translator.general.modpack.translate.file.no", info_level=2)
                            else:
                                Self.日志("log.core.translator.general.structure.unknown", info_level=3)
                else:
                    Self.日志("log.core.translator.general.structure.unknown", info_level=3)
            elif Path(file0).is_dir():
                文件夹名称 = Path(file0).name
                匹配方法 = {
                    "ftbquests": ("FTBQuests", Self.翻译FTB任务, {}),
                    "betterquesting": ("BetterQuesting", Self.翻译BQ任务, {}),
                    "scripts": ("CraftTweaker ZenScripts", Self.翻译ZS脚本, {}),
                    "CustomMainMenu": ("Custom Main Menu", Self.翻译CMM菜单, {}),
                    "fancymenu": ("FancyMenu", Self.翻译FM菜单, {}),
                    "hqm": ("Hardcore Questing Mode", Self.翻译HQM任务, {}),
                    "patchouli_books": ("Patchouli", Self.翻译帕秋莉手册, {}),
                }
                模式, 函数, 参数 = 匹配方法.get(文件夹名称, ("General ModPack", Self.翻译整合包, {"all_mode": all_mode}))
                Self.日志("log.core.translator.general.model", model=模式, info_level=0)
                函数(path=file0, **参数)
                返回内容 = Path(file0)
        except Exception:
            Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=3)
        Self.日志("log.core.translator.succeed", path=返回内容.resolve(), info_level=0)
        return 返回内容.resolve()
        
        
测试 = False
if __name__ == "__main__" and 测试:
    参数 = {
        "LLM1_API_URL": "https://api.deepseek.com/chat/completions",
        "LLM1_MODEL": "deepseek-v4-flash",
        "LLM1_API_KWARGS": {"extra_body": {"thinking": {"type": "disabled"}}},
        "LLM1_MAX_WORKERS": 2500,
        "LLM0_API_URL": "http://127.0.0.1:25564/v1/chat/completions",
        "LLM0_MODEL": "Gemma4-26B-A4B",
        "LLM0_MAX_WORKERS": 1,
        "LLM_TIER_CASCADE": True,
        "LLM0_MIN_COUNT": 100000,
        "LLM_TIER_CASCADE_RATIO": 0.8,
        "TRANSLATOR_BATCH": 5,
        "LLM_CONTEXTS": 3,
        "EMB_API_URL": "http://127.0.0.1:25564/v1/embeddings",
        "EMB_MODEL": "text-embedding-bge-large-en-v1.5",
        "TRANSLATOR_ORIGINAL_REFERENCE": False,
        "LANGUAGE": "zh_CN",
        "TRANSLATOR_CACHE_NAME": "Translator_Cache",
        "VEC_FILE_NAME": "Vectors2",
        "VEC_QUANTIZATION": "Float32",
        "EMB_MAX_WORKERS": 5,
        "DEBUG_MODE": True,
        "TRANSLATOR_CACHE_READ": False,
        "TRANSLATOR_CACHE_WRITE": False,
        "LLM_TIER_INTERLEAVE": False,
        "INDEX_MODE": "GSQMoEPlus",
        "INDEX_SQ": "GSQ2"
        #"LANGUAGE_OUTPUT": "文言",
    }
    翻译 = Translator(参数)
    #翻译.翻译BQ任务(r"mods")
    #翻译.翻译ZS脚本(r"C:\Users\FengMang\Desktop\TranslatorMinecraft\mods")
    #翻译.翻译通用文件(r"C:\Users\FengMang\Downloads\ansiblecrafting-1.0.0-beta-mc1.20.1-fabric.jar")
    #翻译.翻译CMM菜单(r"mods")
    #翻译.翻译FM菜单(r"fancymenu")
    #翻译.翻译HQM任务(r"hqm")
    翻译.翻译通用文件(r"zh_cn.json")
    #翻译.翻译通用文件(r"en_us.lang")
    #翻译.翻译未知伤亡语言文件(r"E:\SteamLibrary\steamapps\common\Casualties Unknown Demo\CasualtiesUnknown_Data\Lang\ZH.json", r"E:\SteamLibrary\steamapps\common\Casualties Unknown Demo\CasualtiesUnknown_Data\Lang\EN.json")
    #翻译.翻译未知伤亡dll模组(r"翻译")
