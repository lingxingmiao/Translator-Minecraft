import threading

from TranslatorLib import (np, zipfile, json, ast, eb, re, defaultdict, token_calibrator, Path, ThreadPoolExecutor, uuid, bisect, SimpleNamespace, random, asyncio, aiohttp, faiss, copy, random, InconsistentValues, Mods, Loop,
                           TranslatorPersistence, Config, HARDWARE_INFO, NOT_IMPORT, GPU_ACC, TranslatorFile)
_语言写锁 = threading.Lock()
class 总结上下文管理器: #占位
    def __init__(Self): pass
    async def add(Self, 原文, 译文): pass
    async def get(Self, 当前原文): return []
class 翻译上下文管理器: #VibeCoding
    def __init__(Self, App, 初始列表):
        Self.Translator: Translator = App
        Self.数据字典 = {}
        Self.已翻译索引列表 = []
        Self.已翻译原文集合 = set()
        Self.索引到原文列表 = []
        Self.线程锁 = None
        Self.原文索引缓存 = {}
        Self.switch = True
        Self.向量索引 = None
        if Self.Translator.Config.TRANSLATOR_CONTEXTS_MODE != "token":
            Self.数据字典 = {原文: "" for 原文 in 初始列表}
            for 索引, 原文 in enumerate(初始列表):
                Self.原文索引缓存[原文] = 索引
                Self.索引到原文列表.append(原文)
        elif Self.Translator.Config.TRANSLATOR_CONTEXTS_MODE == "token":
            Self.上下文 = []
        Self.len = len(初始列表)
    async def add(Self, 原文, 译文):
        if Self.线程锁 is None: Self.线程锁 = asyncio.Lock()
        async with Self.线程锁:
            if Self.Translator.Config.TRANSLATOR_CONTEXTS_MODE == "token":
                Self.上下文.append({"role": "user", "content": 原文})
                Self.上下文.append({"role": "assistant", "content": 译文})
            else:
                if 原文 in Self.数据字典:
                    Self.数据字典[原文] = 译文
                    if 译文 and 译文 != "" and 原文 not in Self.已翻译原文集合:
                        Self.已翻译原文集合.add(原文)
                        bisect.insort(Self.已翻译索引列表, Self.原文索引缓存[原文])
                else:
                    Self.数据字典[原文] = 译文
                    索引 = len(Self.索引到原文列表)
                    Self.原文索引缓存[原文] = 索引
                    Self.索引到原文列表.append(原文)
                if Self.Translator.Config.TRANSLATOR_CONTEXTS_MODE == "vector" and Self.Translator is not None:
                    翻译参考 = [[原文, 译文, 原文]]
                    Self.向量索引, _, _, _ = await TranslatorPersistence.增量索引(Self.Translator, 翻译参考, "context_vector_idx", Self.Translator.Config.INDEX_CONTEXTS_MODE, Self.Translator.Config.TRANSLATOR_CONTEXTS)
    async def get(Self, 当前原文):
        if Self.线程锁 is None: Self.线程锁 = asyncio.Lock()
        if not Self.switch: return []
        模式 = Self.Translator.Config.TRANSLATOR_CONTEXTS_MODE
        单条模式 = not isinstance(当前原文, list)
        原文列表 = [当前原文] if 单条模式 else 当前原文
        if 模式 == "token":
            上下文切片 = Self.上下文[: int(Self.Translator.Config.TRANSLATOR_CONTEXTS * 2)]
            return 上下文切片 if 单条模式 else [上下文切片 for _ in 原文列表]
        所有上下文 = {}
        if 模式 == "vector":
            if getattr(Self, '向量索引', None) is None: return 所有上下文
            缓存 = getattr(Self.Translator, '增量索引缓存', {}).get("context_vector_idx")
            if not 缓存: return 所有上下文
            查询译文列表 = []
            for i, 原文 in enumerate(原文列表):
                译文 = Self.数据字典.get(str(原文), "")
                if 译文:
                    查询译文列表.append([译文])
            if not 查询译文列表: return 所有上下文
            查询向量结果 = await Self.Translator.Builder.并行生成向量(查询译文列表, 查询=True)
            向量矩阵 = np.atleast_2d(np.asarray(查询向量结果[0], dtype=np.float32))
            if GPU_ACC: 向量矩阵 = 向量矩阵.get()
            k = min(Self.Translator.Config.TRANSLATOR_CONTEXTS, Self.向量索引.ntotal)
            faiss.normalize_L2(向量矩阵)
            _, 索引矩阵 = Self.向量索引.search(向量矩阵, k)
            缓存原文 = 缓存.get("ids", [])
            缓存文本 = 缓存.get("texts", [])
            async with Self.线程锁:
                for q_idx in range(len(查询译文列表)):
                    for idx in 索引矩阵[q_idx]:
                        if idx < len(缓存原文):
                            遍历原文 = 缓存原文[idx]
                            原文_ = 缓存文本[idx] if idx < len(缓存文本) else ""
                            译文_ = Self.数据字典.get(str(遍历原文), "")
                            if not 译文_ or str(遍历原文) == str(原文列表[q_idx]):
                                continue
                            所有上下文[原文_] = 译文_
        elif 模式 == "space":
            if not Self.数据字典 or not Self.已翻译索引列表: return 所有上下文
            async with Self.线程锁:
                for i, 原文 in enumerate(原文列表):
                    当前原文str = str(原文)
                    if 当前原文str not in Self.数据字典:
                        continue
                    当前全局索引 = Self.原文索引缓存[当前原文str]
                    pos = bisect.bisect_left(Self.已翻译索引列表, 当前全局索引)
                    start = max(0, pos - Self.Translator.Config.TRANSLATOR_CONTEXTS)
                    选取索引列表 = Self.已翻译索引列表[start:pos]
                    for 索引 in 选取索引列表:
                        遍历原文 = Self.索引到原文列表[索引]
                        所有上下文[遍历原文] = Self.数据字典[遍历原文]
        return 所有上下文
        
class Translator: # AI禁止直接编辑该类
    def __init__(Self, App: Config):
        Self.增量索引缓存          = {} # 增量索引缓存必须要在这 否则持久化文件会OOM 原地飞升Python星球
        Self.Config               = App.Config
        Self.Locale               = App.Locale
        Self.Log                  = App.Log
        Self.Lang                 = App.Lang
        Self.tqdm                 = App.RichTqdm
        Self.DiffTqdm             = App.DiffTqdm
        Self.日志                 = App.日志
        Self.Builder              = App.Builder
        Self.Index                = App.Index
        Self.File                 = App.File
        Self.Module               = App.Module
        Self.Modpack              = App.Modpack
        Self.Quantization         = App.Quantization
        Self.Network              = App.Network
        Self.CacheTokenCalibrator = App.CacheTokenCalibrator
        Self.CacheTranslator      = App.CacheTranslator
        if GPU_ACC: Self.日志("log.core.numpy.gpu", type=HARDWARE_INFO['type'], acc_type=HARDWARE_INFO['acc_type'], version=HARDWARE_INFO['version'], acc_version=HARDWARE_INFO['acc_version'], deviceid=HARDWARE_INFO['device_id'], count=HARDWARE_INFO["device_count"], info_level=0)
        else: Self.日志("log.core.numpy.cpu", type=HARDWARE_INFO['type'], acc_type=HARDWARE_INFO['acc_type'], version=HARDWARE_INFO['version'], acc_version=HARDWARE_INFO['acc_version'], e=HARDWARE_INFO['error'], info_level=0)
        for index in NOT_IMPORT: Self.日志("log.core.not_import", index=index, info_level=1) # ←未安装库警告 ↑Numpy信息
        Self.正则表达式预编译 = SimpleNamespace()
        Self.正则表达式预编译.翻译剔除方法 = re.compile(r'(?i)^\{[^}]*\}$|^[^\w\u4e00-\u9fa5]{1,2}$|^[ivxlcdm]+$|^\d+$|^(?:ulv|lv|mv|hv|ev|iv|luv|zpm|uv|uhv|uev|uiv|uxv|umv|opv|max)$')
        Self.正则表达式预编译.模型输出剔除 = re.compile(r'(?s)(?:<think>.*?</think>|\[think\].*?\[/think\]|<context>.*?</context>|<rag-input>.*?</rag-input>)\s*')
        Self.正则表达式预编译.模型输出转换 = {}
        Self.正则表达式预编译.单词索引分割 = re.compile(r'[ _\-:]+')  
        Self.正则表达式预编译.翻译批量处理 = re.compile(r'<([a-zA-Z][\w:-]*)[^>]*>')
        Self.正则表达式预编译.翻译剔除占位 = re.compile(r'^[\s\W_§]*((%[0-9$.\-+]*[sdfegxXoObBc%])|(\\[nrt"\\]))*[\s\W_§]*$')
        Self.线程锁 = SimpleNamespace()
        Self.线程锁.正则表达式预编译 = SimpleNamespace()
        Self.线程锁.事件循环 = None
        Self.线程锁.正则表达式预编译.模型输出转换 = None # asyncio.Lock()
        Self.线程锁.上下文计数 = None # asyncio.Lock()
        Self.线程锁.Token学习器 = None # asyncio.Lock()
    def __enter__(Self):
        return Self
    async def 生成翻译(Self, 总条目数: int, 请求列表: dict, 上下文管理器: 翻译上下文管理器, 用户提示: str, 请求提示词: list, 使用模型: set,
            就绪事件: asyncio.Event,翻译索引: int, 优先分配列表: list, 任务状态列表: list, 串行事件: asyncio.Event, 总结模式: bool=False):
        当前循环 = asyncio.get_running_loop()
        if Self.线程锁.事件循环 is not 当前循环:
            Self.线程锁.事件循环 = 当前循环
            Self.线程锁.上下文计数 = asyncio.Lock()
            Self.线程锁.Token学习器 = asyncio.Lock()
            Self.线程锁.正则表达式预编译.模型输出转换 = asyncio.Lock()
            
        附属文本, 消息结果 = "", ""
        响应值, 会话, 层级, 工作ID = None, None, None, None
        成功获取过会话, 降级逐条 = False, False
        返回字典, 参考内容 = {}, {}
        错误计数列表 = [0, 0, 0, 0]
        # 0: 请求错误次数
        # 1: 连接错误次数
        # 2: 批量翻译错误次数
        # 3: 逐条翻译错误次数
        原文列表 = list(请求列表.keys())
        参考列表 = list(请求列表.values())
        
        async def 重试抽象(错误计数索引, 请求结果):
            错误计数列表[错误计数索引] += 1
            if 错误计数列表[错误计数索引] >= 层级.get("max_retry", Self.Config.LLM_MAX_RETRY):
                Self.日志("log.core.translator.generate.error", e=eb.format_exc(), output=请求结果, info_level=2)
                return True
            else:
                Self.日志("log.core.translator.generate.retry", e=eb.format_exc(), info_level=1)
                任务状态列表[翻译索引] = "Retrying"
                基础等待 = (层级["retry_coef"] ** (错误计数列表[错误计数索引] - 1)) * 层级["retry_time"]
                await asyncio.sleep(基础等待 + random.uniform(0, 基础等待 * 0.3))
                return False
        
        # ↓批发翻译重试部分
        原始原文列表, 原始参考列表 = 原文列表.copy(), 参考列表.copy()
        if len(原文列表) != 1:
            原文列表, 参考列表 = [原文列表], [参考列表]
            
        while True:
            try:
                for 文本, 参考 in zip(原文列表, 参考列表):
                    try:
                        参考内容 = {}
                        消息 = 请求提示词.copy()
                        上下文 = await 上下文管理器.get(文本)
                        if Self.Config.TRANSLATOR_CONTEXTS_MODE == "token":
                            消息.extend(上下文)
                        else:
                            附属文本 = 附属文本 + f"<context>{" ".join([f"{k}={v}" for k, v in 上下文.items()])}</context>" if 上下文 else ""
                        for 参考组 in 参考: # 摊开RAG参考文本 输出格式 {参考原文: 参考译文, ...}
                            if 参考组 and isinstance(参考组[0], (list, tuple)): 展开列表 = 参考组
                            else: 展开列表 = [参考组]
                            for 单原文参考 in 展开列表:
                                try:
                                    参考内容[单原文参考[0]] = 单原文参考[1]
                                except: pass
                    
                        while True:
                            if 成功获取过会话 or not any(优先分配列表): # 优先分配给重试而不是新任务
                                会话, 层级, 工作ID = Self.Network.get_llm(总条目数, 请求列表, 总结模式)
                            else: return # 尝试获取会话 获取成功了滚出去 没成功重新构建新鲜的上下文
                            if 会话 is None and 工作ID is None and 层级 is None:
                                if 成功获取过会话:
                                    优先分配列表[翻译索引] = True
                                    await asyncio.sleep(0.01) # 删了卡死
                                    continue
                                else: return # 不是第二次获取也滚出去
                            else: break # 获取到了滚出去
                        if not 就绪事件.is_set():
                            就绪事件.set()
                        优先分配列表[翻译索引] = False
                        任务状态列表[翻译索引] = "Working" if not 降级逐条 else "ItemByItem"
                        成功获取过会话 = True
                    
                        附属文本 = 附属文本 + ("" if not 参考内容 else f"<rag-input>{" ".join([f"{k}={v}" for k, v in 参考内容.items()])}</rag-input>")
                        用户文本 = 附属文本 + 用户提示.format(text=json.dumps(文本) if (批处理模式 := 层级["batch_mode"].lower()) == "list" else "\n".join(f"{批处理模式}{t}{Self.正则表达式预编译.翻译批量处理.sub(r'</\1>', 批处理模式)}" for t in 文本) if not (降级逐条 or isinstance(文本, str)) else 文本)
                        消息.append({"role": "user", "content": Self.Module.清理孤立代理(用户文本)})
                        消息[:] = [({**m, "content": Self.Module.清理孤立代理(m["content"])} if isinstance(m.get("content"), str) else m) for m in 消息]
                    
                        Json数据 = {
                            "messages"         : 消息,
                            "top_p"            : 层级["top_p"],
                            "top_k"            : 层级["top_k"],
                            "temperature"      : 层级["temperature"],
                            "seed"             : 层级["seed"],
                            "repeat_penalty"   : 层级["repeat_penalty"],
                            "presence_penalty" : 层级["presence_penalty"],
                            "frequency_penalty": 层级["frequency_penalty"],
                            "stream"           : False}
                        Json数据.update(层级["api_kwargs"])
                        if (not 层级["url"]) and (层级["model"]): # xllamacpp部分
                            模型 = await TranslatorPersistence.获取语言模型(Self, 层级)
                            响应值 = 模型.handle_chat_completions(Json数据)
                        else:
                            Json数据["model"] = 层级["model"]
                            async with 会话.post(url=层级["url"], json=Json数据, timeout=aiohttp.ClientTimeout(
                                    connect=层级["conn_timeout"],
                                    sock_read=层级["timeout"])) as 响应体:
                                if 响应体.status >= 400:
                                    错误信息 = await Self.Module.POST获取错误(响应体, None)
                                    raise aiohttp.ClientResponseError(
                                        request_info=响应体.request_info,
                                        history=响应体.history,
                                        status=响应体.status,
                                        message=错误信息
                                    )
                                响应体.raise_for_status()
                                响应值 = await 响应体.json()
                        Self.Network.close_llm(工作ID)
                        任务状态列表[翻译索引] = "Done"
                        Token结果 = 响应值.get("usage", {})
                        输入Token = Token结果.get("prompt_tokens"    , 0)
                        输出Token = Token结果.get("completion_tokens", 0)
                        原始消息 = 响应值["choices"][0]["message"]["content"]
                        消息结果 = Self.正则表达式预编译.模型输出剔除.sub("", 原始消息)
                        async with Self.线程锁.正则表达式预编译.模型输出转换:
                            if not 批处理模式 in Self.正则表达式预编译.模型输出转换:
                                Self.正则表达式预编译.模型输出转换[批处理模式] = re.compile(fr'{批处理模式}(.*?){Self.正则表达式预编译.翻译批量处理.sub(r'</\1>', 批处理模式)}', re.S)
                        if 总结模式: # ↓总结输出是散文 与批处理格式无关 直接整段作为单条结果
                            尝试解析 = [消息结果]
                        elif 批处理模式 == "list": # ↓列表模式解析JSON 容忍代码围栏与前后废话
                            尝试解析 = Self.Module.解析模型输出JSON(消息结果)
                            if 尝试解析 is None: # ↓不是JSON 单条任务接受裸译文 批量任务按批量错误降级逐条
                                if 降级逐条 or isinstance(文本, str): 尝试解析 = [消息结果]
                                else: raise InconsistentValues(消息结果, 文本)
                            elif isinstance(尝试解析, str): 尝试解析 = [尝试解析] # ↓模型只回了单个JSON字符串
                            elif isinstance(尝试解析, dict): 尝试解析 = list(尝试解析.values()) # ↓模型回了JSON对象 取值为译文序列
                            elif not isinstance(尝试解析, list): 尝试解析 = [] # ↓非法类型 交给长度校验判失败
                        else:
                            尝试解析 = Self.正则表达式预编译.模型输出转换[批处理模式].findall(消息结果)
                        消息结果 = 尝试解析 if 尝试解析 else ([] if 批处理模式 == "list" else [消息结果])
                        if not 总结模式 and not len(消息结果) == len(文本 if not (降级逐条 or isinstance(文本, str)) else [文本]) : raise InconsistentValues(消息结果, 文本)
                        消息结果 = {k: v for k, v in zip(文本 if not (降级逐条 or 总结模式 or isinstance(文本, str)) else [文本], 消息结果)}
                        if token_calibrator is not None: # 学习模型分词器
                            async with Self.线程锁.Token学习器:
                                Self.CacheTokenCalibrator.添加Token(层级["model"], 文本, 输入Token)
                                Self.CacheTokenCalibrator.添加Token(层级["model"], 原始消息, 输出Token)
                        async with Self.线程锁.上下文计数:
                            缓存命中Token = Self.Module.按路径取值(Token结果, Self.Config.LLM_TOKEN_CACHE_HIT_FIELD , 0)
                            层级["token_in"]         += 输入Token
                            层级["token_out"]        += 输出Token
                            层级["token_cache_hit"]  += 缓存命中Token
                            层级["token_cache_miss"] += 输入Token - 缓存命中Token
                        if Self.Config.TRANSLATOR_CONTEXTS_MODE == "token":
                            await 上下文管理器.add(消息[-1]["content"], 原始消息)
                        else:
                            for index0, index1 in 消息结果.items():
                                await 上下文管理器.add(index0, index1)
                        if Self.Config.LOGS_TRANSLATOR_INFO:
                            for index0, index1 in 消息结果.items():
                                Self.日志("log.core.translator.generate", i=index0, o=index1)
                        使用模型.add(层级["model"])
                        返回字典.update(消息结果)
                        if Self.Config.TRANSLATOR_CACHE_WRITE and not 总结模式:
                            Self.CacheTranslator.翻译缓存(返回字典, 语言=Self.Config.LANGUAGE_OUTPUT)
                    except InconsistentValues as err: # 逐条阶段单条失败 只丢这一条
                        Self.Network.close_llm(工作ID)
                        if not 降级逐条: raise # 整批阶段的错误交给外层计数
                        错误计数列表[3] += 1
                        返回字典[文本] = 文本 # 这条回落原文 缓存层会过滤 原文==译文
                        Self.日志("log.core.translator.generate.error", e=eb.format_exc(), output=文本, info_level=2)
                        if 错误计数列表[3] > Self.Config.TRANSLATOR_ITEM_RETRY:
                            Self.日志("log.core.translator.generate.batch.err.max", info_level=3, e=err)
                            任务状态列表[翻译索引] = "Fail"
                            return None
                        continue # 真的跳到下一条
                Self.日志("log.core.translator.generate.request.outputs.debug", messages=用户文本, item=消息结果, info_level=4)
                return 返回字典
            except InconsistentValues as err: # 长度不一致错误 只有整批阶段会进来
                Self.Network.close_llm(工作ID)
                if 错误计数列表[2] < Self.Config.TRANSLATOR_BATCH_RETRY:
                    Self.日志("log.core.translator.generate.batch.retry", info_level=1)
                    任务状态列表[翻译索引] = "Retrying"
                elif 错误计数列表[2] == Self.Config.TRANSLATOR_BATCH_RETRY:
                    Self.日志("log.core.translator.generate.batch.err", info_level=1)
                    任务状态列表[翻译索引] = "Retrying"
                    降级逐条 = True
                    原文列表 = 原始原文列表
                    参考列表 = 原始参考列表
                else:
                    Self.日志("log.core.translator.generate.batch.err.max", info_level=3, e=err)
                    任务状态列表[翻译索引] = "Fail"
                    return None
                错误计数列表[2] += 1
                continue
            except aiohttp.ClientResponseError as err: # 请求错误
                Self.Network.close_llm(工作ID)
                错误信息 = getattr(err, 'message', '') or str(err)
                错误等级 = 3 if int(err.status) in Self.Config.LLM_POAT_CONFIG[1] else 2
                Self.日志(f"log.core.post.{err.status}.err", info_level=错误等级, err=错误信息)
                if 错误等级 == 3 or await 重试抽象(0, 错误信息): # 致命错误
                    任务状态列表[翻译索引] = "Fail"
                    return None
            except (
                    asyncio.TimeoutError           ,  # 异步操作超时
                    aiohttp.ClientConnectorError   ,  # 无法建立TCP连接
                    aiohttp.ClientPayloadError     ,  # 客户端负载/数据错误
                    aiohttp.ClientConnectionError  ,  # 客户端连接异常 父类
                    aiohttp.ServerDisconnectedError,  # 服务端断开连接
                    Exception                      ,  # 基本全部拦截
                ) as err:
                Self.Network.close_llm(工作ID)
                错误信息 = str(err)
                if await 重试抽象(1, 错误信息):
                    任务状态列表[翻译索引] = "Fail"
                    return None
            finally:
                if not 就绪事件.is_set():
                    就绪事件.set()
                if not 串行事件.is_set():
                    串行事件.set()

    async def 翻译语言列表(Self, texts: list, 参考列表: list=None, 使用模型: set=None, 索引ID: str=None, 获取参考文本: bool = False, 纯译文: bool = False) -> list:
        # 参数:
        #     Self:    Translator实例 隐式传入
        #     texts:   需要翻译的列表 格式: [[键, 原文, 文件位置], ...]
        #     参考列表: 参考内容 格式: [[原文, 本地化译文], ...]
        #     使用模型: 添加使用过的翻译模型 不修改指针
        #     索引ID:   LANG索引的共享ID
        #     纯译文:   True时重组返回列表只返回纯译文 忽略原文引用拼接(TRANSLATOR_ORIGINAL_REFERENCE)
        # 返回:
        #     list:    [[键, 译文, 文件位置], ...]

        去翻译字典, 翻译参考列表, 未翻译列表, 待翻译, 重组列表 = {}, [], [], [], []
        未翻译列表文本组件缓存, 翻译列表, 工作列表, 工作返回列表, 提取记录 = [], [], [], [], []
        优先分配列表, 任务状态列表 = [], []
        单词集合, 向量索引 = set(), None
        #去翻译字典 格式: {Key: 译文, ...}
        #未翻译列表 格式: [[Key, 原文, 文件位置], ...]
        #工作列表   格式: [[异步任务, 原文数], ...]
        #工作返回列表   格式: [生成翻译返回值, ...]
        参考字典, 文本组件缓存, 缓存映射 = {}, {}, {}
        单词映射, 文本映射, 索引映射, 翻译映射 = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        #单词映射 文本映射 索引映射 格式: {索引文本: [[原文参考, 译文参考], ...], ...}
        #翻译映射 格式: {原文: [[原文参考, 译文参考], ...], ...}
        #缓存映射 格式: {原文: 译文, ...}
        分组误差 = 0
        if 使用模型 == None: 使用模型 = set()
        if 索引ID is None: 索引ID = uuid.uuid4().hex
        
        # ↓没有输入直接返回
        if texts == []: return []
        
        # ↓剔除{}包裹内容、长度为1-2不含英文数字下划线Unicode中文、纯罗马数字、电压等级标签
        # ↓如果键与文本相同就删除
        texts = [index for index in texts
                 if  not bool(Self.正则表达式预编译.翻译剔除方法.match(str(index[1])))
                 and not bool(Self.正则表达式预编译.翻译剔除占位.match(str(index[1])))
                 and not f"{index[0]}" == f"{index[1]}"]
        输入复制 = texts
        
        # ↓texts与参考列表合并为翻译参考列表 格式:[[texts原文, texts键, 参考列表译文]]
        if 参考列表 != None:
            for item in 参考列表:
                try:
                    参考字典[str(item[0])] = item[1]
                except Exception:
                    Self.日志("log.core.parsing.reference.word.error", e=eb.format_exc(), item=item, info_level=2)
            for index in texts:
                键 = str(index[0])
                if 键 in 参考字典:
                    翻译参考列表.append([index[1], 键, 参考字典[键]])
        
        # ↓使用翻译参考列表构建 索引ID 索引
        参考向量索引, 参考键文本, 参考原文文本, 参考译文文本 = await TranslatorPersistence.增量索引(Self, 翻译参考列表, 索引ID, Self.Config.INDEX_LANG_MODE, Self.Config.INDEX_LANG_K)
        参考文本文件 = [[k, v] for k, v in  zip(参考原文文本, 参考译文文本)]

        # ↓没有参考文本时 *未翻译列表=texts* 有的化尝试分离为 *未翻译列表* 与 *去翻译列表*
        if 参考键文本:
            参考字典 = dict(zip(参考键文本, 参考译文文本))
            for index in texts:
                参考键 = str(index[0])
                if 参考键 in 参考字典:
                    去翻译字典[参考键] = 参考字典[参考键]
                else:
                    未翻译列表.append(index)
            del 参考字典
        else:
            未翻译列表 = texts
        del 参考键文本, 参考原文文本, 参考译文文本, 翻译参考列表, 参考列表, texts
        if 纯译文: 纯译文列表 = list(未翻译列表) # 纯译文模式快照参考分离后的未翻译列表 只导出本次处理条目

        # ↓翻译缓存命中 *未翻译列表* 分离为 *未翻译列表* 与 *去翻译列表*
        if Self.Config.TRANSLATOR_CACHE_READ:
            翻译缓存 = Self.CacheTranslator.翻译缓存(语言=Self.Config.LANGUAGE_OUTPUT)
            原始长度 = len(未翻译列表)
            待翻译 = []
            for index in Self.tqdm(未翻译列表, desc="tqdm.translator.cache.use"):
                if index[1] in 翻译缓存:
                    去翻译字典[str(index[0])] = 翻译缓存[index[1]]
                else:
                    待翻译.append(index)
            未翻译列表 = 待翻译
            成功缓存 = 原始长度 - len(待翻译)
            命中率 = (成功缓存 / 原始长度) if 原始长度 > 0 else 0.0
            Self.日志("log.core.translator.cache.hit", rate=f"{命中率:.4%}", hit=成功缓存, total=原始长度, info_level=0)
        
        # ↓文本组件-解析
        for index in 未翻译列表:
            try:
                解析数据 = ast.literal_eval(index[1])
                if isinstance(解析数据, (dict, list)):
                    提取记录 = []
                    Self.Module.文本组件深度优先搜索(解析数据, [], 提取记录)
                    if 提取记录:
                        当前路径映射 = []
                        for 路径, 文本 in 提取记录:
                            当前路径映射.append((路径, 文本))
                            未翻译列表文本组件缓存.append([文本, 文本, index[2]])
                        文本组件缓存[index[1]] = (解析数据, 当前路径映射)
                    else: 未翻译列表文本组件缓存.append(index)
                else:     未翻译列表文本组件缓存.append(index)
            except:       未翻译列表文本组件缓存.append(index)
        未翻译列表 =       未翻译列表文本组件缓存
    
        # ↓ANN前处理与开始 没有向量文件跳过
        向量文件, 文本文件 = await TranslatorPersistence.参考词预处理(Self, 查询=False)
        # ↓构建翻译参考 无ANN
        for index in 未翻译列表:
            翻译映射[index[1]] = []
        if 文本文件 and 未翻译列表: # 未翻译列表 为空本质跳过
            Self.日志("log.core.index.search.start", info_level=0) # 索引开始
            
            # ↓获取所以
            向量索引 = TranslatorPersistence.缓存索引(Self, 向量文件=向量文件, 文本文件=文本文件)
            索引器   = TranslatorPersistence.快捷搜索器(Self, 向量文件) 
            
            # ↓通用函数 减少重复代码
            async def 索引抽象(输入集合, 索引数量, 索引, 索引列表, 输出映射):
                if not 输入集合: return  # 滚木输入集合直接跳过 防止滚木向量传入faiss触发维度断言
                输入列表 = list(输入集合)
                索引结果矩阵 = await 索引器.search(texts=输入列表, k=索引数量, index=索引)
                for index0, index1 in enumerate(输入列表): # i0为索引 i1为文本
                    输出映射[index1] = [索引列表[i] for i in 索引结果矩阵[index0] if 0 <= i < len(索引列表)] # 剔除无效与越界索引后添加文本索引 文本索引[文本]=[索引文本, ...]
            def 范围匹配抽象(映射, 模糊范围):
                for indexk, indexv0 in 映射.copy().items(): # 匹配映射长度范围 误差模糊范围超过直接删除 原地修改
                    映射[indexk] = [indexv1 for indexv1 in indexv0 if abs(len(indexk) - len(indexv1)) <= 模糊范围]
                    if not 映射[indexk]:
                        del 映射[indexk]
                        
            # ↓ANN单词
            try:
                if Self.Config.INDEX_WORD_K and 向量索引 is not None:
                    for index in 翻译映射: 单词集合.update(index.split()) # 单词列表为集合 split返回为列表所以使用update
                    单词集合 = set(w for w in 单词集合 if len(w.strip()) > 1) # 只保留长度大于1的单词
                    单词集合 = set(w for w in 单词集合 if w.lower() not in {w.lower() for w in Self.Config.INDEX_QUESTS_BASIC_WORDS}) # 创建黑名单后剔除黑名单包含的单词
                    await 索引抽象(单词集合, Self.Config.INDEX_WORD_K, 向量索引, 文本文件, 单词映射)
                    范围匹配抽象(单词映射, Self.Config.INDEX_WORD_RANGE)
            except Exception:
                Self.日志("log.core.index.search.err", e=eb.format_exc(), info_level=2)
            
            # ↓ANN文本
            try:
                if Self.Config.INDEX_TEXT_K and 向量索引 is not None:
                    await 索引抽象(翻译映射, Self.Config.INDEX_TEXT_K, 向量索引, 文本文件, 文本映射)
                    范围匹配抽象(文本映射, Self.Config.INDEX_TEXT_RANGE)
            except Exception:
                Self.日志("log.core.index.search.err", e=eb.format_exc(), info_level=2)
                
            # ↓ANN索引
            try:
                if Self.Config.INDEX_LANG_K and 参考向量索引 is not None:
                    await 索引抽象(翻译映射, Self.Config.INDEX_LANG_K, 参考向量索引, 参考文本文件, 索引映射)
                    范围匹配抽象(索引映射, Self.Config.INDEX_LANG_RANGE)
            except Exception:
                Self.日志("log.core.index.search.err", e=eb.format_exc(), info_level=2)
            Self.日志("log.core.index.search.end", info_level=0) # 索引结束
            
            # ↓构建翻译参考
            for index0 in 翻译映射:
                if Self.Config.INDEX_WORD_K:
                    for index1 in Self.正则表达式预编译.单词索引分割.split(index0):
                        if index1 in 单词映射:
                            翻译映射[index0].extend(单词映射[index1])
                if Self.Config.INDEX_TEXT_K:
                    翻译映射[index0].extend(文本映射[index0])
                if Self.Config.INDEX_LANG_K:
                    翻译映射[index0].extend(索引映射[index0])
        # ↓释放ANN临时数据 降低峰值内存
        del 单词映射, 文本映射, 索引映射, 单词集合, 参考文本文件
        if 向量索引 is not None: del 向量索引
        if 参考向量索引 is not None: del 参考向量索引
            
        # ↓返回ANN数据 Tool调用
        if 获取参考文本: return 翻译映射
        
        # ↓创建上下文管理器
        上下文管理器 = 翻译上下文管理器(Self, 翻译映射)
        总条目数 = 上下文管理器.len # len() 不好看所以加一个.len变量用来获取输入数量
        if 总条目数 == 0: # 无待翻译项：跳过总结与翻译主流程，直接返回缓存/参考已命中的结果
            return [[index[0], 去翻译字典.get(str(index[0]), index[1]), index[2]] for index in (纯译文列表 if 纯译文 else 输入复制)]
        
        # ↓翻译映射分组
        条目键迭代器 = iter(翻译映射)
        批次大小 = Self.Config.TRANSLATOR_BATCH
        if 总条目数 > 0:
            总组数 = max(1, round(总条目数 / 批次大小))
            基础数量, 余数 = divmod(总条目数, 总组数)
            for _ in range(总组数):
                组大小 = 基础数量
                分组误差 += 余数
                if 分组误差 >= 总组数:
                    组大小 += 1
                    分组误差 -= 总组数
                if 组大小 > 0:
                    组 = {}
                    for _ in range(组大小):
                        键 = next(条目键迭代器, None)
                        if 键 is None: break
                        组[键] = 翻译映射[键]
                    if 组: 翻译列表.append(组)
        # 翻译列表 格式:[{"原文": [[参考原文, 参考译文], ...(多参考)], ...(多原文)}, ...(分组)]
        
        # ↓进度条显示
        进度条 = Self.tqdm(total=总条目数, desc="tqdm.translator.generate")
        进度条结束 = False # 主流程完成全部轮次后置True 控制管理进度条协程生命周期
        进度上报 = getattr(Self, "进度上报", None)
        async def 管理进度条(): # ↑精炼复用进度条
            while not 进度条结束: #↓if防止 工作列表.clear() 与 进度条任务.cancel() 之间的微小间隔导致进度条显示0
                if 工作列表: 进度条.n = sum(i[1] for i in 工作列表 if i[0].done()); 进度条.refresh() # 更新进度条并刷新
                if 进度上报 is not None and getattr(Self, "进度模式", None) != "modpack":
                    try: 进度上报(进度条.n, 总条目数, "item")
                    except Exception: pass
                await asyncio.sleep(1/Self.Config.TQDM_FPS) # 没有完成等待刷新
        扩散进度条 = Self.DiffTqdm(tasks=[], desc="tqdm.translator.generate", disable=not Self.Config.TQDM_DIFF)
        async def 管理扩散进度条():
            while not 进度条结束: # 持续刷新直到主流程结束取消；多轮精炼时每轮 任务 引用更新后 refresh 自动适配
                扩散进度条.refresh() # 内部自动管理
                await asyncio.sleep(1/Self.Config.TQDM_FPS)
                
        
        # ↓开始调用大模型翻译
        Self.日志("log.core.translator.generate.start", item=总条目数, info_level=0) # 翻译开始
        内容总结 = ""
        if Self.Config.TRANSLATOR_SUMMARY:
            待总结文本, 当前总长 = [], 0
            for index in 翻译映射.keys():
                当前总长 += len(index) + len("<rt></rt>")
                if 当前总长 > Self.Config.TRANSLATOR_SUMMARY_MAX_TEXT: break
                待总结文本.append(index)
            Self.日志("log.core.translator.summary.start", item=len(待总结文本), info_level=0) # 翻译前总结开始
            总结就绪事件 = asyncio.Event()
            总结串行事件 = asyncio.Event()
            总结结果 = await Self.生成翻译(1, {tuple(待总结文本): []}, 总结上下文管理器(),
                            Self.Config.TRANSLATOR_SUMMARY_USER_PROMPT, [{"role": "system", "content": Self.Config.TRANSLATOR_SUMMARY_SYSTEM_PROMPT.format(lang=Self.Config.LANGUAGE_MODEL)}],
                            set(), 总结就绪事件, 0, [False], [""], 总结串行事件, True)
            内容总结 = next(iter(总结结果.values())) if 总结结果 else ""
            Self.日志("log.core.translator.summary.end", summary=内容总结, info_level=0) # 翻译前总结结束 记录返回内容
        请求提示词 = [{"role": "system", "content": Self.Config.TRANSLATOR_SYSTEM_PROMPT.format(lang=Self.Config.LANGUAGE_MODEL)}]  # 预先构建提示词
        if 内容总结:
            请求提示词[0]["content"] += f"{Self.Config.TRANSLATOR_SUMMARY_TEXT}\n{内容总结}"
        请求文本 = Self.Config.TRANSLATOR_USER_PROMPT.format(lang=Self.Config.LANGUAGE_MODEL, text="{text}")  # 预先构建文本
        进度条任务 = asyncio.create_task(管理进度条()) # 刷新进度条
        扩散进度条任务 = asyncio.create_task(管理扩散进度条())
        try:
            for 轮次 in (range(1) if Self.Config.TRANSLATOR_REFINE_ROUNDS == 0 else Self.tqdm(range(Self.Config.TRANSLATOR_REFINE_ROUNDS + 1), desc="tqdm.translator.refine")): # 翻译精炼
                优先分配列表 = [False for _ in range(len(翻译列表))]
                if 轮次 == 0 and Self.Config.TRANSLATOR_REFINE_ROUNDS > 0: 上下文管理器.switch = False # 第一局不用上下文
                if 轮次 == 1: 上下文管理器.switch = True # 第二局开上下文
                任务状态列表 = ["Unassigned" for _ in range(len(翻译列表))] # Fail Unassigned Working Retrying Done
                扩散进度条.任务 = 任务状态列表
                扩散进度条.数量 = len(翻译列表)
                扩散进度条.n = 0
                信号量 = asyncio.Semaphore(Self.Config.TRANSLATOR_ALLOCATION_CONCURRENT)
                async def 运行任务(indexq, index):
                    async with 信号量:
                        while True:
                            就绪事件 = asyncio.Event()
                            串行事件 = asyncio.Event()
                            任务 = asyncio.create_task(Self.生成翻译(总条目数, index, 上下文管理器, 请求文本, 请求提示词, 使用模型, 就绪事件, indexq, 优先分配列表, 任务状态列表, 串行事件))
                            await 就绪事件.wait()
                            if 任务.done():
                                await asyncio.sleep(0.01)
                                continue
                            工作列表.append([任务, len(index)])
                            if Self.Config.TRANSLATOR_MODE.lower() == "serial":
                                await 串行事件.wait()
                            return 任务
                任务列表 = [asyncio.create_task(运行任务(i, idx)) for i, idx in enumerate(翻译列表)]
                await asyncio.gather(*任务列表)
                try:
                    工作返回列表 = await asyncio.gather(*(i[0] for i in 工作列表))
                except:
                    Self.日志("log.core.translator.error", info_level=2, e=eb.format_exc())
                进度条.n = 总条目数; 进度条.refresh() # 强制补渲染最终100% 防止最后一个任务完成瞬间 clear() 导致进度条停格
                扩散进度条.refresh() # 渲染最终状态 防止取消前的竞态残留 Retrying/Working 快照
                工作列表.clear() # 完全下班 如有循环文则仅上下文共享
                if not 轮次 == Self.Config.TRANSLATOR_REFINE_ROUNDS: 进度条.n = 0 # 不是最后一轮就清零 等待进度条下一次复用
        finally:
            进度条结束 = True # 通知管理进度条协程下班 防止空转
            进度条任务.cancel() # 进度条下班
            扩散进度条任务.cancel()
        
        # ↓合并翻译结果
        for index0 in 工作返回列表: # 格式: [生成翻译返回值, ...]
            if index0 is None: continue # 去除 None 异常返回
            for index1, index2 in index0.items(): # index0={原文: 译文, ...}
                缓存映射[index1] = index2
        del 翻译列表, 工作列表, 工作返回列表, 上下文管理器 # 释放批量任务与上下文数据 降低峰值内存
                
        # ↓写回翻译缓存
        if Self.Config.TRANSLATOR_CACHE_WRITE:
            Self.CacheTranslator.翻译缓存(缓存映射, 语言=Self.Config.LANGUAGE_OUTPUT)
        if "翻译缓存" in locals(): del 翻译缓存

        # ↓重组返回列表
        for index in (纯译文列表 if 纯译文 else 输入复制): # 纯译文模式只重组本次处理条目(未翻译列表) 格式: [[Key, 原文, 文件位置], ...]
            if index[1] in 文本组件缓存:
                解析数据, 当前路径映射 = 文本组件缓存[index[1]]
                目标对象 = copy.deepcopy(解析数据)
                for 路径, 原文本 in 当前路径映射:
                    译文 = 缓存映射.get(原文本, 原文本)
                    if not 纯译文 and Self.Config.TRANSLATOR_ORIGINAL_REFERENCE and 译文 != 原文本:
                        译文 = Self.Config.TRANSLATOR_ORIGINAL_REFERENCE_FORMAT.format(o=翻译映射.get(原文本, []), t=译文)
                    当前对象 = 目标对象
                    for p in 路径[:-1]:
                        当前对象 = 当前对象[p]
                    当前对象[路径[-1]] = 译文
                重组列表.append([index[0], json.dumps(目标对象, ensure_ascii=False), index[2]])
            else:
                译文 = 缓存映射.get(index[1])
                if 译文 is None:
                    译文 = 去翻译字典.get(str(index[0]), index[1])
                elif not 纯译文 and Self.Config.TRANSLATOR_ORIGINAL_REFERENCE:
                    译文 = Self.Config.TRANSLATOR_ORIGINAL_REFERENCE_FORMAT.format(o=翻译映射.get(index[1], []), t=译文)
                重组列表.append([index[0], 译文, index[2]])
        
        return 重组列表
            

    def 翻译语言文件(Self, file0: str,  file1: str="", 索引ID: str=None, output_path: str = "", export_inspection: bool = False, output_lang_str: bool = False, read_error: bool = True, 使用模型: set = set(), 保存语言: str = "", 保留原路径: bool = False):
        if 索引ID is None: 索引ID = uuid.uuid4().hex
        output_path = Self.Module.输出路径处理(output_path)
        输出列表 = []
        if 使用模型 == None: 使用模型 = set()
        可翻译源文件, 源文件, 参考文件, 压缩路径, 输出扩展名, file2 = Self.File.读取资源文件(file0, file1, read_error)
        Self.本次读取条目数 = getattr(Self, "本次读取条目数", 0) + len(可翻译源文件)
        翻译列表 = Loop.运行(Self.翻译语言列表(可翻译源文件, 参考文件, 使用模型, 索引ID, 纯译文=export_inspection)) #翻译核心
        
        保存语言代码 = 保存语言 or Self.Config.LANGUAGE_OUTPUT
        if not 保存语言: # ↓检查是否为1.6-1.10 因为这几个版本要用zh_CN en_US
            模组版本 = Self.File.从模组读取游戏版本(file0)
            if 模组版本:
                try:
                    版本段 = str(模组版本).split('.')
                    主, 次 = (int(版本段[0]), int(版本段[1])) if len(版本段) >= 2 else (int(版本段[0]), 0)
                    if (1, 6) <= (主, 次) <= (1, 10):
                        保存语言代码 = Self.Module.语言代码区域转大写(保存语言代码)
                except Exception: pass
                
        if export_inspection:
            原文对照 = {str(index[0]): index[1] for index in 可翻译源文件}  # 键→原文 用于构造 键: [原文, 译文] 对照
            输出字典 = {str(index[0]): [原文对照.get(str(index[0]), ""), index[1]] for index in 翻译列表}  # {键: [原文, 译文]}
            输出文本 = "{\n" + ",\n".join(f"  {json.dumps(键, ensure_ascii=False)}: {json.dumps(值, ensure_ascii=False)}" for 键, 值 in 输出字典.items()) + "\n}"  # 一行一个键 值保持单行
            with open(str(Path(f"{output_path}/{保存语言代码}.translang")), 'w+', encoding='utf-8') as f:
                f.write(输出文本)
            Self.日志("log.core.translator.succeed", path=Path(f"{output_path}/{保存语言代码}.translang").resolve(), info_level=0)
            return Path(f"{output_path}/{保存语言代码}.translang")
        
        else:
            分组 = defaultdict(list)
            for a, b, c in 翻译列表:
                分组[c].append([a, b])
            翻译列表 = dict(分组)
            翻译映射字典 = {文件路径: {条目[0]: 条目[1] for 条目 in 文件条目列表} for 文件路径, 文件条目列表 in 翻译列表.items()}
            输出列表 = []
            翻译输出列表 = []
            for index in 源文件:
                翻译输出列表 = []
                当前文件映射 = 翻译映射字典.get(index[1], {})
                for index1 in index[0]:
                    if index1.strip().startswith(('#', '//')):
                        翻译输出列表.append(index1)
                    else:
                        当前键 = index1.split('=', 1)[0]
                        if 当前键 in 当前文件映射:
                            翻译输出列表.append(f"{当前键}={当前文件映射[当前键]}")
                        else:
                            翻译输出列表.append(index1)
                if 翻译输出列表:
                    输出列表.append([index[1], 翻译输出列表])
            if output_lang_str and 保留原路径 and 压缩路径:
                # 保留 jar 内真实 assets/<命名空间>/lang 路径；多个模组撞同一命名空间时按键并集
                def _行转字典(行列表, 解析JSON):
                    结果 = {}
                    for 行 in 行列表:
                        if not 行.strip() or 行.lstrip().startswith(("#", "//")) or "=" not in 行:
                            continue
                        k, v = 行.split("=", 1)
                        if 解析JSON:
                            try:
                                v = json.loads(v)
                            except Exception:
                                pass
                        结果[k] = v
                    return 结果
                压缩根 = Path(压缩路径).resolve()
                for 原路径, 行列表 in 输出列表:
                    try:
                        相对 = Path(原路径).resolve().relative_to(压缩根)
                    except Exception:
                        continue
                    if len(相对.parts) < 4 or 相对.parts[0] != "assets":
                        continue
                    目标 = Path(output_path) / 相对.parent / f"{保存语言代码}{相对.suffix}"
                    目标.parent.mkdir(parents=True, exist_ok=True)
                    是JSON = 相对.suffix.lower() == ".json"
                    新数据 = _行转字典(行列表, 是JSON)
                    with _语言写锁:
                        if 是JSON:
                            旧数据 = {}
                            if 目标.is_file():
                                旧数据 = Self.File.读取Json文件(str(目标)) or {}
                            if not isinstance(旧数据, dict):
                                旧数据 = {}
                            旧数据.update(新数据)
                            目标.write_text(json.dumps(旧数据, ensure_ascii=False, indent=4), encoding="utf-8")
                        else:
                            旧数据 = {}
                            if 目标.is_file():
                                旧数据 = _行转字典(目标.read_text(encoding="utf-8", errors="ignore").splitlines(), False)
                            旧数据.update(新数据)
                            目标.write_text("\n".join(f"{k}={v}" for k, v in 旧数据.items()), encoding="utf-8")
                Self.日志("log.core.translator.succeed", path=str(output_path), info_level=0)
                return Path(output_path)
            if 压缩路径 and (not output_lang_str):
                for index in 输出列表:
                    Self.File.保存语言文件(f"{Path(index[0]).parent}/{保存语言代码}{Path(index[0]).suffix}", index[1])
                压缩文件夹Path = Path(压缩路径)
                try:
                    压缩源 = file0 if Path(file0).suffix.lower() in TranslatorFile.归档可写后缀集() else file1
                    if 压缩源 and Path(压缩源).is_file():
                        with zipfile.ZipFile(压缩源, 'r') as 手册压缩包:
                            for 内部文件 in 手册压缩包.namelist():
                                if "/patchouli_books/" in f"/{内部文件}" and not 内部文件.endswith('/'):
                                    if not (压缩文件夹Path / 内部文件).exists():
                                        手册压缩包.extract(内部文件, 压缩路径)
                        for 手册根 in 压缩文件夹Path.glob("assets/*/patchouli_books"):
                            if 手册根.is_dir():
                                Mods().执行(Self, "PatchouliLanguage", str(手册根), 索引ID=索引ID)
                except Exception:
                    Self.日志("log.module.book.load.error", file=str(file0), e=eb.format_exc(), info_level=1)
                if file2[0] == False:
                    文档内容 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(
                        name=Path(file0).stem,
                        lang=保存语言代码,
                        model=", ".join(m for m in 使用模型 if m and m != "null") or Self.Lang("log.core.package.zip.hit"),
                        author=Self.Config.PACK_AUTHOR or "海盐青茫")
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
                with zipfile.ZipFile(f"{output_path}/{Path(file0).stem}-{保存语言代码}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                    for 压缩文件 in 压缩文件夹Path.rglob('*'):
                        if 压缩文件.is_file():
                            f.write(压缩文件, arcname=压缩文件.relative_to(压缩文件夹Path))
                Self.日志("log.core.translator.succeed", path=Path(f"{output_path}/{Path(file0).stem}-{保存语言代码}.zip").resolve(), info_level=0)
                return Path(f"{output_path}/{Path(file0).stem}-{保存语言代码}.zip")
            else:
                if not Path(output_path).suffix:
                    if not 输出扩展名:
                        Self.日志("log.core.translator.skip", file=str(file0), info_level=0)
                        return Path(f"{output_path}")
                    output_path = str(Path(f"{output_path}/{保存语言代码}{输出扩展名}"))
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
            if path2.exists():
                for index in 文件匹配:
                    参考文件列表.extend(p for p in path2.rglob(index))
                Self.日志(f"log.core.file.{日志类型}.read.start", info_level=0)
                with ThreadPoolExecutor(max_workers=读取并发) as 执行器:
                    结果集 = 执行器.map(读取方法, 参考文件列表)
                    for 结果 in Self.tqdm(结果集, total=len(参考文件列表), desc="tqdm.file.read"):
                        参考列表.extend(结果)
        Self.日志(f"log.core.file.{日志类型}.read.end", info_level=0)
        过滤后 = []
        if 过滤方法 is None:
            过滤后 = list(翻译列表)
        else:
            try:
                for 条目 in 翻译列表:
                    if 过滤方法(条目):
                        过滤后.append(条目)
            except Exception:
                Self.日志(f"log.module.{日志类型}.clean.error", index=条目, e=eb.format_exc(), info_level=2)
        待翻译 = [[条目[0], 条目[1], 条目[2] if len(条目) > 2 else ""] for 条目 in 过滤后]
        Self.本次读取条目数 = getattr(Self, "本次读取条目数", 0) + len(待翻译)
        使用模型 = set()
        翻译函数黑名单 = {"传入使用模型"}
        翻译参数 = {k: v for k, v in 总参数.items() if not k in 翻译函数黑名单}
        保存参数 = {"使用模型": 使用模型} if "传入使用模型" in 总参数 else {}
        翻译结果 = Loop.运行(Self.翻译语言列表(待翻译, 参考列表, 使用模型=使用模型, **翻译参数))
        分组 = defaultdict(list)
        for 项目 in 翻译结果:
            分组[分组键方法(项目)].append(项目)
        with ThreadPoolExecutor(max_workers=写入并发) as 执行器:
            任务 = 执行器.map(lambda x: 应用方法(x, **保存参数), 分组.values())
            for _ in Self.tqdm(任务, total=len(分组), desc="tqdm.translator.use"):
                pass
        Self.日志("log.core.translator.succeed", path=输出方法(path1) if 输出方法 else path1.resolve(), info_level=0)
    def 翻译整合包(Self, path: str, all_mode: bool = False):
        翻译列表路径 = {}
        进度上报 = getattr(Self, "进度上报", None)
        Self.进度模式 = "modpack"
        使用模型 = set()
        保存语言代码 = Self.Config.LANGUAGE_OUTPUT
        索引ID = uuid.uuid4().hex
        if Self.Config.MODPACK_AUTO_DOWNLOAD and hasattr(Self, "Modpack"):
            整合包根目录 = Path(path)
            if not Path(f"{整合包根目录}/manifest.json").is_file() and not Path(f"{整合包根目录}/modrinth.index.json").is_file():
                父目录 = 整合包根目录.parent
                if Path(f"{父目录}/manifest.json").is_file() or Path(f"{父目录}/modrinth.index.json").is_file():
                    整合包根目录 = 父目录
            Self.Modpack.下载缺失模组(整合包根目录, Path(path))
        if Path(f"{path}/mods").is_dir():
            I18n模组ID = [] if all_mode else Self.File.从资源包文件夹获取I18n翻译模组ID(path)
            模组ID = Self.File.从模组文件夹获取模组ID(path)
            if Self.File.从模组文件夹获取游戏版本(path):
                保存语言代码 = Self.Module.语言代码区域转大写(保存语言代码)
            模组ID字典 = {item[0]: item[1] for item in 模组ID}
            I18n缺失模组ID = []
            for index in 模组ID字典:
                if index not in I18n模组ID:
                    I18n缺失模组ID.append([index, 模组ID字典[index]])
            缓存路径 = f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/ModPack_Translation-{保存语言代码}/"
            Path(缓存路径).mkdir(parents=True, exist_ok=True)
            def 翻译单个模组(模组信息):
                模组ID, 模组文件 = 模组信息
                try:
                    if not 模组ID or re.search(r'[<>:"/\\|?*\x00-\x1f]', str(模组ID)):
                        模组ID = Path(str(模组文件)).stem
                        模组ID = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', 模组ID) or "unknown"
                    保存路径 = 缓存路径
                    Self.翻译语言文件(file0=f"{path}/mods/{模组文件}", file1="", output_path=保存路径, output_lang_str=True, 保留原路径=True, read_error=False, 索引ID=索引ID, 使用模型=使用模型, 保存语言=保存语言代码)
                except FileNotFoundError:
                    Self.日志("log.core.translator.modpack.error.mod", e="", mod=模组ID, info_level=0)
                except Exception:
                    Self.日志("log.core.translator.modpack.error.mod", e=eb.format_exc(), mod=模组ID, info_level=1)
            with ThreadPoolExecutor(max_workers=Self.Config.MODPACK_MOD_CONCURRENT) as 执行器:
                模组总数 = len(I18n缺失模组ID)
                for 序号, _ in enumerate(Self.tqdm(执行器.map(翻译单个模组, I18n缺失模组ID), total=模组总数, desc="tqdm.translator.mod"), 1):
                    if 进度上报 is not None:
                        try: 进度上报(序号, 模组总数, "mod")
                        except Exception: pass
            # ↓模组 jar 里的非语言文件内容(GuideME 指南页这类): 语言包不覆盖它们, 而且这些和 I18n 资源包无关, 所以所有模组都跑一遍
            def 收集模组内容(模组信息):
                模组ID, 模组文件 = 模组信息
                try:
                    return Self.File.模组内翻译产物(Self, f"{path}/mods/{模组文件}", 缓存路径)
                except Exception:
                    Self.日志("log.core.translator.modpack.error.mod", e=eb.format_exc(), mod=模组ID, info_level=1)
                    return 0
            with ThreadPoolExecutor(max_workers=Self.Config.MODPACK_MOD_CONCURRENT) as 执行器:
                内容总数 = len(模组ID)
                for 序号, _ in enumerate(Self.tqdm(执行器.map(收集模组内容, 模组ID), total=内容总数, desc="tqdm.translator.mod"), 1):
                    if 进度上报 is not None:
                        try: 进度上报(序号, 内容总数, "mod2")
                        except Exception: pass
            with open(f"{str(缓存路径)}/pack.mcmeta", "w+", encoding="utf-8") as f:
                文档内容 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(name="", lang=保存语言代码, model=", ".join(m for m in (使用模型 or []) if m and m != "null") or Self.Lang("log.core.package.zip.hit"), author=Self.Config.PACK_AUTHOR or "海盐青茫")
                f.write(json.dumps({
                    "pack": {
                        "description": 文档内容,
                        "pack_format": 9999,
                        "supported_formats": [0, 9999],
                        "min_format": 0,
                        "max_format": 9999}
                }, ensure_ascii=False, indent=4))
            Path(f"{path}/resourcepacks/").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(f"{path}/resourcepacks/ModPack_Translation-{保存语言代码}.zip", 'w', zipfile.ZIP_DEFLATED) as f:
                for 压缩文件 in Path(缓存路径).rglob('*'):
                    if 压缩文件.is_file():
                        f.write(压缩文件, arcname=压缩文件.relative_to(缓存路径))
            翻译列表路径[f"/resourcepacks/ModPack_Translation-{保存语言代码}.zip"] = ["file"]
        视图 = TranslatorFile.归档视图(路径=path)      # 目录视图: 懒查询, 不遍历整棵树
        计划列表 = Mods().生成计划(视图, "整合包")
        计划总数 = len(计划列表)
        for 序号, 计划 in enumerate(计划列表, 1):
            根 = f"{path}/{计划.根}" if 计划.根 else path
            计划.根 = 根
            try:
                Mods().执行计划(Self, 计划, 原始路径=根, 索引ID=索引ID)
            except Exception:
                Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=2)
                continue
            finally:
                if 进度上报 is not None:
                    try: 进度上报(序号, 计划总数, "plan")
                    except Exception: pass
            翻译列表路径[计划.注册路径 or f"/{计划.根}"] = ["path", 计划.打包扩展名]
        def 登记翻译产物(返回路径):
            if not 返回路径: return
            try: 相对路径 = Path(返回路径).resolve().relative_to(Path(path).resolve()).as_posix()
            except ValueError: return
            翻译列表路径[f"/{相对路径}"] = ["file"]
        for index in frozenset(["resources", "kubejs/assets"]):
            文件夹路径 = f"{path}/{index}"
            if Path(文件夹路径).is_dir():
                所有文件夹 = [p.name for p in Path(文件夹路径).iterdir() if p.is_dir()]
                if "nuclearcraft" in frozenset(所有文件夹) and Path(f"{文件夹路径}/nuclearcraft/addons/").is_dir():
                    for index2 in Self.tqdm(Path(f"{文件夹路径}/nuclearcraft/addons/").glob("*.zip"), desc="tqdm.translator.nuclearcraftaddonspack"):
                        登记翻译产物(Self.翻译语言文件(file0=index2, output_path=f"{文件夹路径}/nuclearcraft/addons/", 索引ID=索引ID))
                for 文件夹 in Self.tqdm(所有文件夹, desc="tqdm.translator.resource"):
                    lang_dir = Path(f"{文件夹路径}/{文件夹}/lang")
                    if lang_dir.is_dir():
                        for f in lang_dir.iterdir():
                            if f.is_file() and f.name.lower() == f"{Self.Config.LANGUAGE_INPUT}.lang".lower():
                                登记翻译产物(Self.翻译语言文件(file0=f, output_path=str(lang_dir), 索引ID=索引ID))
                                break
                        for f in lang_dir.iterdir():
                            if f.is_file() and f.name.lower() == f"{Self.Config.LANGUAGE_INPUT}.json".lower():
                                登记翻译产物(Self.翻译语言文件(file0=f, output_path=str(lang_dir), 索引ID=索引ID))
                                break
        模组目录 = Path(f"{path}/mods")
        清单路径 = Path(f"{path}/vaultpatcher/translatorvault_runtime.json")
        运行时清单 = None
        if 清单路径.is_file():
            try:              运行时清单 = json.loads(清单路径.read_text(encoding="utf-8"))
            except Exception: 运行时清单 = None
        if isinstance(运行时清单, dict):
            for name in 运行时清单.get("mods", []):
                if (模组目录 / name).is_file():
                    翻译列表路径[f"/mods/{name}"] = ["file"]
            for name in 运行时清单.get("plugins", []):
                if Path(f"{path}/vaultpatcher/plugins/{name}").is_file():
                    翻译列表路径[f"/vaultpatcher/plugins/{name}"] = ["file"]
        else:
            if 模组目录.is_dir():
                for jar in sorted(模组目录.glob("vaultpatcher*.jar")):
                    if jar.is_file():
                        翻译列表路径[f"/mods/{jar.name}"] = ["file"]
            插件目录 = Path(f"{path}/vaultpatcher/plugins")
            if 插件目录.is_dir():
                for jar in sorted(插件目录.glob("translatorvault*.jar")):
                    if jar.is_file():
                        翻译列表路径[f"/vaultpatcher/plugins/{jar.name}"] = ["file"]
        模块目录 = Path(f"{path}/vaultpatcher/modules")
        if 模块目录.is_dir():
            翻译列表路径["/vaultpatcher/modules"] = ["path", {".json"}]
        配置目录 = Path(f"{path}/config/vaultpatcher_asm")
        if 配置目录.is_dir():
            翻译列表路径["/config/vaultpatcher_asm"] = ["path", {".json"}]

        Self.增量索引缓存.pop(索引ID, None)
        Self.日志("log.core.translator.succeed", path=Path(f"{path}/resourcepacks/ModPack_Translation-{保存语言代码}.zip").resolve(), info_level=0)
        return 翻译列表路径
    def 翻译任何(Self, 目标, file1=None, 输出目录=None, all_mode: bool = False, export_inspection: bool = False):
        目标 = Path(目标)
        if 输出目录 is None:
            输出目录 = Path(f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
        Path(输出目录).mkdir(parents=True, exist_ok=True)

        if 目标.is_file():
            Self.本次读取条目数 = 0
            命中 = Mods().文件探测(Self, 目标)
            if 命中:
                类型名, _分数 = 命中[0]
                Self.日志("log.core.translator.general.model", model=类型名, info_level=0)
                类型 = Mods().翻译类型注册表.get(类型名)
                if Mods().取执行器(Self, 类型名) == "翻译语言文件":
                    产物 = Mods().执行(Self, 类型名, str(目标), file1=str(file1) if file1 else "",
                                     output_path=str(输出目录), export_inspection=export_inspection)
                else:
                    产物 = Mods().执行(Self, 类型名, str(目标), path2=str(file1) if file1 else None)
                if not Self.本次读取条目数:
                    Self.日志("log.core.translator.general.nothing.read", path=类型名, info_level=2)
                    return Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
                规则 = getattr(类型, "产物", None)
                if 规则 == "源文件":
                    return 目标
                if 规则 == "输出语言文件":
                    return 目标.parent / f"{Self.Config.LANGUAGE_OUTPUT}.json"
                return Path(产物) if 产物 else 目标
            if TranslatorFile.归档可打开(目标):
                return Self.翻译归档(目标, file1=file1, 输出目录=输出目录,
                                all_mode=all_mode, export_inspection=export_inspection)
            返回 = Self.翻译语言文件(file0=str(目标), file1=str(file1) if file1 else "",
                                  output_path=str(输出目录), export_inspection=export_inspection)
            if not Self.本次读取条目数:
                Self.日志("log.core.translator.general.nothing.read", path=目标.name, info_level=2)
                return Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
            return Path(返回)
        if 目标.is_dir():
            return Self.翻译目录(目标, all_mode=all_mode)

        return Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")

    def 翻译归档(Self, file0, file1=None, 输出目录=None, all_mode: bool = False, export_inspection: bool = False):
        file0 = Path(file0)
        缓存文件夹2 = Path(f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
        缓存文件夹 = Path(输出目录) if 输出目录 else Path(f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
        缓存文件夹.mkdir(parents=True, exist_ok=True)

        视图 = TranslatorFile.归档视图(file0)
        容器, 初判计划 = TranslatorFile.路由(视图, file0.suffix)
        Self.日志("log.core.translator.general.model", model=容器, info_level=0)
        TranslatorFile.归档解压(file0, 缓存文件夹, 条目上限=Self.Config.ARCHIVE_MAX_ENTRIES,
                字节上限=Self.Config.ARCHIVE_MAX_BYTES)
        实例根 = Self.File.实例根定位(缓存文件夹)
        计划列表 = 初判计划 if 容器 == "整合包" else Mods().生成计划(TranslatorFile.归档视图(路径=缓存文件夹), 容器)
        if not 计划列表 and 容器 != "整合包":
            计划列表 = Mods().保守计划()
        返回 = None
        if 容器 == "整合包":
            Self.本次读取条目数 = 0
            返回 = Self.File.打包整合包(Self, 实例根, 缓存文件夹, all_mode=all_mode)
            if not Self.本次读取条目数:
                Self.日志("log.core.translator.general.modpack.translate.file.no", info_level=2)
                返回 = None
        else:
            就地翻译过 = False
            自带产物 = None
            for 计划 in 计划列表:
                产出 = Self.File.归档执行一个计划(Self, file0, file1, 缓存文件夹, 计划, 缓存文件夹, 缓存文件夹2, export_inspection)
                if isinstance(产出, Path):
                    自带产物 = 产出
                elif 产出:
                    就地翻译过 = True
            模块目录 = 缓存文件夹 / "vaultpatcher" / "modules"
            有硬编码 = 模块目录.is_dir() and any(模块目录.glob("*.json"))
            if 有硬编码 and file0.suffix.lower() == ".jar":
                返回 = Self.File.打包硬编码包(file0, 缓存文件夹, 自带产物, Self)
            elif 自带产物 is not None:
                返回 = 自带产物
            elif 就地翻译过:
                返回 = Self.File.打包整个包(file0, 缓存文件夹)
            else:
                Self.日志("log.core.translator.general.modpack.translate.file.no", info_level=2)
        return Path(返回) if 返回 else Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")

    def 翻译目录(Self, 目标, all_mode: bool = False):
        视图 = TranslatorFile.归档视图(路径=目标)
        容器, 计划列表 = TranslatorFile.路由(视图)
        Self.日志("log.core.translator.general.model", model=容器, info_level=0)
        Self.本次读取条目数 = 0
        if 容器 == "整合包":
            Self.翻译整合包(path=str(目标), all_mode=all_mode)
            if not Self.本次读取条目数:
                Self.日志("log.core.translator.general.modpack.translate.file.no", info_level=2)
                return Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
            return Path(目标)
        读到过 = 0
        for 计划 in 计划列表:
            计划.根 = str(Path(目标) / 计划.根) if 计划.根 else str(目标)
            try:
                Self.本次读取条目数 = 0
                Mods().执行计划(Self, 计划, 原始路径=计划.根)
                读到过 += Self.本次读取条目数
            except Exception:
                Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=2)
        if not 读到过:
            Self.日志("log.core.translator.general.nothing.read", path=Path(目标).name, info_level=2)
            return Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
        return Path(目标)

    def 翻译通用文件(Self, file0, file1 = None, all_mode: bool = False, export_inspection = False):
        file0 = Path(file0).resolve()
        file1 = Path(file1).resolve() if file1 else None
        Self.日志("log.core.translator.general.generate.file.input", file0=file0, file1=file1, info_level=0)
        缓存文件夹 = Path(f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
        缓存文件夹.mkdir(parents=True, exist_ok=True)
        Self.日志("log.core.translator.general.generate.start", info_level=0)
        try:
            返回内容 = Self.翻译任何(file0, file1=file1, 输出目录=缓存文件夹, all_mode=all_mode, export_inspection=export_inspection)
        except Exception:
            Self.日志("log.core.translator.general.error.unknown", e=eb.format_exc(), info_level=3)
            返回内容 = None
        finally:
            Self.增量索引缓存.clear()
        if 返回内容 is None:
            返回内容 = Path(f"{Self.Config.LOGS_FILE_PATH}/{Self.Config.LOGS_FILE_NAME}.log")
        Self.日志("log.core.translator.succeed", path=返回内容.resolve(), info_level=0)
        return 返回内容.resolve()