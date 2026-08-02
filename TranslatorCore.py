from TranslatorLib import (np, zipfile, json, ast, eb, re, partial, defaultdict, token_calibrator, Path, ThreadPoolExecutor, Callable, uuid, bisect, SimpleNamespace, Any, random, asyncio, aiohttp, shutil, faiss, copy, random, InconsistentValues,
                           TranslatorPersistence, Config, HARDWARE_INFO, NOT_IMPORT, GPU_ACC)
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
        Self.增量索引缓存          = {}                        # 增量索引缓存必须要在这 否则持久化文件会OOM
        Self.Config               = App.Config
        Self.Locale               = App.Locale
        Self.Lang                 = App.Lang
        Self.tqdm                 = App.RichTqdm
        Self.DiffTqdm             = App.DiffTqdm
        Self.日志                 = App.日志
        Self.Builder              = App.Builder
        Self.Index                = App.Index
        Self.File                 = App.File
        Self.Module               = App.Module
        Self.Quantization         = App.Quantization
        Self.Network              = App.Network
        Self.CacheTokenCalibrator = App.CacheTokenCalibrator
        Self.CacheTranslator      = App.CacheTranslator
        if GPU_ACC: Self.日志("log.core.numpy.gpu", type=HARDWARE_INFO['type'], acc_type=HARDWARE_INFO['acc_type'], version=HARDWARE_INFO['version'], acc_version=HARDWARE_INFO['acc_version'], deviceid=HARDWARE_INFO['device_id'], count=HARDWARE_INFO["device_count"], info_level=0)
        else: Self.日志("log.core.numpy.cpu", type=HARDWARE_INFO['type'], acc_type=HARDWARE_INFO['acc_type'], version=HARDWARE_INFO['version'], acc_version=HARDWARE_INFO['acc_version'], e=HARDWARE_INFO['error'], info_level=0)
        for index in NOT_IMPORT: Self.日志("log.core.not_import", index=index, info_level=1) # ←未安装库警告 ↑Numpy信息
        Self.正则表达式预编译 = SimpleNamespace()
        Self.正则表达式预编译.翻译剔除方法 = re.compile(r'^\{[^}]*\}$|^[^\w\u4e00-\u9fa5]{1,2}$')
        Self.正则表达式预编译.模型输出剔除 = re.compile(r'(?s)(?:<think>.*?</think>|\[think\].*?\[/think\]|<context>.*?</context>|<rag-input>.*?</rag-input>)\s*')
        Self.正则表达式预编译.模型输出转换 = re.compile(r'<rt>(.*?)</rt>', re.S)
        Self.正则表达式预编译.单词索引分割 = re.compile(r'[ _\-:]+')  
        Self.线程锁 = SimpleNamespace()
        Self.线程锁.上下文计数 = None # asyncio.Lock()
        Self.线程锁.Token学习器 = None # asyncio.Lock()
    def __enter__(Self):
        return Self
    async def 生成翻译(Self, 总条目数: int, 请求列表: dict, 上下文管理器: 翻译上下文管理器, 用户提示: str, 请求提示词: list, 使用模型: set, 就绪事件: asyncio.Event, 翻译索引: int, 优先分配列表: list, 任务状态列表: list, 总结模式: bool=False):
        if Self.线程锁.上下文计数 is None: Self.线程锁.上下文计数 = asyncio.Lock()
        if Self.线程锁.Token学习器 is None: Self.线程锁.Token学习器 = asyncio.Lock()
        附属文本, 消息结果 = "", ""
        响应值 = None
        成功获取过会话, 降级逐条 = False, False
        返回字典, 参考内容 = {}, {}
        错误计数列表 = [0, 0, 0]
        # 0: 请求错误次数
        # 1: 连接错误次数
        # 2: 批量翻译错误次数
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
        原始原文列表, 原始参考列表 = 原文列表, 参考列表
        if len(原文列表) != 1:
            原文列表, 参考列表 = [原文列表], [参考列表]
        
        while True:
            try:
                for 文本, 参考 in zip(原文列表, 参考列表):
                    参考内容 = {}
                    消息 = 请求提示词.copy()
                    上下文 = await 上下文管理器.get(文本)
                    if Self.Config.TRANSLATOR_CONTEXTS_MODE == "token":
                        请求列表.extend(上下文)
                    else:
                        附属文本 = 附属文本 + f"<context>{" ".join([f"{k}={v}" for k, v in 上下文.items()])}</context>" if 上下文 else ""
                    for 参考组 in 参考: # 摊开RAG参考文本 输出格式 {参考原文: 参考译文, ...}
                        for 单原文参考 in 参考组:
                            try:
                                参考内容[单原文参考[0]] = 单原文参考[1]
                            except: pass
                    附属文本 = 附属文本 + ("" if not 参考内容 else f"<rag-input>{" ".join([f"{k}={v}" for k, v in 参考内容.items()])}</rag-input>")
                    用户文本 = 附属文本 + 用户提示.format(text="\n".join(f"<rt>{t}</rt>" for t in 文本) if not 降级逐条 else 文本)
                    消息.append({"role": "user", "content": 用户文本})
                    
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
                    Json数据 = {
                        "messages"         : 消息,
                        "model"            : 层级["model"],
                        "top_p"            : 层级["top_p"],
                        "top_k"            : 层级["top_k"],
                        "temperature"      : 层级["temperature"],
                        "seed"             : 层级["seed"],
                        "repeat_penalty"   : 层级["repeat_penalty"],
                        "presence_penalty" : 层级["presence_penalty"],
                        "frequency_penalty": 层级["frequency_penalty"],
                        "stream"           : False}
                    Json数据.update(层级["api_kwargs"])
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
                    尝试解析 =  Self.正则表达式预编译.模型输出转换.findall(消息结果)
                    消息结果 = 尝试解析 if 尝试解析 else [消息结果]
                    if not 总结模式 and not len(消息结果) == len(文本 if not 降级逐条 else [文本]) : raise InconsistentValues(消息结果, 文本)
                    消息结果 = {k: v for k, v in zip(文本 if not (降级逐条 or 总结模式) else [文本], 消息结果)}
                    if token_calibrator is not None: # 学习模型分词器
                        async with Self.线程锁.Token学习器:
                            Self.CacheTokenCalibrator.添加Token(层级["model"], 文本, 输入Token)
                            Self.CacheTokenCalibrator.添加Token(层级["model"], 原始消息, 输出Token)
                    async with Self.线程锁.上下文计数:
                        Self.Config.LLM_TOKEN_IN         += 输入Token
                        Self.Config.LLM_TOKEN_OUT        += 输出Token
                        Self.Config.LLM_TOKEN_CACHE_HIT  += Self.Module.按路径取值(Token结果, Self.Config.LLM_TOKEN_CACHE_HIT_FIELD , 0)
                        Self.Config.LLM_TOKEN_CACHE_MISS += Self.Module.按路径取值(Token结果, Self.Config.LLM_TOKEN_CACHE_MISS_FIELD, 0)
                    if Self.Config.TRANSLATOR_CONTEXTS_MODE == "token":
                        await 上下文管理器.add(消息[-1]["content"], 原始消息)
                    else:
                        for index0, index1 in 消息结果.items():
                            await 上下文管理器.add(index0, index1)
                    for index0, index1 in 消息结果.items():
                        Self.日志("log.core.translator.generate", i=index0, o=index1)
                    使用模型.add(层级["model"])
                    返回字典.update(消息结果)
                Self.日志("log.core.translator.generate.request.outputs.debug", messages=用户文本, item=消息结果, info_level=4)
                return 返回字典
            except InconsistentValues as err: # 长度不一致错误
                Self.Network.close_llm(工作ID)
                if 错误计数列表[2] == Self.Config.TRANSLATOR_BATCH_RETRY:
                    Self.日志("log.core.translator.generate.batch.err", info_level=1)
                    任务状态列表[翻译索引] = "Retrying"
                    降级逐条 = True
                    原文列表 = 原始原文列表
                    参考列表 = 原始参考列表
                elif 错误计数列表[2] < Self.Config.TRANSLATOR_BATCH_RETRY:
                    Self.日志("log.core.translator.generate.batch.retry", info_level=1)
                    任务状态列表[翻译索引] = "Retrying"
                elif 错误计数列表[2] > Self.Config.TRANSLATOR_BATCH_RETRY:
                    Self.日志("log.core.translator.generate.batch.err.max", info_level=3, e=err)
                    任务状态列表[翻译索引] = "Fail"
                    return None
                错误计数列表[2] += 1
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

    async def 翻译语言列表(Self, texts: list, 参考列表: list=None, 使用模型: set=None, 索引ID: str=uuid.uuid4().hex, 获取参考文本: bool = False) -> list:
        # 参数:
        #     Self:    Translator实例 隐式传入
        #     texts:   需要翻译的列表 格式: [[键, 原文, 文件位置], ...]
        #     参考列表: 参考内容 格式: [[原文, 本地化译文], ...]
        #     使用模型: 添加使用过的翻译模型 不修改指针
        #     索引ID:   LANG索引的共享ID
        # 返回:
        #     list:    [[键, 译文, 文件位置], ...]

        命中缓存, 去翻译列表, 翻译参考列表, 未翻译列表, 待翻译, 重组列表 = [], [], [], [], [], []
        未翻译列表文本组件缓存, 翻译列表, 工作列表, 工作返回列表, 提取记录 = [], [], [], [], []
        优先分配列表, 任务状态列表 = [], []
        单词集合, 文本集合, 索引集合 = set(), set(), set()
        #去翻译列表 格式: [[Key, 译文, 文件位置], ...]
        #未翻译列表 格式: [[Key, 原文, 文件位置], ...]
        #工作列表   格式: [[异步任务, 原文数], ...]
        #工作返回列表   格式: [生成翻译返回值, ...]
        参考字典, 文本组件缓存, 返回映射, 缓存映射 = {}, {}, {}, {}
        单词映射, 文本映射, 索引映射, 翻译映射 = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        #单词映射 文本映射 索引映射 格式: {索引文本: [[原文参考, 译文参考], ...], ...}
        #翻译映射 格式: {原文: [[原文参考, 译文参考], ...], ...}
        #返回映射 缓存映射 格式: {原文: 译文, ...}
        分组索引位置, 分组误差 = 0, 0
        if 使用模型 == None: 使用模型 = set()
        
        # ↓没有输入直接返回
        if texts == []: return []
        
        # ↓是否为任务模式 翻译流程为True
        try: QuestsMode = True if isinstance(texts[0][0], list) else False
        except: QuestsMode = False
        
        # ↓任务模式下剔除{}包裹内容与长度为1-2并且不含英文数字下划线Unicode中文
        texts = [texts[index] for index in range(len(texts)) if not bool(Self.正则表达式预编译.翻译剔除方法.match(texts[index][1]))] if QuestsMode else texts
        # ↓如果键与文本相同就删除
        texts = [index for index in texts if not f"{index[0]}" == f"{index[1]}"]
        
        输入复制 = texts.copy()
        
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
                    去翻译列表.append([index[0], 参考字典[参考键], index[2]])
                else:
                    未翻译列表.append(index)
        else:
            未翻译列表 = texts.copy()

        # ↓翻译缓存命中 *未翻译列表* 分离为 *未翻译列表* 与 *去翻译列表*
        if Self.Config.TRANSLATOR_CACHE_READ:
            翻译缓存 = Self.CacheTranslator.翻译缓存(语言=Self.Config.LANGUAGE_OUTPUT)
            原始长度 = len(未翻译列表)
            for index in Self.tqdm(未翻译列表, desc="tqdm.translator.cache.use"):
                if index[1] in 翻译缓存:
                    命中缓存.append([index[0], 翻译缓存[index[1]], index[2]])
                else:
                    待翻译.append(index)
            未翻译列表[:] = 待翻译
            成功缓存 = len(命中缓存)
            去翻译列表.extend(命中缓存)
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
                else: 未翻译列表文本组件缓存.append(index)
            except: 未翻译列表文本组件缓存.append(index)
        未翻译列表[:] = 未翻译列表文本组件缓存
    
        # ↓ANN前处理与开始 没有向量文件跳过
        翻译文本列表 = [index[1] for index in 未翻译列表] 
        向量文件, 文本文件 = await TranslatorPersistence.参考词预处理(Self, 查询=False)
        if 向量文件 and 文本文件 and 未翻译列表: # 未翻译列表 为空本质跳过
            Self.日志("log.core.index.search.start", info_level=0) # 索引开始
            
            # ↓获取所以
            向量索引 = TranslatorPersistence.缓存索引(Self, 向量文件=向量文件, 文本文件=文本文件)
            
            # ↓通用函数 减少重复代码
            async def 索引抽象(输入集合, 索引数量, 索引, 索引列表, 输出映射):
                if not 输入集合: return  # 滚木输入集合直接跳过 防止滚木向量传入faiss触发维度断言
                输入列表 = await Self.Builder.并行生成向量([[index, "", ""] for index in 输入集合], 查询=True) # 格式化后生成检索向量 返回格式[向量, [生成文本, 额外, 额外]]
                向量列表 = np.asarray(输入列表[0], dtype=np.float32) # 提取向量部分
                if 向量列表.shape[0] == 0: return  # 滚木向量直接跳过 防止faiss维度断言崩溃
                Self.Quantization.PCA应用懒加载(向量列表, 向量文件) # PCA降维 原地修改
                Self.Quantization.TT应用懒加载(向量列表, 向量文件) # TT解压 原地修改
                向量列表 = 向量列表.get() if GPU_ACC else 向量列表 # GPU转换CPU
                faiss.normalize_L2(向量列表) # L2归一化 原地修改
                for _ in Self.tqdm(range(1), desc="tqdm.index.search"):
                    索引结果矩阵 = 索引.search(向量列表, 索引数量)[1] # ANN
                for index0, index1 in zip(range(len(向量列表)), 输入列表[1][0]): # i0为向量 i1为文本
                    输出映射[index1] = [索引列表[i] for i in 索引结果矩阵[index0] if i >= 0] # 剔除无效索引后添加文本索引 文本索引[文本]=[索引文本, ...]
            def 范围匹配抽象(映射, 模糊范围):
                for indexk, indexv0 in 映射.copy().items(): # 匹配映射长度范围 误差模糊范围超过直接删除 原地修改
                    映射[indexk] = [indexv1 for indexv1 in indexv0 if abs(len(indexk) - len(indexv1)) <= 模糊范围]
                    if not 映射[indexk]:
                        del 映射[indexk]
                        
            # ↓ANN单词
            if Self.Config.INDEX_WORD_K and 向量索引 is not None:
                for index in 翻译文本列表: 单词集合.update(index.split()) # 单词列表为集合 split返回为列表所以使用update
                单词集合 = set(w for w in 单词集合 if len(w.strip()) > 1) # 只保留长度大于1的单词
                单词集合 = set(w for w in 单词集合 if w.lower() not in {w.lower() for w in Self.Config.INDEX_QUESTS_BASIC_WORDS}) # 创建黑名单后剔除黑名单包含的单词
                await 索引抽象(单词集合, Self.Config.INDEX_WORD_K, 向量索引, 文本文件, 单词映射)
                范围匹配抽象(单词映射, Self.Config.INDEX_WORD_RANGE)
            
            # ↓ANN文本
            if Self.Config.INDEX_TEXT_K and 向量索引 is not None:
                for index in 翻译文本列表: 文本集合.add(index) # 文本转换集合 由于直接是文本所以使用add
                await 索引抽象(文本集合, Self.Config.INDEX_TEXT_K, 向量索引, 文本文件, 文本映射)
                范围匹配抽象(文本映射, Self.Config.INDEX_TEXT_RANGE)
                
            # ↓ANN索引
            if Self.Config.INDEX_LANG_K and 参考向量索引 is not None:
                for index in 翻译文本列表: 索引集合.add(index)
                await 索引抽象(索引集合, Self.Config.INDEX_LANG_K, 参考向量索引, 参考文本文件, 索引映射)
                范围匹配抽象(索引映射, Self.Config.INDEX_LANG_RANGE)
            Self.日志("log.core.index.search.end", info_level=0) # 索引结束
            
            # ↓构建翻译参考
            for index0 in 翻译文本列表:
                if Self.Config.INDEX_WORD_K:
                    for index1 in Self.正则表达式预编译.单词索引分割.split(index0):
                        if index1 in 单词映射:
                            翻译映射[index0].extend(单词映射[index1])
                if Self.Config.INDEX_TEXT_K:
                    翻译映射[index0].extend(文本映射[index0])
                if Self.Config.INDEX_LANG_K:
                    翻译映射[index0].extend(索引映射[index0])
        else:
            # ↓构建翻译参考 无ANN
            for index in 翻译文本列表:
                翻译映射[index] = []
        for index in 翻译映射.keys(): # 初始化返回内容
            返回映射[index] = index
            
        # ↓返回ANN数据 Tool调用
        if 获取参考文本: return 翻译映射
        
        # ↓创建上下文管理器
        上下文管理器 = 翻译上下文管理器(Self, [k for k in 翻译映射.keys()])
        总条目数 = 上下文管理器.len # len() 不好看所以加一个.len变量用来获取输入数量
        
        # ↓翻译映射分组
        条目键 = list(翻译映射.keys())
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
                    下一索引 = 分组索引位置 + 组大小
                    翻译列表.append({k: 翻译映射[k] for k in 条目键[分组索引位置:下一索引]})
                    分组索引位置 = 下一索引
        # 翻译列表 格式:[{"原文": [[参考原文, 参考译文], ...(多参考)], ...(多原文)}, ...(分组)]
        
        # ↓进度条显示
        进度条 = Self.tqdm(total=总条目数, desc="tqdm.translator.generate")
        async def 管理进度条(): # ↑精炼复用进度条
            while 进度条.n < 总条目数: #↓if防止 工作列表.clear() 与 进度条任务.cancel() 之间的微小间隔导致进度条显示0
                if 工作列表: 进度条.n = sum(i[1] for i in 工作列表 if i[0].done()); 进度条.refresh() # 更新进度条并刷新
                await asyncio.sleep(1/Self.Config.TQDM_FPS) # 没有完成等待刷新
        扩散进度条 = Self.DiffTqdm(tasks=[], desc="tqdm.translator.generate", disable=not Self.Config.TQDM_DIFF)
        async def 管理扩散进度条():
            while True: # 持续刷新直到主流程结束取消；多轮精炼时每轮 任务 引用更新后 refresh 自动适配
                扩散进度条.refresh() # 内部自动管理
                await asyncio.sleep(1/Self.Config.TQDM_FPS)
                
        
        # ↓开始调用大模型翻译
        Self.日志("log.core.translator.generate.start", item=总条目数, info_level=0) # 翻译开始
        内容总结 = ""
        if Self.Config.TRANSLATOR_SUMMARY:
            # ↓按字符预算截取全部待翻译原文 交给生成翻译统一包裹<rt>标签 (键必须可哈希故用tuple)
            待总结文本, 当前总长 = [], 0
            for index in 翻译映射.keys():
                当前总长 += len(index) + len("<rt></rt>")
                if 当前总长 > Self.Config.TRANSLATOR_SUMMARY_MAX_TEXT: break
                待总结文本.append(index)
            总结就绪事件 = asyncio.Event()
            总结结果 = await Self.生成翻译(1, {tuple(待总结文本): []}, 总结上下文管理器(),
                            Self.Config.TRANSLATOR_SUMMARY_USER_PROMPT, [{"role": "system", "content": Self.Config.TRANSLATOR_SUMMARY_SYSTEM_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT)}],
                            set(), 总结就绪事件, 0, [False], [""], True)
            内容总结 = next(iter(总结结果.values())) if 总结结果 else ""
        请求提示词 = [{"role": "system", "content": Self.Config.TRANSLATOR_SYSTEM_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT)}]  # 预先构建提示词
        if 内容总结:
            请求提示词[0]["content"] += f"{Self.Config.TRANSLATOR_SUMMARY_TEXT}\n{内容总结}"
        请求文本 = Self.Config.TRANSLATOR_USER_PROMPT.format(lang=Self.Config.LANGUAGE_OUTPUT, text="{text}")  # 预先构建文本
        进度条任务 = asyncio.create_task(管理进度条()) # 刷新进度条
        扩散进度条任务 = asyncio.create_task(管理扩散进度条())
        for 轮次 in Self.tqdm(range(Self.Config.TRANSLATOR_REFINE_ROUNDS + 1), desc="tqdm.translator.refine"): # 翻译精炼
            优先分配列表 = [False for _ in range(len(翻译列表))]
            if 轮次 == 0 and Self.Config.TRANSLATOR_REFINE_ROUNDS > 0: 上下文管理器.switch = False # 第一局不用上下文
            if 轮次 == 1: 上下文管理器.switch = True # 第二局开上下文
            任务状态列表 = ["Unassigned" for _ in range(len(翻译列表))] # Fail Unassigned Working Retrying Done
            扩散进度条.任务 = 任务状态列表
            扩散进度条.数量 = len(翻译列表)
            扩散进度条.n = 0
            for indexq, index in enumerate(翻译列表): # ↑预分配工作列表长度
                while True:
                    就绪事件 = asyncio.Event()
                    任务 = asyncio.create_task(Self.生成翻译(总条目数, index, 上下文管理器, 请求文本, 请求提示词, 使用模型, 就绪事件, indexq, 优先分配列表, 任务状态列表))
                    await 就绪事件.wait()
                    if 任务.done(): continue # ←↑检查生成翻译有没有第一时间获取到会话
                    else:
                        工作列表.append([任务, len(index)])
                        break
            try:
                工作返回列表 = await asyncio.gather(*(i[0] for i in 工作列表))
            except:
                Self.日志("log.core.translator.error", info_level=2, e=eb.format_exc())
            扩散进度条.refresh() # 渲染最终状态 防止取消前的竞态残留 Retrying/Working 快照
            工作列表.clear() # 完全下班 如有循环文则仅上下文共享
            if not 轮次 == Self.Config.TRANSLATOR_REFINE_ROUNDS: 进度条.n = 0 # 不是最后一轮就清零 等待进度条下一次复用
        进度条任务.cancel() # 进度条下班
        扩散进度条任务.cancel()
        
        # ↓重组返回映射
        for index0 in 工作返回列表: # 格式: [生成翻译返回值, ...]
            if index0 is None: continue # 去除 None 异常返回
            for index1, index2 in index0.items(): # index0={原文: 译文, ...} ↓有无翻译对照
                缓存映射[index1] = index2
                返回映射[index1] = Self.Config.TRANSLATOR_ORIGINAL_REFERENCE_FORMAT.format(o=翻译映射[index1], t=index2) if Self.Config.TRANSLATOR_ORIGINAL_REFERENCE else index2
                
        # ↓写回翻译缓存
        Self.CacheTranslator.翻译缓存(缓存映射, 语言=Self.Config.LANGUAGE_OUTPUT)

        # ↓重组返回列表
        去翻译字典 = {str(item[0]): item[1] for item in 去翻译列表}
        for index in 输入复制: # 格式: [[Key, 原文, 文件位置], ...]
            if index[1] in 文本组件缓存:
                解析数据, 当前路径映射 = 文本组件缓存[index[1]]
                目标对象 = copy.deepcopy(解析数据) 
                for 路径, 原文本 in 当前路径映射:
                    译文 = 返回映射.get(原文本, 原文本)
                    当前对象 = 目标对象
                    for p in 路径[:-1]:
                        当前对象 = 当前对象[p]
                    当前对象[路径[-1]] = 译文
                重组列表.append([index[0], json.dumps(目标对象, ensure_ascii=False), index[2]])
            else:
                译文 = 返回映射.get(index[1])
                if 译文 is None:
                    译文 = 去翻译字典.get(str(index[0]), index[1])
                重组列表.append([index[0], 译文, index[2]])
        
        return 重组列表
            

    def 翻译语言文件(Self, file0: str,  file1: str="", 索引ID: str=uuid.uuid4().hex, output_path: str = "", export_inspection: bool = False, output_lang_str: bool = False, read_error: bool = True, 使用模型: set = set()):
        output_path = Self.Module.输出路径处理(output_path)
        输出列表 = []
        if 使用模型 == None: 使用模型 = set()
        可翻译源文件, 源文件, 参考文件, 压缩路径, 输出扩展名, file2 = Self.File.读取资源文件(file0, file1, read_error)
        翻译列表 = asyncio.run(Self.翻译语言列表(可翻译源文件, 参考文件, 使用模型, 索引ID)) #翻译核心
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
                try:
                    压缩源 = file0 if Path(file0).suffix.lower() in {".zip", ".jar"} else file1
                    if 压缩源 and Path(压缩源).is_file():
                        with zipfile.ZipFile(压缩源, 'r') as 手册压缩包:
                            for 内部文件 in 手册压缩包.namelist():
                                if "/patchouli_books/" in f"/{内部文件}" and not 内部文件.endswith('/'):
                                    if not (压缩文件夹Path / 内部文件).exists():
                                        手册压缩包.extract(内部文件, 压缩路径)
                        for 手册根 in 压缩文件夹Path.glob("assets/*/patchouli_books"):
                            if 手册根.is_dir():
                                Self.翻译帕秋莉手册语言版本(str(手册根), 索引ID=索引ID)
                except Exception:
                    Self.日志("log.module.book.load.error", file=str(file0), e=eb.format_exc(), info_level=1)
                if file2[0] == False:
                    文档内容 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(
                        name=Path(file0).stem,
                        lang=Self.Config.LANGUAGE_OUTPUT,
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
        try:
            for 条目 in 翻译列表:
                if 过滤方法(条目):
                    过滤后.append(条目)
        except Exception:
            Self.日志(f"log.module.{日志类型}.clean.error", index=条目, e=eb.format_exc(), info_level=2)
        待翻译 = [[条目[0], 条目[1], 条目[2] if len(条目) > 2 else ""] for 条目 in 过滤后]
        使用模型 = set()
        翻译函数黑名单 = {"传入使用模型"}
        翻译参数 = {k: v for k, v in 总参数.items() if not k in 翻译函数黑名单}
        保存参数 = {"使用模型": 使用模型} if "传入使用模型" in 总参数 else {}
        翻译结果 = asyncio.run(Self.翻译语言列表(待翻译, 参考列表, 使用模型=使用模型, **翻译参数))
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
        资源路径 = Path(f"{path}/resources")
        if 资源路径.is_dir():
            for 命名空间目录 in 资源路径.iterdir():
                if not 命名空间目录.is_dir(): continue
                语言目录 = 命名空间目录 / "lang"
                if not 语言目录.is_dir(): continue
                for 语言文件 in 语言目录.iterdir():
                    if 语言文件.is_file() and 语言文件.name.lower() == f"{Self.Config.LANGUAGE_INPUT}.lang".lower():
                        Self.翻译语言文件(file0=str(语言文件), output_path=str(语言目录), **参数)
                        break
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
    def 翻译帕秋莉手册语言版本(Self, patchouli根目录, 索引ID=None):
        根 = Path(patchouli根目录)
        if not 根.is_dir():
            return
        源语言 = Self.Config.LANGUAGE_INPUT.lower()
        目标语言 = Self.Config.LANGUAGE_OUTPUT.lower()
        if 源语言 == 目标语言:
            return
        额外参数 = {}
        if 索引ID is not None: 额外参数["索引ID"] = 索引ID
        for 书籍目录 in 根.iterdir():
            if not 书籍目录.is_dir():
                continue
            源目录 = next((d for d in 书籍目录.iterdir() if d.is_dir() and d.name.lower() == 源语言), None)
            if 源目录 is None:
                continue
            目标目录 = 书籍目录 / 目标语言
            try:
                if 目标目录.exists():
                    shutil.rmtree(目标目录, ignore_errors=True)
                shutil.copytree(源目录, 目标目录)
            except Exception:
                Self.日志("log.module.book.load.error", file=str(源目录), e=eb.format_exc(), info_level=1)
                continue
            try:
                Self.翻译帕秋莉手册(path=str(目标目录), **额外参数)
            except Exception:
                Self.日志("log.module.book.write.error", file=str(目标目录), path="", e=eb.format_exc(), info_level=1)
    def 翻译数据包(Self, path, path2=None, **参数):
        path = Path(path)
        if path.is_file():
            缓存文件夹 = Path(f"{Self.Config.PATH_CACHE}/{uuid.uuid4().hex}/")
            with zipfile.ZipFile(path, 'r') as zf: zf.extractall(缓存文件夹)
        else: 缓存文件夹 = path
        Self.翻译流程(缓存文件夹, ["*.json", "*.mcmeta", "*.mcfunction"], Self.File.读取单个数据包文件, Self.Module.过滤键文本, lambda x: x[0][0], Self.File.应用数据包翻译, Self.Config.DATA_READ_MAX_CONCURRENT, Self.Config.DATA_WRITE_MAX_CONCURRENT, "data", path2=path2, **参数)
    def 翻译未知伤亡语言文件(Self, path, path2=None, **参数):
        Self.翻译流程(path, ["*.json"], Self.File.读取未知伤亡语言文件, Self.Module.过滤键文本, lambda x: x[2], Self.File.保存未知伤亡语言文件, Self.Config.LANG_READ_MAX_CONCURRENT, Self.Config.LANG_WRITE_MAX_CONCURRENT, "lang", 输出方法=lambda p: p.parent / f"{Self.Config.LANGUAGE_OUTPUT}.json", 传入使用模型=True, path2=path2, **参数)
    def 翻译未知伤亡dll模组(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.dll", Self.File.读取单个DLL文件, Self.Module.过滤DLL文本, lambda x: x[0][0], Self.File.应用DLL翻译, Self.Config.DLL_READ_MAX_CONCURRENT, Self.Config.DLL_WRITE_MAX_CONCURRENT, "dll", path2=path2, **参数)
    def 翻译MMT_JSON文件(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.json", Self.File.读取单个MMT文件, Self.Module.过滤键文本, lambda x: x[0][0], Self.File.应用MMT翻译, Self.Config.LANG_READ_MAX_CONCURRENT, Self.Config.LANG_WRITE_MAX_CONCURRENT, "mmt", path2=path2, **参数)
    def 翻译MMT_TXT文件(Self, path, path2=None, **参数):
        Self.翻译流程(path, "*.txt", Self.File.读取单个MMT_TXT文件, lambda x: x[1] and x[1].strip(), lambda x: x[0][0], Self.File.应用MMT_TXT翻译, Self.Config.LANG_READ_MAX_CONCURRENT, Self.Config.LANG_WRITE_MAX_CONCURRENT, "mmt", path2=path2, **参数)
    def 翻译整合包(Self, path: str, all_mode: bool = False):
        翻译列表路径 = {}
        使用模型 = set()
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
                文档内容 = Self.Config.PACK_META_TEMPLATE_TRANSLATE.format(name="", lang=Self.Config.LANGUAGE_OUTPUT, model=", ".join(m for m in (使用模型 or []) if m and m != "null") or Self.Lang("log.core.package.zip.hit"), author=Self.Config.PACK_AUTHOR or "海盐青茫")
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
                           Self.翻译MMT_JSON文件(path=file0, path2=file1)
                           返回内容 = Path(file0)
                       elif "name" in 文件数据:
                           Self.日志("log.core.translator.general.model", model="Casualties: Unknown Language File", info_level=0)
                           Self.翻译未知伤亡语言文件(path=file0, path2=file1)
                           返回内容 = Path(file0).parent / f"{Self.Config.LANGUAGE_OUTPUT}.json" # 返回翻译输出而非源文件
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
                elif 文件0扩展名 == ".txt":
                    try:
                        with open(file0, "r", encoding="utf-8") as _f:
                             _头 = _f.read(200)
                        if "### FILE_NAME ###" in _头 or "### AI_PROMPT ###" in _头:
                            Self.日志("log.core.translator.general.model", model="Mine Mod Translator Mod TXT", info_level=0)
                            Self.翻译MMT_TXT文件(path=file0, path2=file1)
                            Self.日志("log.core.translator.succeed", path=Path(file0).resolve(), info_level=0)
                            返回内容 = Path(file0)
                    except Exception:
                        pass
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
                            if 根前缀.rstrip('/') == 目标文件夹名.rstrip('/'):
                                return True
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
                                    返回内容 = Path(Self.Config.LOGS_FILE_PATH) / Self.Config.LOGS_FILE_NAME
                            else:
                                Self.日志("log.core.translator.general.structure.unknown", info_level=3)
                                返回内容 = Path(Self.Config.LOGS_FILE_PATH) / Self.Config.LOGS_FILE_NAME
                else:
                    Self.日志("log.core.translator.general.structure.unknown", info_level=3)
                    返回内容 = Path(Self.Config.LOGS_FILE_PATH) / Self.Config.LOGS_FILE_NAME
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
            返回内容 = Path(file0)
        if 返回内容 is None:
            返回内容 = Path(file0)
        Self.日志("log.core.translator.succeed", path=返回内容.resolve(), info_level=0)
        return 返回内容.resolve()
        
        
测试 = True
if __name__ == "__main__" and 测试:
    参数 = {
        "LLM1_API_URL": "",
        "LLM1_API_KEY": "",
        "LLM1_MAX_WORKERS": 50,
        "LLM1_MODEL": "LongCat-2.0",
        "LLM1_API_KWARGS": {"thinking": {"type": "disabled"}},
        #"LLM1_API_KWARGS": {"extra_body": {"thinking": {"type": "disabled"}}},
        "LLM1_MAX_WORKERS": 5,
        "LLM0_API_URL": "http://127.0.0.1:25564/v1/chat/completions",
        "LLM0_MODEL": "hy-mt2-1.8b",
        "LLM0_MAX_WORKERS": 4,
        "LLM2_API_URL": "http://127.0.0.1:25564/v1/chat/completions",
        "LLM2_MODEL": "hy-mt2-1.8b:2",
        "LLM2_MAX_WORKERS": 4,
        "LLM3_API_URL": "http://127.0.0.1:25564/v1/chat/completions",
        "LLM3_MODEL": "hy-mt2-1.8b:3",
        "LLM3_MAX_WORKERS": 4,
        "LLM4_API_URL": "http://127.0.0.1:25564/v1/chat/completions",
        "LLM4_MODEL": "hy-mt2-1.8b:4",
        "LLM4_MAX_WORKERS": 1,
        "LLM1_MIN_COUNT": 10000000,
        "LLM2_MIN_COUNT": 0000000,
        "LLM3_MIN_COUNT": 10000000,
        "LLM4_MIN_COUNT": 10000000,
        #"EMB_API_URL": "http://127.0.0.1:25564/v1/embeddings",
        #"EMB_MODEL": "text-embedding-bge-small-en-v1.5",
        "TRANSLATOR_BATCH": 2.5,
        "TRANSLATOR_CONTEXTS": 3,
        "TRANSLATOR_ORIGINAL_REFERENCE": False,
        "LANGUAGE": "zh_CN",
        "TRANSLATOR_CACHE_NAME": "Translator_Cache",
        "EMB_MAX_WORKERS": 1,
        "DEBUG_MODE": True,
        "TRANSLATOR_CACHE_READ": False,
        "TRANSLATOR_CACHE_WRITE": False,
        "TQDM_DIFF": False,
        #"INDEX_MODE": "GSQMoEPlus",
        #"INDEX_SQ": "GSQ8"
        #"LANGUAGE_OUTPUT": "文言",
    }
    from TranslatorLib import Config
    配置 = Config(参数)
    翻译 = Translator(配置)
    #翻译.翻译BQ任务(r"mods")
    #翻译.翻译ZS脚本(r"C:\Users\FengMang\Desktop\TranslatorMinecraft\mods")
    #翻译.翻译通用文件(r"C:\Users\FengMang\Downloads\ansiblecrafting-1.0.0-beta-mc1.20.1-fabric.jar")
    #翻译.翻译CMM菜单(r"mods")
    #翻译.翻译FM菜单(r"fancymenu")
    #翻译.翻译HQM任务(r"hqm")
    翻译.翻译通用文件(r"C:\Users\FengMang\AppData\Roaming\PrismLauncher\instances\supersymmetry-0.1.15.2\minecraft\config\betterquesting\resources\supersymmetry\lang\en_us.lang")
    #翻译.翻译通用文件(r"en_us.lang")
    #翻译.翻译未知伤亡语言文件(r"E:\SteamLibrary\steamapps\common\Casualties Unknown Demo\CasualtiesUnknown_Data\Lang\ZH.json", r"E:\SteamLibrary\steamapps\common\Casualties Unknown Demo\CasualtiesUnknown_Data\Lang\EN.json")
    #翻译.翻译未知伤亡dll模组(r"翻译")
