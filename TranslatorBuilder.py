from TranslatorLib import (np, threading, ThreadPoolExecutor, as_completed, time, eb,
                           Log, Locale, TranslatorPersistence, RuntimeConfig)

class Builder:
    def __init__(Self, Config: dict = None):
        Config = Config or {}
        Self.Config = RuntimeConfig(**Config)
        Self.日志 = Log(Config=Config).写入日志
        Self.Locale = Locale(Config=Config)
        Self.Lang = Self.Locale.Lang
        Self.tqdm = Self.Locale.Tqdm
        Self.线程锁 = threading.Lock()
        Self.嵌入模型 = None
        Self.重排序模型 = None
    def 生成向量(Self, text: list, outputs: list = None, outputs1: list = None):
        重试次数 = 0
        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            with Self.线程锁:
                try:
                    额外参数 = {}
                    if Self.Config.EMB_REASONING_FRAME.lower() == "sentencetransformer":
                        if Self.Config.EMB_ENCODE_PROMPT_NAME:
                            额外参数["prompt_name"] = Self.Config.EMB_ENCODE_PROMPT_NAME
                        if Self.Config.EMB_MODEL_NORMALIZE:
                            额外参数["normalize_embeddings"] = Self.Config.EMB_MODEL_NORMALIZE
                        向量列表 = Self.嵌入模型.encode(text, **额外参数)
                    elif Self.Config.EMB_REASONING_FRAME.lower() == "fastembed":
                        向量列表 = list(Self.嵌入模型.embed(text))
                    向量列表 = np.asarray(向量列表, dtype=np.float32)
                    if Self.Config.VEC_DIM_CLIP != -1:
                        向量列表 = 向量列表[:, :Self.Config.VEC_DIM_CLIP]
                    return [向量列表, [text, outputs, outputs1]]
                except Exception:
                    Self.日志("log.core.locally.generate.vectors.error", e=eb.format_exc(), info_level=2)
                    return [None, [text, outputs, outputs1]]
        else:
            会话 = TranslatorPersistence.获取会话(Self.Config.EMB_API_URL, Self.Config.EMB_API_KEY, Self.Config.EMB_MODEL, Self.Config.EMB_MAX_WORKERS, Self.Config.EMB_RETRY_COEF, Self.Config.EMB_MAX_RETRY)
            while 重试次数 < Self.Config.EMB_MAX_RETRY:
                try:
                    请求结果 = 会话.post(url=Self.Config.EMB_API_URL, json={"input": text,"model": Self.Config.EMB_MODEL}, timeout=(Self.Config.EMB_CONN_TIMEOUT, Self.Config.EMB_TIMEOUT))
                    请求结果.raise_for_status()
                    请求结果 = 请求结果.json()
                    向量列表 = []
                    for index in range(len(text)):
                        向量列表.append(请求结果['data'][index]['embedding'])
                    向量列表 = np.asarray(向量列表, dtype=np.float32)
                    if Self.Config.VEC_DIM_CLIP != -1:
                        向量列表 = 向量列表[:, :Self.Config.VEC_DIM_CLIP]
                    return [向量列表, [text, outputs, outputs1]]
                except Exception:
                    重试次数 += 1
                    if 重试次数 >= Self.Config.EMB_MAX_RETRY:
                        Self.日志("log.core.api.generate.vectors.error", e=eb.format_exc(), info_level=3)
                        return [None, [text, outputs, outputs1]]
                    else:
                        Self.日志("log.core.api.generate.vectors.retry", e=eb.format_exc(), info_level=2)
                        time.sleep((Self.Config.EMB_RETRY_COEF ** (重试次数 - 1)) * Self.Config.EMB_RETRY_TIME)
    def 并行生成向量(Self, texts: list, use_cache: bool = True) -> list:
        Self.日志("log.core.vector.generate.start", info_level=0)
        if not texts:
            Self.日志("log.core.generated.vector.nan", texts=texts, info_level=3)
            return [np.array([], dtype=np.float32).reshape(0, 0), [[], [], []]]

        if use_cache:
            缓存命中, 待生成文本 = TranslatorPersistence.查询向量缓存(texts)
        else:
            缓存命中, 待生成文本 = {}, texts
        if 缓存命中:
            命中数 = len(缓存命中); 总数 = len(texts)
            Self.日志("log.core.vector.cache.hit", rate=f"{命中数/总数:.4%}", hit=命中数, total=总数, info_level=0)

        if not 待生成文本:
            维度 = next(iter(缓存命中.values())).shape[0]
            合并向量 = np.empty((len(texts), 维度), dtype=np.float32)
            合并请求文本, 合并额外返回, 合并额外返回1 = [], [], []
            for item in texts:
                合并向量[len(合并请求文本)] = 缓存命中[item[0]]
                合并请求文本.append(item[0])
                合并额外返回.append(item[1])
                合并额外返回1.append(item[2])
            Self.日志("log.core.vector.generate.end", info_level=0)
            return [合并向量, [合并请求文本, 合并额外返回, 合并额外返回1]]

        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            Self.嵌入模型 = TranslatorPersistence.获取嵌入模型(Self=Self)
        最大字符数 = Self.Config.EMB_MAX_TOKENS * Self.Config.EMB_TOKENSTOTEXT_RATIO
        分组结果, 当前组, 当前总长 = [], [], 0.0
        for index in 待生成文本:
            长度 = len(index[0])
            if 当前总长 + 长度 > 最大字符数:
                分组结果.append(当前组)
                当前组, 当前总长 = [], 0.0
            当前组.append(index)
            当前总长 += 长度
        if 当前组: 分组结果.append(当前组)
        待处理文本列表原文 = [[item[0] for item in group] for group in 分组结果]
        待处理文本列表额外输出 = [[item[1] for item in group] for group in 分组结果]
        待处理文本列表额外输出1 = [[item[2] for item in group] for group in 分组结果]
        总待生成条数 = sum(len(g) for g in 分组结果)
        合并向量 = None
        维度 = None
        当前偏移 = 0
        偏移锁 = threading.Lock()
        提前完成缓存 = []
        合并请求文本 = []
        合并额外返回 = []
        合并额外返回1 = []
        新增缓存条目 = {}
        with ThreadPoolExecutor(max_workers=Self.Config.EMB_MAX_WORKERS) as 执行器:
            未来任务映射 = {
                执行器.submit(Self.生成向量, 原文组, 额外输出, 额外输出1): 原文组
                for 原文组, 额外输出, 额外输出1 in zip(待处理文本列表原文, 待处理文本列表额外输出, 待处理文本列表额外输出1)
            }
            for 单个任务 in Self.tqdm(as_completed(未来任务映射), total=len(未来任务映射), desc="tqdm.vectors.generate"):
                结果 = 单个任务.result()
                if 结果[0] is None: continue
                块 = 结果[0]
                块长度 = 块.shape[0]
                for i, 原文 in enumerate(结果[1][0]):
                    新增缓存条目[原文] = 块[i].copy()
                with 偏移锁:
                    if 维度 is None:
                        维度 = 块.shape[1]
                        合并向量 = np.empty((总待生成条数, 维度), dtype=np.float32)
                        for 缓存块, 缓存元数据 in 提前完成缓存:
                            合并向量[当前偏移:当前偏移+缓存块.shape[0]] = 缓存块
                            当前偏移 += 缓存块.shape[0]
                            合并请求文本.extend(缓存元数据[0])
                            合并额外返回.extend(缓存元数据[1])
                            合并额外返回1.extend(缓存元数据[2])
                        提前完成缓存.clear()
                    elif 块.shape[1] != 维度:
                        continue
                    合并向量[当前偏移:当前偏移+块长度] = 块
                    当前偏移 += 块长度
                合并请求文本.extend(结果[1][0])
                合并额外返回.extend(结果[1][1])
                合并额外返回1.extend(结果[1][2])
        最终生成向量 = 合并向量[:当前偏移] if 合并向量 is not None else np.array([], dtype=np.float32).reshape(0, 维度 or 0)
        if 新增缓存条目 and use_cache:
            # 仅更新内存缓存并标脏，真正写盘交由后台定时线程节流批量执行，避免并发生成时频繁全量读写
            TranslatorPersistence.更新向量缓存(新增缓存条目)
        总条数 = len(texts)
        全量向量 = np.empty((总条数, 维度 or 0), dtype=np.float32) if 维度 else np.array([], dtype=np.float32).reshape(0, 0)
        全量文本, 全量额外, 全量额外1 = [], [], []
        生成索引 = 0
        for item in texts:
            if item[0] in 缓存命中:
                全量向量[len(全量文本)] = 缓存命中[item[0]]
            else:
                全量向量[len(全量文本)] = 最终生成向量[生成索引]
                生成索引 += 1
            全量文本.append(item[0])
            全量额外.append(item[1])
            全量额外1.append(item[2])
        Self.日志("log.core.vector.generate.end", info_level=0)
        return [全量向量, [全量文本, 全量额外, 全量额外1]]
    def 选择相似度最高译文(Self, 请求消息: list):
        请求次数 = 0
        if (not Self.Config.RERANKER_API_URL) and (Self.Config.RERANKER_MODEL):
            with Self.线程锁:
                try:
                    相似度 = Self.重排序模型.predict([(请求消息[0], 候选) for 候选 in 请求消息[1]], show_progress_bar=False)
                    return [请求消息[0], 请求消息[1][相似度.argmax()], 相似度]
                except Exception:
                    Self.日志("log.core.translator.cache.locally.error", e=eb.format_exc(), info_level=2)
                    return [请求消息[0], 请求消息[1][0], [0 for _ in range(len(请求消息[1]))]]
        else:
            from TranslatorLib import TranslatorPersistence
            请求内容 = {
                "model": Self.Config.RERANKER_MODEL,
                "documents": 请求消息[1],
                "query": 请求消息[0],
                "instruct": Self.Config.RERANKER_INSTRUCT
            }
            会话 = TranslatorPersistence.获取会话(Self.Config.RERANKER_API_URL, Self.Config.RERANKER_API_KEY, Self.Config.RERANKER_MODEL, Self.Config.RERANKER_MAX_WORKERS, Self.Config.RERANKER_RETRY_COEF, Self.Config.RERANKER_MAX_RETRY)
            while 请求次数 < Self.Config.RERANKER_MAX_RETRY:
                try:
                    相似度 = []
                    请求结果 = 会话.post(url=Self.Config.RERANKER_API_URL, json=请求内容, timeout=(Self.Config.RERANKER_CONN_TIMEOUT, Self.Config.RERANKER_TIMEOUT))
                    请求结果.raise_for_status()
                    请求结果 = 请求结果.json()
                    请求结果 = 请求结果["output"]["results"]
                    for index in 请求结果:
                        相似度.append(index["document"]["text"])
                    return [请求消息[0], 请求结果[0]["document"]["text"], 相似度]
                except Exception:
                    Self.日志("log.core.translator.cache.generate.messages.error", messages=请求消息[1], e=eb.format_exc(), info_level=1)
                    请求次数 += 1
                    if 请求次数 >= Self.Config.RERANKER_MAX_RETRY:
                        Self.日志("log.core.translator.cache.generate.error", e=eb.format_exc(), output=请求结果, info_level=2)
                        return [请求消息[0], 请求消息[1][0], [0 for _ in range(len(请求消息[1]))]]
                    else:
                        Self.日志("log.core.translator.cache.generate.retry", e=eb.format_exc(), output=请求结果, info_level=1)
                        time.sleep((Self.Config.RERANKER_RETRY_COEF ** (请求次数 - 1)) * Self.Config.RERANKER_RETRY_TIME)
    def 获取相似度最高译文(Self, 输入字典: dict, 强制重排: bool=False):
        请求列表 = []
        剔除列表 = []
        返回列表 = []
        for index in 输入字典:
            if len(输入字典[index]) == 1 and 强制重排 == False:
                剔除列表.append([index, 输入字典[index][0], [0]])
            else:
                请求列表.append([index, 输入字典[index]])
        if (not Self.重排序模型) and (not Self.Config.RERANKER_API_URL) and (Self.Config.RERANKER_MODEL) and (请求列表):
            Self.重排序模型 = TranslatorPersistence.获取重排模型(Self=Self)
        Self.日志("log.core.translator.cache.generate.start", item=len(请求列表), info_level=0)
        with ThreadPoolExecutor(max_workers=Self.Config.RERANKER_MAX_WORKERS) as 执行器:
            未来任务映射 = {
                执行器.submit(
                    Self.选择相似度最高译文,
                    请求消息 = index,
                ): index
                for index in 请求列表
            }
            for 单个任务 in Self.tqdm(as_completed(未来任务映射), total=len(未来任务映射), desc="tqdm.translator.cache.generate"):
                返回列表.append(单个任务.result())
        Self.日志("log.core.translator.cache.generate.end", info_level=0)
        返回列表 += 剔除列表
        return 返回列表
