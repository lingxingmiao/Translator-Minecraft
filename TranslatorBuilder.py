from TranslatorLib import (np, eb, asyncio, random, aiohttp, Image, io, base64, re,
                           TranslatorPersistence, Config)

class Builder:
    def __init__(Self, App: Config):
        Self.Config      = App.Config
        Self.日志         = App.日志
        Self.Network     = App.Network
        Self.Lang        = App.Lang
        Self.tqdm        = App.RichTqdm
        Self.CacheVector = App.CacheVector
        Self.DiffTqdm    = App.DiffTqdm
        Self.Module      = App.Module
        Self.嵌入模型     = None
        Self.图像嵌入模型 = None
        Self.重排序模型   = None
        Self.媒体标记     = None
    async def 生成向量(Self, 原文: list, 会话: aiohttp.ClientSession, 层级: dict, 工作ID: str, 查询: bool):
        
        重试次数 = 0
        响应值 = None
        向量列表 = []
        
        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            try:
                额外参数 = {}
                if Self.Config.EMB_REASONING_FRAME.lower() == "sentencetransformer":
                    if Self.Config.EMB_ENCODE_PROMPT_NAME:
                        额外参数["prompt_name"] = Self.Config.EMB_ENCODE_PROMPT_NAME
                    if Self.Config.EMB_MODEL_NORMALIZE:
                        额外参数["normalize_embeddings"] = Self.Config.EMB_MODEL_NORMALIZE
                    向量列表 = await asyncio.to_thread(Self.嵌入模型.encode, [Self.Config.EMB_PROMPT_NAME[查询].format(t=i) for i in 原文], **额外参数)
                elif Self.Config.EMB_REASONING_FRAME.lower() == "fastembed":
                    向量列表 = await asyncio.to_thread(lambda: list(Self.嵌入模型.embed([Self.Config.EMB_PROMPT_NAME[查询].format(t=i) for i in 原文])))
                向量列表 = np.asarray(向量列表, dtype=np.float32)
                if Self.Config.VEC_DIM_CLIP != -1:
                    向量列表 = 向量列表[:, :Self.Config.VEC_DIM_CLIP]
                Self.Network.close_emb(工作ID)
                return (向量列表, 原文)
            except Exception:
                Self.日志("log.core.locally.generate.vectors.error", e=eb.format_exc(), info_level=2)
                Self.Network.close_emb(工作ID)
                return (None, 原文)
        else:
            
            Json数据 = {
                "input": [Self.Config.EMB_PROMPT_NAME[查询].format(t=i) for i in 原文],
                "model": 层级["model"]
            }
            Json数据.update(层级["api_kwargs"])
            
            while True:
                try:
                    async with 会话.post(url=层级["url"], json=Json数据, timeout=aiohttp.ClientTimeout(connect=层级["conn_timeout"], sock_read=层级["timeout"])) as 响应体:
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
                    for index in range(len(原文)):
                        向量列表.append(响应值['data'][index]['embedding'])
                    向量列表 = np.asarray(向量列表, dtype=np.float32)
                    if Self.Config.VEC_DIM_CLIP != -1:
                        向量列表 = 向量列表[:, :Self.Config.VEC_DIM_CLIP]
                    Self.Network.close_emb(工作ID)
                    return (向量列表, 原文)
                except:
                    重试次数 += 1
                    if 重试次数 >= 层级["max_retry"]:
                        Self.日志("log.core.api.generate.vectors.error", e=eb.format_exc(), info_level=3)
                        Self.Network.close_emb(工作ID)
                        return (None, 原文)
                    else:
                        Self.日志("log.core.api.generate.vectors.retry", e=eb.format_exc(), info_level=2)
                        基础等待 = (层级["retry_coef"] ** (重试次数 - 1)) * 层级["retry_time"]
                        await asyncio.sleep(基础等待 + random.uniform(0, 基础等待 * 0.3))
                        
    async def 并行生成向量(Self, texts: list, use_cache: bool = True, 查询: bool = False) -> list:
        工作列表, 分组结果, 当前组 = [], [], []
        缓存映射, 唯一待生成 = {}, {}
        当前总长 = 0.0
        生成维度, 缓存维度 = 384, 0
        
        文本数量 = len(texts)
        if 文本数量 == 0:
            Self.日志("log.core.generated.vector.nan", texts=texts, info_level=3)
            return [np.array([], dtype=np.float32).reshape(0, 0), [[], [], []]]

        # ↓预分配加速
        最终返回向量 = None
        最终返回文本 = [None]  * 文本数量
        最终返回附加A = [None] * 文本数量
        最终返回附加B = [None] * 文本数量
        
        # 构建文本索引映射
        文本索引映射 = {}
        for i, item in enumerate(texts):
            文本索引映射.setdefault(item[0], []).append(i)

        # ↓查询缓存
        缓存命中, 待生成文本 = Self.CacheVector.查询向量缓存(texts) if use_cache else ({}, texts)
        
        if 缓存命中:
            命中数 = len(缓存命中)
            Self.日志("log.core.vector.cache.hit", rate=f"{命中数/文本数量:.4%}", hit=命中数, total=文本数量, info_level=0)
            缓存维度 = next(iter(缓存命中.values())).shape[0]
            最终返回向量 = np.empty((文本数量, 缓存维度), dtype=np.float32) 
            for index0, index1 in enumerate(texts):
                原文 = index1[0]
                if 原文 in 缓存命中:
                    最终返回向量[index0]  = 缓存命中[原文]
                    最终返回文本[index0]  = index1[0]
                    最终返回附加A[index0] = index1[1]
                    最终返回附加B[index0] = index1[2]

        # ↓全部命中返回
        if not 待生成文本:
            return [最终返回向量, [最终返回文本, 最终返回附加A, 最终返回附加B]]

        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            Self.嵌入模型 = TranslatorPersistence.获取嵌入模型(Self=Self)

        # ↓Token缩放分组
        最大字符数 = Self.Config.EMB_MAX_TOKENS * Self.Config.EMB_TOKENSTOTEXT_RATIO
        for item in 待生成文本:
            if item[0] not in 缓存命中:
                唯一待生成[item[0]] = item
        for index in 唯一待生成.values():
            长度 = len(index[0])
            if 当前总长 + 长度 > 最大字符数:
                分组结果.append(当前组)
                当前组, 当前总长 = [], 0.0
            当前组.append(index)
            当前总长 += 长度
        if 当前组: 分组结果.append(当前组)
        总条目数 = sum(len(g) for g in 分组结果)
        
        进度条 = Self.tqdm(total=总条目数, desc="tqdm.vectors.generate")
        async def 管理进度条():
            while 进度条.n < 总条目数: #↓if防止 工作列表.clear() 与 进度条任务.cancel() 之间的微小间隔导致进度条显示0
                if 工作列表: 进度条.n = sum(i[1] for i in 工作列表 if i[0].done()); 进度条.refresh() # 更新进度条并刷新
                await asyncio.sleep(1/Self.Config.TQDM_FPS) # 没有完成等待刷新
        
        Self.日志("log.core.vector.generate.start", info_level=0)
        进度条任务 = asyncio.create_task(管理进度条()) # 刷新进度条
        for index in 分组结果: 
            原文 = [i[0] for i in index]
            while True:
                会话, 模型层级, 工作ID = Self.Network.get_emb()
                if 会话 is not None and 工作ID is not None and 模型层级 is not None: break
                await asyncio.sleep(0) # 别删 删了用不了
            工作列表.append([asyncio.create_task(Self.生成向量(原文, 会话, 模型层级, 工作ID, 查询)), len(index)])

        for 任务 in asyncio.as_completed([t[0] for t in 工作列表]):
            返回值为None = False
            结果 = await 任务
            向量, 原文 = 结果 if isinstance(结果, tuple) and len(结果) == 2 else (结果, [])
            if 向量 is None: 向量, 返回值为None = np.random.randn(len(原文), 生成维度).astype(np.float32), True
            if 最终返回向量 is None:
                生成维度 = 向量.shape[1]
                最终返回向量 = np.empty((文本数量, 生成维度), dtype=np.float32)
            for index0, index1 in enumerate(原文):
                if index1 in 文本索引映射:
                    for index2 in 文本索引映射[index1]:
                        最终返回向量[index2]  = 向量[index0]
                        最终返回文本[index2]  = texts[index2][0]
                        最终返回附加A[index2] = texts[index2][1]
                        最终返回附加B[index2] = texts[index2][2]
                if use_cache and not 返回值为None:
                    缓存映射[index1] = 向量[index0]
        进度条.n = 总条目数; 进度条.refresh()
        进度条任务.cancel() # 进度条下班
        进度条.close()
        if 缓存维度 != 0 and 生成维度 != 缓存维度: Self.日志("log.core.generated.vector.dim.mismatch.err", info_level=3)
        Self.日志("log.core.vector.generate.end", info_level=0)
        
        if 缓存映射 and use_cache: 
            Self.CacheVector.更新向量缓存(缓存映射)
            
        return [最终返回向量, [最终返回文本, 最终返回附加A, 最终返回附加B]]

    @staticmethod
    def 图像转base64(图像, 最大边长: int = 448) -> str:
        图像 = 图像.convert("RGBA")
        if 最大边长 and max(图像.size) > 最大边长:
            图像.thumbnail((最大边长, 最大边长))
        buf = io.BytesIO()
        图像.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    async def 获取媒体标记(Self, 会话: aiohttp.ClientSession, 层级: dict) -> str:
        if Self.媒体标记:
            return Self.媒体标记
        try:
            匹配 = re.match(r"(https?://[^/]+)", 层级["url"])
            props_url = (匹配.group(1) if 匹配 else 层级["url"]) + "/props"
            async with 会话.get(props_url, timeout=aiohttp.ClientTimeout(connect=层级["conn_timeout"], sock_read=层级["timeout"])) as 响应体:
                if 响应体.status >= 400:
                    raise aiohttp.ClientResponseError(
                        request_info=响应体.request_info,
                        history=响应体.history,
                        status=响应体.status,
                        message=Self.Lang("log.core.image.props.error")
                    )
                属性 = await 响应体.json()
            Self.媒体标记 = (属性 or {}).get("media_marker") or "<__media__>"
        except Exception:
            Self.媒体标记 = "<__media__>"
        return Self.媒体标记

    async def 生成图像向量(Self, 图像列表: list, 路径列表: list, 会话: aiohttp.ClientSession, 层级: dict, 工作ID: str) -> tuple:
        重试次数 = 0
        响应值 = None
        向量列表 = []
        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            try:
                if Self.Config.EMB_REASONING_FRAME.lower() == "sentencetransformer":
                    向量列表 = await asyncio.to_thread(Self.图像嵌入模型.encode, 图像列表)
                elif Self.Config.EMB_REASONING_FRAME.lower() == "fastembed":
                    向量列表 = await asyncio.to_thread(lambda: list(Self.图像嵌入模型.embed(图像列表)))
                向量列表 = np.asarray(向量列表, dtype=np.float32)
                if Self.Config.VEC_DIM_CLIP != -1:
                    向量列表 = 向量列表[:, :Self.Config.VEC_DIM_CLIP]
                Self.Network.close_emb(工作ID)
                return (向量列表, 路径列表)
            except Exception:
                Self.日志("log.core.image.locally.generate.vectors.error", e=eb.format_exc(), info_level=2)
                Self.Network.close_emb(工作ID)
                return (None, 路径列表)
        else:
            if not Self.Config.EMB_API_URL:
                Self.日志("log.core.image.no.api.url", info_level=3)
                Self.Network.close_emb(工作ID)
                return (None, 路径列表)
            媒体标记 = await Self.获取媒体标记(会话, 层级)
            try:
                base64列表 = [Self.图像转base64(i, 最大边长=getattr(Self.Config, "EMB_IMG_MAX_SIDE", 448)) for i in 图像列表]
                Json数据 = {
                    "input": {"prompt_string": " ".join([媒体标记] * len(base64列表)), "multimodal_data": base64列表},
                    "model": 层级["model"],
                    "encoding_format": "float",
                }
                Json数据.update(层级["api_kwargs"])
            except Exception:
                Self.日志("log.core.image.convert.error", e=eb.format_exc(), info_level=2)
                Self.Network.close_emb(工作ID)
                return (None, 路径列表)
            while True:
                try:
                    async with 会话.post(url=层级["url"], json=Json数据, timeout=aiohttp.ClientTimeout(connect=层级["conn_timeout"], sock_read=层级["timeout"])) as 响应体:
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
                    数据列表 = 响应值.get('data') or []
                    if len(数据列表) != len(图像列表):
                        raise RuntimeError(Self.Lang(
                            "log.core.image.api.generate.vectors.count.mismatch",
                            响应数=len(数据列表), 图像数=len(图像列表)))
                    向量列表 = [数据列表[index]['embedding'] for index in range(len(数据列表))]
                    向量列表 = np.asarray(向量列表, dtype=np.float32)
                    if Self.Config.VEC_DIM_CLIP != -1:
                        向量列表 = 向量列表[:, :Self.Config.VEC_DIM_CLIP]
                    Self.Network.close_emb(工作ID)
                    return (向量列表, 路径列表)
                except:
                    重试次数 += 1
                    if 重试次数 >= 层级["max_retry"]:
                        Self.日志("log.core.image.api.generate.vectors.error", e=eb.format_exc(), info_level=3)
                        Self.Network.close_emb(工作ID)
                        return (None, 路径列表)
                    else:
                        Self.日志("log.core.image.api.generate.vectors.retry", e=eb.format_exc(), info_level=2)
                        基础等待 = (层级["retry_coef"] ** (重试次数 - 1)) * 层级["retry_time"]
                        await asyncio.sleep(基础等待 + random.uniform(0, 基础等待 * 0.3))

    async def 并行生成图像向量(Self, texts: list, use_cache: bool = True) -> list:
        工作列表, 分组结果, 当前组 = [], [], []
        缓存映射, 唯一待生成 = {}, {}
        当前数量 = 0
        生成维度, 缓存维度 = 384, 0
        文本数量 = len(texts)
        if 文本数量 == 0:
            Self.日志("log.core.image.empty.input", info_level=3)
            return [np.array([], dtype=np.float32).reshape(0, 0), [[], [], []]]

        最终返回向量 = None
        最终返回文本 = [None] * 文本数量
        最终返回附加A = [None] * 文本数量
        最终返回附加B = [None] * 文本数量

        文本索引映射 = {}
        for i, item in enumerate(texts):
            文本索引映射.setdefault(item[1], []).append(i)

        缓存命中, 待生成文本 = Self.CacheVector.查询向量缓存(texts) if use_cache else ({}, texts)
        if 缓存命中:
            命中数 = len(缓存命中)
            Self.日志("log.core.image.cache.hit", rate=f"{命中数/文本数量:.4%}", hit=命中数, total=文本数量, info_level=0)
            缓存维度 = next(iter(缓存命中.values())).shape[0]
            最终返回向量 = np.empty((文本数量, 缓存维度), dtype=np.float32)
            for index0, index1 in enumerate(texts):
                原文 = index1[1]
                if 原文 in 缓存命中:
                    最终返回向量[index0] = 缓存命中[原文]
                    最终返回文本[index0] = index1[0]
                    最终返回附加A[index0] = index1[1]
                    最终返回附加B[index0] = index1[2]

        if not 待生成文本:
            return [最终返回向量, [最终返回文本, 最终返回附加A, 最终返回附加B]]

        if (not Self.Config.EMB_API_URL) and (Self.Config.EMB_MODEL):
            Self.图像嵌入模型 = TranslatorPersistence.获取图像嵌入模型(Self=Self)

        每批最大 = getattr(Self.Config, "EMB_IMG_MAX_BATCH", 8)
        for item in 待生成文本:
            if item[1] not in 缓存命中:
                唯一待生成[item[1]] = item
        for index in 唯一待生成.values():
            if 当前数量 >= 每批最大:
                分组结果.append(当前组)
                当前组, 当前数量 = [], 0
            当前组.append(index)
            当前数量 += 1
        if 当前组: 分组结果.append(当前组)
        总条目数 = sum(len(g) for g in 分组结果)

        进度条 = Self.tqdm(total=总条目数, desc="tqdm.vectors.generate")
        async def 管理进度条():
            while 进度条.n < 总条目数:
                if 工作列表: 进度条.n = sum(i[1] for i in 工作列表 if i[0].done()); 进度条.refresh()
                await asyncio.sleep(1/Self.Config.TQDM_FPS)

        Self.日志("log.core.vector.generate.start", info_level=0)
        进度条任务 = asyncio.create_task(管理进度条())
        for index in 分组结果:
            图像列表 = [i[0] for i in index]
            路径列表 = [i[1] for i in index]
            while True:
                会话, 模型层级, 工作ID = Self.Network.get_emb()
                if 会话 is not None and 工作ID is not None and 模型层级 is not None: break
                await asyncio.sleep(0)
            工作列表.append([asyncio.create_task(Self.生成图像向量(图像列表, 路径列表, 会话, 模型层级, 工作ID)), len(index)])

        for 任务 in asyncio.as_completed([t[0] for t in 工作列表]):
            返回值为None = False
            结果 = await 任务
            向量, 原文 = 结果 if isinstance(结果, tuple) and len(结果) == 2 else (结果, [])
            if 向量 is None: 向量, 返回值为None = np.random.randn(len(原文), 生成维度).astype(np.float32), True
            if 最终返回向量 is None:
                生成维度 = 向量.shape[1]
                最终返回向量 = np.empty((文本数量, 生成维度), dtype=np.float32)
            for index0, index1 in enumerate(原文):
                if index1 in 文本索引映射:
                    for index2 in 文本索引映射[index1]:
                        最终返回向量[index2] = 向量[index0]
                        最终返回文本[index2] = texts[index2][0]
                        最终返回附加A[index2] = texts[index2][1]
                        最终返回附加B[index2] = texts[index2][2]
                if use_cache and not 返回值为None:
                    缓存映射[index1] = 向量[index0]
        进度条.n = 总条目数; 进度条.refresh()
        进度条任务.cancel()
        进度条.close()
        if 缓存维度 != 0 and 生成维度 != 缓存维度: Self.日志("log.core.image.vector.dim.mismatch.err", 缓存维度=缓存维度, 生成维度=生成维度, info_level=3)
        Self.日志("log.core.vector.generate.end", info_level=0)

        if 缓存映射 and use_cache:
            Self.CacheVector.更新向量缓存(缓存映射)

        return [最终返回向量, [最终返回文本, 最终返回附加A, 最终返回附加B]]
    async def 选择相似度最高译文(Self, 请求消息: list):
        请求次数 = 0
        响应值 = None
        工作ID = None
        if (not Self.Config.RERANKER_API_URL) and (Self.Config.RERANKER_MODEL):
            try:
                相似度 = await asyncio.to_thread(Self.重排序模型.predict, [(请求消息[0], 候选) for 候选 in 请求消息[1]], show_progress_bar=False)
                return [请求消息[0], 请求消息[1][相似度.argmax()], 相似度]
            except Exception:
                Self.日志("log.core.translator.cache.locally.error", e=eb.format_exc(), info_level=2)
                return [请求消息[0], 请求消息[1][0], [0 for _ in range(len(请求消息[1]))]]
        else:
            请求内容 = {
                "model": Self.Config.RERANKER_MODEL,
                "documents": 请求消息[1],
                "query": 请求消息[0],
                "instruct": Self.Config.RERANKER_INSTRUCT
            }
            while 请求次数 < Self.Config.RERANKER_MAX_RETRY:
                try:
                    while True:
                        会话, 模型层级, 工作ID = Self.Network.reranker_emb()
                        if 会话 is not None and 工作ID is not None and 模型层级 is not None: break
                        await asyncio.sleep(0) # 别删 删了用不了
                    相似度 = []
                    async with 会话.post(url=模型层级["url"], json=请求内容, timeout=aiohttp.ClientTimeout(connect=模型层级["conn_timeout"], sock_read=模型层级["timeout"])) as 响应体:
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
                    响应值 = 响应值["output"]["results"]
                    for index in 响应值:
                        相似度.append(index["document"]["text"])
                    Self.Network.close_reranker(工作ID)
                    return [请求消息[0], 响应值[0]["document"]["text"], 相似度]
                except Exception:
                    if 工作ID is not None: Self.Network.close_reranker(工作ID)
                    Self.日志("log.core.translator.cache.generate.messages.error", messages=请求消息[1], e=eb.format_exc(), info_level=1)
                    请求次数 += 1
                    if 请求次数 >= Self.Config.RERANKER_MAX_RETRY:
                        Self.日志("log.core.translator.cache.generate.error", e=eb.format_exc(), output=响应值, info_level=2)
                        return [请求消息[0], 请求消息[1][0], [0 for _ in range(len(请求消息[1]))]]
                    else:
                        Self.日志("log.core.translator.cache.generate.retry", e=eb.format_exc(), output=响应值, info_level=1)
                        基础等待 = (Self.Config.RERANKER_RETRY_COEF ** (请求次数 - 1)) * Self.Config.RERANKER_RETRY_TIME
                        await asyncio.sleep(基础等待 + random.uniform(0, 基础等待 * 0.3))
    async def 获取相似度最高译文(Self, 输入字典: dict, 强制重排: bool=False):
        请求列表 = []
        剔除列表 = []
        返回列表 = []
        工作列表 = []
        for index in 输入字典:
            if len(输入字典[index]) == 1 and 强制重排 == False:
                剔除列表.append([index, 输入字典[index][0], [0]])
            else:
                请求列表.append([index, 输入字典[index]])
        if (not Self.重排序模型) and (not Self.Config.RERANKER_API_URL) and (Self.Config.RERANKER_MODEL) and (请求列表):
            Self.重排序模型 = TranslatorPersistence.获取重排模型(Self=Self)
        Self.日志("log.core.translator.cache.generate.start", item=len(请求列表), info_level=0)
        if 请求列表:
            进度条 = Self.tqdm(total=len(请求列表), desc="tqdm.translator.cache.generate")
            async def 管理进度条():
                while 进度条.n < len(请求列表):
                    if 工作列表: 进度条.n = sum(1 for i in 工作列表 if i.done()); 进度条.refresh()
                    await asyncio.sleep(1/Self.Config.TQDM_FPS)
            进度条任务 = asyncio.create_task(管理进度条())
            for index in 请求列表:
                工作列表.append(asyncio.create_task(Self.选择相似度最高译文(请求消息=index)))
            for 单个任务 in asyncio.as_completed(工作列表):
                返回列表.append(await 单个任务)
            进度条.n = len(请求列表); 进度条.refresh() # 强制补渲染最终100% 防止最后一个任务完成瞬间 cancel() 导致进度条停格
            进度条任务.cancel()
            进度条.close()
        Self.日志("log.core.translator.cache.generate.end", info_level=0)
        返回列表 += 剔除列表
        return 返回列表
