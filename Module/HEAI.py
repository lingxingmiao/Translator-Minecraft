from TranslatorLib import (faiss, FastAPI, HTTPException, uvicorn, np, fastapi, json,
                           Config, TranslatorPersistence, GPU_ACC, eb)

class HEAI: # Just Enough Artificial Intelligence 移植 Had Enough Artificial Intelligence
    def __init__(Self, App: Config):
        Self.Config       = App.Config
        Self.tqdm         = App.RichTqdm
        Self.日志         = App.日志
        Self.Builder      = App.Builder
        Self.Quantization = App.Quantization
        Self.Index        = App.Index
        Self.向量索引     = None
        Self.文本文件     = None
        Self.向量文件     = None
    async def 添加(Self, 物品列表):
        if 物品列表 and isinstance(物品列表[0], str):
            物品列表 = [[名称, 名称, ""] for 名称 in 物品列表]
        if 物品列表 and isinstance(物品列表[0], list):
            物品列表 = [[f"{名称[0]} | {名称[1]}" if len(名称) > 1 else 名称[0], 名称[0], ""] for 名称 in 物品列表]
        Self.向量文件, Self.文本文件 = await TranslatorPersistence.参考词预处理(Self, 物品列表, 查询=False)
        Self.向量索引 = TranslatorPersistence.缓存索引(Self, 向量文件=Self.向量文件, 文本文件=Self.文本文件)
    async def 检索(Self, 查询列表):
        if Self.文本文件 is None and Self.向量文件 is None:
            Self.向量文件, Self.文本文件 = await TranslatorPersistence.参考词预处理(Self, 查询=False)
        if Self.向量文件 is None:
            return [[] for _ in 查询列表]
        返回列表 = [[] for _ in range(len(查询列表))]
        输入列表 = await Self.Builder.并行生成向量([[查询词, "", ""] for 查询词 in 查询列表], 查询=True)
        向量列表 = np.asarray(输入列表[0], dtype=np.float32)
        Self.Quantization.PCA应用懒加载(向量列表, Self.向量文件)
        Self.Quantization.TT应用懒加载(向量列表, Self.向量文件)
        向量列表 = 向量列表.get() if GPU_ACC else 向量列表
        faiss.normalize_L2(向量列表)
        if Self.向量索引 is None:
            Self.向量索引 = TranslatorPersistence.缓存索引(Self, 向量文件=Self.向量文件, 文本文件=Self.文本文件)
        for _ in Self.tqdm(range(1), desc="tqdm.index.search"):
            索引结果矩阵 = Self.向量索引.search(向量列表, max(Self.Config.INDEX_TEXT_K, 64))[1]
        for index0 in range(len(向量列表)):
            返回列表[index0] = [Self.文本文件[i][1] for i in 索引结果矩阵[index0] if i >= 0]
        return 返回列表

 
app = FastAPI()
@app.post("/add")
async def 添加索引(请求: fastapi.Request):
    try:
        负载 = json.loads((await 请求.body()).decode("utf-8"))
        数据 = 负载.get("data", [])
        await heai.添加(数据)
        return {"status": "ok", "message": "索引构建成功"}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的 JSON 格式")
    except Exception:
        eb.print_exc()
        raise HTTPException(status_code=500, detail="索引构建失败")
@app.post("/search")
async def 语义检索(请求: fastapi.Request):
    try:
        负载 = json.loads((await 请求.body()).decode("utf-8"))
        数据 = 负载.get("data", [])
        结果 = await heai.检索(数据)
        def 转原生(对象):
            if hasattr(对象, 'tolist'): return 对象.tolist()
            if isinstance(对象, list): return [转原生(项) for 项 in 对象]
            return 对象
 
        return {"status": "ok", "result": 转原生(结果)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的 JSON 格式")
    except Exception:
        eb.print_exc()
        raise HTTPException(status_code=500, detail="语义检索失败")
 
if __name__ == "__main__":
    配置覆盖 = {"VEC_FILE_NAME": "HEAI", "VEC_RERANKER": False,
            "EMB_MODEL": "intfloat/multilingual-e5-large", # 在此处更换模型，当前教程环境支持的模型见 https://deepwiki.com/qdrant/fastembed/6-supported-models
            "INDEX_MODE": ["IP"],
            "INDEX_TEXT_K": 32, # 搜索数量
            "EMB_PROMPT_NAME": ["passage: {t}", "query: {t}"],
            "EMB_TOKENSTOTEXT_RATIO": 1.0
            } # 更多配置/功能见 TranslatorConfig.py 文件
    配置 = Config(配置覆盖)
    heai = HEAI(配置)
    print("HEAI服务器运行，监听端口 27865...")
    uvicorn.run(app, host="127.0.0.1", port=27865)