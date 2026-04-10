from TranslatorLib import Path as pt, time, os, eb, re, FastMCP, MCPConfig
from TranslatorCore import Translator
mcp = FastMCP("TranslationMinecraft")

配置文件 = {}
def 设置时间():
    时间 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + f"{int((time.time() % 1) * 10000):04d}"
    配置文件['LOGS_FILE_NAME'] = f"logs-{时间}"
def 路径处理(path: str):
    return str(pt(re.sub(r'(\\\\|//|\\)', '/', path)).resolve())
def 目录处理(path: str):
    path = pt(re.sub(r'(\\\\|//|\\)', '/', path)).resolve()
    if path.is_file():
        return path.parent
    elif path.is_dir():
        return path
@mcp.tool()
def SetTranslatorConfig(config_kwargs: dict):
    """
    ### 工具功能
    设置或更新翻译软件的全局配置参数。
    **重要**：此工具具有状态保持性。调用后，后续所有翻译相关工具将立即使用此最新配置。
    建议在进行翻译任务前优先调用此工具以确保环境正确。
    如果用户没有说明请不要随意改变配置。

    ### 输入参数说明
    接收一个字典 (dict)，键 (Key) 为配置项名称，值 (Value) 为对应的设置值。
    仅传递需要修改的参数，未传递的参数通常保持原有设置（取决于后端实现，建议首次调用传递完整配置）。

    ### 配置项详解 (Config Schema)
    
    #### 1. 语言设置 (Language)
    - `LANGUAGE_INPUT` (str): 输入语言代码，默认 "en_us" (例如：zh_cn, ja_jp)
    - `LANGUAGE_OUTPUT` (str): 输出语言代码，默认 "zh_cn"
    - `LANGUAGE` (str): 软件界面及日志语言，默认 "zh_CN"

    #### 2. 大语言模型设置 (LLM)
    - `LLM_API_URL` (str): API 请求地址，默认 "" (必填项之一)
    - `LLM_API_KEY` (str): API 密钥，默认 "" (必填项之一)
    - `LLM_API_KWARGS` (dict): API 而外参数，默认 {}
    - `LLM_MODEL` (str): 模型名称，默认 ""
    - `LLM_TEMP` (float): 温度参数 (0.0-1.0)，越低越确定，默认 0.00
    - `LLM_TOP_K` (int): 采样参数，默认 60
    - `LLM_TOP_P` (float): 采样参数，默认 0.70
    - `LLM_PROMPT_LOCATION` (str): 提示语位置，默认 "system"
    - `LLM_CONTEXTS` (bool): 是否启用上下文记忆，默认 True
    - `LLM_CONTEXTS_LENGTH` (int): 上下文长度限制，默认 65536
    - `LLM_MAX_WORKERS` (int): 最大并发数，默认 24
    - `LLM_MAX_BATCH` (int): 单次批处理数量，默认 3
    - `LLM_MAX_RETRY` (int): 失败重试次数，默认 128
    - `LLM_ORIGINAL_REFERENCE` (bool): 输出是否包含原文，示例: 我喜欢你(I like you)，默认 True

    #### 3. 嵌入模型设置 (Embedding)
    - `EMB_API_URL` (str): 嵌入模型 API 地址，默认 ""
    - `EMB_API_KEY` (str): 嵌入模型 API 密钥，默认 ""
    - `EMB_MODEL` (str): 模型型号或 HF 仓库名，默认 "nomic-ai/nomic-embed-text-v1.5"
    - `EMB_MODEL_ACC_MODE` (str): 加速模式 (None, ONNX, float64, etc.)，默认 "bfloat16"
    - `EMB_MAX_TOKENS` (int): 最大输入 Token 数，默认 2048
    - `EMB_TOKENSTOTEXT_RATIO` (float): Token 转字数比例，默认 3.0
    - `EMB_MAX_WORKERS` (int): 最大并发数，默认 24
    - `EMB_MAX_RETRY` (int): 失败重试次数，默认 128

    #### 4. 向量存储设置 (Vector Store) 修改量化相关设置可能导致出现错误
    - `VEC_FILE_PATH` (str): 向量文件存储目录，默认 r"./Vectors"
    - `VEC_FILE_NAME` (str): 向量文件名，默认 "Vectors"
    - `VEC_QUANTIZATION` (str): 量化类型，["Float32", "Float16", "Float16_E0M15", "BFloat16", "Float8_E4M3", "Q8_K_X", "Q6_K_X", "Q4_K_X", "Q3_K_X", "Q2_K_X"] 从中选择，默认 "Q4_K_X"
    - `VEC_QUANTILE` (float): Q系列量化裁切分位数，范围0.99-1.0，默认0.998
    - `VEC_QUANTIZATION_BLOCK_SIZE` (int): 量化块大小，需要2的倍数，最大256，默认 32

    #### 5. 缓存与路径 (Cache & Path)
    - `TRANSLATOR_CACHE_WRITE` (bool): 是否写入翻译缓存，默认 True
    - `TRANSLATOR_CACHE_READ` (bool): 是否读取翻译缓存，默认 True
    - `TRANSLATOR_CACHE_PATH` (str): 缓存目录，默认 r"./Translator_Cache"
    - `TRANSLATOR_CACHE_NAME` (str): 缓存文件名，默认 "Translator_Cache"
    - `PATH_CACHE` (str): 通用缓存路径，默认 r"./Cache"
    - `LANG_PATH` (str): 语言包路径，默认 r"./Lang"
    - `DEBUG_MODE` (bool): 是否开启调试模式，默认 False

    #### 6. 索引设置 (Indexing - Faiss/HNSW)
    - `INDEX_K` (int): 检索提示数量，默认 3
    - `INDEX_MODE` (str): 索引模式 (RefineFlat, HNSWSQ)，默认 "RefineFlat"
    - `INDEX_SQ` (str): 索引量化 (Q4, Q6, Q8, F16, BF16)，默认 "Q6"
    - `INDEX_HNSW_M` (int): HNSW M 参数，默认 128
    - `INDEX_HNSW_CONSTRUCTION` (int): 构建复杂度，默认 720
    - `INDEX_HNSW_SEARCH` (int): 搜索复杂度，默认 480
    - `INDEX_REFINEFLAT_K_FACTOR` (float): 精炼因子，默认 2.0
    
    #### 任务并发设置 (Task Concurrency)
    - `QUESTS_FTB_READ_MAX_CONCURRENT` (int): FTBQuests 读取最大并发数，默认 4
    - `QUESTS_FTB_WRITE_MAX_CONCURRENT` (int): FTBQuests 写入最大并发数，默认 4
    - `QUESTS_BQ_READ_MAX_CONCURRENT` (int): BetterQuesting 读取最大并发数，默认 4
    - `QUESTS_BQ_WRITE_MAX_CONCURRENT` (int): BetterQuesting 写入最大并发数，默认 4

    ### 使用示例 (Example)
    ```python
    # 仅修改语言和部分 LLM 参数
    config = {
        "LANGUAGE_INPUT": "ja_jp",
        "LANGUAGE_OUTPUT": "zh_cn",
        "LLM_API_KEY": "sk-xxxxxx",
        "LLM_TEMP": 0.5
    }
    SetTranslatorConfig(config)
    ```

    ### 返回说明 (Returns)
    - **成功**: 返回字符串 "配置已更新！当前生效参数：[参数列表]"
    - **失败**: 返回字符串 "配置参数无效：'[错误参数名]'" 或错误详情
    """
    global 配置文件
    try:
        配置文件 = config_kwargs
        modified_items = list(配置文件.keys())
        return f"配置已更新！当前生效参数：{', '.join(modified_items)}\n(下次翻译将自动使用这些新设置)"
    except AttributeError as e:
        错误详情 = str(e)
        返回配置 = 错误详情.split(": ")[1].strip().split(" '")[0] if "' " in 错误详情 else 错误详情
        return f"配置参数无效：'{返回配置}'\n请检查 SetTranslatorConfig 的文档，确认参数名称是否正确。"
    except Exception as e:
        return f"更新配置时发生未知错误:\n{eb.format_exc()}"
@mcp.tool()
def GetTranslatorConfig():
    """查询当前的翻译器配置状态。
    Returns:
        str: 返回当前所有生效的配置项及其值（JSON格式或易读文本）。
    """
    global 配置文件
    config_dict = vars(配置文件)
    key_items = {k: v for k, v in config_dict.items() if not k.startswith("_")}
    return f"当前配置状态：\n" + "\n".join([f"- {k}: {v}" for k, v in key_items.items()])
@mcp.tool()
def TranslatorCore(File0: str, File1: str = None, ExportInspection: bool = False, AllMode: bool = False) -> str:
    """
    ### 工具功能
    此函数为程序核心功能
    翻译语言文件或包含语言文件的压缩包
    支持类型:
      - .zip,光影/资源包
      - .lang/.json语言文件
      - .zip/.mrpack/已安装的整合包文件夹,整合包(模组包)(设计仅支持CurseForge Modrint MultiMC格式的整合包, 其他格式可能通用)(翻译包含 KubeJS配置文件夹、资源文件夹(非资源包文件夹)、模组文件夹、FTB任务文件夹、BQ任务文件夹)
      - .zip/ftbquests文件夹,FTBQuests(FTB任务 1.12以上版本,传入路径翻译自动覆盖源文件)
      - .zip/betterquesting文件夹,BetterQuesting(更好的任务 1.7以上版本,传入路径翻译自动覆盖源文件)
    此工具依赖全局配置，**使用前请确保已调用 `SetTranslatorConfig`。
    最低工作参数: LLM_API_URL、LLM_MODEL
    如果用户没有说明请不要随意改变配置。

    ### 参数详解
    - `File0` (str): **源文件路径**。
      - 翻译File0内容, 程序会自动处理内部路径 
    - `File1` (str): **参考文件路径** (可选)。
      - 留空：仅基于 File0 进行翻译。
      - 指定路径：作为目标语言参考或对比基准（例如：提供已有的目标语言文件用于合并或差异计算）。
    - `OutputPath` (str): **输出路径** (可选)。
      - 留空：默认保存在当前工作目录。
    - `ExportInspection` (bool): **导出审核文件** (可选)。
      - 翻译 Minecraft jar模组、zip光影/资源包、.lang/.json语言文件 此项才可以正常工作
      - False (默认): 直接输出翻译完成的文件。
      - True: 输出用于人工校对/审核的中间格式文件 (.translatorlang)，忽略常规输出命名规则。
    - `AllMode` (bool): **整合包翻译模组跳过模式** (可选)。
      - 翻译.zip/.mrpack/已安装的整合包路径,整合包(模组包) 此项才可以正常工作
      - True: 翻译所有可翻译的模组。
      - False (默认): 剔除I18n模组已有的汉化。需要资源包文件夹内已有 I18n自动汉化更新资源(安装 I18n自动汉化更新 模组启动游戏自动生成) 才可以正常工作

    ### 返回内容
    - **成功**: 返回字符串，包含 "函数执行完成" 及详细的处理日志（包含处理信息、文件路径等）。
    - **失败**: 返回字符串，包含 "发生未知错误" 及具体的 traceback 堆栈信息。

    ### 注意事项
    1. 压缩包模式会自动处理内部结构，无需手动解压。
    2. 目标语言代码由全局配置 `LANGUAGE_OUTPUT` 决定。
    3. 如果用户描述为中文且返回日志文件包含 未找到I18nUpdateMod资源包,可能导致翻译时长增加 可提醒用户安装 I18n自动汉化更新 模组(https://www.mcmod.cn/class/1188.html)
    """
    File0 = 路径处理(File0)
    File1 = 路径处理(File1) if File1 else ""
    try:
        设置时间()
        translator = Translator(Config=配置文件)
        translator.翻译通用文件(
            file0=File0, 
            file1=File1, 
            export_inspection=ExportInspection,
            all_mode=AllMode
        )
        logs = translator.调用额外函数("读取日志")
        return f"函数执行完成\n{logs}"
    except Exception:
        return f"发生未知错误:\n{eb.format_exc()}"

@mcp.tool()
def ImportDictMiniSystemPrompt(File: str, Mode: str = "dense"):
    """
    ### 工具功能
    导入翻译提示词（仅支持 DictMini.json 文件）。
    此工具不完全依赖全局配置, 但是依赖嵌入模型(EMB_MODEL), 默认已经设置。
    此工具文件默认保存 VEC_FILE_PATH/VEC_FILE_NAME 下。
    如果用户没有说明请不要随意改变配置。
    
    ### 参数详解
    - `File` (str): **导入文件**。
      - 导入 DictMini.json 文件。
      - 文件来源: https://github.com/CFPATools/i18n-dict。
    - `Mode` (str): **导入模式** (可选)。
      - dense: 稠密模式, 相同单词的翻译值会被压缩为一个值。
      - sparse: 稀疏模式, 解析每个翻译值为一个独立值。
      
    ### 返回内容
    - **成功**: 返回字符串，包含 "函数执行完成" 及详细的处理日志（包含处理信息、文件路径等）。
    - **失败**: 返回字符串，包含 "发生未知错误" 及具体的 traceback 堆栈信息。
    """
    File = 路径处理(File)
    try:
        设置时间()
        translator = Translator(Config=配置文件)
        translator.导入DictMini参考词(
            file=File,
            mode=Mode
        )
        logs = translator.调用额外函数("读取日志")
        return f"函数执行完成\n{logs}"
    except Exception:
        return f"发生未知错误:\n{eb.format_exc()}"
@mcp.tool()
def ImportDictMiniTranslatorCache(File: str):
    """
    ### 工具功能
    导入翻译缓存（仅支持 DictMini.json 文件）, 翻译缓存可用于命中翻译文本。
    此工具不完全依赖全局配置, 默认已经设置。
    此工具文件默认保存 TRANSLATOR_CACHE_PATH/TRANSLATOR_CACHE_NAME 下。
    如果用户没有说明请不要随意改变配置。
    
    ### 参数详解
    - `File` (str): **导入文件**。
      - 导入 DictMini.json 文件。
      - 文件来源: https://github.com/CFPATools/i18n-dict。
      
    ### 返回内容
    - **成功**: 返回字符串，包含 "函数执行完成" 及详细的处理日志（包含处理信息、文件路径等）。
    - **失败**: 返回字符串，包含 "发生未知错误" 及具体的 traceback 堆栈信息。
    """
    File = 路径处理(File)
    try:
        设置时间()
        translator = Translator(Config=配置文件)
        translator.导入DictMini缓存(
            file=File,
        )
        logs = translator.调用额外函数("读取日志")
        return f"函数执行完成\n{logs}"
    except Exception:
        return f"发生未知错误:\n{eb.format_exc()}"
@mcp.tool()
def ImportSystemPrompt(Path: str):
    """
    ### 工具功能
    导入翻译提示词（支持 Minecraft jar模组、zip光影/资源包、该软件专属格式pkl语言文件）。
    此工具不完全依赖全局配置, 但是依赖嵌入模型(EMB_MODEL), 默认已经设置。
    此工具文件默认保存 VEC_FILE_PATH/VEC_FILE_NAME 下。
    如果用户没有说明请不要随意改变配置。
    
    ### 参数详解
    - `Path` (str): **导入路径**。
      - 导入该路径下的所有 .jar .zip 与 .pkl 文件。
      
    ### 返回内容
    - **成功**: 返回字符串，包含 "函数执行完成" 及详细的处理日志（包含处理信息、文件路径等）。
    - **失败**: 返回字符串，包含 "发生未知错误" 及具体的 traceback 堆栈信息。
    """
    Path = 目录处理(Path)
    try:
        设置时间()
        translator = Translator(Config=配置文件)
        translator.导入参考词(
            path=Path, 
        )
        logs = translator.调用额外函数("读取日志")
        return f"函数执行完成\n{logs}"
    except Exception:
        return f"发生未知错误:\n{eb.format_exc()}"
@mcp.tool()
def DictMiniConvertsDataset(File: str, Mode: str = "Alpaca", OutputFile: str = "dataset.jsonl"):
    """
    ### 工具功能
    将DictMini文件转换为大语言模型训练/微调数据集（仅支持 DictMini.json 文件）。
    如果用户没有说明请不要随意改变配置。
    
    ### 参数详解
    - `File` (str): **导入文件**。
      - 导入 DictMini.json 文件。
      - 文件来源: https://github.com/CFPATools/i18n-dict。
    - `Mode` (str): **导入模式** (可选)。
      - Alpaca: Alpaca格式文件，主要用于 hiyouga/LLaMAFactory。
      - ChatML: ChatML格式文件，通用格式。
    - `OutputFile` (str): **输出文件** (可选)。
      - 输出文件路径，必须为.jsonl扩展名。
      - 默认: dataset.jsonl
      - 示例输入 dataset.jsonl、C:/Users/%username%/Desktop/dataset.jsonl
    
    ### 返回内容
    - **成功**: 返回字符串，包含 "函数执行完成" 及详细的处理日志（包含处理信息、文件路径等）。
    - **失败**: 返回字符串，包含 "发生未知错误" 及具体的 traceback 堆栈信息。
    """
    File = 路径处理(File)
    OutputFile = 路径处理(OutputFile)
    try:
        设置时间()
        translator = Translator(Config=配置文件)
        translator.DictMini转换数据集(
            file=File,
            mode=Mode,
            output_file=OutputFile
        )
        logs = translator.调用额外函数("读取日志")
        return f"函数执行完成\n{logs}"
    except Exception:
        return f"发生未知错误:\n{eb.format_exc()}"
@mcp.tool()
def SeparateLangUpdate(File0: str, File1: str = None, OutputPath: str = None, Mode: str = "none"):
    """
    ### 工具功能
    输出新添加的键值或人工翻译文件（支持 Minecraft jar模组、zip光影/资源包、.lang/.json语言文件）。
    此工具不依赖全局配置。
    如果用户没有说明请不要随意改变配置。

    ### 参数详解
    - `File0` (str): **源文件路径**。
      - 如果是压缩包 (.jar/.zip)：将解压并处理包内的语言文件。
      - 如果是语言文件 (.lang/.json)：直接处理该文件。
    - `File1` (str): **参考文件路径** (可选)。
      - 留空：仅基于 File0 进行处理。
      - 指定路径：作为目标语言参考或对比基准（例如：提供已有的目标语言文件用于合并或差异计算）。
    - `OutputPath` (str): **输出路径** (可选)。
      - 留空：默认保存在当前工作目录。
    - `Mode` (str): **处理模式** (可选)。
      - none (默认): 默认处理模式，返回File0缺失的键值对的语言文件。
      - extra: 翻译提示词附加处理模式，返回ile0缺失的键值对的此工具专有.translatorlang格式文件。
      - 还有种模式为翻译后检查模式，请查看TranslatorLang的ExportInspection值。
      
    ### 返回内容
    - **成功**: 返回字符串，包含 "函数执行完成" 及详细的处理日志（包含处理信息、文件路径等）。
    - **失败**: 返回字符串，包含 "发生未知错误" 及具体的 traceback 堆栈信息。
    
    ### 注意事项
    1. 压缩包模式会自动处理内部结构，无需手动解压。
    2. 输出语言由全局配置 `LANGUAGE_INPUT` 与 `LANGUAGE_OUTPUT` 决定。
    """
    File0 = 路径处理(File0)
    File1 = 路径处理(File1) if File1 else ""
    OutputPath = 目录处理(OutputPath) if OutputPath else ""
    try:
        设置时间()
        translator = Translator(Config=配置文件)
        translator.分离语言文件更新(
            file0=File0,
            file1=File1,
            output_path=OutputPath,
            mode=Mode
        )
        logs = translator.调用额外函数("读取日志")
        return f"函数执行完成\n{logs}"
    except Exception:
        return f"发生未知错误:\n{eb.format_exc()}"
@mcp.tool()
def MergeLangUpdate(File0: str, NotLangFile: str, File1: str = None, OutputPath: str = None):
    """
    ### 工具功能
    合并新添加的键值或人工翻译文件（支持 Minecraft jar模组、zip光影/资源包、.lang/.json语言文件）。
    此工具不依赖全局配置。
    如果用户没有说明请不要随意改变配置。

    ### 参数详解
    - `File0` (str): **源文件路径**。
      - 如果是压缩包 (.jar/.zip)：将解压并处理包内的语言文件。
      - 如果是语言文件 (.lang/.json)：直接处理该文件。
    - `File1` (str): **参考文件路径** (可选)。
      - 留空：仅基于 File0 进行处理。
    - `NotLangFile` (str): **待合并文件**。
      - 待合并的语言文件 (支持 Minecraft jar模组、zip光影/资源包、.lang/.json/.translatorlang语言文件)。
    - `OutputPath` (str): **输出路径** (可选)。
      - 留空：默认保存在当前工作目录。

      
    ### 返回内容
    - **成功**: 返回字符串，包含 "函数执行完成" 及详细的处理日志（包含处理信息、文件路径等）。
    - **失败**: 返回字符串，包含 "发生未知错误" 及具体的 traceback 堆栈信息。
    
    ### 注意事项
    1. 压缩包模式会自动处理内部结构，无需手动解压。
    2. 输出语言由全局配置 `LANGUAGE_INPUT` 与 `LANGUAGE_OUTPUT` 决定。
    """
    File0 = 路径处理(File0)
    File1 = 路径处理(File1) if File1 else ""
    OutputPath = 目录处理(OutputPath) if OutputPath else ""
    NotLangFile = 目录处理(NotLangFile)
    try:
        设置时间()
        translator = Translator(Config=配置文件)
        translator.合并语言文件更新(
            file0=File0,
            file1=File1,
            output_path=OutputPath,
            notlang_file=NotLangFile
        )
        logs = translator.调用额外函数("读取日志")
        return f"函数执行完成\n{logs}"
    except Exception:
        return f"发生未知错误:\n{eb.format_exc()}"

def ViewFolders(path='.') -> dict:
    """查看当前文件夹下有什么文件与文件夹"""
    result = {}
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            result[name] = {'type': 'directory'}
        else:
            result[name] = {'type': 'file'}
    return result
@mcp.tool()
def ViewTextFile(File: str) -> str:
    """查看文件，只能查看文本类格式"""
    with open(File, 'r', encoding='utf-8') as f:
        return f.read()
    

if __name__ == "__main__":
    print("TranslatorMCP服务器运行中")
    mcp.run(transport="http", host=MCPConfig["host"], port=MCPConfig["port"])