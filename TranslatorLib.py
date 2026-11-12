from __future__ import annotations 

__lazy_modules__ = [ # 仅 Python 3.15 可用
    "ast", "asyncio", "atexit", "base64", "bisect", "collections",
    "concurrent.futures", "contextlib", "copy", "dataclasses", "datetime",
    "enum", "functools", "hashlib", "heapq", "importlib", "io", "json",
    "locale", "logging", "logging.handlers", "math", "os", "pathlib",
    "pickle", "pkgutil", "queue", "random", "re", "shlex", "shutil",
    "sqlite3", "struct", "sys", "threading", "time", "tomllib",
    "traceback", "types", "typing", "urllib.parse", "urllib3.util.retry",
    "tarfile", "uuid", "warnings", "zipfile",

    "aiohttp", "bm25s", "cupy", "datasets", "dnfile", "faiss",
    "fastapi", "fastapi.middleware.cors", "fastapi.responses",
    "fastapi.security", "huggingface_hub", "numba", "numpy", "PIL.Image",
    "requests", "requests.adapters", "rich.align", "rich.color",
    "rich.console", "rich.panel", "rich.style", "rich.text", "slowapi",
    "slowapi.errors", "slowapi.util", "token_calibrator", "tqdm",
    "tqdm.rich", "ujson", "uvicorn", "xllamacpp",
]

import traceback as eb
import threading
import importlib
import zipimport
import mimetypes
import warnings
import datetime
import pkgutil
import hashlib
import tarfile
import zipfile
import tomllib
import asyncio
import logging
import secrets
import shutil
import locale
import bisect
import pickle
import random
import atexit
import base64
import struct
import heapq
import types
import queue
import shlex
import time
import uuid
import math
import copy
import ast
import sys
import re
import io
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") # 防止私有内存核爆 向量生成加速 NumPy减速
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1") # 防止 HuggingFace Hub 下载的模型文件被当作符号链接,

from io import BytesIO
from re import compile as _re_compile, sub as _re_sub, MULTILINE
from json import dumps as _json_dumps, loads as _json_loads
from enum import IntEnum
from types import SimpleNamespace, MethodType
from urllib.parse import quote
from typing import Callable, Dict, Any, Union, Optional, List, TextIO, Tuple, Type, TYPE_CHECKING
from urllib3.util.retry import Retry
from logging import FileHandler
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path, PurePosixPath
from requests.adapters import HTTPAdapter
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from collections import defaultdict, deque, OrderedDict, Counter
from dataclasses import dataclass, replace
#需要安装↓ gguf numpy aiohttp requests faiss
import gguf
import numpy
import faiss
import aiohttp
import requests
#需要安装↓ rich tqdm
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.style import Style
from rich.color import Color
from tqdm.rich import tqdm as rich_tqdm
from tqdm import tqdm as tqdm_tqdm, TqdmExperimentalWarning
#可选服务安装↓ uvicorn fastapi slowapi
try:
    import uvicorn, fastapi, slowapi
    from fastapi import FastAPI, UploadFile, HTTPException, status, Depends, Security, Form, Request, BackgroundTasks
    from fastapi.responses import FileResponse, PlainTextResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
except ImportError:
    uvicorn, fastapi, slowapi = None, None, None
#下面可选安装 numba ujson cupy (pythonnet dnfile)
NOT_IMPORT = []
try:
    import ujson as json
except:
    import json
try:
    import dnfile
except:
    dnfile = None
    NOT_IMPORT.append("dnfile")
try:
    import token_calibrator
except:
    class token_calibrator:
        class TokenCalibrator: pass
        class TokenEstimator: pass
    NOT_IMPORT.append("token_calibrator")
try:
    import huggingface_hub
except:
    class huggingface_hub:
        class hf_hub_download: pass
    NOT_IMPORT.append("huggingface_hub")
try:
    import datasets
except:
    class datasets:
        class load_dataset: pass
    NOT_IMPORT.append("datasets")
try:
    from PIL import Image
except:
    class Image:
        class open: pass
        class save: pass
    NOT_IMPORT.append("pillow")
try:
    import bm25s
except:
    bm25s = None
    NOT_IMPORT.append("bm25s")
try:
    import xllamacpp
except:
    class xllamacpp: pass
    NOT_IMPORT.append("xllamacpp")
try:
    import libarchive
except:
    libarchive = None
    NOT_IMPORT.append("libarchive-c")
try:
    import compression.zstd as zstd # type: ignore
except:
    try:
        import zstandard as zstd
    except:
        NOT_IMPORT.append("zstandard")
        zstd = None
ConfigFile = Path("config.cfg").resolve()
ConfigFile.parent.mkdir(parents=True, exist_ok=True)
if ConfigFile.is_file():
    with open(ConfigFile, "r", encoding="utf-8") as f:
        Config = json.load(f)
else:
    Config = {
        "GPU_Accelerator": True,
        "GPU_Device_ID": None
    }
    with open(ConfigFile, "w+", encoding="utf-8") as f:
        json.dump(Config, f, indent=4)
        
CPU_ACC, GPU_ACC = False, False
gpu_error, numba_error = "", ""
try:
    if not Config["GPU_Accelerator"]:
        raise ValueError("GPU_ACC is set to False")

    import cupy as np  # type: ignore
    if Config["GPU_Device_ID"] is not None:
        np.cuda.runtime.setDevice(Config["GPU_Device_ID"])

    np.dot(np.random.rand(2, 2), np.random.rand(2, 2))
    if not np.cuda.is_available():
        raise ValueError("CUDA not available")

    GPU_ACC = True
except ValueError:
    np = numpy
except Exception as e:
    np = numpy
    gpu_error = e
try:
    import numba  # type: ignore
    from numba import njit  # type: ignore
    CPU_ACC = True
except Exception as e:
    numba = None
    numba_error = e
    def njit(*a, **kw):
        def _wrap(f): return f
        return _wrap(a[0]) if a and callable(a[0]) else _wrap
HARDWARE_INFO = {
    "type": "GPU" if GPU_ACC else "CPU",
    "acc_type": "CuPy" if GPU_ACC else ("Numba" if CPU_ACC else "None"),
    "version": numpy.__version__,
    "acc_version": numba.__version__ if CPU_ACC else ("CuPy" if GPU_ACC else "None"),
    **({"device_count": np.cuda.runtime.getDeviceCount(),
        "device_id": np.cuda.runtime.getDevice()} if GPU_ACC else {}),
    "error": "None" if GPU_ACC else (numba_error if not CPU_ACC else gpu_error)
}
APIConfigFile = Path("config-api.cfg").resolve()
APIConfigFile.parent.mkdir(parents=True, exist_ok=True)
if APIConfigFile.is_file():
    with open(APIConfigFile, "r", encoding="utf-8") as f:
        APIConfig = json.load(f)
else:
    APIConfig = {"LOGS_GLOBAL": False}
    with open(APIConfigFile, "w+", encoding="utf-8") as f:
        json.dump(APIConfig, f, indent=4)
            
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning) # 屏蔽rich.tqdm警告
            
_prange = numba.prange if numba is not None else range


def CleanVRAM():
    if GPU_ACC:
        np.get_default_memory_pool().free_all_blocks()
        np.get_default_pinned_memory_pool().free_all_blocks()
def 格式化时间(秒数):
    if 秒数 is None or 秒数 < 0 or not math.isfinite(秒数):
        return "--:--:--"
    秒数 = int(秒数)
    小时 = 秒数 // 3600
    分钟 = (秒数 % 3600) // 60
    秒 = 秒数 % 60
    return f"{小时:02d}:{分钟:02d}:{秒:02d}"
class diff_tqdm:
    def __init__(Self, tasks, iterable=None, desc=None, mininterval=None, disable=False, **kwargs):
        Self.任务 = tasks
        Self._last_tasks_ref = tasks
        Self.数量 = len(tasks)
        Self.desc = desc
        Self.开始时间 = time.monotonic()
        Self.n = 0
        Self.上次完成 = 0
        Self.上次时间 = Self.开始时间
        Self.缓存速率 = 0.0
        Self.禁用显示 = disable
    def refresh(Self):
        if Self.禁用显示: return
        if not Self.任务: return
        if Self._last_tasks_ref is not Self.任务:
            Self._last_tasks_ref = Self.任务
            Self.数量 = len(Self.任务)
            Self.上次完成 = 0
            Self.上次时间 = time.monotonic()
            Self.缓存速率 = 0.0
            Self.开始时间 = Self.上次时间
        方块列表 = []
        当前完成 = 0
        for 元素 in Self.任务:
            if 元素 == "Done": # 完成
                方块列表.append("\033[38;2;0;220;0m█\033[0m")
                当前完成 += 1
            elif 元素 == "Fail": # 失败
                方块列表.append("\033[38;2;235;0;0m█\033[0m")
                当前完成 += 1
            elif 元素 == "Retrying": # 重试中
                方块列表.append("\033[38;2;245;130;0m█\033[0m") 
            elif 元素 == "Working": # 工作中
                方块列表.append("\033[38;2;255;255;0m█\033[0m") 
            elif 元素 == "ItemByItem": # 逐条处理
                方块列表.append("\033[38;2;0;0;220m█\033[0m") 
            else: # Unassigned 未分配
                方块列表.append("\033[38;2;192;192;192;0m█\033[0m") 
        Self.n = 当前完成 
        方块字符串 = "".join(方块列表)
        总任务数 = Self.数量
        当前时间 = time.monotonic()
        已用秒数 = 当前时间 - Self.开始时间
        if 当前完成 > Self.上次完成:
            时间差 = 当前时间 - Self.上次时间
            新增数 = 当前完成 - Self.上次完成
            if 时间差 > 0:
                瞬时速率 = 新增数 / 时间差
                Self.缓存速率 = 0.4 * 瞬时速率 + 0.6 * Self.缓存速率
            else:
                Self.缓存速率 = 当前完成 / 已用秒数 if 已用秒数 > 0 else 0.0
            Self.上次完成 = 当前完成
            Self.上次时间 = 当前时间
        处理速率 = Self.缓存速率
        百分比 = (当前完成 / 总任务数) * 100 if 总任务数 else 0.0
        剩余秒数 = (总任务数 - 当前完成) / 处理速率 if 处理速率 > 0 else float('inf')
        已用显示 = 格式化时间(已用秒数)
        剩余显示 = 格式化时间(剩余秒数)
        信息行 = f"{Self.desc} {百分比:3.1f}% {当前完成}/{总任务数} [ {已用显示} < {剩余显示} , {处理速率:.1f} it/s ] {方块字符串}"
        print(f"\r{信息行}")

class PDict(Dict):
    def __init__(Self, 文件路径, 保存间隔, 清理间隔, 留存间隔, 豁免键=[]):
        super().__init__()
        Self.文件路径 = Path(文件路径)
        Self.保存间隔 = 保存间隔
        Self.清理间隔 = 清理间隔
        Self.留存间隔 = 留存间隔
        Self.豁免键 = {键 for 键 in 豁免键}
        Self.正在保存 = False
        Self.正在清理 = False
        Self.已停止 = False
        Self.保存任务 = None
        Self.清理任务 = None
        Self.写入时间 = {}
        Self.保存异常 = None
        Self.文件路径.parent.mkdir(parents=True, exist_ok=True)
        Self.加载()
        Self.seve = Self.保存
    def 加载(Self):
        try:
            with open(Self.文件路径, "rb") as 文件:
                存档 = pickle.load(文件)
        except FileNotFoundError:
            return
        except Exception as 异常:
            Self.保存异常 = 异常
            return
        if not isinstance(存档, dict):
            return
        if "数据" in 存档:
            数据, 存档时间 = (存档.get("数据") or {}), (存档.get("写入时间") or {})
        else:
            数据, 存档时间 = 存档, {}
        当前时间 = time.time()
        for 键, 值 in 数据.items():
            super().__setitem__(键, 值)
            Self.写入时间[键] = 存档时间.get(键, 当前时间)
    def 快照(Self):
        return {"数据": dict(Self), "写入时间": dict(Self.写入时间)}
    def 写入文件(Self, 存档):
        临时文件 = Self.文件路径.with_name(f"{Self.文件路径.name}.tmp")
        with open(临时文件, "wb") as 文件:
            pickle.dump(存档, 文件, protocol=pickle.HIGHEST_PROTOCOL)
        临时文件.replace(Self.文件路径)
    async def 保存(Self):
        Self.正在保存 = True
        Self.保存异常 = None
        try:
            await asyncio.to_thread(Self.写入文件, Self.快照())
        except Exception as 异常:
            Self.保存异常 = 异常
        finally:
            Self.正在保存 = False
    def 触发保存(Self):
        return asyncio.create_task(Self.保存())
    async def 运行保存(Self):
        while not Self.已停止:
            await asyncio.sleep(Self.保存间隔)
            if not Self.正在保存:
                Self.触发保存()
                continue
            while Self.正在保存 and not Self.已停止:
                await asyncio.sleep(Self.保存间隔)
    async def 运行清理(Self):
        while not Self.已停止:
            await asyncio.sleep(Self.清理间隔)
            await Self.清理()
    async def 清理(Self):
        Self.正在清理 = True
        try:
            截止时间 = time.time() - Self.留存间隔
            过期键 = [键 for 键, 写入时间 in Self.写入时间.items()
                      if 键 not in Self.豁免键 and 写入时间 < 截止时间]
            for 键 in 过期键:
                Self.写入时间.pop(键, None)
                if super().__contains__(键):
                    super().__delitem__(键)
        finally:
            Self.正在清理 = False
    def 启动(Self):
        Self.已停止 = False
        Self.保存任务 = asyncio.create_task(Self.运行保存())
        Self.清理任务 = asyncio.create_task(Self.运行清理())
        return Self.保存任务
    async def 停止(Self, 最后保存: bool = True):
        Self.已停止 = True
        for 任务 in (Self.保存任务, Self.清理任务):
            if 任务 is not None:
                任务.cancel()
                try:
                    await 任务
                except asyncio.CancelledError:
                    pass
        Self.保存任务 = Self.清理任务 = None
        if 最后保存:
            await Self.保存()
    def __setitem__(Self, 键, 值):
        super().__setitem__(键, 值)
        Self.写入时间[键] = time.time()
    def __delitem__(Self, 键):
        Self.写入时间.pop(键, None)
        super().__delitem__(键)
    def pop(Self, 键, *默认值):
        值 = super().pop(键, *默认值)
        Self.写入时间.pop(键, None)
        return 值
    def update(Self, *参数, **关键字):
        super().update(*参数, **关键字)
        当前时间 = time.time()
        for 键 in dict(*参数, **关键字):
            Self.写入时间[键] = 当前时间
    def setdefault(Self, 键, 默认值=None):
        值 = super().setdefault(键, 默认值)
        Self.写入时间.setdefault(键, time.time())
        return 值
    def 添加豁免(Self, *键):
        Self.豁免键.update(键)
    def 移除豁免(Self, *键):
        for 键 in 键:
            Self.豁免键.discard(键)
    def clear(Self):
        保留 = {键: super().__getitem__(键) for 键 in Self.豁免键 if super().__contains__(键)}  # 豁免键整条保留
        Self.写入时间.clear()
        super().clear()
        for 键, 值 in 保留.items():
            super().__setitem__(键, 值)
    async def __aenter__(Self):
        Self.启动()
        return Self
    async def __aexit__(Self, 异常类型, 异常, 回溯):
        await Self.停止()

def 彩色文本(文本: str, 颜色序列: list[str] | None = None) -> Text:
    if 颜色序列 is None:
        颜色序列 = ["#5555FF", "#AA55FF", "#FF5555", "#FFAA00", "#FFFF55", "#55FF55"]
    rgb序列 = [Color.parse(c).get_truecolor() for c in 颜色序列]
    颜色数量 = len(rgb序列)
    行列表 = 文本.split('\n')
    最大宽度 = max(len(行) for 行 in 行列表) if 行列表 else 1
    结果文本 = Text()
    for 行 in 行列表:
        for 列索引, 字符 in enumerate(行):
            if 字符 == ' ':
                结果文本.append(字符)
                continue
            比例 = 列索引 / max(最大宽度 - 1, 1)
            段索引浮点 = 比例 * (颜色数量 - 1)
            起始颜色索引 = int(段索引浮点)
            结束颜色索引 = min(起始颜色索引 + 1, 颜色数量 - 1)
            局部比例 = 段索引浮点 - 起始颜色索引
            颜色1, 颜色2 = rgb序列[起始颜色索引], rgb序列[结束颜色索引]
            红 = int(颜色1[0] + (颜色2[0] - 颜色1[0]) * 局部比例)
            绿 = int(颜色1[1] + (颜色2[1] - 颜色1[1]) * 局部比例)
            蓝 = int(颜色1[2] + (颜色2[2] - 颜色1[2]) * 局部比例)
            十六进制颜色 = f"#{红:02x}{绿:02x}{蓝:02x}"
            结果文本.append(字符, style=Style(color=十六进制颜色))
        结果文本.append('\n')
    return 结果文本


class Parser:
    def __init__(Self, 应用):
        Self.应用 = 应用
        Self.工具 = 应用.File
        Self.日志 = 应用.日志
        Self.Config = 应用.Config
        Self.Module = 应用.Module
        Self.Lang = 应用.Lang

    def 读取Json文件(Self, 文件):
        return Self.工具.读取Json文件(文件)

    def 写入Json文件(Self, 文件路径, 数据, 缩进=4):
        return Self.工具.写入Json文件(文件路径, 数据, 缩进)

    def 设置路径值(Self, 数据, 路径, 译文):
        return Self.工具.设置路径值(数据, 路径, 译文)

    def 递归提取文本(Self, *参数, **关键字):
        return Self.工具.递归提取文本(*参数, **关键字)

    def 读取(Self, 文件路径) -> list:
        raise NotImplementedError(f"{type(Self).__name__} 未实现 读取")

    def 写回(Self, 翻译列表) -> None:
        raise NotImplementedError(f"{type(Self).__name__} 未实现 写回")

class InconsistentValues(Exception):
    def __init__(Self, a: str, b: str):
        Self.a = a
        Self.b = b
        super().__init__(f"The values of the two variables are different: {a} and {b}")
        

if GPU_ACC:
    加速方法 = "CuPy"
    加速版本 = np.__version__
elif CPU_ACC:
    加速方法 = "Numba"
    加速版本 = numba.__version__
else:
    加速方法 = "None"
    加速版本 = ""

#Pagga
文本 = 彩色文本("""████████╗██████╗  █████╗ ███╗   ██╗███████╗██╗      █████╗ ███╗   ███╗ ██████╗
╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║     ██╔══██╗████╗ ████║██╔════╝
   ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ███████║██╔████╔██║██║
   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██║██║╚██╔╝██║██║
   ██║   ██║  ██║██║  ██║██║ ╚████║███████║███████╗██║  ██║██║ ╚═╝ ██║╚██████╗
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝""")

信息文本 = Text.from_markup(f"""
[bold]TranslatorMinecraft Core[/bold]
[bright_green]Version:[/] Release 1.7 Bata.1
[bright_green]NumPy Accelerator:[/] {加速方法} {加速版本}""")

总文本 = Text.assemble(文本, 信息文本)

Console(force_terminal=True, color_system="auto").print(
    Panel(
        Align(总文本, align="center"),
        title="[blue]TranslatorMinecraft Core[/blue]",
        border_style="blue",
        padding=(1, 2),
        width=110
    )
)
__all__ = [
    "Tuple",
    "aiohttp", "APIConfig", "Any", "as_completed", "ast", "asynccontextmanager", "asyncio", "atexit",  # A
    "bisect", "base64", "bm25s",  # B
    "Callable", "copy", "Counter",  # C
    "dataclass", "defaultdict", "deque", "Dict", "datetime", "diff_tqdm", "datasets",  # D
    "eb",  # E
    "faiss", "FileHandler",  # F
    "GPU_ACC", "gguf", # G
    "HARDWARE_INFO", "hashlib", "HTTPAdapter", "heapq", "huggingface_hub",  # H
    "io", "InconsistentValues", "Image", "importlib",  # I
    "json",  # J
    # K (无)
    "List", "locale", "logging", "libarchive",  # L
    "math", "MethodType", "mimetypes",  # M
    "np", "numpy", "njit", "numba", "NOT_IMPORT", # N
    "Optional", "os",  # O
    "partial", "Path", "pickle", "PurePosixPath", "pkgutil", "PDict",  # P
    "queue", "QueueHandler", "QueueListener", "quote",  # Q
    "random", "re", "replace", "requests", "Retry", "RotatingFileHandler", "rich_tqdm",  # R
    "shlex", "shutil", "SimpleNamespace", "struct", "secrets",  # S
    "ThreadPoolExecutor", "threading", "time", "tomllib", "tqdm_tqdm", "token_calibrator", "Type", # T
    "Union", "uuid",  # U
    # V, W (无)
    "xllamacpp", # X
    # Y (无)
    "tarfile", # T
    "zipfile", "zipimport", "zstd",  # Z
    "_prange", # _
]

# API 导入
if all(v is not None for v in [uvicorn, fastapi, slowapi]):
    __all__.extend([
        "fastapi", "UploadFile", "HTTPException", "status", "Depends", "Security", "Form", "Request", "BackgroundTasks",
        "FileResponse", "PlainTextResponse", "HTTPBearer", "HTTPAuthorizationCredentials", "CORSMiddleware", "Limiter",
        "_rate_limit_exceeded_handler", "get_remote_address", "RateLimitExceeded", "GZipMiddleware"])

if TYPE_CHECKING:
    from TranslatorPersistence import TranslationCache, VectorCache, TokenCalibratorCache
if TYPE_CHECKING:
    import TranslatorCore
    import TranslatorFile
    import TranslatorTool
    import TranslatorModule
    import TranslatorLog
    import TranslatorQuantization
    import TranslatorLocale
    import TranslatorBuilder
    import TranslatorIndex
    import TranslatorNetwork
    import TranslatorModpack
    import TranslatorMods

def Mods() -> "TranslatorMods":
    import TranslatorMods as _Class
    return _Class

import TranslatorIndex
IndexGSQ = TranslatorIndex.IndexGSQ
import TranslatorPersistence
import TranslatorQuantization
from TranslatorPersistence import TranslationCache, VectorCache, TokenCalibratorCache

import TranslatorLoop
Loop = TranslatorLoop.Loop
TranslatorLoop = TranslatorLoop.TranslatorLoop

def Locale(Config) -> "TranslatorLocale.Locale":
    from TranslatorLocale import Locale as _Class
    return _Class(Config)

def Quantization(Config) -> "TranslatorQuantization.Quantization":
    from TranslatorQuantization import Quantization as _Class
    return _Class(Config)

def Log(Config) -> "TranslatorLog.Log":
    from TranslatorLog import Log as _Class
    return _Class(Config)

def Module(Config) -> "TranslatorModule.Module":
    from TranslatorModule import Module as _Class
    return _Class(Config)

def File(Config) -> "TranslatorFile.File":
    from TranslatorFile import File as _Class
    return _Class(Config)


def Translator(Config) -> "TranslatorCore.Translator":
    from TranslatorCore import Translator as _Class
    return _Class(Config)

def Tool(Config) -> "TranslatorTool.Tool":
    from TranslatorTool import Tool as _Class
    return _Class(Config)

def Builder(Config) -> "TranslatorBuilder.Builder":
    from TranslatorBuilder import Builder as _Class
    return _Class(Config)

def Index(Config) -> "TranslatorIndex.Index":
    from TranslatorIndex import Index as _Class
    return _Class(Config)

def Network(Config) -> "TranslatorNetwork.Network":
    from TranslatorNetwork import Network as _Class
    return _Class(Config)

def Modpack(Config) -> "TranslatorModpack.Modpack":
    from TranslatorModpack import Modpack as _Class
    return _Class(Config)

from TranslatorConfig import RuntimeConfig, DEFAULT_CONFIG, DefaultConfig, Config

import TranslatorFile

__all__.extend([
    # A (无)
    "Builder", # B
    "Config", # C
    # E (无)
    "DEFAULT_CONFIG", "DefaultConfig", # D
    # E (无)
    "File", # F
    # G, H (无)
    "Index", "IndexGSQ", # I
    "加载模组", # J
    # K (无)
    "Locale", "Log", "Loop", # L
    "Module", "Modpack", "Mods", # M
    "Network", # N
    # O (无)
    "Parser", # P
    "Quantization", # Q
    "RuntimeConfig", # R
    # S (无)
    "Translator", "Tool", "TranslatorPersistence", "TranslationCache", "VectorCache", "TokenCalibratorCache", "TranslatorIndex", "TranslatorQuantization", "TranslatorFile", "TranslatorLoop", # T
    # U, V, W, X, Y, Z (无)
])


模组目录 = None
模组查找路径 = []
模组包 = types.ModuleType("mods")
模组包.__package__ = "mods"
sys.modules.setdefault("mods", 模组包)

class 压缩模组查找器:
    def __init__(Self, 压缩文件路径: str, 相对路径: str = ""):
        Self.压缩文件路径 = 压缩文件路径
        Self.相对路径 = 相对路径 or 压缩文件路径

    def find_spec(Self, 完整名称: str, 路径=None, 目标=None):
        if not 完整名称.startswith("mods."):
            return None
        模块名 = 完整名称[len("mods."):]
        if "." in 模块名 or not 模块名.isidentifier():
            return None
        来源 = f"{Self.压缩文件路径}/{模块名}.py"
        try:
            with zipfile.ZipFile(Self.压缩文件路径) as 压缩包:
                if f"{模块名}.py" not in 压缩包.namelist():
                    return None
        except Exception:
            return None
        return importlib.machinery.ModuleSpec(
            完整名称,
            importlib.machinery.SourceFileLoader(完整名称, 来源),
            origin=来源,
        )

def 加载压缩模组(压缩文件路径: Path):
    压缩文件路径 = Path(压缩文件路径).resolve()
    相对路径 = 压缩文件路径.relative_to(模组目录).as_posix()
    绝对路径 = str(压缩文件路径)
    with zipfile.ZipFile(压缩文件路径) as 压缩包:
        模块名列表 = [
            Path(名称).stem
            for 名称 in 压缩包.namelist()
            if 名称.endswith(".py")
            and "/" not in 名称.rstrip("/")
            and not 名称.startswith("_")
            and Path(名称).stem.isidentifier()
        ]
    if not 模块名列表:
        return []
    if 绝对路径 not in 模组查找路径:
        模组查找路径.append(绝对路径)
    if not any(getattr(查找器, "压缩文件路径", None) == 绝对路径 for 查找器 in sys.meta_path):
        sys.meta_path.append(压缩模组查找器(绝对路径, 相对路径))
    return 模块名列表

def 加载模组(配置, 实例=None):
    # 配置: DefaultConfig 实例 —— 核心基础设施(归档上限等)靠它取配置项
    # 实例: 带 Lang/Log 的实例(Config); 模组注册期报错要用它出本地化文案,
    #       不传就退回 配置 自己(它可能还没有 Lang, 那时注册期文案走中文兜底)
    global 模组目录, 模组查找路径
    模组路径 = Path(配置.MODS_PATH)
    模组目录 = 模组路径 if 模组路径.is_absolute() else Path(__file__).resolve().parent / 模组路径
    模组查找路径 = [str(模组目录)]
    模组包.__path__ = 模组查找路径
    模组包.__spec__ = importlib.machinery.ModuleSpec("mods", None, is_package=True)
    模组包.__spec__.submodule_search_locations = 模组查找路径
    Mods().绑定配置(配置)
    Mods().绑定实例(配置 if 实例 is None else 实例)
    Mods().绑定模组配置(getattr(配置, "MOD_CONFIG", None))
    try:
        for 压缩文件 in sorted(模组目录.glob("*.zip")):
            加载压缩模组(压缩文件)
    except Exception:
        eb.print_exc()

    for _, 模块名, _ in pkgutil.iter_modules(模组查找路径):
        if 模块名.startswith("_"):
            continue
        try:
            importlib.import_module(f'mods.{模块名}')
        except Exception:
            eb.print_exc()

Config.模组加载器 = staticmethod(加载模组)
