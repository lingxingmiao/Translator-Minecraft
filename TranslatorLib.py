import traceback as eb
import threading
import hashlib
import zipfile
import sqlite3
import tomllib
import asyncio
import logging
import shutil
import pickle
import random
import atexit
import queue
import time
import uuid
import math
import ast
import os
import re
import io

from types import SimpleNamespace
from urllib.parse import quote
from typing import Callable, Dict, Any, Union, Optional, List
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path, PurePosixPath
from requests.adapters import HTTPAdapter
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from dataclasses import dataclass, replace
#需要安装↓
import numpy
import faiss
import ujson as json
import snbtlib
#需要安装↓
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.style import Style
from rich.color import Color
from tqdm.rich import tqdm
#from tqdm import tqdm
#Codna需要安装↓
import requests
#可选服务安装↓
try:
    import uvicorn, fastapi, slowapi
except ImportError:
    uvicorn, fastapi, slowapi = None, None, None
    
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
        
GPU_ACC = False
GPU_ERROR = None
HARDWARE_INFO = {"device_count": "1", "device_id": None}
try:
    if not Config["GPU_Accelerator"]:
        raise ValueError("GPU_ACC is set to False")
    
    import cupy as np
    if Config["GPU_Device_ID"] is not None:
        np.cuda.runtime.setDevice(Config["GPU_Device_ID"])
    
    np.dot(np.random.rand(2, 2), np.random.rand(2, 2))
    if not np.cuda.is_available():
        raise ValueError("CUDA not available")
    
    HARDWARE_INFO = {
        "type": "GPU",
        "version": np.__version__,
        "device_count": np.cuda.runtime.getDeviceCount(),
        "device_id": np.cuda.runtime.getDevice(),
        "error": ""
    }
    
    GPU_ACC = True
except ValueError:
    np = numpy
    HARDWARE_INFO = {
        "type": "CPU",
        "version": np.__version__,
        "device_count": None,
        "device_id": None,
        "error": ""
    }
    GPU_ERROR = False
except Exception as e:
    np = numpy
    HARDWARE_INFO = {
        "type": "CPU",
        "version": np.__version__,
        "device_count": None,
        "device_id": None,
        "error": e
    }
    
    GPU_ERROR = e
if uvicorn and fastapi and slowapi:
    APIConfigFile = Path("config-api.cfg").resolve()
    APIConfigFile.parent.mkdir(parents=True, exist_ok=True)
    if APIConfigFile.is_file():
        with open(APIConfigFile, "r", encoding="utf-8") as f:
            APIConfig = json.load(f)
    else:
        APIConfig = {
            "server": {
                    "LLM_API_URL": "",
                    "LLM_API_KEY": "",
                    "LLM_MODEL": "",
                    "EMB_API_URL": "",
                    "EMB_API_KEY": "",
                    "EMB_MODEL": "",
                    "LOGS_GLOBAL": False
                },
            "api": {
                    "host": "127.0.0.1",
                    "port": 25561,
                    "ssl_keyfile": None,
                    "ssl_certfile": None,
                    "max_concurrent": 4,
                    "current-limiting": "8/minute",
                    "task_states_file": "task_states.db",
                    "task_states_save_interval": 30.0,
                    "task_states_cleanup_hours": 24.0,
                    "task_states_cleanup_interval": 300.0,
                    "transalator_file_exists_del": True
                },
            "keys": []
        }
        with open(APIConfigFile, "w+", encoding="utf-8") as f:
            json.dump(APIConfig, f, indent=4)

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

#Pagga
文本 = 彩色文本("""████████╗██████╗  █████╗ ███╗   ██╗███████╗██╗      █████╗ ███╗   ███╗ ██████╗
╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║     ██╔══██╗████╗ ████║██╔════╝
   ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ███████║██╔████╔██║██║
   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██║██║╚██╔╝██║██║
   ██║   ██║  ██║██║  ██║██║ ╚████║███████║███████╗██║  ██║██║ ╚═╝ ██║╚██████╗
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝""")

信息文本 = Text.from_markup(f"""
[bold]TranslatorMinecraft Core[/bold]
[bright_green]Version:[/] Release 1.5
[bright_green]GPU Accelerator:[/] {"True" if GPU_ACC else GPU_ERROR}""")

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
    'np', "threading", "eb", "hashlib",
    "zipfile", "pickle", "json", "ast", "os",
    "re", "partial", "defaultdict", "Path", 'HARDWARE_INFO',
    "ThreadPoolExecutor", "as_completed", "Callable", "Dict", "Any",
    "requests", "math", "tqdm", "dataclass", "faiss",
    "replace", "snbtlib", "time", "shutil", "atexit",
    "GPU_ACC", "numpy", "PurePosixPath", "random", "deque",
    "tomllib", "APIConfig", "Union", "asynccontextmanager",
    "Optional", "List", "io", "asyncio", "uuid",
    "sqlite3", "HTTPAdapter", "SimpleNamespace", "queue", "quote",
    "logging", "QueueHandler", "QueueListener", "RotatingFileHandler",]
