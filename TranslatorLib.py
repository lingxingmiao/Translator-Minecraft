import multiprocessing as mp
import traceback as eb
import threading
import hashlib
import zipfile
import tomllib
import asyncio
import shutil
import pickle
import json
import glob
import time
import uuid
import math
import ast
import os
import re
import io

from typing import Callable, Dict, Any, Union, Optional, List
from pathlib import Path, PurePosixPath
from functools import partial
from contextlib import asynccontextmanager
from collections import defaultdict
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor, as_completed
#需要安装↓
import faiss
import numpy
#需要安装↓
from pyhocon import ConfigFactory, HOCONConverter
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from tqdm.rich import tqdm
#Codna需要安装↓
import requests
#可选服务安装↓
try:
    import uvicorn, fastapi, slowapi
except ImportError:
    uvicorn, fastapi, slowapi = None, None, None
try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None
    
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
        "error": ""
    }
    GPU_ERROR = False
except Exception as e:
    np = numpy
    HARDWARE_INFO = {
        "type": "CPU",
        "version": np.__version__,
        "error": e
    }
    
    GPU_ERROR = e
if FastMCP:
    MCPConfigFile = Path("config-mcp.cfg").resolve()
    MCPConfigFile.parent.mkdir(parents=True, exist_ok=True)
    if MCPConfigFile.is_file():
        with open(MCPConfigFile, "r", encoding="utf-8") as f:
            MCPConfig = json.load(f)
    else:
        MCPConfig = {
            "host": "127.0.0.1",
            "port": 25560
        }
        with open(MCPConfigFile, "w+", encoding="utf-8") as f:
            json.dump(MCPConfig, f, indent=4)
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
                    "current-limiting": "8/minute"
                },
            "keys": []
        }
        with open(APIConfigFile, "w+", encoding="utf-8") as f:
            json.dump(APIConfig, f, indent=4)
            
显示内容 = f"""[cyan]████████╗██████╗  █████╗ ███╗   ██╗███████╗██╗      █████╗ ███╗   ███╗ ██████╗
[cyan]╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║     ██╔══██╗████╗ ████║██╔════╝
   [cyan]██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ███████║██╔████╔██║██║
   [cyan]██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██║██║╚██╔╝██║██║
   [cyan]██║   ██║  ██║██║  ██║██║ ╚████║███████║███████╗██║  ██║██║ ╚═╝ ██║╚██████╗
   [cyan]╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝

TranslatorMinecraft Core
[bright_green]Version:[/] Release 1.4
[bright_green]GPU Accelerator:[/] {"True" if GPU_ACC else GPU_ERROR}"""
Console(force_terminal=True, color_system="auto").print(Panel(Align(显示内容, align="center"),title="[blue]TranslatorMinecraft Core[/blue]",border_style="blue",padding=(1, 2),width=110))

__all__ = [
    'np', "mp", "threading", "eb", "hashlib",
    "zipfile", "pickle", "json", "ast", "os",
    "re", "partial", "defaultdict", "Path", 'HARDWARE_INFO',
    "ThreadPoolExecutor", "as_completed", "Callable", "Dict", "Any",
    "faiss", "requests", "math", "tqdm", "dataclass",
    "replace", "ConfigFactory", "HOCONConverter", "time", "shutil",
    "GPU_ACC", "FastMCP", "MCPConfig", "numpy", "PurePosixPath",
    "tomllib", "glob", "APIConfig", "Union", "asynccontextmanager",
    "Optional", "List", "io", "asyncio", "uuid"]
