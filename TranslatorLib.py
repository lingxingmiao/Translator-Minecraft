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
#Codna需要安装↓
import requests
#Codna需要安装↓
from tqdm import tqdm
#可选服务安装↓
try:
    import uvicorn, fastapi, slowapi
except ImportError:
    uvicorn, fastapi, slowapi = None, None, None
try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None

GPU_ACC = os.getenv("FENGMANG_GPU_ACC", "true").lower() == "false"
GPU_DEVICE_ID = os.getenv("FENGMANG_GPU_DEVICE_ID", None)
if GPU_DEVICE_ID is not None:
    GPU_DEVICE_ID = int(GPU_DEVICE_ID)

try:
    if not GPU_ACC:
        raise ValueError("GPU_ACC is set to False")
    
    import cupy as np
    if GPU_DEVICE_ID is not None:
        np.cuda.runtime.setDevice(GPU_DEVICE_ID)
    
    np.dot(np.random.rand(2, 2), np.random.rand(2, 2))
    if not np.cuda.is_available():
        raise ValueError("CUDA not available")
    
    HARDWARE_INFO = {
        "type": "GPU",
        "version": np.__version__,
        "device_count": np.cuda.runtime.getDeviceCount(),
        "device_id": np.cuda.runtime.getDevice(),
        "error": str("")
    }
    
    GPU_ACC = True
except Exception as e:
    import numpy as np
    HARDWARE_INFO = {
        "type": "CPU",
        "version": np.__version__,
        "error": str(e)
    }
    
    GPU_ACC = False
if FastMCP:
    MCPConfigFile = Path("mcp-config.cfg").resolve()
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
    APIConfigFile = Path("api-config.cfg").resolve()
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
__all__ = [
    'np', "mp", "threading", "eb", "hashlib",
    "zipfile", "pickle", "json", "ast", "os",
    "re", "partial", "defaultdict", "Path", 'HARDWARE_INFO',
    "ThreadPoolExecutor", "as_completed", "Callable", "Dict", "Any",
    "faiss", "requests", "math", "tqdm", "dataclass",
    "replace", "ConfigFactory", "HOCONConverter", "time",
    "GPU_ACC", "FastMCP", "MCPConfig", "numpy", "PurePosixPath",
    "tomllib", "glob", "APIConfig", "Union",
    "Optional", "List", "io", "asyncio",
    "asynccontextmanager", "uuid", "shutil"]
