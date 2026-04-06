import multiprocessing as mp
import traceback as eb
import threading
import hashlib
import zipfile
import tomllib
import pickle
import random
import json
import glob
import time
import math
import ast
import os
import re
import io

from pathlib import Path, PurePosixPath
from functools import partial
from collections import defaultdict
from typing import Callable, Dict, Any, Union, Optional, List
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
    import uvicorn, fastapi
except ImportError:
    uvicorn, fastapi = None, None
try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None

GPU_ACC = os.getenv("FENGMANG_GPU_ACC", "true").lower() == "true"
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
if fastapi:
    APIConfigFile = Path("api-config.cfg").resolve()
    APIConfigFile.parent.mkdir(parents=True, exist_ok=True)
    if APIConfigFile.is_file():
        with open(APIConfigFile, "r", encoding="utf-8") as f:
            APIConfig = json.load(f)
    else:
        APIConfig = {
            "server": {},
            "api": {
                    "host": "127.0.0.1",
                    "port": 25561
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
    "replace", "random", "ConfigFactory", "HOCONConverter", "time",
    "GPU_ACC", "FastMCP", "MCPConfig", "numpy", "PurePosixPath",
    "tomllib", "glob", "APIConfig", "Union",
    "Optional", "List", "io", "uvicorn"]