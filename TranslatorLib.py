import traceback as eb
import threading
import hashlib
import zipfile
import sqlite3
import tomllib
import asyncio
import logging
import clr; clr.AddReference("System"); import System  # type: ignore
import shutil
import locale
import bisect
import pickle
import random
import atexit
import dnfile
import queue
import shlex
import time
import uuid
import math
import ast
import re
import io

from io import BytesIO
from re import compile as _re_compile, sub as _re_sub
from json import dumps as _json_dumps, loads as _json_loads
from enum import IntEnum
from types import SimpleNamespace
from urllib.parse import quote
from typing import Callable, Dict, Any, Union, Optional, List, TextIO
from urllib3.util.retry import Retry
from logging import FileHandler
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path, PurePosixPath
from requests.adapters import HTTPAdapter
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, replace
#需要安装↓
import numpy
import faiss
import ujson as json
import dnfile
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
            
class snbtlib:
    @staticmethod
    def loads(s: str, format: bool = False) -> Any:
        tokens = snbtlib._tokenize(s)
        idx = [0]
        
        def parse_value():
            if idx[0] >= len(tokens): return None
            t = tokens[idx[0]]
            if t['type'] == 'BEGIN_DICT':
                idx[0] += 1; return snbtlib._parse_dict(tokens, idx)
            elif t['type'] == 'BEGIN_LIST':
                idx[0] += 1; return snbtlib._parse_list(tokens, idx)
            idx[0] += 1; return t['value']
            
        result = parse_value()
        return _json_dumps(result, ensure_ascii=False, indent=4) if format else result

    @staticmethod
    def dumps(obj: Any, indent: int = 0, compact: bool = False) -> str:
        if isinstance(obj, str): obj = _json_loads(obj)
        lines = []
        snbtlib._serialize(obj, lines, indent)
        text = ''.join(lines)
        return text if not compact else snbtlib._compatible(text)

    @staticmethod
    def load(fp: TextIO, **kwargs) -> Any:
        return snbtlib.loads(fp.read(), **kwargs)

    @staticmethod
    def dump(obj: Any, fp: TextIO, **kwargs):
        fp.write(snbtlib.dumps(obj, **kwargs))

    @staticmethod
    def _tokenize(text: str) -> List[Dict[str, Any]]:
        text = text.replace('\r', '')
        text = _re_sub(_re_compile(r'^\s+?//.*$', _re_compile.MULTILINE), '', text)
        text = _re_sub(_re_compile(r'^\s+?#.*$', _re_compile.MULTILINE), '', text)
        
        tokens, i, n = [], 0, len(text)
        while i < n:
            ch = text[i]
            if ch.isspace() and ch != '\n': i += 1; continue
            if ch in '{[:]}:,': tokens.append(snbtlib._mk_token(ch, ch)); i += 1; continue
            if ch in '-0123456789':
                j = i
                while j < n and (text[j] in '-0123456789.eE' or (text[j] in '+-' and j > i and text[j-1] in 'eE')): j += 1
                tokens.append({'type': 'NUMBER', 'value': '$number$' + text[i:j]}); i = j; continue
            if ch == '"':
                j, buf = i + 1, []
                while j < n:
                    if text[j] == '\\' and j + 1 < n: buf.append(text[j+1]); j += 2
                    elif text[j] == '"': break
                    else: buf.append(text[j]); j += 1
                tokens.append({'type': 'STRING_QUOTED', 'value': ''.join(buf)}); i = j + 1; continue
            if ch.isalpha() or ch == '_':
                j = i
                while j < n and (text[j].isalnum() or text[j] in '_'): j += 1
                word = text[i:j]
                if word in ('true', 'false'): tokens.append({'type': 'BOOL', 'value': word == 'true'})
                else: tokens.append({'type': 'STRING', 'value': word})
                i = j; continue
            if ch == '\n': tokens.append({'type': 'ENTER', 'value': '\n'}); i += 1; continue
            i += 1
        return tokens

    @staticmethod
    def _mk_token(ch: str, val: str) -> Dict[str, Any]:
        mapping = {'{': 'BEGIN_DICT', '}': 'END_DICT', '[': 'BEGIN_LIST', ']': 'END_LIST', ':': 'COLON', ',': 'ENTER', '\n': 'ENTER'}
        return {'type': mapping.get(ch, 'UNKNOWN'), 'value': val}

    @staticmethod
    def _parse_dict(tokens: List[Dict[str, Any]], idx: List[int]) -> Dict[str, Any]:
        result = {}
        while idx[0] < len(tokens):
            t = tokens[idx[0]]
            if t['type'] == 'END_DICT': idx[0] += 1; break
            if t['type'] in ('STRING', 'STRING_QUOTED', 'NUMBER'):
                key = t['value']
                if key.startswith('$number$'): key = key[8:]
                if t['type'] == 'STRING_QUOTED': key = f'"{key}"'
                idx[0] += 1
                if idx[0] < len(tokens) and tokens[idx[0]]['type'] == 'COLON': idx[0] += 1
                if idx[0] < len(tokens):
                    vt = tokens[idx[0]]
                    if vt['type'] == 'BEGIN_DICT': idx[0] += 1; result[key] = snbtlib._parse_dict(tokens, idx)
                    elif vt['type'] == 'BEGIN_LIST': idx[0] += 1; result[key] = snbtlib._parse_list(tokens, idx)
                    else: idx[0] += 1; result[key] = vt['value']
            else: idx[0] += 1
        return result

    @staticmethod
    def _parse_list(tokens: List[Dict[str, Any]], idx: List[int]) -> Any:
        if idx[0] < len(tokens) and tokens[idx[0]]['value'] == 'B':
            idx[0] += 1
            if idx[0] < len(tokens) and tokens[idx[0]]['type'] == 'COLON': idx[0] += 1
            result = b''
            while idx[0] < len(tokens):
                t = tokens[idx[0]]
                if t['type'] == 'END_LIST': idx[0] += 1; return result
                if t['type'] == 'NUMBER':
                    val = t['value'][8:] if t['value'].startswith('$number$') else t['value']
                    if val.endswith('b'): val = val[:-1]
                    try: result += int(val).to_bytes(1, 'big')
                    except: pass
                idx[0] += 1
            return result
        result = []
        while idx[0] < len(tokens):
            t = tokens[idx[0]]
            if t['type'] == 'END_LIST': idx[0] += 1; break
            if t['type'] == 'BEGIN_DICT': idx[0] += 1; result.append(snbtlib._parse_dict(tokens, idx))
            elif t['type'] == 'BEGIN_LIST': idx[0] += 1; result.append(snbtlib._parse_list(tokens, idx))
            elif t['type'] in ('STRING', 'STRING_QUOTED', 'NUMBER', 'BOOL'): result.append(t['value']); idx[0] += 1
            else: idx[0] += 1
        return result

    @staticmethod
    def _serialize(obj: Any, out: List[str], indent: int):
        tab = '\t' * indent
        if isinstance(obj, dict):
            if not obj: out.append('{ }\n')
            else:
                out.append('{\n')
                for k, v in obj.items():
                    out.append(tab + '\t' + str(k) + ': ')
                    snbtlib._serialize_value(v, out, indent + 1)
                out.append(tab + '}\n')
        elif isinstance(obj, list):
            if not obj: out.append('[ ]\n')
            elif len(obj) == 1 and not isinstance(obj[0], (dict, list, bytes)):
                inner = []; snbtlib._serialize_value(obj[0], inner, indent); out.append('[' + ''.join(inner).strip() + ']\n')
            else:
                if isinstance(obj[0], str) and obj[0] == 'I;': out.append('[I;\n'); items = obj[1:]
                else: out.append('[\n'); items = obj
                for v in items: out.append(tab + '\t'); snbtlib._serialize_value(v, out, indent + 1)
                out.append(tab + ']\n')
        else: snbtlib._serialize_value(obj, out, indent)

    @staticmethod
    def _serialize_value(val: Any, out: List[str], indent: int):
        if isinstance(val, (dict, list)): snbtlib._serialize(val, out, indent)
        elif isinstance(val, str): out.append(val[8:] + '\n' if val.startswith('$number$') else f'"{val}"\n')
        elif isinstance(val, bool): out.append(('true' if val else 'false') + '\n')
        elif isinstance(val, bytes):
            out.append("[B;\n")
            for b in val: out.append('\t' * (indent + 1) + str(b) + 'b\n')
            out.append('\t' * indent + ']\n')
        elif isinstance(val, (int, float)): out.append(str(val) + '\n')
        else: out.append(str(val) + '\n')

    @staticmethod
    def _compatible(text: str) -> str:
        if not text: return ''
        lines = text.splitlines()
        for i in range(len(lines) - 1):
            curr, nxt = lines[i].rstrip(), lines[i + 1].lstrip()
            if curr and nxt and curr[-1] not in '[{' and nxt[0] not in ']}': lines[i] = curr + ','
        return '\n'.join(lines)
    
class fancymenulib:
    """
    FancyMenu 布局文件解析与生成器。

    用法：
        data = fancymenulib.loads(text)
        text = fancymenulib.dumps(data)
        data = fancymenulib.load(open('layout.txt'))
        fancymenulib.dump(data, open('layout.txt', 'w'))
    """

    @staticmethod
    def loads(text: str, auto_convert: bool = True) -> Dict[str, Any]:
        """将布局文本解析为字典。"""
        lines = [l.strip() for l in text.splitlines() if l.strip() != '']
        return fancymenulib._parse(text, lines, auto_convert)

    @staticmethod
    def dumps(obj: Dict[str, Any], indent: int = 0) -> str:
        """将字典序列化为 FancyMenu 布局文本。"""
        return fancymenulib._serialize(obj, indent)

    @staticmethod
    def load(fp: TextIO, auto_convert: bool = True) -> Dict[str, Any]:
        """从文件对象读取并解析。"""
        return fancymenulib.loads(fp.read(), auto_convert)

    @staticmethod
    def dump(obj: Dict[str, Any], fp: TextIO, indent: int = 0):
        """将对象写入文件。"""
        fp.write(fancymenulib.dumps(obj, indent))

    @staticmethod
    def _parse(text: str, lines: List[str], auto_convert: bool) -> Dict[str, Any]:
        it = iter(lines)
        return fancymenulib._parse_block(it, auto_convert)

    @staticmethod
    def _parse_block(it, auto_convert: bool) -> OrderedDict:
        raw = OrderedDict()
        for line in it:
            line = line.strip()
            if line == '}':
                break

            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]

                converted = fancymenulib._convert_value(value) if auto_convert else value
                raw.setdefault(key, []).append(converted)
                continue

            if line.endswith('{'):
                block_name = line[:-1].strip()
                block_content = fancymenulib._parse_block(it, auto_convert)
                raw.setdefault(block_name, []).append(block_content)
                continue

        return fancymenulib._unwrap(raw)

    @staticmethod
    def _unwrap(raw: OrderedDict) -> OrderedDict:
        """将长度为1的列表拆包为单值。"""
        result = OrderedDict()
        for k, vlist in raw.items():
            result[k] = vlist[0] if len(vlist) == 1 else vlist
        return result

    @staticmethod
    def _convert_value(value: str) -> Any:
        low = value.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @staticmethod
    def _serialize(obj: Dict[str, Any], indent: int = 0) -> str:
        lines = []
        fancymenulib._serialize_block(obj, lines, indent)
        return '\n'.join(lines) + '\n'

    @staticmethod
    def _serialize_block(obj: Dict[str, Any], lines: List[str], indent_level: int):
        prefix = '\t' * indent_level
        for key, val in obj.items():
            if isinstance(val, list):
                for item in val:
                    fancymenulib._write_field(key, item, lines, prefix)
            else:
                fancymenulib._write_field(key, val, lines, prefix)

    @staticmethod
    def _write_field(key: str, val, lines: List[str], prefix: str):
        if isinstance(val, dict):
            lines.append(f'{prefix}{key} {{')
            fancymenulib._serialize_block(val, lines, len(prefix) + 1)
            lines.append(f'{prefix}}}')
        else:
            formatted = fancymenulib._format_value(val)
            lines.append(f'{prefix}{key} = {formatted}')

    @staticmethod
    def _format_value(val) -> str:
        if isinstance(val, bool):
            return 'true' if val else 'false'
        if isinstance(val, (int, float)):
            return str(val)
        return str(val)
    
# ============================================================
# BIT MASK 表
# ============================================================
BIT_MASK = [0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff,
            0x1ff, 0x3ff, 0x7ff, 0xfff, 0x1fff, 0x3fff, 0x7fff, 0xffff,
            0x1ffff, 0x3ffff, 0x7ffff, 0xfffff, 0x1fffff, 0x3fffff, 0x7fffff,
            0xffffff, 0x1ffffff, 0x3ffffff, 0x7ffffff, 0xfffffff, 0x1fffffff,
            0x3fffffff, 0x7fffffff, 0xffffffff]


# ============================================================
# FileVersion 枚举
# ============================================================
class _FileVersion(IntEnum):
    INITIAL = 0
    QUESTS = 1
    SETS = 2
    LORE = 3
    UNCOMPLETED_DISABLED = 4
    LORE_AUDIO = 5
    BAGS = 6
    LOCK = 7
    BAG_LIMITS = 8
    TEAMS = 9
    TEAM_SETTINGS = 10
    DEATHS = 11
    REMOVED_QUESTS = 12
    REPEATABLE_QUESTS = 13
    TRIGGER_QUESTS = 14
    OPTION_LINKS = 15
    NO_ITEM_IDS = 16
    NO_ITEM_IDS_FIX = 17
    PARENT_COUNT = 18
    REPUTATION = 19
    REPUTATION_KILL = 20
    REPUTATION_BARS = 21
    CUSTOM_PRECISION_TYPES = 22
    COMMAND_REWARDS = 23
    LATEST = COMMAND_REWARDS

    def contains(self, other: '_FileVersion') -> bool:
        return self >= other
    def lacks(self, other: '_FileVersion') -> bool:
        return self < other


# ============================================================
# 字段位宽定义
# ============================================================
class _Bits:
    BYTE = 8
    SHORT = 16
    INT = 32
    BOOLEAN = 1
    EMPTY = 0
    NBT_LENGTH = 15
    PACKET_ID = 5
    NAME_LENGTH = 5
    PLAYERS = 16
    QUESTS = 10
    TASKS = 4
    REWARDS = 3
    QUEST_SETS = 5
    ITEM_PROGRESS = 30
    QUEST_NAME_LENGTH = 5
    QUEST_DESCRIPTION_LENGTH = 16
    QUEST_POS_X = 9
    QUEST_POS_Y = 8
    TASK_TYPE = 4
    TASK_ITEM_COUNT = 6
    TASK_REQUIREMENT = 32
    QUEST_REWARD = 3
    ITEM_PRECISION = 30
    GROUP_ITEMS = 6
    GROUP_COUNT = 10
    TIER_COUNT = 7
    WEIGHT = 19
    COLOR = 4
    PASS_CODE = 7
    LIMIT = 10
    TEAMS = 10
    TEAM_ACTION_ID = 4
    LIVES = 8
    TEAM_ERROR = 2
    TEAM_REWARD_SETTING = 2
    TEAM_LIVES_SETTING = 1
    OP_ACTION = 3
    BAG_TIER = 3
    DEATHS = 12
    TEAM_PROGRESS = 7
    TASK_LOCATION_COUNT = 3
    WORLD_COORDINATE = 32
    LOCATION_VISIBILITY = 2
    TICKS = 10
    HOURS = 32
    REPEAT_TYPE = 2
    TRIGGER_TYPE = 2
    TASK_MOB_COUNT = 3
    KILL_COUNT = 16
    MOB_ID_LENGTH = 10
    TRACKER_TYPE = 2
    PORTAL_TYPE = 2
    REPUTATION = 8
    REPUTATION_VALUE = 32
    REPUTATION_REWARD = 3
    REPUTATION_SETTING = 3
    REPUTATION_MARKER = 5

    @staticmethod
    def quests(ver: _FileVersion) -> int:
        return 7 if ver.lacks(_FileVersion.SETS) else 10
    @staticmethod
    def players(ver: _FileVersion) -> int:
        return 10 if ver.lacks(_FileVersion.REPEATABLE_QUESTS) else 16
    @staticmethod
    def task_type(ver: _FileVersion) -> int:
        return 3 if ver.lacks(_FileVersion.REPUTATION_KILL) else 4
    @staticmethod
    def item_precision(ver: _FileVersion) -> int:
        return 2 if ver.lacks(_FileVersion.CUSTOM_PRECISION_TYPES) else 30


# ============================================================
# _BitReader
# ============================================================
class _BitReader:
    def __init__(self, data: bytes):
        self._s = BytesIO(data)
        self._buf = 0
        self._bits = 0
        self._ver: _FileVersion = _FileVersion.INITIAL
        self._enc = 'utf-8'

    def set_encoding(self, enc: str):
        self._enc = enc

    def _rd(self) -> int:
        raw = self._s.read(1)
        return raw[0] if raw else 0

    def read_byte(self) -> int:
        return self.read_data(_Bits.BYTE)

    def read_bool(self) -> bool:
        return self.read_data(_Bits.BOOLEAN) != 0

    def read_data(self, n: int) -> int:
        if n == 0:
            return 0
        data = 0
        read = 0
        while True:
            left = n - read
            if self._bits >= left:
                data |= (self._buf & BIT_MASK[left]) << read
                self._buf >>= left
                self._bits -= left
                read += left
                break
            else:
                data |= self._buf << read
                read += self._bits
                self._buf = self._rd()
                self._bits = 8
        return data

    def read_version(self) -> _FileVersion:
        self._ver = _FileVersion(self.read_byte())
        return self._ver

    @property
    def version(self) -> _FileVersion:
        return self._ver

    @property
    def encoding(self) -> str:
        return self._enc

    def read_str(self, bits: int) -> Optional[str]:
        n = self.read_data(bits)
        if n == 0:
            return None
        return bytearray(self.read_byte() for _ in range(n)).decode(self._enc, errors='replace')

    def read_nbt(self) -> Optional[str]:
        if not self.read_bool():
            return None
        return bytearray(self.read_byte() for _ in range(self.read_data(_Bits.NBT_LENGTH))).hex()

    def read_item(self) -> Optional[str]:
        if self._ver.contains(_FileVersion.NO_ITEM_IDS):
            return self.read_str(_Bits.SHORT)
        self.read_data(_Bits.SHORT)
        return None

    def read_stack(self, use_size: bool = False) -> Optional[dict]:
        name = self.read_item()
        if name is None:
            return None
        size = self.read_data(_Bits.SHORT) if use_size else 1
        damage = self.read_data(_Bits.SHORT)
        return {"item": name, "damage": damage, "count": size, "nbt": self.read_nbt()}


# ============================================================
# _BitWriter
# ============================================================
class _BitWriter:
    def __init__(self):
        self._s = BytesIO()
        self._buf = 0
        self._bits = 0
        self._enc = 'utf-8'

    def set_encoding(self, enc: str):
        self._enc = enc

    def write_byte(self, v: int):
        self.write_data(v, _Bits.BYTE)

    def write_bool(self, v: bool):
        self.write_data(1 if v else 0, _Bits.BOOLEAN)

    def write_data(self, v: int, n: int):
        if n == 0:
            return
        v &= BIT_MASK[n]
        while True:
            space = 8 - self._bits
            if n <= space:
                self._buf |= v << self._bits
                self._bits += n
                break
            else:
                self._buf |= (v & BIT_MASK[space]) << self._bits
                self._s.write(bytes([self._buf]))
                v >>= space
                n -= space
                self._buf = 0
                self._bits = 0

    def flush(self):
        if self._bits > 0:
            self._s.write(bytes([self._buf]))
        self._bits = 0
        self._buf = 0

    def get_bytes(self) -> bytes:
        self.flush()
        return self._s.getvalue()

    def write_str(self, s: Optional[str], bits: int):
        if s is None:
            self.write_data(0, bits)
        else:
            try:
                b = s.encode(self._enc)
            except (UnicodeEncodeError, UnicodeDecodeError):
                b = s.encode('utf-8', errors='replace')
            n = min(len(b), (1 << bits) - 1)
            self.write_data(n, bits)
            for i in range(n):
                self.write_byte(b[i])

    def write_nbt(self, hex_str: Optional[str]):
        if hex_str is None:
            self.write_bool(False)
        else:
            self.write_bool(True)
            raw = bytes.fromhex(hex_str)
            self.write_data(len(raw), _Bits.NBT_LENGTH)
            for b in raw:
                self.write_byte(b)

    def write_item(self, name: Optional[str], ver: _FileVersion):
        if ver.contains(_FileVersion.NO_ITEM_IDS):
            self.write_str(name, _Bits.SHORT)
        else:
            self.write_data(0, _Bits.SHORT)

    def write_stack(self, stack: Optional[dict], use_size: bool, ver: _FileVersion):
        if stack is None:
            self.write_item(None, ver)
            return
        self.write_item(stack.get("item"), ver)
        if use_size:
            self.write_data(stack.get("count", 1), _Bits.SHORT)
        self.write_data(stack.get("damage", 0), _Bits.SHORT)
        self.write_nbt(stack.get("nbt"))


# ============================================================
# hqmlib - 主 API
# ============================================================
class hqmlib:
    """
    用法：
        data = hqmlib.load('quests.hqm', encoding='gbk')
        data['quests'][0]['name'] = '新任务名'
        hqmlib.dump_to_hqm(data, 'quests_new.hqm')
        hqmlib.dump(data, 'output.json')
    """

    # ---- 读取 ----

    @staticmethod
    def loads(data: bytes, encoding: str = 'utf-8') -> Dict[str, Any]:
        r = _BitReader(data)
        r.set_encoding(encoding)
        ver = r.read_version()
        result: Dict[str, Any] = {"_file_version": ver.name, "_file_version_ordinal": ver.value}

        if ver.contains(_FileVersion.LOCK):
            result["lock_code"] = r.read_str(_Bits.PASS_CODE)
        result["main_description"] = r.read_str(_Bits.QUEST_DESCRIPTION_LENGTH) if ver.contains(_FileVersion.LORE) else "No description"
        result["quest_sets"] = hqmlib._parse_sets(r, ver)
        result["reputation"] = hqmlib._parse_reputation(r, ver)
        result["quests"] = hqmlib._parse_quests(r, ver)
        result["bag_tiers"] = hqmlib._parse_tiers(r, ver)
        result["bag_groups"] = hqmlib._parse_groups(r, ver)
        return result

    @staticmethod
    def load(path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        with open(path, 'rb') as f:
            return hqmlib.loads(f.read(), encoding)

    # ---- JSON 导出 ----

    @staticmethod
    def dumps(obj: Dict[str, Any], indent: int = 2) -> str:
        return json.dumps(obj, indent=indent, ensure_ascii=False)

    @staticmethod
    def dump(obj: Dict[str, Any], path: str, indent: int = 2):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(hqmlib.dumps(obj, indent))

    # ---- HQM 写入 ----

    @staticmethod
    def dumps_to_hqm(obj: Dict[str, Any], encoding: str = 'utf-8', version: Optional[int] = None) -> bytes:
        w = _BitWriter()
        w.set_encoding(encoding)
        if version is not None:
            ver = _FileVersion(version)
        else:
            ver_id = obj.get("_file_version_ordinal", _FileVersion.LATEST.value)
            ver = _FileVersion(ver_id)
        w.write_byte(ver.value)

        if ver.contains(_FileVersion.LOCK):
            w.write_str(obj.get("lock_code"), _Bits.PASS_CODE)
        if ver.contains(_FileVersion.LORE):
            w.write_str(obj.get("main_description", "No description"), _Bits.QUEST_DESCRIPTION_LENGTH)

        hqmlib._write_sets(w, obj.get("quest_sets", []), ver)
        hqmlib._write_reputation(w, obj.get("reputation", []), ver)
        hqmlib._write_quests(w, obj.get("quests", []), ver)
        hqmlib._write_tiers(w, obj.get("bag_tiers", []), ver)
        hqmlib._write_groups(w, obj.get("bag_groups", []), ver)
        return w.get_bytes()

    @staticmethod
    def dump_to_hqm(obj: Dict[str, Any], path: str, encoding: str = 'utf-8', version: Optional[int] = None):
        with open(path, 'wb') as f:
            f.write(hqmlib.dumps_to_hqm(obj, encoding, version))

    # ================================================================
    # 读取内部方法
    # ================================================================

    @staticmethod
    def _parse_sets(r, ver):
        enc = r.encoding
        sets = []
        if ver.contains(_FileVersion.SETS):
            for i in range(r.read_data(_Bits.QUEST_SETS)):
                s = {"id": i, "name": r.read_str(_Bits.QUEST_NAME_LENGTH),
                     "description": r.read_str(_Bits.QUEST_DESCRIPTION_LENGTH), "reputation_bars": []}
                if ver.contains(_FileVersion.REPUTATION_BARS):
                    for _ in range(r.read_data(_Bits.BYTE)):
                        s["reputation_bars"].append(r.read_data(_Bits.INT))
                sets.append(s)
        else:
            sets.append({"id": 0, "name": "Automatically generated",
                         "description": "This set was automatically generated.", "reputation_bars": []})
        return sets

    @staticmethod
    def _parse_reputation(r, ver):
        reps = []
        if ver.contains(_FileVersion.REPUTATION):
            for _ in range(r.read_data(_Bits.REPUTATION)):
                markers = []
                for _ in range(r.read_data(_Bits.REPUTATION_MARKER)):
                    markers.append({"name": r.read_str(_Bits.QUEST_NAME_LENGTH),
                                    "value": r.read_data(_Bits.REPUTATION_VALUE)})
                reps.append({"id": r.read_data(_Bits.REPUTATION), "name": r.read_str(_Bits.QUEST_NAME_LENGTH),
                             "neutral_name": r.read_str(_Bits.QUEST_NAME_LENGTH), "markers": markers})
        return reps

    @staticmethod
    def _parse_quests(r, ver):
        quests = []
        for qid in range(r.read_data(_Bits.quests(ver))):
            if not r.read_bool():
                continue
            q = {"id": qid, "name": r.read_str(_Bits.QUEST_NAME_LENGTH),
                 "description": r.read_str(_Bits.QUEST_DESCRIPTION_LENGTH),
                 "x": r.read_data(_Bits.QUEST_POS_X), "y": r.read_data(_Bits.QUEST_POS_Y),
                 "big_icon": r.read_bool(),
                 "quest_set_id": r.read_data(_Bits.QUEST_SETS) if ver.contains(_FileVersion.SETS) else 0}

            q["icon"] = r.read_stack(False) if (ver.contains(_FileVersion.SETS) and r.read_bool()) else None

            q["requirements"] = []
            if r.read_bool():
                q["requirements"] = [r.read_data(_Bits.QUESTS) for _ in range(r.read_data(_Bits.QUESTS))]

            q["option_links"] = []
            if ver.contains(_FileVersion.OPTION_LINKS) and r.read_bool():
                q["option_links"] = [r.read_data(_Bits.QUESTS) for _ in range(r.read_data(_Bits.QUESTS))]

            q["repeat_info"] = hqmlib._parse_repeat(r, ver)
            if ver.contains(_FileVersion.TRIGGER_QUESTS):
                q["trigger_type"] = r.read_data(_Bits.TRIGGER_TYPE)
                if q["trigger_type"] == 2:  # TASK_TRIGGER — isUseTaskCount() 返回 true
                    q["trigger_tasks"] = r.read_data(_Bits.TASKS)
            else:
                q["trigger_type"] = 0

            if ver.contains(_FileVersion.PARENT_COUNT) and r.read_bool():
                q["parent_requirement_count"] = r.read_data(_Bits.QUESTS)

            q["tasks"] = hqmlib._parse_tasks(r, ver)
            q["rewards"] = hqmlib._parse_rewards(r, ver)
            q["reward_choices"] = hqmlib._parse_rewards(r, ver)
            q["command_rewards"] = hqmlib._parse_cmd_rewards(r, ver)

            q["reputation_rewards"] = []
            for _ in range(r.read_data(_Bits.REPUTATION_REWARD)):
                q["reputation_rewards"].append({"reputation_id": r.read_data(_Bits.REPUTATION),
                                                "value": r.read_data(_Bits.REPUTATION_VALUE)})
            quests.append(q)
        return quests

    @staticmethod
    def _parse_repeat(r, ver):
        if ver.lacks(_FileVersion.REPEATABLE_QUESTS):
            return {"type": 0}
        info = {"type": r.read_data(_Bits.REPEAT_TYPE)}
        if info["type"] == 1:
            info["hours"] = r.read_data(_Bits.HOURS)
        elif info["type"] == 2:
            info["days"] = r.read_data(_Bits.HOURS)
        return info

    @staticmethod
    @staticmethod
    def _parse_tasks(r, ver):
        """
        Java TaskType 枚举序数:
          0=CONSUME  1=CRAFT  2=LOCATION  3=CONSUME_QDS
          4=DETECT   5=KILL   6=DEATH     7=REPUTATION  8=REPUTATION_KILL
        类型 0,1,3,4 均继承 QuestTaskItems，共享 save/load(items)
        """
        tasks = []
        for _ in range(r.read_data(_Bits.TASKS)):
            tt = r.read_data(_Bits.task_type(ver))
            task = {"type": tt, "description": r.read_str(_Bits.QUEST_NAME_LENGTH),
                    "long_description": r.read_str(_Bits.QUEST_DESCRIPTION_LENGTH)}
            if tt in (0, 1, 3, 4):  # CONSUME / CRAFT / CONSUME_QDS / DETECT → items
                task["items"] = hqmlib._parse_items(r, ver)
            elif tt == 2:  # LOCATION
                task["locations"] = hqmlib._parse_locations(r)
            elif tt == 5:  # KILL (mob)
                task["mobs"] = hqmlib._parse_mobs(r)
            elif tt == 6:  # DEATH
                task["deaths"] = r.read_data(_Bits.DEATHS)
            elif tt == 7:  # REPUTATION (target)
                task["reputation"] = hqmlib._parse_rep_settings(r)
            elif tt == 8:  # REPUTATION_KILL
                task["reputation_kill"] = hqmlib._parse_rep_kill(r)
            tasks.append(task)
        return tasks

    @staticmethod
    def _parse_items(r, ver):
        items = []
        for _ in range(r.read_data(_Bits.TASK_ITEM_COUNT)):
            if r.read_bool():
                name = r.read_item()
                dmg = r.read_data(_Bits.SHORT)
                nbt = r.read_nbt()
                req = r.read_data(_Bits.TASK_REQUIREMENT)
                prec = hqmlib._read_precision(r, ver)
                items.append({"type": "item", "item": name, "damage": dmg, "nbt": nbt, "required": req, "precision": prec})
            else:
                items.append({"type": "fluid", "nbt": r.read_nbt()})
        return items

    @staticmethod
    def _read_precision(r, ver):
        if ver.lacks(_FileVersion.CUSTOM_PRECISION_TYPES):
            MAP = ["PRECISE", "Fuzzy", "NBT", "PRECISE"]
            i = r.read_data(2)
            return MAP[i] if i < len(MAP) else "PRECISE"
        return r.read_str(_Bits.ITEM_PRECISION)

    @staticmethod
    def _parse_locations(r):
        locs = []
        for _ in range(r.read_data(_Bits.TASK_LOCATION_COUNT)):
            loc = {"icon": r.read_stack(False) if r.read_bool() else None,
                   "name": r.read_str(_Bits.NAME_LENGTH),
                   "x": r.read_data(_Bits.WORLD_COORDINATE), "y": r.read_data(_Bits.WORLD_COORDINATE),
                   "z": r.read_data(_Bits.WORLD_COORDINATE), "dimension": r.read_data(_Bits.BYTE),
                   "radius": r.read_data(_Bits.INT), "visible": r.read_data(_Bits.LOCATION_VISIBILITY)}
            locs.append(loc)
        return locs

    @staticmethod
    def _parse_mobs(r):
        mobs = []
        for _ in range(r.read_data(_Bits.TASK_MOB_COUNT)):
            m = {"icon": r.read_stack(False) if r.read_bool() else None,
                 "name": r.read_str(_Bits.NAME_LENGTH), "mob_id": r.read_str(_Bits.MOB_ID_LENGTH),
                 "count": r.read_data(_Bits.KILL_COUNT), "exact": r.read_bool()}
            mobs.append(m)
        return mobs

    @staticmethod
    def _parse_rep_kill(r):
        settings = []
        for _ in range(r.read_data(_Bits.REPUTATION_SETTING)):
            rep_id = r.read_data(_Bits.REPUTATION)
            lower = r.read_data(_Bits.REPUTATION_MARKER) if r.read_bool() else None
            upper = r.read_data(_Bits.REPUTATION_MARKER) if r.read_bool() else None
            settings.append({"reputation_id": rep_id, "lower_marker_id": lower,
                             "upper_marker_id": upper, "inverted": r.read_bool()})
        return {"settings": settings, "kills": r.read_data(_Bits.DEATHS)}

    @staticmethod
    def _parse_rep_settings(r):
        """REPUTATION (target) task body — same settings format as rep_kill without kills"""
        settings = []
        for _ in range(r.read_data(_Bits.REPUTATION_SETTING)):
            rep_id = r.read_data(_Bits.REPUTATION)
            lower = r.read_data(_Bits.REPUTATION_MARKER) if r.read_bool() else None
            upper = r.read_data(_Bits.REPUTATION_MARKER) if r.read_bool() else None
            settings.append({"reputation_id": rep_id, "lower_marker_id": lower,
                             "upper_marker_id": upper, "inverted": r.read_bool()})
        return {"settings": settings}

    @staticmethod
    def _parse_rewards(r, ver):
        if not r.read_bool():
            return []
        return [r.read_stack(True) for _ in range(r.read_data(_Bits.REWARDS))]

    @staticmethod
    def _parse_cmd_rewards(r, ver):
        if not ver.contains(_FileVersion.COMMAND_REWARDS) or not r.read_bool():
            return []
        return [r.read_str(_Bits.QUEST_DESCRIPTION_LENGTH) for _ in range(r.read_data(_Bits.REWARDS))]

    @staticmethod
    def _parse_tiers(r, ver):
        """GroupTier: name + color + 5 weights (BagTier.values().length=5)"""
        tiers = []
        for _ in range(r.read_data(_Bits.TIER_COUNT)):
            name = r.read_str(_Bits.QUEST_NAME_LENGTH)
            color = r.read_data(_Bits.COLOR)
            weights = [r.read_data(_Bits.WEIGHT) for _ in range(5)]
            tiers.append({"name": name, "color": color, "weights": weights})
        return tiers

    @staticmethod
    def _parse_groups(r, ver):
        """Group: id + name + tier(TIER_COUNT) + items + limit"""
        groups = []
        for _ in range(r.read_data(_Bits.GROUP_COUNT)):
            gid = r.read_data(_Bits.GROUP_COUNT) if ver.contains(_FileVersion.BAG_LIMITS) else -1
            name = r.read_str(_Bits.QUEST_NAME_LENGTH)
            tier = r.read_data(_Bits.TIER_COUNT)
            items = []
            for _ in range(r.read_data(_Bits.GROUP_ITEMS)):
                s = r.read_stack(True)
                if s:
                    items.append(s)
            limit = 0
            if ver.contains(_FileVersion.BAG_LIMITS) and r.read_bool():
                limit = r.read_data(_Bits.LIMIT)
            groups.append({"id": gid, "name": name, "tier": tier, "items": items, "limit": limit})
        return groups

    # ================================================================
    # 写入内部方法
    # ================================================================

    @staticmethod
    def _write_sets(w, sets, ver):
        if ver.contains(_FileVersion.SETS):
            w.write_data(len(sets), _Bits.QUEST_SETS)
            for i, s in enumerate(sets):
                w.write_str(s.get("name", f"Set {i}"), _Bits.QUEST_NAME_LENGTH)
                w.write_str(s.get("description", ""), _Bits.QUEST_DESCRIPTION_LENGTH)
                if ver.contains(_FileVersion.REPUTATION_BARS):
                    bars = s.get("reputation_bars", [])
                    w.write_data(len(bars), _Bits.BYTE)
                    for bar in bars:
                        w.write_data(bar, _Bits.INT)

    @staticmethod
    def _write_reputation(w, reps, ver):
        if ver.contains(_FileVersion.REPUTATION):
            w.write_data(len(reps), _Bits.REPUTATION)
            for rep in reps:
                w.write_data(rep.get("id", 0), _Bits.REPUTATION)
                w.write_str(rep.get("name"), _Bits.QUEST_NAME_LENGTH)
                w.write_str(rep.get("neutral_name"), _Bits.QUEST_NAME_LENGTH)
                markers = rep.get("markers", [])
                w.write_data(len(markers), _Bits.REPUTATION_MARKER)
                for m in markers:
                    w.write_str(m.get("name"), _Bits.QUEST_NAME_LENGTH)
                    w.write_data(m.get("value", 0), _Bits.REPUTATION_VALUE)

    @staticmethod
    def _write_quests(w, quests, ver):
        max_id = max((q.get("id", 0) for q in quests), default=0)
        slot_count = max_id + 1  # Java size() = max ID + 1
        w.write_data(slot_count, _Bits.quests(ver))
        qmap = {q.get("id"): q for q in quests}
        for qid in range(slot_count):
            q = qmap.get(qid)
            if q is None:
                w.write_bool(False)
                continue
            w.write_bool(True)
            hqmlib._write_single_quest(w, q, ver)

    @staticmethod
    def _write_single_quest(w, q, ver):
        w.write_str(q.get("name", ""), _Bits.QUEST_NAME_LENGTH)
        w.write_str(q.get("description", ""), _Bits.QUEST_DESCRIPTION_LENGTH)
        w.write_data(q.get("x", 0), _Bits.QUEST_POS_X)
        w.write_data(q.get("y", 0), _Bits.QUEST_POS_Y)
        w.write_bool(q.get("big_icon", False))
        if ver.contains(_FileVersion.SETS):
            w.write_data(q.get("quest_set_id", 0), _Bits.QUEST_SETS)
        if ver.contains(_FileVersion.SETS):
            icon = q.get("icon")
            if icon and icon.get("item"):
                w.write_bool(True)
                w.write_stack(icon, False, ver)
            else:
                w.write_bool(False)
        reqs = q.get("requirements", [])
        if reqs:
            w.write_bool(True)
            w.write_data(len(reqs), _Bits.QUESTS)
            for r in reqs:
                w.write_data(r, _Bits.QUESTS)
        else:
            w.write_bool(False)
        if ver.contains(_FileVersion.OPTION_LINKS):
            links = q.get("option_links", [])
            if links:
                w.write_bool(True)
                w.write_data(len(links), _Bits.QUESTS)
                for link in links:
                    w.write_data(link, _Bits.QUESTS)
            else:
                w.write_bool(False)
        hqmlib._write_repeat(w, q.get("repeat_info", {"type": 0}), ver)
        w.write_data(q.get("trigger_type", 0), _Bits.TRIGGER_TYPE)
        if q.get("trigger_type", 0) == 2:  # TASK_TRIGGER -> isUseTaskCount() == true
            w.write_data(q.get("trigger_tasks", 0), _Bits.TASKS)
        if ver.contains(_FileVersion.PARENT_COUNT):
            parent = q.get("parent_requirement_count")
            if parent is not None:
                w.write_bool(True)
                w.write_data(parent, _Bits.QUESTS)
            else:
                w.write_bool(False)
        hqmlib._write_tasks(w, q.get("tasks", []), ver)
        hqmlib._write_rewards(w, q.get("rewards", []), ver)
        hqmlib._write_rewards(w, q.get("reward_choices", []), ver)
        hqmlib._write_cmd_rewards(w, q.get("command_rewards", []), ver)
        rep_rewards = q.get("reputation_rewards", [])
        w.write_data(len(rep_rewards), _Bits.REPUTATION_REWARD)
        for rr in rep_rewards:
            w.write_data(rr.get("reputation_id", 0), _Bits.REPUTATION)
            w.write_data(rr.get("value", 0), _Bits.REPUTATION_VALUE)

    @staticmethod
    def _write_repeat(w, info, ver):
        if ver.lacks(_FileVersion.REPEATABLE_QUESTS):
            return
        w.write_data(info.get("type", 0), _Bits.REPEAT_TYPE)
        if info.get("type") == 1:
            w.write_data(info.get("hours", 0), _Bits.HOURS)
        elif info.get("type") == 2:
            w.write_data(info.get("days", 0), _Bits.HOURS)

    @staticmethod
    def _write_tasks(w, tasks, ver):
        w.write_data(len(tasks), _Bits.TASKS)
        for task in tasks:
            w.write_data(task.get("type", 0), _Bits.task_type(ver))
            w.write_str(task.get("description", ""), _Bits.QUEST_NAME_LENGTH)
            w.write_str(task.get("long_description", ""), _Bits.QUEST_DESCRIPTION_LENGTH)
            tt = task.get("type", 0)
            if tt in (0, 1, 3, 4):  # CONSUME/CRAFT/CONSUME_QDS/DETECT -> items
                hqmlib._write_items(w, task.get("items", []), ver)
            elif tt == 2:  # LOCATION
                hqmlib._write_locations(w, task.get("locations", []))
            elif tt == 5:  # KILL
                hqmlib._write_mobs(w, task.get("mobs", []))
            elif tt == 6:  # DEATH
                w.write_data(task.get("deaths", 0), _Bits.DEATHS)
            elif tt == 7:  # REPUTATION
                hqmlib._write_rep_settings(w, task.get("reputation", {}))
            elif tt == 8:  # REPUTATION_KILL
                hqmlib._write_rep_kill(w, task.get("reputation_kill", {}))

    @staticmethod
    def _write_items(w, items, ver):
        w.write_data(len(items), _Bits.TASK_ITEM_COUNT)
        for item in items:
            if item.get("type") == "item":
                w.write_bool(True)
                w.write_item(item.get("item"), ver)
                w.write_data(item.get("damage", 0), _Bits.SHORT)
                w.write_nbt(item.get("nbt"))
                w.write_data(item.get("required", 1), _Bits.TASK_REQUIREMENT)
                hqmlib._write_precision(w, item.get("precision", "PRECISE"), ver)
            else:
                w.write_bool(False)
                w.write_nbt(item.get("nbt"))

    @staticmethod
    def _write_precision(w, precision, ver):
        if ver.lacks(_FileVersion.CUSTOM_PRECISION_TYPES):
            MAP = {"PRECISE": 0, "Fuzzy": 1, "NBT": 2}
            w.write_data(MAP.get(precision, 0), 2)
        else:
            w.write_str(precision, _Bits.ITEM_PRECISION)

    @staticmethod
    def _write_locations(w, locs):
        w.write_data(len(locs), _Bits.TASK_LOCATION_COUNT)
        for loc in locs:
            icon = loc.get("icon")
            w.write_bool(icon is not None and icon.get("item") is not None)
            if icon and icon.get("item"):
                w.write_stack(icon, False, _FileVersion.LATEST)
            w.write_str(loc.get("name"), _Bits.NAME_LENGTH)
            w.write_data(loc.get("x", 0), _Bits.WORLD_COORDINATE)
            w.write_data(loc.get("y", 0), _Bits.WORLD_COORDINATE)
            w.write_data(loc.get("z", 0), _Bits.WORLD_COORDINATE)
            w.write_data(loc.get("dimension", 0), _Bits.BYTE)
            w.write_data(loc.get("radius", 0), _Bits.INT)
            w.write_data(loc.get("visible", 0), _Bits.LOCATION_VISIBILITY)

    @staticmethod
    def _write_mobs(w, mobs):
        w.write_data(len(mobs), _Bits.TASK_MOB_COUNT)
        for m in mobs:
            icon = m.get("icon")
            w.write_bool(icon is not None and icon.get("item") is not None)
            if icon and icon.get("item"):
                w.write_stack(icon, False, _FileVersion.LATEST)
            w.write_str(m.get("name"), _Bits.NAME_LENGTH)
            w.write_str(m.get("mob_id"), _Bits.MOB_ID_LENGTH)
            w.write_data(m.get("count", 1), _Bits.KILL_COUNT)
            w.write_bool(m.get("exact", False))

    @staticmethod
    def _write_rep_settings(w, rs):
        """Write REPUTATION task body — settings only, no kills"""
        settings = rs.get("settings", [])
        w.write_data(len(settings), _Bits.REPUTATION_SETTING)
        for s in settings:
            w.write_data(s.get("reputation_id", 0), _Bits.REPUTATION)
            lower = s.get("lower_marker_id")
            w.write_bool(lower is not None)
            if lower is not None:
                w.write_data(lower, _Bits.REPUTATION_MARKER)
            upper = s.get("upper_marker_id")
            w.write_bool(upper is not None)
            if upper is not None:
                w.write_data(upper, _Bits.REPUTATION_MARKER)
            w.write_bool(s.get("inverted", False))

    @staticmethod
    def _write_rep_kill(w, rk):
        settings = rk.get("settings", [])
        w.write_data(len(settings), _Bits.REPUTATION_SETTING)
        for s in settings:
            w.write_data(s.get("reputation_id", 0), _Bits.REPUTATION)
            lower = s.get("lower_marker_id")
            w.write_bool(lower is not None)
            if lower is not None:
                w.write_data(lower, _Bits.REPUTATION_MARKER)
            upper = s.get("upper_marker_id")
            w.write_bool(upper is not None)
            if upper is not None:
                w.write_data(upper, _Bits.REPUTATION_MARKER)
            w.write_bool(s.get("inverted", False))
        w.write_data(rk.get("kills", 0), _Bits.DEATHS)

    @staticmethod
    def _write_rewards(w, rewards, ver):
        if rewards:
            w.write_bool(True)
            w.write_data(len(rewards), _Bits.REWARDS)
            for stack in rewards:
                w.write_stack(stack, True, ver)
        else:
            w.write_bool(False)

    @staticmethod
    def _write_cmd_rewards(w, cmds, ver):
        if ver.contains(_FileVersion.COMMAND_REWARDS):
            if cmds:
                w.write_bool(True)
                w.write_data(len(cmds), _Bits.REWARDS)
                for cmd in cmds:
                    w.write_str(cmd, _Bits.QUEST_DESCRIPTION_LENGTH)
            else:
                w.write_bool(False)

    @staticmethod
    def _write_tiers(w, tiers, ver):
        w.write_data(len(tiers), _Bits.TIER_COUNT)
        for t in tiers:
            w.write_str(t.get("name"), _Bits.QUEST_NAME_LENGTH)
            w.write_data(t.get("color", 0), _Bits.COLOR)
            weights = t.get("weights", [])
            if len(weights) != 5:
                weights = [t.get("weight", 1)] * 5  # backward compat
            for wt in weights:
                w.write_data(wt, _Bits.WEIGHT)

    @staticmethod
    def _write_groups(w, groups, ver):
        w.write_data(len(groups), _Bits.GROUP_COUNT)
        for i, g in enumerate(groups):
            if ver.contains(_FileVersion.BAG_LIMITS):
                w.write_data(g.get("id", i), _Bits.GROUP_COUNT)
            w.write_str(g.get("name"), _Bits.QUEST_NAME_LENGTH)
            w.write_data(g.get("tier", 0), _Bits.TIER_COUNT)
            items = g.get("items", [])
            w.write_data(len(items), _Bits.GROUP_ITEMS)
            for stack in items:
                w.write_stack(stack, True, ver)
            if ver.contains(_FileVersion.BAG_LIMITS):
                limit = g.get("limit", 0)
                if limit > 0:
                    w.write_bool(True)
                    w.write_data(limit, _Bits.LIMIT)
                else:
                    w.write_bool(False)

hqmlib = hqmlib()
snbtlib = snbtlib()
fancymenulib = fancymenulib()

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
[bright_green]Version:[/] Release 1.6 Bata 1
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
    "zipfile", "pickle", "json", "ast", "fancymenulib",
    "re", "partial", "defaultdict", "Path", 'HARDWARE_INFO',
    "ThreadPoolExecutor", "as_completed", "Callable", "Dict", "Any",
    "requests", "math", "tqdm", "dataclass", "faiss",
    "replace", "time", "shutil", "atexit",
    "GPU_ACC", "numpy", "PurePosixPath", "random", "deque",
    "APIConfig", "Union", "asynccontextmanager",
    "Optional", "List", "io", "asyncio", "uuid",
    "sqlite3", "HTTPAdapter", "SimpleNamespace", "queue", "quote",
    "logging", "QueueHandler", "QueueListener", "RotatingFileHandler", 
    "shlex", "FileHandler", "Retry", "locale", "bisect",
    "System", "dnfile", #Unity
    "tomllib", "snbtlib", "hqmlib" #Minecraft
] 
