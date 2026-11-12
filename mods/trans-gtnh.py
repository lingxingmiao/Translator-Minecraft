# 可用配置(覆盖键 MOD_TRANS_GTNH_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   ["*.cfg", "*.lang"]
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import Path, Parser, Mods, eb, re, _re_compile, Any, MULTILINE, Dict, Optional, TextIO, List

模组参数 = Mods().模组配置("trans-gtnh", {
    "FILE_MATCH": ["*.cfg", "*.lang"],
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)

_块开始 = re.compile(r"^(\w[\w.\-]*)\s*\{")
# S: 键两种写法: S:"带引号 键"=值   /   S:裸键=值(GregTech.lang 里绝大多数是裸键)
_S条目 = re.compile(r'^(\s*)S:("(?:[^"\\]|\\.)*"|[^\s=]+)=(.*)$')
_语言块提示 = re.compile(r"^\s*[\w.\-]+\s*\{", re.M)
_S条目计数 = re.compile(r'^\s*S:(?:"(?:[^"\\]|\\.)*"|[^\s=]+)=', re.M)
_值不像文本 = re.compile(r"^(?:[-+]?\d+(?:\.\d+)?|\[[\d,.\s]*\]|\{.*\})$")
_资源ID = re.compile(r"^[a-z0-9_.\-]+:[a-z0-9_/.\-]+$", re.I)
_字面量 = frozenset({"true", "false", "null", "none"})
_有文字 = re.compile(r"[A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def 可取文本(值: str) -> bool:
    文本 = 值.strip()
    if not 文本 or 文本.lower() in _字面量:
        return False
    if "://" in 文本 or _资源ID.match(文本):
        return False
    return not (_值不像文本.match(文本) and not _有文字.search(文本))


class gtnhlib:
    """
    GTNH .lang 文件格式解析与生成器。
    数据格式：
        {
            'header': ['# Configuration file', ...],
            'block_name': 'languagefile',
            'entries': [
                {'type': 'S', 'key': 'Lang.Key', 'value': '翻译文本'},
                {'type': 'B', 'key': 'someFlag', 'value': True},
            ]
        }
    
    用法：
        data = gtnhlib.loads(text)
        text = gtnhlib.dumps(data)
        data = gtnhlib.load(open('GregTech.lang', encoding='utf-8'))
        gtnhlib.dump(data, open('output.lang', 'w', encoding='utf-8'))
    """
    
    _BLOCK_START = _re_compile(r'^(\w[\w.\-]*)\s*\{')
    _BLOCK_END = _re_compile(r'^\s*\}')
    _S_ENTRY = _re_compile(r'^\s+S:(?:"((?:[^"\\]|\\.)*)"|([^\s=]+))=(.*)$', MULTILINE)
    _B_ENTRY = _re_compile(r'^\s+B:(\w+)=(true|false)$')
    _COMMENT_OR_BLANK = _re_compile(r'^\s*(#.*)?$')

    @staticmethod
    def loads(text: str) -> Dict[str, Any]:
        """将 .lang 文本解析为结构化字典。"""
        lines = text.splitlines()
        header: List[str] = []
        block_name: Optional[str] = None
        entries: List[Dict[str, Any]] = []
        in_block = False
        
        for line in lines:
            m_start = gtnhlib._BLOCK_START.match(line)
            if m_start and not in_block:
                block_name = m_start.group(1)
                in_block = True
                continue
            
            if in_block and gtnhlib._BLOCK_END.match(line):
                in_block = False
                continue
            
            if in_block:
                m_s = gtnhlib._S_ENTRY.match(line)
                if m_s:
                    entries.append({'type': 'S', 'key': m_s.group(1), 'value': m_s.group(2)})
                    continue
                
                m_b = gtnhlib._B_ENTRY.match(line)
                if m_b:
                    entries.append({'type': 'B', 'key': m_b.group(1), 'value': m_b.group(2) == 'true'})
                    continue
                
                if gtnhlib._COMMENT_OR_BLANK.match(line):
                    continue
                
                entries.append({'type': '_RAW', '_line': line})
            else:
                header.append(line)
        
        return {'header': header, 'block_name': block_name or 'languagefile', 'entries': entries}

    @staticmethod
    def dumps(data: Dict[str, Any], block_name: Optional[str] = None) -> str:
        """将结构化字典序列化为 .lang 文本。"""
        bn = block_name or data.get('block_name', 'languagefile')
        header = data.get('header', ['# Configuration file'])
        entries = data.get('entries', [])
        
        lines: List[str] = []
        
        if header:
            lines.extend(header)
        else:
            lines.extend(['# Configuration file', ''])
        
        lines.append(f'{bn} {{')
        
        for entry in entries:
            etype = entry.get('type')
            if etype == 'S':
                lines.append(f'\tS:"{entry["key"]}"={entry["value"]}')
            elif etype == 'B':
                val_str = 'true' if entry['value'] else 'false'
                lines.append(f'\tB:{entry["key"]}={val_str}')
            elif etype == '_RAW':
                lines.append(entry['_line'])
        
        lines.append('}')
        lines.append('')
        
        return '\n'.join(lines)

    @staticmethod
    def load(fp: TextIO, **kwargs) -> Dict[str, Any]:
        """从文件对象读取并解析。"""
        return gtnhlib.loads(fp.read(), **kwargs)

    @staticmethod
    def dump(data: Dict[str, Any], fp: TextIO, **kwargs):
        """将数据结构写入文件。"""
        fp.write(gtnhlib.dumps(data, **kwargs))

    # ---- 便捷操作 ----

    @staticmethod
    def get_entry(data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """按 key 查找第一条匹配的条目，返回 value 或 default。"""
        for entry in data.get('entries', []):
            if entry.get('key') == key:
                return entry.get('value', default)
        return default

    @staticmethod
    def set_entry(data: Dict[str, Any], key: str, value: Any, etype: str = 'S') -> bool:
        """按 key 查找并修改第一条匹配条目，不存在则追加。返回是否修改了已有条目。"""
        for entry in data.get('entries', []):
            if entry.get('key') == key and entry.get('type') == etype:
                entry['value'] = value
                return True
        data.setdefault('entries', []).append({'type': etype, 'key': key, 'value': value})
        return False

    @staticmethod
    def remove_entry(data: Dict[str, Any], key: str, etype: Optional[str] = None) -> bool:
        """按 key 删除第一条匹配条目。etype 为 None 表示任意类型。返回是否成功删除。"""
        entries = data.get('entries', [])
        for i, entry in enumerate(entries):
            if entry.get('key') == key:
                if etype is None or entry.get('type') == etype:
                    entries.pop(i)
                    return True
        return False


# ============================================================
# BIT MASK 表
# ============================================================


class GTNHParser(Parser):

    def 读取(Self, 文件路径):
        结果 = []
        路径 = Path(文件路径)
        try:
            行列表 = 路径.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except Exception:
            Self.日志("log.module.lang.load.error", mod="GTNH", file=str(路径), e=eb.format_exc(), info_level=2)
            return 结果
        结构 = gtnhlib.loads("\n".join(行列表))
        类别 = 结构.get("block_name") or ""
        for 行 in 行列表:
            块 = _块开始.match(行)
            if 块:
                类别 = 块.group(1)
                continue
            条目 = _S条目.match(行)
            if not 条目:
                continue
            键 = 条目.group(2)
            值 = 条目.group(3)
            # 引号只是为了包住特殊字符, 不该跟着进翻译器 —— 剥掉, 写回时按原样补回
            if len(值) >= 2 and 值.startswith('"') and 值.endswith('"'):
                值 = 值[1:-1].replace('\\"', '"')
            if not 可取文本(值):
                continue
            结果.append([[str(路径), 键], 值, 类别])
        return 结果

    def 写回(Self, 翻译列表):
        文件映射 = {}
        for 条目 in 翻译列表:
            文件路径 = 条目[0][0]
            键 = 条目[0][1]
            译文 = 条目[1]
            if 译文 is None or 译文 == "":
                continue
            文件映射.setdefault(文件路径, {})[键] = str(译文).replace("\r\n", "\n").replace("\n", "")
        for 文件路径, 替换字典 in 文件映射.items():
            路径 = Path(文件路径)
            try:
                原始 = 路径.read_bytes()
                行列表 = 原始.decode("utf-8-sig").splitlines()
            except Exception:
                Self.日志("log.module.lang.load.error", mod="GTNH", file=str(路径), e=eb.format_exc(), info_level=2)
                continue
            输出 = []
            for 行 in 行列表:
                条目 = _S条目.match(行)
                if 条目 and 条目.group(2) in 替换字典:
                    译文 = str(替换字典[条目.group(2)])
                    原值 = 条目.group(3)
                    if len(原值) >= 2 and 原值.startswith('"') and 原值.endswith('"'):
                        译文 = '"' + 译文.replace('"', '\\"') + '"'
                    输出.append(f"{条目.group(1)}S:{条目.group(2)}={译文}")
                    continue
                输出.append(行)
            try:
                # 写字节: 保住原来的 BOM, 也不让 Windows 把 \n 翻成 \r\n
                数据 = "\n".join(输出).encode("utf-8") + b"\n"
                路径.write_bytes((b"\xef\xbb\xbf" if 原始.startswith(b"\xef\xbb\xbf") else b"") + 数据)
            except Exception:
                Self.日志("log.module.lang.write.error", mod="GTNH", file=str(路径), e=eb.format_exc(), info_level=2)


def 执行(Self, 根, path2=None, **参数):
    解析 = GTNHParser(Self)
    Self.翻译流程(
        根, Mods().取值(Self, ("trans-gtnh", "FILE_MATCH")), 解析.读取, None,
        lambda x: x[0][0], 解析.写回,
        Mods().取值(Self, ("trans-gtnh", "READ_CONCURRENT")),
        Mods().取值(Self, ("trans-gtnh", "WRITE_CONCURRENT")),
        "lang", path2=path2, **参数)


def 探测配置(Self, 路径):
    路径 = Path(路径)
    # GTNH 的语言条目有时在 .cfg, 有时就叫 GregTech.lang —— 只看内容指纹
    if 路径.suffix.lower() not in {".cfg", ".lang"}:
        return False
    try:
        文本 = 路径.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return False
    if "{" not in 文本 or not _语言块提示.search(文本):
        return False
    return len(_S条目计数.findall(文本)) >= 4


Mods().注册翻译类型("GTNH Language Files", 执行, 解析器类=GTNHParser, 显示名="GTNH", 文件探测=探测配置, 探测分数=75, 产物="源文件")

Mods().注册翻译类型到归档("GTNH Language Files", "config", 注册路径="/config", 打包扩展名={".lang"}, 最大深度=2)
