# 可用配置(覆盖键 MOD_TRANS_FTBQUESTS_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   "*.snbt"
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (Path, json, re, Any, List, _json_dumps, _json_loads, _re_compile, _re_sub, MULTILINE, Parser, Mods, eb, Dict, TextIO)


# ============================================================
# snbtlib — SNBT (Stringified NBT) 解析与序列化器
# MIT License — Copyright (c) 2021 Tryanks
#
# 纯 Python 实现的 Minecraft SNBT 格式解析器, 支持 FTB Quests (.snbt) 及标准 SNBT 数据类型。
# 用法:
#     data = snbtlib.loads('{key: "value", num: 42b}')
#     text = snbtlib.dumps(data)
#     data = snbtlib.load(open('quest.snbt', encoding='utf-8'))
#     snbtlib.dump(data, open('quest.snbt', 'w', encoding='utf-8'))


# ============================================================


class snbtlib:
    """
    SNBT (Stringified Named Binary Tag) 解析器与序列化器。
    支持的数据类型:
      - Byte:          0b, 1b  → 内部存储为 "$number$0b"
      - Short:         0s, 1s  → 内部存储为 "$number$0s"
      - Int:           0, -1   → 内部存储为 "$number$0"
      - Long:          0L, 1L  → 内部存储为 "$number$0L"
      - Float:         0.0f    → 内部存储为 "$number$0.0f"
      - Double:        0.0d    → 内部存储为 "$number$0.0d"
      - String:        "value", unquoted_word
      - Boolean:       true, false
      - Compound:      {key: value, ...}  → dict
      - List:          [a, b, c]          → list
      - Byte Array:    [B; 0b, 1b, ...]   → bytes
      - Int Array:     [I; 0, 1, ...]     → list (首元素 "I;" 标记)
      - Long Array:    [L; 0L, 1L, ...]   → list (首元素 "L;" 标记)
    数值采用 "$number$" 前缀保留原始字符串形式，
    以保证 round-trip 忠实还原（含 NBT 类型后缀）。
    """
    _NUMBER_PREFIX = '$number$'
    _NUMBER_SUFFIXES = frozenset('bslfdBSLFD')
    # ---- 公开 API -------------------------------------------------
    @staticmethod

    def loads(s: str, format: bool = False) -> Any:
        """解析 SNBT 字符串，返回 dict/list/bytes 等 Python 对象。
        Args:
            s: SNBT 格式文本
            format: 若为 True，返回格式化 JSON 字符串
        Returns:
            解析后的 Python 对象，或 JSON 字符串（format=True 时）
        """
        tokens = snbtlib._tokenize(s)
        idx = [0]
        result = snbtlib._parse_value(tokens, idx)
        return _json_dumps(result, ensure_ascii=False, indent=4) if format else result
    @staticmethod

    def dumps(obj: Any, indent: int = 0, compact: bool = False) -> str:
        """将 Python 对象序列化为 SNBT 字符串。
        Args:
            obj: 待序列化的 dict/list/bytes 等对象。
                 若传入 JSON 字符串，会自动解析为 Python 对象。
            indent: 起始缩进层级（内部递归使用）
            compact: 若为 True，在各项之间插入逗号（兼容 BetterQuesting 格式）
        Returns:
            SNBT 格式字符串
        """
        if isinstance(obj, str):
            obj = _json_loads(obj)
        lines = []
        snbtlib._serialize(obj, lines, indent)
        text = ''.join(lines)
        return text if not compact else snbtlib._compatible(text)
    @staticmethod

    def load(fp: TextIO, **kwargs) -> Any:
        """从文件对象读取并解析 SNBT。"""
        return snbtlib.loads(fp.read(), **kwargs)
    @staticmethod

    def dump(obj: Any, fp: TextIO, **kwargs):
        """将对象序列化为 SNBT 并写入文件对象。"""
        fp.write(snbtlib.dumps(obj, **kwargs))
    # ---- 词法分析 (Tokenizer) -------------------------------------
    @staticmethod

    def _tokenize(text: str) -> List[Dict[str, Any]]:
        """将 SNBT 文本切分为 token 列表。
        处理流程:
        1. 移除 \\r
        2. 移除行注释（// 和 #）
        3. 逐字符扫描生成 token
        """
        text = text.replace('\r', '')
        # 移除单行注释: // 和 # 开头的行
        text = _re_sub(_re_compile(r'^\s*//.*$', MULTILINE), '', text)
        text = _re_sub(_re_compile(r'^\s*#.*$', MULTILINE), '', text)
        tokens, i, n = [], 0, len(text)
        while i < n:
            ch = text[i]
            # 跳过空白字符（保留换行用于分隔）
            if ch.isspace() and ch != '\n':
                i += 1
                continue
            # 结构符号
            if ch in '{[}],:;':
                tokens.append(snbtlib._mk_token(ch))
                i += 1
                continue
            # 数值（含负号、小数、科学计数法、NBT 类型后缀）
            if ch in '-0123456789':
                j = i
                # 读取数值主体
                while j < n and (text[j] in '-0123456789.eE'
                                 or (text[j] in '+-' and j > i and text[j-1] in 'eE')):
                    j += 1
                # 读取 NBT 类型后缀 (b/B, s/S, l/L, f/F, d/D)
                if j < n and text[j] in snbtlib._NUMBER_SUFFIXES:
                    j += 1
                tokens.append({'type': 'NUMBER',
                               'value': snbtlib._NUMBER_PREFIX + text[i:j]})
                i = j
                continue
            # 引号字符串
            if ch == '"':
                j, buf = i + 1, []
                while j < n:
                    if text[j] == '\\' and j + 1 < n:
                        # 处理转义序列：解析为实际字符
                        esc = text[j + 1]
                        if esc == 'n':
                            buf.append('\n')
                        elif esc == 't':
                            buf.append('\t')
                        elif esc == 'r':
                            buf.append('\r')
                        elif esc == '\\':
                            buf.append('\\')
                        elif esc == '"':
                            buf.append('"')
                        elif esc == 'u' and j + 5 < n:
                            # Unicode escape \uXXXX
                            try:
                                buf.append(chr(int(text[j+2:j+6], 16)))
                                j += 4  # extra skip handled below
                            except (ValueError, IndexError):
                                buf.append(text[j+1:j+6])  # 解析失败保留原始\uXXXX
                                j += 4  # 跳过全部\uXXXX防止后续解析混乱
                        else:
                            # 未知转义，保留原样
                            buf.append(esc)
                        j += 2
                    elif text[j] == '"':
                        break
                    else:
                        buf.append(text[j])
                        j += 1
                tokens.append({'type': 'STRING_QUOTED', 'value': ''.join(buf)})
                i = j + 1
                continue
            # 标识符 / 无引号字符串 / 布尔值
            if ch.isalpha() or ch == '_':
                j = i
                while j < n and (text[j].isalnum() or text[j] in '_./-'):
                    j += 1
                word = text[i:j]
                if word in ('true', 'false'):
                    tokens.append({'type': 'BOOL', 'value': word == 'true'})
                else:
                    tokens.append({'type': 'STRING', 'value': word})
                i = j
                continue
            # 换行符
            if ch == '\n':
                tokens.append({'type': 'ENTER', 'value': '\n'})
                i += 1
                continue
            # 未知字符，跳过
            i += 1
        return tokens
    @staticmethod

    def _mk_token(ch: str) -> Dict[str, Any]:
        """根据字符创建对应类型的 token。"""
        _MAP = {
            '{': 'BEGIN_DICT', '}': 'END_DICT',
            '[': 'BEGIN_LIST', ']': 'END_LIST',
            ':': 'COLON', ';': 'SEMICOLON',
            ',': 'ENTER', '\n': 'ENTER',
        }
        return {'type': _MAP.get(ch, 'UNKNOWN'), 'value': ch}
    # ---- 语法分析 (Parser) ----------------------------------------
    @staticmethod

    def _parse_value(tokens: List[Dict[str, Any]], idx: List[int]) -> Any:
        """解析单个顶层值。"""
        if idx[0] >= len(tokens):
            return None
        t = tokens[idx[0]]
        if t['type'] == 'BEGIN_DICT':
            idx[0] += 1
            return snbtlib._parse_dict(tokens, idx)
        elif t['type'] == 'BEGIN_LIST':
            idx[0] += 1
            return snbtlib._parse_list(tokens, idx)
        idx[0] += 1
        return t['value']
    @staticmethod

    def _parse_dict(tokens: List[Dict[str, Any]], idx: List[int]) -> Dict[str, Any]:
        """解析 Compound: {key: value, ...}"""
        result = {}
        while idx[0] < len(tokens):
            t = tokens[idx[0]]
            if t['type'] == 'END_DICT':
                idx[0] += 1
                break
            if t['type'] in ('STRING', 'STRING_QUOTED', 'NUMBER'):
                key = t['value']
                # 数值键去掉 $number$ 前缀
                if key.startswith(snbtlib._NUMBER_PREFIX):
                    key = key[len(snbtlib._NUMBER_PREFIX):]
                # 引号字符串键保留外围引号标识
                if t['type'] == 'STRING_QUOTED':
                    key = f'"{key}"'
                idx[0] += 1
                # 跳过冒号
                if idx[0] < len(tokens) and tokens[idx[0]]['type'] == 'COLON':
                    idx[0] += 1
                # 解析值
                if idx[0] < len(tokens):
                    vt = tokens[idx[0]]
                    if vt['type'] == 'BEGIN_DICT':
                        idx[0] += 1
                        result[key] = snbtlib._parse_dict(tokens, idx)
                    elif vt['type'] == 'BEGIN_LIST':
                        idx[0] += 1
                        result[key] = snbtlib._parse_list(tokens, idx)
                    else:
                        idx[0] += 1
                        result[key] = vt['value']
            else:
                idx[0] += 1
        return result
    @staticmethod

    def _parse_list(tokens: List[Dict[str, Any]], idx: List[int]) -> Any:
        """解析 List / ByteArray / IntArray / LongArray。
        返回:
            - 普通列表 → list
            - ByteArray [B; ...] → bytes
            - IntArray  [I; ...] → list (首元素 "I;")
            - LongArray [L; ...] → list (首元素 "L;")
        """
        # 检测是否为特殊数组类型 [B; / [I; / [L;
        if idx[0] < len(tokens):
            first_val = tokens[idx[0]]['value']
            if first_val in ('B', 'I', 'L'):
                # 确认后面紧跟 ';'（以 STRING 或 COLON 形式）
                next_idx = idx[0] + 1
                if next_idx < len(tokens):
                    nt = tokens[next_idx]
                    nt_val = nt['value']
                    # 'B' ';' 或 'I' ';' 或 'L' ';'
                    if nt_val == ';' or nt_val == ':':
                        arr_type = first_val
                        idx[0] += 1  # 跳过类型标识 B/I/L
                        if nt['type'] == 'COLON' or nt_val in (';', ':'):
                            idx[0] += 1  # 跳过分隔符
                        return snbtlib._parse_typed_array(tokens, idx, arr_type)
        # 普通列表
        result = []
        while idx[0] < len(tokens):
            t = tokens[idx[0]]
            if t['type'] == 'END_LIST':
                idx[0] += 1
                break
            if t['type'] == 'BEGIN_DICT':
                idx[0] += 1
                result.append(snbtlib._parse_dict(tokens, idx))
            elif t['type'] == 'BEGIN_LIST':
                idx[0] += 1
                result.append(snbtlib._parse_list(tokens, idx))
            elif t['type'] in ('STRING', 'STRING_QUOTED', 'NUMBER', 'BOOL'):
                result.append(t['value'])
                idx[0] += 1
            else:
                idx[0] += 1
        return result
    @staticmethod

    def _parse_typed_array(tokens: List[Dict[str, Any]],
                           idx: List[int], arr_type: str) -> Any:
        """解析 [B; ...] / [I; ...] / [L; ...] 类型数组。"""
        if arr_type == 'B':
            # ByteArray → bytes
            result = b''
            while idx[0] < len(tokens):
                t = tokens[idx[0]]
                if t['type'] == 'END_LIST':
                    idx[0] += 1
                    return result
                if t['type'] == 'NUMBER':
                    val = t['value']
                    if val.startswith(snbtlib._NUMBER_PREFIX):
                        val = val[len(snbtlib._NUMBER_PREFIX):]
                    # 去掉 'b'/'B' 后缀
                    if val.lower().endswith('b'):
                        val = val[:-1]
                    try:
                        result += int(val).to_bytes(1, 'big', signed=True)
                    except (ValueError, OverflowError):
                        pass
                idx[0] += 1
            return result
        else:
            # IntArray / LongArray → list with prefix marker
            prefix = arr_type + ';'
            result = [prefix]
            while idx[0] < len(tokens):
                t = tokens[idx[0]]
                if t['type'] == 'END_LIST':
                    idx[0] += 1
                    break
                if t['type'] in ('STRING', 'STRING_QUOTED', 'NUMBER', 'BOOL'):
                    result.append(t['value'])
                    idx[0] += 1
                elif t['type'] == 'BEGIN_DICT':
                    idx[0] += 1
                    result.append(snbtlib._parse_dict(tokens, idx))
                elif t['type'] == 'BEGIN_LIST':
                    idx[0] += 1
                    result.append(snbtlib._parse_list(tokens, idx))
                else:
                    idx[0] += 1
            return result
    # ---- 序列化 (Serializer) --------------------------------------
    @staticmethod

    def _serialize(obj: Any, out: List[str], indent: int):
        """递归序列化对象到输出行列表。"""
        tab = '\t' * indent
        if isinstance(obj, dict):
            if not obj:
                out.append('{ }\n')
            else:
                out.append('{\n')
                for k, v in obj.items():
                    out.append(tab + '\t' + str(k) + ': ')
                    snbtlib._serialize_value(v, out, indent + 1)
                out.append(tab + '}\n')
        elif isinstance(obj, list):
            if not obj:
                out.append('[ ]\n')
            elif len(obj) == 1 and not isinstance(obj[0], (dict, list, bytes)):
                # 单元素简单值 → 单行
                inner = []
                snbtlib._serialize_value(obj[0], inner, indent)
                out.append('[' + ''.join(inner).strip() + ']\n')
            else:
                # 检测是否为特殊数组类型 [I; ...] 或 [L; ...]
                if (len(obj) > 0 and isinstance(obj[0], str)
                        and obj[0] in ('I;', 'L;')):
                    out.append('[' + obj[0] + '\n')
                    items = obj[1:]
                else:
                    out.append('[\n')
                    items = obj
                for v in items:
                    out.append(tab + '\t')
                    snbtlib._serialize_value(v, out, indent + 1)
                out.append(tab + ']\n')
        elif isinstance(obj, bytes):
            # ByteArray → [B; ...]
            if not obj:
                out.append('[B; ]\n')
            else:
                out.append('[B;\n')
                for byte_val in obj:
                    out.append(tab + '\t' + str(byte_val) + 'b\n')
                out.append(tab + ']\n')
        else:
            snbtlib._serialize_value(obj, out, indent)
    @staticmethod

    def _serialize_value(val: Any, out: List[str], indent: int):
        """序列化单个值（非容器类型）。"""
        if isinstance(val, (dict, list, bytes)):
            snbtlib._serialize(val, out, indent)
        elif isinstance(val, str):
            if val.startswith(snbtlib._NUMBER_PREFIX):
                # 数值：去掉 $number$ 前缀，输出原始形式
                out.append(val[len(snbtlib._NUMBER_PREFIX):] + '\n')
            else:
                # 普通字符串：转义后加引号
                # 控制字符必须一并转义(模型给出的多行段落会带真实换行), 否则字符串被换行撑破引号,
                # FTBQ 解析整份文件时报 New line without closing string 而丢掉全部译文。
                escaped = (val.replace('\\', '\\\\').replace('"', '\\"')
                           .replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'))
                out.append(f'"{escaped}"\n')
        elif isinstance(val, bool):
            out.append(('true' if val else 'false') + '\n')
        elif isinstance(val, (int, float)):
            out.append(str(val) + '\n')
        elif val is None:
            out.append('null\n')
        else:
            out.append(str(val) + '\n')
    @staticmethod

    def _compatible(text: str) -> str:
        """紧凑兼容模式：在需要的地方插入逗号分隔符。
        用于 BetterQuesting 等需要逗号分隔的 SNBT 变体格式。
        """
        if not text:
            return ''
        lines = text.splitlines()
        for i in range(len(lines) - 1):
            curr = lines[i].rstrip()
            nxt = lines[i + 1].lstrip()
            if not curr or not nxt:
                continue
            # 当前行不以 [ { ; 结尾，下一行不以 ] } 开头 → 需要逗号
            if curr[-1] not in '[{;' and nxt[0] not in ']}':
                lines[i] = curr + ','
        return '\n'.join(lines)


snbtlib = snbtlib()          # 解析库实例(随模组走)
模组参数 = Mods().模组配置("trans-ftbquests", {
    "FILE_MATCH": "*.snbt",
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


def 尝试提取内嵌JSON(文件, 路径前缀, 字符串, 输出列表):
    try:
        解析结果 = json.loads(字符串)
    except (json.JSONDecodeError, TypeError):
        输出列表.append([[文件, 路径前缀], 字符串])
        return
    if isinstance(解析结果, dict):
        if "text" in 解析结果:
            输出列表.append([[文件, 路径前缀 + [解析结果, "text"]], 解析结果["text"]])
        elif "translate" in 解析结果:
            输出列表.append([[文件, 路径前缀 + [解析结果, "translate"]], 解析结果["translate"]])
    else:
        输出列表.append([[文件, 路径前缀], 字符串])


class FTBQuestParser(Parser):

    def 读取(Self, 文件路径: str):
        index = 文件路径
        文本列表 = []
        路径对象 = Path(文件路径)
        # 新版 FTBQ 把文案集中在 quests/lang/<语言>.snbt(键名自带 .title/.quest_desc), 只翻源语言那份
        语言文件 = 路径对象.parent.name.lower() == "lang"
        if 语言文件 and 路径对象.stem.lower() != str(Self.Config.LANGUAGE_INPUT).lower():
            return 文本列表
        try:
            SNBT文件 = snbtlib.loads(Path(index).read_text(encoding='utf-8'))
        except Exception:
            Self.日志("log.module.quests.load.error", mod="FTB Quests", file=index, e=eb.format_exc(), info_level=2)
            return 文本列表
        目标字段 = {"title", "subtitle", "text", "pages", "Lore", "Name"}
        可能含JSON的字段 = {"description", "text"}
        def 提取(obj, 路径):
            if isinstance(obj, dict):
                for 键, 值 in obj.items():
                    当前路径 = 路径 + [键]
                    if 语言文件 or 键 in 目标字段:      # 语言文件里每条都是要翻的文案
                        if isinstance(值, str):
                            文本列表.append([[index, 当前路径], 值])
                        elif isinstance(值, list):
                            for 序号, 元素 in enumerate(值):
                                if isinstance(元素, str):
                                    文本列表.append([[index, 当前路径 + [序号]], 元素])
                                else:
                                    提取(元素, 当前路径 + [序号])
                        else:
                            提取(值, 当前路径)
                    elif 键 in 可能含JSON的字段:
                        if isinstance(值, str):
                            尝试提取内嵌JSON(index, 当前路径, 值, 文本列表)
                        elif isinstance(值, list):
                            for 序号, 元素 in enumerate(值):
                                if isinstance(元素, str):
                                    尝试提取内嵌JSON(index, 当前路径 + [序号], 元素, 文本列表)
                                else:
                                    提取(元素, 当前路径 + [序号])
                        else:
                            提取(值, 当前路径)
                    else:
                        提取(值, 当前路径)
            elif isinstance(obj, list):
                for 序号, 元素 in enumerate(obj):
                    提取(元素, 路径 + [序号])
        提取(SNBT文件, [])
        return 文本列表

    def 写回(Self, index: list, mode: str = None):
        mode = mode or ('H' if 'quests' in str(index[0][0] if index else '') else 'L')
        文件映射 = {}
        for 条目 in index:
            文件路径 = 条目[0][0]
            路径段 = 条目[0][1]
            译文 = 条目[1]
            文件映射.setdefault(文件路径, []).append((路径段, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            源路径 = Path(文件路径)
            # 语言文件不覆盖原文: 照 FTBQ 的规矩另存一份 <目标语言>.snbt, en_us 原样留着
            目标路径 = 源路径
            if 源路径.parent.name.lower() == "lang":
                目标路径 = 源路径.with_name(f"{Self.Config.LANGUAGE_OUTPUT}{源路径.suffix}")
            SNBT文件 = snbtlib.loads(源路径.read_text(encoding='utf-8'))
            for 路径, 译文 in 替换列表:
                try:
                    当前对象 = SNBT文件
                    父对象 = None  # 【新增】用来记录倒数第二层的容器
                    for i, 段 in enumerate(路径):
                        if isinstance(段, dict):
                            # --- 处理内嵌 JSON 的逻辑 (保持不变) ---
                            解析对象 = 段
                            剩余路径 = 路径[i+1:]
                            目标 = 解析对象
                            for p in 剩余路径[:-1]:
                                if isinstance(目标, list): p = int(p)
                                目标 = 目标[p]
                            最后键 = 剩余路径[-1]
                            if isinstance(目标, list): 最后键 = int(最后键)
                            目标[最后键] = 译文
                            新字符串 = json.dumps(解析对象, ensure_ascii=False)
                            break  # 遇到 JSON 提前跳出
                        # 【核心修复】：在向下钻取前，先记录当前的容器为“父对象”
                        父对象 = 当前对象
                        if isinstance(当前对象, list):
                            段 = int(段)
                        当前对象 = 当前对象[段]
                    else:
                        # 循环正常结束（没有遇到 JSON 提前 break），说明到了最底层
                        if 父对象 is not None:
                            最后键 = 路径[-1]
                            if isinstance(父对象, list):
                                最后键 = int(最后键)
                            # 【核心修复】：使用“父对象”来赋值，而不是已经变成字符串的“当前对象”
                            父对象[最后键] = 译文
                        continue
                    # --- 写回 JSON 字符串的逻辑 (保持不变) ---
                    当前对象 = SNBT文件
                    for j, 段 in enumerate(路径):
                        if isinstance(段, dict): break
                        if isinstance(当前对象, list): 段 = int(段)
                        当前对象 = 当前对象[段]
                    父容器 = SNBT文件
                    for j, 段 in enumerate(路径):
                        if isinstance(段, dict):
                            前一段 = 路径[j-1]
                            if isinstance(父容器, list): 前一段 = int(前一段)
                            父容器[前一段] = 新字符串
                            break
                        if isinstance(父容器, list): 段 = int(段)
                        父容器 = 父容器[段]
                except Exception:
                    Self.日志("log.module.quests.write.error", file=文件路径, path=str(路径), e=eb.format_exc(), info_level=2)
            目标路径.write_text(snbtlib.dumps(SNBT文件, compact=False if mode=="H" else True), encoding='utf-8', newline="")


# ============================================================
# 单个 .snbt 文件输入 —— 内容级探测
#
# .snbt 是 FTB Library 的通用存档后缀, 整个整合包里有上百个同名文件:
#   config/ftbchunks-world.snbt / ftbessentials.snbt / ftbteams-server.snbt ...
# 只认后缀必然误判, 所以按"内容特征 + 目录形态"认领 FTBQ 自己的文件:
#   quests/lang/<语言>.snbt         文案表     键形如 chapter.<16位ID>.title
#   quests/chapters/*.snbt          章节文件   filename + chapter_groups + quests
#   quests/chapter_groups.snbt      章节分组   chapter_groups(内容太弱, 靠目录兜底)
#   quests/data.snbt                任务书配置 default_quest_shape + progression_mode
#   quests/reward_tables/*.snbt     奖励表     rewards + order_index
# 认不出来就返回 False, 交给别的翻译类型(或保守的语言文件流程)处理。
# ============================================================

_探测缓存 = {}

_LANG_KEY_PAT = _re_compile(r'^\s*(?:chapter|chapter_group|quest|task|reward|reward_table)'
                            r'\.[0-9A-Fa-f]{16}\.(?:title|subtitle|chapter_subtitle|quest_subtitle'
                            r'|quest_desc|text|description|chapter_desc)\s*:')
# 文案表每行的键都内嵌一个 16 位十六进制 ID —— 用于"行数太少没能充分抽样"时的兜底确认
_LANG_ID_PAT = _re_compile(r'\.[0-9A-Fa-f]{16}\.')
_任务书配置名 = frozenset({'data.snbt', 'chapter_groups.snbt'})


def 探测SNBT(Self, 路径):
    """单个 .snbt 文件是不是 FTBQ 的文件 —— 按内容判定, 不只看后缀。"""
    路径对象 = Path(路径)
    if 路径对象.suffix.lower() != '.snbt':
        return False
    路径文本 = str(路径对象)
    if 路径文本 in _探测缓存:
        return _探测缓存[路径文本]
    结果 = False
    try:
        文本 = 路径对象.read_text(encoding='utf-8', errors='ignore')
        # ① 文案表: 抽样命中足够多的 键.ID.字段 行即认定, 不整份解析(最大那份约 600KB)
        命中 = 0
        for 行 in 文本.split('\n', 400):
            if _LANG_KEY_PAT.match(行):
                命中 += 1
                if 命中 >= 4:
                    break
        结果 = 命中 >= 4
        if not 结果 and 路径对象.stem.lower() in ('zh_cn', 'en_us'):
            # 文案表可能整份都带缩进/是空壳, 再顺序看一眼有没有 16 位 ID 键
            结果 = bool(_LANG_ID_PAT.search(文本))
        if not 结果:
            根 = snbtlib.loads(文本)
            if isinstance(根, dict):
                有分组 = isinstance(根.get('chapter_groups'), list)
                有任务 = 'quests' in 根
                有奖励表 = 'rewards' in 根 and 'order_index' in 根
                结果 = (
                    ('filename' in 根 and (有任务 or 有分组))            # 章节文件
                    or (有分组 and 有任务)                              # 分组 + 任务齐全
                    or ('default_quest_shape' in 根 and 'progression_mode' in 根)  # data.snbt
                    or 有奖励表                                         # 奖励表
                )
        if not 结果:
            # ② 兜底: 内容特征太弱时认目录形态 —— .../quests/<chapters|lang|reward_tables>/*.snbt
            父名 = 路径对象.parent.name.lower()
            结果 = (父名 in ('chapters', 'lang', 'reward_tables') and 路径对象.parent.parent.name.lower() == 'quests') \
                or (父名 == 'quests' and 路径对象.name.lower() in _任务书配置名)
    except Exception:
        结果 = False
    _探测缓存[路径文本] = 结果
    return 结果


Mods().注册翻译类型(
    "FTBQuests", "翻译流程", 文件匹配=("trans-ftbquests", "FILE_MATCH"), 解析器类=FTBQuestParser,
    过滤="Module.过滤键文本", 分组键="文件",
    读并发=("trans-ftbquests", "READ_CONCURRENT"), 写并发=("trans-ftbquests", "WRITE_CONCURRENT"),
    日志类型="quests", 显示名="FTBQuests",
    文件探测=探测SNBT, 探测分数=80, 产物="源文件")


Mods().注册翻译类型到整合包("FTBQuests", "config/ftbquests", 注册路径="/config/ftbquests",
                     打包扩展名={".json", ".lang", ".snbt"})


Mods().注册翻译类型到归档("FTBQuests", "ftbquests", 注册路径="/ftbquests",
                     打包扩展名={".snbt"}, 最大深度=2)
