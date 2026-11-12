# 可用配置(覆盖键 MOD_TRANS_FANCYMENU_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   "*.txt"
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (re, Any, List, Path, Parser, Mods, eb, OrderedDict, Dict, TextIO)


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


fancymenulib = fancymenulib()
模组参数 = Mods().模组配置("trans-fancymenu", {
    "FILE_MATCH": "*.txt",
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


def 执行(Self, 根, path2=None, **参数):
    """customization/*.txt + locals/*.local —— 整条流程归本模组"""
    解析 = FMParser(Self)
    if Path(f"{根}/customization").is_dir():
        Self.翻译流程(f"{根}/customization", Mods().取值(Self, ("trans-fancymenu", "FILE_MATCH")), 解析.读取,
                    Self.Module.过滤键文本, lambda x: x[0][0], 解析.写回,
                    Mods().取值(Self, ("trans-fancymenu", "READ_CONCURRENT")),
                    Mods().取值(Self, ("trans-fancymenu", "WRITE_CONCURRENT")), "menu", path2=path2, **参数)
    if Path(f"{根}/locals").is_dir():
        参数表 = {"file0": f"{根}/locals/{Self.Config.LANGUAGE_INPUT}.local", "output_path": f"{根}/locals", "output_lang_str": True}
        参数表["file1"] = f"{根}/locals/{Self.Config.LANGUAGE_OUTPUT}.local" if any(
            Self.Config.LANGUAGE_OUTPUT.lower() in p.name.lower() for p in Path(f"{根}/locals").iterdir()
            if p.is_file() and p.suffix == '.local') else ""
        Self.翻译语言文件(**参数表, **参数)


class FMParser(Parser):

    def 读取(Self, 文件路径: str) -> list:
        结果 = []
        try:
            with open(文件路径, 'r', encoding='utf-8') as f:
                布局数据 = fancymenulib.load(f)
        except Exception:
            Self.日志("log.module.menu.load.error", mod="FancyMenu", file=文件路径, e=eb.format_exc(), info_level=2)
            return 结果
        elements = 布局数据.get('element')
        if elements is None:
            return 结果
        if not isinstance(elements, list):
            elements = [elements]
        for el in elements:
            if not isinstance(el, dict):
                continue
            实例ID = el.get('instance_identifier', 'unknown')
            if 'source' in el and isinstance(el['source'], str):
                source文本 = el['source']
                if not re.match(r'^\s*\[[^\]]*\]', source文本):
                    结果.append([[文件路径, 实例ID, 'source'], source文本])
            if 'label' in el:
                结果.append([[文件路径, 实例ID, 'label'], el['label']])
            if 'hoverlabel' in el:
                结果.append([[文件路径, 实例ID, 'hoverlabel'], el['hoverlabel']])
        return 结果

    def 写回(Self, 翻译列表: list) -> None:
        文件映射 = {}
        for 条目 in 翻译列表:
            位置 = 条目[0]
            译文 = 条目[1]
            文件路径 = 位置[0]
            文件映射.setdefault(文件路径, []).append((位置[1], 位置[2], 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            try:
                with open(文件路径, 'r', encoding='utf-8') as f:
                    布局数据 = fancymenulib.load(f)
            except Exception:
                Self.日志("log.module.menu.load.error", mod="FancyMenu", file=文件路径, e=eb.format_exc(), info_level=2)
                continue
            elements = 布局数据.get('element')
            if elements is None:
                continue
            is_single = not isinstance(elements, list)
            if is_single:
                elements = [elements]
            el_map = {}
            for i, el in enumerate(elements):
                if isinstance(el, dict) and 'instance_identifier' in el:
                    el_map[el['instance_identifier']] = i
            for 实例ID, 字段名, 译文 in 替换列表:
                if 实例ID in el_map:
                    idx = el_map[实例ID]
                    elements[idx][字段名] = 译文
                else:
                    Self.日志("log.module.menu.write.error", mod="FancyMenu", file=文件路径, id=实例ID, field=字段名, info_level=2)
            if is_single:
                布局数据['element'] = elements[0]
            else:
                布局数据['element'] = elements
            try:
                with open(文件路径, 'w', encoding='utf-8', newline="") as f:
                    fancymenulib.dump(布局数据, f)
            except Exception:
                Self.日志("log.module.menu.write.error", mod="FancyMenu", file=文件路径, e=eb.format_exc(), info_level=2)


Mods().注册翻译类型("FancyMenu", 执行, 解析器类=FMParser, 显示名="FancyMenu")


Mods().注册翻译类型到整合包("FancyMenu", "config/fancymenu", 注册路径="/config/fancymenu",
                     打包扩展名={".json", ".lang", ".local", ".txt"})


Mods().注册翻译类型到归档("FancyMenu", "fancymenu", 注册路径="/fancymenu",
                     打包扩展名={".txt", ".local", ".json"}, 最大深度=2)
