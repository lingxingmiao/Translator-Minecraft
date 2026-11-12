# 可用配置(覆盖键 MOD_TRANS_HQM_<键>, 全大写):
#   FILE_MATCH        文件匹配 glob   ["*.hqm", "*.json"]
#   READ_CONCURRENT   读取并发        4
#   WRITE_CONCURRENT  写入并发        4

from TranslatorLib import (json, Any, List, re, struct, IntEnum, SimpleNamespace, Parser, Mods, eb, Optional, Dict, BytesIO)


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
        with open(path, 'w', encoding='utf-8', newline='') as f:
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


hqmlib = hqmlib()          # 解析库实例(随模组走)
模组参数 = Mods().模组配置("trans-hqm", {
    "FILE_MATCH": ["*.hqm", "*.json"],
    "READ_CONCURRENT": 4,
    "WRITE_CONCURRENT": 4,
}, 索引指纹=False)


class HQMParser(Parser):

    def 读取(Self, 文件路径: str, mode: str = None) -> list:
        mode = mode or ('L' if str(文件路径).lower().endswith('.hqm') else 'H')
        结果 = []
        try:
            if mode == "H":
                数据 = Self.读取Json文件(文件路径)
            elif mode == "L":
                数据 = hqmlib.load(文件路径, encoding='gbk')
            Self.递归提取文本(数据, [], {"main_description", "name", "description", "long_description", "longDescription"}, lambda 路径, 值: 结果.append([[文件路径, 路径], 值]))
        except Exception: Self.日志("log.module.quests.load.error", mod="Hardcore Questing Mode", file=文件路径, e=eb.format_exc(),  info_level=2)
        return 结果

    def 写回(Self, index: list, mode: str = None) -> None:
        mode = mode or ('L' if any(str(条目[0][0]).lower().endswith('.hqm') for 条目 in index) else 'H')
        文件映射 = {}
        for 条目 in index:
            位置 = 条目[0]
            译文 = 条目[1]
            文件路径 = 位置[0]
            路径列表 = 位置[1]
            文件映射.setdefault(文件路径, []).append((路径列表, 译文))
        for 文件路径, 替换列表 in 文件映射.items():
            try:
                if mode == "H":
                    数据 = Self.读取Json文件(文件路径)
                elif mode == "L":
                    数据 = hqmlib.load(文件路径, encoding='gbk')
            except Exception: Self.日志("log.module.quests.load.error", mod="Hardcore Questing Mode", file=文件路径, e=eb.format_exc(), info_level=2); continue
            for 路径, 译文 in 替换列表:
                try:
                    Self.设置路径值(数据, 路径, 译文)
                except Exception: Self.日志("log.module.quests.write.error", mod="Hardcore Questing Mode",  file=文件路径, path=str(路径), e=eb.format_exc(), info_level=2)
            if mode == "H":
                Self.写入Json文件(文件路径, 数据)
            elif mode == "L":
                hqmlib.dump_to_hqm(数据, 文件路径, encoding='gbk')
    # ========== MC 指令提取 ==========


Mods().注册翻译类型(
    "HardcoreQuestingMode", "翻译流程", 文件匹配=("trans-hqm", "FILE_MATCH"), 解析器类=HQMParser,
    过滤="Module.过滤键文本", 分组键="文件",
    读并发=("trans-hqm", "READ_CONCURRENT"), 写并发=("trans-hqm", "WRITE_CONCURRENT"),
    日志类型="quests", 显示名="HardcoreQuestingMode")


Mods().注册翻译类型到整合包("HardcoreQuestingMode", "config/hqm", 注册路径="/config/hqm",
                     打包扩展名={".json", ".lang", ".hqm"})


Mods().注册翻译类型到归档("HardcoreQuestingMode", "hqm", 注册路径="/hqm",
                     打包扩展名={".hqm", ".json"}, 最大深度=2)
