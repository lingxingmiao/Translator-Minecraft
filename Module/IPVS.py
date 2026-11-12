from TranslatorLib import (io, np, asyncio, Path, struct, Image, faiss, GPU_ACC,
                           zipfile, ThreadPoolExecutor, TranslatorPersistence, Config)

class IbisPaintIPV:
    _HDR_W_OFF = 0x28
    _HDR_H_OFF = 0x2C
    _HDR_ALPHA_OFF = 0x30
    _HDR_BLEND_OFF = 0x34

    def __init__(self, data: bytes):
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data 必须是 bytes / bytearray")
        self._data = bytes(data)
        self.canvas_size = self._parse_canvas_size()
        self._meta_strings = self._parse_header_strings()
        self._png_offsets = self._find_png_offsets()
        if not self._png_offsets:
            raise ValueError("IPV 数据中未找到 PNG 图像块")
    @classmethod
    def from_path(cls, path):
        with open(path, "rb") as fp:
            return cls(fp.read())
    def _parse_canvas_size(self):
        try:
            w = struct.unpack(">H", self._data[0x12:0x14])[0]
            h = struct.unpack(">H", self._data[0x16:0x18])[0]
            if w > 100 and h > 100:
                return (w, h)
        except Exception:
            pass
        return None
    def _read_pascal_string(self, offset):
        if offset >= len(self._data):
            return None, offset
        ln = self._data[offset]
        if 1 <= ln <= 200 and offset + 1 + ln <= len(self._data):
            try:
                s = self._data[offset + 1: offset + 1 + ln].decode("utf-8")
                return s, offset + 1 + ln
            except Exception:
                return None, offset
        return None, offset

    def _parse_header_strings(self):
        meta = {}
        for start in (0x19, 0x3D, 0x4C, 0x57):
            s, _ = self._read_pascal_string(start)
            if s and s.isprintable():
                if start in (0x19,):
                    meta.setdefault("session", s)
                elif start in (0x3D,):
                    meta.setdefault("app", s)
                elif start in (0x4C,):
                    meta.setdefault("version", s)
                elif start in (0x57,):
                    meta.setdefault("device", s)
        if "version" not in meta:
            head = self._data[:0x100]
            for needle in (b"Ver.", b"ver."):
                idx = head.find(needle)
                if idx >= 0:
                    end = head.find(b"\x00", idx)
                    if end < 0:
                        end = idx + 16
                    ver = head[idx:end].decode("utf-8", "replace").strip()
                    if ver:
                        meta["version"] = ver
                        break
        return meta

    def _find_png_offsets(self):
        sig = b"\x89PNG\r\n\x1a\n"
        offsets = []
        idx = 0
        while True:
            i = self._data.find(sig, idx)
            if i < 0:
                break
            offsets.append(i)
            idx = i + 1
        return offsets

    @staticmethod
    def _png_end(data, start):
        p = start + 8
        while p + 8 <= len(data):
            ln = struct.unpack(">I", data[p:p + 4])[0]
            typ = data[p + 4:p + 8]
            p += 8 + ln + 4
            if typ == b"IEND":
                return p
        return -1
    def to_pil(self, prefer_final: bool = True) -> Image.Image:
        offsets = self._png_offsets
        if prefer_final:
            po = offsets[-1]
            end = self._png_end(self._data, po)
            if end > 0:
                img = Image.open(self._bytesio(po, end)).convert("RGBA")
                if self.canvas_size:
                    self.canvas_size = img.size
                return img
        po = offsets[0]
        end = self._png_end(self._data, po)
        return Image.open(self._bytesio(po, end)).convert("RGBA")
    def to_numpy(self, prefer_final: bool = True) -> np.ndarray:
        pil_img = self.to_pil(prefer_final=prefer_final)
        return np.asarray(pil_img, dtype=np.uint8)

    def _bytesio(self, start, end):
        return io.BytesIO(self._data[start:end])
    @property
    def version(self):
        return self._meta_strings.get("version")

    @property
    def device(self):
        return self._meta_strings.get("device")

    @property
    def app(self):
        return self._meta_strings.get("app")

    @property
    def session(self):
        return self._meta_strings.get("session")

    def __repr__(self):
        return (
            "<IbisPaintIPV canvas=%s version=%s device=%s pngs=%d>"
            % (self.canvas_size, self.version, self.device,
               len(self._png_offsets))
        )

class HuaShiJieHSJ:
    def __init__(self, data: bytes):
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data 必须是 bytes / bytearray")
        self._data = bytes(data)
        self._payload = self._unzip(self._data)
        self._parse_header()
        self._parse_index()

    @classmethod
    def from_path(cls, path):
        with open(path, "rb") as fp:
            return cls(fp.read())

    @staticmethod
    def _unzip(raw: bytes) -> bytes:
        if not raw.startswith(b"PK\x03\x04"):
            return raw
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            if not names:
                raise ValueError("ZIP 内没有条目")
            return z.read(names[0])

    @staticmethod
    def _varint(b, i, end):
        v = 0
        s = 0
        while True:
            if i >= end:
                raise ValueError("varint 越界")
            c = b[i]
            i += 1
            v |= (c & 0x7F) << s
            if not (c & 0x80):
                return v, i
            s += 7
            if s > 63:
                raise ValueError("varint 过长")

    @classmethod
    def _fields(cls, b, off, end):
        i = off
        while i < end:
            try:
                key, i = cls._varint(b, i, end)
            except Exception:
                return
            fn, wt = key >> 3, key & 7
            if fn == 0:
                return
            if wt == 0:
                try:
                    v, i = cls._varint(b, i, end)
                except Exception:
                    return
                yield fn, wt, v
            elif wt == 2:
                try:
                    ln, i = cls._varint(b, i, end)
                except Exception:
                    return
                if ln < 0 or i + ln > end:
                    return
                yield fn, wt, b[i:i + ln]
                i += ln
            elif wt == 5:
                if i + 4 > end:
                    return
                yield fn, wt, b[i:i + 4]
                i += 4
            elif wt == 1:
                if i + 8 > end:
                    return
                yield fn, wt, b[i:i + 8]
                i += 8
            else:
                return

    @staticmethod
    def _text(b):
        if not b or len(b) > 512:
            return None
        try:
            s = b.decode("utf-8")
        except Exception:
            return None
        return s if s.isprintable() else None

    def _parse_header(self):
        p = self._payload
        if len(p) < 0x30:
            raise ValueError("payload 太短")
        if not p.startswith(self._MAGIC):
            raise ValueError("不是 HSJ payload（魔数不符）")
        self.header_len = struct.unpack(">I", p[0x08:0x0C])[0]
        self.total_size = struct.unpack(">Q", p[0x10:0x18])[0]
        self.trailer_len = struct.unpack(">I", p[0x20:0x24])[0]
        self._t0 = len(p) - self.trailer_len

    def _parse_index(self):
        tr = self._payload[self._t0:]
        self.index = {}
        self.files = []                      # [(路径, 偏移, 大小)]
        for fn, wt, v in self._fields(tr, 0, len(tr)):
            if fn == 12 and wt == 2:
                path = None
                nums = []
                for g, gw, gv in self._fields(v, 0, len(v)):
                    if gw == 2:
                        s = self._text(gv)
                        if s:
                            path = s
                    elif gw == 0:
                        nums.append(gv)
                if path:
                    off = nums[0] if nums else 0
                    size = nums[1] if len(nums) > 1 else 0
                    self.files.append((path, off, size))
            elif wt == 2:
                s = self._text(v)
                if s:
                    self.index["f%d" % fn] = s
            elif wt == 0:
                self.index["f%d" % fn] = v

        idx = self.index
        self.project_id = idx.get("f1")
        self.modified = idx.get("f2")
        self.storage = idx.get("f37")
        self.cover_size = (idx.get("f3"), idx.get("f4"))
        w = idx.get("f13") or idx.get("f27") or idx.get("f30")
        h = idx.get("f14") or idx.get("f28")
        self.canvas_size = (w, h) if w and h else None
        self.layer_count = idx.get("f31")

    def _entry(self, path):
        for p, off, size in self.files:
            if p == path:
                return off, size
        return None

    def read(self, path):
        hit = self._entry(path)
        if not hit:
            return None
        off, size = hit
        if size <= 0 or off + size > len(self._payload):
            return None
        return self._payload[off:off + size]

    @property
    def cover_bytes(self):
        return self.read("/cover")

    def to_pil(self) -> Image.Image:
        raw = self.cover_bytes
        if raw is None:
            raise ValueError("没有 /cover 资源（成品图）")
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        self.cover_size = img.size
        return img

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.to_pil(), dtype=np.uint8)

    @property
    def app(self):
        return "HuaShiJiePro"

    def __repr__(self):
        return ("<HuaShiJieHSJ canvas=%s cover=%s actions=%d files=%d>"
                % (self.canvas_size, self.cover_size,
                   self.index.get("f31") or 0, len(self.files)))

# ↓可选配置项 不写进 DefaultConfig 由 RuntimeConfig 遇未知键自动注册 未配置时走 getattr 兜底
#   EMB_IMG_DECODE_WORKERS  图像解码并行线程数 默认4 每个线程持有一整幅位图 过高会同时驻留多张大图而爆内存
#   EMB_IMG_DB_CHUNK        每批处理的图像张数 默认128 批内完成 解码→算向量→释放 峰值只与批大小挂钩
class IPVS:
    def __init__(Self, App: Config):
        Self.Config       = App.Config
        Self.日志         = App.日志
        Self.Lang         = App.Lang
        Self.Builder      = App.Builder
        Self.Quantization = App.Quantization
        Self.Index        = App.Index
        Self.tqdm         = App.RichTqdm
    def 添加图像到数据库(Self, path):
        if isinstance(path, (list, tuple, set)):
            图像路径列表 = list(path)
        elif isinstance(path, str):
            p = Path(path)
            if p.is_file():
                图像路径列表 = [str(p)]
            elif p.is_dir():
                支持扩展 = ("*.ipv", "*.IPV", "*.hsj", "*.HSJ",
                          "*.png", "*.PNG",
                          "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.psd", "*.PSD")
                图像路径列表 = list({str(x) for ext in 支持扩展 for x in p.glob("**/" + ext)})
            else:
                raise FileNotFoundError(f"路径不存在: {path}")
        else:
            raise TypeError("path 必须是 str 或 str 列表")
        if not 图像路径列表: return
        _, 文本列表 = asyncio.run(TranslatorPersistence.参考词预处理(Self, 图像=True))
        路径集 = {index[1] for index in (文本列表 or []) if isinstance(index, (list, tuple)) and len(index) > 1}
        待处理路径 = [item for item in 图像路径列表 if item not in 路径集]
        总数 = len(待处理路径)
        if 总数 == 0: return
        最大边长 = getattr(Self.Config, "EMB_IMG_MAX_SIDE", 448) or None
        解码线程 = max(1, min(getattr(Self.Config, "EMB_IMG_DECODE_WORKERS", 4) or 4, 总数))
        每批数量 = max(1, getattr(Self.Config, "EMB_IMG_DB_CHUNK", 128) or 128)
        def 读取图像(图像路径):
            try:
                后缀 = Path(图像路径).suffix.lower()
                if 后缀 == ".ipv":
                    图像 = IbisPaintIPV.from_path(图像路径).to_pil(prefer_final=True)
                elif 后缀 == ".hsj":
                    图像 = HuaShiJieHSJ.from_path(图像路径).to_pil()
                else:
                    with Image.open(图像路径) as 原始:
                        if 最大边长 and 后缀 in (".jpg", ".jpeg"):
                            原始.draft("RGB", (最大边长, 最大边长))
                        图像 = 原始.convert("RGBA")
                if 最大边长 and max(图像.size) > 最大边长:
                    图像.thumbnail((最大边长, 最大边长), Image.BILINEAR)
                return [图像, 图像路径, ""]
            except Exception:
                Self.日志("log.core.file.ipvs.read.error", item=图像路径, info_level=1)
                return None
        进度条 = Self.tqdm(total=总数, desc="转换图像")
        预生成向量, 待入库路径, 待收批次 = {}, [], []
        def 收批次(条目):
            生成列表, 任务 = 条目
            try:
                向量矩阵 = np.asarray(任务.result()[0], dtype=np.float32)
                for 序号, item in enumerate(生成列表):
                    预生成向量[item[1]] = 向量矩阵[序号]
                    待入库路径.append(item[1])
            finally:
                for item in 生成列表:
                    try: item[0].close()
                    except Exception: pass
        try:
            with ThreadPoolExecutor(max_workers=解码线程) as 解码执行器, ThreadPoolExecutor(max_workers=1) as 向量执行器:
                for 起始 in range(0, 总数, 每批数量):
                    批路径 = 待处理路径[起始:起始 + 每批数量]
                    生成列表 = []
                    for 结果 in 解码执行器.map(读取图像, 批路径):
                        进度条.update(1)
                        if 结果 is not None and 结果[1] not in 路径集:
                            生成列表.append(结果)
                    if 生成列表:
                        待收批次.append([生成列表, 向量执行器.submit(lambda 批=生成列表: asyncio.run(Self.Builder.并行生成图像向量(批, use_cache=False)))])
                    if len(待收批次) >= 2:
                        收批次(待收批次.pop(0))
                while 待收批次: 收批次(待收批次.pop(0))
        finally:
            for 条目 in 待收批次:
                for item in 条目[0]:
                    try: item[0].close()
                    except Exception: pass
            进度条.close()
        if not 待入库路径: return
        入库列表 = [[None, 路径, ""] for 路径 in 待入库路径]
        asyncio.run(TranslatorPersistence.参考词预处理(Self, 入库列表, use_cache=False, 图像=True, 预生成向量=预生成向量))
    def 导出结果PNG(Self, 结果列表, 输出目录: str = "IPVSOutput"):
        try:
            目录 = Path(输出目录)
            目录.mkdir(parents=True, exist_ok=True)
            for 旧文件 in 目录.iterdir():
                if 旧文件.is_file(): 旧文件.unlink()
        except Exception:
            Self.日志("log.core.file.ipvs.read.error", item=输出目录, info_level=1)
            return
        for 序号, 项 in enumerate(结果列表, 1):
            路径 = 项[0] if isinstance(项, (list, tuple)) and 项 else 项
            try:
                p = Path(路径)
                后缀 = p.suffix.lower()
                if 后缀 == ".ipv":
                    图像 = IbisPaintIPV.from_path(str(p)).to_pil(prefer_final=True)
                elif 后缀 == ".hsj":
                    图像 = HuaShiJieHSJ.from_path(str(p)).to_pil()
                else:
                    with Image.open(str(p)) as 原始:
                        图像 = 原始.convert("RGBA")
                图像.save(目录 / f"{序号:02d}_{p.parent.name}群_{p.stem}.png")  # <搜索顺序>_<群号>群_<名称>
            except Exception:
                Self.日志("log.core.file.ipvs.read.error", item=str(路径), info_level=1)
    def 使用图像搜索(Self, path, K: int = None) -> list:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"路径不存在: {path}")
        if p.suffix.lower() == ".ipv":
            图像 = IbisPaintIPV.from_path(str(p)).to_pil(prefer_final=True)
        elif p.suffix.lower() == ".hsj":
            图像 = HuaShiJieHSJ.from_path(str(p)).to_pil()
        else:
            图像 = Image.open(str(p)).convert("RGBA")
        向量文件, 文本文件 = asyncio.run(TranslatorPersistence.参考词预处理(Self, 查询=False))
        if not 向量文件 or not 文本文件:
            print("无数据库，请先添加数据")
            return []
        向量索引 = TranslatorPersistence.缓存索引(Self, 向量文件=向量文件, 文本文件=文本文件)
        输入列表 = asyncio.run(Self.Builder.并行生成图像向量([[图像, str(p), ""]], use_cache=False))
        查询向量 = np.asarray(输入列表[0], dtype=np.float32)
        if 查询向量.shape[0] == 0:
            return []
        Self.Quantization.SpecTemp应用懒加载(查询向量, 向量文件)
        查询向量 = 查询向量.get() if GPU_ACC else 查询向量
        faiss.normalize_L2(查询向量)
        K = K or Self.Config.INDEX_TEXT_K
        try:
            K = min(K, 向量索引.ntotal)
        except Exception:
            pass
        距离矩阵, 索引矩阵 = 向量索引.search(查询向量, K)
        结果列表 = []
        for i, 距离 in zip(索引矩阵[0], 距离矩阵[0]):
            if 0 <= i < len(文本文件):
                index = 文本文件[i]
                路径 = index[1] if isinstance(index, (list, tuple)) and len(index) > 1 else index[0]
                结果列表.append([路径, float(距离)])
        Self.导出结果PNG(结果列表)
        return 结果列表
    def 使用文本搜索(Self, 文本: str, K: int = None) -> list:
        向量文件, 文本文件 = asyncio.run(TranslatorPersistence.参考词预处理(Self, 查询=False))
        if not 向量文件 or not 文本文件:
            print("无数据库，请先添加数据")
            return []
        向量索引 = TranslatorPersistence.缓存索引(Self, 向量文件=向量文件, 文本文件=文本文件)
        输入列表 = asyncio.run(Self.Builder.并行生成向量([[文本, "", ""]], 查询=True))
        查询向量 = np.asarray(输入列表[0], dtype=np.float32)
        if 查询向量.shape[0] == 0:
            return []
        Self.Quantization.SpecTemp应用懒加载(查询向量, 向量文件)
        查询向量 = 查询向量.get() if GPU_ACC else 查询向量
        faiss.normalize_L2(查询向量)
        K = K or Self.Config.INDEX_TEXT_K
        try:
            K = min(K, 向量索引.ntotal)
        except Exception:
            pass
        距离矩阵, 索引矩阵 = 向量索引.search(查询向量, K)
        结果列表 = []
        for i, 距离 in zip(索引矩阵[0], 距离矩阵[0]):
            if 0 <= i < len(文本文件):
                index = 文本文件[i]
                路径 = index[1] if isinstance(index, (list, tuple)) and len(index) > 1 else index[0]
                结果列表.append([路径, float(距离)])
        Self.导出结果PNG(结果列表)
        return 结果列表
    
    def 使用文件名搜索(Self, 关键词, 根目录: str = ".", 上限: int = 0) -> list:
        根 = Path(根目录)
        if not 根.is_dir():
            raise FileNotFoundError(f"目录不存在: {根目录}")
        关键词小写 = str(关键词).lower()
        命中 = []
        for 路径 in 根.rglob("*"):
            try:
                if not 路径.is_file(): continue
                if 路径.suffix.lower() not in (".ipv", ".hsj", ".png", ".jpg", ".jpeg", ".psd"): continue
                if 关键词小写 in 路径.name.lower(): 命中.append(str(路径))
            except OSError:
                continue
        命中.sort(key=lambda s: (Path(s).name.lower(), s.lower()))
        if 上限 and 上限 > 0: 命中 = 命中[:上限]
        结果列表 = [[路径, 0.0] for 路径 in 命中]
        Self.导出结果PNG(结果列表)
        return 结果列表