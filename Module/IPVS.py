from TranslatorLib import (io, np, asyncio, Path, struct, Image, faiss, GPU_ACC,
                           ThreadPoolExecutor, TranslatorPersistence, Config)

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
        self.layer_names = self._parse_layer_names()
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

    def _parse_layer_names(self):
        tail = self._data[max(0, len(self._data) - 15000):]
        names = []
        seen = set()
        i = 0
        while i < len(tail) - 40:
            if tail[i:i + 4] == b"\x03\x00\x04\x02":
                start = i - 4
                if start >= 0 and start + 0x60 <= len(tail):
                    rec = tail[start:]
                    ln = rec[0x22]
                    if 1 <= ln <= 40 and 0x23 + ln <= len(rec):
                        try:
                            name = rec[0x23:0x23 + ln].decode("utf-8")
                            if name.isprintable() and len(name) >= 2 and name not in seen:
                                seen.add(name)
                                names.append(name)
                        except Exception:
                            pass
                i += 1
            else:
                i += 1
        return names
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
            "<IbisPaintIPV canvas=%s version=%s device=%s layers=%d pngs=%d>"
            % (self.canvas_size, self.version, self.device,
               len(self.layer_names), len(self._png_offsets))
        )

class IPVS:
    def __init__(Self, App: Config):
        Self.Config       = App.Config
        Self.日志         = App.日志
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
                支持扩展 = ("*.ipv", "*.IPV", "*.png", "*.PNG",
                          "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.psd", "*.PSD")
                图像路径列表 = list({str(x) for ext in 支持扩展 for x in p.glob("**/" + ext)})
            else:
                raise FileNotFoundError(f"路径不存在: {path}")
        else:
            raise TypeError("path 必须是 str 或 str 列表")
        图像列表 = []
        def 读取图像(图像路径):
            try:
                后缀 = Path(图像路径).suffix.lower()
                if 后缀 == ".ipv":
                    图像 = IbisPaintIPV.from_path(图像路径).to_pil(prefer_final=True)
                else:
                    图像 = Image.open(图像路径).convert("RGBA")
                return [图像, 图像路径, ""]
            except Exception as e:
                Self.日志("log.core.file.ipvs.read.error", item=图像路径, info_level=1)
                return None
        with ThreadPoolExecutor(max_workers=30) as 执行器:
            for 结果 in Self.tqdm(执行器.map(读取图像, 图像路径列表),
                                  total=len(图像路径列表), desc="转换图像"):
                if 结果 is not None:
                    图像列表.append(结果)
        _, 文本列表 = asyncio.run(TranslatorPersistence.参考词预处理(Self, 图像=True))
        文本列表 = 文本列表 or []
        路径集 = {index[1] for index in 文本列表 if isinstance(index, (list, tuple)) and len(index) > 1}
        生成列表 = [item for item in 图像列表 if item[1] not in 路径集]
        if 生成列表:
            asyncio.run(TranslatorPersistence.参考词预处理(Self, 生成列表, 图像=True))
        else: return
    def 使用图像搜索(Self, path, K: int = None) -> list:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"路径不存在: {path}")
        if p.suffix.lower() == ".ipv":
            图像 = IbisPaintIPV.from_path(str(p)).to_pil(prefer_final=True)
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
        Self.Quantization.PCA应用懒加载(查询向量, 向量文件)
        Self.Quantization.TT应用懒加载(查询向量, 向量文件)
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
        Self.Quantization.PCA应用懒加载(查询向量, 向量文件)
        Self.Quantization.TT应用懒加载(查询向量, 向量文件)
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
        return 结果列表
    
测试 = True
if __name__ == "__main__":
    配置字典 = {
        "EMB_API_URL": "http://127.0.0.1:11434/v1/embeddings", # API仅支持llama.cpp 这个端口不是Ollama 这是我测试时llama-server挂到11434的
        "EMB_MODEL": "Qwen3-VL-Embedding-8B",
        "EMB_IMG_MAX_BATCH": 1,
        "VEC_CACHE_NAME": "IPVSC",
        "VEC_FILE_NAME": "IPVS",
        "VEC_QUANTIZATION": "Float16_Max"
    }
    配置管理器 = Config(配置字典)
    管理器 = IPVS(配置管理器)
    管理器.添加图像到数据库(r"C:\Users\FengMang\Desktop\模板\1")
    管理器.添加图像到数据库(r"C:\Users\FengMang\Desktop\模板\2")
    #print(管理器.使用PNG搜索(r"C:\Users\FengMang\Desktop\测试.png"))