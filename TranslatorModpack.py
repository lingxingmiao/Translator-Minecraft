from TranslatorLib import (json, eb, Path, ThreadPoolExecutor, requests, time, uuid, hashlib, shutil, Config)

class Modpack: # 当前版本工作内容: 整合包清单解析 自动识别下载CurseForge与Modrinth模组 下载文件LFU缓存
    def __init__(Self, App: Config):
        Self.Config = App.Config
        Self.Module = App.Module
        Self.Locale = App.Locale
        Self.日志 = App.日志
        Self.Lang = App.Lang
        Self.tqdm = App.RichTqdm
        Self.会话 = requests.Session()
        Self.会话.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 TranslatorMinecraft",
        })
        # ↓镜像站配置 解决中国大陆无法访问CurseForge/ModrinthCDN的问题 参考HMCL BMCLAPIDownloadProvider
        Self.镜像映射 = {
            "https://api.curseforge.com": "https://mod.mcimirror.top/curseforge",
            "https://edge.forgecdn.net": "https://mod.mcimirror.top",
            "https://api.modrinth.com": "https://mod.mcimirror.top/modrinth",
            "https://cdn.modrinth.com": "https://mod.mcimirror.top",
        }
        # ↓下载文件LFU缓存: {url_hash: [缓存文件名, 访问频率, 最后访问代数, 文件大小]}
        Self.下载缓存元数据 = {}
        Self.下载缓存轮次 = 0
        Self.下载缓存锁 = __import__("threading").Lock()
        Self._加载下载缓存()
    def _加载下载缓存(Self):
        """加载下载缓存元数据"""
        try:
            缓存路径 = Path(f"{Self.Config.MODPACK_DOWNLOAD_CACHE_PATH}/{Self.Config.MODPACK_DOWNLOAD_CACHE_NAME}.json")
            if 缓存路径.is_file():
                with open(缓存路径, "r", encoding="utf-8") as f:
                    数据 = json.load(f)
                Self.下载缓存元数据 = {k: list(v) for k, v in 数据.get("cache", {}).items()}
                Self.下载缓存轮次 = int(数据.get("round", 0))
        except Exception:
            Self.日志("log.modpack.cache.load.error", e=eb.format_exc(), info_level=1)
    def _保存下载缓存(Self):
        """保存下载缓存元数据"""
        try:
            缓存目录 = Path(Self.Config.MODPACK_DOWNLOAD_CACHE_PATH)
            缓存目录.mkdir(parents=True, exist_ok=True)
            缓存路径 = 缓存目录 / f"{Self.Config.MODPACK_DOWNLOAD_CACHE_NAME}.json"
            with open(缓存路径, "w", encoding="utf-8") as f:
                json.dump({"round": Self.下载缓存轮次, "cache": Self.下载缓存元数据}, f, ensure_ascii=False, indent=2)
        except Exception:
            Self.日志("log.modpack.cache.save.error", e=eb.format_exc(), info_level=1)
    def _下载缓存淘汰(Self):
        """LFU缓存淘汰: 频率/代数差评分淘汰 + 超上限淘汰最低分"""
        if not Self.Config.MODPACK_DOWNLOAD_CACHE:
            return
        缓存目录 = Path(Self.Config.MODPACK_DOWNLOAD_CACHE_PATH)
        宽限期 = Self.Config.MODPACK_DOWNLOAD_CACHE_DECAY_GRACE
        衰减阈值 = Self.Config.MODPACK_DOWNLOAD_CACHE_DECAY_THRESHOLD
        with Self.下载缓存锁:
            Self.下载缓存轮次 += 1
            当前轮次 = Self.下载缓存轮次
            过期条目 = []
            for 哈希 in list(Self.下载缓存元数据.keys()):
                频率 = Self.下载缓存元数据[哈希][1]
                代数差 = 当前轮次 - Self.下载缓存元数据[哈希][2]
                if 代数差 <= 宽限期: # 宽限期内不淘汰
                    continue
                if 频率 / (代数差 + 1) < 衰减阈值:
                    过期条目.append(哈希)
            for 哈希 in 过期条目:
                Self._删除缓存文件(哈希)
            # ↓超上限按评分淘汰最低分
            上限 = Self.Config.MODPACK_DOWNLOAD_CACHE_MAX_SIZE
            if len(Self.下载缓存元数据) > 上限:
                def _分数(哈希):
                    频率 = Self.下载缓存元数据[哈希][1]
                    代数差 = 当前轮次 - Self.下载缓存元数据[哈希][2] + 1
                    return 频率 / 代数差
                排序列表 = sorted(Self.下载缓存元数据.keys(), key=_分数)
                淘汰数 = len(Self.下载缓存元数据) - 上限
                for 哈希 in 排序列表[:淘汰数]:
                    Self._删除缓存文件(哈希)
                Self.日志("log.modpack.cache.evict", evicted=淘汰数, remain=len(Self.下载缓存元数据), info_level=0)
            elif 过期条目:
                Self.日志("log.modpack.cache.evict", evicted=len(过期条目), remain=len(Self.下载缓存元数据), info_level=0)
            Self._保存下载缓存()
    def _删除缓存文件(Self, 哈希):
        """删除缓存文件并移除元数据"""
        缓存目录 = Path(Self.Config.MODPACK_DOWNLOAD_CACHE_PATH)
        文件名 = Self.下载缓存元数据.get(哈希, [None])[0]
        if 文件名:
            try:
                (缓存目录 / 文件名).unlink(missing_ok=True)
            except Exception:
                pass
        Self.下载缓存元数据.pop(哈希, None)
    def _下载缓存查询(Self, url: str) -> Path:
        """查询下载缓存 命中返回缓存文件路径并更新频率 未命中返回None"""
        if not Self.Config.MODPACK_DOWNLOAD_CACHE:
            return None
        哈希 = hashlib.sha256(url.encode("utf-8")).hexdigest()
        with Self.下载缓存锁:
            条目 = Self.下载缓存元数据.get(哈希)
            if 条目:
                缓存文件 = Path(f"{Self.Config.MODPACK_DOWNLOAD_CACHE_PATH}/{条目[0]}")
                if 缓存文件.is_file(): # 命中
                    条目[1] += 1            # 频率+1
                    条目[2] = Self.下载缓存轮次 # 最后代数
                    Self.日志("log.modpack.cache.hit", hash=哈希[:12], file=条目[0], info_level=0)
                    return 缓存文件
                else: # 文件丢失 移除无效元数据
                    Self.下载缓存元数据.pop(哈希, None)
        return None
    def _下载缓存写入(Self, url: str, 源文件: Path, 文件大小: int):
        """将下载完成的文件写入缓存"""
        if not Self.Config.MODPACK_DOWNLOAD_CACHE:
            return
        try:
            缓存目录 = Path(Self.Config.MODPACK_DOWNLOAD_CACHE_PATH)
            缓存目录.mkdir(parents=True, exist_ok=True)
            哈希 = hashlib.sha256(url.encode("utf-8")).hexdigest()
            缓存文件名 = f"{哈希}.jar"
            目标路径 = 缓存目录 / 缓存文件名
            if 目标路径.exists(): # 已存在 直接更新元数据
                目标路径.unlink(missing_ok=True)
            shutil.copy2(源文件, 目标路径)
            with Self.下载缓存锁:
                Self.下载缓存元数据[哈希] = [缓存文件名, 1, Self.下载缓存轮次, 文件大小]
                Self._保存下载缓存()
            Self.日志("log.modpack.cache.write", hash=哈希[:12], file=缓存文件名, size=文件大小, info_level=0)
        except Exception:
            Self.日志("log.modpack.cache.write.error", e=eb.format_exc(), info_level=1)
    def 镜像替换(Self, url: str) -> str:
        """替换URL为镜像地址(启用镜像且配置镜像时)"""
        if not Self.Config.MODPACK_USE_MIRROR:
            return url
        for 原地址, 镜像地址 in Self.镜像映射.items():
            if url.startswith(原地址):
                return url.replace(原地址, 镜像地址)
        return url
    #====================================================================================================↓整合包清单解析↓====================================================================================================#
    def 解析整合包清单(Self, 整合包根目录: str) -> list:
        """识别整合包模组清单 返回统一格式列表:
        CurseForge: {"来源": "curseforge", "projectID": str, "fileID": str, "required": bool, "path": str, "fileName": str, "url": str}
        Modrinth:   {"来源": "modrinth", "path": str, "downloads": list, "hashes": dict, "fileSize": int, "env": dict}
        """
        整合包根目录 = Path(整合包根目录)
        清单列表 = []
        manifest路径 = 整合包根目录 / "manifest.json"
        if manifest路径.is_file():
            try:
                with open(manifest路径, "r", encoding="utf-8") as f:
                    数据 = json.load(f)
                for 文件 in 数据.get("files", []):
                    清单列表.append({
                        "来源": "curseforge",
                        "projectID": str(文件.get("projectID", "")),
                        "fileID": str(文件.get("fileID", "")),
                        "required": 文件.get("required", True),
                        "path": 文件.get("path", ""),
                        "fileName": 文件.get("fileName", "") or "",
                        "url": 文件.get("url", "") or "",
                    })
                return 清单列表
            except Exception:
                Self.日志("log.modpack.manifest.parse.error", file=str(manifest路径), e=eb.format_exc(), info_level=2)
        modrinth索引路径 = 整合包根目录 / "modrinth.index.json"
        if modrinth索引路径.is_file():
            try:
                with open(modrinth索引路径, "r", encoding="utf-8") as f:
                    数据 = json.load(f)
                for 文件 in 数据.get("files", []):
                    清单列表.append({
                        "来源": "modrinth",
                        "path": 文件.get("path", ""),
                        "downloads": 文件.get("downloads", []) or [],
                        "hashes": 文件.get("hashes", {}) or {},
                        "fileSize": 文件.get("fileSize", 0),
                        "env": 文件.get("env", {}) or {},
                    })
                return 清单列表
            except Exception:
                Self.日志("log.modpack.manifest.parse.error", file=str(modrinth索引路径), e=eb.format_exc(), info_level=2)
        return 清单列表
    #====================================================================================================↑整合包清单解析↑====================================================================================================#
    #====================================================================================================↓下载链接解析↓====================================================================================================#
    def CurseForge获取文件信息(Self, projectID: str, fileID: str) -> dict:
        """通过CurseForge API获取模组文件信息
        自动降级顺序: 官方API(需Key) -> 镜像站(MCIM) -> 网页公开接口 -> CDN直链兜底
        单个来源请求失败(网络/5xx)时自动尝试下一来源, 全部失败抛出最后一次异常"""
        APIKey = Self.Config.MODPACK_CURSEFORGE_API_KEY
        来源列表 = []
        if APIKey: # ↓1. 官方API
            来源列表.append(("官方API", "https://api.curseforge.com/v1/mods/{projectID}/files/{fileID}", {"x-api-key": APIKey, "Accept": "application/json"}))
        if Self.Config.MODPACK_USE_MIRROR: # ↓2. 镜像站(MCIM)
            来源列表.append(("镜像站", "https://api.curseforge.com/v1/mods/{projectID}/files/{fileID}", {"Accept": "application/json"}))
        # ↓3. 网页公开接口(直连 不走镜像)
        来源列表.append(("网页公开接口", "https://www.curseforge.com/api/v1/mods/{projectID}/files/{fileID}", {"Accept": "application/json"}))
        最后异常 = None
        for 来源名称, 模板地址, 请求头 in 来源列表:
            try:
                请求地址 = 模板地址.format(projectID=projectID, fileID=fileID)
                请求地址 = Self.镜像替换(请求地址)
                # ↓5xx/网络错误重试(限时退避)
                for 尝试 in range(Self.Config.MODPACK_DOWNLOAD_RETRY):
                    try:
                        响应 = Self.会话.get(请求地址, headers=请求头, timeout=Self.Config.MODPACK_DOWNLOAD_TIMEOUT)
                        if 响应.status_code < 500:
                            响应.raise_for_status()
                            break
                        最后异常 = requests.exceptions.HTTPError(f"{响应.status_code} Server Error: {响应.reason} for url: {请求地址}")
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                        最后异常 = eb.format_exc()
                    if 尝试 + 1 < Self.Config.MODPACK_DOWNLOAD_RETRY:
                        time.sleep(Self.Config.MODPACK_DOWNLOAD_RETRY_TIME)
                else:
                    Self.日志("log.modpack.curseforge.fallback", source=来源名称, projectID=projectID, fileID=fileID, e=str(最后异常), info_level=1)
                    continue
                数据 = 响应.json()
                # ↓兼容多种响应结构: {"data": {...}} / {...} / 列表
                文件数据 = {}
                if isinstance(数据, dict):
                    文件数据 = 数据.get("data", 数据) or {}
                    if isinstance(文件数据, list) and 文件数据:
                        文件数据 = 文件数据[0]
                elif isinstance(数据, list) and 数据:
                    文件数据 = 数据[0]
                文件数据 = 文件数据 or {}
                if not 文件数据: # ↓空响应视为失败 继续下一来源
                    Self.日志("log.modpack.curseforge.fallback", source=来源名称, projectID=projectID, fileID=fileID, e="响应为空", info_level=1)
                    continue
                return {
                    "id": str(文件数据.get("id", fileID)),
                    "fileName": 文件数据.get("fileName", "") or "",
                    "downloadUrl": 文件数据.get("downloadUrl", "") or "",
                }
            except Exception:
                Self.日志("log.modpack.curseforge.fallback", source=来源名称, projectID=projectID, fileID=fileID, e=eb.format_exc(), info_level=1)
        最后异常 = 最后异常 or eb.format_exc()
        raise RuntimeError(Self.Lang("log.modpack.curseforge.all.failed", projectID=projectID, fileID=fileID, e=最后异常))
    def CurseForge构造CDN链接(Self, fileID: str, fileName: str) -> str:
        """CurseForge禁止分发时构造edge.forgecdn.net CDN直链"""
        try:
            文件ID数字 = int(fileID)
            return Self.镜像替换(f"https://edge.forgecdn.net/files/{文件ID数字 // 1000}/{文件ID数字 % 1000}/{fileName}")
        except Exception:
            return ""
    def Modrinth哈希查询(Self, 哈希值: str, 算法: str = "sha1") -> str:
        """通过Modrinth API按文件哈希查询下载链接"""
        请求地址 = f"https://api.modrinth.com/v2/version_file/{哈希值}?algorithm={算法}"
        请求地址 = Self.镜像替换(请求地址)
        响应 = Self.会话.get(请求地址, timeout=Self.Config.MODPACK_DOWNLOAD_TIMEOUT)
        响应.raise_for_status()
        数据 = 响应.json()
        for 文件 in 数据.get("files", []):
            if 文件.get("hashes", {}).get(算法) == 哈希值:
                return Self.镜像替换(文件["url"])
        if 数据.get("files"):
            return Self.镜像替换(数据["files"][0]["url"])
        raise ValueError(Self.Lang("log.modpack.resolve.hash.not.found", hash=哈希值))
    def 获取下载链接(Self, 条目: dict) -> str:
        """根据清单条目获取模组文件下载链接"""
        if 条目["来源"] == "modrinth":
            if 条目.get("downloads"):
                return Self.镜像替换(条目["downloads"][0])
            sha1 = 条目.get("hashes", {}).get("sha1")
            if sha1:
                return Self.Modrinth哈希查询(sha1, "sha1")
            sha512 = 条目.get("hashes", {}).get("sha512")
            if sha512:
                return Self.Modrinth哈希查询(sha512, "sha512")
            raise ValueError(Self.Lang("log.modpack.entry.missing.download", name=条目.get("path", "")))
        elif 条目["来源"] == "curseforge":
            if 条目.get("url"):
                return 条目["url"]
            文件名 = 条目.get("fileName", "")
            try:
                文件信息 = Self.CurseForge获取文件信息(条目["projectID"], 条目["fileID"])
                文件名 = 文件信息["fileName"] or 文件名
                下载链接 = 文件信息["downloadUrl"]
                if not 下载链接 and 文件名: # 禁止分发 尝试CDN直链
                    下载链接 = Self.CurseForge构造CDN链接(条目["fileID"], 文件名)
                if 下载链接:
                    return 下载链接
            except Exception as e:
                Self.日志("log.modpack.curseforge.cdn.fallback", projectID=条目["projectID"], fileID=条目["fileID"], e=str(e), info_level=1)
            # ↓API全部失败/无链接 用清单里的文件名构造CDN直链兜底(需要文件名)
            if 文件名:
                return Self.CurseForge构造CDN链接(条目["fileID"], 文件名)
            raise ValueError(Self.Lang("log.modpack.entry.missing.download", name=条目.get("path", "")))
        raise ValueError(Self.Lang("log.modpack.entry.unknown.source", source=条目.get("来源", "")))
    #====================================================================================================↑下载链接解析↑====================================================================================================#
    #====================================================================================================↓模组下载↓====================================================================================================#
    def 下载文件(Self, url: str, 目标路径: Path) -> bool:
        """下载文件到目标路径 优先LFU缓存命中 未命中则流式下载并写入缓存 失败自动重试"""
        目标路径 = Path(目标路径)
        目标路径.parent.mkdir(parents=True, exist_ok=True)
        url = Self.镜像替换(url)
        # ↓先检查LFU缓存
        缓存文件 = Self._下载缓存查询(url)
        if 缓存文件 is not None:
            try:
                shutil.copy2(缓存文件, 目标路径)
                return True
            except Exception:
                Self.日志("log.modpack.cache.copy.error", e=eb.format_exc(), info_level=1)
        # ↓未命中 正常下载
        最大重试 = max(1, Self.Config.MODPACK_DOWNLOAD_RETRY)
        for 尝试 in range(最大重试):
            try:
                with Self.会话.get(url, stream=True, timeout=Self.Config.MODPACK_DOWNLOAD_TIMEOUT) as 响应:
                    响应.raise_for_status()
                    文件大小 = 0
                    with open(目标路径, "wb") as f:
                        for 块 in 响应.iter_content(chunk_size=1024 * 256):
                            if 块:
                                f.write(块)
                                文件大小 += len(块)
                # ↓下载完成写入缓存
                Self._下载缓存写入(url, 目标路径, 文件大小)
                return True
            except Exception:
                if 尝试 + 1 < 最大重试:
                    time.sleep(Self.Config.MODPACK_DOWNLOAD_RETRY_TIME)
                else:
                    raise
        return False
    def 下载缺失模组(Self, 整合包根目录: str, 模组根目录: str) -> int:
        """根据整合包清单自动下载缺失模组到模组根目录 返回下载成功数量"""
        整合包根目录 = Path(整合包根目录)
        模组根目录 = Path(模组根目录)
        清单列表 = Self.解析整合包清单(整合包根目录)
        if not 清单列表:
            Self.日志("log.modpack.manifest.not.found", path=整合包根目录, info_level=1)
            return 0
        来源统计 = {}
        for 条目 in 清单列表:
            来源统计[条目["来源"]] = 来源统计.get(条目["来源"], 0) + 1
        Self.日志("log.modpack.manifest.parsed", curseforge=来源统计.get("curseforge", 0), modrinth=来源统计.get("modrinth", 0), count=len(清单列表), info_level=0)
        # ↓构建下载任务列表
        任务列表 = []
        模组根解析 = 模组根目录.resolve()
        for 条目 in 清单列表:
            if not 条目.get("required", True): # 可选模组默认跳过
                continue
            if 条目["来源"] == "modrinth" and 条目.get("env", {}).get("client", "required") == "unsupported":
                continue # 客户端不支持的模组跳过
            try:
                下载链接 = Self.获取下载链接(条目)
            except Exception:
                Self.日志("log.modpack.resolve.url.error", name=条目.get("path") or f"{条目.get('projectID', '')}/{条目.get('fileID', '')}", e=eb.format_exc(), info_level=2)
                continue
            if not 下载链接:
                Self.日志("log.modpack.resolve.url.error", name=条目.get("path") or f"{条目.get('projectID', '')}/{条目.get('fileID', '')}", e="downloadUrl为空", info_level=2)
                continue
            # ↓计算目标相对路径
            if 条目["来源"] == "modrinth" and 条目.get("path"):
                相对路径 = 条目["path"].replace("\\", "/").lstrip("/")
            else:
                文件名 = Path(下载链接.split("?")[0]).name
                if not 文件名 or not 文件名.lower().endswith((".jar", ".zip")):
                    文件名 = f"{条目.get('projectID', 'mod')}-{条目.get('fileID', '')}.jar"
                相对路径 = f"mods/{文件名}"
            # ↓路径安全校验 防止路径穿越
            目标路径 = (模组根目录 / 相对路径).resolve()
            try:
                if not str(目标路径).startswith(str(模组根解析)):
                    Self.日志("log.modpack.download.skip", name=相对路径, info_level=2)
                    continue
            except Exception:
                pass
            if 目标路径.exists(): # 已存在跳过
                Self.日志("log.modpack.download.skip", name=相对路径, info_level=1)
                continue
            任务列表.append([目标路径, 下载链接, 相对路径])
        if not 任务列表:
            Self.日志("log.modpack.download.all.succeed", count=0, info_level=0)
            return 0
        # ↓并发下载
        成功数 = 0
        def 下载单个(任务):
            目标路径, 下载链接, 相对路径 = 任务
            try:
                Self.日志("log.modpack.download.start", name=相对路径, info_level=0)
                Self.下载文件(下载链接, 目标路径)
                Self.日志("log.modpack.download.succeed", name=相对路径, path=目标路径, info_level=0)
                return True
            except Exception:
                Self.日志("log.modpack.download.error", name=相对路径, e=eb.format_exc(), info_level=2)
                return False
        with ThreadPoolExecutor(max_workers=Self.Config.MODPACK_DOWNLOAD_CONCURRENT) as 执行器:
            for 结果 in Self.tqdm(执行器.map(下载单个, 任务列表), total=len(任务列表), desc="tqdm.modpack.download"):
                if 结果:
                    成功数 += 1
        Self._下载缓存淘汰() # 下载完成后执行LFU淘汰
        Self.日志("log.modpack.download.all.succeed", count=成功数, info_level=0)
        return 成功数
    #====================================================================================================↑模组下载↑====================================================================================================#
