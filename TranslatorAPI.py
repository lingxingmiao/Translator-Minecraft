from TranslatorLib import APIConfig, Dict, uvicorn, Path as pt, time, json
from TranslatorCore import Translator
#需要安装↓
from fastapi import FastAPI, UploadFile, HTTPException, status, Depends, Security, Form
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

FastAPI = FastAPI(title="TranslationMinecraft")

def 设置时间():
    时间 = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + f"{int((time.time() % 1) * 10000):04d}"
    APIConfig["server"]['LOGS_FILE_NAME'] = f"logs-{时间}"
安全 = HTTPBearer(auto_error=False)
async def 验证Key(凭证: HTTPAuthorizationCredentials = Security(安全)):
    if 凭证 is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    密钥 = 凭证.credentials
    with open("api-config.cfg", 'r', encoding='utf-8') as f:
        data = json.load(f)
    有效密钥 = data.get("keys", [])
    if not 有效密钥:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: No valid API keys found")
    if 密钥 not in 有效密钥:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    return 密钥
@FastAPI.post("/translate", dependencies=[Depends(验证Key)])
async def translate(
    file0: UploadFile, file_name0: str = Form(...),
    file1: UploadFile = None, file_name1: str = Form(None),
    all_mode: bool = Form(False), export_inspection: bool = Form(False)) -> Dict:
    设置时间()
    Translator实例 = Translator(APIConfig["server"])
    缓存路径0 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    缓存路径1 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    pt(缓存路径0).mkdir(parents=True, exist_ok=True)
    pt(缓存路径1).mkdir(parents=True, exist_ok=True)
    with open(f"{缓存路径0}/{file_name0}", "wb") as f:
        f.write(await file0.read())
    file0 = f"{缓存路径0}/{file_name0}"
    if not file1 == None:
        with open(f"{缓存路径1}/{file_name1}", "wb") as f:
            f.write(await file1.read())
        file1 = f"{缓存路径1}/{file_name1}"
    翻译结果 = Translator实例.翻译通用文件(file0=file0, file1=file1, all_mode=all_mode, export_inspection=export_inspection)
    if 翻译结果 is None:
        raise HTTPException(status_code=500, detail=Translator实例.Lang("log.api.translate.error.none"))
    return FileResponse(
        path=翻译结果,
        filename=翻译结果.name,
        media_type="application/octet-stream"
    )
@FastAPI.post("/separatelangupdate", dependencies=[Depends(验证Key)])
async def SeparateLangUpdate(
    file0: UploadFile, file_name0: str = Form(...),
    file1: UploadFile = None, file_name1: str = Form(None),
    Mode: str = Form("none")):
    设置时间()
    Translator实例 = Translator(APIConfig["server"])
    缓存路径0 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    缓存路径1 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    缓存路径2 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    pt(缓存路径0).mkdir(parents=True, exist_ok=True)
    pt(缓存路径1).mkdir(parents=True, exist_ok=True)
    pt(缓存路径2).mkdir(parents=True, exist_ok=True)
    with open(f"{缓存路径0}/{file_name0}", "wb") as f:
        f.write(await file0.read())
    file0 = f"{缓存路径0}/{file_name0}"
    if not file1 == None:
        with open(f"{缓存路径1}/{file_name1}", "wb") as f:
            f.write(await file1.read())
        file1 = f"{缓存路径1}/{file_name1}"
    处理结果 = Translator实例.分离语言文件更新(file0=file0, file1=file1, output_path=缓存路径2, mode=Mode)
    if 处理结果 is None:
        raise HTTPException(status_code=500, detail=Translator实例.Lang("log.api.langupdate.separate.error.none"))
    return FileResponse(
        path=处理结果,
        filename=处理结果.name,
        media_type="application/octet-stream"
    )
@FastAPI.post("/mergelangupdate", dependencies=[Depends(验证Key)])
async def MergeLangUpdate(
    file0: UploadFile, notlang_file: UploadFile,
    file_name0: str = Form(...), nolang_file_name: str = Form(...),
    file1: UploadFile = None, file_name1: str = Form(None)):
    设置时间()
    Translator实例 = Translator(APIConfig["server"])
    缓存路径0 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    缓存路径1 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    缓存路径2 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    缓存路径3 = f"{Translator实例.Config.PATH_CACHE}/{Translator实例.Module.随机16进制字符串(4)}/"
    pt(缓存路径0).mkdir(parents=True, exist_ok=True)
    pt(缓存路径1).mkdir(parents=True, exist_ok=True)
    pt(缓存路径2).mkdir(parents=True, exist_ok=True)
    pt(缓存路径3).mkdir(parents=True, exist_ok=True)
    with open(f"{缓存路径0}/{file_name0}", "wb") as f:
        f.write(await file0.read())
    file0 = f"{缓存路径0}/{file_name0}"
    with open(f"{缓存路径1}/{nolang_file_name}", "wb") as f:
        f.write(await notlang_file.read())
    notlang_file = f"{缓存路径1}/{nolang_file_name}"
    if not file1 == None:
        with open(f"{缓存路径2}/{file_name1}", "wb") as f:
            f.write(await file1.read())
        file1 = f"{缓存路径2}/{file_name1}"
    处理结果 = Translator实例.合并语言文件更新(file0=file0, file1=file1, output_path=缓存路径3, notlang_file=notlang_file)
    if 处理结果 is None:
        raise HTTPException(status_code=500, detail=Translator实例.Lang("log.api.langupdate.merge.error.none"))
    return FileResponse(
        path=处理结果,
        filename=处理结果.name,
        media_type="application/octet-stream"
    )
    
if __name__ == "__main__":
    # ANSI Shadow
    蓝色 = '\033[34m'
    绿色 = '\033[92m'
    默认色 = '\033[0m'
    print(f"""
    {蓝色}
████████╗██████╗  █████╗ ███╗   ██╗███████╗██╗      █████╗ ███╗   ███╗ ██████╗
╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║     ██╔══██╗████╗ ████║██╔════╝
   ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ███████║██╔████╔██║██║     
   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██║██║╚██╔╝██║██║     
   ██║   ██║  ██║██║  ██║██║ ╚████║███████║███████╗██║  ██║██║ ╚═╝ ██║╚██████╗
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝                 
    {默认色}
        {绿色}Server:{默认色} TranslatorMinecraft API Version 1.4 Beta 2
========================================================================================
        """)
    uvicorn.run(FastAPI, host=APIConfig["api"]["host"], port=APIConfig["api"]["port"])