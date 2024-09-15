from fastapi import FastAPI, File, UploadFile

from utils.data_load_utils import get_ids
app = FastAPI()
 
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    uuidf = get_ids()
    filename = f"/gemini/code/temp/images/{uuidf}-{file.filename}"
    with open(filename, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    return {"filename": f"http://direct.virtaicloud.com:48344/{uuidf}-{file.filename}"}


from fastapi import File, UploadFile
from typing import List

@app.post("/upload/files/")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        contents = await file.read()
        # 这里可以保存文件到服务器，或者进行其他处理
        uploaded_files.append({"filename": file.filename, "size": len(contents)})
    return {"message": "文件上传成功", "files": uploaded_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9002)