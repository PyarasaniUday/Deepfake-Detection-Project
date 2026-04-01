from fastapi import FastAPI, UploadFile
import shutil
from predict import predict_image #type:ignore

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)

    return {"prediction": result}