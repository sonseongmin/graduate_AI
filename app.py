# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import tempfile, os
from processor import analyze_video

app = FastAPI(title="Fitness AI Server", version="0.1.0")

class AnalyzeResult(BaseModel):
    exercise_name: str
    count_total: int
    count_incorrect: int
    feedback: str
    elapsed_time: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(exercise: str = Form(...), file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = analyze_video(tmp_path, exercise)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
