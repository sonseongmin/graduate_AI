# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import tempfile, os
from processor import analyze_video_init, analyze_video_run

app = FastAPI(title="Fitness AI Server", version="0.1.0")
pose_model = analyze_video_init()
class AnalyzeResult(BaseModel):
    exercise_name: str
    count_total: int
    count_incorrect: int
    feedback: list[str]
    elapsed_time: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(exercise: str = Form(...), category: str = Form(None), file: UploadFile = File(...)):
    exercise = exercise or category
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        raw_result = analyze_video_run(tmp_path, exercise)

        if "error" in raw_result:
            raise HTTPException(status_code=400, detail=raw_result["error"])

        # ✅ AnalyzeResult 스키마에 맞춰 변환
        result = analyze_video(tmp_path, exercise)
        return AnalyzeResult(
            exercise_name=exercise,
            count_total=raw_result.get("count_total", 0),
            count_incorrect=raw_result.get("count_incorrect", 0),
            feedback=raw_result.get("feedback", "분석 결과 없음"),
            elapsed_time=raw_result.get("elapsed_time", 0.0),
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
