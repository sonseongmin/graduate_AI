# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import tempfile, os
from processor import analyze_video_run  # ✅ run 버전 불러오기
import mediapipe as mp

app = FastAPI(title="Fitness AI Server", version="0.1.0")

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
async def analyze(exercise: str = Form(...), file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # ✅ Mediapipe Pose 객체를 열어서 같이 전달
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            raw_result = analyze_video_run(tmp_path, exercise, pose)

        if "error" in raw_result:
            raise HTTPException(status_code=400, detail=raw_result["error"])

        return AnalyzeResult(
            exercise_name=raw_result.get("exercise_name", exercise),
            count_total=raw_result.get("count_total", 0),
            count_incorrect=raw_result.get("count_incorrect", 0),
            feedback=raw_result.get("feedback", []),
            elapsed_time=raw_result.get("elapsed_time", 0.0),
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
