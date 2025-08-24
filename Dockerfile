# 베이스 이미지 (Python 3.10 이상 권장)
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 컨테이너 포트
EXPOSE 8001

# FastAPI 실행 명령
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]