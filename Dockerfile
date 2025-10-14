# ==================================
# 1️⃣ Base image (Python 3.10)
# ==================================
FROM python:3.10-slim

# ==================================
# 2️⃣ Working directory
# ==================================
WORKDIR /app

# ==================================
# 3️⃣ System dependencies
# ==================================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ==================================
# 4️⃣ Python dependencies
# ==================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


# ==================================
# 5️⃣ Copy source code
# ==================================
COPY TransRAC_main/models/best_classifier_hybrid.pt /app/models/best_classifier_hybrid.pt
COPY . .

# ==================================
# 6️⃣ Expose port & run FastAPI
# ==================================
EXPOSE 8001
CMD ["uvicorn", "TransRAC_main.tools.last.app:app", "--host", "0.0.0.0", "--port", "8001"]
