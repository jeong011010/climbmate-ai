FROM python:3.10-slim

# 🚀 최적화된 패키지 설치 (필수만)
RUN apt-get update && apt-get install -y \
    git curl \
    libgl1 libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

WORKDIR /app

# 🚀 파이썬 라이브러리 설치 (캐시 없이 설치)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# 프로젝트 전체 복사
COPY . .

# 데이터베이스 디렉토리 생성
RUN mkdir -p /app/backend/models /app/backend/data

# 🚀 메모리 최적화 환경변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# FastAPI 백엔드 포트
EXPOSE 8000

# 기본 실행: FastAPI 백엔드
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]