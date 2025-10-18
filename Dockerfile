FROM python:3.10-slim

# 필수 패키지 + 빌드 툴 설치
RUN apt-get update && apt-get install -y \
    git curl wget unzip \
    libgl1 libglib2.0-0 libglib2.0-dev \
    build-essential gcc g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 라이브러리 설치 (캐시 없이 설치)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 프로젝트 전체 복사
COPY . .

# 데이터베이스 디렉토리 생성
RUN mkdir -p /app/backend/models /app/backend/data

# FastAPI 백엔드 포트
EXPOSE 8000

# 기본 실행: FastAPI 백엔드
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]