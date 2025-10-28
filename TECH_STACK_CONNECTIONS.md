# ClimbMate 기술 스택 연결도 (피그마용)
## 아이콘은 직접, 연결만 표시

---

## 🎯 전체 기술 스택 목록

### 프론트엔드
- React 19.1.1
- Vite 7.1.7
- TailwindCSS 3.4.18
- Axios 1.12.2
- React Router DOM 7.9.4
- vite-plugin-pwa 1.1.0

### 백엔드 프레임워크
- FastAPI 0.119.0
- Uvicorn 0.37.0
- Python 3.13

### AI/ML 모델
- YOLOv8-seg (Ultralytics 8.2.103)
- PyTorch
- scikit-learn 1.3.2+
- ~~CLIP (제거됨)~~
- 규칙기반 HSV 분석
- ML 색상 분류 모델

### OpenAI
- OpenAI 1.0.0+ (GPT-4 Vision)

### 이미지 처리
- OpenCV 4.11.0+
- Pillow 10.4.0+
- NumPy 1.26.4+

### AWS 인프라
- AWS Lightsail (또는 EC2)

### 컨테이너 인프라
- Docker
- Docker Compose
- Nginx
- Let's Encrypt (Certbot)

### 비동기 처리
- Redis 7.x
- Celery 5.3.0

### 데이터베이스
- SQLite

---

## 🔗 연결 관계 (화살표)

### 사용자 → AWS 인프라

```
사용자
  ↓ (HTTPS)
climbmate.store 도메인
  ↓ (DNS)
AWS Lightsail / EC2
  ↓ (Port 443)
Nginx (Docker 내부)
```

---

### Nginx → 서비스들

```
Nginx
  ├→ React (Port 3000)
  └→ FastAPI (Port 8000)
```

---

### React → Backend

```
React
  ↓ (Axios HTTP 요청)
FastAPI
```

---

### FastAPI → AI 모델들

```
FastAPI
  ├→ YOLOv8-seg (홀드 감지)
  ├→ 규칙기반 HSV 분석 (색상 분류)
  ├→ ML 색상 분류 모델 (scikit-learn)
  ├→ 자체 ML 모델 (난이도 분석)
  └→ GPT-4 Vision (OpenAI API)
```

---

### FastAPI → 라이브러리들

```
FastAPI
  ├→ OpenCV (이미지 처리)
  ├→ Pillow (이미지 I/O)
  ├→ NumPy (수치 연산)
  └→ scikit-learn (ML)
```

---

### FastAPI → 비동기 처리

```
FastAPI
  ↓ (Port 6379)
Redis
  ↓
Celery Worker
```

---

### FastAPI ↔ 데이터베이스

```
FastAPI
  ↔ (양방향)
SQLite
```

---

### 피드백 → ML 학습

```
사용자 피드백
  ↓
FastAPI (/api/feedback)
  ↓
SQLite (저장)
  ↓
ml_trainer.py (학습)
  ↓
새로운 모델 (.pkl 파일)
  ↓
FastAPI (다음 분석에 적용)
```

---

## 📊 AI 처리 파이프라인 (순차적)

```
이미지 입력
  ↓
YOLOv8-seg
  ↓ (홀드 마스크)
규칙기반 HSV 분석
  ↓
ML 색상 모델 (scikit-learn)
  ↓ (색상 분류 완료)
[3단계 하이브리드 분석]
  ├→ 1순위: 자체 ML 모델
  ├→ 2순위: GPT-4 Vision
  └→ 3순위: 규칙기반 분석
  ↓ (최종 결과)
SQLite 저장
  ↓
React로 전달
```

---

## ☁️ AWS 인프라 구조

```
AWS Lightsail / EC2 인스턴스
  └─ Docker Compose
      ├─ Nginx (리버스 프록시)
      ├─ React (frontend 컨테이너)
      ├─ FastAPI (backend 컨테이너)
      ├─ Redis (작업 큐)
      ├─ Celery Worker (비동기 작업)
      └─ Certbot (SSL 인증서 갱신)
```

---

## 🔐 SSL/보안

```
Let's Encrypt (Certbot)
  ↓ (SSL 인증서)
Nginx
  ↓ (HTTPS)
React + FastAPI
```

---

## 📦 피그마 배치용 정리

### 가장 큰 박스: AWS Lightsail/EC2
안에 들어갈 것:
- Docker Compose (두 번째 큰 박스)

### 두 번째 큰 박스: Docker Compose (EC2 내부)
안에 들어갈 것들:
- Nginx + Let's Encrypt
- React
- FastAPI
- Redis
- Celery Worker

### 외부 박스들 (EC2 밖):
- 사용자
- 도메인 (climbmate.store)

### DB 위치:
- SQLite DB (Docker Volume, EC2 내부)

### FastAPI 내부 (작은 박스들):
- YOLOv8-seg
- 규칙기반 HSV
- ML 색상 모델
- 자체 ML 모델
- GPT-4 Vision
- OpenCV
- NumPy
- scikit-learn

---

## ⚡ 화살표 종류별 정리

### 일반 흐름 (→)
- 사용자 → 도메인
- 도메인 → Nginx
- Nginx → React
- Nginx → FastAPI
- React → FastAPI
- FastAPI → Redis
- Redis → Celery

### 양방향 (↔)
- FastAPI ↔ SQLite

### 내부 호출 (→)
- FastAPI → YOLOv8
- FastAPI → 색상 분류
- FastAPI → 난이도 분석
- FastAPI → OpenCV
- FastAPI → GPT-4 Vision

### 순차 처리 (↓)
- 이미지 ↓ YOLO ↓ 색상 ↓ 난이도 ↓ 결과

---

## 📝 포트 번호 표시

```
Nginx: 80, 443
React: 3000
FastAPI: 8000
Redis: 6379
```

---

## ✨ 강조할 부분

### CLIP → 규칙+ML 변경
```
[제거]
CLIP ViT-B/32

[추가]
규칙기반 HSV 분석
  +
ML 색상 모델 (scikit-learn)
```

### 하이브리드 난이도 분석
```
1순위: 자체 ML
2순위: GPT-4 Vision  
3순위: 규칙기반
```

---

## 🎨 배치 순서 (위→아래)

```
1. 사용자
2. climbmate.store 도메인
3. [AWS Lightsail/EC2 박스 시작 - 가장 큰 박스]
   3-1. [Docker Compose 박스 시작 - 두 번째 큰 박스]
        3-1-1. Nginx + Let's Encrypt
        3-1-2. React (왼쪽)
        3-1-3. FastAPI (오른쪽)
               - YOLOv8
               - 규칙+ML 색상
               - 난이도 분석
               - GPT-4
        3-1-4. Redis (중앙 하단)
        3-1-5. Celery (하단)
   3-2. [Docker Compose 박스 끝]
   3-3. SQLite DB (Docker Volume)
4. [AWS Lightsail/EC2 박스 끝]
```

---

## 📌 기술 스택 그룹핑

### 프론트엔드 그룹 (파랑)
- React
- Vite
- TailwindCSS
- Axios

### 백엔드 그룹 (초록)
- FastAPI
- Uvicorn
- Python

### AI 그룹 (주황)
- YOLOv8
- 규칙기반 HSV
- ML 색상 모델
- GPT-4 Vision
- scikit-learn

### AWS 그룹 (주황)
- AWS Lightsail / EC2

### 컨테이너 그룹 (회색)
- Docker
- Docker Compose
- Nginx
- Redis
- Celery

### 데이터 그룹 (파랑)
- SQLite

---

작성일: 2025-10-27
용도: 피그마 아키텍처 다이어그램 작성용

