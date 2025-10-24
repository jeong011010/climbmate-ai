# ClimbMate AI 개발 노트

## 🚀 현재 작업 상태

### 📌 **체크포인트: 서버 완벽 동작 상태** (커밋: 5f79dda)
**날짜**: 2025-10-21  
**상태**: ✅ 서버에서 모든 기능이 완벽하게 동작하는 안정적인 상태

#### 🏗️ **현재 아키텍처**
```
브라우저: 이미지 업로드 + 진행률 표시
    ↓
서버: 작업 큐 관리 (FastAPI + Celery)
    ↓
Celery Worker: YOLO + CLIP + GPT-4 분석
    ↓
결과 반환 (비동기 폴링)
```

#### ✅ **완벽 동작하는 기능들**
- **YOLO 홀드 감지**: 정확한 홀드 위치 및 크기 감지
- **CLIP 색상 분석**: AI 기반 정확한 색상 분류
- **GPT-4 문제 생성**: 한국어로 상세한 문제 분석
- **비동기 처리**: Celery + Redis로 안정적인 백그라운드 처리
- **실시간 진행률**: 폴링 방식으로 사용자에게 진행 상황 표시
- **데이터베이스**: SQLite로 분석 결과 및 피드백 저장
- **프론트엔드**: React로 완전한 UI 구현

#### ⚠️ **현재 제약사항**
- **메모리 사용량**: 서버 메모리 1.9GB < 필요한 메모리 2GB
- **Celery Worker 크래시**: 메모리 부족으로 인한 SIGKILL 오류 발생 가능
- **모든 AI 모델이 서버에서 실행**: 브라우저 오프로딩 없음

#### 🎯 **이 체크포인트의 의미**
- 모든 핵심 기능이 완벽하게 구현된 상태
- 서버 메모리만 충분하다면 완벽하게 동작
- 브라우저 오프로딩 시도 전의 안정적인 베이스라인
- 향후 최적화 작업의 기준점

---

### 최근 수정 사항 (2025-10-20)

#### ✅ 모든 핵심 기능 완성! (커밋: 5c58d1b)
- **GPT-4 한국어 응답**: 프롬프트를 한국어로 변경하여 모든 분석 결과가 한국어로 제공
- **프론트엔드 데이터 표시**: 실제 데이터 구조에 맞게 필드명 수정
- **분석 가능 문제 수**: `statistics.analyzable_problems` 추가로 프론트엔드에 표시
- **세분화된 진행률**: 0% → 10% → 30% → 50% → 70% → 95% → 100% 단계별 표시
- **불필요한 로그 정리**: 프로덕션 환경에 맞게 디버깅 로그 제거

#### ✅ 색상별 문제 분석 완성 (커밋: 8babec9)
- **7개 문제 동시 생성**: pink, green, gray, blue, yellow, brown, red
- **정확한 색상명 표시**: "unknown" 대신 실제 색상명 (purple, blue, red 등)
- **홀드 수 필터링**: 3개 미만 홀드는 자동 제외
- **GPT-4 분석 성공**: 모든 문제에 대해 정상적인 JSON 응답

#### ✅ Celery Redis 직렬화 에러 수정 (커밋: 414acea)
- **문제**: `ValueError: Exception information must include the exception type`
- **해결 방법**:
  - `celery_app.py`: `result_extended=True` 추가하여 예외 메타데이터 확장
  - `backend/ai_tasks.py`: 예외를 `raise` 대신 에러 dict 반환하도록 수정
  - `backend/main.py`: `/api/analyze-status` 엔드포인트 에러 처리 개선
- **영향**: 홀드 감지 실패 시에도 안정적으로 에러 메시지 반환

## 🛠 기술 스택

### 백엔드
- **FastAPI**: Python 웹 프레임워크
- **Celery**: 비동기 작업 큐
- **Redis**: Celery 브로커
- **YOLO**: 홀드 감지 모델
- **CLIP**: 색상 분석 AI
- **GPT-4**: 문제 생성 AI

### 프론트엔드
- **React**: JavaScript 프레임워크
- **Vite**: 빌드 도구
- **Tailwind CSS**: 스타일링

### 인프라
- **Docker**: 컨테이너화
- **Nginx**: 리버스 프록시
- **AWS Lightsail**: 클라우드 호스팅
- **Let's Encrypt**: SSL 인증서

## 🌐 배포 정보

### 도메인
- **프로덕션**: `https://climbmate.store`
- **SSL**: Let's Encrypt 자동 갱신

### 서버 정보
- **플랫폼**: AWS Lightsail Ubuntu 24.04
- **인스턴스**: 2GB RAM, 1 vCPU
- **디스크**: 40GB SSD

## 📁 프로젝트 구조

```
climbmate/
├── backend/           # FastAPI 백엔드
│   ├── main.py       # 메인 API 서버
│   ├── ai_tasks.py   # Celery 비동기 작업
│   ├── gpt4_analyzer.py
│   └── hybrid_analyzer.py
├── frontend/         # React 프론트엔드
│   ├── src/
│   │   ├── App.jsx
│   │   └── clientAI.js
│   └── dist/        # 빌드된 정적 파일
├── holdcheck/        # YOLO 홀드 감지
│   ├── preprocess.py
│   └── clustering.py
├── docker-compose.yml
├── nginx/
│   └── nginx.conf
└── certbot/         # SSL 인증서
```

## 🔧 개발 환경 설정

### 로컬 개발
```bash
# 백엔드 실행
cd backend
python -m uvicorn main:app --reload

# 프론트엔드 실행
cd frontend
npm run dev

# Celery Worker (로컬)
celery -A backend.ai_tasks worker --loglevel=info
```

### Docker 개발
```bash
# 전체 서비스 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 특정 서비스 재시작
docker-compose restart backend
```

## 🚨 알려진 이슈 및 해결 방법

### 1. Python 코드 수정 후 반영되지 않음
- **원인**: Docker 이미지에 코드가 복사되어 있어 볼륨 마운트로는 반영 안 됨
- **해결**: 
  ```bash
  docker compose build backend celery-worker  # 이미지 재빌드 필수!
  docker compose up -d backend celery-worker
  ```

### 2. docker-compose 명령어 없음 (EC2)
- **원인**: 최신 Docker는 `docker compose` (하이픈 없음) 사용
- **해결**: `docker-compose` → `docker compose`로 변경

### 3. 메모리 최적화
- **CLIP 모델**: 첫 요청 시 로딩 (메모리 절약)
- **YOLO 모델**: 싱글톤 패턴으로 캐싱
- **Redis**: 결과 1시간 후 자동 삭제 (`result_expires=3600`)

## 📝 API 엔드포인트

### 동기 분석 (기존)
- `POST /api/analyze` - 전체 분석 (동기)

### 비동기 분석 (신규)
- `POST /api/analyze-stream` - 비동기 분석 시작
- `GET /api/analyze-status/{task_id}` - 진행률 조회

### 기타
- `GET /api/health` - 서버 상태 확인

## 🔄 배포 프로세스

### 코드 배포
```bash
# 1. 로컬에서 수정 후 커밋
git add -A
git commit -m "수정 내용"
git push origin main

# 2. EC2에서 배포
ssh ubuntu@your-server
cd ~/climbmate-ai
git pull origin main

# ⚠️ 중요: docker compose (하이픈 없음) 사용
docker compose build backend celery-worker
docker compose up -d backend celery-worker

# 로그 확인
docker compose logs -f celery-worker
```

### 전체 재시작 (문제 발생 시)
```bash
docker compose down
docker compose up -d
docker compose logs -f
```

### SSL 인증서 갱신
```bash
sudo docker compose exec certbot certbot renew
sudo docker compose restart nginx
```

## 🎯 다음 작업 계획

1. ✅ ~~Celery Worker 코드 동기화 문제 해결~~ - 완료
2. ✅ ~~Celery Redis 예외 처리 개선~~ - 완료
3. **프론트엔드 실시간 진행률 UI 개선**
4. **브라우저 모델 로딩 최적화**
5. **GPT-4 분석 결과 캐싱**
6. **성능 모니터링 대시보드 추가**

## 📞 연락처 및 참고

- **GitHub**: https://github.com/jeong011010/climbmate-ai
- **도메인**: https://climbmate.store
- **개발자**: 김재즈

---

## 🎨 최신 업데이트 (2025-10-24)

### ✅ 홀드 상세 정보 및 색상 피드백 시스템 구축

#### 🎯 주요 기능 추가

**1. 홀드 클릭 시 세그먼테이션 윤곽선 표시**
- SVG 오버레이로 홀드 contour 실시간 렌더링
- 선택된 홀드는 노란색 점선 + 애니메이션
- 같은 문제 그룹의 홀드는 초록색 윤곽선

**2. 홀드 상세 정보 UI**
```jsx
선택된 홀드 컴포넌트:
├─ 문제 그룹 색상 (해당 홀드가 속한 문제의 색상)
├─ 홀드 실제 색상 (AI가 감지한 RGB/HSV 색상)
│  ├─ 원형 색상 샘플 (실제 RGB 표시)
│  ├─ RGB(R, G, B) 값
│  └─ 색상 이름 (AI 감지)
└─ 위치 정보 (X, Y, HSV)
```

**3. 홀드 색상 피드백 시스템 (ML 학습용)**
- 사용자가 AI 예측이 틀렸을 때 올바른 색상 제출
- 데이터베이스에 자동 저장 (`hold_color_feedback` 테이블)
- 피드백 30개 이상 시 ML 재학습 준비 알림

**4. 데이터베이스 확장**
```sql
CREATE TABLE hold_color_feedback (
    id, problem_id, hold_id,
    center_x, center_y,
    rgb_r, rgb_g, rgb_b,
    hsv_h, hsv_s, hsv_v,
    lab_l, lab_a, lab_b,
    color_stats (JSON),
    predicted_color,
    user_correct_color,
    created_at
)
```

**5. ML 색상 학습 로직**
```python
# backend/ml_trainer.py
- train_color_model(): Random Forest 색상 분류 모델
- extract_color_features(): 23개 특징 벡터 추출
- predict_color(): 학습된 모델로 색상 예측
- Top-3 예측 지원
```

#### 🔧 색상 추출 정확도 개선 (7단계 개선)

**1단계: 마스크 경계 제거 (Erosion)**
- 배경 픽셀 혼입 방지
- 5x5 kernel로 경계 제거

**2단계: 원본 이미지 사용**
- 명도 보정 비활성화 (색상 왜곡 방지)
- CLIP AI와 동일한 원본 데이터 사용

**3단계: 초크 자국 제거 비활성화**
- V > 200 필터링 제거 (흰색 홀드 왜곡 방지)

**4단계: Outlier 제거 비활성화**
- 중앙값 ± σ 필터링 제거
- 원본 픽셀 모두 사용

**5단계: 중앙값 기반 색상 추출**
- 평균 대신 중앙값 사용 (outlier에 강함)

**6단계: 무채색 판단 개선**
```python
S < 30:
  V < 60: 검정 ⚫
  V > 200: 흰색 ⚪
  60-200: 회색 ⬜
```

**7단계: 상식적인 HSV 색상 분류**
```python
유채색 (OpenCV H: 0-180):
- 0-8, 170-180: 빨강 🔴
- 8-18: 주황 🟠
- 18-30: 노랑 🟡
- 30-45: 연두 🟢
- 45-80: 초록 🟢
- 80-95: 민트 🫧
- 95-130: 파랑 🔵
- 130-150: 보라 🟣
- 150-170: 핑크 🩷
```

#### 🐛 해결된 주요 문제

**문제 1: 색상 왜곡**
- 원인: 명도 보정이 색상을 과도하게 변경
- 해결: 원본 이미지 사용 (CLIP AI 방식)
- 효과: 노란색→주황, 빨강→초록 문제 해결 ✅

**문제 2: 검정색 홀드가 회색으로 감지**
- 원인: 초크 자국, 반사광, 배경 픽셀 혼입
- 해결: 마스크 경계 제거 + 중앙값 사용
- 효과: RGB(133,155,147) → RGB(40,40,40) ✅

**문제 3: 흰색 홀드가 회색으로 감지**
- 원인: 초크 제거 필터가 흰색 픽셀까지 제거
- 해결: 초크 제거 비활성화
- 효과: 밝은 곳의 흰색 홀드 정확히 감지 ✅

**문제 4: 피드백 제출 422 에러**
- 원인: Pydantic 타입 검증 실패 (dict, list)
- 해결: Optional[Dict[str, Any]] 명시적 타입 지정
- 효과: 피드백 정상 제출 ✅

**문제 5: 피드백 제출 시 hold_id 타입 에러**
- 원인: hold_id가 숫자로 전송 (문자열 기대)
- 해결: String() 변환
- 효과: 피드백 저장 성공 ✅

#### 🚀 배포 자동화

**배포 스크립트 추가**
```bash
# server_deploy.sh
1. git pull (최신 코드)
2. docker system prune (디스크 정리 ~7GB)
3. docker compose down (컨테이너 중지)
4. docker compose build (이미지 재빌드)
5. docker compose up -d (컨테이너 시작)
6. 상태 확인 (디스크, Docker, 컨테이너)
```

**사용법:**
```bash
ssh ubuntu@ip-172-31-12-99
cd ~/climbmate-ai
./server_deploy.sh
```

#### 📊 시스템 흐름

```
사용자 홀드 클릭
    ↓
홀드 상세 정보 표시
    ├─ 문제 그룹 색상 (clustering 결과)
    ├─ 실제 홀드 색상 (AI 감지)
    │   ├─ RGB 원형 샘플
    │   ├─ RGB(R,G,B) 값
    │   └─ 색상 이름
    └─ 위치 (X, Y, HSV)
    ↓
색상 피드백 버튼 클릭
    ↓
올바른 색상 선택 (12색 + 서브 색상)
    ↓
데이터베이스 저장 (hold_color_feedback)
    ├─ RGB, HSV, LAB 특징
    ├─ 통계 특징 (JSON)
    ├─ AI 예측 vs 사용자 피드백
    └─ ML 학습용 데이터 축적
    ↓
피드백 30개 이상 → ML 재학습 가능
```

#### 🎨 색상 분류 시스템

**현재 방식: 규칙 기반 (use_clip_ai=True이지만 실제로는 규칙 사용 중)**

**색상 추출 파이프라인:**
```
원본 이미지
    ↓
마스크 경계 제거 (erosion 5x5)
    ↓
원본 픽셀 추출 (필터링 없음)
    ↓
HSV 중앙값 계산
    ↓
상식적 HSV 분류
    ├─ 무채색: S < 30
    │   ├─ 검정: V < 60
    │   ├─ 흰색: V > 200
    │   └─ 회색: V 60-200
    └─ 유채색: H 각도 기반 12색
```

#### 🔍 Git 커밋 히스토리

```bash
5dc2deb - feat: 상식적인 HSV 기반 색상 분류
fefea3b - feat: 자동 배포 스크립트 추가
690c161 - feat: 규칙 기반도 원본 이미지 사용
937a670 - feat: CLIP AI 색상 프롬프트 개선
fc02b7c - feat: 상대적 명도 보정
bbf6ff3 - feat: 명도 정규화 추가
61ff000 - fix: 피드백 저장 에러 수정
37f288a - feat: 홀드 색상 피드백 ML 학습 시스템
42286e1 - feat: 검정색/흰색 색상 추출 정확도 개선
d7e77c1 - fix: use_clip_ai=False 경로에도 contour 추가
d6e39af - fix: SVG 오버레이 렌더링 및 색상 데이터 전달
```

#### 🎯 다음 단계

1. **사용자 피드백 수집** (30-100개 목표)
2. **ML 색상 모델 재학습** (`train_color_model()`)
3. **CLIP AI vs 규칙 기반 정확도 비교**
4. **최적 색상 분류 방식 선택**

---
*마지막 업데이트: 2025-10-24*
