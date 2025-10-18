# 🚀 ClimbMate 배포 가이드

## 빠른 시작 (로컬 개발)

### 1. 환경변수 설정 (선택)

```bash
# GPT-4 Vision 사용 시
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 2. 실행

```bash
./start.sh
```

그러면 자동으로:
- 백엔드: http://localhost:8000
- 프론트엔드: http://localhost:3000

---

## Docker로 배포

### 1. Docker Compose 사용 (권장)

```bash
# GPT-4 사용 시
export OPENAI_API_KEY="sk-your-key"

# 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build
```

### 2. 개별 Docker 실행

**백엔드:**
```bash
docker build -t climbmate-backend .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="sk-your-key" \
  -v $(pwd)/backend/climbmate.db:/app/backend/climbmate.db \
  -v $(pwd)/backend/models:/app/backend/models \
  climbmate-backend
```

**프론트엔드:**
```bash
cd frontend
docker build -f Dockerfile.frontend -t climbmate-frontend .
docker run -p 3000:80 climbmate-frontend
```

---

## 클라우드 배포

### Render.com 배포 (무료)

**백엔드:**
1. GitHub에 푸시
2. Render.com → New Web Service
3. 저장소 연결
4. 설정:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Environment Variables: `OPENAI_API_KEY`

**프론트엔드:**
1. Render.com → New Static Site
2. Build Command: `cd frontend && npm install && npm run build`
3. Publish Directory: `frontend/dist`

### Vercel 배포 (프론트엔드)

```bash
cd frontend
vercel --prod
```

### Railway 배포 (백엔드)

```bash
railway init
railway up
```

---

## 환경변수

### 백엔드 (.env)
```
OPENAI_API_KEY=sk-xxx  # 선택
```

### 프론트엔드 (.env)
```
VITE_API_URL=https://your-backend-url.com
```

---

## 데이터 관리

### 데이터베이스 백업

```bash
# 로컬
cp backend/climbmate.db backend/climbmate.db.backup

# Docker
docker cp climbmate-backend-1:/app/backend/climbmate.db ./backup.db
```

### 모델 파일 백업

```bash
tar -czf models_backup.tar.gz backend/models/
```

---

## 성능 최적화

### 1. YOLO 모델 경량화
```python
# preprocess.py에서
conf=0.5  # 더 높은 confidence threshold
```

### 2. CLIP 배치 크기 조정
```python
# preprocess.py에서
batch_size=128  # GPU 메모리에 따라 조정
```

### 3. GPT-4 사용 조건 설정
```python
# 50개 데이터 축적 후 자체 모델 우선 사용
# hybrid_analyzer.py에서 자동 처리
```

---

## 모니터링

### API 문서
http://localhost:8000/docs

### 통계 확인
http://localhost:8000/api/stats

### 헬스체크
http://localhost:8000/api/health

---

## 트러블슈팅

### 1. "OPENAI_API_KEY not found"
→ 환경변수 설정 또는 없이 실행 (규칙 기반만 사용)

### 2. "Database is locked"
→ 동시 접근 이슈, 잠시 후 재시도

### 3. "CUDA out of memory"
→ CPU 모드로 실행 또는 배치 크기 감소

### 4. 프론트엔드 빌드 에러
→ `rm -rf node_modules && npm install`

---

## 비용 최적화

### GPT-4 Vision 사용 전략

**Phase 1 (0-50 피드백):**
- 모든 분석에 GPT-4 사용
- 비용: ~$1-2

**Phase 2 (50-100 피드백):**
- 자체 모델 학습 시작
- GPT-4 + 자체 모델 병행
- 비용: ~$0.5-1

**Phase 3 (100+ 피드백):**
- 자체 모델 우선 사용
- 신뢰도 낮을 때만 GPT-4
- 비용: ~$0.1-0.2

---

## 다음 단계

1. ✅ 데이터 수집 (피드백 50개 목표)
2. ✅ 자체 모델 학습 (`/api/train` 호출)
3. ✅ 정확도 모니터링
4. ✅ GPT-4 의존도 감소
5. ✅ 완전 독립 운영

