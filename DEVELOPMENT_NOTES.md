# ClimbMate AI 개발 노트

## 🚀 현재 작업 상태

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
*마지막 업데이트: 2025-10-20*
