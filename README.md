# 🧗‍♀️ ClimbMate - AI 클라이밍 문제 분석

> 볼더링 벽 사진만 찍으면 AI가 홀드를 감지하고, 문제를 분류하고, 난이도와 유형을 분석합니다.

## ✨ 주요 기능

### 📸 이미지 분석
- **홀드 자동 감지**: YOLOv8-seg 기반 정확한 홀드 감지
- **색상 인식**: CLIP AI 기반 정확한 색상 분류 (10가지 색상 지원)
- **문제 그룹화**: 같은 색상 홀드끼리 자동 그룹화 (최소 3개 홀드 이상)

### 🎯 난이도 & 유형 분석
- **GPT-4 Vision 통합**: 한국어 분석 결과 제공
- **실시간 진행률**: 세분화된 분석 단계 표시 (0% → 10% → 30% → 50% → 70% → 95% → 100%)
- **다중 문제 분석**: 색상별로 여러 문제 동시 분석

### 📊 분석 결과
- **난이도**: V0-V10 등급 시스템
- **유형**: dynamic, static, crimp, sloper, balance 등
- **상세 분석**: 홀드 간격, 크기, 배치 기반 기술적 분석
- **실용적 팁**: 코어 활용, 모멘텀 사용 등 구체적 조언

---

## 🚀 빠른 시작

### 요구사항
- Python 3.10+
- Node.js 18+
- (선택) OpenAI API 키

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone <repository>
cd climbmate

# 2. Python 가상환경 (권장)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 4. 실행
./start.sh
```

**접속:**
- 프론트엔드: http://localhost:3000
- 백엔드 API: http://localhost:8000
- API 문서: http://localhost:8000/docs

---

## 🎯 분석 과정

### 1단계: 홀드 감지 (10%)
- YOLOv8-seg 모델로 홀드 자동 감지
- 바운딩 박스와 마스크 생성

### 2단계: 색상 분석 (30%)
- CLIP AI로 각 홀드의 색상 분류
- 10가지 색상 자동 인식 (red, blue, green, yellow, purple, pink, gray, brown, white, black)

### 3단계: 문제 생성 (50%)
- 같은 색상 홀드끼리 그룹화
- 최소 3개 홀드 이상인 그룹만 문제로 인식

### 4단계: GPT-4 분석 (70-95%)
- 각 문제별로 GPT-4 Vision API 호출
- 한국어로 난이도, 유형, 분석, 팁 제공
- 실시간 진행률 표시 (1/7, 2/7, ...)

### 5단계: 완료 (100%)
- 모든 분석 결과 통합
- 프론트엔드에 결과 전달

---

## 📱 모바일 PWA

### 설치 방법
1. 모바일 브라우저에서 접속
2. "홈 화면에 추가" 클릭
3. 앱처럼 사용!

### 기능
- ✅ 오프라인 캐싱
- ✅ 카메라 직접 촬영
- ✅ 바텀시트 UI
- ✅ 터치 최적화

---

## 🛠️ 기술 스택

### Frontend
- React 19 + Vite
- Tailwind CSS
- Axios
- PWA (Service Worker)

### Backend
- FastAPI
- YOLOv8-seg
- CLIP AI
- GPT-4 Vision
- scikit-learn

### Database
- SQLite (경량, 파일 기반)

### ML Pipeline
- Random Forest / Gradient Boosting
- 특징 추출: 25개 특징
- Cross-validation

---

## 📊 API 엔드포인트

### POST /api/analyze
이미지 분석 (홀드 감지 + 난이도/유형)

### POST /api/feedback
사용자 피드백 저장

### POST /api/train
자체 모델 학습 (50+ 데이터 필요)

### GET /api/stats
통계 및 모델 성능 조회

---

## 🎯 로드맵

### ✅ 완료된 기능
- [x] YOLO 기반 홀드 감지
- [x] CLIP AI 색상 분류 (10가지 색상)
- [x] GPT-4 Vision 통합 (한국어 응답)
- [x] 실시간 진행률 표시
- [x] 다중 문제 동시 분석
- [x] 색상별 문제 그룹화
- [x] PWA 모바일 최적화
- [x] Celery 비동기 작업 큐
- [x] Redis 캐시 시스템

### 🚧 개발 예정
- [ ] 벽 각도 자동 감지
- [ ] 홀드 타입 세부 분류 (crimp, sloper, jug 등)
- [ ] 사용자 피드백 시스템
- [ ] 자체 ML 모델 학습
- [ ] 실시간 협업 기능

---

## 🤝 기여

피드백과 개선 제안을 환영합니다!

---

## 📄 라이선스

MIT License

---

**Made with ❤️ for climbers**

