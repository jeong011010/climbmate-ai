# 🧗‍♀️ ClimbMate - AI 클라이밍 문제 분석

> 볼더링 벽 사진만 찍으면 AI가 홀드를 감지하고, 문제를 분류하고, 난이도와 유형을 분석합니다.

## ✨ 주요 기능

### 📸 이미지 분석
- **홀드 자동 감지**: YOLOv8-seg 기반
- **색상 인식**: CLIP AI 기반 정확한 색상 분류
- **문제 그룹화**: 같은 색상 홀드끼리 자동 그룹화

### 🎯 난이도 & 유형 분석
- **하이브리드 AI 시스템**:
  1. 자체 학습 ML 모델 (50+ 피드백 후)
  2. GPT-4 Vision (API 키 있을 때)
  3. 규칙 기반 분석 (백업)

### 📊 학습 데이터 축적
- **사용자 피드백 수집**
- **자동 모델 학습** (50개 데이터부터)
- **점진적 정확도 향상**

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

## 🎓 학습 데이터 축적 전략

### Phase 1: 데이터 수집 (0-50개)
```
1. 클라이밍 벽 사진 업로드
2. AI 분석 결과 확인
3. "📝 피드백" 버튼으로 실제 난이도/유형 입력
4. 데이터 자동 저장
```

### Phase 2: 모델 학습 (50+개)
```
1. 50개 이상 피드백 축적
2. 헤더에 "✅ AI 학습 가능" 표시
3. 백엔드에서 자동으로 모델 학습
4. 이후 분석부터 자체 모델 우선 사용
```

### Phase 3: 독립 운영 (100+개)
```
1. 자체 모델 정확도 향상
2. GPT-4 의존도 감소 (비용 절감)
3. 빠른 응답 속도
```

---

## 💰 비용 구조

### GPT-4 Vision 비용
- 이미지 1장당: ~$0.01
- 월 100건 분석: ~$1
- 월 1000건 분석: ~$10

### 비용 절감 전략
1. **초기**: GPT-4로 정확도 확보
2. **중기**: 데이터 축적
3. **장기**: 자체 모델로 전환 (무료!)

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

- [x] YOLO 기반 홀드 감지
- [x] CLIP AI 색상 분류
- [x] GPT-4 Vision 통합
- [x] 사용자 피드백 시스템
- [x] 자체 ML 모델 학습
- [x] 하이브리드 분석 시스템
- [x] PWA 모바일 최적화
- [ ] 벽 각도 자동 감지
- [ ] 홀드 타입 세부 분류
- [ ] 실시간 협업 기능

---

## 🤝 기여

피드백과 개선 제안을 환영합니다!

---

## 📄 라이선스

MIT License

---

**Made with ❤️ for climbers**

