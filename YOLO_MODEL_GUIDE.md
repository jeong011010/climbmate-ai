# 🎯 YOLO 모델 전환 가이드

## 📋 사용 가능한 모델

### 1. Roboflow 모델 (기본)
- 파일: `holdcheck/roboflow_weights/weights.pt`
- 용도: 기존 훈련된 모델

### 2. Alternative 모델 (새 모델)
- 파일: `holdcheck/roboflow_weights/weights_alternative.pt`
- 용도: 성능 비교 테스트용

---

## 🔄 모델 전환 방법

### 로컬 개발 환경:

**Alternative 모델 사용:**
```bash
export YOLO_MODEL=alternative
python -m uvicorn backend.main:app --reload
```

**기본 모델로 복귀:**
```bash
export YOLO_MODEL=roboflow
# 또는
unset YOLO_MODEL
```

---

### Docker 환경:

#### 방법 1: 환경 변수로 설정
```bash
# Alternative 모델 사용
YOLO_MODEL=alternative docker compose up -d

# 기본 모델 사용
YOLO_MODEL=roboflow docker compose up -d
```

#### 방법 2: .env 파일 생성
```bash
# .env 파일 생성
echo "YOLO_MODEL=alternative" > .env

# Docker Compose 실행
docker compose up -d
```

#### 방법 3: docker-compose.yml 수정 (영구 설정)
```yaml
environment:
  - YOLO_MODEL=alternative  # 또는 roboflow
```

---

## 🧪 성능 비교 테스트 절차

### 1단계: Roboflow 모델 테스트
```bash
# 기본 모델로 실행
docker compose down
docker compose up -d

# 같은 이미지 여러 개로 테스트
# - 홀드 감지 개수 기록
# - 색상 정확도 기록
# - 처리 시간 기록
```

### 2단계: Alternative 모델 테스트
```bash
# 모델 전환
docker compose down
YOLO_MODEL=alternative docker compose up -d

# 동일한 이미지로 테스트
# - 결과 비교
```

### 3단계: 결과 비교
- ✅ 홀드 감지율 (얼마나 많은 홀드를 찾았나?)
- ✅ False Positive (잘못 감지한 것은 없나?)
- ✅ False Negative (놓친 홀드는 없나?)
- ✅ 세그멘테이션 품질 (윤곽선이 정확한가?)
- ✅ 처리 속도

---

## 📊 현재 진행률 구조

### 홀드 감지 (10% ~ 50%):
- 10%: YOLO 모델 로딩
- 20%: 이미지 전처리
- 30%: 홀드 감지 진행
- 50%: 홀드 감지 완료 ✅

### 색상 분석 (50% ~ 60%):
- 52%: 색상 분석 중
- 58%: 색상 분석 완료 ✅

### 문제 생성 (60% ~ 65%):
- 60%: 문제 그룹 생성
- 65%: 완료 ✅

### GPT-4 분석 (65% ~ 95%):
- 각 문제별 진행률 표시

---

## 💡 팁

**더 나은 모델 선택 기준:**
1. **정확도** > 속도
2. **False Negative 최소화** (놓친 홀드 없게)
3. **세그멘테이션 품질** (윤곽선 정확도)

**테스트 추천:**
- 다양한 조명 조건
- 다양한 홀드 밀도
- 다양한 색상 조합

