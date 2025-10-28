# 🎨 색상 분류 ML 모델 학습 가이드

## 현재 상황
- 📊 **896개 데이터** 수집 완료
- 🤖 ML 모델 학습 준비 완료
- 🔄 규칙 기반 → ML 기반으로 전환

## 학습 방법

### 방법 1: EC2 서버에서 직접 학습 (추천)

```bash
# 1. 프로젝트 디렉토리로 이동
cd /path/to/climbmate

# 2. Python 가상환경 활성화
source venv/bin/activate  # 또는 해당 venv 경로

# 3. 학습 스크립트 실행
python train_color_model_896.py

# 4. 결과 확인
# - backend/models/color_model.pkl
# - backend/models/color_encoder.pkl

# 5. 서버 재시작 (모델 로드)
docker-compose restart backend
# 또는
pm2 restart backend
```

### 방법 2: API 엔드포인트 사용

```bash
# POST 요청으로 학습 트리거
curl -X POST https://climbmate.store/api/train-color-model

# 또는 브라우저에서
# https://climbmate.store/api/train-color-model
```

## 학습 프로세스

### 1단계: 데이터 준비
```
📊 896개 데이터 로드
🔄 gray → white 자동 변환
📈 색상별 분포 확인
⚠️ 샘플 부족 색상 필터링 (< 5개)
```

### 2단계: 특징 추출 (21차원)
```
1-3.   RGB 값 (R, G, B)
4-6.   HSV 값 (H, S, V)
7-9.   LAB 값 (L, A, B)
10-14. HSV 통계 (평균, 표준편차)
15-17. RGB 통계 (표준편차)
18-19. LAB 통계 (A, B 평균)
20.    홀드 크기 (정규화)
21.    홀드 원형도
```

### 3단계: 모델 학습
```
🤖 알고리즘: Random Forest
🌲 트리 개수: 200
📊 최대 깊이: 15
⚖️ 클래스 가중치: balanced (불균형 처리)
```

### 4단계: 평가
```
✅ 훈련 정확도
✅ 테스트 정확도 (80/20 분할)
✅ Cross-Validation (3-fold)
```

## 모델 사용 우선순위

학습 완료 후 자동으로 적용:

```python
# 1순위: ML 모델 (신뢰도 >= 0.70)
if ml_color and ml_confidence >= 0.70:
    return ml_color
    
# 2순위: 규칙 기반
else:
    return rule_based_color
```

## 예상 성능

### 초기 학습 (896개)
- 예상 정확도: **75-85%**
- 규칙 기반보다 **10-20% 향상** 예상

### 피드백 축적 후 (1000+개)
- 예상 정확도: **85-90%**
- 지속적 개선

## 재학습 주기

```bash
# 피드백 100개 추가될 때마다 재학습 권장
python train_color_model_896.py
docker-compose restart backend
```

## 트러블슈팅

### 데이터 부족
```
❌ 각 색상마다 최소 5개 필요
✅ 피드백 더 수집하거나 해당 색상 제외
```

### 학습 실패
```
❌ 특징 추출 실패
✅ hold_color_feedback 테이블 스키마 확인
✅ color_stats JSON 형식 확인
```

### 모델 미적용
```
❌ 모델 로드 실패
✅ backend/models/ 디렉토리 확인
✅ 서버 재시작 확인
✅ 로그에서 "🤖 ML 예측" 메시지 확인
```

## 성능 모니터링

```python
# 로그에서 확인
🤖 ML 예측: 홀드 3 → blue (신뢰도: 0.85)  # ML 사용
📏 룰 기반: 홀드 5 → red (신뢰도: 0.90)   # 규칙 사용
```

ML 예측이 많이 보이면 성공!


