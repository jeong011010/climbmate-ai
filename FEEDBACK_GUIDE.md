# 🎨 피드백 기반 색상 규칙 개선 가이드

## 📋 워크플로우

### 1️⃣ 피드백 데이터 초기화 (처음 시작할 때만)
```bash
docker compose exec backend python3 reset_all_feedback.py
```

### 2️⃣ 앱에서 피드백 제공
- 잘못 분류된 홀드 색상을 수정하며 피드백 제출
- **최소 50-100개 피드백** 권장 (각 색상당 10개 이상)

### 3️⃣ 피드백 데이터 확인
```bash
docker compose exec backend python3 check_feedback_data.py
```

**출력 예시:**
```
📊 전체 피드백: 120개
✅ user_correct_color 분포:
  purple: 25개
  pink: 20개
  blue: 18개
  white: 15개
  ...
```

### 4️⃣ 규칙 자동 조정 (피드백이 충분히 쌓이면)
```bash
docker compose exec backend python3 auto_tune_color_rules.py
docker compose restart backend
```

**이 스크립트가 하는 일:**
- 피드백 데이터의 HSV 분포 분석
- `color_ranges.json`의 HSV 범위 자동 조정
- 백업 파일 생성 (`color_ranges.json.backup`)

### 5️⃣ 효과 확인
- 같은 이미지로 재분석
- 정확도가 개선되었는지 확인
- 필요시 2-4 반복

---

## 🔧 수동 조정 (고급)

자동 조정이 만족스럽지 않으면:

### 1. `color_ranges.json` 직접 수정
```json
{
  "white": {
    "hsv_ranges": [{
      "h": [0, 180],
      "s": [0, 15],    // ← 채도 범위 조정
      "v": [200, 255]  // ← 명도 범위 조정
    }]
  }
}
```

### 2. `holdcheck/clustering.py` 수정
```python
def classify_color_simple_hsv(h, s, v):
    if v >= 200 and s <= 15:  # ← 조건 수정
        return "white", 0.95
```

### 3. 변경사항 적용
```bash
git add -A
git commit -m "🎨 색상 범위 수동 조정"
git push origin main

# EC2에서
cd ~/climbmate-ai
git pull origin main
docker compose restart backend
```

---

## 📊 현재 색상 범위 (규칙 기반)

| 색상 | Hue | Saturation | Value |
|------|-----|------------|-------|
| **black** | 0-180 | 0-60 | 0-150 |
| **white** | 0-180 | 0-15 | 200-255 |
| **red** | 0-10, 170-180 | 100-255 | 100-255 |
| **orange** | 10-25 | 100-255 | 100-255 |
| **yellow** | 25-40 | 100-255 | 150-255 |
| **green** | 40-75 | 100-255 | 100-255 |
| **mint** | 75-100 | 30-255 | 90-255 |
| **blue** | 100-125 | 50-255 | 110-255 |
| **purple** | 125-160 | 60-255 | 100-255 |
| **pink** | 160-175 | 80-255 | 200-255 |

---

## ⚠️ 주의사항

1. **피드백 품질이 중요합니다**
   - 명확한 오분류만 피드백
   - 애매한 경우는 스킵

2. **충분한 데이터가 필요합니다**
   - 각 색상당 최소 10개 이상
   - 전체 50-100개 권장

3. **백업 필수**
   - `auto_tune_color_rules.py`는 자동으로 백업 생성
   - 문제 발생 시 복원 가능

4. **점진적 개선**
   - 한 번에 큰 변화 X
   - 조금씩 개선하며 확인

---

## 🎯 현재 상태

- ✅ **규칙 기반 분류**: 활성화 (`color_ranges.json`)
- ✅ **피드백 수집**: 활성화 (DB 저장)
- ✅ **자동 튜닝**: 준비 완료 (`auto_tune_color_rules.py`)
- ⏸️ **실시간 학습**: 비활성화 (수동 트리거 방식)

**장점:**
- 피드백이 쌓일 때까지 안정적인 규칙 기반 유지
- 원할 때 수동으로 규칙 개선
- 롤백 가능 (백업)

