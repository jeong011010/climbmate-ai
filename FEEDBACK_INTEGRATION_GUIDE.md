# 🎨 색상 피드백 시스템 통합 가이드

## 🎯 개요

사용자가 프론트엔드에서 직접 잘못 분류된 홀드의 색상을 수정할 수 있는 시스템입니다.

### 2가지 방법

1. **간단한 방법** (관리자용): Streamlit UI
2. **프로덕션** (사용자용): React 프론트엔드 통합 ✅

---

## 🚀 방법 1: Streamlit UI (관리자용)

### 실행
```bash
# 로컬
streamlit run holdcheck/color_feedback_ui.py --server.port 8501

# EC2
streamlit run holdcheck/color_feedback_ui.py --server.port 8501 --server.address 0.0.0.0
```

### 접속
```
http://localhost:8501  # 로컬
http://your-ec2-ip:8501  # EC2
```

**장점**: 코드 0줄, 즉시 사용  
**단점**: 일반 사용자 접근 어려움

---

## ⚡ 방법 2: React 프론트엔드 통합 (프로덕션)

### 1단계: 컴포넌트 추가 ✅

파일 추가됨:
- `frontend/src/components/ColorFeedback.jsx`
- `frontend/src/components/ColorFeedback.css`

### 2단계: App.jsx에 통합

```jsx
import ColorFeedback from './components/ColorFeedback'

function App() {
  const [problems, setProblems] = useState({})
  const [imageUrl, setImageUrl] = useState('')

  // 분석 완료 후
  const handleAnalysisComplete = (result) => {
    setProblems(result.problems)
    setImageUrl(result.imageUrl)
  }

  // 피드백 제출 후
  const handleFeedbackSubmit = (feedbacks) => {
    console.log('피드백 저장됨:', feedbacks)
    // 필요시 재분석 또는 UI 업데이트
  }

  return (
    <div>
      {/* 기존 분석 UI */}
      <AnalysisComponent onComplete={handleAnalysisComplete} />

      {/* 피드백 컴포넌트 추가 */}
      {problems && Object.keys(problems).length > 0 && (
        <ColorFeedback
          problems={problems}
          imageUrl={imageUrl}
          onFeedbackSubmit={handleFeedbackSubmit}
        />
      )}
    </div>
  )
}
```

### 3단계: 백엔드 API ✅

엔드포인트 추가됨 (`backend/main.py`):

#### POST `/api/color-feedback`
```javascript
// 요청
const response = await fetch('/api/color-feedback', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    feedbacks: [
      {
        hold_id: 0,
        predicted_color: "yellow",
        correct_color: "orange",
        problem_id: "g1",
        timestamp: "2025-01-23T10:00:00Z"
      }
    ]
  })
})

// 응답
{
  "status": "success",
  "message": "3개의 피드백이 저장되었습니다",
  "feedback_count": 3,
  "next_steps": "다음 분석부터 개선된 색상 분류가 적용됩니다"
}
```

#### GET `/api/feedback-stats`
```javascript
// 응답
{
  "status": "success",
  "total_feedbacks": 45,
  "last_updated": "2025-01-23T10:00:00Z",
  "color_stats": {
    "yellow": { "name": "노란색", "range_count": 2, "priority": 6 },
    "red": { "name": "빨간색", "range_count": 1, "priority": 4 }
  }
}
```

---

## 📱 UI 흐름

### 1. 초기 상태 (피드백 모드 OFF)
```
┌─────────────────────────────────┐
│  ✏️ 색상 수정하기  [버튼]       │
└─────────────────────────────────┘
```

### 2. 피드백 모드 ON
```
┌─────────────────────────────────────────────┐
│  ✅ 피드백 모드 종료    수정: 3개  💾 저장  │
├─────────────────────────────────────────────┤
│  💡 잘못 분류된 홀드의 올바른 색상 선택    │
├─────────────────────────────────────────────┤
│  [빨간색 그룹]                              │
│    홀드 #0  [빨강 ▼]  ✏️                   │
│    홀드 #1  [주황 ▼]                       │
│                                              │
│  [노란색 그룹]                              │
│    홀드 #3  [노랑 ▼]                       │
└─────────────────────────────────────────────┘
```

### 3. 피드백 저장 완료
```
✅ 3개의 피드백이 저장되었습니다!
다음 분석부터 개선된 색상 분류가 적용됩니다.
```

---

## 🔧 커스터마이징

### 색상 옵션 변경
```jsx
// ColorFeedback.jsx
const COLOR_OPTIONS = [
  { value: 'black', label: '검정', color: '#000000' },
  { value: 'custom', label: '커스텀', color: '#FF00FF' },  // 추가
  // ...
]
```

### 자동 재분석 추가
```jsx
const handleFeedbackSubmit = async (feedbacks) => {
  // 피드백 저장
  await fetch('/api/color-feedback', { ... })
  
  // 자동 재분석
  const reanalysis = await fetch('/api/analyze-colors', { ... })
  setProblems(reanalysis.problems)
}
```

### 신뢰도 낮은 홀드 강조
```jsx
// 신뢰도 70% 미만 자동 하이라이트
{hold.clip_confidence < 0.7 && (
  <span className="low-confidence-badge">⚠️ 확인 필요</span>
)}
```

---

## 📊 데이터 흐름

```
사용자
  ↓ (색상 수정)
React 컴포넌트
  ↓ (POST /api/color-feedback)
FastAPI 백엔드
  ↓ (save_user_feedback)
clustering.py
  ↓ (자동 학습)
color_ranges.json 업데이트
  ↓ (다음 분석 시 적용)
개선된 색상 분류 🎉
```

---

## 🎯 실제 사용 예시

### 시나리오: 노란색이 주황색으로 잘못 분류됨

1. **사용자**: "✏️ 색상 수정하기" 클릭
2. **UI**: 피드백 모드 활성화
3. **사용자**: 홀드 #5의 색상을 "주황" → "노랑"으로 변경
4. **UI**: 수정된 홀드에 ✏️ 표시
5. **사용자**: "💾 피드백 저장" 클릭
6. **백엔드**: 
   - 피드백 데이터 저장
   - HSV 범위 자동 조정
   - `color_ranges.json` 업데이트
7. **다음 분석**: 비슷한 홀드가 노란색으로 정확하게 분류됨!

---

## 🔒 보안 고려사항

### Rate Limiting
```python
# backend/main.py에 추가
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/api/color-feedback")
@limiter.limit("10/minute")  # 분당 10회 제한
async def submit_color_feedback(...):
    ...
```

### 인증
```jsx
// 로그인한 사용자만 피드백 가능
const response = await fetch('/api/color-feedback', {
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  ...
})
```

---

## 📈 모니터링

### 피드백 통계 확인
```bash
curl http://localhost:8000/api/feedback-stats
```

### 로그 확인
```bash
# 백엔드 로그
docker logs -f climbmate-backend | grep "피드백"

# 출력 예시:
# 📝 색상 피드백 수신: 3개
#    오분류 패턴:
#    yellow -> orange: 2건
#    red -> pink: 1건
# ✅ 피드백 반영 완료! (총 45건)
```

---

## 🚀 배포 체크리스트

- [x] React 컴포넌트 추가
- [x] 백엔드 API 추가
- [x] CORS 설정 확인
- [ ] Git 푸시
- [ ] EC2 배포
- [ ] 프론트엔드 빌드
- [ ] 테스트

---

## 🎓 고급 기능 (선택)

### 1. 실시간 프리뷰
```jsx
// 색상 변경 시 즉시 시각화
const handleColorChange = (holdId, newColor) => {
  // 이미지에 오버레이 업데이트
  updateImageOverlay(holdId, newColor)
}
```

### 2. 자동 제안
```jsx
// AI가 대안 색상 제안
{hold.clip_confidence < 0.7 && (
  <div className="suggestions">
    추천: {hold.alternative_colors.join(', ')}
  </div>
)}
```

### 3. 피드백 히스토리
```jsx
// 사용자의 과거 피드백 표시
const MyFeedbacks = () => {
  const [history, setHistory] = useState([])
  
  useEffect(() => {
    fetch('/api/user-feedbacks').then(...)
  }, [])
  
  return <FeedbackHistory items={history} />
}
```

---

**다음 단계**: Git 푸시 및 EC2 배포

