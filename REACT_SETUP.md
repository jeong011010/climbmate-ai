# 🚀 ClimbMate React 프론트엔드 설정

## 📁 프로젝트 구조

```
climbmate/
├── backend/              # FastAPI 백엔드
│   ├── main.py          # API 서버
│   └── Dockerfile
├── frontend/            # React 프론트엔드 (생성 중...)
│   ├── src/
│   │   ├── App.js
│   │   ├── components/
│   │   ├── services/
│   │   └── styles/
│   └── public/
└── holdcheck/           # 기존 Streamlit (참고용)
```

## 🛠 설치 및 실행

### 1. Backend (FastAPI)

```bash
# requirements.txt에 추가
pip install fastapi uvicorn python-multipart

# 실행
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend (React)

```bash
# 프로젝트 생성 (진행 중...)
npx create-react-app frontend --template cra-template-pwa

# 의존성 설치
cd frontend
npm install axios react-dropzone

# 개발 서버 실행
npm start  # http://localhost:3000
```

## 🌐 배포

### Backend: Railway / Fly.io
```bash
# Railway
railway init
railway up

# Fly.io
fly launch
fly deploy
```

### Frontend: Vercel / Netlify
```bash
# Vercel
npm install -g vercel
vercel

# Netlify
npm install -g netlify-cli
netlify deploy
```

## 📱 PWA 기능

React PWA 템플릿에 기본 포함:
- ✅ Service Worker
- ✅ Offline 지원
- ✅ 홈 화면 추가
- ✅ 푸시 알림 (선택)

## 🎯 API 엔드포인트

**Base URL**: `http://localhost:8000`

### POST /api/analyze
이미지 분석 요청

**Request:**
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('wall_angle', 'overhang'); // optional

const response = await axios.post('/api/analyze', formData);
```

**Response:**
```json
{
  "problems": [
    {
      "id": "ai_blue",
      "color_name": "blue",
      "color_rgb": [50, 120, 200],
      "hold_count": 8,
      "analysis": {
        "difficulty": {
          "grade": "V4-V5",
          "level": "중급",
          "confidence": 0.65
        },
        "climb_type": {
          "primary_type": "다이나믹",
          "types": ["다이나믹", "코디네이션"]
        }
      }
    }
  ],
  "statistics": {
    "total_holds": 25,
    "total_problems": 4,
    "analyzable_problems": 3
  }
}
```

## 🎨 React 컴포넌트 구조

```
src/
├── App.js                    # 메인 앱
├── components/
│   ├── ImageUpload.js       # 이미지 업로드
│   ├── ProblemList.js       # 문제 목록
│   ├── ProblemDetail.js     # 문제 상세
│   ├── Statistics.js        # 통계 카드
│   └── Loading.js           # 로딩 스피너
├── services/
│   └── api.js               # API 호출
└── styles/
    └── App.css              # 스타일
```

## 🚀 빠른 시작 (React 생성 완료 후)

1. **Backend 실행:**
```bash
cd backend
docker build -t climbmate-api .
docker run -p 8000:8000 climbmate-api
```

2. **Frontend 실행:**
```bash
cd frontend
npm start
```

3. **브라우저 접속:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs

## 💡 다음 단계

React 프로젝트 생성이 완료되면:
1. `frontend/src/App.js` 수정
2. API 연동
3. PWA 설정 활성화
4. 배포

완료되면 알려드리겠습니다! 🎉

