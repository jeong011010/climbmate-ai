# 🧗‍♀️ ClimbMate - React Frontend

## ✅ 완성!

### 🚀 **빠른 시작 (로컬 개발)**

**1. Backend 실행:**
```bash
cd /Users/kimjazz/Desktop/project/climbmate
docker restart climbmate-app
# API: http://localhost:8000
```

**2. Frontend 실행:**
```bash
cd /Users/kimjazz/Desktop/project/climbmate/frontend
npm run dev
# Frontend: http://localhost:3000
```

**3. 브라우저 접속:**
- Frontend: **http://localhost:3000** ✨
- Backend API Docs: **http://localhost:8000/docs**

---

## 📱 **모바일에서 테스트**

1. **내 IP 확인:**
```bash
ipconfig getifaddr en0  # Mac
# 예: 192.168.0.10
```

2. **모바일 브라우저에서:**
- `http://192.168.0.10:3000`

---

## 🌐 **배포하기**

### **Frontend → Vercel (무료, 추천)**

```bash
cd frontend
npm install -g vercel
vercel login
vercel

# 환경 변수 설정
vercel env add VITE_API_URL
# 입력: https://your-api-url.com
```

**또는 Netlify:**
```bash
npm install -g netlify-cli
netlify login
netlify deploy --prod
```

### **Backend → Railway (무료 $5 크레딧)**

1. Railway 가입: https://railway.app
2. New Project → Deploy from GitHub
3. `backend/` 폴더 선택
4. 환경 변수 설정:
   - `PORT=8000`
   - Python 버전, 의존성 자동 감지
5. 배포 완료! → URL 복사

**또는 Fly.io:**
```bash
cd backend
fly launch
fly deploy
```

---

## 📦 **빌드 (프로덕션)**

```bash
cd frontend
npm run build
# dist/ 폴더 생성 → 정적 파일 호스팅
```

---

## 🎯 **프로젝트 구조**

```
climbmate/
├── frontend/                # React (Vite)
│   ├── src/
│   │   ├── App.jsx         # 메인 앱
│   │   └── App.css         # 스타일
│   ├── vite.config.js      # PWA 설정
│   └── package.json
├── backend/                 # FastAPI
│   ├── main.py             # API 서버
│   └── Dockerfile
└── holdcheck/              # Python 모듈
    ├── preprocess.py       # YOLO
    └── clustering.py       # CLIP AI
```

---

## 🔥 **기능**

✅ 이미지 업로드  
✅ AI 홀드 감지 (YOLO)  
✅ 색상 그룹핑 (CLIP AI)  
✅ 난이도 분석 (V-등급)  
✅ 문제 유형 분석 (다이나믹/스태틱)  
✅ 모바일 최적화  
✅ PWA 지원 (설치 가능)  
✅ 반응형 디자인  

---

## 💡 **다음 단계**

1. **아이콘 추가:**
   - `frontend/public/pwa-192x192.png`
   - `frontend/public/pwa-512x512.png`

2. **환경 변수 설정:**
   ```bash
   cd frontend
   echo "VITE_API_URL=http://localhost:8000" > .env
   ```

3. **HTTPS 설정 (PWA 필수):**
   - Let's Encrypt SSL
   - Cloudflare

4. **성능 최적화:**
   - 이미지 압축
   - Lazy loading
   - Code splitting

---

## 🐛 **문제 해결**

### CORS 에러
- Backend `main.py`에서 `allow_origins` 수정
- 프로덕션: 특정 도메인만 허용

### API 연결 안됨
- Backend가 실행 중인지 확인: `docker ps`
- 포트 확인: 8000번 포트가 열려있는지
- Proxy 설정 확인: `vite.config.js`

---

## 🎉 **지금 바로 테스트!**

```bash
# Terminal 1: Backend
docker restart climbmate-app

# Terminal 2: Frontend
cd frontend && npm run dev

# 브라우저 열기
open http://localhost:3000
```

**완성되었습니다! 🚀✨**

