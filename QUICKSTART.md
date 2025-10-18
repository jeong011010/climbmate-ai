# 🚀 ClimbMate 빠른 시작 가이드

## ✅ 현재 실행 중!

### 🌐 **접속 주소:**

- **React Frontend**: http://localhost:3000 ✨ (메인)
- **FastAPI Docs**: http://localhost:8000/docs 📚
- **Streamlit (기존)**: http://localhost:8501 💻

---

## 📱 **사용 방법**

### 1. **이미지 업로드**
- 📸 버튼 클릭 → 클라이밍 벽 사진 선택

### 2. **벽 각도 선택 (선택사항)**
- 🏔️ 오버행 / 슬랩 / 직벽 선택

### 3. **분석 시작**
- 🔍 버튼 클릭 → AI가 자동 분석 (30초 소요)

### 4. **문제 선택**
- 🖼️ 이미지에서 홀드 클릭
- 📊 아래에 상세 분석 표시

---

## ⚡ **속도 최적화 완료**

### **개선 사항:**
- ✅ YOLO conf: 0.25 → 0.4 (확실한 홀드만)
- ✅ 마스크 정제: 5회 → 1회
- ✅ 배치 처리: 64개씩
- ✅ 모델 캐싱: YOLO + CLIP

### **예상 시간:**
- 첫 번째 이미지: ~30-40초
- 두 번째 이미지: ~15-20초 (캐싱 효과)

---

## 🎨 **새로운 UI 기능**

### ✅ **이미지 클릭으로 문제 선택**
- 홀드 클릭 → 해당 문제 자동 선택
- 클릭한 위치에서 가장 가까운 홀드 찾기

### ✅ **인라인 상세 분석**
- 이미지 바로 아래 분석 결과 표시
- 난이도 + 유형 한눈에 확인

### ✅ **색상 코딩된 이미지**
- 각 문제별 색상으로 홀드 표시
- 번호 라벨로 홀드 ID 표시

### ✅ **모바일 최적화**
- 터치 친화적 UI
- 반응형 레이아웃
- PWA 지원 (설치 가능)

---

## 📱 **모바일에서 테스트**

### **같은 Wi-Fi에서:**

1. **내 IP 확인:**
```bash
ipconfig getifaddr en0  # Mac
# 예: 192.168.0.10
```

2. **모바일 브라우저에서:**
- `http://192.168.0.10:3000`

3. **홈 화면에 추가:**
- Safari: 공유 → 홈 화면에 추가
- Chrome: 메뉴 → 홈 화면에 추가

---

## 🐛 **문제 해결**

### **Frontend 안 열림**
```bash
cd /Users/kimjazz/Desktop/project/climbmate/frontend
npm run dev
```

### **Backend API 안됨**
```bash
docker restart climbmate-app
docker logs -f climbmate-app
```

### **CORS 에러**
- 이미 해결됨 (allow_origins="*")

### **분석이 너무 느림**
- 첫 번째는 모델 로딩으로 30-40초 정상
- 두 번째부터 15-20초로 단축

---

## 🚀 **다음 단계**

### **배포하기:**

**1. Frontend → Vercel**
```bash
cd frontend
vercel
```

**2. Backend → Railway**
```bash
cd ..
railway init
railway up
```

**3. 환경 변수 설정**
```bash
# Frontend .env
VITE_API_URL=https://your-api.railway.app

# Railway
PORT=8000
```

---

## 💡 **개발 팁**

### **Frontend만 수정 시:**
- 자동 리로드됨 (Vite HMR)

### **Backend 수정 시:**
```bash
# 로컬 파일 수정 후
cat backend/main.py | docker exec -i climbmate-app tee /app/backend/main.py > /dev/null
docker restart climbmate-app
```

### **Python 코드 수정 시:**
```bash
# holdcheck/ 폴더 수정 후
docker restart climbmate-app
```

---

## 🎉 **완성!**

**지금 바로 테스트:** http://localhost:3000

**주요 기능:**
- ✅ AI 홀드 감지 (YOLO)
- ✅ 색상 그룹핑 (CLIP AI)
- ✅ 난이도 분석 (V-등급)
- ✅ 문제 유형 분석 (다이나믹/스태틱)
- ✅ 이미지 클릭 선택
- ✅ 모바일 최적화
- ✅ PWA 지원

**즐거운 클라이밍 되세요!** 🧗‍♀️✨

