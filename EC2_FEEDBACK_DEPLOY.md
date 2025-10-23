# 🎨 피드백 시스템 EC2 배포

## 🚀 한 줄 명령어

```bash
ssh ubuntu@your-ec2 "cd climbmate && git pull && docker-compose restart backend frontend"
```

---

## 📋 단계별 (SSH 접속 후)

```bash
# 1. 프로젝트 디렉토리
cd climbmate

# 2. 최신 코드
git pull origin main

# 3. 백엔드 재시작
docker-compose restart backend

# 4. 프론트엔드 재빌드
cd frontend
npm run build
cd ..

# 5. 프론트엔드 재시작
docker-compose restart frontend

# 또는 한 번에
docker-compose restart
```

---

## ✅ 확인

### 백엔드 API 테스트
```bash
# 피드백 API 확인
curl http://localhost:8000/api/feedback-stats

# 응답 예시:
# {"status":"success","total_feedbacks":0,"last_updated":"2025-01-23"...}
```

### 프론트엔드 확인
```bash
# 브라우저에서
http://your-domain.com

# 분석 완료 후 "✏️ 색상 수정하기" 버튼 확인
```

---

## 🎯 피드백 수집 2가지 방법

### 방법 1: 프론트엔드 (사용자용) ✅
→ 자동으로 적용됨 (추가 작업 없음)

### 방법 2: Streamlit (관리자용)
```bash
# EC2에서 실행
cd climbmate
streamlit run holdcheck/color_feedback_ui.py --server.port 8501 --server.address 0.0.0.0

# 접속
http://your-ec2-ip:8501
```

**주의**: 보안그룹에서 8501 포트 열어야 함

---

## 🔧 문제 해결

### API 오류
```bash
# 로그 확인
docker logs -f climbmate-backend | grep "피드백"
```

### 프론트엔드 버튼 안 보임
```bash
# 빌드 다시
cd frontend
npm run build

# 캐시 클리어
docker-compose down
docker-compose up -d
```

---

끝! 🎉

