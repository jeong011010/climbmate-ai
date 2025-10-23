# 🚀 EC2 업데이트 명령어

## 📦 새로운 색상 분류 시스템 배포

```bash
# 1. EC2 접속
ssh -i "your-key.pem" ubuntu@your-ec2-ip

# 2. 프로젝트 디렉토리로 이동
cd /path/to/climbmate

# 3. 최신 코드 가져오기
git pull origin main

# 4. 필요한 패키지 설치 (Streamlit UI용, 선택사항)
# 이미 설치되어 있으면 건너뛰기
pip install streamlit

# 5. 백엔드 재시작 (Docker 사용 시)
docker-compose restart backend

# 또는 직접 실행 시
sudo systemctl restart climbmate-backend

# 6. 완료! 🎉
```

---

## ⚡ 빠른 적용 (한 줄 명령어)

```bash
ssh ubuntu@your-ec2 "cd climbmate && git pull && docker-compose restart backend"
```

---

## 🎨 피드백 UI 실행 (선택사항)

색상 범위를 조정하고 싶으면:

```bash
# EC2에서
cd /path/to/climbmate
streamlit run holdcheck/color_feedback_ui.py --server.port 8501

# 브라우저에서
http://your-ec2-ip:8501
```

**보안그룹 설정**: 8501 포트 열어야 함

---

## 📊 적용 확인

```bash
# 로그 확인
docker logs -f climbmate-backend

# 또는
tail -f /var/log/climbmate/backend.log
```

새로운 로그에서 이런 메시지 확인:
```
⚡ 룰 기반 색상 클러스터링 시작 (CLIP 없음, 초고속)
✅ 룰 기반 클러스터링 완료 (⚡ 0.13초)
```

---

## 🔧 트러블슈팅

### 문제 1: 패키지 없음
```bash
pip install scikit-learn numpy opencv-python
```

### 문제 2: 권한 오류
```bash
sudo chown -R ubuntu:ubuntu /path/to/climbmate
```

### 문제 3: 포트 충돌
```bash
# 기존 프로세스 종료
sudo lsof -ti:8000 | xargs kill -9
```

---

## 💡 성능 확인

배포 전후 비교:

```bash
# 분석 속도 측정
curl -X POST http://your-ec2-ip:8000/analyze \
  -F "file=@test.jpg" \
  --trace-time
```

**예상 결과:**
- 기존: ~5-8초
- 새로운 방식: ~1-2초 (CLIP 제외)

---

## 📝 주의사항

1. **CLIP 사용 안 함**: 기본적으로 룰 기반 사용
2. **학습 데이터**: `color_ranges.json` 자동 생성됨
3. **백업**: 기존 설정 백업 권장
   ```bash
   cp holdcheck/color_ranges.json holdcheck/color_ranges.json.backup
   ```

---

끝! 🎉

