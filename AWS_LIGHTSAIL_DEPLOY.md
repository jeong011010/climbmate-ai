# 🚀 AWS Lightsail로 ClimbMate 백엔드 배포하기

## 📋 준비물
- AWS 계정
- OpenAI API Key
- 신용카드 (프리티어 가능)

---

## 1️⃣ AWS Lightsail 인스턴스 생성

### 1.1 Lightsail 콘솔 접속
1. [AWS Lightsail 콘솔](https://lightsail.aws.amazon.com/) 접속
2. "인스턴스 생성" 클릭

### 1.2 인스턴스 설정
1. **인스턴스 위치**: 
   - 서울 (ap-northeast-2) 선택 ✅

2. **플랫폼 선택**:
   - "Linux/Unix" 선택

3. **블루프린트 선택**:
   - "OS 전용" 탭 클릭
   - **Ubuntu 22.04 LTS** 선택 ✅

4. **인스턴스 플랜 선택**:
   ```
   추천: $40/월 (4GB RAM, 2 vCPU, 80GB SSD)
   - 4GB RAM: AI 모델 로드에 충분
   - 2 vCPU: 추론 속도 양호
   - 4TB 트래픽: 대부분의 경우 충분
   ```

5. **인스턴스 이름**:
   - 예: `climbmate-backend`

6. **"인스턴스 생성"** 클릭

---

## 2️⃣ 방화벽 설정

### 2.1 포트 열기
1. 생성된 인스턴스 클릭
2. "네트워킹" 탭 클릭
3. "IPv4 방화벽" 섹션에서 다음 규칙 추가:

```
애플리케이션: Custom
프로토콜: TCP
포트: 8000
설명: Backend API
```

---

## 3️⃣ SSH 접속 및 초기 설정

### 3.1 SSH 접속
1. 인스턴스 페이지에서 "SSH를 사용하여 연결" 클릭
2. 또는 터미널에서:
   ```bash
   # SSH 키 다운로드 후
   ssh -i LightsailDefaultKey-ap-northeast-2.pem ubuntu@YOUR_PUBLIC_IP
   ```

### 3.2 시스템 업데이트
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

---

## 4️⃣ Docker 설치

```bash
# Docker 설치 스크립트 실행
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker ubuntu

# Docker Compose 설치
sudo apt-get install docker-compose-plugin -y

# 재접속 (권한 적용)
exit
# 다시 SSH 접속
```

### 4.1 Docker 확인
```bash
docker --version
docker compose version
```

---

## 5️⃣ Git 설치 및 코드 클론

```bash
# Git 설치
sudo apt-get install git -y

# 코드 클론
git clone https://github.com/YOUR_USERNAME/climbmate.git
cd climbmate
```

> **프라이빗 레포지토리인 경우:**
> ```bash
> # Personal Access Token 사용
> git clone https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/climbmate.git
> ```

---

## 6️⃣ 환경 변수 설정

```bash
# .env 파일 생성
nano .env
```

아래 내용 입력:
```env
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

**저장**: `Ctrl + X` → `Y` → `Enter`

---

## 7️⃣ 백엔드 배포

### 7.1 백엔드만 Docker로 실행

백엔드만 먼저 배포하려면 `docker-compose.yml` 수정:

```bash
nano docker-compose.yml
```

프론트엔드 서비스 주석처리:
```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    volumes:
      - ./backend/climbmate.db:/app/backend/climbmate.db
      - ./backend/models:/app/backend/models
    restart: unless-stopped

  # frontend:  # 주석처리
  #   build:
  #     context: ./frontend
  #     dockerfile: Dockerfile.frontend
  #   ...
```

### 7.2 Docker 이미지 빌드 및 실행

```bash
# 백엔드 빌드 및 실행
docker compose up -d --build

# 로그 확인
docker compose logs -f backend
```

**빌드 시간**: 약 5-10분 소요 (모델 다운로드 포함)

---

## 8️⃣ 배포 확인

### 8.1 헬스체크
```bash
# 로컬에서 확인
curl http://localhost:8000/api/health

# 외부에서 확인 (브라우저 또는 터미널)
curl http://YOUR_PUBLIC_IP:8000/api/health
```

**예상 응답:**
```json
{"status":"healthy","message":"ClimbMate API is running"}
```

### 8.2 API 문서 확인
브라우저에서 접속:
```
http://YOUR_PUBLIC_IP:8000/docs
```

### 8.3 GPT-4 상태 확인
```bash
curl http://YOUR_PUBLIC_IP:8000/api/gpt4-status
```

---

## 9️⃣ 프론트엔드 설정 업데이트

이제 로컬 프론트엔드에서 백엔드 주소를 변경해야 합니다:

### 9.1 프론트엔드 환경 변수 수정

```bash
# 로컬 개발 환경 (frontend/.env)
VITE_API_URL=http://YOUR_PUBLIC_IP:8000
```

### 9.2 프론트엔드 로컬 실행
```bash
cd frontend
npm install
npm run dev
```

이제 `http://localhost:3000`에서 프론트엔드에 접속하면 AWS 백엔드와 통신합니다!

---

## 🔟 운영 관리

### 10.1 로그 확인
```bash
# 실시간 로그
docker compose logs -f backend

# 최근 100줄
docker compose logs --tail=100 backend
```

### 10.2 재시작
```bash
# 백엔드 재시작
docker compose restart backend

# 완전 재시작 (이미지 재빌드)
docker compose down
docker compose up -d --build
```

### 10.3 디스크 용량 확인
```bash
df -h
docker system df
```

### 10.4 사용하지 않는 Docker 이미지 정리
```bash
docker system prune -a
```

---

## 🔒 보안 설정 (선택사항)

### 11.1 UFW 방화벽 설정
```bash
# UFW 활성화
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8000/tcp    # Backend
sudo ufw enable

# 상태 확인
sudo ufw status
```

### 11.2 자동 보안 업데이트
```bash
sudo apt-get install unattended-upgrades -y
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## 📊 모니터링

### 12.1 리소스 사용량 확인
```bash
# 시스템 리소스
htop

# Docker 컨테이너 리소스
docker stats
```

### 12.2 자동 재시작 설정 (이미 적용됨)
`docker-compose.yml`에 `restart: unless-stopped` 옵션이 설정되어 있어서 서버 재부팅 시 자동으로 시작됩니다.

---

## 💰 비용 예상

### AWS Lightsail $40/월 플랜
- 인스턴스: $40/월 (4GB RAM, 2 vCPU)
- 트래픽: 4TB 포함 (초과 시 $0.09/GB)
- 스토리지: 80GB 포함
- **총 예상 비용**: $40-50/월

### 비용 절감 팁
- 프리티어: 첫 3개월 무료 (일부 플랜)
- 스냅샷: 필요할 때만 생성
- 불필요한 인스턴스 정지

---

## 🚨 트러블슈팅

### Q1. Docker 빌드 실패
```bash
# 메모리 부족 시 Swap 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Q2. 포트 8000 접속 안 됨
- Lightsail 방화벽에서 포트 8000 열었는지 확인
- Docker 컨테이너 실행 중인지 확인: `docker compose ps`
- UFW 방화벽 확인: `sudo ufw status`

### Q3. 모델 로딩 시간 오래 걸림
- 첫 실행 시 YOLO/CLIP 모델 다운로드로 5-10분 소요
- 이후에는 캐시되어 빠름

### Q4. 메모리 부족 에러
- 4GB RAM 플랜으로 업그레이드
- 또는 Swap 메모리 추가

---

## ✅ 배포 완료 체크리스트

- [ ] AWS Lightsail 인스턴스 생성
- [ ] 방화벽 포트 8000 열기
- [ ] SSH 접속 및 Docker 설치
- [ ] 코드 클론
- [ ] 환경 변수 (.env) 설정
- [ ] `docker compose up -d --build` 실행
- [ ] 헬스체크 확인 (`http://YOUR_IP:8000/api/health`)
- [ ] API 문서 확인 (`http://YOUR_IP:8000/docs`)
- [ ] 프론트엔드에서 백엔드 연결 테스트

---

## 🎉 배포 완료!

**백엔드 접속 주소:**
- API: `http://YOUR_PUBLIC_IP:8000`
- Docs: `http://YOUR_PUBLIC_IP:8000/docs`
- Health: `http://YOUR_PUBLIC_IP:8000/api/health`

**다음 단계:**
1. 프론트엔드도 배포하려면 → Vercel이나 Netlify 사용 (무료!)
2. 도메인 연결하려면 → Route 53 또는 Cloudflare
3. HTTPS 적용하려면 → Let's Encrypt + Nginx

---

## 📞 추가 도움말

**AWS Lightsail 공식 문서:**
https://lightsail.aws.amazon.com/ls/docs

**문제 발생 시:**
```bash
# 로그 확인
docker compose logs -f backend

# 컨테이너 상태 확인
docker compose ps

# 서버 리소스 확인
htop
df -h
```

