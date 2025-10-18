# 🔄 기존 EC2 서버 재사용 가이드

## 📋 1단계: 서버 스펙 확인

### 1.1 필수 요구사항
현재 ClimbMate 백엔드가 필요로 하는 스펙:
- **최소**: 2GB RAM, 1 vCPU
- **권장**: 4GB RAM, 2 vCPU
- **저장공간**: 최소 10GB 여유 공간
- **OS**: Ubuntu 20.04 이상 (또는 다른 Linux)

### 1.2 스펙 확인 방법

SSH 접속 후:
```bash
# CPU 확인
nproc
lscpu | grep "Model name"

# 메모리 확인
free -h

# 디스크 확인
df -h

# OS 확인
cat /etc/os-release
```

**판단 기준:**
- RAM 2GB 미만 → ⚠️ Swap 메모리 추가 필요
- RAM 4GB 이상 → ✅ 바로 사용 가능!

---

## 🧹 2단계: 기존 스프링부트 정리

### 2.1 실행 중인 Java 프로세스 확인
```bash
# Java 프로세스 확인
ps aux | grep java

# 포트 사용 확인
sudo lsof -i :8080
sudo lsof -i :8000
```

### 2.2 스프링부트 중지

**방법 1: Systemd 서비스인 경우**
```bash
# 서비스 확인
sudo systemctl list-units --type=service | grep -i spring
sudo systemctl list-units --type=service | grep -i climb

# 서비스 중지 및 비활성화
sudo systemctl stop climbmate
sudo systemctl disable climbmate
```

**방법 2: 수동 실행 중인 경우**
```bash
# 프로세스 ID 확인
ps aux | grep java

# 프로세스 종료
sudo kill -9 <PID>
```

**방법 3: Docker 컨테이너인 경우**
```bash
# 실행 중인 컨테이너 확인
docker ps

# 컨테이너 중지 및 삭제
docker stop <container_name>
docker rm <container_name>
```

### 2.3 스프링부트 파일 정리 (선택사항)
```bash
# 백업 디렉토리 생성
mkdir -p ~/old_projects/climbmate_spring_backup
cd ~

# 기존 스프링부트 프로젝트 백업
mv climbmate ~/old_projects/climbmate_spring_backup/

# 또는 그냥 삭제 (확실한 경우)
rm -rf ~/climbmate
```

---

## 🐳 3단계: Docker 설치 (없는 경우)

### 3.1 Docker 설치 여부 확인
```bash
docker --version
docker compose version
```

### 3.2 Docker 설치
설치가 안 되어 있다면:
```bash
# Docker 설치 스크립트
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# Docker Compose 플러그인 설치
sudo apt-get install docker-compose-plugin -y

# 재접속 (권한 적용)
exit
# 다시 SSH 접속
```

---

## 🚀 4단계: 새 ClimbMate 백엔드 배포

### 4.1 기존 프로젝트 백업 (있다면)
```bash
# 홈 디렉토리로 이동
cd ~

# 기존 climbmate 폴더가 있다면 백업
if [ -d "climbmate" ]; then
    mv climbmate climbmate_old_$(date +%Y%m%d)
fi
```

### 4.2 새 코드 클론
```bash
# Git 설치 (없는 경우)
sudo apt-get install git -y

# 새 코드 클론
git clone https://github.com/YOUR_USERNAME/climbmate.git
cd climbmate
```

### 4.3 환경 변수 설정
```bash
# .env 파일 생성
nano .env
```

아래 내용 입력:
```env
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

**저장**: `Ctrl + X` → `Y` → `Enter`

### 4.4 방화벽 설정 (AWS 콘솔)

**AWS 콘솔에서:**
1. EC2 → 인스턴스 → 해당 인스턴스 클릭
2. "보안" 탭 → "보안 그룹" 클릭
3. "인바운드 규칙 편집" 클릭
4. 다음 규칙 추가:

```
유형: Custom TCP
프로토콜: TCP
포트 범위: 8000
소스: 0.0.0.0/0 (또는 Anywhere-IPv4)
설명: ClimbMate Backend API
```

### 4.5 배포 실행
```bash
# Docker Compose로 백엔드 실행
docker compose up -d --build

# 로그 확인
docker compose logs -f backend
```

**빌드 시간**: 첫 실행 시 5-10분 소요

---

## 🔍 5단계: 배포 확인

### 5.1 헬스체크
```bash
# 로컬에서 확인
curl http://localhost:8000/api/health

# 외부에서 확인
curl http://YOUR_EC2_PUBLIC_IP:8000/api/health
```

**예상 응답:**
```json
{"status":"healthy","message":"ClimbMate API is running"}
```

### 5.2 API 문서 확인
브라우저에서:
```
http://YOUR_EC2_PUBLIC_IP:8000/docs
```

### 5.3 GPT-4 상태 확인
```bash
curl http://YOUR_EC2_PUBLIC_IP:8000/api/gpt4-status
```

---

## ⚙️ 6단계: 자동 시작 설정 (선택사항)

서버 재부팅 시 자동으로 Docker 컨테이너가 시작되도록 설정:

### 6.1 Docker 자동 시작
```bash
# Docker 서비스 자동 시작 활성화
sudo systemctl enable docker

# 확인
sudo systemctl is-enabled docker
```

### 6.2 재부팅 테스트
```bash
# 재부팅
sudo reboot

# 재접속 후 확인
docker compose ps
curl http://localhost:8000/api/health
```

`docker-compose.yml`에 `restart: unless-stopped` 옵션이 있어서 자동으로 재시작됩니다.

---

## 🔄 7단계: 포트 충돌 해결 (필요 시)

### 7.1 기존 스프링부트가 8080 사용 중
```bash
# 8080 포트 사용 프로세스 확인
sudo lsof -i :8080

# 종료
sudo kill -9 <PID>
```

### 7.2 다른 서비스가 8000 사용 중
```bash
# 8000 포트 사용 프로세스 확인
sudo lsof -i :8000

# 종료
sudo kill -9 <PID>
```

### 7.3 포트 변경 (최후의 수단)
`docker-compose.yml` 수정:
```yaml
backend:
  ports:
    - "8001:8000"  # 호스트 포트를 8001로 변경
```

---

## 💰 비용 절감 팁

### 8.1 기존 EC2 타입 확인
```bash
# 인스턴스 메타데이터에서 타입 확인
curl http://169.254.169.254/latest/meta-data/instance-type
```

### 8.2 인스턴스 타입 업그레이드/다운그레이드

**AWS 콘솔에서:**
1. 인스턴스 중지
2. "작업" → "인스턴스 설정" → "인스턴스 유형 변경"
3. 적절한 타입 선택:
   - **t3.medium** (2 vCPU, 4GB) → 추천!
   - **t3a.medium** (2 vCPU, 4GB) → 더 저렴
4. 인스턴스 시작

---

## 🚨 트러블슈팅

### Q1. 메모리 부족 에러
```bash
# Swap 메모리 추가 (2GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 확인
free -h
```

### Q2. Docker 권한 에러
```bash
# docker 그룹에 사용자 추가
sudo usermod -aG docker $USER

# 재접속 필요
exit
```

### Q3. 디스크 용량 부족
```bash
# 불필요한 Docker 이미지/컨테이너 삭제
docker system prune -a

# 로그 파일 정리
sudo journalctl --vacuum-time=7d
```

### Q4. Java와 Python 충돌?
걱정 없습니다! Docker로 실행되므로:
- ✅ Java와 Python이 서로 격리됨
- ✅ 포트만 겹치지 않으면 OK
- ✅ 호스트 시스템에 Python 설치 불필요

---

## 📊 비교: 스프링부트 vs FastAPI+Docker

| 항목 | 스프링부트 | FastAPI+Docker |
|------|-----------|----------------|
| 메모리 사용 | ~500MB | ~1-2GB (AI 모델 포함) |
| 시작 시간 | 10-30초 | 5초 |
| AI 모델 | ❌ 통합 어려움 | ✅ 완벽 통합 |
| 배포 | JAR 파일 | Docker 컨테이너 |
| 포트 | 8080 | 8000 |

**결론**: 같은 서버에서 병렬 실행 가능! (포트만 다르게)

---

## ✅ 재사용 체크리스트

- [ ] 서버 스펙 확인 (최소 2GB RAM)
- [ ] 기존 스프링부트 중지
- [ ] Docker 설치 (없으면)
- [ ] 코드 클론
- [ ] 환경 변수 (.env) 설정
- [ ] AWS 보안 그룹에서 포트 8000 열기
- [ ] `docker compose up -d --build` 실행
- [ ] 헬스체크 확인
- [ ] 프론트엔드에서 연결 테스트

---

## 🎉 완료!

**기존 EC2 서버 재사용 완료!**

**새 백엔드 주소:**
- API: `http://YOUR_EC2_IP:8000`
- Docs: `http://YOUR_EC2_IP:8000/docs`
- Health: `http://YOUR_EC2_IP:8000/api/health`

**다음 단계:**
1. 프론트엔드에서 백엔드 URL 업데이트
2. 테스트
3. 기존 스프링부트는 백업 보관 또는 삭제

---

## 💡 추가 팁

### 스프링부트도 계속 사용하고 싶다면?
```bash
# 스프링부트: 8080 포트
# FastAPI: 8000 포트
# 둘 다 병렬 실행 가능!

# Nginx로 리버스 프록시 설정
# /api/v1 → FastAPI (AI 분석)
# /api/v2 → 스프링부트 (기존 기능)
```

이렇게 하면 기존 시스템을 유지하면서 새로운 AI 기능을 추가할 수 있습니다!

