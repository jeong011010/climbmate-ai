# 🚀 ClimbMate 배포 가이드

## 📋 배포 아키텍처

```
┌─────────────────┐
│   Frontend      │ (Port 3000)
│   React + PWA   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    Backend      │ (Port 8000)
│  FastAPI + AI   │
│  - YOLOv8       │
│  - CLIP         │
│  - GPT-4 API    │
└─────────────────┘
```

**권장 서버 스펙:**
- **CPU**: 2 vCPU 이상
- **RAM**: 4GB 이상
- **Storage**: 20GB 이상
- **OS**: Ubuntu 22.04 LTS

---

## 🛠️ 1. 사전 준비

### 1.1 필수 소프트웨어 설치

```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose 설치
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Git 설치
sudo apt-get install git
```

### 1.2 환경 변수 설정

```bash
# 프로젝트 루트에 .env 파일 생성
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key-here
EOF
```

---

## 🚀 2. 배포 실행

### 2.1 코드 클론

```bash
# 서버에 접속 후
git clone <your-repository-url>
cd climbmate
```

### 2.2 Docker 이미지 빌드 및 실행

```bash
# 백그라운드로 실행
sudo docker-compose up -d --build

# 로그 확인
sudo docker-compose logs -f
```

### 2.3 상태 확인

```bash
# 컨테이너 상태 확인
sudo docker-compose ps

# 헬스체크
curl http://localhost:8000/api/health
curl http://localhost:3000
```

---

## 🌐 3. 도메인 및 SSL 설정 (선택사항)

### 3.1 Nginx 설치

```bash
sudo apt-get install nginx certbot python3-certbot-nginx
```

### 3.2 Nginx 설정

```bash
# /etc/nginx/sites-available/climbmate 파일 생성
sudo nano /etc/nginx/sites-available/climbmate
```

```nginx
# Frontend
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# Backend API
server {
    listen 80;
    server_name api.your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        
        # 이미지 업로드를 위한 타임아웃 설정
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        client_max_body_size 50M;
    }
}
```

### 3.3 SSL 인증서 발급

```bash
# 심볼릭 링크 생성
sudo ln -s /etc/nginx/sites-available/climbmate /etc/nginx/sites-enabled/

# Nginx 재시작
sudo systemctl restart nginx

# Let's Encrypt SSL 인증서 발급
sudo certbot --nginx -d your-domain.com -d api.your-domain.com
```

---

## 🔧 4. 운영 관리

### 4.1 서비스 시작/중지

```bash
# 시작
sudo docker-compose up -d

# 중지
sudo docker-compose down

# 재시작
sudo docker-compose restart

# 특정 서비스만 재시작
sudo docker-compose restart backend
sudo docker-compose restart frontend
```

### 4.2 로그 확인

```bash
# 전체 로그
sudo docker-compose logs -f

# 백엔드 로그만
sudo docker-compose logs -f backend

# 최근 100줄만
sudo docker-compose logs --tail=100 backend
```

### 4.3 업데이트 배포

```bash
# 코드 업데이트
git pull origin main

# 재빌드 및 재시작
sudo docker-compose up -d --build

# 사용하지 않는 이미지 정리
sudo docker system prune -a
```

### 4.4 데이터베이스 백업

```bash
# SQLite DB 백업
cp backend/climbmate.db backend/climbmate.db.backup.$(date +%Y%m%d)

# 주기적 백업 크론잡 설정
crontab -e
# 매일 새벽 3시에 백업
0 3 * * * cp /path/to/climbmate/backend/climbmate.db /path/to/backups/climbmate.db.$(date +\%Y\%m\%d)
```

---

## 📊 5. 모니터링

### 5.1 리소스 사용량 확인

```bash
# Docker 컨테이너 리소스 사용량
sudo docker stats

# 시스템 리소스
htop
df -h
```

### 5.2 헬스체크 설정

```bash
# 크론잡으로 헬스체크 (매 5분)
*/5 * * * * curl -f http://localhost:8000/api/health || echo "Backend is down!" | mail -s "ClimbMate Alert" your-email@example.com
```

---

## 🚨 6. 트러블슈팅

### 6.1 메모리 부족

```bash
# Swap 메모리 추가 (2GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 6.2 포트 충돌

```bash
# 포트 사용 중인 프로세스 확인
sudo lsof -i :8000
sudo lsof -i :3000

# 프로세스 종료
sudo kill -9 <PID>
```

### 6.3 Docker 이미지 재빌드

```bash
# 캐시 없이 완전 재빌드
sudo docker-compose build --no-cache
sudo docker-compose up -d
```

---

## 💰 7. 예상 비용 (월별)

### AWS EC2 (t3.medium)
- 인스턴스: $30/월
- 스토리지 (20GB): $2/월
- 트래픽 (1TB): $90/월
- **총 예상 비용**: ~$122/월

### DigitalOcean (Basic Droplet)
- 인스턴스 (2 vCPU, 4GB): $24/월
- 트래픽 (4TB 포함): $0
- **총 예상 비용**: ~$24/월 ✅ (추천)

### Google Cloud (e2-medium)
- 인스턴스: $25/월
- 스토리지: $2/월
- 트래픽: 변동
- **총 예상 비용**: ~$30-50/월

---

## 📱 8. 프론트엔드 환경 변수 수정

배포 시 프론트엔드의 API URL을 수정해야 합니다:

```bash
# frontend/.env.production 파일 생성
VITE_API_URL=https://api.your-domain.com
```

또는 docker-compose.yml에서 수정:
```yaml
frontend:
  environment:
    - VITE_API_URL=https://api.your-domain.com
```

---

## ✅ 배포 완료 체크리스트

- [ ] Docker 및 Docker Compose 설치
- [ ] OPENAI_API_KEY 환경 변수 설정
- [ ] `docker-compose up -d --build` 실행
- [ ] 헬스체크 확인 (`curl http://localhost:8000/api/health`)
- [ ] 프론트엔드 접속 확인 (`http://localhost:3000`)
- [ ] 도메인 연결 (선택사항)
- [ ] SSL 인증서 설치 (선택사항)
- [ ] 데이터베이스 백업 크론잡 설정
- [ ] 모니터링 설정
- [ ] 방화벽 설정 (포트 80, 443, 8000, 3000 허용)

---

## 🎉 배포 완료!

이제 ClimbMate가 프로덕션 환경에서 실행됩니다!

**접속 URL:**
- Frontend: `http://your-server-ip:3000` (또는 `https://your-domain.com`)
- Backend API: `http://your-server-ip:8000` (또는 `https://api.your-domain.com`)
- API Docs: `http://your-server-ip:8000/docs`

---

## 📞 지원

문제가 발생하면 로그를 확인하세요:
```bash
sudo docker-compose logs -f --tail=100
```

