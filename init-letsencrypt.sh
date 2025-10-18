#!/bin/bash

# 도메인 설정 (여기를 실제 도메인으로 변경하세요)
DOMAIN="YOUR_DOMAIN"
EMAIL="your-email@example.com"  # SSL 인증서 알림 받을 이메일

echo "🔐 Let's Encrypt SSL 인증서 초기 발급 시작..."
echo "도메인: $DOMAIN"
echo "이메일: $EMAIL"
echo ""

# 기존 인증서 삭제 (선택사항)
read -p "기존 인증서를 삭제하고 새로 발급받으시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo rm -rf ./certbot/conf/live/$DOMAIN
    sudo rm -rf ./certbot/conf/archive/$DOMAIN
    sudo rm -rf ./certbot/conf/renewal/$DOMAIN.conf
    echo "✅ 기존 인증서 삭제 완료"
fi

# 필요한 디렉토리 생성
mkdir -p ./certbot/conf
mkdir -p ./certbot/www

# Nginx 설정에서 SSL 부분 임시 제거 (초기 발급용)
echo "📝 임시 Nginx 설정 생성 중..."
cat > ./nginx/nginx-init.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    client_max_body_size 50M;
    
    server {
        listen 80;
        server_name YOUR_DOMAIN;

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 200 'SSL certificate initialization in progress...';
            add_header Content-Type text/plain;
        }
    }
}
EOF

# YOUR_DOMAIN을 실제 도메인으로 교체
sed -i.bak "s/YOUR_DOMAIN/$DOMAIN/g" ./nginx/nginx-init.conf

# 임시 Nginx 컨테이너 시작
echo "🚀 임시 Nginx 시작 중..."
docker compose down nginx 2>/dev/null || true
docker run -d --name nginx-temp \
    -p 80:80 \
    -v $(pwd)/nginx/nginx-init.conf:/etc/nginx/nginx.conf:ro \
    -v $(pwd)/certbot/www:/var/www/certbot:ro \
    nginx:alpine

# SSL 인증서 발급
echo "📜 SSL 인증서 발급 중..."
docker compose run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN

# 임시 Nginx 중지
echo "🛑 임시 Nginx 중지 중..."
docker stop nginx-temp
docker rm nginx-temp

# 실제 도메인으로 설정 파일 업데이트
echo "📝 실제 Nginx 설정 업데이트 중..."
sed -i.bak "s/YOUR_DOMAIN/$DOMAIN/g" ./nginx/nginx.conf
sed -i.bak "s/YOUR_DOMAIN/$DOMAIN/g" ./docker-compose.yml

# 인증서 권한 설정
sudo chmod -R 755 ./certbot/conf/live
sudo chmod -R 755 ./certbot/conf/archive

echo ""
echo "✅ SSL 인증서 발급 완료!"
echo ""
echo "이제 전체 스택을 시작하세요:"
echo "  docker compose up -d"
echo ""
echo "도메인 접속: https://$DOMAIN"

