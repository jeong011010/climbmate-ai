#!/bin/bash

echo "🚀 ClimbMate 배포 시작..."
echo "================================"

# 서버 정보 (수정 필요)
SERVER_USER="ubuntu"
SERVER_HOST="ip-172-31-12-99"  # 또는 실제 IP 주소
SERVER_PATH="~/climbmate-ai"

echo "📡 서버 연결 중: ${SERVER_USER}@${SERVER_HOST}"

# SSH로 서버에서 실행
ssh ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
cd ~/climbmate-ai

echo ""
echo "📥 Step 1/6: 최신 코드 가져오기..."
git pull origin main

echo ""
echo "🧹 Step 2/6: Docker 정리 (디스크 공간 확보)..."
docker system prune -f
docker builder prune -a -f

echo ""
echo "🛑 Step 3/6: 기존 컨테이너 중지..."
docker compose down

echo ""
echo "🔨 Step 4/6: 이미지 재빌드..."
docker compose build --no-cache backend celery-worker frontend

echo ""
echo "▶️  Step 5/6: 컨테이너 시작..."
docker compose up -d

echo ""
echo "⏳ Step 6/6: 서비스 준비 대기 (20초)..."
sleep 20

echo ""
echo "📊 배포 상태 확인:"
docker compose ps

echo ""
echo "💾 디스크 상태:"
df -h | grep "/dev/root"

echo ""
echo "🐳 Docker 상태:"
docker system df

echo ""
echo "✅ 배포 완료!"
echo "================================"
echo "📝 로그 확인: docker compose logs -f"
echo "🌐 접속: https://climbmate.store"
ENDSSH

echo ""
echo "✅ 배포 스크립트 실행 완료!"
