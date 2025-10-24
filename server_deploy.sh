#!/bin/bash

echo "🚀 ClimbMate 서버 배포 시작..."
echo "================================"

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
echo "📝 로그 확인: docker compose logs -f backend celery-worker"
echo "🌐 접속: https://climbmate.store"
