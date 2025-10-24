#!/bin/bash

echo "🚀 Backend만 빠르게 재배포..."
echo "================================"

cd ~/climbmate-ai

echo ""
echo "📥 Step 1/3: 최신 코드 가져오기..."
git pull origin main

echo ""
echo "🔨 Step 2/3: Backend 이미지 재빌드..."
docker compose build --no-cache backend

echo ""
echo "▶️  Step 3/3: Backend 컨테이너 재시작..."
docker compose up -d --force-recreate backend

echo ""
echo "⏳ 10초 대기 (서비스 준비)..."
sleep 10

echo ""
echo "📊 Backend 로그 확인:"
docker compose logs backend | tail -30

echo ""
echo "✅ Backend 재배포 완료!"
echo "🌐 확인: https://climbmate.store"

