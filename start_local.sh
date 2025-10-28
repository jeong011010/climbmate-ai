#!/bin/bash

echo "🚀 ClimbMate 로컬 서버 시작..."
echo "================================"

cd "$(dirname "$0")"

echo ""
echo "🐳 Step 1/3: Docker 상태 확인..."
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker가 실행되지 않았습니다!"
    echo "📝 Docker Desktop을 실행해주세요:"
    echo "   open -a Docker"
    exit 1
fi
echo "✅ Docker 실행 중"

echo ""
echo "🛑 Step 2/3: 기존 컨테이너 중지..."
docker compose down

echo ""
echo "▶️  Step 3/3: 로컬 서버 시작..."
docker compose up --build -d

echo ""
echo "⏳ 서비스 준비 대기 (10초)..."
sleep 10

echo ""
echo "📊 서비스 상태:"
docker compose ps

echo ""
echo "✅ 로컬 서버 시작 완료!"
echo "================================"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:8000"
echo "📋 API Docs: http://localhost:8000/docs"
echo ""
echo "📝 로그 확인: docker compose logs -f"
echo "🛑 종료하기:  docker compose down"



