#!/bin/bash

echo "🚀 ClimbMate 시작..."

# 환경변수 체크
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
    echo "   GPT-4 Vision 없이 실행됩니다 (규칙 기반 분석만 사용)"
    echo ""
    echo "   GPT-4를 사용하려면:"
    echo "   export OPENAI_API_KEY='sk-your-key'"
    echo ""
fi

# 백엔드 시작
echo "📡 백엔드 시작 중..."
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 잠시 대기
sleep 3

# 프론트엔드 시작
echo "🎨 프론트엔드 시작 중..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ ClimbMate가 시작되었습니다!"
echo ""
echo "   🎨 프론트엔드: http://localhost:3000"
echo "   📡 백엔드 API: http://localhost:8000"
echo "   📊 API 문서: http://localhost:8000/docs"
echo ""
echo "   종료하려면 Ctrl+C를 누르세요"
echo ""

# 종료 시그널 대기
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait

