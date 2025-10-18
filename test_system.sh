#!/bin/bash

echo "🧪 ClimbMate 시스템 테스트"
echo ""

# 1. 백엔드 헬스체크
echo "1️⃣ 백엔드 헬스체크..."
curl -s http://localhost:8000/api/health | python3 -m json.tool
echo ""

# 2. 통계 확인
echo "2️⃣ 모델 통계 확인..."
curl -s http://localhost:8000/api/stats | python3 -m json.tool
echo ""

# 3. GPT-4 상태 확인
echo "3️⃣ GPT-4 Vision 상태..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "   ⚠️  OPENAI_API_KEY 없음 (규칙 기반만 사용)"
else
    echo "   ✅ OPENAI_API_KEY 설정됨"
fi
echo ""

# 4. 데이터베이스 확인
echo "4️⃣ 데이터베이스 확인..."
if [ -f "backend/climbmate.db" ]; then
    SIZE=$(ls -lh backend/climbmate.db | awk '{print $5}')
    echo "   ✅ DB 존재 (크기: $SIZE)"
else
    echo "   ⚠️  DB 없음 (첫 실행 시 자동 생성)"
fi
echo ""

# 5. 모델 파일 확인
echo "5️⃣ 학습된 모델 확인..."
if [ -f "backend/models/difficulty_model.pkl" ]; then
    echo "   ✅ 난이도 모델 존재"
else
    echo "   ⚠️  난이도 모델 없음 (50+ 피드백 후 학습)"
fi

if [ -f "backend/models/type_model.pkl" ]; then
    echo "   ✅ 유형 모델 존재"
else
    echo "   ⚠️  유형 모델 없음 (50+ 피드백 후 학습)"
fi
echo ""

echo "✅ 시스템 테스트 완료"

