#!/usr/bin/env python3
"""GPT-4 분석 로컬 테스트"""
import os
import sys
import asyncio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from backend.gpt4_analyzer import analyze_with_gpt4_vision, get_gpt4_status

async def test_gpt4():
    # 1. GPT-4 상태 확인
    status = get_gpt4_status()
    print("🔍 GPT-4 상태:")
    print(f"   사용 가능: {status['available']}")
    print(f"   API 키 설정: {status['api_key_set']}")
    print(f"   모델: {status.get('model', 'N/A')}")
    
    if not status['available'] or not status['api_key_set']:
        print("\n❌ GPT-4 사용 불가 (API 키 확인 필요)")
        return
    
    # 2. 더미 데이터로 프롬프트 생성 테스트
    print("\n📝 프롬프트 생성 테스트...")
    
    dummy_holds = [
        {"id": 0, "color_name": "blue", "center": [100, 200], "area": 500},
        {"id": 1, "color_name": "blue", "center": [150, 300], "area": 450},
        {"id": 2, "color_name": "blue", "center": [200, 400], "area": 480},
    ]
    
    dummy_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # 1x1 투명 PNG
    
    print(f"   홀드 수: {len(dummy_holds)}")
    print(f"   이미지: base64 (길이={len(dummy_image)})")
    
    # 3. 실제 GPT-4 호출 (간단한 테스트)
    print("\n🚀 GPT-4 호출 중...")
    try:
        result = await analyze_with_gpt4_vision(
            image_base64=dummy_image,
            holds_info=dummy_holds,
            wall_angle="face",
            rule_based={"difficulty": "V3", "type": "일반"}
        )
        
        print("\n✅ GPT-4 응답 수신")
        print(f"   난이도: {result.get('difficulty', 'N/A')}")
        print(f"   타입: {result.get('type', 'N/A')}")
        print(f"   신뢰도: {result.get('confidence', 0):.2f}")
        print(f"   루트 스텝 수: {len(result.get('route', []))}")
        
        if result.get('reasoning'):
            print(f"   분석 (처음 100자): {result['reasoning'][:100]}...")
        
        # 거부 응답 체크
        if 'sorry' in str(result.get('raw_response', '')).lower():
            print("\n⚠️  GPT-4 거부 응답 감지!")
            print(f"   원본: {result.get('raw_response', '')[:200]}")
        else:
            print("\n✅ GPT-4 정상 분석 완료")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test_gpt4())

