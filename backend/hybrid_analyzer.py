"""
하이브리드 분석 시스템
GPT-4 Vision + 자체 ML 모델 + 규칙 기반 분석을 통합
"""

from typing import Dict, List, Optional
from gpt4_analyzer import analyze_with_gpt4_vision, get_gpt4_status, translate_and_enhance_gpt4_result
import os

async def hybrid_analyze(
    image_base64: str,
    holds_data: List[Dict],
    wall_angle: Optional[str] = None,
    rule_based_analysis: Optional[Dict] = None
) -> Dict:
    """
    하이브리드 분석 시스템
    
    우선순위:
    1. 자체 ML 모델 (50개 이상 데이터 축적 후)
    2. GPT-4 Vision (API 키 있을 때)
    3. 규칙 기반 분석 (백업)
    
    Args:
        image_base64: Base64 인코딩된 이미지
        holds_data: 홀드 데이터
        wall_angle: 벽 각도
        rule_based_analysis: 규칙 기반 분석 결과 (백업용)
    
    Returns:
        최종 분석 결과 (난이도, 유형, 신뢰도, 사용된 방법)
    """
    
    result = {
        'difficulty': {'grade': 'V?', 'confidence': 0.0},
        'type': {'primary_type': '분석 불가', 'confidence': 0.0},
        'method_used': 'none',
        'methods_tried': []
    }
    
    # 1️⃣ GPT-4 Vision 시도 (ML 모델은 현재 사용 안 함)
    gpt4_status = get_gpt4_status()
    
    print(f"🔍 GPT-4 상태 확인: {gpt4_status}")
    print(f"🔍 API 키 존재: {bool(os.getenv('OPENAI_API_KEY'))}")
    
    if gpt4_status['available'] and os.getenv('OPENAI_API_KEY'):
        print("🤖 GPT-4 Vision 사용 중...")
        
        gpt4_result = analyze_with_gpt4_vision(image_base64, holds_data, wall_angle)
        print(f"🔍 GPT-4 결과: {gpt4_result}")
        
        if gpt4_result.get('used_gpt4'):
            # 거부 응답인지 확인
            reasoning = gpt4_result.get('reasoning', '')
            if any(keyword in reasoning.lower() for keyword in ['sorry', "can't", 'cannot', 'unable']):
                print(f"   ⚠️ GPT-4 거부 응답 감지: {reasoning[:50]}...")
                result['methods_tried'].append('gpt4_refused')
                # 규칙 기반 분석으로 대체
                if rule_based_analysis:
                    print("📏 GPT-4 거부로 인한 규칙 기반 분석 사용...")
                    result['difficulty'] = {
                        'grade': rule_based_analysis.get('difficulty', {}).get('grade', 'V?'),
                        'confidence': rule_based_analysis.get('difficulty', {}).get('confidence', 0.3)
                    }
                    result['type'] = {
                        'primary_type': rule_based_analysis.get('climb_type', {}).get('primary_type', '일반'),
                        'confidence': rule_based_analysis.get('climb_type', {}).get('confidence', 0.3)
                    }
                    result['method_used'] = 'rule_based_fallback'
                    result['methods_tried'].append('rule_based_fallback')
                    result['gpt4_reasoning'] = f"GPT-4 거부: {reasoning}"
                    print(f"   ✅ 규칙 기반 결과: {result['difficulty']['grade']}, {result['type']['primary_type']}")
                    return result
            else:
                # 정상 응답 처리 - 한글 번역 및 상세 분석 추가
                enhanced_result = translate_and_enhance_gpt4_result(gpt4_result)
                
                result['difficulty'] = {
                    'grade': gpt4_result.get('difficulty', 'V?'),
                    'confidence': gpt4_result.get('confidence', 0.5)
                }
                result['type'] = {
                    'primary_type': gpt4_result.get('type', '일반'),
                    'confidence': gpt4_result.get('confidence', 0.5)
                }
                result['method_used'] = 'gpt4_vision'
                result['methods_tried'].append('gpt4_vision')
                result['gpt4_reasoning'] = enhanced_result['detailed_analysis']
                result['gpt4_movements'] = enhanced_result.get('movements', [])
                result['gpt4_challenges'] = enhanced_result.get('challenges', [])
                result['gpt4_tips'] = enhanced_result.get('tips', [])
                
                print(f"   ✅ GPT-4 결과: {gpt4_result.get('difficulty')}, {gpt4_result.get('type')}")
                print(f"   ✅ GPT-4 상세 분석 생성 완료")
                return result
        else:
            result['methods_tried'].append('gpt4_failed')
            print(f"   ⚠️ GPT-4 사용 실패: {gpt4_result}")
    else:
        result['methods_tried'].append('gpt4_unavailable')
        print("   ⚠️ GPT-4 Vision 사용 불가 (API 키 없음)")
    
    # 3️⃣ 규칙 기반 분석 (백업)
    if rule_based_analysis:
        print("📏 규칙 기반 분석 사용...")
        
        result['difficulty'] = {
            'grade': rule_based_analysis.get('difficulty', {}).get('grade', 'V?'),
            'confidence': rule_based_analysis.get('difficulty', {}).get('confidence', 0.3)
        }
        result['type'] = {
            'primary_type': rule_based_analysis.get('climb_type', {}).get('primary_type', '일반'),
            'confidence': rule_based_analysis.get('climb_type', {}).get('confidence', 0.3)
        }
        result['method_used'] = 'rule_based'
        result['methods_tried'].append('rule_based')
        
        print(f"   ✅ 규칙 기반 결과: {result['difficulty']['grade']}, {result['type']['primary_type']}")
        print(f"   ✅ 규칙 기반 analysis_method: {result['method_used']}")
    
    return result

def get_analysis_method_stats() -> Dict:
    """분석 방법별 통계"""
    gpt4_stat = get_gpt4_status()
    
    return {
        'ml_model_available': False,  # ML 모델은 현재 사용 안 함
        'gpt4_available': gpt4_stat['available'] and bool(os.getenv('OPENAI_API_KEY')),
        'rule_based_available': True,
        'recommended_method': (
            'gpt4_vision' if gpt4_stat['available'] and os.getenv('OPENAI_API_KEY') else
            'rule_based'
        )
    }

