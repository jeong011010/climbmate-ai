import os
import base64
from typing import Dict, List, Optional
import json
import re
import asyncio
from PIL import Image
import io

# OpenAI 클라이언트 (환경변수에서 API 키 로드)
try:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    GPT4_AVAILABLE = True
except:
    GPT4_AVAILABLE = False
    print("⚠️ OpenAI API 사용 불가 (환경변수 OPENAI_API_KEY 필요)")

def resize_image_for_gpt4(image_base64: str, max_size: int = 512) -> str:
    """
    🚀 GPT-4 전송용 이미지 크기 축소 (속도 최적화)
    - 큰 이미지를 작게 만들어 API 응답 속도 향상
    - 홀드 인식에는 고해상도가 필요하지 않음
    """
    try:
        # Base64 디코딩
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data))
        
        # 이미 작으면 그대로 반환
        if max(img.size) <= max_size:
            return image_base64
        
        # 비율 유지하며 리사이즈
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Base64로 재인코딩
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ 이미지 리사이즈 실패 (원본 사용): {e}")
        return image_base64

async def analyze_with_gpt4_vision(
    image_base64: str,
    holds_info: List[Dict],
    wall_angle: Optional[str] = None
) -> Dict:
    """
    GPT-4 Vision으로 클라이밍 문제 분석
    
    Args:
        image_base64: Base64 인코딩된 이미지
        holds_info: 홀드 정보 리스트
        wall_angle: 벽 각도 (overhang/slab/face)
    
    Returns:
        {
            'difficulty': 'V3',
            'type': '다이나믹',
            'confidence': 0.75,
            'reasoning': '...',
            'used_gpt4': True
        }
    """
    
    if not GPT4_AVAILABLE:
        return {
            'difficulty': 'V?',
            'type': '분석 불가',
            'confidence': 0.0,
            'reasoning': 'GPT-4 API 사용 불가',
            'used_gpt4': False
        }
    
    try:
        # 홀드 정보 요약
        num_holds = len(holds_info)
        color_groups = {}
        for hold in holds_info:
            color = hold.get('color_name', 'unknown')
            color_groups[color] = color_groups.get(color, 0) + 1
        
        # 평균 크기 계산
        areas = [h.get('area', 0) for h in holds_info]
        avg_area = sum(areas) / len(areas) if areas else 0
        
        # 거리 계산
        import numpy as np
        centers = [h.get('center', [0, 0]) for h in holds_info]
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                distances.append(dist)
        max_dist = max(distances) if distances else 0
        
        # 🚀 이미지 크기 축소 (API 속도 향상)
        image_base64_optimized = resize_image_for_gpt4(image_base64, max_size=512)
        
        # 프롬프트 구성
        wall_info = f"\n벽 각도: {wall_angle}" if wall_angle else ""
        
        prompt = f"""이 클라이밍 벽 이미지를 분석해주세요. {num_holds}개의 홀드가 있습니다.

다음을 제공해주세요:
1. 난이도 (V0-V10)
2. 스타일 (dynamic/static/crimp/sloper/balance)
3. 간단한 분석

JSON 형식으로 응답해주세요:
{{
  "difficulty": "V3",
  "type": "dynamic",
  "confidence": 0.75,
  "reasoning": "홀드 간격이 넓어서 다이나믹한 움직임이 필요합니다",
  "movements": ["큰 리치", "다이나믹 점프"],
  "challenges": ["밸런스 유지"],
  "tips": ["코어를 활용하세요", "모멘텀을 사용하세요"]
}}"""

        # 🚀 GPT-4o-mini 사용 (2-3배 빠름 + 저렴함)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # gpt-4o에서 변경
            messages=[{
                "role": "system",
                "content": "You are a climbing coach. Analyze bouldering routes and respond in JSON format only."
            }, {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64_optimized}",
                            "detail": "low"
                        }
                    }
                ]
            }],
            max_tokens=150,
            temperature=0.3,
            timeout=8  # 타임아웃 단축
        )
        
        # 응답 파싱
        content = response.choices[0].message.content
        print(f"📝 GPT-4 응답: {content}")
        
        # JSON 추출
        try:
            # JSON 블록 찾기
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # JSON이 없으면 텍스트 파싱
                result = parse_text_response(content)
        except:
            result = parse_text_response(content)
        
        result['used_gpt4'] = True
        result['raw_response'] = content
        
        return result
        
    except Exception as e:
        print(f"❌ GPT-4 Vision 분석 실패: {e}")
        return {
            'difficulty': 'V?',
            'type': '분석 실패',
            'confidence': 0.0,
            'reasoning': str(e),
            'used_gpt4': False
        }

def translate_and_enhance_gpt4_result(gpt4_result):
    """GPT-4 결과를 한글로 번역하고 상세 분석 추가"""
    
    # 기본 번역 매핑
    difficulty_map = {
        'V0': 'V0 (초급)', 'V1': 'V1 (초급)', 'V2': 'V2 (초급)',
        'V3': 'V3 (중급)', 'V4': 'V4 (중급)', 'V5': 'V5 (중급)',
        'V6': 'V6 (고급)', 'V7': 'V7 (고급)', 'V8': 'V8 (고급)',
        'V9': 'V9 (전문가)', 'V10': 'V10 (전문가)', 'V?': 'V? (미분석)'
    }
    
    type_map = {
        'dynamic': '다이나믹',
        'static': '스태틱', 
        'crimp': '크림프',
        'sloper': '슬로퍼',
        'traverse': '트래버스',
        'campusing': '캠퍼싱',
        'balance': '밸런스',
        'lunge': '런지',
        'dyno': '다이노',
        '일반': '일반'
    }
    
    # 기본 결과
    result = {
        'difficulty': difficulty_map.get(gpt4_result.get('difficulty', 'V?'), 'V? (미분석)'),
        'type': type_map.get(gpt4_result.get('type', '일반'), '일반'),
        'confidence': gpt4_result.get('confidence', 0.0),
        'reasoning': gpt4_result.get('reasoning', ''),
        'movements': gpt4_result.get('movements', []),
        'challenges': gpt4_result.get('challenges', []),
        'tips': gpt4_result.get('tips', [])
    }
    
    # 상세 분석 생성
    detailed_analysis = generate_detailed_analysis(gpt4_result)
    result['detailed_analysis'] = detailed_analysis
    
    return result

def generate_detailed_analysis(gpt4_result):
    """GPT-4 결과를 바탕으로 간결한 한글 분석 생성"""
    
    difficulty = gpt4_result.get('difficulty', 'V?')
    climb_type = gpt4_result.get('type', '일반')
    reasoning = gpt4_result.get('reasoning', '')
    movements = gpt4_result.get('movements', [])
    challenges = gpt4_result.get('challenges', [])
    tips = gpt4_result.get('tips', [])
    
    analysis_parts = []
    
    # 1. 난이도 분석 (간소화)
    if difficulty.startswith('V'):
        v_num = difficulty[1:]
        if v_num.isdigit():
            v_level = int(v_num)
            if v_level <= 2:
                analysis_parts.append(f"🟢 **초급** (V{v_level}) - 기본 기술로 해결 가능")
            elif v_level <= 5:
                analysis_parts.append(f"🟡 **중급** (V{v_level}) - 기술과 체력 필요")
            else:
                analysis_parts.append(f"🔴 **고급** (V{v_level}) - 높은 수준의 기술과 체력 요구")
    
    # 2. 클라이밍 유형 분석 (간소화)
    type_analysis = {
        'dynamic': "💥 **다이나믹**: 폭발적 움직임과 점프 필요",
        'static': "🧘 **스태틱**: 신중한 움직임과 균형 중요",
        'crimp': "🤏 **크림프**: 작은 홀드, 손가락 힘 중요",
        'sloper': "🤚 **슬로퍼**: 둥근 홀드, 접촉력과 균형",
        'traverse': "↔️ **트래버스**: 옆으로 이동, 지구력 중요",
        'balance': "⚖️ **밸런스**: 균형 유지, 코어 근력 필요"
    }
    
    if climb_type in type_analysis:
        analysis_parts.append(type_analysis[climb_type])
    
    # 3. 필요한 동작 분석 (한글화)
    if movements:
        korean_movements = translate_movements(movements)
        analysis_parts.append(f"🎯 **필요 동작**: {', '.join(korean_movements)}")
    
    # 4. 주요 도전과제 (한글화)
    if challenges:
        korean_challenges = translate_challenges(challenges)
        analysis_parts.append(f"⚠️ **도전과제**: {', '.join(korean_challenges)}")
    
    # 5. 실용적인 팁 (한글화)
    if tips:
        korean_tips = translate_tips(tips)
        analysis_parts.append(f"💡 **팁**: {', '.join(korean_tips)}")
    
    # 6. GPT-4 원본 분석 (한글화)
    if reasoning:
        korean_reasoning = translate_reasoning(reasoning)
        analysis_parts.append(f"🤖 **분석**: {korean_reasoning}")
    
    return "\n".join(analysis_parts)

def translate_movements(movements):
    """동작을 한국어로 번역"""
    movement_map = {
        'dynamic moves': '다이나믹 움직임',
        'balance': '균형',
        'coordination': '협응',
        'crimping': '크림핑',
        'static moves': '스태틱 움직임',
        'power': '파워',
        'precision': '정밀함',
        'reach': '리치',
        'footwork': '풋워크',
        'momentum': '모멘텀'
    }
    return [movement_map.get(move.lower(), move) for move in movements]

def translate_challenges(challenges):
    """도전과제를 한국어로 번역"""
    challenge_map = {
        'reach': '리치',
        'precision': '정밀함',
        'power': '파워',
        'hold type transitions': '홀드 전환',
        'balance': '균형',
        'coordination': '협응',
        'endurance': '지구력',
        'flexibility': '유연성',
        'strength': '근력'
    }
    return [challenge_map.get(challenge.lower(), challenge) for challenge in challenges]

def translate_tips(tips):
    """팁을 한국어로 번역"""
    tip_map = {
        'use momentum effectively': '모멘텀 활용',
        'focus on precise foot placements': '정확한 발 배치',
        'commit to dynamic moves': '다이나믹 움직임에 집중',
        'adapt grip quickly': '빠른 그립 전환',
        'maintain balance': '균형 유지',
        'use core strength': '코어 근력 활용',
        'breathe steadily': '안정적인 호흡',
        'plan your route': '루트 계획',
        'stay relaxed': '긴장 완화'
    }
    return [tip_map.get(tip.lower(), tip) for tip in tips]

def translate_reasoning(reasoning):
    """분석 내용을 한국어로 번역"""
    # 간단한 번역 매핑
    translations = {
        'varied hold types': '다양한 홀드 유형',
        'wide spacing': '넓은 간격',
        'dynamic movement': '다이나믹 움직임',
        'precise footwork': '정확한 풋워크',
        'intermediate level': '중급 수준',
        'advanced level': '고급 수준',
        'beginner level': '초급 수준',
        'requires': '요구',
        'needs': '필요',
        'challenging': '도전적',
        'difficult': '어려운',
        'moderate': '보통',
        'easy': '쉬운'
    }
    
    result = reasoning
    for eng, kor in translations.items():
        result = result.replace(eng, kor)
    
    return result

def parse_text_response(text: str) -> Dict:
    """텍스트 응답에서 난이도/유형 추출"""
    result = {
        'difficulty': 'V?',
        'type': '일반',
        'confidence': 0.5,
        'reasoning': text
    }
    
    # 거부 응답 감지
    refusal_keywords = [
        "sorry", "can't", "cannot", "unable", "refuse", "decline",
        "inappropriate", "unsafe", "policy", "guidelines",
        "죄송", "할 수 없", "거부", "분석할 수 없", "불가능"
    ]
    
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in refusal_keywords):
        result['reasoning'] = "GPT-4가 이미지 분석을 거부했습니다. 규칙 기반 분석을 사용합니다."
        return result
    
    # V-grade 찾기
    v_match = re.search(r'V(\d+)', text, re.IGNORECASE)
    if v_match:
        result['difficulty'] = f"V{v_match.group(1)}"
    
    # 유형 찾기 (영어/한국어 모두 지원)
    types_en = ['dynamic', 'static', 'crimp', 'sloper', 'traverse', 'campusing', 'balance', 'lunge', 'dyno']
    types_kr = ['다이나믹', '스태틱', '크림프', '슬로퍼', '트래버스', '캠퍼싱', '밸런스', '런지', '다이노']
    
    for t in types_en + types_kr:
        if t.lower() in text_lower:
            result['type'] = t
            break
    
    return result

def get_gpt4_status() -> Dict:
    """GPT-4 사용 가능 여부"""
    return {
        'available': GPT4_AVAILABLE,
        'api_key_set': bool(os.getenv("OPENAI_API_KEY")),
        'model': 'gpt-4o' if GPT4_AVAILABLE else None
    }

