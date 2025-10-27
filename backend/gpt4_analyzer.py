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
        image_base64_optimized = resize_image_for_gpt4(image_base64, max_size=768)
        
        # 프롬프트 구성 - 훨씬 더 자세하게
        wall_info = f"벽 각도: {wall_angle} (overhang은 경사벽, slab은 슬랩, face는 수직)" if wall_angle else "벽 각도: 수직으로 추정"
        
        # 홀드 분포 정보
        color_distribution = ", ".join([f"{color}: {count}개" for color, count in color_groups.items()])
        
        prompt = f"""당신은 경험이 풍부한 클라이밍 코치입니다. 이 볼더링 문제를 매우 상세하게 분석해주세요.

## 📊 문제 정보
- 홀드 개수: {num_holds}개
- {wall_info}
- 색상 분포: {color_distribution}
- 평균 홀드 크기: {avg_area:.0f}px²
- 최대 홀드 간격: {max_dist:.0f}px

## 🎯 분석 요청사항

**1. 난이도 평가 (V0-V17)**
- V-grade를 정확히 판단하세요
- 홀드 크기, 간격, 벽 각도, 홀드 유형을 모두 고려
- 초급(V0-V2), 중급(V3-V5), 고급(V6-V9), 전문가(V10+) 구분

**2. 클라이밍 스타일 분석**
- **주요 스타일** 1개를 선택: dynamic(다이나믹), static(스태틱), crimp(크림프), sloper(슬로퍼), pinch(핀치), balance(밸런스), power(파워), endurance(지구력), technical(기술), coordination(협응)
- **부가 스타일** 2-3개를 추가로 선택
- 각 스타일이 필요한 이유를 구체적으로 설명

**3. 상세 분석**
- 이 문제의 핵심 난이도 요인 (크럭스 구간)
- 필요한 구체적인 동작들 (예: "왼손 언더클링에서 오른발 하이스텝", "오른손 다이노로 톱홀드 잡기")
- 주요 도전과제 3-5가지
- 실전 팁 3-5가지 (구체적인 신체 사용법, 시퀀스 제안)
- 비슷한 난이도의 다른 문제와 비교

**응답 형식 (JSON):**
{{
  "difficulty": "V4",
  "confidence": 0.85,
  "primary_type": "dynamic",
  "secondary_types": ["power", "coordination"],
  "reasoning": "홀드 간격이 평균보다 30% 넓어 다이나믹한 움직임이 필수적입니다. 특히 중간 구간에서 큰 리치가 요구되며, 이는 V4 수준의 난이도를 형성합니다. 벽 각도가 {wall_angle}이므로 상체 근력도 중요합니다.",
  "key_factors": [
    "중간 홀드 간격 60cm - 큰 리치 필요",
    "슬로퍼 홀드 2개 - 오픈 그립 필수",
    "벽 각도로 인한 상체 근력 요구"
  ],
  "crux": "3번째에서 5번째 홀드로 이동하는 구간. 오른손 언더클링 상태에서 왼손으로 슬로퍼를 잡아야 하며, 발 위치 선택이 핵심입니다.",
  "movements": [
    "시작: 양손 매칭으로 첫 홀드 안정화",
    "중간: 오른발 하이스텝 후 왼손 다이나믹 무브",
    "크럭스: 슬로퍼 홀드에서 바디텐션 유지하며 다음 홀드로 전환",
    "마무리: 마지막 홀드는 매칭 후 컨트롤된 동작으로"
  ],
  "challenges": [
    "큰 리치 구간 - 작은 체격은 점프 필요",
    "슬로퍼 홀드 - 접촉력과 바디텐션 필수",
    "지속적인 상체 근력 - 지구력 관리 중요",
    "발 위치 선택 - 작은 풋홀드 정확히 사용"
  ],
  "tips": [
    "시작 전 전체 시퀀스를 미리 시각화하세요",
    "크럭스 구간에서 호흡을 멈추지 말고 계속 호흡하세요",
    "슬로퍼 홀드는 팔을 곧게 펴고 체중을 실어 마찰력 극대화",
    "발 위치를 먼저 정한 후 손 무브를 시도하세요",
    "다이나믹 구간에서는 주저하지 말고 과감하게 commit하세요"
  ],
  "comparison": "일반적인 V3보다 한 단계 높은 수준입니다. 비슷한 스타일의 체육관 세팅과 비교하면 크럭스 난이도가 더 높습니다."
}}

**중요:** 반드시 위 JSON 형식으로만 답변하고, 추가 설명 없이 JSON만 출력하세요."""

        # 🚀 GPT-4o 사용 (고품질 분석)
        response = await client.chat.completions.create(
            model="gpt-4o",  # 다시 gpt-4o 사용
            messages=[{
                "role": "system",
                "content": "You are an expert climbing coach with 15+ years of experience. Analyze bouldering problems in detail and respond ONLY with valid JSON. No markdown, no explanations, just pure JSON."
            }, {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64_optimized}",
                            "detail": "high"  # low에서 high로 변경 (더 상세한 분석)
                        }
                    }
                ]
            }],
            max_tokens=800,  # 150에서 대폭 증가 (상세한 답변용)
            temperature=0.4,  # 약간 높여서 창의적 분석
            timeout=15  # 타임아웃 증가
        )
        
        # 응답 파싱
        content = response.choices[0].message.content
        print(f"📝 GPT-4 응답 (처음 500자): {content[:500]}...")
        
        # JSON 추출
        try:
            # JSON 블록 찾기 (마크다운 코드 블록 제거)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    json_str = json_match.group()
                else:
                    json_str = content.strip()
            
            result = json.loads(json_str)
            
            # 새로운 필드 처리
            # primary_type이 있으면 type으로 매핑
            if 'primary_type' in result:
                result['type'] = result['primary_type']
            elif 'type' not in result:
                result['type'] = '일반'
            
            # secondary_types 추가
            if 'secondary_types' not in result:
                result['secondary_types'] = []
            
            # 새 필드들 추가
            if 'key_factors' not in result:
                result['key_factors'] = []
            if 'crux' not in result:
                result['crux'] = ''
            if 'comparison' not in result:
                result['comparison'] = ''
            
            # movements, challenges, tips는 기존과 동일
            if 'movements' not in result:
                result['movements'] = []
            if 'challenges' not in result:
                result['challenges'] = []
            if 'tips' not in result:
                result['tips'] = []
            
            print(f"✅ GPT-4 JSON 파싱 성공:")
            print(f"   - 난이도: {result.get('difficulty')}")
            print(f"   - 주요 스타일: {result.get('type')}")
            print(f"   - 부가 스타일: {result.get('secondary_types')}")
            print(f"   - 핵심 요인: {len(result.get('key_factors', []))}개")
            print(f"   - 크럭스: {'있음' if result.get('crux') else '없음'}")
            
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
            print(f"   원본 응답: {content[:200]}...")
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
        'pinch': '핀치',
        'traverse': '트래버스',
        'campusing': '캠퍼싱',
        'balance': '밸런스',
        'power': '파워',
        'endurance': '지구력',
        'technical': '기술',
        'coordination': '협응',
        'lunge': '런지',
        'dyno': '다이노',
        '일반': '일반'
    }
    
    # 주요 타입 번역
    primary_type = gpt4_result.get('type', gpt4_result.get('primary_type', '일반'))
    primary_type_kr = type_map.get(primary_type, primary_type)
    
    # 부가 타입 번역
    secondary_types = gpt4_result.get('secondary_types', [])
    secondary_types_kr = [type_map.get(t, t) for t in secondary_types]
    
    # 기본 결과
    result = {
        'difficulty': difficulty_map.get(gpt4_result.get('difficulty', 'V?'), 'V? (미분석)'),
        'type': primary_type_kr,
        'secondary_types': secondary_types_kr,
        'confidence': gpt4_result.get('confidence', 0.0),
        'reasoning': gpt4_result.get('reasoning', ''),
        'key_factors': gpt4_result.get('key_factors', []),
        'crux': gpt4_result.get('crux', ''),
        'movements': gpt4_result.get('movements', []),
        'challenges': gpt4_result.get('challenges', []),
        'tips': gpt4_result.get('tips', []),
        'comparison': gpt4_result.get('comparison', '')
    }
    
    # 상세 분석 생성 (훨씬 더 자세하게)
    detailed_analysis = generate_detailed_analysis_v2(gpt4_result, result)
    result['detailed_analysis'] = detailed_analysis
    
    return result

def generate_detailed_analysis_v2(gpt4_result, translated_result):
    """GPT-4 결과를 바탕으로 매우 상세한 한글 분석 생성"""
    
    difficulty = translated_result.get('difficulty', 'V?')
    primary_type = translated_result.get('type', '일반')
    secondary_types = translated_result.get('secondary_types', [])
    reasoning = translated_result.get('reasoning', '')
    key_factors = translated_result.get('key_factors', [])
    crux = translated_result.get('crux', '')
    movements = translated_result.get('movements', [])
    challenges = translated_result.get('challenges', [])
    tips = translated_result.get('tips', [])
    comparison = translated_result.get('comparison', '')
    
    analysis_parts = []
    
    # 1. 난이도 및 타입 요약
    type_desc = f"**{primary_type}**"
    if secondary_types:
        type_desc += f" (부가: {', '.join(secondary_types)})"
    
    analysis_parts.append(f"🎯 **난이도**: {difficulty}")
    analysis_parts.append(f"🧗 **클라이밍 스타일**: {type_desc}")
    analysis_parts.append("")
    
    # 2. 종합 분석
    if reasoning:
        analysis_parts.append(f"📊 **종합 분석**")
        analysis_parts.append(reasoning)
        analysis_parts.append("")
    
    # 3. 핵심 난이도 요인
    if key_factors:
        analysis_parts.append(f"🔑 **핵심 난이도 요인**")
        for i, factor in enumerate(key_factors, 1):
            analysis_parts.append(f"  {i}. {factor}")
        analysis_parts.append("")
    
    # 4. 크럭스 구간
    if crux:
        analysis_parts.append(f"⚡ **크럭스 (가장 어려운 구간)**")
        analysis_parts.append(crux)
        analysis_parts.append("")
    
    # 5. 필요한 동작 시퀀스
    if movements:
        analysis_parts.append(f"🎬 **동작 시퀀스**")
        for i, movement in enumerate(movements, 1):
            analysis_parts.append(f"  {i}. {movement}")
        analysis_parts.append("")
    
    # 6. 주요 도전과제
    if challenges:
        analysis_parts.append(f"⚠️ **주요 도전과제**")
        for i, challenge in enumerate(challenges, 1):
            analysis_parts.append(f"  {i}. {challenge}")
        analysis_parts.append("")
    
    # 7. 실전 팁
    if tips:
        analysis_parts.append(f"💡 **실전 공략 팁**")
        for i, tip in enumerate(tips, 1):
            analysis_parts.append(f"  {i}. {tip}")
        analysis_parts.append("")
    
    # 8. 비교 분석
    if comparison:
        analysis_parts.append(f"📈 **비교 분석**")
        analysis_parts.append(comparison)
    
    return "\n".join(analysis_parts)

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

