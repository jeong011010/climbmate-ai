import os
import base64
from typing import Dict, List, Optional, Any
import json
import re
import asyncio

# OpenAI 클라이언트 (환경변수에서 API 키 로드)
try:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    GPT4_AVAILABLE = True
except:
    GPT4_AVAILABLE = False
    print("⚠️ OpenAI API 사용 불가 (환경변수 OPENAI_API_KEY 필요)")

# 이미지 리사이즈 함수 제거 (원본 이미지 사용)
# def resize_image_for_gpt4(image_base64: str, max_size: int = 512) -> str:
#     """원본 이미지를 그대로 사용하도록 변경"""
#     return image_base64

def _grade_to_int(grade: str) -> Optional[int]:
    if not isinstance(grade, str):
        return None
    grade = grade.strip().upper()
    if grade.startswith('V') and grade[1:].isdigit():
        return int(grade[1:])
    return None

def _int_to_grade(v: int) -> str:
    v = max(0, min(12, int(v)))
    return f"V{v}"

def _aggregate_results(results: List[Dict]) -> Dict:
    if not results:
        return {}
    # difficulty: median
    grades = [g for g in (_grade_to_int(r.get('difficulty')) for r in results) if g is not None]
    if grades:
        grades.sort()
        mid = grades[len(grades)//2]
        agg_diff = _int_to_grade(mid)
    else:
        agg_diff = results[0].get('difficulty', 'V?')
    # primary_type: majority
    counts: Dict[str,int] = {}
    for r in results:
        t = r.get('type') or r.get('primary_type') or '일반'
        counts[t] = counts.get(t, 0) + 1
    agg_type = max(counts.items(), key=lambda x: x[1])[0]
    # secondary_types: union up to 3
    sec_union = []
    seen = set()
    for r in results:
        for t in r.get('secondary_types', []) or []:
            if t not in seen:
                seen.add(t)
                sec_union.append(t)
    sec_union = sec_union[:3]
    # confidence: average
    confs = [float(r.get('confidence', 0.6)) for r in results]
    avg_conf = sum(confs)/len(confs)
    # merge brief reasoning
    reasons = [r.get('reasoning','') for r in results if r.get('reasoning')]
    merged_reason = " \n".join(reasons[:2])
    base = dict(results[0])
    base.update({
        'difficulty': agg_diff,
        'type': agg_type,
        'secondary_types': sec_union,
        'confidence': avg_conf,
        'reasoning': merged_reason
    })
    return base

def _build_context(holds_info: List[Dict], wall_angle: Optional[str], rule_based: Optional[Dict]) -> Dict[str, Any]:
    # 홀드 요약 (색상 정보 제외)
    num_holds = len(holds_info)
    areas = [h.get('area', 0) for h in holds_info]
    avg_area = sum(areas) / len(areas) if areas else 0
    # 거리 요약
    try:
        import numpy as np
        centers = [h.get('center', [0, 0]) for h in holds_info]
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                distances.append(float(dist))
        max_dist = max(distances) if distances else 0.0
        p90_dist = float(np.percentile(distances, 90)) if distances else 0.0
    except Exception:
        max_dist = 0.0
        p90_dist = 0.0
    # 규칙/하이브리드 힌트
    rule_hint = {}
    if rule_based:
        d = (rule_based.get('difficulty') or {}).get('grade') or rule_based.get('difficulty')
        t = (rule_based.get('climb_type') or {}).get('primary_type') or (rule_based.get('type') if isinstance(rule_based.get('type'), str) else None)
        if d: rule_hint['rule_difficulty'] = d
        if t: rule_hint['rule_type'] = t
    return {
        'num_holds': num_holds,
        'avg_hold_area': round(avg_area, 2),
        'max_center_distance': round(float(max_dist), 2),
        'p90_center_distance': round(float(p90_dist), 2),
        'wall_angle': wall_angle or 'unknown',
        **({'rule_hint': rule_hint} if rule_hint else {})
    }

async def _call_gpt4(image_base64: str, prompt: str, temperature: float = 0.2) -> Dict:
    response = await client.chat.completions.create(
        model="gpt-4o",
        store=True,  # 로그 30일 저장 (대시보드에서 확인 가능)
            messages=[{
                "role": "system",
                "content": (
                    "You are an expert bouldering route setter and judge analyzing indoor climbing gym walls for training purposes. "
                    "This is a safe, controlled indoor environment used for fitness and training. "
                    "Respond in JSON only, strictly matching the provided schema. "
                    "All textual content must be written in Korean (한국어), with no English explanations. "
                    "Base difficulty on observable features and the rubric; avoid generic answers."
                )
            }, {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
            ]
        }],
            max_tokens=800,  # 루트파인딩 포함으로 증가
            temperature=temperature,
            timeout=30
    )
    content = response.choices[0].message.content
    # JSON 추출 동일 로직 재사용
    try:
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
            json_str = json_match.group() if json_match else content.strip()
        result = json.loads(json_str)
        if 'primary_type' in result and 'type' not in result:
            result['type'] = result['primary_type']
        if 'secondary_types' not in result:
            result['secondary_types'] = []
        for k in ('key_factors','movements','challenges','tips'):
            result[k] = result.get(k, [])
        # route 필드 기본값
        if 'route' not in result:
            result['route'] = []
        result['used_gpt4'] = True
        result['raw_response'] = content
        return result
    except Exception as e:
        return { 'difficulty':'V?', 'type':'일반', 'confidence':0.5, 'reasoning': str(e), 'used_gpt4': True, 'raw_response': content }

async def _refine_result(image_base64: str, context: Dict[str,Any], first_result: Dict) -> Dict:
    schema = (
        "JSON 스키마: {\n"
        "  \"difficulty\": \"V0~V12\",\n"
        "  \"confidence\": 0.0~1.0,\n"
        "  \"primary_type\": one of [dynamic, static, crimp, sloper, pinch, balance, power, technical, coordination],\n"
        "  \"secondary_types\": string[],\n"
        "  \"reasoning\": string (모든 텍스트는 한국어),\n"
        "  \"key_factors\": string[] (한국어),\n"
        "  \"crux\": string (한국어),\n"
        "  \"movements\": string[] (한국어),\n"
        "  \"challenges\": string[] (한국어),\n"
        "  \"tips\": string[] (한국어),\n"
        "  \"comparison\": string,\n"
        "  \"route\": [{\"step\": int, \"hold_id\": int, \"hold_color\": string, \"action\": string (한국어), \"difficulty\": string (한국어: 쉬움/중간/어려움)}]\n"
        "}"
    )
    rubric = (
        "검토 규칙: 컨텍스트(홀드 수/간격/각도/규칙 힌트)와 1차 결과가 일치하는지 점검하고, 필요시 난이도/타입을 '약간'만 조정. "
        "근거 없는 큰 변경 금지. JSON만 출력. 모든 텍스트는 한국어로 작성."
    )
    prompt = (
        "다음 1차 분석을 검토하고 필요한 최소 수정만 적용해 더 일관된 결과로 보정하세요.\n"
        f"컨텍스트: {json.dumps(context, ensure_ascii=False)}\n"
        f"1차 결과: {json.dumps(first_result, ensure_ascii=False)}\n"
        f"{rubric}\n"
        f"{schema}"
    )
    return await _call_gpt4(image_base64, prompt, temperature=0.2)

async def analyze_with_gpt4_vision(
    image_base64: str,
    holds_info: List[Dict],
    wall_angle: Optional[str] = None,
    rule_based: Optional[Dict] = None
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
        # 컨텍스트 구축(룰 힌트 포함)
        context = _build_context(holds_info, wall_angle, rule_based)
        
        # 홀드 요약만 제공 (색상 정보 제외)
        holds_summary = f"{len(holds_info)}개 홀드"
        
        # 홀드 배치 정보: 위치(x,y) 제공
        max_holds_in_prompt = int(os.getenv('CLIMBMATE_GPT_MAX_HOLDS', '15'))
        holds_list = []
        # 크기 우선 정렬 (중요 홀드 먼저)
        sorted_holds = sorted(enumerate(holds_info), key=lambda x: x[1].get('area', 0), reverse=True)
        for idx, (i, h) in enumerate(sorted_holds[:max_holds_in_prompt]):
            center = h.get('center', [0, 0])
            holds_list.append({
                "id": i,
                "x": round(center[0], 1),
                "y": round(center[1], 1),  # 큰 y = 아래쪽(이미지 좌표계)
                "area": round(h.get('area', 0), 1)
            })

        rubric = (
            "난이도 루브릭: V0-1(큰 홀드/짧은 동작), V2-3(중간 간격/기본 기술), "
            "V4-5(긴 리치/힘 요구/정확한 풋워크), V6+(고난도 시퀀스/파워 또는 정밀도 상). "
            "오버행이면 같은 패턴에서 0.5~1단계 상향. 큰 간격, 크림프/슬로퍼, 크럭스의 난이도 증가 요인을 구체적으로 반영."
        )

        schema = (
            "JSON 스키마: {\n"
            "  \"difficulty\": \"V0~V12\",\n"
            "  \"confidence\": 0.0~1.0,\n"
            "  \"primary_type\": one of [dynamic, static, crimp, sloper, pinch, balance, power, technical, coordination],\n"
            "  \"secondary_types\": string[] (최대 3개),\n"
            "  \"reasoning\": string (모든 텍스트는 한국어, 간결하게),\n"
            "  \"key_factors\": string[] (한국어, 최대 3개),\n"
            "  \"crux\": string (한국어, 간결),\n"
            "  \"movements\": string[] (한국어, 최대 2개),\n"
            "  \"challenges\": string[] (한국어, 최대 2개),\n"
            "  \"tips\": string[] (한국어, 최대 2개),\n"
            "  \"comparison\": string (간결),\n"
            "  \"route\": [{\"step\": int, \"hold_id\": int, \"action\": string (한국어, 구체적 기술명), \"difficulty\": string (쉬움/중간/어려움)}] (최대 10개 스텝)\n"
            "}"
        )

        # 홀드를 y좌표 내림차순으로 정렬해 "시작→끝" 순서 명확화
        sorted_by_position = sorted(enumerate(holds_info), key=lambda x: x[1].get('center', [0, 0])[1], reverse=True)
        
        # 이미지 높이 추정 (가장 큰 y값 기준)
        max_y = max([h[1].get('center', [0, 0])[1] for h in sorted_by_position])
        
        # 바닥 근처 홀드 필터링 (하단 20% 이내는 발홀드 가능성 높음)
        floor_threshold = max_y * 0.80  # 상위 80% 이상 y값 = 바닥 근처
        
        # 시작 홀드 선정: 바닥 근처가 아니면서 하단에 있는 홀드
        start_candidates = []
        for idx, hold in sorted_by_position:
            y_pos = hold.get('center', [0, 0])[1]
            area = hold.get('area', 0)
            # 바닥 근처(큰 y)가 아니거나, 작은 홀드면 손 홀드로 간주
            if y_pos < floor_threshold or area < 3000:
                start_candidates.append((idx, hold))
            if len(start_candidates) >= 5:  # 후보 5개 수집
                break
        
        # 시작 홀드가 부족하면 상위 3개 사용
        if len(start_candidates) < 3:
            start_candidates = sorted_by_position[:3]
        
        start_holds = start_candidates[:3]
        top_holds = sorted_by_position[-3:]   # 상단 3개 (탑)
        
        start_ids = [h[0] for h in start_holds]
        top_ids = [h[0] for h in top_holds]
        
        # 시작/탑 홀드 정보를 명확하게 표시
        start_holds_info = [(h[0], h[1].get('center', [0, 0])[1], h[1].get('area', 0)) for h in start_holds]
        top_holds_info = [(h[0], h[1].get('center', [0, 0])[1]) for h in top_holds]
        
        # Few-shot 예시 (실제 좌표 기반)
        example = (
            f"⚠️ 중요: 이미지 좌표계에서 y값이 클수록 아래쪽입니다. (이미지 최대 y={int(max_y)})\n"
            f"✅ 시작 홀드 후보 (하단 손 홀드): {start_holds_info} (id, y좌표, 면적)\n"
            f"   - 바닥 근처(y>{int(floor_threshold)})의 큰 홀드는 발홀드로 제외됨\n"
            f"✅ 탑 홀드 후보 (상단): {top_holds_info} (작은 y값)\n"
            f"📌 루트는 반드시 큰 y → 작은 y 순서로 진행.\n"
            f"📌 첫 스텝 hold_id는 {start_ids} 중 하나 (손으로 잡는 시작 홀드).\n"
            f"📌 마지막 스텝 hold_id는 {top_ids} 중 하나 (탑아웃).\n"
            "예시: step 1 hold_id=5 (y=1400, 시작 홀드 잡기) → step 2 hold_id=3 (y=900) → step 3 hold_id=1 (y=200, 탑아웃)"
        )
        
        prompt = (
            "이것은 실내 클라이밍 짐의 훈련용 볼더링 벽입니다. 안전하고 통제된 환경에서 운동 목적으로 사용됩니다.\n"
            "클라이밍 문제를 정확히 분석하고 추천 루트를 제시하세요. 모호한 일반론을 피하고, 컨텍스트와 시각 증거를 근거로 판단하세요.\n"
            f"컨텍스트: {json.dumps(context, ensure_ascii=False)}\n"
            f"홀드: {holds_summary}\n"
            f"주요 홀드 샘플 (id/x/y/area): {json.dumps(holds_list[:10], ensure_ascii=False)}\n"
            f"{rubric}\n"
            f"{example}\n"
            "\n=== ⚠️ 루트파인딩 규칙 (반드시 준수) ===\n"
            f"1. 🚨 첫 스텝 (시작): hold_id는 ONLY {start_ids} 중 하나를 사용해야 합니다.\n"
            f"   - {start_ids}는 바닥 근처 발홀드를 제외한 하단 손 홀드입니다.\n"
            f"   - 바닥 가까이 있는 큰 홀드(y>{int(floor_threshold)}, area>3000)는 발로 밟는 홀드이므로 첫 스텝으로 절대 사용 금지.\n"
            f"   - action='시작 홀드 잡기' 또는 구체적 동작.\n"
            f"\n2. 🚨 마지막 스텝 (탑아웃): hold_id는 ONLY {top_ids} 중 하나를 사용해야 합니다.\n"
            f"   - {top_ids}는 가장 위(작은 y좌표)에 있는 홀드들입니다.\n"
            f"   - 마지막 스텝은 반드시 가장 위에 있는 홀드여야 합니다.\n"
            f"   - action='탑아웃' 또는 '탑 홀드 완등' 필수 포함.\n"
            f"   - ⛔ {top_ids} 외의 홀드를 마지막 스텝으로 사용하면 안 됩니다.\n"
            "\n3. 중간 스텝: 각 스텝의 y값이 점진적으로 감소(아래→위). 각 구간의 핵심 무브/기술을 action에 명시.\n"
            f"4. 발홀드 인식: 바닥 근처(y>{int(floor_threshold)})의 큰 홀드는 발로 디디는 홀드. 루트 스텝에 명시하지 않음.\n"
            "5. action 설명 예시:\n"
            "   - 평범: '시작 홀드 잡기', '중간 홀드로 이동', '탑아웃'\n"
            "   - 구체적: '다이노로 멀리 떨어진 슬로퍼 제압', '크로스 무브로 상단 크림프', '토훅 걸고 밸런싱', '데드포인트', '매칭 후 크로스', '플래깅으로 균형', '힐훅 걸어 당기기'\n"
            "6. 스텝 개수: 홀드 개수와 난이도에 맞게 3~10개. 복잡한 문제는 더 많은 스텝 허용.\n"
            "7. hold_id 검증: 제공된 홀드 리스트의 id만 사용 (0~N-1).\n"
            "\n출력은 반드시 순수 JSON(추가 텍스트/마크다운 금지). 스키마를 준수하세요.\n"
            f"{schema}"
        )
        # 앙상블/리파인 설정 (기본값 축소: rate limit 방지)
        ens_n = int(os.getenv('CLIMBMATE_GPT_ENS_N', '1'))  # 기본 1회로 축소
        enable_refine = os.getenv('CLIMBMATE_GPT_REFINE', '0') == '1'  # 기본 OFF
        
        # 단일 호출 (temp=0.1로 안정성 유지)
        result = await _call_gpt4(image_base64, prompt, temperature=0.1)
        
        # 거부 응답 재시도 (최대 2회, 점진적 완화)
        retry_count = 0
        while result.get('raw_response') and "sorry" in result['raw_response'].lower() and retry_count < 2:
            retry_count += 1
            print(f"⚠️  GPT-4 거부 응답 감지 ({retry_count}/2), 재시도 중...")
            
            # 재시도 전략
            if retry_count == 1:
                # 1차: 더 강한 안전 컨텍스트
                retry_prompt = (
                    "This is an educational analysis of an indoor climbing gym training wall for fitness purposes. "
                    "The image shows artificial climbing holds on a wall in a safe, controlled environment.\n\n" + prompt
                )
            else:
                # 2차: 프롬프트 단순화 (루트 제외, 분석만)
                simple_schema = schema.replace('"route": [{', '"route": [')
                retry_prompt = prompt.replace(
                    "루트파인딩: 이미지에서 보이는 홀드 순서로 시작부터 끝까지 추천 경로를 3~7 스텝으로 제시하세요. "
                    "각 스텝마다 action(어떤 손/발로 어떻게 잡는지 한국어로), difficulty(쉬움/중간/어려움)를 명시하세요.\n",
                    ""
                ).replace(schema, simple_schema)
            
            result = await _call_gpt4(image_base64, retry_prompt, temperature=0.0)
        
        # 옵션: 리파인 패스 (env=1일 때만)
        if enable_refine and result.get('difficulty') and result.get('type'):
            refined = await _refine_result(image_base64, context, result)
            if refined.get('difficulty') and refined.get('type'):
                result = refined
        
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

def translate_and_enhance_gpt4_result(gpt4_result, holds_info=None):
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
    
    # 보조 필드 한국어 매핑(간이)
    movements_kr = translate_movements(gpt4_result.get('movements', []))
    challenges_kr = translate_challenges(gpt4_result.get('challenges', []))
    tips_kr = translate_tips(gpt4_result.get('tips', []))
    reasoning_kr = translate_reasoning(gpt4_result.get('reasoning', ''))

    # 루트 검증 및 재정렬 (y좌표 기준 큰→작은 순서)
    route = gpt4_result.get('route', [])
    if holds_info:
        validated_route = validate_and_fix_route(route, holds_info)
    else:
        print("⚠️ 경고: holds_info가 없어 루트 검증을 건너뜁니다.")
        validated_route = route
    
    # 기본 결과 (한국어 필드 반영)
    result = {
        'difficulty': difficulty_map.get(gpt4_result.get('difficulty', 'V?'), 'V? (미분석)'),
        'type': primary_type_kr,
        'secondary_types': secondary_types_kr,
        'confidence': gpt4_result.get('confidence', 0.0),
        'reasoning': reasoning_kr,
        'key_factors': gpt4_result.get('key_factors', []),
        'crux': gpt4_result.get('crux', ''),
        'movements': movements_kr,
        'challenges': challenges_kr,
        'tips': tips_kr,
        'comparison': gpt4_result.get('comparison', ''),
        'route': validated_route
    }
    
    # 상세 분석 생성 (훨씬 더 자세하게)
    detailed_analysis = generate_detailed_analysis_v2(gpt4_result, result)
    result['detailed_analysis'] = detailed_analysis
    
    # 원본 필드 유지 (used_gpt4, raw_response 등)
    result['used_gpt4'] = gpt4_result.get('used_gpt4', True)
    if 'raw_response' in gpt4_result:
        result['raw_response'] = gpt4_result['raw_response']
    
    return result

def validate_and_fix_route(route, holds_info):
    """
    루트 검증 및 수정: y좌표 기준 아래→위 순서로 재정렬
    - GPT-4가 잘못된 순서로 반환해도 강제로 수정
    - 가장 위 홀드가 반드시 마지막 스텝이 되도록 보장
    """
    if not route or not holds_info:
        return route
    
    # hold_id가 유효한 스텝만 필터링
    valid_steps = []
    for step in route:
        hold_id = step.get('hold_id')
        if hold_id is not None and 0 <= hold_id < len(holds_info):
            valid_steps.append(step)
    
    if len(valid_steps) < 2:
        return route
    
    # 각 스텝에 y좌표 추가
    steps_with_y = []
    for step in valid_steps:
        hold_id = step.get('hold_id')
        hold = holds_info[hold_id]
        y_pos = hold.get('center', [0, 0])[1]
        steps_with_y.append({
            'step': step,
            'y': y_pos,
            'hold_id': hold_id
        })
    
    # y좌표 기준 내림차순 정렬 (큰 y → 작은 y = 아래 → 위)
    sorted_steps = sorted(steps_with_y, key=lambda x: x['y'], reverse=True)
    
    # 🚨 추가 검증: 마지막 스텝이 가장 위 홀드인지 확인
    # 만약 루트에 가장 위 홀드가 없다면, 가장 위에 있는 홀드를 마지막에 추가
    all_holds_y = [(idx, holds_info[idx].get('center', [0, 0])[1]) for idx in range(len(holds_info))]
    top_hold_id = min(all_holds_y, key=lambda x: x[1])[0]  # 가장 작은 y = 가장 위
    
    print(f"🔍 루트 검증: 전체 홀드 수={len(holds_info)}, 루트 스텝 수={len(sorted_steps)}")
    print(f"🔍 가장 위 홀드: id={top_hold_id}, y={all_holds_y[top_hold_id][1]}")
    print(f"🔍 현재 루트 홀드 IDs: {[item['hold_id'] for item in sorted_steps]}")
    print(f"🔍 현재 루트 y좌표: {[item['y'] for item in sorted_steps]}")
    
    route_hold_ids = [item['hold_id'] for item in sorted_steps]
    if top_hold_id not in route_hold_ids:
        # 가장 위 홀드가 루트에 없으면 강제로 추가
        print(f"⚠️ 경고: 가장 위 홀드 (id={top_hold_id}, y={all_holds_y[top_hold_id][1]})가 루트에 없음. 강제로 추가합니다.")
        top_hold = holds_info[top_hold_id]
        sorted_steps.append({
            'step': {
                'hold_id': top_hold_id,
                'action': '탑아웃',
                'difficulty': '중간'
            },
            'y': top_hold.get('center', [0, 0])[1],
            'hold_id': top_hold_id
        })
        # 재정렬
        sorted_steps = sorted(sorted_steps, key=lambda x: x['y'], reverse=True)
    else:
        print(f"✅ 가장 위 홀드가 루트에 포함되어 있습니다.")
    
    # step 번호 재할당
    fixed_route = []
    for idx, item in enumerate(sorted_steps, start=1):
        step = item['step'].copy()
        step['step'] = idx
        
        # action 수정: 첫 스텝은 "시작", 마지막 스텝은 "탑아웃" 포함
        if idx == 1:
            if '시작' not in step.get('action', ''):
                step['action'] = '시작 홀드 잡기'
        elif idx == len(sorted_steps):
            if '탑' not in step.get('action', ''):
                step['action'] = '탑아웃'
            # 마지막 스텝이 가장 위 홀드인지 최종 확인
            if item['hold_id'] != top_hold_id:
                print(f"⚠️ 경고: 마지막 스텝이 가장 위 홀드가 아님. hold_id={item['hold_id']}, top_hold_id={top_hold_id}")
        
        fixed_route.append(step)
    
    return fixed_route

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

