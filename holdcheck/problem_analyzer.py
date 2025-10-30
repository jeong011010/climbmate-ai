"""
🧗‍♀️ ClimbMate 문제 분석 모듈
- 클라이밍 난이도 추정 (V-scale)
- 문제 유형 분석 (dynamic, static, power, etc.)
- 홀드 통계 분석
"""

import numpy as np


def analyze_problem(hold_data, group_id=None, wall_angle=None):
    """
    🧗‍♀️ AI 기반 클라이밍 문제 분석 (난이도 + 유형 추정)
    
    Args:
        hold_data: 전체 홀드 정보 리스트
        group_id: 분석할 그룹 ID (None이면 전체)
        wall_angle: 벽 각도 ("overhang", "slab", "face") - 사용자 입력
        
    Returns:
        dict: {
            'difficulty': 난이도 정보,
            'climb_type': 유형 정보,
            'statistics': 기본 통계
        }
    """
    # 그룹 필터링
    if group_id is not None:
        filtered_holds = [h for h in hold_data if h.get('group') == group_id]
    else:
        filtered_holds = hold_data
    
    if not filtered_holds or len(filtered_holds) < 1:
        return None
    
    # 🎯 1. 난이도 분석
    difficulty = analyze_difficulty(filtered_holds)
    
    # 🧗‍♀️ 2. 문제 유형 분석
    climb_type = analyze_climbing_type(filtered_holds, wall_angle)
    
    # 📊 3. 기본 통계
    centers = np.array([h['center'] for h in filtered_holds])
    areas = np.array([h.get('area', 2000) for h in filtered_holds])
    
    # 거리 분석
    distances = []
    if len(filtered_holds) > 1:
        for i, h1 in enumerate(filtered_holds):
            for h2 in filtered_holds[i+1:]:
                dist = np.linalg.norm(np.array(h1['center']) - np.array(h2['center']))
                distances.append(dist)
    
    statistics = {
        'num_holds': len(filtered_holds),
        'avg_hold_size': f"{np.mean(areas):.0f}px²",
        'total_height': f"{np.max(centers[:, 1]) - np.min(centers[:, 1]):.0f}px",
        'total_width': f"{np.max(centers[:, 0]) - np.min(centers[:, 0]):.0f}px",
        'avg_distance': f"{np.mean(distances):.0f}px" if distances else "0px",
        'max_distance': f"{np.max(distances):.0f}px" if distances else "0px"
    }
    
    return {
        'difficulty': difficulty,
        'climb_type': climb_type,
        'statistics': statistics
    }


def analyze_difficulty(filtered_holds):
    """🎯 난이도 분석 (개선 버전)"""
    num_holds = len(filtered_holds)
    areas = np.array([h.get('area', 2000) for h in filtered_holds])
    centers = np.array([h['center'] for h in filtered_holds])
    
    # 거리 계산
    distances = []
    consecutive_distances = []  # 인접 홀드 간 거리
    if num_holds > 1:
        # 모든 홀드 간 거리
        for i, h1 in enumerate(filtered_holds):
            for h2 in filtered_holds[i+1:]:
                dist = np.linalg.norm(np.array(h1['center']) - np.array(h2['center']))
                distances.append(dist)
        
        # 높이 순으로 정렬하여 연속 거리 계산
        sorted_holds = sorted(filtered_holds, key=lambda h: h['center'][1], reverse=True)
        for i in range(len(sorted_holds) - 1):
            dist = np.linalg.norm(
                np.array(sorted_holds[i]['center']) - np.array(sorted_holds[i+1]['center'])
            )
            consecutive_distances.append(dist)
    
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    max_distance = max(distances) if distances else 0
    avg_distance = np.mean(distances) if distances else 0
    avg_consecutive_distance = np.mean(consecutive_distances) if consecutive_distances else 0
    
    # 홀드 크기 분산 (일관성)
    area_std = np.std(areas)
    
    # 높이 변화
    heights = [h['center'][1] for h in filtered_holds]
    height_range = max(heights) - min(heights) if num_holds > 1 else 0
    
    # 수평 변화
    horizontal_coords = [h['center'][0] for h in filtered_holds]
    horizontal_range = max(horizontal_coords) - min(horizontal_coords) if num_holds > 1 else 0
    
    difficulty_score = 0
    factors = {}
    
    # 1. 홀드 크기 분석 (가중치 증가)
    small_hold_ratio = len([a for a in areas if a < 1200]) / num_holds
    if min_area < 600 or avg_area < 1000:
        difficulty_score += 5
        hold_size_level = "매우 작음 (크림프)"
        factors['hold_size'] = f"극소형 홀드 (평균 {int(avg_area)}px²)"
    elif avg_area < 1500:
        difficulty_score += 4
        hold_size_level = "작음"
        factors['hold_size'] = f"작은 홀드 (평균 {int(avg_area)}px²)"
    elif avg_area < 2500:
        difficulty_score += 2
        hold_size_level = "보통"
        factors['hold_size'] = f"보통 크기 홀드 (평균 {int(avg_area)}px²)"
    elif avg_area < 4000:
        difficulty_score += 1
        hold_size_level = "큼"
        factors['hold_size'] = f"큰 홀드 (평균 {int(avg_area)}px²)"
    else:
        difficulty_score += 0
        hold_size_level = "매우 큼 (쥬그)"
        factors['hold_size'] = f"매우 큰 홀드 (평균 {int(avg_area)}px²)"
    
    # 2. 연속 홀드 간격 분석 (실제 등반 경로)
    if avg_consecutive_distance > 200:
        difficulty_score += 5
        distance_level = "매우 큰 점프"
        factors['distance'] = f"다이나믹한 큰 점프 (평균 {int(avg_consecutive_distance)}px)"
    elif avg_consecutive_distance > 150:
        difficulty_score += 4
        distance_level = "큰 점프"
        factors['distance'] = f"큰 점프 필요 (평균 {int(avg_consecutive_distance)}px)"
    elif avg_consecutive_distance > 100:
        difficulty_score += 2
        distance_level = "보통 간격"
        factors['distance'] = f"보통 간격 (평균 {int(avg_consecutive_distance)}px)"
    elif avg_consecutive_distance > 60:
        difficulty_score += 1
        distance_level = "좁은 간격"
        factors['distance'] = f"좁은 간격 (평균 {int(avg_consecutive_distance)}px)"
    else:
        difficulty_score += 0
        distance_level = "매우 좁은 간격"
        factors['distance'] = f"매우 좁은 간격 (평균 {int(avg_consecutive_distance)}px)"
    
    # 3. 홀드 개수 분석 (적당한 개수가 적당한 난이도)
    if num_holds < 4:
        difficulty_score += 4
        holds_level = "매우 적음"
        factors['num_holds'] = f"{num_holds}개 - 극소수 홀드로 매우 어려움"
    elif num_holds < 6:
        difficulty_score += 3
        holds_level = "적음"
        factors['num_holds'] = f"{num_holds}개 - 적은 홀드로 어려움"
    elif num_holds < 10:
        difficulty_score += 1
        holds_level = "보통"
        factors['num_holds'] = f"{num_holds}개 - 적당한 개수"
    elif num_holds < 15:
        difficulty_score += 0
        holds_level = "많음"
        factors['num_holds'] = f"{num_holds}개 - 많은 홀드로 쉬움"
    else:
        difficulty_score -= 1
        holds_level = "매우 많음"
        factors['num_holds'] = f"{num_holds}개 - 매우 많은 홀드로 쉬움"
    
    # 4. 높이 변화 분석
    if height_range > 600:
        difficulty_score += 3
        height_level = "매우 큰 변화"
        factors['height'] = f"높이 변화 {int(height_range)}px - 체력 소모 큼"
    elif height_range > 400:
        difficulty_score += 2
        height_level = "큰 변화"
        factors['height'] = f"높이 변화 {int(height_range)}px - 보통"
    elif height_range > 200:
        difficulty_score += 1
        height_level = "보통 변화"
        factors['height'] = f"높이 변화 {int(height_range)}px - 적당함"
    else:
        height_level = "작은 변화"
        factors['height'] = f"높이 변화 {int(height_range)}px - 트래버스"
    
    # 5. 수평 변화 (트래버스)
    if horizontal_range > 500 and height_range < 200:
        difficulty_score += 2
        factors['traverse'] = f"긴 트래버스 (수평 {int(horizontal_range)}px)"
    
    # 6. 홀드 크기 일관성
    if area_std > 1000:
        difficulty_score += 1
        factors['consistency'] = "홀드 크기 편차가 커서 적응 어려움"
    
    # V-등급 매핑 (더 세밀하게)
    difficulty_score = max(0, difficulty_score)  # 음수 방지
    
    if difficulty_score <= 2:
        grade = "V0"
        level = "입문"
    elif difficulty_score <= 4:
        grade = "V1"
        level = "초급"
    elif difficulty_score <= 6:
        grade = "V2"
        level = "초급+"
    elif difficulty_score <= 8:
        grade = "V3"
        level = "초중급"
    elif difficulty_score <= 10:
        grade = "V4"
        level = "중급"
    elif difficulty_score <= 12:
        grade = "V5"
        level = "중급+"
    elif difficulty_score <= 14:
        grade = "V6"
        level = "중고급"
    elif difficulty_score <= 16:
        grade = "V7"
        level = "고급"
    elif difficulty_score <= 18:
        grade = "V8"
        level = "고급+"
    else:
        grade = "V9+"
        level = "전문가"
    
    # 신뢰도 계산 (더 보수적)
    confidence = 0.3 + min(num_holds / 20, 0.3)  # 30% ~ 60%
    
    return {
        "grade": grade,
        "level": level,
        "score": difficulty_score,
        "confidence": confidence,
        "factors": factors,
        "details": {
            "hold_size": hold_size_level,
            "distance": distance_level,
            "num_holds": holds_level,
            "height_change": height_level
        }
    }


def analyze_climbing_type(filtered_holds, wall_angle=None):
    """🧗‍♀️ 클라이밍 문제 유형 분석 (개선 버전)"""
    
    num_holds = len(filtered_holds)
    if num_holds < 1:
        return {"primary_type": "unknown", "secondary_types": [], "confidence": 0.0}
    
    areas = np.array([h.get('area', 2000) for h in filtered_holds])
    centers = np.array([h['center'] for h in filtered_holds])
    
    # 거리 계산
    distances = []
    if num_holds > 1:
        sorted_holds = sorted(filtered_holds, key=lambda h: h['center'][1], reverse=True)
        for i in range(len(sorted_holds) - 1):
            dist = np.linalg.norm(
                np.array(sorted_holds[i]['center']) - np.array(sorted_holds[i+1]['center'])
            )
            distances.append(dist)
    
    avg_area = np.mean(areas)
    avg_distance = np.mean(distances) if distances else 0
    
    # 높이/수평 변화
    heights = [h['center'][1] for h in filtered_holds]
    horizontal_coords = [h['center'][0] for h in filtered_holds]
    height_range = max(heights) - min(heights) if num_holds > 1 else 0
    horizontal_range = max(horizontal_coords) - min(horizontal_coords) if num_holds > 1 else 0
    
    # 유형 점수 계산
    type_scores = {}
    
    # Dynamic (다이나믹)
    if avg_distance > 150:
        type_scores['dynamic'] = 0.9
    elif avg_distance > 100:
        type_scores['dynamic'] = 0.6
    else:
        type_scores['dynamic'] = 0.2
    
    # Power (파워)
    if avg_area < 1500 and avg_distance > 120:
        type_scores['power'] = 0.8
    elif avg_area < 2000:
        type_scores['power'] = 0.5
    else:
        type_scores['power'] = 0.2
    
    # Technical (테크니컬)
    if num_holds >= 8 and avg_distance < 100:
        type_scores['technical'] = 0.8
    elif num_holds >= 6:
        type_scores['technical'] = 0.5
    else:
        type_scores['technical'] = 0.3
    
    # Static (정적)
    if avg_distance < 80:
        type_scores['static'] = 0.7
    elif avg_distance < 120:
        type_scores['static'] = 0.4
    else:
        type_scores['static'] = 0.1
    
    # Balance (밸런스)
    if avg_area > 2500:
        type_scores['balance'] = 0.6
    else:
        type_scores['balance'] = 0.3
    
    # Crimp (크림프)
    if avg_area < 1200:
        type_scores['crimp'] = 0.8
    elif avg_area < 1800:
        type_scores['crimp'] = 0.5
    else:
        type_scores['crimp'] = 0.2
    
    # Sloper (슬로퍼)
    if avg_area > 3000:
        type_scores['sloper'] = 0.7
    else:
        type_scores['sloper'] = 0.2
    
    # Coordination (조정력)
    if num_holds >= 8 and avg_distance > 100:
        type_scores['coordination'] = 0.7
    else:
        type_scores['coordination'] = 0.3
    
    # 벽 각도 보정
    if wall_angle == "overhang":
        type_scores['power'] = min(1.0, type_scores.get('power', 0) + 0.2)
        type_scores['dynamic'] = min(1.0, type_scores.get('dynamic', 0) + 0.1)
    elif wall_angle == "slab":
        type_scores['balance'] = min(1.0, type_scores.get('balance', 0) + 0.2)
        type_scores['technical'] = min(1.0, type_scores.get('technical', 0) + 0.1)
    
    # 상위 유형 선택
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    
    primary_type = sorted_types[0][0] if sorted_types else "unknown"
    primary_confidence = sorted_types[0][1] if sorted_types else 0.0
    
    secondary_types = [t[0] for t in sorted_types[1:4] if t[1] > 0.4]
    
    return {
        "primary_type": primary_type,
        "secondary_types": secondary_types,
        "confidence": primary_confidence,
        "all_scores": type_scores
    }

