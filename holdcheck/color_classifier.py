"""
🎨 ClimbMate 색상 분류 시스템
- 홀드 색상 추출 및 그룹핑
- color_ranges.json 기반 룰 분류
- ML 모델 기반 색상 예측 (선택적)
"""

import numpy as np
import cv2
import os
import json
from pathlib import Path

# 🚀 성능 최적화: 전역 캐시
_color_ranges_cache = None

# 🤖 ML 모델 캐시
_ml_color_model = None
_ml_color_encoder = None
_ml_model_loaded = False

# 🔇 Runtime verbosity control
VERBOSE = os.getenv('CLIMBMATE_VERBOSE', '0') == '1'


# ============================================================================
# 🛠️ 유틸리티 함수
# ============================================================================

def hsv_to_rgb_fast(h, s, v):
    """⚡ 빠른 HSV → RGB 변환 (수학적 변환, cv2보다 빠름)"""
    h, s, v = float(h), float(s) / 255.0, float(v) / 255.0
    
    # H를 [0, 360) 범위로 변환
    h_deg = h * 2.0  # OpenCV H[0-179] → [0-358]도
    
    c = v * s
    x = c * (1 - abs((h_deg / 60.0) % 2 - 1))
    m = v - c
    
    if 0 <= h_deg < 60:
        r, g, b = c, x, 0
    elif 60 <= h_deg < 120:
        r, g, b = x, c, 0
    elif 120 <= h_deg < 180:
        r, g, b = 0, c, x
    elif 180 <= h_deg < 240:
        r, g, b = 0, x, c
    elif 240 <= h_deg < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]


# ============================================================================
# 🤖 ML 모델 관리
# ============================================================================

def reset_ml_model_cache():
    """🔄 ML 모델 캐시 초기화 (재학습 후 호출)"""
    global _ml_color_model, _ml_color_encoder, _ml_model_loaded
    _ml_color_model = None
    _ml_color_encoder = None
    _ml_model_loaded = False
    print("   🔄 ML 모델 캐시 초기화 완료")


def load_ml_color_model():
    """🤖 ML 색상 분류 모델 로드 (캐싱)"""
    global _ml_color_model, _ml_color_encoder, _ml_model_loaded
    
    if _ml_model_loaded:
        return _ml_color_model, _ml_color_encoder
    
    try:
        import sys
        backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend')
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        from ml_trainer import COLOR_MODEL_PATH, COLOR_ENCODER_PATH
        import pickle
        
        if os.path.exists(COLOR_MODEL_PATH) and os.path.exists(COLOR_ENCODER_PATH):
            with open(COLOR_MODEL_PATH, 'rb') as f:
                _ml_color_model = pickle.load(f)
            with open(COLOR_ENCODER_PATH, 'rb') as f:
                _ml_color_encoder = pickle.load(f)
            print("   ✅ ML 색상 분류 모델 로드 완료")
            _ml_model_loaded = True
            return _ml_color_model, _ml_color_encoder
        else:
            print("   ⚠️ ML 모델 파일 없음")
            _ml_model_loaded = True
            return None, None
    except Exception as e:
        print(f"   ⚠️ ML 모델 로드 실패: {e}")
        _ml_model_loaded = True
        return None, None


def predict_with_ml(hold_features):
    """🤖 ML 모델로 색상 예측"""
    global _ml_color_model, _ml_color_encoder
    
    if _ml_color_model is None or _ml_color_encoder is None:
        return None, 0.0
    
    try:
        # 특징 추출
        from ml_trainer import extract_color_features
        features = extract_color_features(hold_features)
        
        # 예측
        prediction = _ml_color_model.predict([features])[0]
        probabilities = _ml_color_model.predict_proba([features])[0]
        confidence = float(max(probabilities))
        
        # 레이블 디코딩
        color_name = _ml_color_encoder.inverse_transform([prediction])[0]
        
        return color_name, confidence
    except Exception as e:
        if VERBOSE:
            print(f"   ⚠️ ML 예측 실패: {e}")
        return None, 0.0


# ============================================================================
# 📂 색상 범위 설정 관리
# ============================================================================

def load_color_ranges(config_path="holdcheck/color_ranges.json"):
    """색상 범위 설정 파일 로드 (사용자 피드백 반영)"""
    global _color_ranges_cache
    
    if _color_ranges_cache is not None:
        return _color_ranges_cache
    
    # 파일이 있으면 로드
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            _color_ranges_cache = json.load(f)
            if VERBOSE:
                print(f"✅ 색상 범위 설정 로드: {config_path}")
            return _color_ranges_cache
    
    # 없으면 기본값 생성
    _color_ranges_cache = get_default_color_ranges_data()
    save_color_ranges(_color_ranges_cache, config_path)
    print(f"✅ 기본 색상 범위 생성: {config_path}")
    return _color_ranges_cache


def save_color_ranges(ranges, config_path="holdcheck/color_ranges.json"):
    """색상 범위 설정 저장"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(ranges, f, indent=2, ensure_ascii=False)
    if VERBOSE:
        print(f"💾 색상 범위 저장: {config_path}")


def reload_color_ranges():
    """색상 범위 캐시 강제 리로드 (피드백 학습 후 즉시 적용)"""
    global _color_ranges_cache
    _color_ranges_cache = None
    print("🔄 색상 범위 캐시 초기화 - 다음 분석부터 새 범위 적용")


def get_default_color_ranges_data():
    """기본 색상 범위 데이터 (JSON 직렬화 가능)"""
    return {
        "version": "1.0",
        "last_updated": "2025-01-01",
        "feedback_count": 0,
        "colors": {
            "black": {
                "name": "검정색",
                "priority": 1,
                "hsv_ranges": [
                    {"h": [0, 180], "s": [0, 60], "v": [0, 150]}
                ],
                "rgb_conditions": [
                    {"type": "max_value", "threshold": 80},
                    {"type": "achromatic", "brightness_max": 150, "channel_diff_max": 50}
                ]
            },
            "white": {
                "name": "흰색",
                "priority": 2,
                "hsv_ranges": [
                    {"h": [0, 180], "s": [0, 30], "v": [150, 255]}
                ],
                "rgb_conditions": [
                    {"type": "min_value", "threshold": 80},
                    {"type": "achromatic", "brightness_min": 80, "brightness_max": 255, "channel_diff_max": 50}
                ]
            },
            "red": {
                "name": "빨간색",
                "priority": 4,
                "hsv_ranges": [
                    {"h": [0, 10], "s": [100, 255], "v": [100, 255]},
                    {"h": [165, 180], "s": [120, 255], "v": [80, 255]}
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "r", "min_value": 150, "diff_threshold": 50}
                ]
            },
            "orange": {
                "name": "주황색",
                "priority": 5,
                "hsv_ranges": [
                    {"h": [10, 25], "s": [100, 255], "v": [100, 255]}
                ],
                "rgb_conditions": [
                    {"type": "two_channel_high", "channels": ["r", "g"], "r_min": 150, "g_min": 80, "b_max": 120, "r_over_g": True}
                ]
            },
            "yellow": {
                "name": "노란색",
                "priority": 6,
                "hsv_ranges": [
                    {"h": [25, 40], "s": [100, 255], "v": [150, 255]}
                ],
                "rgb_conditions": [
                    {"type": "two_channel_high", "channels": ["r", "g"], "r_min": 150, "g_min": 150, "b_max": 150, "similar": True}
                ]
            },
            "green": {
                "name": "초록색",
                "priority": 7,
                "hsv_ranges": [
                    {"h": [40, 75], "s": [100, 255], "v": [100, 255]}
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "g", "min_value": 100, "diff_threshold": 30}
                ]
            },
            "mint": {
                "name": "민트색",
                "priority": 8,
                "hsv_ranges": [
                    {"h": [75, 100], "s": [30, 255], "v": [90, 255]}
                ],
                "rgb_conditions": [
                    {"type": "two_channel_high", "channels": ["g", "b"], "g_min": 150, "b_min": 150, "r_max": 150}
                ]
            },
            "blue": {
                "name": "파란색",
                "priority": 9,
                "hsv_ranges": [
                    {"h": [100, 125], "s": [50, 255], "v": [110, 255]}
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "b", "min_value": 100, "diff_threshold": 30}
                ]
            },
            "purple": {
                "name": "보라색",
                "priority": 10,
                "hsv_ranges": [
                    {"h": [125, 160], "s": [60, 255], "v": [100, 255]}
                ],
                "rgb_conditions": [
                    {"type": "two_channel_high", "channels": ["r", "b"], "r_min": 100, "b_min": 100, "g_diff": 20}
                ]
            },
            "pink": {
                "name": "분홍색",
                "priority": 11,
                "hsv_ranges": [
                    {"h": [160, 170], "s": [50, 120], "v": [180, 255]}
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "r", "min_value": 180, "g_min": 100, "b_min": 100}
                ]
            },
            "brown": {
                "name": "갈색",
                "priority": 12,
                "hsv_ranges": [
                    {"h": [0, 10], "s": [80, 200], "v": [50, 150]}
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "r", "min_value": 80, "max_value": 150, "dark": True}
                ]
            }
        }
    }


# ============================================================================
# 🎨 색상 분류 함수
# ============================================================================

def classify_color_by_hsv(h, s, v, rgb, colors_config):
    """HSV 범위 기반 색상 분류 (color_ranges.json 우선 사용)"""
    
    # 🔥 1️⃣ 먼저 color_ranges.json 기반 분류 시도
    sorted_colors = sorted(colors_config.items(), key=lambda x: x[1].get("priority", 999))
    
    for color_name, config in sorted_colors:
        # HSV 범위 체크
        if "hsv_ranges" in config:
            for hsv_range in config["hsv_ranges"]:
                h_min, h_max = hsv_range["h"]
                s_min, s_max = hsv_range["s"]
                v_min, v_max = hsv_range["v"]
                
                # Hue는 원형이므로 특별 처리
                h_match = False
                if h_min <= h_max:
                    h_match = h_min <= h <= h_max
                else:  # 예: [170, 10] (빨강)
                    h_match = h >= h_min or h <= h_max
                
                if h_match and s_min <= s <= s_max and v_min <= v <= v_max:
                    confidence = calculate_confidence_hsv(h, s, v, hsv_range)
                    return color_name, confidence, f"color_ranges.json: H={h}, S={s}, V={v}"
        
        # RGB 조건 체크 (보조)
        if "rgb_conditions" in config:
            for condition in config["rgb_conditions"]:
                if check_rgb_condition(rgb, condition):
                    confidence = 0.8  # RGB 조건은 약간 낮은 신뢰도
                    return color_name, confidence, f"color_ranges.json RGB: {rgb}"
    
    # 2️⃣ 매칭 실패 - 폴백
    return find_nearest_color_hsv(h, s, v, colors_config)


def classify_color_by_rgb(rgb, colors_config):
    """RGB 조건 기반 색상 분류"""
    r, g, b = rgb
    
    sorted_colors = sorted(colors_config.items(), key=lambda x: x[1].get("priority", 999))
    
    for color_name, config in sorted_colors:
        if "rgb_conditions" in config:
            for condition in config["rgb_conditions"]:
                if check_rgb_condition(rgb, condition):
                    confidence = 0.85
                    return color_name, confidence, f"RGB: {rgb}"
    
    # 매칭 실패
    return "unknown", 0.5, "No match"


def check_rgb_condition(rgb, condition):
    """RGB 조건 체크"""
    r, g, b = rgb
    cond_type = condition.get("type")
    
    if cond_type == "max_value":
        return max(r, g, b) < condition["threshold"]
    
    elif cond_type == "min_value":
        return min(r, g, b) > condition["threshold"]
    
    elif cond_type == "achromatic":
        brightness = max(r, g, b)
        channel_diff = max(r, g, b) - min(r, g, b)
        
        checks = []
        if "brightness_min" in condition:
            checks.append(brightness >= condition["brightness_min"])
        if "brightness_max" in condition:
            checks.append(brightness <= condition["brightness_max"])
        if "channel_diff_max" in condition:
            checks.append(channel_diff < condition["channel_diff_max"])
        
        return all(checks) if checks else False
    
    elif cond_type == "dominant_channel":
        channel = condition["channel"]
        min_val = condition.get("min_value", 0)
        diff_thresh = condition.get("diff_threshold", 30)
        
        channel_val = {"r": r, "g": g, "b": b}[channel]
        other_vals = [v for k, v in {"r": r, "g": g, "b": b}.items() if k != channel]
        
        return (channel_val >= min_val and 
                all(channel_val > ov + diff_thresh for ov in other_vals))
    
    elif cond_type == "two_channel_high":
        channels = condition["channels"]
        vals = {"r": r, "g": g, "b": b}
        
        checks = []
        for ch in channels:
            if f"{ch}_min" in condition:
                checks.append(vals[ch] >= condition[f"{ch}_min"])
            if f"{ch}_max" in condition:
                checks.append(vals[ch] <= condition[f"{ch}_max"])
        
        # 추가 조건
        if condition.get("r_over_g"):
            checks.append(r > g)
        if condition.get("similar"):
            checks.append(abs(r - g) < 50)
        if "g_diff" in condition:
            checks.append(r > g + condition["g_diff"] and b > g + condition["g_diff"])
        
        return all(checks)
    
    return False


def calculate_confidence_hsv(h, s, v, hsv_range):
    """HSV 매칭 신뢰도 계산"""
    h_min, h_max = hsv_range["h"]
    s_min, s_max = hsv_range["s"]
    v_min, v_max = hsv_range["v"]
    
    # 중심에 가까울수록 높은 신뢰도
    h_center = (h_min + h_max) / 2
    s_center = (s_min + s_max) / 2
    v_center = (v_min + v_max) / 2
    
    h_dist = min(abs(h - h_center), 180 - abs(h - h_center)) / 90  # 정규화
    s_dist = abs(s - s_center) / 127.5
    v_dist = abs(v - v_center) / 127.5
    
    # 거리 기반 신뢰도
    avg_dist = (h_dist + s_dist + v_dist) / 3
    confidence = 1.0 - avg_dist * 0.3  # 최대 0.3 감소
    
    return max(0.5, min(1.0, confidence))


def find_nearest_color_hsv(h, s, v, colors_config):
    """가장 가까운 색상 찾기 (폴백)"""
    # 무채색 체크
    if s < 50:
        if v < 80:
            return "black", 0.6, "Fallback: dark achromatic"
        else:
            return "white", 0.6, "Fallback: achromatic"
    
    # Hue 기반 분류
    if h < 10:
        return "red", 0.5, "Fallback: low hue"
    elif h < 25:
        return "orange", 0.5, "Fallback: hue range"
    elif h < 40:
        return "yellow", 0.5, "Fallback: hue range"
    elif h < 75:
        return "green", 0.5, "Fallback: hue range"
    elif h < 100:
        return "mint", 0.5, "Fallback: hue range"
    elif h < 125:
        return "blue", 0.5, "Fallback: hue range"
    elif h < 160:
        return "purple", 0.5, "Fallback: hue range"
    elif h < 170:
        return "pink", 0.5, "Fallback: hue range"
    else:
        return "red", 0.5, "Fallback: high hue"


# ============================================================================
# 🎨 메인 색상 클러스터링 함수
# ============================================================================

def rule_based_color_clustering(hold_data, vectors, config_path="holdcheck/color_ranges.json", 
                                confidence_threshold=0.7, use_hsv=True):
    """
    ⚡ 룰 기반 색상 클러스터링 (CLIP 대체, 초고속)
    
    RGB/HSV 색상 범위로 직접 분류 - CLIP보다 10-20배 빠름!
    사용자 피드백으로 정확도 지속 개선 가능
    
    Args:
        hold_data: 홀드 데이터 (dominant_rgb 또는 dominant_hsv 필요)
        vectors: 사용 안 함 (호환성 유지)
        config_path: 색상 범위 설정 파일 경로
        confidence_threshold: 신뢰도 임계값 (낮으면 unknown으로 분류)
        use_hsv: HSV 공간 사용 여부 (더 정확함)
    
    Returns:
        hold_data: 그룹 정보가 추가된 홀드 데이터
    """
    if len(hold_data) == 0:
        return hold_data
    
    import time
    start_time = time.time()
    
    print(f"\n⚡ 룰 기반 색상 클러스터링 시작 (CLIP 없음, 초고속)")
    print(f"   홀드 개수: {len(hold_data)}개")
    print(f"   색상 공간: {'HSV' if use_hsv else 'RGB'}")
    
    # 색상 범위 로드
    ranges_data = load_color_ranges(config_path)
    colors_config = ranges_data["colors"]
    
    # 각 홀드를 색상으로 분류
    color_groups = {}
    classification_details = []
    
    for hold_idx, hold in enumerate(hold_data):
        # RGB/HSV 값 가져오기
        if "dominant_hsv" in hold:
            h, s, v = hold["dominant_hsv"]
        elif "dominant_rgb" in hold:
            rgb = hold["dominant_rgb"]
            hsv_arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
            hsv_bgr = cv2.cvtColor(hsv_arr, cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = hsv_bgr
        else:
            h, s, v = 0, 0, 128  # 기본값
            rgb = [128, 128, 128]
        
        if "dominant_rgb" not in hold:
            # ⚡ 성능 최적화: HSV → RGB 수학적 변환 (cv2보다 빠름)
            rgb = hsv_to_rgb_fast(h, s, v)
        else:
            rgb = hold["dominant_rgb"]
        
        # 🤖 1단계: ML 모델 예측 시도 (우선 순위)
        ml_color, ml_confidence = None, 0.0
        if not _ml_model_loaded:  # 아직 로드 안 되었으면 시도
            load_ml_color_model()  # 모델 로드 시도
        if _ml_color_model is not None:  # ML 모델이 있을 때만 호출
            ml_color, ml_confidence = predict_with_ml(hold)
        
        if ml_color and ml_confidence >= 0.70:
            # ML 모델이 높은 신뢰도로 예측 → 사용
            color_name = ml_color
            confidence = ml_confidence
            matched_rule = f"ML_Model (confidence: {ml_confidence:.2f})"
        else:
            # ML 모델 없거나 신뢰도 낮음 → 룰 기반 사용
            if use_hsv:
                color_name, confidence, matched_rule = classify_color_by_hsv(
                    h, s, v, rgb, colors_config
                )
            else:
                color_name, confidence, matched_rule = classify_color_by_rgb(
                    rgb, colors_config
                )
        
        # 신뢰도 낮으면 unknown
        if confidence < confidence_threshold:
            color_name = "unknown"
        
        # 홀드에 정보 추가 (CLIP 호환)
        hold["clip_color_name"] = color_name
        hold["clip_confidence"] = confidence
        hold["color_method"] = "ml_model" if ml_color and ml_confidence >= 0.70 else "rule_based"
        hold["matched_rule"] = matched_rule
        
        # 그룹핑
        if color_name not in color_groups:
            color_groups[color_name] = []
        color_groups[color_name].append(hold)
        
        classification_details.append({
            "hold_id": hold.get("id", hold_idx),
            "rgb": rgb,
            "hsv": [h, s, v],
            "color": color_name,
            "confidence": confidence,
            "rule": matched_rule
        })
    
    # 그룹 ID 할당 (색상 이름 기준 정렬)
    color_order = ["black", "white", "red", "orange", "yellow", 
                   "lime", "green", "mint", "blue", "purple", "pink", "brown", "unknown"]
    
    group_idx = 0
    for color_name in color_order:
        if color_name in color_groups:
            for hold in color_groups[color_name]:
                hold["group"] = f"g{group_idx}"
            group_idx += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ 룰 기반 클러스터링 완료 (⚡ {elapsed:.2f}초)")
    print(f"   생성된 그룹 수: {len(color_groups)}개")
    for color_name in color_order:
        if color_name in color_groups:
            count = len(color_groups[color_name])
            avg_conf = np.mean([h["clip_confidence"] for h in color_groups[color_name]])
            print(f"   {color_name}: {count}개 홀드 (평균 신뢰도: {avg_conf:.2f})")
    
    return hold_data

