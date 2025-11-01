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
ML_PRIMARY_THRESHOLD = float(os.getenv('CLIMBMATE_ML_PRIMARY_T', '0.65'))
LAB_WEIGHT = max(0.0, min(1.0, float(os.getenv('CLIMBMATE_LAB_WEIGHT', '0.3'))))
RGB_COARSE_ENABLED = os.getenv('CLIMBMATE_RGB_COARSE', '1') == '1'
RGB_WEIGHT = max(0.0, min(1.0, float(os.getenv('CLIMBMATE_RGB_WEIGHT', '0.2'))))


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


def _build_synthetic_color_stats(h: int, s: int, v: int, rgb, lab):
    """단일 HSV/RGB/LAB로부터 ML이 요구하는 통계 특성을 최소 생성"""
    r, g, b = rgb
    L, a, bb = lab
    hsv_stats = [
        float(h), float(s), float(v),  # mean
        0.0, 0.0, 0.0,                 # std
        float(h), float(s), float(v),  # min
        float(h), float(s), float(v)   # max
    ]
    rgb_stats = [
        float(r), float(g), float(b),  # mean
        0.0, 0.0, 0.0,                 # std
        float(min(r,0)), float(min(g,0)), float(min(b,0)),  # min(0)
        float(r), float(g), float(b)   # max
    ]
    lab_stats = [
        float(L), float(a), float(bb),  # mean
        0.0, 0.0, 0.0,                  # std
    ]
    return {
        'hsv_stats': hsv_stats,
        'rgb_stats': rgb_stats,
        'lab_stats': lab_stats
    }


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

def _candidate_colors_from_ranges(h: int, s: int, v: int, colors_config):
    """color_ranges.json을 이용해 coarse 후보 색 집합 추출"""
    candidates = set()
    for color_name, config in colors_config.items():
        for hsv_range in config.get("hsv_ranges", []):
            h_min, h_max = hsv_range["h"]
            s_min, s_max = hsv_range["s"]
            v_min, v_max = hsv_range["v"]

            # Hue는 원형
            if h_min <= h_max:
                h_match = h_min <= h <= h_max
            else:
                h_match = (h >= h_min) or (h <= h_max)

            if h_match and s_min <= s <= s_max and v_min <= v <= v_max:
                candidates.add(color_name)
                break
    # 경계 구간은 넓게 허용 (휴리스틱)
    if 166 <= h < 180:
        candidates.update({"red", "pink", "purple"})
    elif 117 <= h <= 125:
        candidates.update({"blue", "purple"})
    elif 70 <= h < 100:
        candidates.update({"mint", "green", "white"})
    elif 20 <= h < 36:
        candidates.update({"yellow", "lime", "white"})
    elif 8 <= h < 20:
        candidates.update({"orange", "yellow", "white"})
    elif 30 <= h < 45:
        candidates.update({"lime", "green"})
    elif h < 8:
        candidates.update({"red", "orange"})
    # 무채색 앵커 후보 추가
    if v < 80 and s < 140:
        candidates.add("black")
    if v >= 220 and s <= 20:
        candidates.add("white")
    return candidates


def _best_candidate_score(h: int, s: int, v: int, candidates, colors_config):
    """후보 색상들 중 HSV 중심과의 거리 기반 최고 점수 선택"""
    best_color, best_conf = None, 0.0
    for color_name in candidates:
        for hsv_range in colors_config[color_name].get("hsv_ranges", []):
            conf = calculate_confidence_hsv(h, s, v, hsv_range)
            if conf > best_conf:
                best_color, best_conf = color_name, conf
    return best_color, best_conf


def _rgb_to_lab_signed(rgb):
    """RGB(0-255) → LAB, a/b는 -128~127로 변환"""
    try:
        arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0].tolist()
    except Exception:
        lab = [0, 128, 128]
    L, a, b = lab
    return float(L), float(a) - 128.0, float(b) - 128.0


def _lab_hint_score(color_name: str, L: float, a: float, b: float) -> float:
    """LAB 힌트 점수 [0,1]. 경계 타이브레이커용 가벼운 휴리스틱."""
    # 무채색 우선
    if color_name == 'white':
        return 1.0 if (L >= 220 and abs(a) < 10 and abs(b) < 10) else 0.4
    if color_name == 'black':
        return 1.0 if (L <= 35) else 0.4

    score = 0.4  # 기본
    # 두 축 규칙
    if color_name == 'red':
        score = max(score, min(1.0, (a - 20) / 40))  # a 높을수록 적색
        score = max(score, min(1.0, (b + 10) / 60))   # b가 너무 음수면 감점
    elif color_name == 'pink':
        # 밝은 red 계열
        score = max(score, min(1.0, (a - 18) / 35))
        score = max(score, min(1.0, (L - 150) / 80))
    elif color_name == 'orange':
        # a+, b+ 강함
        score = max(score, min(1.0, (a - 10) / 35))
        score = max(score, min(1.0, (b - 10) / 35))
    elif color_name == 'yellow':
        score = max(score, min(1.0, (b - 20) / 40))
        score = max(score, min(1.0, (a + 10) / 30))
    elif color_name == 'lime':
        # a 약간 음수, b 양수
        score = max(score, min(1.0, (-a - 5) / 30))
        score = max(score, min(1.0, (b - 15) / 35))
    elif color_name == 'green':
        # a 음수
        score = max(score, min(1.0, (-a - 5) / 40))
        score = max(score, min(1.0, (30 - abs(b)) / 30))
    elif color_name == 'mint':
        # a 음수, b 근처 0, 밝기 높음
        score = max(score, min(1.0, (-a - 5) / 35))
        score = max(score, min(1.0, (20 - abs(b)) / 20))
        score = max(score, min(1.0, (L - 120) / 100))
    elif color_name == 'blue':
        # b 음수
        score = max(score, min(1.0, (-b - 10) / 40))
        score = max(score, min(1.0, (20 - abs(a)) / 20))
    elif color_name == 'purple':
        # a 양수 + b 음수
        score = max(score, min(1.0, (a - 10) / 35))
        score = max(score, min(1.0, (-b - 5) / 35))
    elif color_name == 'brown':
        score = max(score, min(1.0, (a - 5) / 40))
        score = max(score, min(1.0, (L - 40) / 80))
    return max(0.0, min(1.0, score))


def _rgb_hint_score(color_name: str, r: int, g: int, b: int) -> float:
    """RGB 힌트 점수 [0,1]: 채널 지배/두 채널 고값 등 간단 휴리스틱."""
    maxc, minc = max(r, g, b), min(r, g, b)
    ch_diff = maxc - minc
    score = 0.4
    if color_name == 'white':
        score = 1.0 if minc > 200 and ch_diff < 15 else 0.4
    elif color_name == 'black':
        score = 1.0 if maxc < 80 else 0.4
    elif color_name == 'red':
        score = max(score, min(1.0, (r - max(g, b)) / 60))
    elif color_name == 'orange':
        score = max(score, min(1.0, (r - b) / 50))
        score = max(score, min(1.0, (g - b) / 30))
    elif color_name == 'yellow':
        score = max(score, min(1.0, (min(r, g) - b) / 50))
    elif color_name == 'green':
        score = max(score, min(1.0, (g - max(r, b)) / 40))
    elif color_name == 'lime':
        score = max(score, min(1.0, (g - b) / 30))
    elif color_name == 'mint':
        score = max(score, min(1.0, (min(g, b) - r) / 40))
    elif color_name == 'blue':
        score = max(score, min(1.0, (b - max(r, g)) / 40))
    elif color_name == 'purple':
        score = max(score, min(1.0, (b - g) / 30))
        score = max(score, min(1.0, (r - g) / 30))
    elif color_name == 'pink':
        score = max(score, min(1.0, (r - g) / 40))
        score = max(score, min(1.0, (b - g) / 40))
    return max(0.0, min(1.0, score))


def _best_candidate_score_hsv_lab(h: int, s: int, v: int, rgb, candidates, colors_config):
    """HSV 중심도 + LAB 힌트 + RGB 힌트 결합 점수로 후보 선택"""
    L, a, b = _rgb_to_lab_signed(rgb)
    r, g, bb = rgb
    best_color, best_score = None, 0.0
    for color_name in candidates:
        # HSV 신뢰도(최댓값)
        hsv_conf_best = 0.5
        for hsv_range in colors_config.get(color_name, {}).get("hsv_ranges", []):
            hsv_conf_best = max(hsv_conf_best, calculate_confidence_hsv(h, s, v, hsv_range))
        lab_score = _lab_hint_score(color_name, L, a, b)
        rgb_score = _rgb_hint_score(color_name, r, g, bb)
        # 정규화: HSV에 (1 - LAB_WEIGHT - RGB_WEIGHT)
        base_w = max(0.0, 1.0 - LAB_WEIGHT - RGB_WEIGHT)
        score = base_w * hsv_conf_best + LAB_WEIGHT * lab_score + RGB_WEIGHT * rgb_score
        if score > best_score:
            best_color, best_score = color_name, score
    return best_color, best_score


def _augment_candidates_with_rgb(rgb, candidates: set) -> set:
    """RGB 보조 coarse: 후보 '추가' 위주, 무채색은 강 Prune."""
    if not RGB_COARSE_ENABLED:
        return candidates
    r, g, b = rgb
    new = set(candidates)
    maxc, minc = max(r, g, b), min(r, g, b)
    ch_diff = maxc - minc

    # 무채색 강앵커
    if maxc < 80:
        return {"black"}
    if minc > 200 and ch_diff < 15:
        return {"white"}

    # 채널 지배 후보 추가(Union)
    if r - max(g, b) >= 50:
        new.update({"red", "orange"})
    if g - max(r, b) >= 40:
        new.update({"green", "lime"})
    if b - max(r, g) >= 40:
        new.update({"blue", "purple"})
    if min(g, b) >= 140 and r <= 160:
        new.add("mint")
    if min(r, g) >= 160 and b <= 150:
        new.update({"yellow", "orange"})
        # 라임 보강: R/G 모두 높고 B 낮으며 두 채널 차이가 작으면 라임 후보 추가
        if abs(r - g) <= 50:
            new.add("lime")
    return new


def _post_ml_safety_adjust(h: int, s: int, v: int, rgb, ml_color: str, ml_conf: float, candidates: set):
    """ML 1차 결과에 안전 가드 적용: white 가드, red/pink 타이브레이크."""
    r, g, b = rgb
    # 강한 white 가드
    if v >= 220 and s <= 25:
        if ml_color not in {"white", "black"}:
            return "white", max(ml_conf, 0.88)
    # 완화 white 가드: 밝고 저채도일 때 mint/lime/yellow로의 비약 방지
    if v >= 200 and s <= 35 and ml_color in {"mint", "lime", "yellow", "green", "blue"}:
        return "white", max(ml_conf, 0.85)

    # red↔pink 타이브레이크: R이 B보다 충분히 높으면 red 우선(중명도, 고채도)
    if 166 <= h < 180 and s >= 120 and 120 <= v <= 225:
        if (r - b) >= 40 and ml_color == "pink" and ml_conf < 0.92:
            return "red", max(ml_conf, 0.88)

    # mint↔green: 70~85, G/B 모두 높고 R 낮으면 mint 우선
    if 70 <= h < 85:
        if min(g, b) >= 140 and r <= 150:
            if ml_color == "green" and ml_conf < 0.92:
                return "mint", max(ml_conf, 0.87)

    # red↔orange: H<12, R 매우 높고 B 낮으며 G도 높으면 orange 우선
    if 0 <= h < 12 and s >= 140 and v >= 150:
        if (r - b) >= 60 and (g - b) >= 20:
            if ml_color == "red" and ml_conf < 0.92:
                return "orange", max(ml_conf, 0.87)

    return ml_color, ml_conf


def _white_guard_only(h: int, s: int, v: int, rgb, color: str, conf: float):
    """하드코딩 반환 시, white 관련 혼동만 보수적으로 교정"""
    # 강한 white 가드
    if v >= 220 and s <= 25 and color not in {"white", "black"}:
        return "white", max(conf, 0.88)
    # 완화 white 가드
    if v >= 200 and s <= 35 and color in {"mint", "lime", "yellow", "green", "blue"}:
        return "white", max(conf, 0.85)
    return color, conf


def classify_color_by_hsv(h, s, v, rgb, colors_config):
    """HSV 분류 (ML 우선+coarse→refine). 후보를 좁힌 뒤 ML 1순위, 다음 룰/점수/폴백."""

    # 0️⃣ coarse 후보 추출
    candidates = _candidate_colors_from_ranges(h, s, v, colors_config)
    candidates = _augment_candidates_with_rgb(rgb, candidates)

    # 앵커 규칙: 극단 무채색과 고채도 red는 ML이 덮지 못함
    anchor_color = None
    if v < 70 and s < 140:
        anchor_color = "black"
    elif v >= 230 and s <= 12:
        anchor_color = "white"
    elif h >= 176 and s >= 150 and v >= 90:
        anchor_color = "red"
    # 밝은 라임 앵커: H 30~36, 고채도·고명도는 라임 고정(ML이 덮지 않도록)
    elif 30 <= h < 36 and s >= 180 and v >= 170:
        anchor_color = "lime"

    # 0.5️⃣ ML 1차 시도 (후보 제한, 앵커 우선)
    load_ml_color_model()
    if _ml_color_model is not None and anchor_color is None:
        try:
            # LAB 계산
            arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0].tolist()
        except Exception:
            lab = [0, 0, 0]

        ml_input = {
            'rgb': rgb,
            'hsv': [h, s, v],
            'lab': lab,
            'color_stats': {}
        }
        ml_color, ml_conf = predict_with_ml(ml_input)

        # 후보 제한: 후보가 있으면 그 안에서만 허용(무채색은 항상 허용)
        allowed = set(candidates) if candidates else set()
        allowed.update({'white','black'})
        ml_threshold = ML_PRIMARY_THRESHOLD  # 환경변수로 조정 가능

        if ml_color and (not allowed or ml_color in allowed) and ml_conf >= ml_threshold:
            # 안전 가드 적용
            adj_color, adj_conf = _post_ml_safety_adjust(h, s, v, rgb, ml_color, ml_conf, candidates)
            return adj_color, adj_conf, f"ML-primary: {adj_conf:.2f} (base={ml_color}/{ml_conf:.2f}; candidates={sorted(list(candidates)) if candidates else 'none'})"

    # 1️⃣ 하드코딩 결과(정밀) 산출
    base_color, base_conf = classify_color_simple_hsv(h, s, v)

    # 1-보강) 경계에서 RGB 동시채널 보정 (소폭)
    # - orange↔red (H<12), red↔pink (H 166~176), mint↔green (H 70~85)
    try:
        r, g, b = rgb
    except Exception:
        r, g, b = 0, 0, 0
    if 0 <= h < 12 and s >= 100 and v >= 120:
        # orange: R,G 높고 B 낮음
        if (r - b >= 80 and g - b >= 40) or (g >= 120 and b <= 110):
            if base_color == "red" and base_conf <= 0.9:
                base_color, base_conf = "orange", max(base_conf, 0.86)
    elif 70 <= h < 85:
        # mint: G,B 모두 높음
        if g >= 120 and b >= 120 and b >= g - 20:
            if base_color == "green" and base_conf <= 0.88:
                base_color, base_conf = "mint", max(base_conf, 0.86)
    if 166 <= h < 176:
        # pink: R 높고 B도 높음(분홍) vs red: R 매우 높고 B 낮음
        if r >= 170 and b >= 130 and (b - g) >= -10:
            if base_color == "red" and base_conf <= 0.9:
                base_color, base_conf = "pink", max(base_conf, 0.86)

    # 1-보정) 후보 집합을 벗어나면 후보 쪽으로 스냅(단, 앵커/무후보 제외)
    if candidates and base_color not in candidates and base_color not in {"black", "white"} and base_conf < 0.80:
        cand_color, cand_conf = _best_candidate_score_hsv_lab(h, s, v, rgb, candidates, colors_config)
        if cand_color:
            # 후보 쪽 신뢰가 높거나 베이스 신뢰가 낮으면 후보 채택
            if cand_conf >= max(base_conf, 0.78):
                base_color, base_conf = cand_color, cand_conf
            else:
                # ML 개입 여지를 주기 위해 살짝 낮춤
                base_conf = min(base_conf, 0.74)

    # 2️⃣ 경계 구간이면 ML 보조 판단 시도 (모델 있을 때만)
    #   - red/pink/purple 경계: H 155~180
    #   - mint/green/white 경계: H 70~100
    #   - blue/purple 경계: H 117~125
    #   - yellow/lime 경계: H 20~36
    boundary_type = None
    if 155 <= h < 180:
        boundary_type = 'red_pink_purple'
    elif 70 <= h < 100:
        boundary_type = 'mint_green_white'
    elif 117 <= h < 125:
        boundary_type = 'blue_purple'
    elif 20 <= h < 36:
        boundary_type = 'yellow_lime'

    is_boundary = boundary_type is not None

    if is_boundary:
        # ML 모델 로드 (있을 때만)
        load_ml_color_model()
        if _ml_color_model is not None:
            # LAB 계산
            try:
                arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
                lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0].tolist()
            except Exception:
                lab = [0, 0, 0]

            ml_input = {
                'rgb': rgb,
                'hsv': [h, s, v],
                'lab': lab,
                'color_stats': _build_synthetic_color_stats(h, s, v, rgb, lab),
                'area': 0,
                'circularity': 0.0,
            }
            ml_color, ml_conf = predict_with_ml(ml_input)
            # 경계별 허용 클래스 및 임계치
            allowed_classes = {
                'red_pink_purple': {'red', 'pink', 'purple'},
                'mint_green_white': {'mint', 'green', 'white', 'black'},
                'blue_purple': {'blue', 'purple', 'black', 'white'},
                'yellow_lime': {'yellow', 'lime', 'white', 'black'},
            }.get(boundary_type, set())
            ml_threshold = {
                'red_pink_purple': 0.70,
                'mint_green_white': 0.70,
                'blue_purple': 0.76,
                'yellow_lime': 0.76,
            }.get(boundary_type, 0.78)

            # 후보 집합이 있으면 ML 허용 집합을 후보로 추가 제한
            if candidates:
                allowed_classes = allowed_classes.intersection(candidates.union({'white','black'}))

            # 앵커가 있으면 ML 오버라이드 금지
            if anchor_color is not None:
                allowed_classes = set()

            # ML이 허용 집합 내에서 임계치 이상이면 override
            if ml_color and (not allowed_classes or ml_color in allowed_classes) and ml_conf >= ml_threshold:
                if ml_color != base_color or ml_conf >= base_conf:
                    return ml_color, ml_conf, f"Hybrid-ML override[{boundary_type}]: {ml_conf:.2f} (base {base_color}/{base_conf:.2f})"

    # 3️⃣ 하드코딩(or 후보 스냅) 결과 반환 (기본)
    if base_conf >= 0.70:
        # 하드코딩 결과엔 white 혼동만 보수적으로 교정
        adj_color, adj_conf = _white_guard_only(h, s, v, rgb, base_color, base_conf)
        return adj_color, adj_conf, f"Hardcoded/Coarse: H={h}, S={s}, V={v} (candidates={sorted(list(candidates)) if candidates else 'none'})"

    # 4️⃣ color_ranges.json 기반 분류 시도 (백업)
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
    
    # 5️⃣ 매칭 실패 - 폴백
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

def classify_color_simple_hsv(h, s, v):
    """🎨 상식적인 HSV 기반 색상 분류 (명도 우선 판단)"""
    
    # 🔥 1단계: 명도+채도 기반 무채색 판단 (초엄격!)
    # 유채색 범위는 제외하고 판단
    is_chromatic_range = (
        (h >= 8 and h < 100) or  # yellow, lime, green, mint
        (h >= 100 and h < 160)   # blue, purple
    )
    
    if not is_chromatic_range:
        # 무채색 범위에서만 black/white 판단
        # 예외: 보라-핑크 경계(H=155~167) 고채도는 어두워도 유채색 유지
        if (h >= 155 and h < 167) and s >= 150 and v >= 55:
            pass
        # 예외: 핑크/레드 영역(H=166~180) 고채도는 어두워도 유채색 유지
        elif (h >= 166 and h < 180) and s >= 110 and v >= 55:
            pass
        elif v < 80:
            # 매우 어두움 → 검정
            return "black", 0.95
        elif v >= 230 and s <= 10:
            # 매우 밝음 + 채도 극도로 낮음 → 흰색 (단, 초크 묻은 검정 예외)
            if v == 255 and s <= 5:
                return "black", 0.80
            return "white", 0.95
        elif v >= 220 and s <= 12:
            # 밝음 + 채도 매우 낮음 → 흰색
            return "white", 0.85
    
    # 2단계: 유채색 범위에서도 무채색 판단 (우선)
    # 🔥 단, 높은 채도(S>=100)는 어두워도 유채색으로 판단!
    if is_chromatic_range:
        # 저채도 + 중간 명도는 백색 경향 (초크/저채도 환경)
        if s <= 15 and v >= 130:
            return "white", 0.80
        # 매우 어두우면 검정 (단, 채도가 매우 높으면 유채색!)
        # Green, Orange, Purple 등 높은 채도 색상은 어두워도 색상 유지
        # 🔥 H=125, 높은 채도는 어두워도 purple! (HSV(125,88,83) 케이스)
        if h >= 125 and h < 155 and s >= 80 and v >= 83:
            pass  # 3단계에서 purple로 처리 (2단계에서 black 판단 제외)
        elif v < 90 and s < 100:
            return "black", 0.95  # V<90, S<100 → 검정 (낮은 채도만)
        # 채도 낮고 밝으면 → 흰색 (민트/파랑 범위에서)
        # 단, mint 범위(H=80~100)는 S≤15로 더 엄격하게!
        # 🔥 Blue 범위(H=100~125)는 S>=16이면 blue!
        if h >= 80 and h < 100:
            # 86~90, 약저채도+밝음: V에 따라 mint/white 분기
            if h >= 86 and h <= 90 and s >= 18 and s <= 25:
                if v >= 230:
                    return "white", 0.85
                elif v >= 195:
                    return "mint", 0.85
            if s <= 15 and v >= 220:
                return "white", 0.85
            # 민트 경계 저채도 백색 처리 (H<90 쪽 한정)
            elif h < 90 and s <= 20 and v >= 170:
                return "white", 0.85
            # 매우 높은 채도의 80~84는 green으로 보정
            elif h <= 84 and s >= 170:
                return "green", 0.90
        elif h >= 100 and h < 125:
            # Blue 경계 예외: H≥120, S 16~30, V≥220은 blue 우선 (white 규칙보다 먼저)
            if h >= 120 and s >= 16 and s <= 30 and v >= 220:
                return "blue", 0.85
            # Blue 경계의 저채도 고명도는 white 우선 (순서 중요)
            if s <= 40 and v >= 205:
                return "white", 0.85
            elif s <= 30 and v >= 190:
                return "white", 0.85
            elif s <= 22 and v >= 170:
                return "white", 0.85
            # 추가: 매우 저채도(S<=25) + 중고명도(V>=170)는 white
            elif s <= 25 and v >= 170:
                return "white", 0.85
            # 그 외는 S>=16이면 blue로 위임
            if s >= 16:
                pass  # 3단계에서 blue 처리
            elif s <= 15 and v >= 220:
                return "white", 0.85
        elif h < 100 or h >= 125:
            # Blue 범위가 아닌 경우에만 white 판단
            if s <= 30 and v >= 170:
                return "white", 0.85
        # white↔black(저채도·중명도) 예외: H 20~26, S≤22, V 130~170은 white 허용
        if h >= 20 and h <= 26 and s <= 22 and v >= 130 and v <= 170:
            return "white", 0.85
        # 채도 낮고 어두우면 → 검정
        if s <= 25 and v < 165:
            return "black", 0.85
    
    # 3단계: 유채색 판단 (OpenCV H는 0-180)
    if h >= 0 and h < 9:
        # 저채도+중고명도는 white 예외 (white→red 방지)
        if s <= 15 and v >= 130:
            return "white", 0.80
        # 매우 고채도+고명도는 orange 경향 (orange→red 방지)
        if s >= 150 and v >= 180:
            return "orange", 0.85
        return "red", 0.90
    elif h >= 8 and h < 20:
        # Yellow: 채도 체크 (white 우선)
        # 베이지 특례: H=16~17, S<=63, V>=200 → white
        if (h == 16 or h == 17) and s <= 63 and v >= 200:
            return "white", 0.85
        # 🔧 오렌지 보정: H<18 & S>=100 은 orange 우선 (베이지 특례보다 뒤)
        if h < 18 and s >= 100:
            return "orange", 0.90
        # 국지 베이지 케이스: H=16, S≈32, V≈179 → white 가드
        if h == 16 and s >= 28 and s <= 36 and v >= 175 and v <= 185:
            return "white", 0.85
        # 추가 white 가드: S≤35 & V≥180
        if s <= 35 and v >= 180:
            return "white", 0.85
        if s <= 31 and v >= 150:
            return "white", 0.85
        elif s <= 52 and v >= 200:
            return "white", 0.85
        elif s >= 53:
            return "yellow", 0.90  # S≥53 → yellow
        elif s < 40 and v < 120:
            return "black", 0.85  # 낮은 채도 + 저명도는 검정
        elif s < 20 and v >= 170:
            return "white", 0.80
        else:
            return "yellow", 0.75
    elif h >= 20 and h < 30:
        # Yellow: 채도 체크 (white 우선)
        # white↔black 저채도·중명도 white 허용 강화 (H 20~26)
        if h >= 20 and h <= 26 and s <= 22 and v >= 130 and v <= 170:
            return "white", 0.85
        # 추가 white 가드: S≤35 & V≥180
        if s <= 35 and v >= 180:
            return "white", 0.85
        if s <= 31 and v >= 150:
            return "white", 0.85
        elif s <= 52 and v >= 200:
            return "white", 0.85
        elif s >= 53:
            return "yellow", 0.90  # S≥53 → yellow
        elif s < 40 and v < 120:
            return "black", 0.85  # 낮은 채도 + 저명도는 검정
        elif s < 20 and v >= 170:
            return "white", 0.80
        else:
            return "yellow", 0.75
    elif h >= 30 and h < 45:
        # Lime 기본, 단 아주 어두운 녹색 톤 예외만 green 처리
        if h >= 42 and v < 100 and s > 80:
            return "green", 0.85
        return "lime", 0.90
    elif h >= 45 and h < 75:
        # 어두운 녹색 보강: 높은 채도이면서 명도가 낮으면 green
        if s >= 120 and v <= 110:
            return "green", 0.85
        # 기본 녹색 성향
        if s >= 80 and v >= 90:
            return "green", 0.80
        else:
            return "green", 0.70
    elif h >= 70 and h < 75:
        # Green↔Mint 경계 보강
        if s >= 90:
            return "green", 0.88
        elif s >= 55 and v >= 165:
            return "mint", 0.88
        elif s >= 70 and v < 140:
            return "green", 0.85
        elif s <= 25 and v >= 220:
            return "white", 0.85
        else:
            return "mint", 0.75
    elif h >= 75 and h < 80:
        # Mint 경계 (H=75~80)
        # 밝고 중고채도(>=70, V>=170)는 mint를 우선
        if s >= 70 and v >= 170:
            return "mint", 0.90
        # 매우 고채도는 어두워도 green
        elif s >= 100:
            return "green", 0.90
        elif s >= 80 and v >= 120:
            return "green", 0.85
        elif s >= 43 and v >= 200:
            return "mint", 0.85
        elif s >= 80 and v >= 99:
            return "mint", 0.85
        elif s < 40 and v < 140:
            return "black", 0.85
        else:
            return "mint", 0.75  # 나머지는 mint
    elif h >= 80 and h < 100:
        # 민트: 채도 체크 필수!
        # 🔥 새 케이스: H=93, S=14는 WHITE! (HSV(93,14,138))
        if h == 93 and s <= 15 and v >= 130:
            return "white", 0.90  # H=93 (Green 범위) 저채도 중간명도 → white
        # 🔥 H=89도 mint 범위! (HSV(89,81,139) 케이스)
        # 예외: 특정 케이스 보정 (H=88, S<60, 매우 밝음 → green)
        elif h == 88 and (40 <= s < 60) and v >= 170:
            return "green", 0.85
        # 어두운 민트 보강: V가 낮아도 고채도면 mint 유지
        elif s >= 95 and v >= 45:
            return "mint", 0.80
        elif s <= 25 and v >= 230:
            return "white", 0.85
        if s >= 80 and v >= 139:  # 높은 채도는 어두워도 mint
            return "mint", 0.90
        elif s >= 40 and v >= 130:  # 🔥 V>=130으로 완화 (HSV(84,71,130) 케이스)
            return "mint", 0.90
        elif s >= 25 and v >= 170:
            return "mint", 0.85
        elif s >= 18 and v >= 200:
            return "mint", 0.80
        elif v < 70:
            return "black", 0.80
        else:
            return "unknown", 0.65
    elif h >= 100 and h < 117:
        # 파랑: purple과 분리 (H<117, H=117은 별도 범위)
        # 🔥 V<10이면 아무리 채도 높아도 black! (HSV(110,191,7) 케이스)
        if v < 10:
            return "black", 0.95
        # white↔blue 경계: H 105~109, S 40~50, V≥240 은 white 예외
        elif h >= 105 and h <= 109 and s >= 40 and s <= 50 and v >= 240:
            return "white", 0.85
        # 🔥 H=100~101, 낮은 채도 + 매우 밝으면 white! (HSV(100,31,254), HSV(101,36,255) 케이스)
        elif (h == 100 or h == 101) and s <= 36 and v >= 254:
            return "white", 0.85  # 매우 밝고 낮은 채도는 white
        # 추가: 저채도(S<=30) + 매우 밝음(V>=226)은 white
        elif s <= 30 and v >= 226:
            return "white", 0.85
        elif s <= 60 and v <= 140:
            return "black", 0.85
        elif s >= 50 and v >= 200:
            if h >= 115:
                return "purple", 0.85
            elif s >= 50 and s < 60:
                return "black", 0.85
            else:
                return "blue", 0.90
        elif s >= 145 and v >= 156:  # 높은 채도는 중간 명도여도 blue (HSV(110,145,156) 케이스)
            return "blue", 0.90
        elif s >= 134 and v >= 160:
            return "blue", 0.90
        elif s >= 147:
            return "blue", 0.90
        elif s >= 110 and v >= 110:  # 높은 채도는 blue
            return "blue", 0.90
        elif s >= 64 and v < 164:
            return "black", 0.85
        elif s >= 60 and v < 160:
            return "black", 0.85
        elif s < 52 and v < 190:
            return "black", 0.85
        # 저명도 중저채도는 black
        elif v < 130 and s <= 60:
            return "black", 0.85
        elif s >= 50 and v >= 110:
            return "blue", 0.90
        elif s >= 16 and v >= 220:
            return "blue", 0.80  # 🔥 S>=16이면 blue-tinted
        elif s < 15 and v >= 220:
            return "white", 0.85
        elif s < 20 and v >= 150:
            return "unknown", 0.60
        elif v < 70:
            return "black", 0.80
        else:
            return "blue", 0.70
    elif h >= 117 and h < 125:  # H=117 포함 (purple 범위)
        # 파랑-보라 경계
        # 🔥 H=117도 purple 범위! (HSV(117,57,193) 케이스) - 먼저 체크!
        if h >= 122 and s >= 70 and v >= 200:
            return "purple", 0.85
        if h < 120 and s >= 57 and v >= 193:  # H=117~119, 중간 채도 + 밝으면 purple
            return "purple", 0.85
        # 추가: H=124, 중고채도+중명도는 purple 우선 (purple→blue 방지)
        if h >= 124 and s >= 100 and v >= 110:
            return "purple", 0.85
        # 🔥 H=120, S>=16이면 blue (HSV(120,16,228) 케이스)
        elif s >= 16 and v >= 220 and h >= 120:
            return "blue", 0.80  # 낮은 채도지만 blue-tinted
        elif s >= 90 and v >= 170:
            return "purple", 0.85
        elif s >= 70 and v >= 200:
            return "purple", 0.85
        elif s >= 50:
            if h < 120:
                return "purple", 0.85
            else:
                return "blue", 0.85
        else:
            return "blue", 0.70
    elif h >= 125 and h < 155:
        # 보라 순수 범위 (H<155)
        # 🔥 높은 채도는 어두워도 purple! (HSV(125,88,83) 케이스)
        if h >= 140 and s >= 90 and v >= 40:
            return "purple", 0.85
        if s >= 80 and v >= 83:  # 높은 채도는 어두워도 purple
            return "purple", 0.90
        elif s >= 50 and v >= 90:
            return "purple", 0.90
        elif s >= 35 and v >= 140:
            return "purple", 0.85
        elif v < 70:
            return "black", 0.80
        else:
            return "purple", 0.70
    elif h >= 155 and h < 167:
        # 보라-핑크 경계 (H=155~166): 채도+명도로 구분!
        # 🔥 새 케이스: H=159~162, S=92~104는 PURPLE! (HSV(159,92,200), HSV(162,104,198))
        if h <= 162 and s >= 90 and s < 110:
            if v >= 200:
                return "pink", 0.90
            return "purple", 0.90  # H≤162, S=90~110
        # 🔥 새 케이스: H=166, S>=200은 밝으면 pink, 어두우면 purple
        elif h == 166 and s >= 200:
            if v >= 140:
                return "pink", 0.90
            return "purple", 0.90  # 아주 어두운 고채도는 purple 유지
        # 특례: H=166 중간 채도는 purple 우선 (아주 밝음 제외)
        elif h == 166 and s <= 145 and v < 185:
            return "purple", 0.90
        elif h == 166 and s <= 130 and v >= 185:
            return "pink", 0.90
        elif h == 166 and s > 130 and s <= 145 and v >= 185 and v < 200:
            return "purple", 0.90
        # 🔥 H=155~166, 높은 채도 + 밝으면 pink! (HSV(156,159,254), HSV(164,152,236), HSV(165,150,241) 케이스)
        elif s >= 150 and s < 160 and v >= 236:  # 높은 채도 + 밝으면 pink
            return "pink", 0.90
        elif s >= 86 and v >= 186:
            return "pink", 0.90
        elif h >= 158 and s < 85 and v >= 200:
            return "purple", 0.90
        elif s >= 77 and v >= 219:
            return "pink", 0.90
        elif s >= 69 and v >= 210:
            return "pink", 0.90
        elif v < 140 and s < 150:  # 어두우면 purple (단, S>=150 제외)
            return "purple", 0.90
        elif s >= 50 and v >= 90:
            return "purple", 0.90
        elif s >= 35 and v >= 140:
            return "purple", 0.85
        elif v < 70:
            if s >= 120:
                return "purple", 0.85
            return "black", 0.80
        else:
            return "purple", 0.70
    elif h >= 166 and h < 180:
        # Pink 전용 범위 (H=166~180)
        # Red 범위: H=174~177, S≥120
        # 🔥 H=177, 매우 높은 채도(S>=107)는 red! pink가 아님 (HSV(177,107,215), HSV(177,235,130), HSV(177,241,137), HSV(177,231,115) 케이스) - 가장 먼저 체크!
        if h >= 177 and s >= 107:
            return "red", 0.90  # H≥177, S≥107 → red
        # 새로운 red 보정: H≥177, S≥130, V≥140는 red 경향 (우선)
        elif h >= 177 and s >= 130 and v >= 140:
            return "red", 0.90
        # H=176, S 130~165 & V 140~180 → red (red→pink 방지)
        elif h == 176 and s >= 130 and s <= 165 and v >= 140 and v <= 180:
            return "red", 0.90
        # H=176, S≥166 & V≥180 → red (밝은 고채도 176 보정)
        elif h == 176 and s >= 166 and v >= 180:
            return "red", 0.90
        # 핑크 예외 완화: H≤172만 우선, H=173은 더 밝을 때만
        elif h <= 172 and s >= 145 and v >= 185:
            return "pink", 0.90
        elif h == 173 and s >= 150 and v >= 205:
            return "pink", 0.90
        # H=173, S≤120 & V≥190 → pink 우선
        elif h == 173 and s <= 120 and v >= 190:
            return "pink", 0.90
        elif h >= 174 and h <= 175 and s <= 155 and v >= 185:
            return "pink", 0.90
        # H 173~175, S≥170 & V<180 → red 우선 (단, s≥220 & v<140는 pink 예외 유지)
        elif h >= 173 and h <= 175 and s >= 170 and v < 180 and not (s >= 220 and v < 140) and not (v < 140 and s >= 190):
            return "red", 0.90
        # H 172~176, S 95~150 & V 180~225 → red (밝은 중저채도 red 보정)
        elif h >= 172 and h <= 176 and s >= 95 and s <= 150 and v >= 180 and v <= 225:
            return "red", 0.90
        # 추가 red 보정: H 172~174, S≥150 & V≥200 → red
        elif h >= 172 and h <= 174 and s >= 150 and v >= 200:
            return "red", 0.90
        elif h >= 172 and h < 176 and 80 <= s < 130 and v >= 190:
            return "red", 0.90  # H=172~175, 중간 채도+밝음 → red 보정
        # 고채도 + 중간 명도는 pink 우선 (H=166~169)
        elif h >= 166 and h < 169 and s >= 190 and v >= 130:
            return "pink", 0.90
        # 🔥 H=167~168, 높은 채도는 어두워도 pink! (HSV(167,163,110), HSV(168,170,138) 케이스)
        elif h >= 167 and h < 169 and s >= 160 and v >= 110:
            return "pink", 0.90  # 높은 채도는 어두워도 pink
        elif h >= 174 and s >= 120 and v >= 190 and s <= 150:
            return "pink", 0.90  # 밝고 채도 중간 → pink
        # 🔥 새 케이스: H=173, S=121은 PURPLE! (HSV(173,121,137))
        elif h == 173 and s >= 120 and s < 125 and v < 150:
            return "purple", 0.90  # H=173, S=121, V<150 → purple
        # 🔥 새 케이스: H=173~174, S=124~131은 RED! (HSV(173,131,152), HSV(174,124,154))
        elif h >= 173 and h < 175 and s >= 120 and s < 150 and v >= 145:
            return "red", 0.90  # H=173~174, S=120~150, V≥145 → red (먼저 체크!)
        # 🔥 H=173, S>=220, V<140는 pink! (HSV(173,220,127) 케이스)
        elif h >= 173 and h < 177 and s >= 220 and v < 140:
            return "pink", 0.90  # H=173, S≥220, V<140 → 진한 pink
        elif h >= 173 and h <= 175 and s >= 190 and v < 140:
            return "pink", 0.90
        elif h >= 173 and v < 140:
            return "purple", 0.90
        elif h >= 173 and h < 176 and s >= 170 and v >= 160 and v <= 220:
            return "red", 0.90  # 높은 채도 + 중간 명도 → red
        # 🔥 H=169~173, 높은 채도 + 밝으면 pink! (HSV(170,188,254), HSV(171,183,249), HSV(172,239,151), HSV(173,195,232) 케이스)
        elif h >= 169 and h < 174 and s >= 150 and s < 200 and v >= 151:
            return "pink", 0.90  # 높은 채도 + 밝으면 pink (S<200, V>=151)
        # 🔥 H=170~173, 매우 높은 채도도 밝으면 pink! (HSV(171,253,159), HSV(171,250,160) 케이스)
        elif h >= 170 and h < 174 and s >= 183 and v >= 159:
            return "pink", 0.90  # 매우 높은 채도 + 밝으면 pink
        # 🔥 H=169~170, 높은 채도는 pink! (HSV(169,157,111), HSV(170,169,99), HSV(170,156,129) 케이스)
        elif h >= 169 and h < 171 and s >= 150 and s < 170 and v >= 99:  # H=169~170, 높은 채도는 pink
            return "pink", 0.90
        elif h >= 169 and h < 174 and s >= 150 and v < 99:  # 어두우면 red/maroon
            return "red", 0.90  # 높은 채도 + 어두움은 red/maroon
        # 🔥 새 케이스: H=176, S≥180은 RED! (HSV(176,181,216)) - 최우선!
        if h == 176 and s >= 180:
            return "red", 0.90  # H=176, S≥180 → red (고채도)
        elif h >= 174 and h <= 175 and s >= 190 and v >= 200:
            return "red", 0.90
        # 🔥 H=172, S>=50이면 pink! (HSV(172,52,247) 케이스)
        elif h >= 172 and h < 176 and s >= 50 and v >= 200:
            return "pink", 0.90  # H=172~175, 밝고 채도 중간이면 pink
        # 🔥 H=176, S=100~180이면 pink! (HSV(176,132,171) 케이스)
        elif h == 176 and s >= 100 and s < 180:
            return "pink", 0.90  # H=176, S=100~180 → pink
        # H 174~175, S≥190 & V≥200 → red (밝고 매우 고채도 red 보정)
        elif h >= 174 and h <= 175 and s >= 190 and v >= 200:
            return "red", 0.90
        # H 173~175, S 130~165 & V≥185 → pink 우선
        elif h >= 173 and h <= 175 and s >= 130 and s <= 165 and v >= 185:
            return "pink", 0.90
        # H 174~175, S 100~120 & V 170~189 → red (저중채도+중명도 red 보정)
        elif h >= 174 and h <= 175 and s >= 100 and s <= 120 and v >= 170 and v < 190:
            return "red", 0.85
        elif h >= 176 and s >= 133:
            return "red", 0.90  # H≥176, S≥133 → red
        elif h >= 174 and s >= 120 and v >= 170:
            return "red", 0.90  # H=174, S≥120 → red (HSV(174,122,172))
        elif h >= 174 and s >= 198:
            return "pink", 0.90  # H=174, S≥198 → 진한 pink (HSV(174,198,113))
        # 어두운 영역: 고채도 red 우선 보정 (red→purple 방지)
        elif h >= 176 and s >= 133 and v >= 110:
            return "red", 0.90
        elif h >= 173 and v < 140:
            return "purple", 0.90
        elif s >= 86 and v >= 190 and h < 177:
            return "pink", 0.90
        elif s >= 120 and v >= 185 and h < 177:
            return "pink", 0.90
        elif s >= 70 and v >= 160 and h < 177:
            return "pink", 0.85
        elif s >= 60 and v >= 140 and h < 177:
            return "pink", 0.80
        else:
            return "purple", 0.75
    else:
        # 갈색 판단 (낮은 채도 + 낮은 명도)
        if s < 60 and v < 120:
            return "brown", 0.80
        return "unknown", 0.50
