#!/usr/bin/env python3
"""
HSV 색상 분류 테스트 스크립트
"""
import sys
import os
import json

# holdcheck 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'holdcheck'))

from color_classifier import classify_color_by_hsv, load_color_ranges

def load_color_ranges_json(config_path="holdcheck/color_ranges.json"):
    """color_ranges.json 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["colors"]
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return {}

def check_hsv_in_range(h, s, v, color_ranges):
    """HSV 값이 color_ranges.json의 어떤 범위에 속하는지 확인"""
    matches = []
    for color_name, config in color_ranges.items():
        for hsv_range in config.get("hsv_ranges", []):
            h_min, h_max = hsv_range["h"]
            s_min, s_max = hsv_range["s"]
            v_min, v_max = hsv_range["v"]
            
            if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
                matches.append(color_name)
    return matches

def test_hsv_cases():
    """테스트 케이스 실행"""
    print("🧪 HSV 색상 분류 테스트\n")
    
    # color_ranges.json 로드 (두 가지 함수 모두 사용)
    ranges_data = load_color_ranges()  # clustering.py의 함수
    color_ranges = ranges_data["colors"]
    
    # 테스트 케이스
    test_cases = [
        # 문제 1: 진한 빨강 → pink 오분류
        {"hsv": (173, 166, 184), "expected": "red", "description": "진한 빨강 1"},
        {"hsv": (174, 215, 139), "expected": "red", "description": "진한 빨강 2 (어두움)"},
        
        # 문제 2: 밝은 회색 → black 오분류
        {"hsv": (13, 12, 153), "expected": "white", "description": "밝은 회색"},
        
        # 추가 테스트 (경계 케이스)
        {"hsv": (0, 255, 255), "expected": "red", "description": "순수 빨강"},
        {"hsv": (165, 120, 80), "expected": "red", "description": "진한 빨강 (경계)"},
        {"hsv": (160, 50, 180), "expected": "pink", "description": "연한 분홍"},
        {"hsv": (0, 0, 255), "expected": "white", "description": "순수 흰색"},
        {"hsv": (0, 0, 0), "expected": "black", "description": "순수 검정"},
    ]
    
    results = []
    for i, case in enumerate(test_cases, 1):
        h, s, v = case["hsv"]
        expected = case["expected"]
        description = case["description"]
        
        # RGB 값 생성 (필요 시)
        import cv2
        import numpy as np
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        rgb = rgb_arr.tolist()
        
        # classify_color_by_hsv로 분류 (color_ranges.json 사용)
        color_name, confidence, matched_rule = classify_color_by_hsv(h, s, v, rgb, color_ranges)
        
        # color_ranges.json 범위 확인
        range_matches = check_hsv_in_range(h, s, v, color_ranges)
        
        # 결과 판정
        is_correct = color_name == expected
        status = "✅" if is_correct else "❌"
        
        print(f"{status} 테스트 {i}: {description}")
        print(f"   HSV({h}, {s}, {v}) → RGB{rgb}")
        print(f"   예상: {expected} | 결과: {color_name} (신뢰도: {confidence:.2f})")
        print(f"   매칭 규칙: {matched_rule}")
        print(f"   color_ranges.json 범위 매칭: {range_matches if range_matches else '없음'}")
        print()
        
        results.append({
            "test_id": i,
            "description": description,
            "hsv": (h, s, v),
            "expected": expected,
            "actual": color_name,
            "confidence": confidence,
            "correct": is_correct,
            "range_matches": range_matches
        })
    
    # 결과 요약
    total = len(results)
    passed = sum(1 for r in results if r["correct"])
    failed = total - passed
    accuracy = passed / total * 100
    
    print("=" * 60)
    print(f"📊 테스트 결과 요약")
    print(f"   전체: {total}개 | 성공: {passed}개 | 실패: {failed}개")
    print(f"   정확도: {accuracy:.1f}%")
    print("=" * 60)
    
    if failed > 0:
        print("\n❌ 실패한 테스트:")
        for r in results:
            if not r["correct"]:
                print(f"   - {r['description']}: HSV{r['hsv']} → {r['actual']} (예상: {r['expected']})")
    else:
        print("\n🎉 모든 테스트 통과!")
    
    return results

if __name__ == "__main__":
    test_hsv_cases()

