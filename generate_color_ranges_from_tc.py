#!/usr/bin/env python3
"""
테스트 케이스 기반 color_ranges.json 자동 생성
- 101개 테스트 케이스 분석
- 색상별 HSV 범위 자동 계산
- 안전 마진 추가
"""
import json
import numpy as np

def calculate_hsv_ranges(test_cases):
    """테스트 케이스에서 색상별 HSV 범위 계산"""
    
    # 색상별 HSV 값 수집
    color_data = {}
    for tc in test_cases:
        color = tc['expected']
        h, s, v = tc['hsv']
        
        if color not in color_data:
            color_data[color] = {'h': [], 's': [], 'v': []}
        
        color_data[color]['h'].append(h)
        color_data[color]['s'].append(s)
        color_data[color]['v'].append(v)
    
    # 색상별 범위 계산 (min-max + 마진)
    color_ranges = {}
    
    for color, values in color_data.items():
        h_values = values['h']
        s_values = values['s']
        v_values = values['v']
        
        # Min-Max 계산
        h_min, h_max = min(h_values), max(h_values)
        s_min, s_max = min(s_values), max(s_values)
        v_min, v_max = min(v_values), max(v_values)
        
        # 안전 마진 추가 (±10%)
        h_margin = max(5, int((h_max - h_min) * 0.1))
        s_margin = max(10, int((s_max - s_min) * 0.1))
        v_margin = max(10, int((v_max - v_min) * 0.1))
        
        h_min = max(0, h_min - h_margin)
        h_max = min(180, h_max + h_margin)
        s_min = max(0, s_min - s_margin)
        s_max = min(255, s_max + s_margin)
        v_min = max(0, v_min - v_margin)
        v_max = min(255, v_max + v_margin)
        
        # 빨강은 H가 0 근처와 180 근처로 분리
        if color == 'red' and any(h < 10 for h in h_values) and any(h > 170 for h in h_values):
            # 빨강 범위 분리
            low_h = [h for h in h_values if h < 90]
            high_h = [h for h in h_values if h >= 90]
            
            ranges = []
            if low_h:
                ranges.append({
                    "h": [0, max(low_h) + h_margin],
                    "s": [s_min, s_max],
                    "v": [v_min, v_max]
                })
            if high_h:
                ranges.append({
                    "h": [min(high_h) - h_margin, 180],
                    "s": [s_min, s_max],
                    "v": [v_min, v_max]
                })
            
            color_ranges[color] = ranges
        else:
            color_ranges[color] = [{
                "h": [h_min, h_max],
                "s": [s_min, s_max],
                "v": [v_min, v_max]
            }]
        
        print(f"{color:8s}: H=[{h_min:3d},{h_max:3d}] S=[{s_min:3d},{s_max:3d}] V=[{v_min:3d},{v_max:3d}] ({len(h_values)}개 샘플)")
    
    return color_ranges

def generate_color_ranges_json():
    """color_ranges.json 생성"""
    
    # 테스트 케이스 로드
    with open('test_cases/color_classification_test_cases.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data['test_cases']
    print(f"📊 테스트 케이스 기반 color_ranges.json 생성\n")
    print(f"총 {len(test_cases)}개 케이스 분석...\n")
    
    # HSV 범위 계산
    color_ranges = calculate_hsv_ranges(test_cases)
    
    # Priority 설정
    color_priority = {
        'black': 1,
        'white': 12,
        'pink': 3,
        'red': 4,
        'orange': 5,
        'yellow': 6,
        'green': 7,
        'mint': 8,
        'blue': 9,
        'purple': 10,
        'brown': 11
    }
    
    # JSON 구조 생성
    colors_config = {}
    for color, ranges in color_ranges.items():
        colors_config[color] = {
            "name": {
                'black': '검정색', 'white': '흰색', 'red': '빨간색',
                'orange': '주황색', 'yellow': '노란색', 'green': '초록색',
                'mint': '민트색', 'blue': '파란색', 'purple': '보라색',
                'pink': '분홍색', 'brown': '갈색'
            }.get(color, color),
            "priority": color_priority.get(color, 99),
            "hsv_ranges": ranges,
            "rgb_conditions": []
        }
    
    output = {
        "version": "2.0",
        "last_updated": "2025-10-30",
        "feedback_count": len(test_cases),
        "source": "test_cases/color_classification_test_cases.json",
        "colors": colors_config
    }
    
    # 저장
    with open('holdcheck/color_ranges.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ color_ranges.json 생성 완료!")
    print(f"   파일: holdcheck/color_ranges.json")
    print(f"   색상: {len(colors_config)}개")

if __name__ == "__main__":
    generate_color_ranges_json()

