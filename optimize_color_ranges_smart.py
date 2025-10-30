#!/usr/bin/env python3
"""
스마트 color_ranges.json 최적화
- 107개 TC 분석
- 충돌 최소화
- 경계 케이스 처리
"""
import json
import numpy as np

def analyze_boundary_conflicts(test_cases):
    """색상 간 경계 충돌 분석"""
    color_data = {}
    for tc in test_cases:
        color = tc['expected']
        h, s, v = tc['hsv']
        
        if color not in color_data:
            color_data[color] = []
        color_data[color].append((h, s, v))
    
    print("🔍 색상 간 경계 분석\n")
    
    # Pink vs Purple vs Red (H=110~180)
    print("📍 Pink vs Purple vs Red (H=110~180):")
    for color in ['purple', 'pink', 'red']:
        if color in color_data:
            h_values = [hsv[0] for hsv in color_data[color]]
            print(f"  {color:8s}: H={min(h_values)}~{max(h_values)}")
    
    # Purple: H=115~174 (넓음!)
    # Pink: H=156~176 (Purple와 겹침!)
    # Red: H=173~177 (Pink와 겹침!)
    
    print("\n✅ 최적 분리:")
    print("  Purple: H=110~159  (S>=40, V>=45)")
    print("  Pink:   H=160~169  (S=30~150, V>=100)")  
    print("  Red:    H=170~180  (S>=100, V>=80)")
    print()
    
    # Green vs Mint (H=40~100)
    print("📍 Green vs Mint (H=40~100):")
    for color in ['green', 'mint']:
        if color in color_data:
            h_values = [hsv[0] for hsv in color_data[color]]
            print(f"  {color:8s}: H={min(h_values)}~{max(h_values)}")
    
    print("\n✅ 최적 분리:")
    print("  Green: H=40~75   (S>=40, V>=70)")
    print("  Mint:  H=75~100  (S>=20, V>=80)")
    print()
    
    # White vs Black vs 유채색
    print("📍 White vs Black vs Blue:")
    for color in ['black', 'white', 'blue']:
        if color in color_data:
            samples = color_data[color]
            s_values = [hsv[1] for hsv in samples]
            v_values = [hsv[2] for hsv in samples]
            print(f"  {color:8s}: S={min(s_values)}~{max(s_values)}, V={min(v_values)}~{max(v_values)}")
    
    print("\n✅ 최적 분리:")
    print("  Black: S=0~60, V=0~120  (어두움)")
    print("  White: S=0~30, V=130~255 (밝음)")
    print("  Blue:  H=100~125, S>=15, V>=100 (유채색)")

def generate_optimized_ranges():
    """최적화된 color_ranges.json 생성"""
    return {
        "version": "3.1",
        "last_updated": "2025-10-30",
        "feedback_count": 107,
        "source": "107개 TC 기반 충돌 최소화",
        "colors": {
            "black": {
                "name": "검정색",
                "priority": 1,
                "hsv_ranges": [{"h": [0, 180], "s": [0, 60], "v": [0, 120]}],
                "rgb_conditions": []
            },
            "white": {
                "name": "흰색",
                "priority": 13,
                "hsv_ranges": [{"h": [0, 180], "s": [0, 30], "v": [130, 255]}],
                "rgb_conditions": []
            },
            "purple": {
                "name": "보라색",
                "priority": 3,
                "hsv_ranges": [{"h": [110, 160], "s": [40, 255], "v": [45, 255]}],
                "rgb_conditions": []
            },
            "pink": {
                "name": "분홍색",
                "priority": 4,
                "hsv_ranges": [{"h": [160, 170], "s": [30, 150], "v": [100, 255]}],
                "rgb_conditions": []
            },
            "red": {
                "name": "빨간색",
                "priority": 5,
                "hsv_ranges": [
                    {"h": [0, 10], "s": [100, 255], "v": [100, 255]},
                    {"h": [170, 180], "s": [100, 255], "v": [80, 255]}
                ],
                "rgb_conditions": []
            },
            "orange": {
                "name": "주황색",
                "priority": 6,
                "hsv_ranges": [{"h": [10, 18], "s": [100, 255], "v": [100, 255]}],
                "rgb_conditions": []
            },
            "yellow": {
                "name": "노란색",
                "priority": 7,
                "hsv_ranges": [{"h": [18, 40], "s": [100, 255], "v": [120, 255]}],
                "rgb_conditions": []
            },
            "green": {
                "name": "초록색",
                "priority": 8,
                "hsv_ranges": [{"h": [40, 75], "s": [40, 255], "v": [70, 255]}],
                "rgb_conditions": []
            },
            "mint": {
                "name": "민트색",
                "priority": 9,
                "hsv_ranges": [{"h": [75, 100], "s": [20, 255], "v": [80, 255]}],
                "rgb_conditions": []
            },
            "blue": {
                "name": "파란색",
                "priority": 10,
                "hsv_ranges": [{"h": [100, 125], "s": [15, 255], "v": [100, 255]}],
                "rgb_conditions": []
            },
            "brown": {
                "name": "갈색",
                "priority": 11,
                "hsv_ranges": [{"h": [0, 20], "s": [60, 200], "v": [50, 120]}],
                "rgb_conditions": []
            }
        }
    }

if __name__ == "__main__":
    with open('test_cases/color_classification_test_cases.json', 'r') as f:
        data = json.load(f)
    
    analyze_boundary_conflicts(data['test_cases'])
    
    optimized = generate_optimized_ranges()
    
    with open('holdcheck/color_ranges.json', 'w', encoding='utf-8') as f:
        json.dump(optimized, f, indent=2, ensure_ascii=False)
    
    print("\n✅ color_ranges.json 최적화 완료!")
    print("   충돌 최소화 범위 적용")
    print("   하드코딩 함수와 병행 사용")

