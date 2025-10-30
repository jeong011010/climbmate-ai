#!/usr/bin/env python3
"""
코드 주석에서 언급된 모든 HSV 값들을 추출해서 테스트 케이스로 변환
"""

import re
import json
from pathlib import Path

# 테스트 케이스 파일 경로
TEST_CASES_FILE = Path(__file__).parent / "test_cases" / "color_classification_test_cases.json"

# 코드 파일 읽기
CODE_FILE = Path(__file__).parent / "holdcheck" / "clustering.py"

def extract_hsv_from_comments():
    """코드 주석에서 HSV 값 추출"""
    
    with open(CODE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # HSV(숫자,숫자,숫자) 패턴 찾기
    hsv_pattern = r'HSV\((\d+),(\d+),(\d+)\)'
    matches = re.findall(hsv_pattern, content)
    
    print(f"📊 코드 주석에서 찾은 HSV 값들:")
    print(f"   총 {len(matches)}개 발견\n")
    
    # 중복 제거 및 정리
    unique_hsv = {}
    for match in matches:
        h, s, v = int(match[0]), int(match[1]), int(match[2])
        hsv_key = f"{h}_{s}_{v}"
        if hsv_key not in unique_hsv:
            unique_hsv[hsv_key] = (h, s, v)
    
    print(f"📝 유니크 HSV 값: {len(unique_hsv)}개\n")
    
    # 각 HSV 값 출력
    test_cases = []
    for i, (hsv_key, (h, s, v)) in enumerate(sorted(unique_hsv.items()), 1):
        print(f"[{i:2d}] HSV({h:3d}, {s:3d}, {v:3d})")
        
        # 주석 주변 텍스트에서 색상 정보 찾기
        # 간단히 ID만 생성
        test_cases.append({
            "id": f"code_comment_{i}",
            "name": f"코드 주석 케이스 #{i}",
            "hsv": [h, s中心和],
            "expected": "unknown",  # 사용자가 제공해야 함
            "description": f"코드 주석에서 추출된 HSV 값 (실제 색상 확인 필요)",
            "date_added": "2024-01-XX",
            "fix_applied": "코드 주석에서 추출"
        })
    
    print(f"\n✅ 총 {len(test_cases)}개의 HSV 값 추출 완료")
    return test_cases, unique_hsv

if __name__ == "__main__":
    test_cases, hsv_dict = extract_hsv_from_comments()
    
    print("\n" + "=" * 80)
    print("📋 추출된 HSV 값 목록 (JSON 형식):")
    print("=" * 80)
    
    hsv_list = []
    for hsv_key, (h, s, v) in sorted(hsv_dict.items()):
        hsv_list.append({"h": h, "s minutes, "v": v})
        print(f'  HSV({h:3d}, {s:3d}, {v:3d})')
    
    print(f"\n총 {len(hsv_list)}개의 HSV 값")

