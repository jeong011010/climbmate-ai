#!/usr/bin/env python3
"""
모든 HSV 피드백 데이터를 테스트 케이스로 추가
"""

import json
from pathlib import Path
from datetime import datetime

# 기존 테스트 케이스 로드
test_file = Path('test_cases/color_classification_test_cases.json')
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 기존 HSV 값들
existing_hsv = {tuple(tc['hsv']) for tc in test_data.get('test_cases', [])}

# 새로 추가할 테스트 케이스들
new_cases = [
    {"hsv": [18, 51, 213], "expected": "white", "name": "베이지 3 - White", "desc": "채도 낮고 밝으면 흰색"},
    {"hsv": [20, 52, 201], "expected": "white", "name": "베이지 4 - White", "desc": "채도 낮고 밝으면 흰색"},
    {"hsv": [22, 31, 219], "expected": "white", "name": "Yellow 범위 White", "desc": "Yellow 범위에서 채도 낮고 밝으면 흰색"},
    {"hsv": [22, 37, 118], "expected": "black", "name": "Yellow 범위 Black", "desc": "Yellow 범위에서 채도 낮고 어두우면 검정"},
    {"hsv": [60, 10, 168], "expected": "white", "name": "Green 범위 White", "desc": "Green 범위에서 채도 극도로 낮고 밝으면 흰색"},
    {"hsv": [75, 36, 65], "expected": "black", "name": "Green 범위 Black", "desc": "Green 범위에서 채도 낮고 어두우면 검정"},
    {"hsv": [79, 39, 135], "expected": "black", "name": "Mint 경계 Black", "desc": "Mint 경계에서 채도 낮고 어두우면 검정"},
    {"hsv": [80, 27, 175], "expected": "mint", "name": "Mint 중간", "desc": "중간 민트"},
    {"hsv": [84, 48, 143], "expected": "mint", "name": "Mint 진한색", "desc": "진한 민트"},
    {"hsv": [87, 30, 173], "expected": "mint", "name": "Mint 중간 2", "desc": "중간 민트"},
    {"hsv": [88, 20, 202], "expected": "mint", "name": "Mint 연한색", "desc": "연한 민트"},
    {"hsv": [104, 27, 156], "expected": "black", "name": "Blue 범위 Black 1", "desc": "Blue 범위에서 채도 낮고 어두우면 검정"},
    {"hsv": [104, 48, 148], "expected": "black", "name": "Blue 범위 Black 2", "desc": "Blue 범위에서 채도 높지만 어두우면 검정"},
    {"hsv": [105, 147, 148], "expected": "blue", "name": "Blue 진한색", "desc": "채도 극도로 높음 → 파랑"},
    {"hsv": [106, 134, 160], "expected": "blue", "name": "Blue 밝은색", "desc": "채도 높고 밝음 → 파랑"},
    {"hsv": [106, 145, 158], "expected": "blue", "name": "Blue 중간 밝기", "desc": "채도 높고 중간 밝기 → 파랑"},
    {"hsv": [108, 52, 202], "expected": "black", "name": "Blue 범위 Black 3", "desc": "S=50~60, 밝지만 채도 애매 → 검정"},
    {"hsv": [108, 64, 163], "expected": "black", "name": "Blue 범위 Black 4", "desc": "채도 높지만 어두움 → 검정"},
    {"hsv": [115, 55, 217], "expected": "purple", "name": "Purple 경계", "desc": "H≥115, 밝으면 → purple"},
    {"hsv": [123, 98, 174], "expected": "purple", "name": "Purple 진한색", "desc": "채도 높으면 → 보라"},
    {"hsv": [157, 69, 216], "expected": "pink", "name": "Pink 밝은색 1", "desc": "매우 밝고 채도 중간 → pink"},
    {"hsv": [158, 77, 219], "expected": "pink", "name": "Pink 밝은색 2", "desc": "밝고 채도 중간 → pink"},
    {"hsv": [161, 63, 152], "expected": "purple", "name": "Purple 어두운색", "desc": "어둡거나 채도 낮음 → purple"},
    {"hsv": [173, 121, 137], "expected": "purple", "name": "Purple 어두운색 2", "desc": "어두움 → purple"},
    {"hsv": [173, 220, 127], "expected": "pink", "name": "Pink 진한색 1", "desc": "H=173, S≥220, V<140 → 진한 pink"},
    {"hsv": [174, 122, 172], "expected": "red", "name": "Red", "desc": "H=174, S≥120 → red"},
    {"hsv": [174, 173, 112], "expected": "purple", "name": "Pink-Purple 경계", "desc": "H=173~174, V<140 → purple"},
    {"hsv": [176, 132, 171], "expected": "pink", "name": "Pink 2", "desc": "H≥176, S=100~132 → pink"},
    {"hsv": [177, 107, 215], "expected": "red", "name": "Red 2", "desc": "H≥177, S≥107 → red"}
]

added_count = 0
for case in new_cases:
    hsv_tuple = tuple(case['hsv'])
    if hsv_tuple not in existing_hsv:
        test_id = f"code_comment_{len(test_data['test_cases']) + 1}"
        test_data['test_cases'].append({
            "id": test_id,
            "name": case['name'],
            "hsv": case['hsv'],
            "expected": case['expected'],
            "description": case['desc'],
            "date_added": datetime.now().strftime("%Y-%m-%d"),
            "fix_applied": "코드 주석에서 추출된 피드백 케이스"
        })
        existing_hsv.add(hsv_tuple)
        added_count += 1

# 저장
test_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(test_file, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"✅ {added_count}개의 테스트 케이스 추가 완료!")
print(f"📊 총 테스트 케이스: {len(test_data['test_cases'])}개")

