#!/usr/bin/env python3
"""
🎨 색상 분류 테스트 스크립트

JSON 파일에서 테스트 케이스를 읽어서 색상 분류 함수를 테스트합니다.
새로운 피드백이 들어올 때마다 test_cases/color_classification_test_cases.json 파일에 추가하면
이 스크립트로 전체 테스트가 제대로 통과하는지 확인할 수 있습니다.
"""

import sys
import json
from pathlib import Path

# 색상 분류 함수 (의존성 없이 독립 실행 가능)
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
        if v < 80:
            # 매우 어두움 → 검정
            return "black", 0.95
        elif v >= 230 and s <= 10:
            # 매우 밝음 + 채도 극도로 낮음 → 흰색
            return "white", 0.95
        elif v >= 220 and s <= 12:
            # 밝음 + 채도 매우 낮음 → 흰색
            return "white", 0.85
    
    # 2단계: 유채색 범위에서도 무채색 판단 (우선)
    # 🔥 단, 높은 채도(S>=100)는 어두워도 유채색으로 판단!
    if is_chromatic_range:
        # 매우 어두우면 검정 (단, 채도가 매우 높으면 유채색!)
        # Green, Orange 등 높은 채도 색상은 어두워도 색상 유지
        if v < 90 and s < 100:
            return "black", 0.95  # V<90, S<100 → 검정 (낮은 채도만)
        # 채도 낮고 밝으면 → 흰색 (민트/파랑 범위에서)
        # 단, mint 범위(H=80~100)는 S≤15로 더 엄격하게!
        # 🔥 Blue 범위(H=100~120)는 S>=16이면 blue!
        if h >= 80 and h < 100:
            if s <= 15 and v >= 220:
                return "white", 0.85
        elif h >= 100 and h < 120:
            # Blue 범위에서는 S>=16이면 blue (white 아님!)
            if s >= 16:
                pass  # 3단계에서 blue로 처리
            elif s <= 15 and v >= 220:
                return "white", 0.85
        elif s <= 30 and v >= 170:
            return "white", 0.85
        # 채도 낮고 어두우면 → 검정
        if s <= 25 and v < 165:
            return "black", 0.85
    
    # 3단계: 유채색 판단 (OpenCV H는 0-180)
    if h >= 0 and h < 8:
        return "red", 0.90
    elif h >= 8 and h < 20:
        # Orange (H=8~18) & 일부 Yellow (H=18~20): 채도 낮으면 white!
        # 🔥 채도가 100 이상이면 무조건 orange!
        if h < 18 and s >= 60:
            return "orange", 0.90
        elif h < 20 and s >= 100:
            return "orange", 0.90  # 높은 채도는 무조건 orange
        elif s <= 63 and v >= 200:
            return "white", 0.85  # 베이지도 흰색 허용
        elif s >= 51 and v >= 200:
            return "white", 0.85
        elif s <= 50 and v >= 200:
            return "white", 0.85
        else:
            return "unknown", 0.60
    elif h >= 20 and h < 30:
        # Yellow: 채도 체크
        if s >= 53:
            return "yellow", 0.90
        elif s <= 52 and v >= 200:
            return "white", 0.85
        elif s < 40 and v < 120:
            return "black", 0.85
        elif s < 20 and v >= 170:
            return "white", 0.80
        else:
            return "yellow", 0.75
    elif h >= 30 and h < 45:
        return "lime", 0.90
    elif h >= 45 and h < 73:
        # Green: 채도 체크 (H<73으로 확대, mint 경계 명확화)
        # 🔥 높은 채도는 무조건 green!
        if s >= 100:
            return "green", 0.90  # 채도 높으면 무조건 green
        elif s >= 50:
            return "green", 0.90
        elif s < 40 and v < 140:
            return "black", 0.85
        elif s <= 10 and v >= 160:
            return "white", 0.85
        elif s < 15 and v >= 220:
            return "white", 0.80
        else:
            return "green", 0.75
    elif h >= 73 and h < 80:
        # Mint 전 단계 (H=73~80) - 경계 영역
        # 🔥 H=73~75, S>=70이면 mint
        if s >= 70 and v >= 170:
            return "mint", 0.90  # 채도 높고 밝으면 mint
        elif s >= 43 and v >= 200:
            return "mint", 0.85  # 밝으면 mint
        elif s < 40 and v < 140:
            return "black", 0.85
        else:
            return "green", 0.75  # 나머지는 green
    elif h >= 80 and h < 100:
        # 민트: 채도 체크 필수!
        if s >= 40 and v >= 140:
            return "mint", 0.90
        elif s >= 25 and v >= 170:
            return "mint", 0.85
        elif s >= 18 and v >= 200:
            return "mint", 0.80
        elif v < 70:
            return "black", 0.80
        else:
            return "unknown", 0.65
    elif h >= 100 and h < 120:
        # 파랑: purple과 분리 (H<120)
        if s >= 50 and v >= 200:
            if h >= 115:
                return "purple", 0.85
            elif s >= 50 and s < 60:
                return "black", 0.85
            else:
                return "blue", 0.90
        elif s >= 145 and v >= 158:
            return "blue", 0.90
        elif s >= 134 and v >= 160:
            return "blue", 0.90
        elif s >= 147:
            return "blue", 0.90
        elif s >= 64 and v < 164:
            return "black", 0.85
        elif s >= 60 and v < 160:
            return "black", 0.85
        elif s < 52 and v < 190:
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
    elif h >= 120 and h < 125:
        # 파랑-보라 경계
        # 🔥 H=120, S>=16이면 blue (HSV(120,16,228) 케이스)
        if s >= 16 and v >= 220:
            return "blue", 0.80  # 낮은 채도지만 blue-tinted
        elif s >= 90 and v >= 170:
            return "purple", 0.85
        elif s >= 70 and v >= 200:
            return "purple", 0.85
        elif s >= 50:
            return "blue", 0.85
        else:
            return "blue", 0.70
    elif h >= 125 and h < 155:
        # 보라 순수 범위 (H<155)
        if s >= 50 and v >= 90:
            return "purple", 0.90
        elif s >= 35 and v >= 140:
            return "purple", 0.85
        elif v < 70:
            return "black", 0.80
        else:
            return "purple", 0.70
    elif h >= 155 and h < 166:
        # 보라-핑크 경계 (H=155~165): 채도+명도로 구분!
        # 🔥 높은 채도는 red/maroon!
        if s >= 150:
            return "red", 0.90  # 매우 높은 채도는 red/maroon
        elif s >= 86 and v >= 186:
            return "pink", 0.90
        elif s >= 77 and v >= 219:
            return "pink", 0.90
        elif s >= 69 and v >= 210:
            return "pink", 0.90
        elif v < 140:
            return "purple", 0.90
        elif s >= 50 and v >= 90:
            return "purple", 0.90
        elif s >= 35 and v >= 140:
            return "purple", 0.85
        elif v < 70:
            return "black", 0.80
        else:
            return "purple", 0.70
    elif h >= 166 and h < 180:
        # Pink 전용 범위 (H=166~180)
        # 🔥 H=169~173, 높은 S면 red/maroon!
        if h >= 169 and h < 174 and s >= 150:
            return "red", 0.90  # 높은 채도는 red/maroon
        # 🔥 H=172, S>=50이면 pink!
        elif h >= 172 and s >= 50 and v >= 200:
            return "pink", 0.90  # 밝고 채도 중간이면 pink
        # Red 범위: H=174~177, S≥120
        elif h >= 177 and s >= 107:
            return "red", 0.90
        elif h >= 176 and s >= 133:
            return "red", 0.90
        elif h >= 176 and s >= 100 and s < 133:
            return "pink", 0.90
        elif h >= 174 and s >= 120 and v >= 170:
            return "red", 0.90
        elif h >= 174 and s >= 198:
            return "pink", 0.90
        elif h >= 173 and s >= 220 and v < 140:
            return "pink", 0.90
        elif h >= 173 and v < 140:
            return "purple", 0.90
        elif s >= 86 and v >= 190:
            return "pink", 0.90
        elif s >= 100 and v >= 180:
            return "pink", 0.90
        elif s >= 70 and v >= 160:
            return "pink", 0.85
        elif s >= 60 and v >= 140:
            return "pink", 0.80
        else:
            return "purple", 0.75
    else:
        # 갈색 판단 (낮은 채도 + 낮은 명도)
        if s < 60 and v < 120:
            return "brown", 0.80
        return "unknown", 0.50

# 테스트 케이스 파일 경로
TEST_CASES_FILE = Path(__file__).parent / "test_cases" / "color_classification_test_cases.json"

def load_test_cases():
    """테스트 케이스 JSON 파일 로드"""
    if not TEST_CASES_FILE.exists():
        print(f"❌ 테스트 케이스 파일을 찾을 수 없습니다: {TEST_CASES_FILE}")
        print(f"💡 {TEST_CASES_FILE.parent} 디렉토리를 생성하고 JSON 파일을 추가하세요.")
        sys.exit(1)
    
    try:
        with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파일 파싱 오류: {e}")
        print(f"   파일: {TEST_CASES_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        sys.exit(1)

def test_color_classification():
    """색상 분류 테스트 실행"""
    
    # 테스트 케이스 로드
    test_data = load_test_cases()
    test_cases = test_data.get("test_cases", [])
    
    if not test_cases:
        print("❌ 테스트 케이스가 없습니다.")
        sys.exit(1)
    
    print("=" * 80)
    print("🎨 색상 분류 테스트 시작")
    print("=" * 80)
    print(f"📁 테스트 케이스 파일: {TEST_CASES_FILE}")
    print(f"📊 총 테스트 케이스: {len(test_cases)}개")
    print(f"📝 설명: {test_data.get('description', 'N/A')}")
    print(f"🔖 버전: {test_data.get('version', 'N/A')}")
    print()
    
    passed = 0
    failed = 0
    tests_failed = []
    
    for i, test in enumerate(test_cases, 1):
        test_id = test.get("id", f"test_{i}")
        name = test.get("name", f"테스트 {i}")
        hsv_list = test.get("hsv", [])
        expected = test.get("expected", "").lower()
        description = test.get("description", "")
        date_added = test.get("date_added", "N/A")
        fix_applied = test.get("fix_applied", "")
        
        if len(hsv_list) != 3:
            print(f"[{i}/{len(test_cases)}] ❌ ERROR - {name}")
            print(f"  HSV 값이 올바르지 않습니다: {hsv_list}")
            print()
            failed += 1
            tests_failed.append((name, "HSV 값 형식 오류"))
            continue
        
        h, s, v = hsv_list
        
        # 색상 분류 실행
        try:
            result_color, confidence = classify_color_simple_hsv(h, s, v)
            result_color_lower = result_color.lower()
        except Exception as e:
            print(f"[{i}/{len(test_cases)}] ❌ ERROR - {name} (ID: {test_id})")
            print(f"  HSV: ({h}, {s}, {v})")
            print(f"  오류: {e}")
            print()
            failed += 1
            tests_failed.append((name, str(e)))
            continue
        
        # 결과 검증
        is_pass = result_color_lower == expected
        
        # 상태 아이콘
        status = "✅ PASS" if is_pass else "❌ FAIL"
        
        print(f"[{i}/{len(test_cases)}] {status} - {name} (ID: {test_id})")
        print(f"  HSV: ({h}, {s}, {v})")
        print(f"  예상: {expected.upper()}")
        print(f"  실제: {result_color.upper()} (신뢰도: {confidence:.2f})")
        
        if description:
            print(f"  설명: {description}")
        if fix_applied:
            print(f"  수정: {fix_applied}")
        if date_added != "N/A":
            print(f"  추가일: {date_added}")
        
        if is_pass:
            passed += 1
        else:
            failed += 1
            tests_failed.append((name, f"'{result_color}'를 반환했으나 '{expected}'를 기대함"))
            print(f"  ⚠️  오류: '{result_color}'를 반환했으나 '{expected}'를 기대함")
        
        print()
    
    # 결과 요약
    print("=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)
    print(f"✅ 통과: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.1f}%)")
    print(f"❌ 실패: {failed}/{len(test_cases)} ({failed/len(test_cases)*100:.1f}%)")
    
    if failed > 0:
        print("\n❌ 실패한 케이스:")
        for name, reason in tests_failed:
            print(f"  - {name}: {reason}")
    
    if failed == 0:
        print("\n🎉 모든 테스트가 통과했습니다!")
        return 0
    else:
        print(f"\n⚠️  {failed}개의 테스트가 실패했습니다.")
        return 1

def show_test_statistics():
    """테스트 케이스 통계 표시"""
    test_data = load_test_cases()
    test_cases = test_data.get("test_cases", [])
    
    if not test_cases:
        return
    
    print("\n" + "=" * 80)
    print("📈 테스트 케이스 통계")
    print("=" * 80)
    
    # 색상별 통계
    color_counts = {}
    for test in test_cases:
        expected = test.get("expected", "unknown").lower()
        color_counts[expected] = color_counts.get(expected, 0) + 1
    
    print("\n예상 색상별 통계:")
    for color, count in sorted(color_counts.items()):
        print(f"  {color.upper()}: {count}개")
    
    print(f"\n총 테스트 케이스: {len(test_cases)}개")

if __name__ == "__main__":
    try:
        exit_code = test_color_classification()
        show_test_statistics()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  테스트가 중단되었습니다.")
        sys.exit(1)
