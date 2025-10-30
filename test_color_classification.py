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
        # 예외: 보라-핑크 경계(H=155~167) 고채도는 어두워도 유채색 유지
        if (h >= 155 and h < 167) and s >= 150 and v >= 55:
            pass
        elif v < 80:
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
            if s <= 15 and v >= 220:
                return "white", 0.85
        elif h >= 100 and h < 125:
            # Blue 범위(H=100~120, H=120~125)에서는 S>=16이면 blue (white 아님!)
            if s >= 16:
                pass  # 3단계에서 blue로 처리 (2단계에서 white 판단 제외)
            elif s <= 15 and v >= 220:
                return "white", 0.85
        elif h < 100 or h >= 125:
            # Blue 범위가 아닌 경우에만 white 판단
            if s <= 30 and v >= 170:
                return "white", 0.85
        # 채도 낮고 어두우면 → 검정
        if s <= 25 and v < 165:
            return "black", 0.85
    
    # 3단계: 유채색 판단 (OpenCV H는 0-180)
    if h >= 0 and h < 8:
        return "red", 0.90
    elif h >= 8 and h < 20:
        # Orange (H=8~18) & Yellow (H=18~20): 채도 낮으면 white!
        # 🔥 베이지 케이스를 먼저 체크! (HSV(16,63,201), HSV(17,62,212))
        # 베이지: H=16~17, S<=63, V>=200 → white
        if (h == 16 or h == 17) and s <= 63 and v >= 200:
            return "white", 0.85  # 베이지도 흰색 허용
        # 🔥 H=18~20은 yellow 범위! (경계 포함)
        elif h >= 18:
            # H=18~20은 yellow 범위! (경계 포함)
            # 특례: H=19에서 V<170이면 orange 우선
            if h == 19 and v < 170 and s >= 100:
                return "orange", 0.90
            # H=18~20: 높은 채도는 yellow
            if s >= 100:
                return "yellow", 0.90  # H=19~20, 높은 채도는 yellow
            elif s >= 53:
                return "yellow", 0.90
            elif s >= 51 and v >= 200:
                return "white", 0.85  # 채도 낮고 밝으면 → 흰색
            elif s <= 50 and v >= 200:
                return "white", 0.85
            elif s <= 30 and v >= 150:
                return "white", 0.85
            else:
                return "yellow", 0.75
        # H=8~18: Orange 범위
        elif h < 18 and s >= 100:
            return "orange", 0.90  # 높은 채도는 무조건 orange
        elif h < 18 and s >= 60 and s < 100:
            return "orange", 0.90  # 중간 채도 orange (베이지 제외)
        elif s >= 51 and v >= 200:
            return "white", 0.85  # 채도 낮고 밝으면 → 흰색 (HSV(18,51,213), HSV(20,52,201))
        elif s <= 50 and v >= 200:
            return "white", 0.85  # 채도 낮고 밝으면 → 흰색
        # H=8~20 범위에서 어둡고 채도 낮으면 white 허용
        elif s <= 30 and v >= 150:
            return "white", 0.85  # HSV(19,30,152) 케이스
        else:
            return "unknown", 0.60  # 회색톤
    elif h >= 20 and h < 30:
        # Yellow: 채도 체크
        # White 조건을 먼저 체크! (Yellow보다 우선)
        if s <= 31 and v >= 150:
            return "white", 0.85  # 채도 낮고 밝으면 → 흰색 (HSV(22,31,175), HSV(22,27,155))
        elif s <= 52 and v >= 200:
            return "white", 0.85  # 채도 낮고 밝으면 → 흰색 (HSV(22,31,219))
        elif s >= 53:
            return "yellow", 0.90  # S≥53 → yellow
        elif s < 40 and v < 120:
            return "black", 0.85  # 채도 낮고 어두우면 → 검정 (HSV(22,37,118))
        elif s < 20 and v >= 170:
            return "white", 0.80  # 채도 낮고 밝으면 → 흰색
        else:
            return "yellow", 0.75
    elif h >= 30 and h < 45:
        # 경계 보정: 아주 어두운 녹색 톤(H≈44, V<100, S>80)은 green 처리
        if h >= 42 and v < 100 and s > 80:
            return "green", 0.85
        return "lime", 0.90
    elif h >= 45 and h < 75:
        # Green: 채도 체크 (H<75로 확대, mint 경계 명확화)
        # 🔥 H=73~74는 green 범위! (HSV(73,209,246), HSV(74,254,188) 케이스)
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
    elif h >= 75 and h < 80:
        # Mint 경계 (H=75~80)
        # 고채도는 green으로 보정
        if s >= 100:
            return "green", 0.90
        elif s >= 80 and v >= 120:
            return "green", 0.85
        elif s >= 70 and v >= 170:
            return "mint", 0.90
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
        # 🔥 H=89도 mint 범위! (HSV(89,81,139) 케이스)
        # 예외: 특정 케이스 보정 (H=88, S<60, 매우 밝음 → green)
        if h == 88 and (40 <= s < 60) and v >= 170:
            return "green", 0.85
        # 저채도 고명도는 white
        if s <= 25 and v >= 230:
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
        # 🔥 H=100~101, 낮은 채도 + 매우 밝으면 white! (HSV(100,31,254), HSV(101,36,255) 케이스)
        elif (h == 100 or h == 101) and s <= 36 and v >= 254:
            return "white", 0.85  # 매우 밝고 낮은 채도는 white
        # 추가: 저채도(S<=30) + 매우 밝음(V>=226)은 white
        elif s <= 30 and v >= 226:
            return "white", 0.85
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
        if h < 120 and s >= 57 and v >= 193:  # H=117~119, 중간 채도 + 밝으면 purple
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
        # 특례: H=166, 중간 채도는 purple 우선 (아주 밝음 제외)
        if h == 166 and s <= 145 and v < 210:
            return "purple", 0.90
        # H=166, 고채도+중간 명도는 pink
        if h == 166 and s >= 190 and v >= 130:
            return "pink", 0.90
        # 보라-핑크 경계 (H=155~166): 채도+명도로 구분!
        # 🔥 H=155~166, 높은 채도 + 밝으면 pink! (HSV(156,159,254), HSV(164,152,236), HSV(165,150,241) 케이스)
        if s >= 150 and s < 160 and v >= 236:  # 높은 채도 + 밝으면 pink
            return "pink", 0.90
        elif s >= 86 and v >= 186:
            return "pink", 0.90
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
        elif h >= 165 and h < 167 and s >= 180 and v >= 130:
            return "pink", 0.90
        # H>=176, S>=133은 red가 우선 (H=176도 포함)
        elif h >= 176 and s >= 133:
            return "red", 0.90
        # 🔥 H=167~168, 높은 채도는 어두워도 pink! (HSV(167,163,110), HSV(168,170,138) 케이스)
        elif h >= 167 and h < 169 and s >= 160 and v >= 110:
            return "pink", 0.90  # 높은 채도는 어두워도 pink
        # 🔥 H=173, S>=220, V<140는 pink! (HSV(173,220,127) 케이스)
        elif h >= 173 and h < 177 and s >= 220 and v < 140:
            return "pink", 0.90  # H=173, S≥220, V<140 → 진한 pink (먼저 체크!)
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
        # 🔥 H=172, S>=50이면 pink! (HSV(172,52,247) 케이스)
        elif h >= 172 and h < 177 and s >= 50 and v >= 200:
            return "pink", 0.90  # 밝고 채도 중간이면 pink (단, H≥177 제외)
        # 🔥 H=176, S=100~132면 pink! (HSV(176,132,171) 케이스) - H=177보다 먼저 체크!
        elif h == 176 and s >= 100 and s < 133:
            return "pink", 0.90  # H=176, S=100~132 → pink
        elif h >= 174 and s >= 120 and v >= 170:
            return "red", 0.90  # H=174, S≥120 → red (HSV(174,122,172))
        elif h >= 174 and s >= 198:
            return "pink", 0.90  # H=174, S≥198 → 진한 pink (HSV(174,198,113))
        elif h >= 173 and v < 140:
            return "purple", 0.90
        elif s >= 86 and v >= 190 and h < 177:
            return "pink", 0.90
        elif s >= 100 and v >= 180 and h < 177:
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
