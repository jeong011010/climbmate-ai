#!/usr/bin/env python3
"""
🎨 색상 분류 테스트 스크립트

JSON 파일에서 테스트 케이스를 읽어서 색상 분류 함수를 테스트합니다.
새로운 피드백이 들어올 때마다 test_cases/color_classification_test_cases.json 파일에 추가하면
이 스크립트로 전체 테스트가 제대로 통과하는지 확인할 수 있습니다.
"""

import sys
import os
import json
from pathlib import Path

# 함수 직접 import 시도
try:
    # 현재 디렉토리에서 직접 실행
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'holdcheck'))
    from clustering import classify_color_simple_hsv
except ImportError:
    # holdcheck 디렉토리에서 실행
    sys.path.insert(0, os.path.dirname(__file__))
    from holdcheck.clustering import classify_color_simple_hsv
except Exception as e:
    print(f"❌ 모듈 import 실패: {e}")
    print("💡 python3 -m pip install scikit-learn numpy opencv-python 실행 후 다시 시도하세요.")
    sys.exit(1)

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
