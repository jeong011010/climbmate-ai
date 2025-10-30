#!/usr/bin/env python3
"""
전체 색상 분류 테스트 케이스 실행
- test_cases/color_classification_test_cases.json 사용
- 101개의 누적된 피드백 케이스 검증
"""
import sys
import os
import json
import cv2
import numpy as np

# holdcheck 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'holdcheck'))

from color_classifier import classify_color_by_hsv, load_color_ranges

def run_all_test_cases():
    """전체 테스트 케이스 실행"""
    
    # 테스트 케이스 로드
    with open('test_cases/color_classification_test_cases.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data['test_cases']
    print(f"🧪 색상 분류 테스트 - {len(test_cases)}개 케이스\n")
    
    # color_ranges.json 로드
    ranges_data = load_color_ranges()
    color_ranges = ranges_data["colors"]
    
    # 결과 저장
    results = []
    passed = 0
    failed = 0
    failed_cases = []
    
    # 색상별 통계
    color_stats = {}
    
    for i, case in enumerate(test_cases, 1):
        h, s, v = case['hsv']
        expected = case['expected']
        description = case.get('description', case.get('name', ''))
        
        # RGB 값 생성
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        rgb = rgb_arr.tolist()
        
        # 분류
        color_name, confidence, matched_rule = classify_color_by_hsv(h, s, v, rgb, color_ranges)
        
        # 결과 판정
        is_correct = color_name == expected
        
        if is_correct:
            passed += 1
        else:
            failed += 1
            failed_cases.append({
                'id': case.get('id', f'test_{i}'),
                'name': case.get('name', ''),
                'hsv': (h, s, v),
                'expected': expected,
                'actual': color_name,
                'confidence': confidence,
                'description': description
            })
        
        # 색상별 통계
        if expected not in color_stats:
            color_stats[expected] = {'total': 0, 'passed': 0, 'failed': 0}
        color_stats[expected]['total'] += 1
        if is_correct:
            color_stats[expected]['passed'] += 1
        else:
            color_stats[expected]['failed'] += 1
        
        results.append({
            'test_id': i,
            'hsv': (h, s, v),
            'expected': expected,
            'actual': color_name,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # 결과 출력
    total = len(results)
    accuracy = passed / total * 100
    
    print("=" * 80)
    print(f"📊 테스트 결과 요약")
    print(f"   전체: {total}개 | 성공: {passed}개 | 실패: {failed}개")
    print(f"   정확도: {accuracy:.1f}%")
    print("=" * 80)
    
    # 색상별 통계
    print("\n📈 색상별 정확도:")
    for color in sorted(color_stats.keys()):
        stats = color_stats[color]
        color_acc = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        status = "✅" if color_acc >= 80 else "⚠️" if color_acc >= 60 else "❌"
        print(f"   {status} {color:8s}: {stats['passed']:2d}/{stats['total']:2d} ({color_acc:5.1f}%)")
    
    # 실패 케이스 분석
    if failed > 0:
        print(f"\n❌ 실패한 케이스 ({failed}개):")
        
        # 오분류 패턴 분석
        misclassification_patterns = {}
        for fc in failed_cases:
            pattern = f"{fc['expected']}→{fc['actual']}"
            if pattern not in misclassification_patterns:
                misclassification_patterns[pattern] = []
            misclassification_patterns[pattern].append(fc)
        
        print("\n🔍 오분류 패턴:")
        for pattern, cases in sorted(misclassification_patterns.items(), key=lambda x: -len(x[1])):
            print(f"   {pattern}: {len(cases)}건")
            for case in cases[:3]:  # 각 패턴당 최대 3개만
                h, s, v = case['hsv']
                print(f"      - HSV({h},{s},{v}): {case['name']}")
            if len(cases) > 3:
                print(f"      ... 외 {len(cases)-3}건")
    else:
        print("\n🎉 모든 테스트 통과!")
    
    return accuracy, results, failed_cases

if __name__ == "__main__":
    accuracy, results, failed_cases = run_all_test_cases()
    
    # 80% 미만이면 경고
    if accuracy < 80:
        print(f"\n⚠️ 경고: 정확도가 80% 미만입니다 ({accuracy:.1f}%)")
        print("   color_ranges.json 튜닝이 필요합니다.")
    elif accuracy >= 90:
        print(f"\n🎉 훌륭합니다! 정확도 {accuracy:.1f}%")
    else:
        print(f"\n✅ 양호합니다. 정확도 {accuracy:.1f}%")

