#!/usr/bin/env python3
"""
🎨 모든 색상 분류 테스트 케이스 종합 테스트 스크립트

이전에 제공받았던 모든 잘못된 케이스들을 모아 한 번에 테스트합니다.
"""

import sys
sys.path.insert(0, 'holdcheck')
from clustering import classify_color_simple_hsv

# 최근 11개 케이스 (최종 테스트)
recent_11 = [
    (159, 94, 213, 'PINK'),
    (157, 69, 216, 'PINK'),
    (115, 55, 217, 'PURPLE'),
    (175, 99, 189, 'PINK'),
    (176, 132, 171, 'PINK'),
    (175, 82, 173, 'PINK'),
    (175, 106, 156, 'PINK'),
    (108, 64, 163, 'BLACK'),
    (108, 52, 202, 'BLACK'),
    (123, 98, 174, 'PURPLE'),
    (174, 122, 172, 'RED'),
]

# 코드 주석에 있는 특정 케이스들
code_examples = [
    (115, 55, 217, 'PURPLE'),  # H≥115, 밝으면 → purple
    (108, 64, 163, 'BLACK'),   # 채도 높지만 어두움 → 검정
    (108, 52, 202, 'BLACK'),   # S=50~60, 밝지만 채도 애매 → 검정
    (84, 48, 143, 'MINT'),     # 진한 민트
    (87, 30, 173, 'MINT'),     # 중간 민트
    (80, 27, 175, 'MINT'),     # 중간 민트
    (88, 20, 202, 'MINT'),     # 연한 민트
]

# 추가로 이전에 제공받았던 케이스들 (필요시 여기에 추가)
additional_cases = [
    # 예시: (h, s, v, 'EXPECTED_COLOR'),
    # 여기에 추가 케이스를 넣어주세요
]

# 중복 제거 및 모든 케이스 통합
all_test_cases = []
seen = set()

for h, s, v, expected in recent_11 + code_examples + additional_cases:
    key = (h, s, v)
    if key not in seen:
        seen.add(key)
        all_test_cases.append((h, s, v, expected))

# 색상별 통계
color_stats = {}
for h, s, v, expected in all_test_cases:
    if expected not in color_stats:
        color_stats[expected] = {'total': 0, 'correct': 0, 'incorrect': []}
    color_stats[expected]['total'] += 1

# 테스트 실행
print("=" * 70)
print("🎨 모든 색상 분류 테스트 케이스 종합 테스트")
print("=" * 70)
print(f"총 테스트 케이스: {len(all_test_cases)}개\n")

correct = 0
incorrect = []
results_by_color = {}

for h, s, v, expected in all_test_cases:
    result, conf = classify_color_simple_hsv(h, s, v)
    result_upper = result.upper()
    expected_upper = expected.upper()
    
    if result_upper == expected_upper:
        status = '✅'
        correct += 1
        color_stats[expected]['correct'] += 1
    else:
        status = '❌'
        incorrect.append((h, s, v, expected, result, conf))
        color_stats[expected]['incorrect'].append((h, s, v, result, conf))
    
    # 색상별 결과 그룹화
    if expected not in results_by_color:
        results_by_color[expected] = []
    results_by_color[expected].append({
        'hsv': (h, s, v),
        'expected': expected,
        'result': result,
        'conf': conf,
        'correct': result_upper == expected_upper
    })
    
    print(f"{status} HSV({h:3d},{s:3d},{v:3d}) → {result:8s} (예상: {expected:8s}) [신뢰도: {conf:.2f}]")

print("\n" + "=" * 70)
print("📊 종합 결과")
print("=" * 70)
print(f"전체 정확도: {correct}/{len(all_test_cases)} ({correct/len(all_test_cases)*100:.1f}%)")

if incorrect:
    print(f"\n❌ 오분류 케이스: {len(incorrect)}개")
    print("-" * 70)
    for h, s, v, expected, result, conf in incorrect:
        print(f"  HSV({h:3d},{s:3d},{v:3d}) → {result:8s} (예상: {expected:8s}) [신뢰도: {conf:.2f}]")

print("\n" + "=" * 70)
print("📈 색상별 상세 통계")
print("=" * 70)

# 색상별 정확도 출력
for color in sorted(color_stats.keys()):
    stats = color_stats[color]
    total = stats['total']
    correct_count = stats['correct']
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    print(f"\n{color:8s}: {correct_count}/{total} ({accuracy:.1f}%)")
    
    if stats['incorrect']:
        print("  오분류:")
        for h, s, v, result, conf in stats['incorrect']:
            print(f"    HSV({h:3d},{s:3d},{v:3d}) → {result:8s} [신뢰도: {conf:.2f}]")

print("\n" + "=" * 70)

if correct == len(all_test_cases):
    print("🎉 완벽합니다! 모든 케이스가 올바르게 분류되었습니다! ✅")
else:
    print(f"⚠️  {len(incorrect)}개의 케이스가 아직 수정이 필요합니다.")
    print("=" * 70)

