#!/usr/bin/env python3
import sys
import os
import json
import cv2
import numpy as np

# holdcheck 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'holdcheck'))

from color_classifier import load_color_ranges, calculate_confidence_hsv, check_rgb_condition, find_nearest_color_hsv


def classify_color_by_hsv_ranges_only(h, s, v, rgb, colors_config):
	"""color_ranges.json 규칙만으로 색상 분류 (하드코딩 우회)"""
	# 우선순위대로 순회
	sorted_colors = sorted(colors_config.items(), key=lambda x: x[1].get("priority", 999))
	for color_name, config in sorted_colors:
		# HSV 범위 체크
		if "hsv_ranges" in config:
			for hsv_range in config["hsv_ranges"]:
				h_min, h_max = hsv_range["h"]
				s_min, s_max = hsv_range["s"]
				v_min, v_max = hsv_range["v"]
				# Hue 원형 처리
				h_match = (h_min <= h <= h_max) if h_min <= h_max else (h >= h_min or h <= h_max)
				if h_match and s_min <= s <= s_max and v_min <= v <= v_max:
					confidence = calculate_confidence_hsv(h, s, v, hsv_range)
					return color_name, confidence, f"color_ranges.json: H={h}, S={s}, V={v}"
		# RGB 조건 체크
		if "rgb_conditions" in config:
			for condition in config["rgb_conditions"]:
				if check_rgb_condition(rgb, condition):
					return color_name, 0.8, f"color_ranges.json RGB: {rgb}"
	# 폴백
	color_name, confidence, rule = find_nearest_color_hsv(h, s, v, colors_config)
	return color_name, confidence, rule


def run_all_test_cases_ranges_only():
	# 테스트 케이스 로드
	with open('test_cases/color_classification_test_cases.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
	cases = data['test_cases']
	print(f"🧪 색상 분류 테스트 (ranges-only) - {len(cases)}개 케이스\n")

	# color_ranges.json 로드
	ranges_data = load_color_ranges()
	colors_config = ranges_data["colors"]

	passed = 0
	failed = 0
	failed_cases = []
	color_stats = {}

	for i, case in enumerate(cases, 1):
		h, s, v = case['hsv']
		expected = case['expected']
		description = case.get('description', case.get('name', ''))
		# HSV -> RGB
		hsv_arr = np.uint8([[[h, s, v]]])
		rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
		rgb = rgb_arr.tolist()
		# 분류
		pred, conf, rule = classify_color_by_hsv_ranges_only(h, s, v, rgb, colors_config)
		is_correct = pred == expected
		if is_correct:
			passed += 1
		else:
			failed += 1
			failed_cases.append({
				'id': case.get('id', f'test_{i}'),
				'name': case.get('name', ''),
				'hsv': (h, s, v),
				'expected': expected,
				'actual': pred,
				'confidence': conf,
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

	# 결과 출력
	total = len(cases)
	accuracy = passed / total * 100
	print("=" * 80)
	print("📊 테스트 결과 요약 (ranges-only)")
	print(f"   전체: {total}개 | 성공: {passed}개 | 실패: {failed}개")
	print(f"   정확도: {accuracy:.1f}%")
	print("=" * 80)
	print("\n📈 색상별 정확도:")
	for color in sorted(color_stats.keys()):
		stats = color_stats[color]
		color_acc = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
		status = "✅" if color_acc >= 80 else "⚠️" if color_acc >= 60 else "❌"
		print(f"   {status} {color:8s}: {stats['passed']:2d}/{stats['total']:2d} ({color_acc:5.1f}%)")

	if failed > 0:
		print(f"\n❌ 실패한 케이스 ({failed}개):")
		mis = {}
		for fc in failed_cases:
			pattern = f"{fc['expected']}→{fc['actual']}"
			mis.setdefault(pattern, []).append(fc)
		print("\n🔍 오분류 패턴:")
		for pattern, cases in sorted(mis.items(), key=lambda x: -len(x[1])):
			print(f"   {pattern}: {len(cases)}건")
			for case in cases[:3]:
				h, s, v = case['hsv']
				print(f"      - HSV({h},{s},{v}): {case['name']}")
			if len(cases) > 3:
				print(f"      ... 외 {len(cases)-3}건")
	else:
		print("\n🎉 모든 테스트 통과!")

	return accuracy


if __name__ == "__main__":
	acc = run_all_test_cases_ranges_only()
	if acc < 80:
		print(f"\n⚠️ 경고: 정확도가 80% 미만입니다 ({acc:.1f}%)")
		print("   color_ranges.json 튜닝이 필요합니다.")
	elif acc >= 90:
		print(f"\n🎉 훌륭합니다! 정확도 {acc:.1f}%")
	else:
		print(f"\n✅ 양호합니다. 정확도 {acc:.1f}%")
