#!/usr/bin/env python3
"""
color-feedback-*.json 파일을 테스트 케이스 JSON으로 병합
- 입력: color-feedback-YYYY-..json (배열)
- 출력: test_cases/color_classification_test_cases.json 업데이트
- 규칙:
  - confirmed=True인 항목 우선 사용 (없어도 user_correct_color가 있으면 사용)
  - expected = user_correct_color (gray는 white로 정규화)
  - 중복 제거: 동일 ID(fb_{id}) 또는 동일 HSV 값 존재 시 스킵
"""

import json
import sys
from pathlib import Path
from datetime import datetime

TEST_CASES_FILE = Path(__file__).parent / "test_cases" / "color_classification_test_cases.json"


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    if len(sys.argv) < 2:
        print("사용법: python3 merge_feedback_to_test_cases.py <color-feedback.json>")
        sys.exit(1)

    fb_path = Path(sys.argv[1]).resolve()
    if not fb_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {fb_path}")
        sys.exit(1)

    feedbacks = load_json(fb_path)
    if not isinstance(feedbacks, list):
        print("❌ 입력 JSON 형식이 배열이 아닙니다.")
        sys.exit(1)

    # 테스트 케이스 로드(없으면 생성)
    if TEST_CASES_FILE.exists():
        test_data = load_json(TEST_CASES_FILE)
    else:
        test_data = {
            "description": "색상 분류 테스트 케이스 - 피드백 누적용",
            "version": "1.0.0",
            "test_cases": []
        }

    existing_cases = test_data.get('test_cases', [])
    existing_ids = {tc.get('id') for tc in existing_cases}
    existing_hsv = {tuple(tc.get('hsv', [])) for tc in existing_cases if 'hsv' in tc}

    added = 0
    candidates = 0

    for fb in feedbacks:
        # confirmed 우선, 단 user_correct_color가 명시된 경우만
        confirmed = bool(fb.get('confirmed'))
        correct = (fb.get('user_correct_color') or '').strip().lower()
        pred = (fb.get('predicted_color') or '').strip().lower()
        hsv = fb.get('hsv') or fb.get('color_stats', {}).get('dominant_hsv')
        # HSV가 dict 형태({"h":..,"s":..,"v":..})인 경우 처리
        if isinstance(hsv, dict):
            if all(k in hsv for k in ('h','s','v')):
                h, s, v = int(hsv['h']), int(hsv['s']), int(hsv['v'])
            else:
                continue
        elif isinstance(hsv, (list, tuple)) and len(hsv) == 3:
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        else:
            continue

        # 후보: confirmed이거나 user_correct_color가 있고 pred와 다를 때
        if not correct:
            continue
        if not confirmed and pred and pred.lower() == correct.lower():
            # 불일치가 아니고 미확정이면 스킵
            continue

        candidates += 1

        if correct == 'gray':
            correct = 'white'

        tc_id = f"fb_{fb.get('id', f'{h}_{s}_{v}') }"
        hsv_key = (h, s, v)

        if tc_id in existing_ids or hsv_key in existing_hsv:
            continue

        name = fb.get('name') or fb.get('description') or f"FB {fb.get('id', '?')}"
        desc = f"피드백 병합: {pred.upper() if pred else 'N/A'} → {correct.upper()}"

        test_case = {
            "id": tc_id,
            "name": name,
            "hsv": [h, s, v],
            "expected": correct,
            "description": desc,
            "date_added": fb.get('created_at') or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "fix_applied": "feedback_merge"
        }

        existing_cases.append(test_case)
        existing_ids.add(tc_id)
        existing_hsv.add(hsv_key)
        added += 1

    test_data['test_cases'] = existing_cases
    test_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    save_json(TEST_CASES_FILE, test_data)

    print(f"✅ 병합 완료: 추가 {added}건 / 후보 {candidates}건")
    print(f"📄 파일: {TEST_CASES_FILE}")


if __name__ == '__main__':
    main()
