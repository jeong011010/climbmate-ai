#!/usr/bin/env python3
"""
테스트 케이스(JSON) → hold_color_feedback 테이블 삽입 스크립트

- 기본 입력: test_cases/color_classification_test_cases.json
- 필요 시 --json-path 옵션으로 커스텀 파일 지정
- --truncate 옵션을 주면 삽입 전에 기존 hold_color_feedback 데이터를 모두 삭제

사용 예시 (로컬):
    python3 import_test_cases_to_feedback.py --truncate

사용 예시 (Docker/EC2):
    docker compose exec backend python3 import_test_cases_to_feedback.py --truncate
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone

import cv2
import numpy as np

from holdcheck.color_classifier import (
    classify_color_by_hsv,
    hsv_to_rgb_fast,
    load_color_ranges,
)


def resolve_db_path() -> str:
    """Docker/로컬 환경에 따라 DB 경로 결정"""
    if os.path.exists('/app/backend/climbmate.db'):
        return '/app/backend/climbmate.db'
    return os.path.join(os.path.dirname(__file__), 'backend', 'climbmate.db')


def hsv_to_lab(rgb):
    """RGB 배열을 Lab 공간으로 변환"""
    arr = np.uint8([[rgb]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0]
    return [int(lab[0]), int(lab[1]), int(lab[2])]


def truncate_feedback_table(conn: sqlite3.Connection) -> int:
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM hold_color_feedback')
    count = cursor.fetchone()[0]
    if count > 0:
        cursor.execute('DELETE FROM hold_color_feedback')
        conn.commit()
    return count


def insert_test_case(cursor: sqlite3.Cursor, idx: int, case: dict, colors_config) -> None:
    h, s, v = case['hsv']
    expected = case.get('expected') or case.get('expected_color') or 'unknown'

    rgb = hsv_to_rgb_fast(h, s, v)
    lab = hsv_to_lab(rgb)

    predicted, confidence, reason = classify_color_by_hsv(h, s, v, rgb, colors_config)

    color_stats = {
        'source': 'test_case_import',
        'test_case_id': case.get('id'),
        'name': case.get('name'),
        'description': case.get('description'),
        'reason': reason,
        'confidence': confidence,
    }

    cursor.execute(
        """
        INSERT INTO hold_color_feedback (
            problem_id, hold_id,
            center_x, center_y,
            rgb_r, rgb_g, rgb_b,
            hsv_h, hsv_s, hsv_v,
            lab_l, lab_a, lab_b,
            color_stats,
            predicted_color,
            user_correct_color,
            confirmed,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            0,
            idx + 1,
            0.0,
            0.0,
            rgb[0],
            rgb[1],
            rgb[2],
            h,
            s,
            v,
            lab[0],
            lab[1],
            lab[2],
            json.dumps(color_stats, ensure_ascii=False),
            predicted,
            expected,
            0,
            datetime.now(timezone.utc).isoformat(timespec='seconds'),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description='테스트 케이스를 hold_color_feedback 테이블에 삽입합니다.')
    parser.add_argument('--json-path', default='test_cases/color_classification_test_cases.json', help='테스트 케이스 JSON 경로')
    parser.add_argument('--truncate', action='store_true', help='삽입 전 기존 hold_color_feedback 데이터를 모두 삭제')
    parser.add_argument('--confirm', action='store_true', help='삽입되는 데이터를 confirmed=1 상태로 저장')
    args = parser.parse_args()

    json_path = os.path.abspath(args.json_path)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'테스트 케이스 파일을 찾을 수 없습니다: {json_path}')

    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    test_cases = payload.get('test_cases')
    if not test_cases:
        raise ValueError('test_cases 배열이 비어있습니다.')

    db_path = resolve_db_path()
    conn = sqlite3.connect(db_path)

    try:
        if args.truncate:
            deleted = truncate_feedback_table(conn)
            print(f'🗑️ 기존 피드백 {deleted}개 삭제 완료')

        cursor = conn.cursor()
        colors_config = load_color_ranges()

        for idx, case in enumerate(test_cases):
            insert_test_case(cursor, idx, case, colors_config)

        if args.confirm:
            cursor.execute('UPDATE hold_color_feedback SET confirmed = 1 WHERE confirmed = 0')

        conn.commit()

        print('✅ 테스트 케이스 삽입 완료')
        print(f'   총 삽입 건수: {len(test_cases)}개')
        if args.confirm:
            print('   모든 데이터 confirmed=1 로 저장되었습니다.')
    finally:
        conn.close()


if __name__ == '__main__':
    main()

