#!/usr/bin/env python3
"""
피드백 JSON 파일을 DB(backend/climbmate.db)의 hold_color_feedback 테이블로 병합합니다.

사용 예:
  python3 scripts/import_feedback_json.py --path /absolute/path/to/color-feedback.json
"""
import argparse
import json
import os
import sqlite3
import time


def normalize_hsv(value):
    if isinstance(value, dict):
        return [int(value.get('h', 0)), int(value.get('s', 0)), int(value.get('v', 128))]
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [int(value[0]), int(value[1]), int(value[2])]
    return [0, 0, 128]


def normalize_rgb(value):
    if isinstance(value, dict):
        return [int(value.get('r', 128)), int(value.get('g', 128)), int(value.get('b', 128))]
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [int(value[0]), int(value[1]), int(value[2])]
    return [128, 128, 128]


def normalize_lab(value):
    if isinstance(value, dict):
        return [int(value.get('l', 0)), int(value.get('a', 0)), int(value.get('b', 0))]
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [int(value[0]), int(value[1]), int(value[2])]
    return [0, 0, 0]


def main():
    parser = argparse.ArgumentParser(description='피드백 JSON → DB 병합')
    parser.add_argument('--path', required=True, help='피드백 JSON 절대 경로')
    parser.add_argument('--db', default=os.path.join(os.path.dirname(__file__), '..', 'backend', 'climbmate.db'))
    args = parser.parse_args()

    json_path = os.path.abspath(args.path)
    db_path = os.path.abspath(args.db)

    if not os.path.exists(json_path):
        raise SystemExit(f'JSON이 존재하지 않습니다: {json_path}')
    if not os.path.exists(db_path):
        raise SystemExit(f'DB가 존재하지 않습니다: {db_path}')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get('feedbacks', [])
    else:
        items = []
    if not isinstance(items, list):
        raise SystemExit('JSON 형식 인식 실패 (feedbacks 배열 또는 루트 리스트 필요)')

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    ins = (
        """
        INSERT INTO hold_color_feedback (
         problem_id, hold_id, center_x, center_y,
         rgb_r, rgb_g, rgb_b,
         hsv_h, hsv_s, hsv_v,
         lab_l, lab_a, lab_b,
         color_stats,
         predicted_color, user_correct_color,
         confirmed, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
        """
    )

    count = 0
    ts = int(time.time())
    for it in items:
        hsv_source = it.get('hsv') or (it.get('color_stats') or {}).get('hsv') or (it.get('color_stats') or {}).get('dominant_hsv')
        rgb_source = it.get('rgb') or (it.get('color_stats') or {}).get('rgb')
        lab_source = it.get('lab') or (it.get('color_stats') or {}).get('lab')

        h, s, v = normalize_hsv(hsv_source)
        r, g, b = normalize_rgb(rgb_source)
        L, A, B = normalize_lab(lab_source)

        pred = (it.get('predicted_color') or it.get('pred') or 'unknown').lower()
        corr = (it.get('user_correct_color') or it.get('correct_color') or it.get('expected') or 'unknown').lower()

        center = it.get('center') or {}
        if isinstance(center, dict):
            cx = float(center.get('x', 0.0))
            cy = float(center.get('y', 0.0))
        elif isinstance(center, (list, tuple)) and len(center) >= 2:
            cx = float(center[0])
            cy = float(center[1])
        else:
            cx, cy = 0.0, 0.0

        problem_id = int(it.get('problem_id') or 0)
        hold_id = int(it.get('hold_id') or 0)

        color_stats = json.dumps({'source': 'json_import', 'ts': ts}, ensure_ascii=False)

        cur.execute(
            ins,
            (problem_id, hold_id, cx, cy, r, g, b, h, s, v, L, A, B, color_stats, pred, corr),
        )
        count += 1

    conn.commit()
    conn.close()
    print(f'✅ 병합 완료: {count}건')


if __name__ == '__main__':
    main()


