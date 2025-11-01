#!/usr/bin/env python3
import os
import sys
import csv
import sqlite3

# project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from holdcheck.color_classifier import (
    classify_color_by_hsv,
    classify_color_simple_hsv,
    load_color_ranges,
)


def main():
    db_path = os.path.join(ROOT, 'backend', 'climbmate.db')
    out_dir = os.path.join(ROOT, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, 'all_feedback_rule_decisions.csv')

    if not os.path.exists(db_path):
        print(f"❌ DB 없음: {db_path}")
        sys.exit(1)

    colors_config = load_color_ranges()["colors"]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, problem_id, hold_id,
               hsv_h, hsv_s, hsv_v,
               rgb_r, rgb_g, rgb_b,
               user_correct_color
        FROM hold_color_feedback
        ORDER BY created_at DESC
        """
    )
    rows = cur.fetchall()
    conn.close()

    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id','problem_id','hold_id','H','S','V','R','G','B',
            'expected','predicted_now','confidence','rule_label',
            'expected_is_rule','predicted_is_rule','reason'
        ])
        for (sid, problem_id, hold_id, h, s, v, r, g, b, expected) in rows:
            expected_l = (expected or '').strip().lower()
            # 현재(하이브리드) 예측
            pred_now, conf, reason = classify_color_by_hsv(int(h), int(s), int(v), [int(r), int(g), int(b)], colors_config)
            # 순수 룰 라벨
            rule_label, _rule_conf = classify_color_simple_hsv(int(h), int(s), int(v))

            expected_is_rule = (expected_l == rule_label) if expected_l else False
            predicted_is_rule = (pred_now == rule_label)

            writer.writerow([
                int(sid), int(problem_id or 0), int(hold_id or 0),
                int(h), int(s), int(v), int(r), int(g), int(b),
                expected_l, pred_now, f"{float(conf):.3f}", rule_label,
                str(expected_is_rule), str(predicted_is_rule), reason
            ])

    print(f"✅ CSV 생성: {csv_out}")


if __name__ == '__main__':
    main()
