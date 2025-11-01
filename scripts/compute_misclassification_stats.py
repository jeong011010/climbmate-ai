#!/usr/bin/env python3
import os
import sys
import sqlite3
from collections import defaultdict, Counter

# project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from holdcheck.color_classifier import classify_color_by_hsv, load_color_ranges


def main():
    db_path = os.path.join(ROOT, 'backend', 'climbmate.db')
    if not os.path.exists(db_path):
        print(f"❌ DB 없음: {db_path}")
        sys.exit(1)

    # load rules once
    colors_config = load_color_ranges()["colors"]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, hsv_h, hsv_s, hsv_v, rgb_r, rgb_g, rgb_b, user_correct_color
        FROM hold_color_feedback
        WHERE user_correct_color IS NOT NULL AND user_correct_color != ''
        """
    )
    rows = cur.fetchall()
    conn.close()

    total = 0
    correct = 0
    per_color = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = Counter()
    examples = defaultdict(list)  # (exp,pred) -> sample ids

    boundary_pairs = {
        frozenset(["pink", "red"]): "pink↔red",
        frozenset(["green", "mint"]): "green↔mint",
        frozenset(["red", "orange"]): "red↔orange",
        frozenset(["blue", "purple"]): "blue↔purple",
        frozenset(["black", "white"]): "black↔white",
        frozenset(["yellow", "lime"]): "yellow↔lime",
    }
    boundary_counts = Counter()

    for (sid, h, s, v, r, g, b, expected) in rows:
        try:
            expected_l = (expected or 'unknown').lower()
            pred, conf, _reason = classify_color_by_hsv(int(h), int(s), int(v), [int(r), int(g), int(b)], colors_config)
        except Exception:
            pred = 'unknown'
            conf = 0.0
            expected_l = (expected or 'unknown').lower()

        total += 1
        per_color[expected_l]["total"] += 1
        if pred == expected_l:
            correct += 1
            per_color[expected_l]["correct"] += 1
        else:
            confusion[(expected_l, pred)] += 1
            if len(examples[(expected_l, pred)]) < 5:
                examples[(expected_l, pred)].append(sid)
            bp_key = frozenset([expected_l, pred])
            if bp_key in boundary_pairs:
                boundary_counts[boundary_pairs[bp_key]] += 1

    acc = (correct / total * 100) if total else 0.0

    # 출력
    print(f"총 표본: {total}")
    print(f"정답: {correct}")
    print(f"정확도: {acc:.1f}%\n")

    print("색상별 정확도:")
    for color in sorted(per_color.keys()):
        t = per_color[color]["total"]
        c = per_color[color]["correct"]
        a = (c / t * 100) if t else 0.0
        print(f"- {color}: {c}/{t} ({a:.1f}%)")

    print("\n상위 오분류 페어(Top 10):")
    for (pair, cnt) in confusion.most_common(10):
        exp, pred = pair
        print(f"- {exp} → {pred}: {cnt}건 (예: {examples[pair]})")

    if boundary_counts:
        print("\n경계 페어 누적:")
        for name, cnt in boundary_counts.most_common():
            print(f"- {name}: {cnt}건")


if __name__ == '__main__':
    main()


