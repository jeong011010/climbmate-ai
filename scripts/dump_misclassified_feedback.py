#!/usr/bin/env python3
import os
import sys
import csv
import json
import sqlite3
from collections import defaultdict

# project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from holdcheck.color_classifier import (
    classify_color_by_hsv,
    load_color_ranges,
    calculate_confidence_hsv,
)


def best_hsv_conf_for_color(h, s, v, colors_config, color_name):
    conf = 0.0
    cfg = colors_config.get(color_name, {})
    for hsv_range in cfg.get("hsv_ranges", []):
        conf = max(conf, calculate_confidence_hsv(h, s, v, hsv_range))
    return conf


def main():
    db_path = os.path.join(ROOT, 'backend', 'climbmate.db')
    out_dir = os.path.join(ROOT, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    json_out = os.path.join(out_dir, 'misclassified_feedback.json')
    csv_out = os.path.join(out_dir, 'misclassified_feedback.csv')

    if not os.path.exists(db_path):
        print(f"❌ DB 없음: {db_path}")
        sys.exit(1)

    colors_config = load_color_ranges()["colors"]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, problem_id, hold_id,
               rgb_r, rgb_g, rgb_b,
               hsv_h, hsv_s, hsv_v,
               lab_l, lab_a, lab_b,
               predicted_color, user_correct_color,
               created_at
        FROM hold_color_feedback
        WHERE user_correct_color IS NOT NULL AND user_correct_color != ''
        ORDER BY created_at DESC
        """
    )
    rows = cur.fetchall()
    conn.close()

    total, mis = 0, 0
    results = []

    for (sid, problem_id, hold_id,
         r, g, b,
         h, s, v,
         L, A, B,
         db_pred, expected,
         created_at) in rows:
        total += 1
        expected_l = (expected or 'unknown').lower()
        # 현재 분류기(룰+ML)로 재평가
        pred, conf, reason = classify_color_by_hsv(int(h), int(s), int(v), [int(r), int(g), int(b)], colors_config)

        if pred != expected_l:
            mis += 1
            # 휴리스틱: 현재 HSV가 각 색의 범위와 얼마나 맞는지
            conf_expected_hsv = best_hsv_conf_for_color(int(h), int(s), int(v), colors_config, expected_l)
            conf_pred_hsv = best_hsv_conf_for_color(int(h), int(s), int(v), colors_config, pred)
            label_suspicion_score = round(conf_pred_hsv - conf_expected_hsv, 3)
            # 경계 힌트
            boundary = None
            pair = frozenset([expected_l, pred])
            if pair == frozenset(["pink", "red"]):
                boundary = "pink↔red"
            elif pair == frozenset(["green", "mint"]):
                boundary = "green↔mint"
            elif pair == frozenset(["red", "orange"]):
                boundary = "red↔orange"
            elif pair == frozenset(["blue", "purple"]):
                boundary = "blue↔purple"
            elif pair == frozenset(["black", "white"]):
                boundary = "black↔white"

            results.append({
                "id": int(sid),
                "problem_id": int(problem_id or 0),
                "hold_id": int(hold_id or 0),
                "hsv": [int(h), int(s), int(v)],
                "rgb": [int(r), int(g), int(b)],
                "lab": [int(L), int(A), int(B)],
                "expected": expected_l,
                "predicted_now": pred,
                "confidence": round(float(conf), 3),
                "db_predicted_at_insert": (db_pred or ''),
                "reason": reason,
                "boundary_pair": boundary,
                "label_suspicion_score": label_suspicion_score,
                "created_at": created_at,
            })

    # 저장(JSON)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump({
            "total": total,
            "misclassified": mis,
            "accuracy": round((total - mis) / total * 100, 1) if total else 0.0,
            "items": results
        }, f, ensure_ascii=False, indent=2)

    # 저장(CSV)
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id","problem_id","hold_id","H","S","V","R","G","B","L","a","b","expected","predicted_now","confidence","db_predicted_at_insert","boundary_pair","label_suspicion_score","created_at","reason"])
        for it in results:
            h, s, v = it["hsv"]
            r, g, b = it["rgb"]
            L, A, B = it["lab"]
            writer.writerow([
                it["id"], it["problem_id"], it["hold_id"],
                h, s, v, r, g, b, L, A, B,
                it["expected"], it["predicted_now"], it["confidence"], it["db_predicted_at_insert"],
                it.get("boundary_pair") or "", it["label_suspicion_score"], it["created_at"], it["reason"]
            ])

    print(f"총 표본: {total}, 오분류: {mis}, 정확도: {((total-mis)/total*100 if total else 0):.1f}%")
    print(f"JSON: {json_out}")
    print(f"CSV : {csv_out}")


if __name__ == '__main__':
    main()
