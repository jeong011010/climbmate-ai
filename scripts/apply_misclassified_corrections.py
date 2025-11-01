#!/usr/bin/env python3
"""
misclassified_rule_decisions.csv를 이용해 hold_color_feedback의 정답 라벨(user_correct_color)을 수정합니다.

규칙:
- expected_is_rule=True & predicted_is_rule=False  → user_correct_color = expected
- expected_is_rule=False & predicted_is_rule=True  → user_correct_color = predicted_now
- 둘 다 True                                  → expected 우선
- 둘 다 False                                 → rule_label가 유효 색이면 rule_label, 아니면 스킵

실행 예:
  python3 scripts/apply_misclassified_corrections.py \
      --csv /Users/kimjazz/Downloads/misclassified_rule_decisions.csv
"""
import argparse
import csv
import os
import sqlite3

VALID_COLORS = {"black","white","red","orange","yellow","lime","green","mint","blue","purple","pink","brown"}


def parse_bool(val):
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("1","true","yes","y","t")


def main():
    parser = argparse.ArgumentParser(description="CSV 기반 정답 라벨 수정")
    parser.add_argument("--csv", required=True, help="CSV 파일 절대 경로")
    parser.add_argument("--db", default=os.path.join(os.path.dirname(__file__), "..", "backend", "climbmate.db"))
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    db_path = os.path.abspath(args.db)

    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    if not os.path.exists(db_path):
        raise SystemExit(f"DB 파일을 찾을 수 없습니다: {db_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    updated = 0
    skipped = 0
    both_false = 0
    both_true = 0

    for r in rows:
        try:
            fb_id = int(r.get("id"))
        except Exception:
            continue

        expected = (r.get("expected") or "").strip().lower()
        predicted_now = (r.get("predicted_now") or "").strip().lower()
        rule_label = (r.get("rule_label") or "").strip().lower()
        exp_true = parse_bool(r.get("expected_is_rule"))
        pred_true = parse_bool(r.get("predicted_is_rule"))

        target = None
        if exp_true and not pred_true:
            target = expected
        elif pred_true and not exp_true:
            target = predicted_now
        elif exp_true and pred_true:
            both_true += 1
            target = expected  # tie → expected 우선
        else:
            # 둘 다 False → rule_label이 유효하면 사용, 아니면 스킵
            both_false += 1
            if rule_label in VALID_COLORS:
                target = rule_label

        if target is None or target not in VALID_COLORS:
            skipped += 1
            continue

        cur.execute(
            "UPDATE hold_color_feedback SET user_correct_color = ?, confirmed = 1 WHERE id = ?",
            (target, fb_id)
        )
        updated += 1

    conn.commit()
    conn.close()

    print(f"✅ 업데이트 완료: {updated}건")
    print(f"   스킵: {skipped}건 (유효 타깃 불명) | 둘다 False: {both_false} | 둘다 True: {both_true}")


if __name__ == "__main__":
    main()


