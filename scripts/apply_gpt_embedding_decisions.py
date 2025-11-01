#!/usr/bin/env python3
"""
GPT/임베딩 판단 CSV를 이용해 hold_color_feedback의 user_correct_color를 업데이트.

우선순위:
  1) gpt_choice: expected/predicted/tie
  2) embedding_choice: expected/predicted/tie
tie면 스킵.

사용 예:
  python3 scripts/apply_gpt_embedding_decisions.py \
    --gpt /Users/kimjazz/Downloads/gpt_decision.csv \
    --embed /Users/kimjazz/Downloads/embedding_decision.csv
"""
import argparse
import csv
import os
import sqlite3

VALID_COLORS = {"black","white","red","orange","yellow","lime","green","mint","blue","purple","pink","brown"}


def load_csv_to_map(path: str, key: str = "id"):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rid = int(r.get(key))
            except Exception:
                continue
            m[rid] = r
    return m


def main():
    parser = argparse.ArgumentParser(description="GPT/임베딩 판단 CSV 적용")
    parser.add_argument("--gpt", required=True, help="gpt_decision.csv 경로")
    parser.add_argument("--embed", required=True, help="embedding_decision.csv 경로")
    parser.add_argument("--db", default=os.path.join(os.path.dirname(__file__), "..", "backend", "climbmate.db"))
    args = parser.parse_args()

    gpt_map = load_csv_to_map(os.path.abspath(args.gpt))
    emb_map = load_csv_to_map(os.path.abspath(args.embed))
    db_path = os.path.abspath(args.db)

    if not os.path.exists(db_path):
        raise SystemExit(f"DB 파일이 없습니다: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # id 존재 여부 확인을 위해 모두 로드
    cur.execute("SELECT id, user_correct_color FROM hold_color_feedback")
    existing_ids = {row[0] for row in cur.fetchall()}

    updated_gpt = 0
    updated_embed = 0
    skipped = 0

    # 합집합 id에 대해 처리
    all_ids = sorted(set(gpt_map.keys()) | set(emb_map.keys()))
    for rid in all_ids:
        if rid not in existing_ids:
            skipped += 1
            continue

        gpt_row = gpt_map.get(rid, {})
        emb_row = emb_map.get(rid, {})

        expected = (gpt_row.get("expected") or emb_row.get("expected") or "").strip().lower()
        predicted_now = (gpt_row.get("predicted_now") or emb_row.get("predicted_now") or "").strip().lower()
        gpt_choice = (gpt_row.get("gpt_choice") or "").strip().lower()
        emb_choice = (emb_row.get("embedding_choice") or "").strip().lower()

        target = None
        source = None
        if gpt_choice in ("expected", "predicted"):
            target = expected if gpt_choice == "expected" else predicted_now
            source = "gpt"
        elif emb_choice in ("expected", "predicted"):
            target = expected if emb_choice == "expected" else predicted_now
            source = "embed"

        if not target or target not in VALID_COLORS:
            skipped += 1
            continue

        cur.execute(
            "UPDATE hold_color_feedback SET user_correct_color = ?, confirmed = 1 WHERE id = ?",
            (target, rid)
        )
        if source == "gpt":
            updated_gpt += 1
        else:
            updated_embed += 1

    conn.commit()
    conn.close()

    print(f"✅ 업데이트 완료 | GPT 기반: {updated_gpt}건, 임베딩 기반: {updated_embed}건, 스킵: {skipped}건")


if __name__ == "__main__":
    main()


