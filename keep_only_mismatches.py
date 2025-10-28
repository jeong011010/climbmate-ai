#!/usr/bin/env python3
"""
🗑️ 일치하는 피드백 삭제, 불일치만 유지
- predicted = correct → 삭제 (이미 잘 맞음)
- predicted ≠ correct → 유지 (학습 필요)
"""

import os
import sqlite3

# DB 경로 (Docker 환경 고려)
if os.path.exists('/app/backend/climbmate.db'):
    DB_PATH = '/app/backend/climbmate.db'
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), 'backend', 'climbmate.db')

print("="*80)
print("🗑️ 일치 데이터 삭제 (불일치만 유지)")
print("="*80)

if not os.path.exists(DB_PATH):
    print(f"❌ 데이터베이스 없음: {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 현재 상태 확인
cursor.execute("SELECT COUNT(*) FROM hold_color_feedback")
total = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) 
    FROM hold_color_feedback 
    WHERE predicted_color = user_correct_color
""")
matches = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) 
    FROM hold_color_feedback 
    WHERE predicted_color != user_correct_color
""")
mismatches = cursor.fetchone()[0]

print(f"\n📊 현재 데이터:")
print(f"   전체: {total}개")
print(f"   일치 (삭제 대상): {matches}개")
print(f"   불일치 (유지): {mismatches}개")

if matches == 0:
    print(f"\n✅ 삭제할 데이터가 없습니다!")
    conn.close()
    exit(0)

print(f"\n⚠️ {matches}개의 일치 데이터를 삭제합니다.")
print(f"   (불일치 {mismatches}개는 유지됩니다)")

# 일치 데이터 삭제
cursor.execute("""
    DELETE FROM hold_color_feedback 
    WHERE predicted_color = user_correct_color
""")
conn.commit()

# 결과 확인
cursor.execute("SELECT COUNT(*) FROM hold_color_feedback")
remaining = cursor.fetchone()[0]

print(f"\n✅ 삭제 완료!")
print(f"   삭제: {matches}개")
print(f"   남은 데이터: {remaining}개")

# 색상별 분포 확인
cursor.execute("""
    SELECT user_correct_color, COUNT(*) as count
    FROM hold_color_feedback
    GROUP BY user_correct_color
    ORDER BY count DESC
""")

rows = cursor.fetchall()

if rows:
    print(f"\n📊 남은 데이터 색상 분포 (오분류만):")
    for color, count in rows:
        print(f"   {color}: {count}개")

# 샘플 확인
cursor.execute("""
    SELECT predicted_color, user_correct_color, COUNT(*) as count
    FROM hold_color_feedback
    GROUP BY predicted_color, user_correct_color
    ORDER BY count DESC
    LIMIT 10
""")

rows = cursor.fetchall()

if rows:
    print(f"\n🔍 주요 오분류 패턴 (Top 10):")
    for pred, correct, count in rows:
        print(f"   {pred} → {correct}: {count}회")

conn.close()

print(f"\n💡 다음 단계:")
print(f"   1. docker compose exec backend python3 auto_tune_color_rules.py")
print(f"   2. docker compose restart backend")
print(f"   3. 오분류만으로 규칙 개선!")
print(f"\n   예상: 오분류 패턴에 집중하여 정확도 대폭 향상!")


