#!/usr/bin/env python3
"""
📊 피드백 데이터 상태 확인
"""

import os
import sqlite3
from collections import Counter

# DB 경로 (Docker 환경 고려)
if os.path.exists('/app/backend/climbmate.db'):
    DB_PATH = '/app/backend/climbmate.db'
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), 'backend', 'climbmate.db')

print("="*80)
print("📊 피드백 데이터 상태 확인")
print("="*80)
print(f"\nDB 경로: {DB_PATH}")

if not os.path.exists(DB_PATH):
    print(f"❌ 데이터베이스 파일이 없습니다!")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 전체 데이터
cursor.execute("SELECT COUNT(*) FROM hold_color_feedback")
total = cursor.fetchone()[0]
print(f"\n📊 전체 피드백: {total}개")

# predicted_color 분포
cursor.execute("SELECT predicted_color, COUNT(*) FROM hold_color_feedback GROUP BY predicted_color ORDER BY COUNT(*) DESC")
predicted_dist = cursor.fetchall()

print(f"\n🔍 predicted_color 분포:")
for color, count in predicted_dist:
    print(f"   {color}: {count}개")

# user_correct_color 분포
cursor.execute("SELECT user_correct_color, COUNT(*) FROM hold_color_feedback GROUP BY user_correct_color ORDER BY COUNT(*) DESC")
correct_dist = cursor.fetchall()

print(f"\n✅ user_correct_color 분포:")
for color, count in correct_dist:
    print(f"   {color}: {count}개")

# 일치율 확인
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN predicted_color = user_correct_color THEN 1 ELSE 0 END) as correct,
        SUM(CASE WHEN predicted_color != user_correct_color THEN 1 ELSE 0 END) as wrong
    FROM hold_color_feedback
    WHERE predicted_color != 'unknown'
""")
total, correct, wrong = cursor.fetchone()

if total and total > 0:
    accuracy = (correct / total * 100) if correct else 0
    print(f"\n📈 예측 정확도:")
    print(f"   일치: {correct}개 ({accuracy:.1f}%)")
    print(f"   불일치: {wrong}개 ({100-accuracy:.1f}%)")

# 샘플 5개 확인
cursor.execute("""
    SELECT id, predicted_color, user_correct_color, created_at
    FROM hold_color_feedback
    ORDER BY id DESC
    LIMIT 5
""")

print(f"\n🔬 최근 데이터 샘플 (5개):")
for row in cursor.fetchall():
    id, pred, correct, created = row
    match = "✅" if pred == correct else "❌"
    print(f"   {match} ID {id}: {pred} → {correct} ({created})")

conn.close()

print(f"\n💡 결론:")
if total == 0:
    print(f"   ✅ 데이터가 깨끗합니다. 새로 시작하세요!")
elif predicted_dist and predicted_dist[0][0] == 'unknown':
    print(f"   ⚠️ unknown 데이터가 있습니다. 삭제 권장!")
else:
    print(f"   ✅ 데이터가 유효합니다. ML 학습 가능!")


