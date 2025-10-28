#!/usr/bin/env python3
"""
🗑️ unknown 피드백 데이터 삭제
- predicted_color='unknown'인 896개 데이터 삭제
- 새로운 피드백부터 ML 학습 시작
"""

import sys
import os
import sqlite3

# DB 경로
DB_PATH = os.path.join(os.path.dirname(__file__), 'backend', 'climbmate.db')

def reset_feedback():
    """unknown 피드백 데이터 삭제"""
    
    print("="*80)
    print("🗑️ 피드백 데이터 초기화")
    print("="*80)
    
    if not os.path.exists(DB_PATH):
        print(f"❌ 데이터베이스 파일이 없습니다: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 현재 데이터 확인
    cursor.execute("SELECT COUNT(*) FROM hold_color_feedback")
    total_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM hold_color_feedback WHERE predicted_color = 'unknown'")
    unknown_count = cursor.fetchone()[0]
    
    print(f"\n📊 현재 데이터:")
    print(f"   전체: {total_count}개")
    print(f"   unknown 예측: {unknown_count}개")
    print(f"   유효 데이터: {total_count - unknown_count}개")
    
    if unknown_count == 0:
        print(f"\n✅ 삭제할 데이터가 없습니다!")
        conn.close()
        return
    
    # 삭제 확인
    print(f"\n⚠️ {unknown_count}개의 unknown 피드백을 삭제합니다.")
    print(f"   (유효한 {total_count - unknown_count}개는 유지됩니다)")
    
    # 삭제 실행
    cursor.execute("DELETE FROM hold_color_feedback WHERE predicted_color = 'unknown'")
    conn.commit()
    
    # 결과 확인
    cursor.execute("SELECT COUNT(*) FROM hold_color_feedback")
    remaining_count = cursor.fetchone()[0]
    
    print(f"\n✅ 삭제 완료!")
    print(f"   삭제: {unknown_count}개")
    print(f"   남은 데이터: {remaining_count}개")
    
    # 색상별 분포 확인
    cursor.execute("""
        SELECT user_correct_color, COUNT(*) as count
        FROM hold_color_feedback
        GROUP BY user_correct_color
        ORDER BY count DESC
    """)
    
    rows = cursor.fetchall()
    
    if rows:
        print(f"\n📊 남은 데이터 색상 분포:")
        for color, count in rows:
            print(f"   {color}: {count}개")
    
    conn.close()
    
    print(f"\n💡 다음 단계:")
    print(f"   1. 새로운 이미지 분석 시작")
    print(f"   2. 색상 피드백 제공")
    print(f"   3. 100개 이상 모이면 ML 학습 가능")

if __name__ == "__main__":
    reset_feedback()

