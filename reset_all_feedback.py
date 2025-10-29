#!/usr/bin/env python3
"""
피드백 데이터 완전 초기화 스크립트
모든 color_feedback 데이터를 삭제합니다.
"""

import sqlite3
import os

# Docker 환경 감지
if os.path.exists('/app/backend/climbmate.db'):
    DB_PATH = '/app/backend/climbmate.db'
else:
    DB_PATH = 'backend/climbmate.db'

def reset_all_feedback():
    """모든 피드백 데이터 삭제"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 삭제 전 개수 확인
    cursor.execute("SELECT COUNT(*) FROM color_feedback")
    before_count = cursor.fetchone()[0]
    
    print(f"🗑️  피드백 데이터 완전 초기화")
    print(f"=" * 80)
    print(f"삭제 전 데이터: {before_count}개")
    
    # 모든 피드백 데이터 삭제
    cursor.execute("DELETE FROM color_feedback")
    conn.commit()
    
    # 삭제 후 확인
    cursor.execute("SELECT COUNT(*) FROM color_feedback")
    after_count = cursor.fetchone()[0]
    
    print(f"삭제 후 데이터: {after_count}개")
    print(f"✅ {before_count}개 피드백 데이터 삭제 완료!")
    print(f"=" * 80)
    
    conn.close()

if __name__ == "__main__":
    reset_all_feedback()

