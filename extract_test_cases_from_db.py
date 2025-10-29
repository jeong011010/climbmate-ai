#!/usr/bin/env python3
"""
DB에서 피드백 데이터를 추출해서 테스트 케이스 JSON 파일에 추가하는 스크립트
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# backend 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from database import get_all_color_feedbacks
except ImportError as e:
    print(f"❌ DB 모듈 import 실패: {e}")
    print("💡 backend/database.py 파일이 있는지 확인하세요.")
    sys.exit(1)

# 테스트 케이스 파일 경로
TEST_CASES_FILE = Path(__file__).parent / "test_cases" / "color_classification_test_cases.json"

def load_existing_test_cases():
    """기존 테스트 케이스 로드"""
    if not TEST_CASES_FILE.exists():
        return {
            "description": "색상 분류 테스트 케이스 - 피드백 누적용",
            "version": "1.0.0",
            "test_cases": []
        }
    
    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_test_cases(test_data):
    """테스트 케이스 저장"""
    with open(TEST_CASES_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 테스트 케이스 파일 저장 완료: {TEST_CASES_FILE}")

def extract_test_cases_from_db():
    """DB에서 피드백 데이터 추출 후 테스트 케이스로 변환"""
    
    print("=" * 80)
    print("📊 DB에서 피드백 데이터 추출 중...")
    print("=" * 80)
    
    try:
        feedbacks = get_all_color_feedbacks()
        print(f"✅ 총 {len(feedbacks)}개의 피드백 데이터 로드 완료")
    except Exception as e:
        print(f"❌ 피드백 데이터 로드 실패: {e}")
        sys.exit(1)
    
    if not feedbacks:
        print("⚠️  피드백 데이터가 없습니다.")
        return
    
    # 잘못된 분류 케이스만 필터링 (predicted != user_correct)
    mismatched = [
        f for f in feedbacks 
        if f['predicted_color'] and f['user_correct_color'] 
        and f['predicted_color'].lower() != f['user_correct_color'].lower()
    ]
    
    print(f"\n📈 분석 결과:")
    print(f"  - 전체 피드백: {len(feedbacks)}개")
    print(f"  - 잘못된 분류: {len(mismatched)}개")
    
    if not mismatched:
        print("\n✅ 모든 피드백이 올바르게 분류되었습니다!")
        return
    
    # 기존 테스트 케이스 로드
    test_data = load_existing_test_cases()
    existing_ids = {tc['id'] for tc in test_data.get('test_cases', [])}
    
    # HSV 값을 기반으로 중복 제거 (같은 HSV는 하나만)
    unique_cases = {}
    new_cases = []
    
    for i, feedback in enumerate(mismatched):
        hsv = feedback['hsv']
        if len(hsv) != 3:
            continue
        
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        hsv_key = f"{h}_{s}_{v}"
        
        # 중복 확인 (같은 HSV 값이면 스킵)
        if hsv_key in unique_cases:
            continue
        
        unique_cases[hsv_key] = True
        
        # 테스트 케이스 ID 생성
        test_id = f"db_{feedback['id']}"
        if test_id in existing_ids:
            continue  # 이미 존재하는 케이스는 스킵
        
        predicted = feedback['predicted_color'].lower()
        correct = feedback['user_correct_color'].lower()
        
        # gray는 white로 변환 (이미 제거했지만 확인)
        if correct == 'gray':
            correct = 'white'
        
        test_case = {
            "id": test_id,
            "name": f"DB Feedback #{feedback['id']} - {correct.upper()}",
            "hsv": [h, s, v],
            "expected": correct,
            "description": f"잘못 분류됨: {predicted.upper()} → {correct.upper()} (문제 #{feedback.get('problem_id', '?')}, 홀드 #{feedback.get('hold_id', '?')})",
            "date_added": feedback.get('created_at', datetime.now().strftime("%Y-%m-%d")),
            "fix_applied": f"AI가 {predicted.upper()}로 예측했으나 실제는 {correct.upper()}"
        }
        
        new_cases.append(test_case)
        existing_ids.add(test_id)
    
    print(f"\n📝 테스트 케이스 변환:")
    print(f"  - 새로운 테스트 케이스: {len(new_cases)}개")
    print(f"  - 중복 제거 후: {len(unique_cases)}개 유니크 케이스")
    
    # 기존 테스트 케이스에 추가
    if 'test_cases' not in test_data:
        test_data['test_cases'] = []
    
    # 기존 케이스와 새 케이스 병합
    all_cases = test_data['test_cases'] + new_cases
    
    # ID로 정렬 (DB 피드백은 최신순, 기존 케이스는 유지)
    test_data['test_cases'] = all_cases
    test_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 저장
    save_test_cases(test_data)
    
    print(f"\n✅ 완료!")
    print(f"  - 총 테스트 케이스: {len(all_cases)}개")
    print(f"  - 기존: {len(test_data['test_cases']) - len(new_cases)}개")
    print(f"  - 신규: {len(new_cases)}개")
    
    # 색상별 통계
    color_stats = {}
    for tc in all_cases:
        color = tc.get('expected', 'unknown').lower()
        color_stats[color] = color_stats.get(color, 0) + 1
    
    print(f"\n📊 색상별 테스트 케이스 통계:")
    for color, count in sorted(color_stats.items()):
        print(f"  {color.upper()}: {count}개")

if __name__ == "__main__":
    try:
        extract_test_cases_from_db()
    except KeyboardInterrupt:
        print("\n\n⚠️  중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

