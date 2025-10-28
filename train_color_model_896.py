#!/usr/bin/env python3
"""
🎨 896개 데이터로 색상 분류 ML 모델 학습
- 규칙 기반 결과를 초기 데이터로 활용
- 사용자 피드백으로 레이블 수정
- gray → white로 자동 변환
"""

import sys
import os

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'holdcheck'))

from backend.database import get_color_training_data, init_db
from backend.ml_trainer import train_color_model

def main():
    """896개 데이터로 색상 ML 모델 학습"""
    
    print("="*80)
    print("🎨 색상 분류 ML 모델 학습")
    print("="*80)
    
    # 데이터베이스 초기화
    try:
        init_db()
        print("✅ 데이터베이스 연결 완료")
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return
    
    # 학습 데이터 로드 (모든 데이터 사용 - 규칙 기반 결과 활용)
    print("\n📊 학습 데이터 로드 중...")
    training_data = get_color_training_data(min_samples=1, confirmed_only=False)
    
    if not training_data:
        print("❌ 학습 데이터가 없습니다!")
        print("   사용자 피드백을 먼저 수집해주세요.")
        return
    
    print(f"✅ 총 {len(training_data)}개 데이터 로드 완료")
    
    # 색상 분포 확인
    from collections import Counter
    color_counts = Counter([d['correct_color'] for d in training_data])
    print(f"\n📊 색상별 데이터 분포:")
    for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
        print(f"   {color}: {count}개 ({count/len(training_data)*100:.1f}%)")
    
    # gray 확인
    if 'gray' in color_counts:
        print(f"\n⚠️ 주의: gray 데이터 {color_counts['gray']}개 발견")
        print(f"   → 자동으로 white로 변환됩니다")
    
    # 색상별 최소 샘플 확인
    min_samples = 5
    insufficient_colors = [color for color, count in color_counts.items() if count < min_samples]
    
    if insufficient_colors:
        print(f"\n⚠️ 샘플이 부족한 색상 ({min_samples}개 미만):")
        for color in insufficient_colors:
            print(f"   {color}: {color_counts[color]}개")
        print(f"   → 학습에서 제외될 수 있습니다")
    
    # 모델 학습
    print(f"\n🤖 색상 분류 모델 학습 시작...")
    print(f"   알고리즘: Random Forest (200 trees)")
    print(f"   특징: RGB + HSV + LAB + 통계 (21차원)")
    
    try:
        test_accuracy, cv_accuracy = train_color_model(training_data)
        
        if test_accuracy > 0:
            print(f"\n🎉 학습 완료!")
            print(f"   테스트 정확도: {test_accuracy*100:.1f}%")
            print(f"   CV 정확도: {cv_accuracy*100:.1f}%")
            print(f"\n💡 다음 단계:")
            print(f"   1. 서버 재시작: docker-compose restart backend")
            print(f"   2. ML 모델이 자동으로 규칙 기반보다 우선 적용됩니다")
            print(f"   3. 더 많은 피드백을 주시면 정확도가 계속 향상됩니다")
        else:
            print(f"\n❌ 학습 실패")
            print(f"   데이터가 부족하거나 문제가 있습니다")
    
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

