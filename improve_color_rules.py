#!/usr/bin/env python3
"""
🎨 896개 피드백 데이터로 color_ranges.json 개선
- 오분류 패턴 분석
- HSV 임계값 자동 조정
- 최적 범위 찾기
"""

import sys
import os
import json
from collections import defaultdict
import numpy as np

# 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.database import get_color_training_data, init_db

def analyze_misclassifications():
    """오분류 패턴 분석"""
    print("="*80)
    print("🎨 색상 분류 오분류 분석")
    print("="*80)
    
    # 데이터베이스 초기화
    try:
        init_db()
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return
    
    # 피드백 데이터 로드
    training_data = get_color_training_data(min_samples=1, confirmed_only=False)
    
    print(f"\n📊 로드된 데이터: {len(training_data)}개")
    
    if len(training_data) == 0:
        print("❌ 데이터가 없습니다!")
        return
    
    # 오분류 분석
    misclassifications = defaultdict(list)
    
    for data in training_data:
        predicted = data.get('predicted_color', 'unknown')
        correct = data.get('correct_color', 'unknown')
        
        if predicted != correct:
            misclassifications[correct].append({
                'predicted': predicted,
                'hsv': [data.get('hsv_h', 0), data.get('hsv_s', 0), data.get('hsv_v', 0)],
                'rgb': [data.get('rgb_r', 0), data.get('rgb_g', 0), data.get('rgb_b', 0)]
            })
    
    # 결과 출력
    print(f"\n📈 오분류 통계:")
    total_errors = sum(len(errors) for errors in misclassifications.values())
    print(f"   총 오분류: {total_errors}개 ({total_errors/len(training_data)*100:.1f}%)")
    
    for color, errors in sorted(misclassifications.items(), key=lambda x: -len(x[1])):
        print(f"\n   {color}:")
        print(f"      오분류 {len(errors)}회")
        
        # 잘못 예측된 색상
        predicted_colors = defaultdict(int)
        for err in errors:
            predicted_colors[err['predicted']] += 1
        
        for pred_color, count in sorted(predicted_colors.items(), key=lambda x: -x[1])[:3]:
            print(f"      → {pred_color}로 잘못 예측: {count}회")
        
        # HSV 범위 제안
        if errors:
            h_values = [e['hsv'][0] for e in errors if e['hsv'][0] > 0]
            s_values = [e['hsv'][1] for e in errors if e['hsv'][1] > 0]
            v_values = [e['hsv'][2] for e in errors if e['hsv'][2] > 0]
            
            if h_values:
                print(f"      H 범위: {min(h_values)}~{max(h_values)}")
            if s_values:
                print(f"      S 범위: {min(s_values)}~{max(s_values)}")
            if v_values:
                print(f"      V 범위: {min(v_values)}~{max(v_values)}")
    
    # 개선 방안
    print(f"\n💡 개선 방안:")
    print(f"   1. color_ranges.json의 HSV 범위 조정")
    print(f"   2. 가장 많은 오분류 색상부터 수정 권장")
    print(f"   3. 대조되는 색상 쌍 주의 (예: pink-red, lime-yellow)")
    
    return misclassifications

def suggest_rules():
    """896개 데이터로 규칙 제안"""
    misclassifications = analyze_misclassifications()
    
    if not misclassifications:
        print("\n✅ 오분류가 없습니다!")
        return
    
    print(f"\n🎯 규칙 개선 제안:")
    print(f"   color_ranges.json 수정이 필요합니다")
    
    # 상위 3개 색상
    top_colors = sorted(misclassifications.items(), key=lambda x: -len(x[1]))[:3]
    
    print(f"\n   최우선 개선 색상:")
    for color, errors in top_colors:
        print(f"   - {color}: {len(errors)}회 오분류")

if __name__ == "__main__":
    suggest_rules()


