#!/usr/bin/env python3
"""
🎯 896개 피드백으로 color_ranges.json 자동 튜닝
- HSV 범위 자동 조정
- 오분류 패턴 기반 최적화
- 규칙 기반 정확도 68.4% → 75%+ 목표
"""

import os
import sys
import json
import sqlite3
import numpy as np
from collections import defaultdict

# DB 경로 (Docker 환경 고려)
if os.path.exists('/app/backend/climbmate.db'):
    DB_PATH = '/app/backend/climbmate.db'
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), 'backend', 'climbmate.db')

# color_ranges.json 경로
COLOR_RANGES_PATH = os.path.join(os.path.dirname(__file__), 'holdcheck', 'color_ranges.json')

print("="*80)
print("🎯 규칙 기반 자동 튜닝 (896개 피드백)")
print("="*80)

# 1. 피드백 데이터 로드
print(f"\n📊 Step 1: 피드백 데이터 로드")
print(f"   DB: {DB_PATH}")

if not os.path.exists(DB_PATH):
    print(f"❌ 데이터베이스 없음!")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        user_correct_color,
        hsv_h, hsv_s, hsv_v,
        rgb_r, rgb_g, rgb_b
    FROM hold_color_feedback
    WHERE user_correct_color != 'unknown'
""")

feedbacks = cursor.fetchall()
conn.close()

print(f"   ✅ {len(feedbacks)}개 피드백 로드")

# 2. 색상별 HSV 데이터 수집
print(f"\n📈 Step 2: 색상별 HSV 분석")

color_data = defaultdict(lambda: {'h': [], 's': [], 'v': []})

for color, h, s, v, r, g, b in feedbacks:
    if h is not None and s is not None and v is not None:
        color_data[color]['h'].append(h)
        color_data[color]['s'].append(s)
        color_data[color]['v'].append(v)

# 3. 각 색상별 최적 범위 계산
print(f"\n🔧 Step 3: 최적 HSV 범위 계산")

optimized_ranges = {}

for color, data in color_data.items():
    if len(data['h']) < 5:  # 최소 5개 필요
        print(f"   ⚠️ {color}: 데이터 부족 ({len(data['h'])}개) - 스킵")
        continue
    
    h_arr = np.array(data['h'])
    s_arr = np.array(data['s'])
    v_arr = np.array(data['v'])
    
    # 통계 계산
    h_mean, h_std = np.mean(h_arr), np.std(h_arr)
    s_mean, s_std = np.mean(s_arr), np.std(s_arr)
    v_mean, v_std = np.mean(v_arr), np.std(v_arr)
    
    # 최적 범위: mean ± 2*std (95% 커버)
    # 단, 최소/최대값 고려하여 너무 좁아지지 않게
    h_min = max(0, int(np.percentile(h_arr, 5)))  # 하위 5%
    h_max = min(180, int(np.percentile(h_arr, 95)))  # 상위 5%
    
    s_min = max(0, int(np.percentile(s_arr, 5)))
    s_max = min(255, int(np.percentile(s_arr, 95)))
    
    v_min = max(0, int(np.percentile(v_arr, 5)))
    v_max = min(255, int(np.percentile(v_arr, 95)))
    
    # Red는 H가 0 근처와 180 근처 두 범위
    if color == 'red':
        # H가 170 이상인 것들 분리
        h_high = h_arr[h_arr > 160]
        h_low = h_arr[h_arr <= 160]
        
        if len(h_high) > 0 and len(h_low) > 0:
            # 두 범위로 분리
            optimized_ranges[color] = {
                'ranges': [
                    {
                        'h': [int(np.percentile(h_low, 5)), int(np.percentile(h_low, 95))],
                        's': [s_min, s_max],
                        'v': [v_min, v_max]
                    },
                    {
                        'h': [int(np.percentile(h_high, 5)), 180],
                        's': [s_min, s_max],
                        'v': [v_min, v_max]
                    }
                ],
                'sample_count': len(h_arr)
            }
            print(f"   ✅ {color}: H[{optimized_ranges[color]['ranges'][0]['h'][0]}-{optimized_ranges[color]['ranges'][0]['h'][1]}] & [{optimized_ranges[color]['ranges'][1]['h'][0]}-180], S[{s_min}-{s_max}], V[{v_min}-{v_max}] ({len(h_arr)}개)")
            continue
    
    optimized_ranges[color] = {
        'ranges': [{
            'h': [h_min, h_max],
            's': [s_min, s_max],
            'v': [v_min, v_max]
        }],
        'sample_count': len(h_arr)
    }
    
    print(f"   ✅ {color}: H[{h_min}-{h_max}], S[{s_min}-{s_max}], V[{v_min}-{v_max}] ({len(h_arr)}개)")

# 4. color_ranges.json 로드
print(f"\n📂 Step 4: color_ranges.json 로드")

if not os.path.exists(COLOR_RANGES_PATH):
    print(f"   ❌ color_ranges.json 없음: {COLOR_RANGES_PATH}")
    exit(1)

with open(COLOR_RANGES_PATH, 'r', encoding='utf-8') as f:
    color_ranges = json.load(f)

print(f"   ✅ 기존 설정 로드 완료")

# 5. 기존 설정과 비교 & 업데이트
print(f"\n🔄 Step 5: HSV 범위 업데이트")

updated_count = 0

for color, opt_data in optimized_ranges.items():
    if color not in color_ranges['colors']:
        print(f"   ⚠️ {color}: color_ranges.json에 없음 - 스킵")
        continue
    
    old_ranges = color_ranges['colors'][color].get('hsv_ranges', [])
    new_ranges = opt_data['ranges']
    
    # 업데이트
    color_ranges['colors'][color]['hsv_ranges'] = new_ranges
    
    updated_count += 1
    
    # 변경 사항 출력
    if old_ranges:
        old_h = old_ranges[0].get('h', [])
        new_h = new_ranges[0].get('h', [])
        print(f"   🔄 {color}: H {old_h} → {new_h}")
    else:
        print(f"   ✨ {color}: 새 범위 추가")

# 6. 저장
print(f"\n💾 Step 6: color_ranges.json 저장")

# 백업
backup_path = COLOR_RANGES_PATH + '.backup'
import shutil
shutil.copy(COLOR_RANGES_PATH, backup_path)
print(f"   📦 백업: {backup_path}")

# 메타데이터 업데이트
color_ranges['version'] = '2.0-auto-tuned'
color_ranges['last_updated'] = __import__('datetime').datetime.now().isoformat()
color_ranges['feedback_count'] = len(feedbacks)

# 저장
with open(COLOR_RANGES_PATH, 'w', encoding='utf-8') as f:
    json.dump(color_ranges, f, indent=2, ensure_ascii=False)

print(f"   ✅ 저장 완료: {COLOR_RANGES_PATH}")

# 7. 결과 요약
print(f"\n" + "="*80)
print(f"🎉 자동 튜닝 완료!")
print(f"="*80)
print(f"   📊 피드백 데이터: {len(feedbacks)}개")
print(f"   🔄 업데이트된 색상: {updated_count}개")
print(f"   💾 백업: {backup_path}")
print(f"\n💡 다음 단계:")
print(f"   1. docker compose restart backend")
print(f"   2. 새 이미지 분석해보기")
print(f"   3. 정확도 향상 확인!")
print(f"\n   예상: 68.4% → 75%+")

