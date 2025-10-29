#!/usr/bin/env python3
"""
디버그: 특정 HSV 값이 color_ranges.json과 매칭되는지 확인
"""
import json

# 테스트 HSV 값 (문제가 되는 홀드)
test_hsv = {
    "h": 105,  # 예시
    "s": 14,   # 예시  
    "v": 163   # 예시
}

# color_ranges.json 로드
with open('holdcheck/color_ranges.json', 'r', encoding='utf-8') as f:
    ranges_data = json.load(f)

print(f"🔍 테스트 HSV: H={test_hsv['h']}, S={test_hsv['s']}, V={test_hsv['v']}\n")
print("=" * 80)

# 각 색상별 매칭 여부 확인
sorted_colors = sorted(ranges_data['colors'].items(), key=lambda x: x[1].get("priority", 999))

for color_name, config in sorted_colors:
    if "hsv_ranges" in config:
        for hsv_range in config["hsv_ranges"]:
            h_min, h_max = hsv_range["h"]
            s_min, s_max = hsv_range["s"]
            v_min, v_max = hsv_range["v"]
            
            h_match = h_min <= test_hsv['h'] <= h_max
            s_match = s_min <= test_hsv['s'] <= s_max
            v_match = v_min <= test_hsv['v'] <= v_max
            
            if h_match and s_match and v_match:
                print(f"✅ {color_name}: 매칭!")
                print(f"   범위: H[{h_min},{h_max}], S[{s_min},{s_max}], V[{v_min},{v_max}]")
                print(f"   우선순위: {config.get('priority', 999)}")
            else:
                print(f"❌ {color_name}: 불일치")
                print(f"   범위: H[{h_min},{h_max}], S[{s_min},{s_max}], V[{v_min},{v_max}]")
                if not h_match:
                    print(f"   → H 불일치 ({test_hsv['h']} ∉ [{h_min},{h_max}])")
                if not s_match:
                    print(f"   → S 불일치 ({test_hsv['s']} ∉ [{s_min},{s_max}])")
                if not v_match:
                    print(f"   → V 불일치 ({test_hsv['v']} ∉ [{v_min},{v_max}])")
    print()

