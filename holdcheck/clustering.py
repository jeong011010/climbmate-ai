"""
🔗 Clustering.py - Legacy Support Wrapper

이 파일은 하위 호환성을 위한 래퍼입니다.
실제 색상 분류 기능은 color_classifier.py로 이동했습니다.

migrate 전:
- 5,945줄, 89개 함수
- 92%가 미사용 레거시 코드

migrate 후:
- color_classifier.py: 600줄, 핵심 기능만
- clustering.py: 이 래퍼 + analyze_problem만

TODO: analyze_problem도 별도 모듈로 분리
"""

# ============================================================================
# 색상 분류 관련 - color_classifier.py로 이동
# ============================================================================
from color_classifier import (
    rule_based_color_clustering,
    load_color_ranges,
    save_color_ranges,
    reload_color_ranges,
    classify_color_by_hsv,
    classify_color_by_rgb,
    load_ml_color_model,
    predict_with_ml,
    reset_ml_model_cache,
    hsv_to_rgb_fast
)

# ============================================================================
# 문제 분석 관련 - legacy/clustering.py에서 임포트
# ============================================================================
import sys
import os
legacy_path = os.path.join(os.path.dirname(__file__), 'legacy')
if legacy_path not in sys.path:
    sys.path.insert(0, legacy_path)

# analyze_problem과 관련 함수들은 아직 legacy에서 가져옴
from legacy.clustering import (
    analyze_problem,
    analyze_difficulty,
    analyze_climbing_type
)

print("⚠️ clustering.py는 하위 호환성을 위한 래퍼입니다.")
print("   색상 분류: color_classifier.py 사용 중")
print("   문제 분석: legacy/clustering.py 사용 중")
