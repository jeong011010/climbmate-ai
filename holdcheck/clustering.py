"""
🔗 Clustering.py - Legacy Support Wrapper

이 파일은 하위 호환성을 위한 래퍼입니다.
실제 기능은 모듈화되어 분리되었습니다.

migrate 후:
- color_classifier.py: 색상 분류 (600줄)
- problem_analyzer.py: 문제 분석 (400줄)
- clustering.py: 하위 호환성 래퍼 (50줄)

총 5,945줄 → 1,050줄 (82% 감소)
"""

import os
import sys

# 🔧 모듈 경로 설정 (Celery worker + 직접 import 모두 호환)
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ============================================================================
# 색상 분류 관련 - color_classifier.py
# ============================================================================
try:
    # 패키지로 import 시도 (holdcheck 패키지에서 호출)
    from .color_classifier import (
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
    from .problem_analyzer import (
        analyze_problem,
        analyze_difficulty,
        analyze_climbing_type
    )
except ImportError:
    # 직접 import 시도 (스크립트에서 직접 호출)
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
    from problem_analyzer import (
        analyze_problem,
        analyze_difficulty,
        analyze_climbing_type
    )

print("✅ clustering.py 래퍼 로드 완료")
print("   색상 분류: color_classifier.py")
print("   문제 분석: problem_analyzer.py")
