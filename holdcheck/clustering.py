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

# ============================================================================
# 색상 분류 관련 - color_classifier.py
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
# 문제 분석 관련 - problem_analyzer.py
# ============================================================================
from problem_analyzer import (
    analyze_problem,
    analyze_difficulty,
    analyze_climbing_type
)

print("✅ clustering.py 래퍼 로드 완료")
print("   색상 분류: color_classifier.py")
print("   문제 분석: problem_analyzer.py")
