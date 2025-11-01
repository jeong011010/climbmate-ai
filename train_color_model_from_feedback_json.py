#!/usr/bin/env python3
"""
피드백 JSON(배열)로부터 경량 ML 색상 분류 모델을 학습합니다.
- 입력: color-feedback-YYYY-..json 파일 경로
- 필터: confirmed=True, user_correct_color 존재
- 라벨: user_correct_color (gray => white)
- 출력: backend/models/color_model.pkl, color_encoder.pkl
"""

import sys
import json
from pathlib import Path

# 경로 설정
ROOT = Path(__file__).parent
BACKEND = ROOT / 'backend'

sys.path.insert(0, str(BACKEND))

from ml_trainer import train_color_model
from holdcheck.color_classifier import reset_ml_model_cache


def load_feedback_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


essential_keys = ['rgb', 'hsv', 'lab']


def normalize_sample(fb: dict):
    # 색상 라벨
    correct = (fb.get('user_correct_color') or '').strip().lower()
    if not correct:
        return None
    if correct == 'gray':
        correct = 'white'

    # HSV/RGB/LAB
    hsv = fb.get('hsv')
    if isinstance(hsv, dict):
        hsv = [int(hsv.get('h', 0)), int(hsv.get('s', 0)), int(hsv.get('v', 128))]
    rgb_obj = fb.get('rgb') or {}
    rgb = [int(rgb_obj.get('r', 128)), int(rgb_obj.get('g', 128)), int(rgb_obj.get('b', 128))]
    lab_obj = fb.get('lab') or {}
    lab = [int(lab_obj.get('l', 0)), int(lab_obj.get('a', 0)), int(lab_obj.get('b', 0))]

    if not (isinstance(hsv, (list, tuple)) and len(hsv) == 3):
        return None

    sample = {
        'rgb': rgb,
        'hsv': hsv,
        'lab': lab,
        'color_stats': fb.get('color_stats') or {},
        'area': fb.get('color_stats', {}).get('area', 0),
        'circularity': fb.get('color_stats', {}).get('circularity', 0),
        'correct_color': correct
    }
    return sample


def build_training_data(feedbacks):
    training = []
    for fb in feedbacks:
        if not isinstance(fb, dict):
            continue
        if not fb.get('confirmed', False):
            continue
        sample = normalize_sample(fb)
        if sample:
            training.append(sample)
    return training


def main():
    if len(sys.argv) < 2:
        print('사용법: python3 train_color_model_from_feedback_json.py <color-feedback.json>')
        sys.exit(1)

    fb_path = Path(sys.argv[1]).resolve()
    if not fb_path.exists():
        print(f'❌ 파일을 찾을 수 없습니다: {fb_path}')
        sys.exit(1)

    feedbacks = load_feedback_json(fb_path)
    if not isinstance(feedbacks, list):
        print('❌ 입력 JSON은 배열이어야 합니다.')
        sys.exit(1)

    training_data = build_training_data(feedbacks)
    print(f'📦 학습 후보(confirmed) 샘플 수: {len(training_data)}')

    test_acc, cv_acc = train_color_model(training_data)
    print(f'🎯 color_model test={test_acc*100:.1f}% / cv={cv_acc*100:.1f}%')

    # 런타임 캐시 초기화 (다음 호출부터 새 모델 사용)
    reset_ml_model_cache()
    print('🔄 런타임 ML 캐시 리셋 완료')


if __name__ == '__main__':
    main()


