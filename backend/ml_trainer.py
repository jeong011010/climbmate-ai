"""
자체 ML 모델 학습 파이프라인
검증된 데이터 50개 이상부터 학습 가능
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from typing import List, Dict, Tuple

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DIFFICULTY_MODEL_PATH = os.path.join(MODEL_DIR, 'difficulty_model.pkl')
TYPE_MODEL_PATH = os.path.join(MODEL_DIR, 'type_model.pkl')
DIFFICULTY_ENCODER_PATH = os.path.join(MODEL_DIR, 'difficulty_encoder.pkl')
TYPE_ENCODER_PATH = os.path.join(MODEL_DIR, 'type_encoder.pkl')

# 🎨 색상 분류 모델 경로
COLOR_MODEL_PATH = os.path.join(MODEL_DIR, 'color_model.pkl')
COLOR_ENCODER_PATH = os.path.join(MODEL_DIR, 'color_encoder.pkl')

def extract_features(holds_data: List[Dict], stats: Dict = None) -> np.ndarray:
    """홀드 데이터로부터 특징 벡터 추출"""
    
    num_holds = len(holds_data)
    
    # 홀드 크기 통계
    areas = [h.get('area', 0) for h in holds_data]
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    std_area = np.std(areas)
    
    # 홀드 위치 통계
    centers = np.array([h.get('center', [0, 0]) for h in holds_data])
    
    # 거리 계산
    distances = []
    consecutive_distances = []
    if num_holds > 1:
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)
        
        # 높이 순으로 정렬하여 연속 거리
        sorted_indices = np.argsort(centers[:, 1])[::-1]
        for i in range(len(sorted_indices) - 1):
            dist = np.linalg.norm(
                centers[sorted_indices[i]] - centers[sorted_indices[i+1]]
            )
            consecutive_distances.append(dist)
    
    max_distance = max(distances) if distances else 0
    avg_distance = np.mean(distances) if distances else 0
    avg_consecutive = np.mean(consecutive_distances) if consecutive_distances else 0
    
    # 높이/수평 범위
    heights = centers[:, 1]
    horizontals = centers[:, 0]
    height_range = np.ptp(heights) if num_holds > 1 else 0
    horizontal_range = np.ptp(horizontals) if num_holds > 1 else 0
    
    # 수평/수직 비율
    movement_ratio = horizontal_range / (height_range + 1)
    
    # 색상 분포
    colors = [h.get('color_name', 'unknown') for h in holds_data]
    unique_colors = len(set(colors))
    
    # 특징 벡터 (25개 특징)
    features = [
        num_holds,              # 1. 홀드 개수
        avg_area,               # 2. 평균 홀드 크기
        min_area,               # 3. 최소 홀드 크기
        max_area,               # 4. 최대 홀드 크기
        std_area,               # 5. 홀드 크기 분산
        max_distance,           # 6. 최대 홀드 간격
        avg_distance,           # 7. 평균 홀드 간격
        avg_consecutive,        # 8. 연속 홀드 평균 간격
        height_range,           # 9. 높이 범위
        horizontal_range,       # 10. 수평 범위
        movement_ratio,         # 11. 이동 비율 (수평/수직)
        unique_colors,          # 12. 고유 색상 수
        # 비율 특징
        len([a for a in areas if a < 1200]) / num_holds,  # 13. 작은 홀드 비율
        len([a for a in areas if a > 3500]) / num_holds,  # 14. 큰 홀드 비율
        # 분포 특징
        np.std(centers[:, 0]) if num_holds > 1 else 0,    # 15. 수평 분산
        np.std(centers[:, 1]) if num_holds > 1 else 0,    # 16. 수직 분산
        # 거리 분산
        np.std(distances) if distances else 0,            # 17. 거리 분산
        # 밀도
        num_holds / (height_range * horizontal_range + 1),  # 18. 홀드 밀도
        # 평균 위치
        np.mean(centers[:, 0]),                           # 19. 평균 X 위치
        np.mean(centers[:, 1]),                           # 20. 평균 Y 위치
        # 최상단/최하단 거리
        np.max(heights) - np.min(heights) if num_holds > 1 else 0,  # 21. 높이 변화
        # 연속 거리 분산
        np.std(consecutive_distances) if consecutive_distances else 0,  # 22. 연속 거리 분산
        # 홀드 크기 범위
        max_area - min_area,                              # 23. 크기 범위
        # 극단값 비율
        len([d for d in distances if d > 150]) / len(distances) if distances else 0,  # 24. 큰 점프 비율
        len([a for a in areas if a < 1000]) / num_holds   # 25. 극소형 홀드 비율
    ]
    
    return np.array(features)

def train_difficulty_model(training_data: List[Dict]) -> Tuple[float, float]:
    """난이도 예측 모델 학습"""
    
    print(f"\n🎓 난이도 모델 학습 시작...")
    print(f"   훈련 데이터: {len(training_data)}개")
    
    # 특징 추출
    X = []
    y = []
    sample_weights = []
    sample_weights = []
    sample_weights = []
    sample_weights = []
    
    for data in training_data:
        features = extract_features(data['holds_data'])
        X.append(features)
        y.append(data['difficulty'])
    
    X = np.array(X)
    y = np.array(y)
    sample_weights = np.array(sample_weights) if len(sample_weights) == len(y) else np.ones(len(y))
    sample_weights = np.array(sample_weights) if len(sample_weights) == len(y) else np.ones(len(y))
    sample_weights = np.array(sample_weights) if len(sample_weights) == len(y) else np.ones(len(y))
    sample_weights = np.array(sample_weights) if len(sample_weights) == len(y) else np.ones(len(y))
    
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # 모델 학습 (Gradient Boosting - 더 정확)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 정확도 평가
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y_encoded, cv=min(5, len(X)))
    cv_accuracy = np.mean(cv_scores)
    
    print(f"   ✅ 훈련 정확도: {train_accuracy*100:.1f}%")
    print(f"   ✅ 테스트 정확도: {test_accuracy*100:.1f}%")
    print(f"   ✅ CV 정확도: {cv_accuracy*100:.1f}%")
    
    # 모델 저장
    with open(DIFFICULTY_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(DIFFICULTY_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"   💾 모델 저장 완료: {DIFFICULTY_MODEL_PATH}")
    
    return test_accuracy, cv_accuracy

def train_type_model(training_data: List[Dict]) -> Tuple[float, float]:
    """유형 예측 모델 학습"""
    
    print(f"\n🎓 유형 모델 학습 시작...")
    print(f"   훈련 데이터: {len(training_data)}개")
    
    # 특징 추출
    X = []
    y = []
    sample_weights = []
    
    for data in training_data:
        features = extract_features(data['holds_data'])
        X.append(features)
        y.append(data['type'])
    
    X = np.array(X)
    y = np.array(y)
    sample_weights = np.array(sample_weights) if len(sample_weights) == len(y) else np.ones(len(y))
    
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 정확도 평가
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y_encoded, cv=min(5, len(X)))
    cv_accuracy = np.mean(cv_scores)
    
    print(f"   ✅ 훈련 정확도: {train_accuracy*100:.1f}%")
    print(f"   ✅ 테스트 정확도: {test_accuracy*100:.1f}%")
    print(f"   ✅ CV 정확도: {cv_accuracy*100:.1f}%")
    
    # 모델 저장
    with open(TYPE_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(TYPE_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"   💾 모델 저장 완료: {TYPE_MODEL_PATH}")
    
    return test_accuracy, cv_accuracy

def predict_difficulty(holds_data: List[Dict]) -> Dict:
    """학습된 모델로 난이도 예측"""
    
    if not os.path.exists(DIFFICULTY_MODEL_PATH):
        return {'grade': None, 'confidence': 0.0, 'available': False}
    
    try:
        # 모델 로드
        with open(DIFFICULTY_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(DIFFICULTY_ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        
        # 특징 추출
        features = extract_features(holds_data)
        features = features.reshape(1, -1)
        
        # 예측
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        # 경계 prior 보정 (HSV 기반 가벼운 multiplier)
        try:
            hsv = [0, 0, 128]
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
            class_list = list(encoder.classes_)
            mult = {c: 1.0 for c in class_list}
            # white guard
            if v >= 220 and s <= 35:
                mult['white'] = 1.3
                for c in ('mint','lime','yellow','green','blue'):
                    if c in mult: mult[c] *= 0.85
            # black guard (저명도)
            if v <= 120 and s <= 90:
                if 'black' in mult: mult['black'] *= 1.35
                if 'white' in mult: mult['white'] *= 0.80
            # red/pink/purple 경계
            if 166 <= h < 180 and s >= 120:
                if 'red' in mult: mult['red'] *= 1.12
                if 'pink' in mult: mult['pink'] *= 0.95
            # pink 강화 (밝고 고채도, 169~174)
            if 169 <= h < 174 and s >= 150 and v >= 150:
                if 'pink' in mult: mult['pink'] *= 1.20
                if 'red' in mult: mult['red'] *= 0.92
                if 'purple' in mult: mult['purple'] *= 0.96
            # pink vs purple: 155~166, 고명도일수록 pink 가중
            if 155 <= h < 166 and s >= 80 and v >= 170:
                if 'pink' in mult: mult['pink'] *= 1.08
                if 'purple' in mult: mult['purple'] *= 0.96
            # mint/green 경계
            if 70 <= h < 85:
                if s >= 70 and v >= 170:
                    if 'mint' in mult: mult['mint'] *= 1.12
                    if 'green' in mult: mult['green'] *= 0.95
                else:
                    if 'mint' in mult: mult['mint'] *= 1.06
            # orange/red 경계
            if 0 <= h < 12 and s >= 140 and v >= 150:
                if 'orange' in mult: mult['orange'] *= 1.08
            # 적용
            probs = {c: float(p) for c, p in zip(class_list, probabilities)}
            probs = {c: probs[c]*mult.get(c,1.0) for c in probs}
            total = sum(probs.values())
            if total > 0:
                probabilities = np.array([probs[c]/total for c in class_list])
                # 상위1 재계산
                prediction = np.argmax(probabilities)
                color = encoder.inverse_transform([prediction])[0]
                confidence = float(np.max(probabilities))
        except Exception as e:
            pass
        
        grade = encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        
        return {
            'grade': grade,
            'confidence': confidence,
            'available': True
        }
    except Exception as e:
        print(f"⚠️ 난이도 예측 실패: {e}")
        return {'grade': None, 'confidence': 0.0, 'available': False}

def predict_type(holds_data: List[Dict]) -> Dict:
    """학습된 모델로 유형 예측"""
    
    if not os.path.exists(TYPE_MODEL_PATH):
        return {'type': None, 'confidence': 0.0, 'available': False}
    
    try:
        # 모델 로드
        with open(TYPE_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(TYPE_ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        
        # 특징 추출
        features = extract_features(holds_data)
        features = features.reshape(1, -1)
        
        # 예측
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        climb_type = encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        
        return {
            'type': climb_type,
            'confidence': confidence,
            'available': True
        }
    except Exception as e:
        print(f"⚠️ 유형 예측 실패: {e}")
        return {'type': None, 'confidence': 0.0, 'available': False}

def get_model_availability() -> Dict:
    """모델 사용 가능 여부 확인"""
    return {
        'difficulty_model': os.path.exists(DIFFICULTY_MODEL_PATH),
        'type_model': os.path.exists(TYPE_MODEL_PATH),
        'color_model': os.path.exists(COLOR_MODEL_PATH)
    }

# 🎨 ===== 색상 분류 모델 ===== 🎨

def extract_color_features(color_data: Dict) -> np.ndarray:
    """🎨 홀드 색상 데이터로부터 특징 벡터 추출"""
    
    rgb = color_data.get('rgb', [128, 128, 128])
    hsv = color_data.get('hsv', [0, 0, 128])
    lab = color_data.get('lab', [0, 0, 0])
    
    # 기본 색상 특징 (9개)
    features = [
        rgb[0], rgb[1], rgb[2],  # RGB
        hsv[0], hsv[1], hsv[2],  # HSV
        lab[0], lab[1], lab[2],  # LAB
    ]
    
    # 통계 특징 추가 (color_stats에서)
    color_stats = color_data.get('color_stats', {})
    
    # HSV 통계 (리스트 형식)
    hsv_stats = color_stats.get('hsv_stats', [])
    if isinstance(hsv_stats, list) and len(hsv_stats) >= 12:
        # [mean_h, mean_s, mean_v, std_h, std_s, std_v, min_h, min_s, min_v, max_h, max_s, max_v]
        features.extend([
            hsv_stats[0] if hsv_stats[0] else 0,  # H 평균
            hsv_stats[1] if hsv_stats[1] else 0,  # S 평균
            hsv_stats[2] if hsv_stats[2] else 0,  # V 평균
            hsv_stats[4] if len(hsv_stats) > 4 else 0,  # S 표준편차
            hsv_stats[5] if len(hsv_stats) > 5 else 0,  # V 표준편차
        ])
    else:
        features.extend([0, 0, 0, 0, 0])
    
    # RGB 통계 (리스트 형식)
    rgb_stats = color_stats.get('rgb_stats', [])
    if isinstance(rgb_stats, list) and len(rgb_stats) >= 9:
        # [mean_r, mean_g, mean_b, std_r, std_g, std_b, min_r, min_g, min_b, max_r, max_g, max_b]
        features.extend([
            rgb_stats[3] if len(rgb_stats) > 3 else 0,  # R 표준편차
            rgb_stats[4] if len(rgb_stats) > 4 else 0,  # G 표준편차
            rgb_stats[5] if len(rgb_stats) > 5 else 0,  # B 표준편차
        ])
    else:
        features.extend([0, 0, 0])
    
    # LAB 통계 (리스트 형식)
    lab_stats = color_stats.get('lab_stats', [])
    if isinstance(lab_stats, list) and len(lab_stats) >= 6:
        # [mean_l, mean_a, mean_b, std_l, std_a, std_b, min_l, min_a, min_b, max_l, max_a, max_b]
        features.extend([
            lab_stats[1] if len(lab_stats) > 1 else 0,  # a 평균 (빨강-녹색)
            lab_stats[2] if len(lab_stats) > 2 else 0,  # b 평균 (파랑-노랑)
        ])
    else:
        features.extend([0, 0])
    
    # 추가 특징
    features.extend([
        color_data.get('area', 0) / 10000,     # 홀드 크기 (정규화)
        color_data.get('circularity', 0)       # 홀드 원형도
    ])
    
    return np.array(features)

def train_color_model(training_data: List[Dict]) -> Tuple[float, float]:
    """🎨 색상 분류 모델 학습"""
    
    print(f"\n🎨 색상 분류 모델 학습 시작...")
    print(f"   훈련 데이터: {len(training_data)}개")
    
    if len(training_data) < 10:
        print(f"   ⚠️ 데이터 부족! 최소 10개 필요 (현재: {len(training_data)}개)")
        return 0.0, 0.0
    
    # 특징 추출
    X = []
    y = []
    sample_weights = []
    
    for data in training_data:
        try:
            # 디버깅: 데이터 구조 확인
            if not isinstance(data, dict):
                print(f"   ⚠️ 데이터 타입 오류: {type(data)}, 데이터: {data}")
                continue
            
            features = extract_color_features(data)
            X.append(features)
            y.append(data['correct_color'])
            # 경계 가중치(데이터 기반 샘플 가중)
            w = 1.0
            hsv = data.get('hsv') or data.get('dominant_hsv') or data.get('color_stats', {}).get('dominant_hsv')
            if isinstance(hsv, (list, tuple)) and len(hsv) == 3:
                h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
                label = (data['correct_color'] or '').lower()
                # pink 경계 강화 (명도↑, 저채도↑일수록 강화)
                if label == 'pink' and 167 <= h < 176 and v >= 160:
                    w *= 1.28
                    if s <= 140 and v >= 185:
                        w *= 1.10
                # black 경계 강화 (더 보수)
                if label == 'black' and v <= 115 and s <= 85:
                    w *= 1.30
                # mint 경계 강화
                if label == 'mint' and 70 <= h < 85 and s >= 70 and v >= 170:
                    w *= 1.20
                # red 경계 강화
                if label == 'red' and 166 <= h < 180 and s >= 120 and v >= 120:
                    w *= 1.08
                    if 169 <= h < 176 and v < 165 and s >= 170:
                        w *= 1.05
                # white 경계 강화 (매우 밝고 저채도)
                if label == 'white' and v >= 230 and s <= 28:
                    w *= 1.15
                # green 경계 강화 (저채도·고명도 경계에서 green 보강)
                if label == 'green' and h < 90 and s <= 65 and v >= 160:
                    w *= 1.15
            sample_weights.append(w)
        except Exception as e:
            print(f"   ⚠️ 특징 추출 실패: {e}, 데이터: {data}")
            continue
    
    if len(X) < 10:
        print(f"   ⚠️ 유효 데이터 부족! (현재: {len(X)}개)")
        return 0.0, 0.0
    
    X = np.array(X)
    y = np.array(y)
    sample_weights = np.array(sample_weights) if len(sample_weights) == len(y) else np.ones(len(y))
    
    unique_colors, counts = np.unique(y, return_counts=True)
    print(f"   색상 분포: {dict(zip(unique_colors, counts))}")
    
    # 샘플이 1개뿐인 클래스 필터링
    min_samples_per_class = 2
    classes_to_keep = unique_colors[counts >= min_samples_per_class]
    
    if len(classes_to_keep) < 2:
        print(f"   ⚠️ 2개 이상의 샘플을 가진 색상 클래스가 부족합니다!")
        print(f"   각 색상마다 최소 {min_samples_per_class}개의 피드백이 필요합니다.")
        return 0.0, 0.0
    
    # 필터링된 데이터만 사용
    mask = np.isin(y, classes_to_keep)
    X = X[mask]
    y = y[mask]
    sample_weights = sample_weights[mask]
    
    print(f"   ✅ 학습에 사용할 데이터: {len(X)}개 (필터링 후)")
    
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # stratify 가능 여부 체크
    unique_encoded, counts_encoded = np.unique(y_encoded, return_counts=True)
    can_stratify = np.all(counts_encoded >= 2)
    
    # 학습/테스트 분할
    if can_stratify:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y_encoded, sample_weights, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y_encoded, sample_weights, test_size=0.2, random_state=42
        )
    
    # 베이스 모델 (Random Forest)
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # 불균형 데이터 처리
    )
    
    # 확률 캘리브레이션 (Isotonic; 데이터 충분 시 더 정확)
    try:
        model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
        model.fit(X_train, y_train, sample_weight=w_train)
    except Exception as e:
        print(f"   ⚠️ Isotonic 캘리브레이션 실패: {e} → sigmoid로 폴백")
        model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
        model.fit(X_train, y_train, sample_weight=w_train)
    
    # 정확도 평가
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    # Cross-validation (각 클래스에 충분한 샘플이 있을 때만)
    cv_folds = min(3, len(X)//10)  # 더 보수적으로 설정
    if cv_folds >= 2 and np.all(counts_encoded >= cv_folds):
        try:
            cv_scores = cross_val_score(model, X, y_encoded, cv=cv_folds)
            cv_accuracy = np.mean(cv_scores)
        except Exception as e:
            print(f"   ⚠️ Cross-validation 실패: {e}")
            cv_accuracy = test_accuracy  # CV 실패 시 test_accuracy 사용
    else:
        print(f"   ⚠️ Cross-validation 스킵 (데이터 부족)")
        cv_accuracy = test_accuracy
    
    print(f"   ✅ 훈련 정확도: {train_accuracy*100:.1f}%")
    print(f"   ✅ 테스트 정확도: {test_accuracy*100:.1f}%")
    print(f"   ✅ CV 정확도: {cv_accuracy*100:.1f}%")
    
    # Feature Importance (가능한 경우에만)
    try:
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
        elif hasattr(model, 'base_estimator_') and hasattr(model.base_estimator_, 'feature_importances_'):
            fi = model.base_estimator_.feature_importances_
        else:
            fi = None
        if fi is not None:
            top_features = np.argsort(fi)[::-1][:5]
            print(f"   🔝 중요 특징 (인덱스): {top_features}")
    except Exception:
        pass
    
    # 모델 저장
    with open(COLOR_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(COLOR_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"   💾 모델 저장 완료: {COLOR_MODEL_PATH}")
    
    return test_accuracy, cv_accuracy

def predict_color(hold_features: Dict) -> Dict:
    """🎨 학습된 모델로 홀드 색상 예측"""
    
    if not os.path.exists(COLOR_MODEL_PATH):
        return {'color': None, 'confidence': 0.0, 'available': False}
    
    try:
        # 모델 로드
        with open(COLOR_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(COLOR_ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        
        # 특징 추출
        features = extract_color_features(hold_features)
        features = features.reshape(1, -1)
        
        # 예측
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # HSV 기반 가벼운 prior 보정 (pink↔red, green↔mint, red↔orange, black↔white)
        try:
            hsv = hold_features.get('hsv') or hold_features.get('dominant_hsv') or hold_features.get('color_stats', {}).get('dominant_hsv')
            if isinstance(hsv, (list, tuple)) and len(hsv) == 3:
                h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
                class_list = list(encoder.classes_)
                mult = {c: 1.0 for c in class_list}
                # black guard: 저명도 저채도는 black 보수 강화 + 주변 감쇠
                if v <= 115 and s <= 85 and 'black' in mult:
                    mult['black'] *= 1.22
                    for c in ('white','blue','green','mint','lime'):
                        if c in mult: mult[c] *= 0.92
                if v <= 95 and 'black' in mult:
                    mult['black'] *= 1.05
                    if 'white' in mult: mult['white'] *= 0.90
                # pink↔red 집중 보수화: hue 165~178 구간 세분화
                if 165 <= h < 178:
                    # 밝고 저채도는 pink 강화 (강화)
                    if v >= 185 and s <= 140 and 'pink' in mult:
                        mult['pink'] *= 1.12
                        if 'red' in mult: mult['red'] *= 0.92
                    # 추가: V 175~195에서 S 기준 완화로 pink 영역 확대
                    if 175 <= v < 195 and s <= 155 and 'pink' in mult:
                        mult['pink'] *= 1.04
                        if 'red' in mult: mult['red'] *= 0.97
                    # 매우 밝고 매우 저채도는 pink를 더 강하게, red 더 감쇠 (강화)
                    if v >= 190 and s <= 135 and 'pink' in mult:
                        mult['pink'] *= 1.08
                        if 'red' in mult: mult['red'] *= 0.92
                    # 어두운 고채도는 red 강화 (강화)
                    if v <= 155 and s >= 180 and 'red' in mult:
                        mult['red'] *= 1.10
                        if 'pink' in mult: mult['pink'] *= 0.94
                    # 추가: 더 어두운·더 고채도에서 red 우선 강화
                    if v <= 150 and s >= 190 and 'red' in mult:
                        mult['red'] *= 1.12
                        if 'pink' in mult: mult['pink'] *= 0.93
                    # 중간 영역은 과증폭 방지로 red 약가중 (강화)
                    if 155 < v < 185 and 150 <= s < 180 and 'red' in mult:
                        mult['red'] *= 1.05
                    # LAB 힌트가 있으면 약하게 반영 (a* 양수 크면 red 쪽)
                    lab = hold_features.get('lab') or hold_features.get('color_stats', {}).get('lab_stats')
                    if isinstance(lab, (list, tuple)) and len(lab) >= 2:
                        a_val = lab[1] if len(lab) == 3 else lab[1]
                        if isinstance(a_val, (int, float)):
                            if a_val >= 32 and v < 190 and 'red' in mult:
                                mult['red'] *= 1.05
                            if a_val <= 18 and v >= 185 and 'pink' in mult:
                                mult['pink'] *= 1.04
                # green↔mint: h<90 저채도·고명도는 green 보수 강화, mint 감쇠
                if h < 90 and 60 <= s <= 85 and 160 <= v <= 190:
                    if 'green' in mult: mult['green'] *= 1.14
                    if 'mint' in mult: mult['mint'] *= 0.94
                if h < 90 and s < 60 and v >= 160:
                    if 'green' in mult: mult['green'] *= 1.16
                    if 'mint' in mult: mult['mint'] *= 0.94
                # mint 보강: 88~96, 중간 명도/중고채도 민트가 unknown으로 빠지는 케이스 보강
                if 88 <= h <= 96 and 80 <= s <= 150 and 105 <= v <= 175:
                    if 'mint' in mult: mult['mint'] *= 1.15
                    if 'green' in mult: mult['green'] *= 0.95
                # green 보강: 80~86, 고채도이면서 저중명도 구간은 green 우선
                if 80 <= h < 86 and s >= 95 and v <= 130:
                    if 'green' in mult: mult['green'] *= 1.10
                    if 'mint' in mult: mult['mint'] *= 0.95
                # blue 가드: 매우 높은 채도 + 저명도(딥 틸)일 때 mint로 치우치는 문제 방지
                if 95 <= h <= 110 and s >= 220 and v <= 90:
                    if 'blue' in mult: mult['blue'] *= 1.22
                    if 'mint' in mult: mult['mint'] *= 0.85
                    if 'green' in mult: mult['green'] *= 0.92
                # red↔orange: 극저h, 고채도·고명도는 오렌지 보강
                if 0 <= h < 9 and s >= 160 and v >= 150:
                    if 'orange' in mult: mult['orange'] *= 1.15
                    if 'red' in mult: mult['red'] *= 0.94
                # h<5의 극저h에서 추가 억제/보강
                if 0 <= h < 5 and s >= 180 and v >= 160:
                    if 'orange' in mult: mult['orange'] *= 1.03
                    if 'red' in mult: mult['red'] *= 0.98
                # RGB 기반 red↔orange 타이브레이크: R 매우 높고 G/B 중간이면 orange 쪽 가중
                try:
                    rgb = hold_features.get('rgb') or hold_features.get('dominant_rgb')
                    if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
                        if r >= 230 and 60 <= g <= 140 and 60 <= b <= 140:
                            gb_ratio = g / max(1, b)
                            if gb_ratio >= 0.8 and (g + b) >= 130:
                                if 'orange' in mult: mult['orange'] *= 1.12
                                if 'red' in mult: mult['red'] *= 0.95
                except Exception:
                    pass
                # purple 보호: 어두운 고채도 보라가 black으로 끌리지 않도록 소폭 보강
                if 150 <= h <= 170 and s >= 110 and v < 100:
                    if 'purple' in mult: mult['purple'] *= 1.06
                    if 'black' in mult: mult['black'] *= 0.97
                # white guard: 매우 밝고 저채도는 white 보강, 주변색 감쇠
                if v >= 230 and s <= 28 and 'white' in mult:
                    mult['white'] *= 1.12
                    for c in ('mint','lime','yellow','green','blue'):
                        if c in mult: mult[c] *= 0.95
                # white vs yellow(베이지 근접) 감쇠: h 18~26, v 200~235, s 28~40이면 yellow 감쇠
                if 18 <= h <= 26 and 200 <= v <= 235 and 28 <= s <= 40:
                    if 'yellow' in mult: mult['yellow'] *= 0.90
                # 적용
                probs = {c: float(p) for c, p in zip(class_list, probabilities)}
                probs = {c: probs[c]*mult.get(c, 1.0) for c in probs}
                total = sum(probs.values())
                if total > 0:
                    probabilities = np.array([probs[c]/total for c in class_list])
                    prediction = np.argmax(probabilities)
        except Exception:
            pass
        
        color = encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        
        # 상위 3개 예측
        top_3_idx = np.argsort(probabilities)[::-1][:3]
        top_3_colors = encoder.inverse_transform(top_3_idx)
        top_3_probs = probabilities[top_3_idx]
        
        return {
            'color': color,
            'confidence': confidence,
            'available': True,
            'top_3': list(zip(top_3_colors, top_3_probs.tolist()))
        }
    except Exception as e:
        print(f"⚠️ 색상 예측 실패: {e}")
        return {'color': None, 'confidence': 0.0, 'available': False}

