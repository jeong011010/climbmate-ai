"""
자체 ML 모델 학습 파이프라인
검증된 데이터 50개 이상부터 학습 가능
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    
    for data in training_data:
        features = extract_features(data['holds_data'])
        X.append(features)
        y.append(data['difficulty'])
    
    X = np.array(X)
    y = np.array(y)
    
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
    
    for data in training_data:
        features = extract_features(data['holds_data'])
        X.append(features)
        y.append(data['type'])
    
    X = np.array(X)
    y = np.array(y)
    
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
    
    for data in training_data:
        try:
            # 디버깅: 데이터 구조 확인
            if not isinstance(data, dict):
                print(f"   ⚠️ 데이터 타입 오류: {type(data)}, 데이터: {data}")
                continue
            
            features = extract_color_features(data)
            X.append(features)
            y.append(data['correct_color'])
        except Exception as e:
            print(f"   ⚠️ 특징 추출 실패: {e}, 데이터: {data}")
            continue
    
    if len(X) < 10:
        print(f"   ⚠️ 유효 데이터 부족! (현재: {len(X)}개)")
        return 0.0, 0.0
    
    X = np.array(X)
    y = np.array(y)
    
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
    
    print(f"   ✅ 학습에 사용할 데이터: {len(X)}개 (필터링 후)")
    
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # stratify 가능 여부 체크
    unique_encoded, counts_encoded = np.unique(y_encoded, return_counts=True)
    can_stratify = np.all(counts_encoded >= 2)
    
    # 학습/테스트 분할
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    # 모델 학습 (Random Forest - 색상 분류에 적합)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # 불균형 데이터 처리
    )
    
    model.fit(X_train, y_train)
    
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
    
    # Feature Importance
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[::-1][:5]
    print(f"   🔝 중요 특징 (인덱스): {top_features}")
    
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

