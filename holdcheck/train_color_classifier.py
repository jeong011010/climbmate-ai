"""
🤖 경량 색상 분류 AI 모델 학습
CLIP보다 훨씬 빠르고 가벼운 전용 모델 (~1-5MB)

3가지 모델 제공:
1. KNN (가장 빠름, 실시간)
2. SVM (정확도 중간)
3. 작은 신경망 (가장 정확, 약간 느림)
"""

import json
import numpy as np
import pickle
from pathlib import Path


def load_feedback_dataset(dataset_path="holdcheck/color_feedback_dataset.json"):
    """피드백 데이터셋 로드"""
    if not Path(dataset_path).exists():
        print(f"❌ 데이터셋 없음: {dataset_path}")
        print("   먼저 피드백 UI에서 데이터를 수집하세요!")
        return None
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ 데이터셋 로드: {dataset['total_samples']}개 샘플")
    return dataset


def prepare_training_data(dataset):
    """학습 데이터 준비"""
    X = []  # 특징 (RGB, HSV, 통계 등)
    y = []  # 라벨 (색상 이름)
    
    for sample in dataset['samples']:
        # 특징 추출: RGB + HSV
        rgb = sample.get('rgb', [0, 0, 0])
        hsv = sample.get('hsv', [0, 0, 0])
        
        # 추가 특징 계산
        r, g, b = rgb
        h, s, v = hsv
        
        # 색상 특징 벡터 (12차원)
        features = [
            r, g, b,  # RGB (3)
            h, s, v,  # HSV (3)
            r / 255.0, g / 255.0, b / 255.0,  # 정규화 RGB (3)
            s / 255.0, v / 255.0,  # 정규화 SV (2)
            max(r, g, b) - min(r, g, b),  # 색상 범위 (1)
        ]
        
        X.append(features)
        y.append(sample['correct_color'])
    
    return np.array(X), np.array(y)


def train_knn_model(X, y, n_neighbors=5):
    """KNN 모델 학습 (가장 빠름)"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    
    print(f"\n🚀 KNN 모델 학습 중... (k={n_neighbors})")
    
    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KNN 학습
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    model.fit(X_scaled, y)
    
    print(f"✅ KNN 학습 완료!")
    
    return model, scaler


def train_svm_model(X, y):
    """SVM 모델 학습 (정확도 중간)"""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    print(f"\n🚀 SVM 모델 학습 중...")
    
    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SVM 학습
    model = SVC(kernel='rbf', probability=True, gamma='auto')
    model.fit(X_scaled, y)
    
    print(f"✅ SVM 학습 완료!")
    
    return model, scaler


def train_neural_network_model(X, y, epochs=100):
    """작은 신경망 학습 (가장 정확)"""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow import keras
    
    print(f"\n🚀 신경망 모델 학습 중... (epochs={epochs})")
    
    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    
    # 신경망 구조 (매우 작음, ~100KB)
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(np.unique(y)), activation='softmax')
    ])
    
    # 컴파일
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 학습
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0
    )
    
    # 최종 정확도
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"✅ 신경망 학습 완료! (검증 정확도: {val_accuracy:.1%})")
    
    return model, scaler, label_encoder


def save_model(model, scaler, model_type, label_encoder=None, 
               output_dir="holdcheck/models"):
    """모델 저장"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == "neural_network":
        # TensorFlow 모델 저장
        model.save(f"{output_dir}/color_classifier_nn.h5")
        
        # 스케일러와 라벨 인코더 저장
        with open(f"{output_dir}/color_classifier_nn_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        with open(f"{output_dir}/color_classifier_nn_labels.pkl", 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"💾 신경망 모델 저장: {output_dir}/color_classifier_nn.h5")
    
    else:
        # Scikit-learn 모델 저장
        model_data = {
            'model': model,
            'scaler': scaler,
            'model_type': model_type
        }
        
        output_path = f"{output_dir}/color_classifier_{model_type}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 {model_type.upper()} 모델 저장: {output_path}")


def load_trained_model(model_type="knn", model_dir="holdcheck/models"):
    """학습된 모델 로드"""
    if model_type == "neural_network":
        import tensorflow as tf
        from tensorflow import keras
        
        model = keras.models.load_model(f"{model_dir}/color_classifier_nn.h5")
        
        with open(f"{model_dir}/color_classifier_nn_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open(f"{model_dir}/color_classifier_nn_labels.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder
    
    else:
        model_path = f"{model_dir}/color_classifier_{model_type}.pkl"
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data['model'], model_data['scaler'], None


def predict_color(model, scaler, rgb, hsv, label_encoder=None, model_type="knn"):
    """색상 예측"""
    r, g, b = rgb
    h, s, v = hsv
    
    # 특징 추출
    features = np.array([[
        r, g, b,
        h, s, v,
        r / 255.0, g / 255.0, b / 255.0,
        s / 255.0, v / 255.0,
        max(r, g, b) - min(r, g, b),
    ]])
    
    # 정규화
    features_scaled = scaler.transform(features)
    
    # 예측
    if model_type == "neural_network":
        probs = model.predict(features_scaled, verbose=0)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        color_name = label_encoder.inverse_transform([pred_idx])[0]
    else:
        color_name = model.predict(features_scaled)[0]
        
        # 신뢰도 계산 (확률 지원하는 경우)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probs))
        else:
            confidence = 0.85  # 기본값
    
    return color_name, confidence


def ml_based_color_clustering(hold_data, vectors, model_type="knn", 
                              model_dir="holdcheck/models"):
    """
    🤖 ML 모델 기반 색상 클러스터링
    
    룰 기반보다 정확하고, CLIP보다 빠름!
    
    Args:
        hold_data: 홀드 데이터
        vectors: 사용 안 함
        model_type: "knn", "svm", "neural_network"
        model_dir: 모델 저장 디렉토리
    
    Returns:
        hold_data: 그룹 정보가 추가된 홀드 데이터
    """
    import time
    import cv2
    
    if len(hold_data) == 0:
        return hold_data
    
    start_time = time.time()
    
    print(f"\n🤖 ML 모델 기반 색상 클러스터링 시작")
    print(f"   모델: {model_type.upper()}")
    print(f"   홀드 개수: {len(hold_data)}개")
    
    # 모델 로드
    try:
        model, scaler, label_encoder = load_trained_model(model_type, model_dir)
    except FileNotFoundError:
        print(f"❌ 모델 없음: {model_dir}/color_classifier_{model_type}")
        print("   먼저 train_color_classifier.py를 실행하여 모델을 학습하세요!")
        return hold_data
    
    # 각 홀드 분류
    color_groups = {}
    
    for hold_idx, hold in enumerate(hold_data):
        # RGB/HSV 가져오기
        if "dominant_hsv" in hold:
            h, s, v = hold["dominant_hsv"]
        else:
            h, s, v = 0, 0, 128
        
        if "dominant_rgb" in hold:
            rgb = hold["dominant_rgb"]
        else:
            hsv_arr = np.uint8([[[h, s, v]]])
            rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
            rgb = rgb_arr.tolist()
        
        # 예측
        color_name, confidence = predict_color(
            model, scaler, rgb, [h, s, v], 
            label_encoder, model_type
        )
        
        # 홀드에 정보 추가
        hold["clip_color_name"] = color_name
        hold["clip_confidence"] = confidence
        hold["color_method"] = f"ml_{model_type}"
        
        # 그룹핑
        if color_name not in color_groups:
            color_groups[color_name] = []
        color_groups[color_name].append(hold)
    
    # 그룹 ID 할당
    color_order = ["black", "white", "gray", "red", "orange", "yellow", 
                   "green", "mint", "blue", "purple", "pink", "brown", "unknown"]
    
    group_idx = 0
    for color_name in color_order:
        if color_name in color_groups:
            for hold in color_groups[color_name]:
                hold["group"] = f"g{group_idx}"
            group_idx += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ ML 클러스터링 완료 (⚡ {elapsed:.2f}초)")
    print(f"   생성된 그룹 수: {len(color_groups)}개")
    for color_name in color_order:
        if color_name in color_groups:
            count = len(color_groups[color_name])
            avg_conf = np.mean([h["clip_confidence"] for h in color_groups[color_name]])
            print(f"   {color_name}: {count}개 홀드 (평균 신뢰도: {avg_conf:.2f})")
    
    return hold_data


def train_all_models():
    """모든 모델 학습 (KNN, SVM, NN)"""
    print("=" * 60)
    print("🤖 경량 색상 분류 AI 모델 학습")
    print("=" * 60)
    
    # 데이터셋 로드
    dataset = load_feedback_dataset()
    if dataset is None:
        return
    
    if dataset['total_samples'] < 10:
        print(f"⚠️ 데이터가 너무 적습니다 ({dataset['total_samples']}개)")
        print("   최소 30개 이상의 피드백을 수집하세요!")
        return
    
    # 데이터 준비
    X, y = prepare_training_data(dataset)
    print(f"\n📊 학습 데이터: {len(X)}개 샘플, {len(np.unique(y))}개 클래스")
    print(f"   클래스: {np.unique(y)}")
    
    # 1. KNN 학습
    try:
        knn_model, knn_scaler = train_knn_model(X, y)
        save_model(knn_model, knn_scaler, "knn")
    except Exception as e:
        print(f"❌ KNN 학습 실패: {e}")
    
    # 2. SVM 학습
    try:
        svm_model, svm_scaler = train_svm_model(X, y)
        save_model(svm_model, svm_scaler, "svm")
    except Exception as e:
        print(f"❌ SVM 학습 실패: {e}")
    
    # 3. 신경망 학습 (TensorFlow 있으면)
    try:
        nn_model, nn_scaler, nn_labels = train_neural_network_model(X, y)
        save_model(nn_model, nn_scaler, "neural_network", nn_labels)
    except ImportError:
        print("⚠️ TensorFlow 미설치 - 신경망 학습 건너뜀")
    except Exception as e:
        print(f"❌ 신경망 학습 실패: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 모든 모델 학습 완료!")
    print("=" * 60)
    print("\n💡 사용 방법:")
    print("   from train_color_classifier import ml_based_color_clustering")
    print("   hold_data = ml_based_color_clustering(hold_data, None, model_type='knn')")


def evaluate_models():
    """학습된 모델 성능 평가"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("\n📊 모델 성능 평가")
    print("=" * 60)
    
    # 데이터셋 로드
    dataset = load_feedback_dataset()
    if dataset is None:
        return
    
    X, y = prepare_training_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 각 모델 평가
    for model_type in ["knn", "svm"]:
        try:
            model, scaler, _ = load_trained_model(model_type)
            
            # 예측
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            # 정확도
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{model_type.upper()} 모델:")
            print(f"  정확도: {accuracy:.1%}")
            print("\n" + classification_report(y_test, y_pred))
        
        except Exception as e:
            print(f"❌ {model_type.upper()} 평가 실패: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate_models()
    else:
        train_all_models()

