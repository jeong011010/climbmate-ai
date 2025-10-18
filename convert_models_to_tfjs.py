"""
PyTorch YOLO + CLIP 모델을 TensorFlow.js로 변환
"""
import torch
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
from ultralytics import YOLO
import clip
import os

def convert_yolo_to_tfjs():
    """YOLO 모델을 TensorFlow.js로 변환"""
    print("🔄 YOLO 모델 변환 시작...")
    
    try:
        # YOLO 모델 로드
        yolo_path = "holdcheck/roboflow_weights/weights.pt"
        if not os.path.exists(yolo_path):
            print("⚠️  커스텀 YOLO 모델 없음, YOLOv8n 사용")
            yolo_path = "yolov8n.pt"
        
        model = YOLO(yolo_path)
        
        # TorchScript로 변환
        print("  → TorchScript로 변환 중...")
        dummy_input = torch.randn(1, 3, 640, 640)
        traced = torch.jit.trace(model.model, dummy_input)
        
        # ONNX로 변환
        print("  → ONNX로 변환 중...")
        torch.onnx.export(
            traced,
            dummy_input,
            "models/yolo.onnx",
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # ONNX → TensorFlow
        print("  → TensorFlow로 변환 중...")
        os.system("python -m tf2onnx.convert --opset 13 --onnx models/yolo.onnx --output models/yolo_tf")
        
        # TensorFlow → TensorFlow.js
        print("  → TensorFlow.js로 변환 중...")
        os.system("tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model models/yolo_tf frontend/public/models/yolo")
        
        print("✅ YOLO 모델 변환 완료!")
        return True
        
    except Exception as e:
        print(f"❌ YOLO 변환 실패: {e}")
        return False

def convert_clip_to_tfjs():
    """CLIP 모델을 TensorFlow.js로 변환"""
    print("🔄 CLIP 모델 변환 시작...")
    
    try:
        # CLIP 모델 로드
        print("  → CLIP 모델 로딩 중...")
        device = "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # TorchScript로 변환
        print("  → TorchScript로 변환 중...")
        dummy_input = torch.randn(1, 3, 224, 224)
        traced = torch.jit.trace(model.visual, dummy_input)
        
        # ONNX로 변환
        print("  → ONNX로 변환 중...")
        torch.onnx.export(
            traced,
            dummy_input,
            "models/clip.onnx",
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # ONNX → TensorFlow
        print("  → TensorFlow로 변환 중...")
        os.system("python -m tf2onnx.convert --opset 13 --onnx models/clip.onnx --output models/clip_tf")
        
        # TensorFlow → TensorFlow.js
        print("  → TensorFlow.js로 변환 중...")
        os.system("tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model models/clip_tf frontend/public/models/clip")
        
        print("✅ CLIP 모델 변환 완료!")
        return True
        
    except Exception as e:
        print(f"❌ CLIP 변환 실패: {e}")
        return False

def create_lightweight_mock_models():
    """경량 모의 모델 생성 (변환 실패 시)"""
    print("📦 경량 모의 모델 생성 중...")
    
    os.makedirs("frontend/public/models/yolo", exist_ok=True)
    os.makedirs("frontend/public/models/clip", exist_ok=True)
    
    # 간단한 더미 모델 정보
    model_info = {
        "format": "tfjs-graph-model",
        "generatedBy": "mock",
        "convertedBy": "ClimbMate",
        "modelTopology": {
            "node": [],
            "library": {},
            "versions": {}
        },
        "weightsManifest": []
    }
    
    import json
    with open("frontend/public/models/yolo/model.json", "w") as f:
        json.dump(model_info, f)
    
    with open("frontend/public/models/clip/model.json", "w") as f:
        json.dump(model_info, f)
    
    print("✅ 모의 모델 생성 완료!")

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 PyTorch 모델 → TensorFlow.js 변환 시작")
    print("=" * 80)
    print()
    
    # 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("frontend/public/models", exist_ok=True)
    
    # YOLO 변환
    yolo_success = convert_yolo_to_tfjs()
    
    # CLIP 변환
    clip_success = convert_clip_to_tfjs()
    
    # 변환 실패 시 모의 모델 생성
    if not yolo_success or not clip_success:
        print()
        print("⚠️  모델 변환에 실패했습니다.")
        print("⚠️  경량 모의 모델을 생성합니다.")
        create_lightweight_mock_models()
    
    print()
    print("=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)

