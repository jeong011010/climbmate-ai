"""
PyTorch YOLO + CLIP 모델을 ONNX로 변환
ONNX Runtime Web으로 브라우저에서 실행
"""
import os
import sys
import torch

def export_yolo_to_onnx():
    """YOLO 모델을 ONNX로 내보내기"""
    print("=" * 80)
    print("🔄 YOLO 모델 → ONNX 변환")
    print("=" * 80)
    
    try:
        from ultralytics import YOLO
        
        # 커스텀 YOLO 모델 로드
        yolo_path = "holdcheck/roboflow_weights/weights.pt"
        if not os.path.exists(yolo_path):
            print(f"⚠️  커스텀 모델 없음: {yolo_path}")
            yolo_path = "yolov8n.pt"
            print(f"📦 YOLOv8n 다운로드 및 사용")
        else:
            print(f"📂 커스텀 모델 사용: {yolo_path}")
        
        model = YOLO(yolo_path)
        
        # ONNX로 내보내기
        print(f"🔄 ONNX 변환 중... (img_size=640)")
        output_dir = "frontend/public/models"
        os.makedirs(output_dir, exist_ok=True)
        
        # export() 메서드 사용
        model.export(
            format="onnx",
            imgsz=640,
            simplify=True,
            dynamic=True
        )
        
        # 생성된 ONNX 파일 찾기
        onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
        if onnx_files:
            src_file = onnx_files[0]
            dst_file = os.path.join(output_dir, "yolo.onnx")
            
            if os.path.exists(src_file):
                os.rename(src_file, dst_file)
                size = os.path.getsize(dst_file) / (1024 * 1024)
                print(f"✅ YOLO ONNX 변환 완료!")
                print(f"📦 파일: {dst_file} ({size:.1f}MB)")
                return True
        
        print(f"❌ ONNX 파일 생성 실패")
        return False
            
    except Exception as e:
        print(f"❌ YOLO 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def export_clip_to_onnx():
    """CLIP 모델을 ONNX로 내보내기"""
    print("\n" + "=" * 80)
    print("🔄 CLIP 모델 → ONNX 변환")
    print("=" * 80)
    
    try:
        import clip
        
        # CLIP 모델 로드
        print("📦 CLIP 모델 로드 중... (ViT-B/32)")
        device = "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        
        # Visual encoder만 내보내기
        print("🔄 Visual Encoder ONNX 변환 중...")
        
        dummy_input = torch.randn(1, 3, 224, 224)
        output_path = "frontend/public/models/clip.onnx"
        
        torch.onnx.export(
            model.visual,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True
        )
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ CLIP ONNX 변환 완료!")
            print(f"📦 파일: {output_path} ({size:.1f}MB)")
            return True
        else:
            print(f"❌ ONNX 파일 생성 실패")
            return False
            
    except Exception as e:
        print(f"❌ CLIP 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_model_info():
    """모델 정보 JSON 생성"""
    print("\n" + "=" * 80)
    print("📝 모델 메타데이터 생성")
    print("=" * 80)
    
    import json
    
    model_info = {
        "yolo": {
            "format": "onnx",
            "path": "/models/yolo.onnx",
            "input_shape": [1, 3, 640, 640],
            "description": "커스텀 YOLO 홀드 세그멘테이션 모델",
            "runtime": "onnxruntime-web"
        },
        "clip": {
            "format": "onnx",
            "path": "/models/clip.onnx",
            "input_shape": [1, 3, 224, 224],
            "model": "ViT-B/32",
            "description": "CLIP 색상 분석 모델 (Visual Encoder)",
            "runtime": "onnxruntime-web"
        },
        "usage": {
            "library": "onnxruntime-web",
            "install": "npm install onnxruntime-web",
            "note": "ONNX Runtime Web을 사용하여 브라우저에서 실행"
        }
    }
    
    info_path = "frontend/public/models/model_info.json"
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 모델 정보 생성: {info_path}")
    return True

if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("🚀 ClimbMate AI 모델 ONNX 변환")
    print("=" * 80)
    print("\n")
    
    # YOLO 변환
    yolo_success = export_yolo_to_onnx()
    
    # CLIP 변환
    clip_success = export_clip_to_onnx()
    
    # 모델 정보 생성
    info_success = create_model_info()
    
    # 결과 요약
    print("\n")
    print("=" * 80)
    print("📊 변환 결과 요약")
    print("=" * 80)
    print(f"  YOLO: {'✅ 성공' if yolo_success else '❌ 실패'}")
    print(f"  CLIP: {'✅ 성공' if clip_success else '❌ 실패'}")
    print(f"  Info: {'✅ 성공' if info_success else '❌ 실패'}")
    print("=" * 80)
    
    if yolo_success and clip_success:
        print("\n🎉 모든 모델 변환 완료!")
        print("\n📝 다음 단계:")
        print("  1. npm install onnxruntime-web")
        print("  2. clientAI.js에서 ONNX Runtime 사용")
        print("  3. frontend 재빌드 및 배포")
        print("\n")
    else:
        print("\n⚠️  변환 실패 - 수동으로 확인 필요")
        print("\n")
    
    sys.exit(0 if (yolo_success and clip_success) else 1)

