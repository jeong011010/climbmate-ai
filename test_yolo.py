#!/usr/bin/env python3
"""
YOLO 모델 로컬 테스트 스크립트
"""
import cv2
import sys
from holdcheck.preprocess import preprocess

def test_yolo(image_path, conf=0.25):
    """YOLO 홀드 감지 테스트"""
    print(f"🔍 이미지 분석 시작: {image_path}")
    print(f"📊 Confidence Threshold: {conf}")
    print(f"🎯 IoU Threshold: 0.5")
    print("=" * 60)
    
    try:
        # YOLO 전처리 실행
        hold_data, masks = preprocess(
            image_path,
            model_path="/Users/kimjazz/Desktop/project/climbmate/holdcheck/roboflow_weights/weights.pt",
            conf=conf,
            use_clip_ai=False
        )
        
        print(f"\n✅ 감지 완료!")
        print(f"📍 감지된 홀드 수: {len(hold_data)}개")
        print("=" * 60)
        
        # 감지된 홀드 상세 정보
        for i, hold in enumerate(hold_data):
            print(f"\n홀드 #{i+1}:")
            print(f"  - 중심: {hold['center']}")
            print(f"  - 면적: {hold['area']:.1f}")
            print(f"  - 원형도: {hold['circularity']:.2f}")
            print(f"  - RGB: {hold.get('dominant_rgb', 'N/A')}")
            print(f"  - HSV: {hold.get('dominant_hsv', 'N/A')}")
        
        print("\n" + "=" * 60)
        print(f"🖼️  결과 이미지: outputs/{image_path.split('/')[-1].replace('.', '_')}_preprocessed.png")
        
        return hold_data, masks
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_yolo.py <이미지_경로> [confidence_threshold]")
        print("예시: python test_yolo.py test_image.jpg 0.25")
        sys.exit(1)
    
    image_path = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    
    test_yolo(image_path, conf)



