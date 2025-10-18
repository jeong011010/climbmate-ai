import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
import torch
import clip
from PIL import Image
import psutil
import gc

# 🚀 메모리 최적화: 스레드 수 제한 (메모리 절약)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except:
    pass

def get_memory_usage():
    """📊 현재 메모리 사용량 반환 (MB 단위)"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # 실제 메모리 사용량 (MB)
        'vms': memory_info.vms / 1024 / 1024,  # 가상 메모리 사용량 (MB)
        'percent': process.memory_percent(),    # 시스템 메모리 대비 비율
        'available': psutil.virtual_memory().available / 1024 / 1024  # 사용 가능한 메모리 (MB)
    }

def log_memory_usage(stage_name):
    """📊 메모리 사용량 로그 출력"""
    memory = get_memory_usage()
    
    # 메모리 사용률에 따른 경고
    if memory['percent'] > 90:
        print(f"🚨 [CRITICAL] [{stage_name}] 메모리 사용량이 90%를 초과했습니다!")
        print(f"   🔴 실제 메모리: {memory['rss']:.1f}MB ({memory['percent']:.1f}%)")
        print(f"   ⚠️  OOM 위험! 컨테이너가 종료될 수 있습니다!")
    elif memory['percent'] > 80:
        print(f"⚠️  [WARNING] [{stage_name}] 메모리 사용량이 80%를 초과했습니다!")
        print(f"   🟡 실제 메모리: {memory['rss']:.1f}MB ({memory['percent']:.1f}%)")
    else:
        print(f"📊 [{stage_name}] 메모리 사용량:")
        print(f"   🔸 실제 메모리: {memory['rss']:.1f}MB ({memory['percent']:.1f}%)")
    
    print(f"   🔸 가상 메모리: {memory['vms']:.1f}MB")
    print(f"   🔸 사용 가능: {memory['available']:.1f}MB")
    return memory

def convert_to_json_safe(data):
    """🚀 JSON 직렬화 가능하도록 데이터 변환"""
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_json_safe(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_safe(item) for item in data]
    elif isinstance(data, tuple):
        return [convert_to_json_safe(item) for item in data]
    else:
        return data
from sklearn.cluster import KMeans

# -------------------------------
# 🚀 모델 싱글톤 (캐싱) - 성능 최적화
# -------------------------------
_clip_model = None
_clip_preprocess = None
_clip_device = None
_yolo_model = None
_yolo_model_path = None

def get_yolo_model(model_path="/app/holdcheck/roboflow_weights/weights.pt"):
    """🚀 YOLO 모델을 싱글톤으로 로드 (메모리 절약 + 속도 향상)"""
    global _yolo_model, _yolo_model_path
    
    if _yolo_model is None or _yolo_model_path != model_path:
        print(f"🔍 YOLO 모델 로딩 중... ({model_path})")
        
        # 메모리 사용량 측정 (로딩 전)
        memory_before = log_memory_usage("YOLO 로딩 전")
        
        # 경량 YOLO 모델 사용 (nano 버전)
        if not os.path.exists(model_path):
            print(f"⚠️ 커스텀 모델 없음: {model_path}")
            print("📦 경량 YOLOv8n 모델 사용")
            _yolo_model = YOLO('yolov8n.pt')  # nano 버전 (6MB vs 50MB)
        else:
            print(f"📦 커스텀 모델 사용: {model_path}")
            _yolo_model = YOLO(model_path)
        _yolo_model_path = model_path
        
        # 메모리 사용량 측정 (로딩 후)
        memory_after = log_memory_usage("YOLO 로딩 후")
        
        # 메모리 증가량 계산
        memory_increase = memory_after['rss'] - memory_before['rss']
        print(f"📊 YOLO 모델 메모리 사용량: +{memory_increase:.1f}MB")
        
        print(f"✅ YOLO 모델 로딩 완료!")
    
    return _yolo_model

def get_clip_model():
    """🤖 CLIP 모델을 싱글톤으로 로드 (메모리 절약)"""
    global _clip_model, _clip_preprocess, _clip_device
    
    if _clip_model is None:
        print("🤖 CLIP 모델 로딩 중...")
        
        # 메모리 사용량 측정 (로딩 전)
        memory_before = log_memory_usage("CLIP 로딩 전")
        
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 환경변수에서 모델 선택 (기본값: 가벼운 ViT-B/16)
        clip_model_name = os.getenv("CLIP_MODEL", "ViT-B/32")  # 최경량 모델 (극한 메모리 절약)
        print(f"📊 사용할 CLIP 모델: {clip_model_name}")
        
        _clip_model, _clip_preprocess = clip.load(clip_model_name, device=_clip_device)
        
        # 메모리 사용량 측정 (로딩 후)
        memory_after = log_memory_usage("CLIP 로딩 후")
        
        # 메모리 증가량 계산
        memory_increase = memory_after['rss'] - memory_before['rss']
        print(f"📊 CLIP 모델 메모리 사용량: +{memory_increase:.1f}MB")
        
        print(f"✅ CLIP 모델 로딩 완료 (Device: {_clip_device})")
    
    return _clip_model, _clip_preprocess, _clip_device

# -------------------------------
# 🤖 CLIP AI 기반 색상 추출
# -------------------------------
def extract_color_with_clip_ai(image, mask):
    """
    🤖 CLIP AI를 사용해서 홀드의 색상을 직접 추출
    
    Args:
        image: 원본 이미지 (BGR)
        mask: 홀드 마스크 (0/1)
    
    Returns:
        color_name: 인식된 색상 이름 (예: "yellow", "red")
        confidence: 신뢰도 (0~1)
        rgb: 대표 RGB 값
        hsv: 대표 HSV 값
        clip_features: CLIP 특징 벡터 (512차원)
    """
    model, preprocess, device = get_clip_model()
    
    # 홀드 영역 추출
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return "unknown", 0.0, [128, 128, 128], [0, 0, 128], np.zeros(512)
    
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    
    # 홀드 크롭
    hold_image = image[y_min:y_max+1, x_min:x_max+1]
    hold_pil = Image.fromarray(cv2.cvtColor(hold_image, cv2.COLOR_BGR2RGB))
    
    # 🔧 마스크 침범 방지: mask_core 생성
    mask_area = mask[y_min:y_max+1, x_min:x_max+1]
    kernel_size = max(3, min(mask_area.shape) // 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_core = cv2.erode((mask_area * 255).astype(np.uint8), kernel, iterations=2)
    mask_core = (mask_core > 127).astype(np.float32)
    
    # 색상 프롬프트 정의 (검정색 우선, 주황색 강화)
    color_prompts = [
        "a black climbing hold", "a very dark black climbing hold", "a dark black climbing hold",  # 검정색 최우선
        "a bright orange climbing hold",
        "a dark orange climbing hold",
        "an orange climbing hold",
        "a bright yellow climbing hold",
        "a yellow climbing hold",
        "a light yellow climbing hold",
        "a dark yellow climbing hold",
        "a red climbing hold", 
        "a dark red climbing hold",
        "a blue climbing hold",
        "a light blue climbing hold",
        "a dark blue climbing hold",
        "a green climbing hold",
        "a light green climbing hold",
        "a dark green climbing hold",
        "a purple climbing hold",
        "a pink climbing hold",
        "a white climbing hold",
        "a gray climbing hold",
        "a brown climbing hold"
    ]
    
    # 색상 매핑 (검정색 우선)
    color_map = {
        "black": ["black", "very dark black", "dark black"],  # 검정색 최우선
        "orange": ["orange", "bright orange", "dark orange"],
        "yellow": ["yellow", "light yellow", "dark yellow", "bright yellow"],
        "red": ["red", "dark red"],
        "blue": ["blue", "light blue", "dark blue"],
        "green": ["green", "light green", "dark green"],
        "purple": ["purple"],
        "pink": ["pink"],
        "white": ["white"],
        "gray": ["gray"],
        "brown": ["brown"]
    }
    
    # 텍스트 특징 추출
    text_tokens = clip.tokenize(color_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 이미지 특징 추출
        image_input = preprocess(hold_pil).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 🎯 검정색 홀드 사전 감지 (개별 함수용) - 다단계 로직
    pixels = hold_image[mask_core > 0]
    is_black_candidate = False
    black_confidence_level = "low"
    
    if len(pixels) > 10:
        avg_rgb = np.mean(pixels, axis=0)
        avg_brightness = np.mean(avg_rgb)
        max_rgb = np.max(avg_rgb)
        
        # RGB 표준편차와 채널 차이 계산
        rgb_std = np.std(pixels, axis=0)
        avg_std = np.mean(rgb_std)
        channel_diff = np.max(avg_rgb) - np.min(avg_rgb)
        
        # 1단계: 진짜 검정색 (매우 어두움)
        if avg_brightness <= 80 and max_rgb <= 100:
            is_black_candidate = True
            black_confidence_level = "very_high"
            print(f"   🖤 개별 함수: 진짜 검정색 (평균: {avg_brightness:.1f}, 최대: {max_rgb:.1f})")
        
        # 2단계: 모든 밝기에서 색상 특성 기반 판별 (개별 함수)
        else:
            # 🚨 색상 특성 체크: 무채색인지 확인
            r, g, b = avg_rgb[0], avg_rgb[1], avg_rgb[2]
            
            # 보라색 특성 체크: Red와 Blue가 높고 Green이 낮음 (더 완화)
            is_purple = (r > g + 3 and b > g + 3)
            
            # 노란색 특성 체크: Red와 Green이 높고 Blue가 낮음 (더 완화)
            is_yellow = (r > b + 3 and g > b + 3)
            
            # 파란색 특성 체크: Blue가 다른 채널보다 높음 (완화)
            is_blue = (b > r + 10 and b > g + 10)
            
            # 빨간색 특성 체크: Red가 다른 채널보다 높음 (완화)
            is_red = (r > g + 10 and r > b + 10)
            
            # 초록색 특성 체크: Green이 다른 채널보다 높음 (완화)
            is_green = (g > r + 10 and g > b + 10)
            
            # 🎯 무채색(검정색/회색/흰색) 조건: 색상 특성이 없고 채널 차이가 작음
            is_achromatic = not (is_purple or is_yellow or is_blue or is_red or is_green)
            
            # 무채색이면 검정색으로 분류 (밝기 무관)
            if is_achromatic and channel_diff < 50:
                is_black_candidate = True
                black_confidence_level = "high"
                print(f"   🖤 개별 함수: 무채색 검정색 (RGB: {avg_rgb}, 채널차: {channel_diff:.1f}, 밝기: {avg_brightness:.1f})")
            elif is_purple:
                print(f"   💜 개별 함수: 보라색 특성 감지 (RGB: {avg_rgb}) - 검정색 제외")
            elif is_yellow:
                print(f"   💛 개별 함수: 노란색 특성 감지 (RGB: {avg_rgb}) - 검정색 제외")
            elif is_blue:
                print(f"   💙 개별 함수: 파란색 특성 감지 (RGB: {avg_rgb}) - 검정색 제외")
            elif is_red:
                print(f"   ❤️ 개별 함수: 빨간색 특성 감지 (RGB: {avg_rgb}) - 검정색 제외")
            elif is_green:
                print(f"   💚 개별 함수: 초록색 특성 감지 (RGB: {avg_rgb}) - 검정색 제외")
    
    # 유사도 계산
    similarities = (image_features @ text_features.T).squeeze().cpu().numpy()
    
    # 🎯 검정색 후보 강제 분류 (신뢰도별)
    if is_black_candidate:
        if black_confidence_level == "very_high":
            confidence = 0.98
        elif black_confidence_level == "high":
            confidence = 0.95
        else:  # medium
            confidence = 0.90
            
        color_name = "black"
        print(f"   ✅ 검정색으로 강제 분류 (개별 함수, 신뢰도: {black_confidence_level})")
    else:
        # 가장 유사한 색상 선택
        best_idx = np.argmax(similarities)
        confidence = float(similarities[best_idx])
        best_prompt = color_prompts[best_idx]
        
        # 색상 이름 추출
        color_name = "unknown"
        for color, keywords in color_map.items():
            if any(keyword in best_prompt for keyword in keywords):
                color_name = color
                break
    
    # 🎯 마스크 침범 방지: 중심부 픽셀만 사용 (경계 제외)
    mask_area = mask[y_min:y_max+1, x_min:x_max+1]
    
    # 모폴로지 침식으로 경계 제거 (침범 방지)
    kernel_size = max(3, min(mask_area.shape) // 10)  # 마스크 크기의 10%
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_core = cv2.erode((mask_area * 255).astype(np.uint8), kernel, iterations=2)
    mask_core = (mask_core > 127).astype(np.float32)
    
    # 중심부 픽셀 추출
    pixels = hold_image[mask_core > 0]
    
    if len(pixels) > 10:  # 충분한 픽셀이 있을 때만
        # 밝은 픽셀만 선택 (상위 30%)
        pixels_hsv = cv2.cvtColor(hold_image, cv2.COLOR_BGR2HSV)[mask_core > 0]
        brightness = pixels_hsv[:, 2]
        bright_threshold = np.percentile(brightness, 70)
        bright_mask = brightness >= bright_threshold
        
        if np.sum(bright_mask) > 10:
            pixels = pixels[bright_mask]
            pixels_hsv = pixels_hsv[bright_mask]
        
        # RGB/HSV 평균
        rgb = np.mean(pixels, axis=0).astype(int)[::-1]  # BGR -> RGB
        hsv = np.mean(pixels_hsv, axis=0).astype(int)
    else:
        # 중심부가 너무 작으면 원본 마스크 사용
        pixels = hold_image[mask_area > 0]
        if len(pixels) > 0:
            pixels_hsv = cv2.cvtColor(hold_image, cv2.COLOR_BGR2HSV)[mask_area > 0]
            rgb = np.mean(pixels, axis=0).astype(int)[::-1]
            hsv = np.mean(pixels_hsv, axis=0).astype(int)
        else:
            rgb = [128, 128, 128]
            hsv = [0, 0, 128]
    
    # CLIP 특징 벡터 반환
    clip_features = image_features.squeeze().cpu().numpy()
    
    print(f"   🎨 CLIP AI: {color_name} (신뢰도: {confidence:.3f})")
    
    return color_name, confidence, rgb.tolist(), hsv.tolist(), clip_features

def extract_colors_with_clip_ai_batch(hold_images, masks):
    """
    🚀 CLIP AI 배치 처리로 여러 홀드의 색상을 한 번에 추출
    
    Args:
        hold_images: 홀드 이미지 리스트 (BGR)
        masks: 홀드 마스크 리스트 (0/1)
    
    Returns:
        results: 각 홀드별 (color_name, confidence, rgb, hsv, clip_features) 리스트
    """
    if not hold_images:
        return []
    
    model, preprocess, device = get_clip_model()
    
    # 🚀 성능 최적화: 검정색 사전 감지를 최대한 간소화 (속도 우선)
    black_candidates = []
    
    for i, (image, mask) in enumerate(zip(hold_images, masks)):
        # 🚀 빠른 샘플링: 마스크에서 임의로 100개 픽셀만 추출
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0:
            continue
        
        # 랜덤 샘플링으로 속도 향상
        sample_count = min(100, len(y_coords))
        sample_indices = np.random.choice(len(y_coords), sample_count, replace=False)
        sampled_y = y_coords[sample_indices]
        sampled_x = x_coords[sample_indices]
        
        pixels = image[sampled_y, sampled_x]
        if len(pixels) > 10:
            # 🚀 극단적 최적화: 매우 간단한 검정색 감지만 수행
            avg_rgb = np.mean(pixels, axis=0)
            avg_brightness = np.mean(avg_rgb)
            channel_diff = np.max(avg_rgb) - np.min(avg_rgb)
            
            # 검정색 후보: 어둡고(< 80) 무채색(채널차 < 30)
            if avg_brightness < 80 and channel_diff < 30:
                black_candidates.append((i, "high"))
    
    # 색상 프롬프트 정의 (다양한 표현 유지)
    color_prompts = [
        "a black climbing hold", "a very dark black climbing hold", "a dark black climbing hold",
        "a white climbing hold", "a bright white climbing hold", "a pure white climbing hold",
        "a gray climbing hold", "a light gray climbing hold", "a dark gray climbing hold",
        "an orange climbing hold", "a bright orange climbing hold", "a dark orange climbing hold",
        "a yellow climbing hold", "a bright yellow climbing hold", "a lemon yellow climbing hold", "a golden yellow climbing hold",
        "a red climbing hold", "a bright red climbing hold", "a dark red climbing hold",
        "a pink climbing hold", "a hot pink climbing hold",
        "a blue climbing hold", "a light blue climbing hold", "a sky blue climbing hold",
        "a green climbing hold", "a bright green climbing hold", "a forest green climbing hold",
        "a mint climbing hold", "a mint green climbing hold", "a turquoise mint climbing hold",
        "a lime climbing hold", "a lime green climbing hold", "a neon lime climbing hold",
        "a purple climbing hold", "a bright purple climbing hold", "a violet climbing hold",
        "a brown climbing hold", "a dark brown climbing hold"
    ]
    
    color_map = {
        "black": ["black", "very dark black", "dark black"],
        "white": ["white", "bright white", "pure white"],
        "gray": ["gray", "light gray", "dark gray"],
        "orange": ["orange", "bright orange", "dark orange"],
        "yellow": ["yellow", "bright yellow", "lemon", "golden"],
        "red": ["red", "bright red", "dark red"],
        "pink": ["pink", "hot pink"],
        "blue": ["blue", "light blue", "sky blue"],
        "green": ["green", "bright green", "forest"],
        "mint": ["mint", "mint green", "turquoise mint"],
        "lime": ["lime", "lime green", "neon lime"],
        "purple": ["purple", "bright purple", "violet"],
        "brown": ["brown", "dark brown"]
    }
    
    # 텍스트 특징 추출 (한 번만)
    text_tokens = clip.tokenize(color_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 🚀 메모리 최적화: 배치 크기를 환경변수로 설정 (기본값: 16)
    batch_size = int(os.getenv("CLIP_BATCH_SIZE", "2"))  # 메모리 절약을 위해 더 작게 설정
    print(f"📊 CLIP 배치 크기: {batch_size}")
    
    all_similarities = []
    all_image_features = []
    valid_indices = []
    
    for batch_start in range(0, len(hold_images), batch_size):
        batch_end = min(batch_start + batch_size, len(hold_images))
        batch_images = hold_images[batch_start:batch_end]
        batch_masks = masks[batch_start:batch_end]
        
        processed_images = []
        batch_valid_indices = []
        
        for i, (image, mask) in enumerate(zip(batch_images, batch_masks)):
            actual_idx = batch_start + i
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0:
                continue
                
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            hold_image = image[y_min:y_max+1, x_min:x_max+1]
            hold_pil = Image.fromarray(cv2.cvtColor(hold_image, cv2.COLOR_BGR2RGB))
            processed_images.append(preprocess(hold_pil))
            batch_valid_indices.append(actual_idx)
        
        if not processed_images:
            continue
        
        # 배치로 이미지 특징 추출
        images_tensor = torch.stack(processed_images).to(device)
        
        with torch.no_grad():
            batch_image_features = model.encode_image(images_tensor)
            batch_image_features = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
            
            # 유사도 계산 (배치)
            batch_similarities = (batch_image_features @ text_features.T).cpu().numpy()
        
        all_similarities.append(batch_similarities)
        all_image_features.append(batch_image_features)
        valid_indices.extend(batch_valid_indices)
        
        # 🚀 메모리 최적화: 배치마다 메모리 정리
        del batch_image_features, batch_similarities, images_tensor
        if 'processed_images' in locals():
            del processed_images
        gc.collect()
    
    if not all_similarities:
        return []
    
    # 모든 배치 결과 합치기
    similarities = np.vstack(all_similarities)
    image_features = torch.cat(all_image_features, dim=0)
    
    # 결과 처리
    results = []
    for i, orig_idx in enumerate(valid_indices):
        # 원본 이미지와 마스크 가져오기
        image = hold_images[orig_idx]
        mask = masks[orig_idx]
        
        # 가장 유사한 색상 선택
        best_idx = np.argmax(similarities[i])
        confidence = float(similarities[i][best_idx])
        best_prompt = color_prompts[best_idx]
        
        # 🎯 검정색 후보에 대한 특별 처리
        is_black_candidate = False
        black_confidence_level = None
        
        for candidate_idx, conf_level in black_candidates:
            if candidate_idx == orig_idx:
                is_black_candidate = True
                black_confidence_level = conf_level
                break
        
        if is_black_candidate:
            print(f"   🖤 홀드 {orig_idx}: 검정색 후보 ({black_confidence_level}) - 강제 검정색 분류")
            
            # 신뢰도에 따른 강제 분류
            if black_confidence_level == "very_high":
                color_name = "black"
                confidence = 0.98
            elif black_confidence_level == "high":
                color_name = "black"
                confidence = 0.95
            else:  # medium
                color_name = "black"
                confidence = 0.90
            
            print(f"      ✅ 검정색으로 강제 분류 (신뢰도: {black_confidence_level}, confidence: {confidence})")
            
            # 추가 검증: 다른 색상으로 분류될 가능성 체크
            other_color_similarities = []
            for j, prompt in enumerate(color_prompts):
                if "black" not in prompt.lower():
                    other_color_similarities.append(similarities[i][j])
            
            if other_color_similarities:
                max_other_similarity = max(other_color_similarities)
                if max_other_similarity > 0.3:  # 다른 색상 유사도가 높으면 경고
                    print(f"      ⚠️ 다른 색상 유사도도 높음: {max_other_similarity:.3f}")
        else:
            # 일반 홀드 처리
            color_name = "unknown"
            for color, keywords in color_map.items():
                if any(keyword in best_prompt for keyword in keywords):
                    color_name = color
                    break
        
        # RGB/HSV 추출 (기존 로직 재사용)
        y_coords, x_coords = np.where(mask > 0)
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        hold_image = image[y_min:y_max+1, x_min:x_max+1]
        
        # 마스크 침범 방지 로직
        mask_area = mask[y_min:y_max+1, x_min:x_max+1]
        kernel_size = max(3, min(mask_area.shape) // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_core = cv2.erode((mask_area * 255).astype(np.uint8), kernel, iterations=2)
        mask_core = (mask_core > 127).astype(np.float32)
        
        pixels = hold_image[mask_core > 0]
        if len(pixels) > 10:
            pixels_hsv = cv2.cvtColor(hold_image, cv2.COLOR_BGR2HSV)[mask_core > 0]
            brightness = pixels_hsv[:, 2]
            bright_threshold = np.percentile(brightness, 70)
            bright_mask = brightness >= bright_threshold
            
            if np.sum(bright_mask) > 10:
                pixels = pixels[bright_mask]
                pixels_hsv = pixels_hsv[bright_mask]
            
            rgb = np.mean(pixels, axis=0).astype(int)[::-1]  # BGR -> RGB
            hsv = np.mean(pixels_hsv, axis=0).astype(int)
        else:
            rgb = [128, 128, 128]
            hsv = [0, 0, 128]
        
        results.append((color_name, confidence, rgb.tolist(), hsv.tolist(), image_features[i].cpu().numpy()))
    
    return results

# -------------------------------
# 📌 Resize + Padding
# -------------------------------
def resize_with_padding(image, target_size=(640, 640), pad_color=(255, 255, 255)):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded, scale, pad_left, pad_top

# -------------------------------
# 📌 원본 크기 복원
# -------------------------------
def restore_mask_to_original(mask, original_shape, scale, pad_left, pad_top):
    h_ori, w_ori = original_shape
    unpadded = mask[pad_top:pad_top + int(h_ori * scale), pad_left:pad_left + int(w_ori * scale)]
    restored = cv2.resize(unpadded, (w_ori, h_ori), interpolation=cv2.INTER_NEAREST)
    return restored

# -------------------------------
# 📌 대표색 추출 (Dominant Color) - 앙상블 방식
# -------------------------------
def remove_outliers(pixels, percentile=5):
    """아웃라이어 제거 (상위/하위 5%)"""
    if len(pixels) == 0:
        return pixels
    lower = np.percentile(pixels, percentile, axis=0)
    upper = np.percentile(pixels, 100 - percentile, axis=0)
    mask = np.all((pixels >= lower) & (pixels <= upper), axis=1)
    return pixels[mask]

def refine_mask_boundary(mask, kernel_size=3, iterations=2):
    """마스크 경계 정제 - 모폴로지 연산으로 부드럽게"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 닫힘 연산 (구멍 메우기)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # 열림 연산 (노이즈 제거)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opened

def detect_background_color(image, masks):
    """배경색 자동 감지 - 나무 벽면 색상 추출"""
    if len(masks) == 0:
        return None
    
    # 모든 홀드 마스크를 합쳐서 배경 영역 찾기
    all_holds_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        all_holds_mask = cv2.bitwise_or(all_holds_mask, (mask * 255).astype(np.uint8))
    
    # 배경 영역 (홀드가 아닌 부분)
    background_mask = cv2.bitwise_not(all_holds_mask)
    
    # 배경에서 샘플링
    background_pixels = image[background_mask > 0]
    
    if len(background_pixels) > 100:
        # 배경색의 평균값 계산
        background_hsv = cv2.cvtColor(background_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        avg_background_hsv = np.mean(background_hsv, axis=0)[0]
        
        print(f"🎨 배경색 감지: HSV({avg_background_hsv[0]:.1f}, {avg_background_hsv[1]:.1f}, {avg_background_hsv[2]:.1f})")
        return avg_background_hsv
    
    return None

def filter_background_pixels(pixels_hsv, background_hsv, threshold=30):
    """배경색과 유사한 픽셀 제거"""
    if background_hsv is None or len(pixels_hsv) == 0:
        return pixels_hsv
    
    # HSV 거리 계산
    h_diff = np.minimum(np.abs(pixels_hsv[:, 0] - background_hsv[0]), 
                        360 - np.abs(pixels_hsv[:, 0] - background_hsv[0]))
    s_diff = np.abs(pixels_hsv[:, 1] - background_hsv[1])
    v_diff = np.abs(pixels_hsv[:, 2] - background_hsv[2])
    
    # 가중치 적용 (H:2, S:1, V:1)
    distance = np.sqrt(2 * h_diff**2 + s_diff**2 + v_diff**2)
    
    # 배경색과 유사한 픽셀 제거
    filtered_mask = distance > threshold
    filtered_pixels = pixels_hsv[filtered_mask]
    
    print(f"🚫 배경색 필터링: {len(pixels_hsv)} → {len(filtered_pixels)} 픽셀")
    return filtered_pixels

def extract_best_color_multiple_methods(pixels_hsv):
    """다중 방법으로 색상 추출 후 최적 선택"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # 방법 1: K-means (기본)
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(3, len(pixels_hsv)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels_hsv)
        dominant_color_kmeans = kmeans.cluster_centers_[np.argmax(np.bincount(labels))]
    except:
        dominant_color_kmeans = np.mean(pixels_hsv, axis=0)
    
    # 방법 2: 채도 기반 가중 평균
    saturation_weights = pixels_hsv[:, 1] / 255.0
    saturation_weights = saturation_weights ** 2  # 채도 가중치 강화
    if np.sum(saturation_weights) > 0:
        dominant_color_weighted = np.average(pixels_hsv, axis=0, weights=saturation_weights)
    else:
        dominant_color_weighted = np.mean(pixels_hsv, axis=0)
    
    # 방법 3: 중앙값 (이상치 제거)
    dominant_color_median = np.median(pixels_hsv, axis=0)
    
    # 방법 4: 히스토그램 피크
    h_hist = np.histogram(pixels_hsv[:, 0], bins=36, range=(0, 360))[0]
    peak_h_idx = np.argmax(h_hist)
    peak_h = peak_h_idx * 10  # 10도 단위로 양자화
    
    # 해당 Hue 범위의 픽셀들만 사용
    h_mask = (pixels_hsv[:, 0] >= peak_h - 10) & (pixels_hsv[:, 0] <= peak_h + 10)
    if np.sum(h_mask) > 0:
        peak_pixels = pixels_hsv[h_mask]
        dominant_color_peak = np.mean(peak_pixels, axis=0)
        dominant_color_peak[0] = peak_h  # Hue는 피크 값 사용
    else:
        dominant_color_peak = np.mean(pixels_hsv, axis=0)
    
    # 각 방법의 품질 점수 계산
    methods = [
        (dominant_color_kmeans, "K-means"),
        (dominant_color_weighted, "채도 가중"),
        (dominant_color_median, "중앙값"),
        (dominant_color_peak, "히스토그램 피크")
    ]
    
    best_color = dominant_color_kmeans
    best_score = -1
    
    for color, method_name in methods:
        # 품질 점수: 채도 * 명도 * 일관성
        saturation = color[1] / 255.0
        brightness = color[2] / 255.0
        
        # 일관성 점수 (주변 픽셀과의 유사도)
        if len(pixels_hsv) > 10:
            distances = np.sqrt(np.sum((pixels_hsv - color)**2, axis=1))
            consistency = 1.0 / (1.0 + np.std(distances))
        else:
            consistency = 1.0
        
        score = saturation * brightness * consistency
        
        if score > best_score:
            best_score = score
            best_color = color
            
        print(f"   {method_name}: HSV({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f}) - 점수: {score:.3f}")
    
    print(f"🏆 최적 색상 선택: HSV({best_color[0]:.1f}, {best_color[1]:.1f}, {best_color[2]:.1f})")
    return best_color

def extract_core_pixels(pixels_hsv, core_ratio=0.7):
    """홀드 중심부 픽셀만 추출 - 가장 순수한 색상"""
    if len(pixels_hsv) == 0:
        return pixels_hsv
    
    # 채도 기준으로 상위 core_ratio%만 선택
    saturation_scores = pixels_hsv[:, 1]  # S 채널
    threshold = np.percentile(saturation_scores, (1 - core_ratio) * 100)
    core_mask = saturation_scores >= threshold
    
    return pixels_hsv[core_mask]

def get_kmeans_dominant_color(pixels, k=3):
    """방법 1: K-means 클러스터링"""
    if len(pixels) == 0:
        return [0, 0, 0]
    kmeans = KMeans(n_clusters=min(k, len(pixels)), n_init=10, random_state=42)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant.tolist() if hasattr(dominant, 'tolist') else list(dominant)

def get_histogram_peak_color(pixels_hsv):
    """방법 2: Histogram peak (Hue 기준)"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # Hue 히스토그램 (18개 구간, 10도씩)
    hist, bins = np.histogram(pixels_hsv[:, 0], bins=18, range=(0, 180))
    peak_bin = np.argmax(hist)
    peak_hue = (bins[peak_bin] + bins[peak_bin + 1]) / 2
    
    # 해당 Hue 근처의 픽셀들만 선택
    hue_range = 10
    mask = np.abs(pixels_hsv[:, 0] - peak_hue) < hue_range
    if np.sum(mask) > 0:
        result = np.mean(pixels_hsv[mask], axis=0)
    else:
        result = np.mean(pixels_hsv, axis=0)
    return result.tolist() if hasattr(result, 'tolist') else list(result)

def get_median_color(pixels):
    """방법 3: Median (중앙값)"""
    if len(pixels) == 0:
        return [0, 0, 0]
    result = np.median(pixels, axis=0)
    return result.tolist() if hasattr(result, 'tolist') else list(result)

def get_weighted_mean_color(pixels_hsv):
    """방법 4: 가중 평균 (채도가 높은 픽셀에 더 큰 가중치)"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # 채도를 가중치로 사용 (채도가 높을수록 순수한 색상)
    weights = pixels_hsv[:, 1] / 255.0 + 0.1  # 0으로 나누기 방지
    weights = weights / np.sum(weights)
    
    # Hue는 원형이므로 특별 처리
    h_rad = np.deg2rad(pixels_hsv[:, 0] / 180.0 * 360.0)
    cos_h = np.sum(np.cos(h_rad) * weights)
    sin_h = np.sum(np.sin(h_rad) * weights)
    weighted_h = np.rad2deg(np.arctan2(sin_h, cos_h)) / 360.0 * 180.0
    if weighted_h < 0:
        weighted_h += 180
    
    weighted_s = np.sum(pixels_hsv[:, 1] * weights)
    weighted_v = np.sum(pixels_hsv[:, 2] * weights)
    
    return [weighted_h, weighted_s, weighted_v]

def colors_are_similar(color1, color2, h_thresh=15, s_thresh=30, v_thresh=30):
    """두 색상이 유사한지 판단"""
    h1, s1, v1 = color1
    h2, s2, v2 = color2
    
    # Hue 원형 거리
    h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2))
    
    return (h_diff < h_thresh and 
            abs(s1 - s2) < s_thresh and 
            abs(v1 - v2) < v_thresh)

def get_black_dominant_color(pixels_hsv):
    """🚨 검정색 홀드 전용 색상 추출"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # 검정색 홀드는 Value가 낮은 픽셀들을 우선적으로 고려
    # Value 기준으로 정렬하여 어두운 픽셀들 우선 선택
    sorted_pixels = sorted(pixels_hsv, key=lambda x: x[2])  # Value 기준 정렬
    
    # 하위 50% 픽셀들만 사용 (가장 어두운 픽셀들)
    dark_pixels = sorted_pixels[:len(sorted_pixels)//2]
    
    if len(dark_pixels) == 0:
        dark_pixels = sorted_pixels[:max(1, len(sorted_pixels)//4)]
    
    # 검정색 홀드의 경우 Value 중심으로 색상 추출
    # Hue와 Saturation은 덜 중요, Value가 가장 중요
    
    # 1. Value의 중간값 사용
    v_values = [p[2] for p in dark_pixels]
    median_v = np.median(v_values)
    
    # 2. Hue는 전체 픽셀의 중간값 사용 (검정색은 Hue가 중요하지 않음)
    h_values = [p[0] for p in dark_pixels]
    median_h = np.median(h_values)
    
    # 3. Saturation은 낮게 설정 (검정색은 채도가 낮음)
    s_values = [p[1] for p in dark_pixels]
    median_s = min(np.median(s_values), 30)  # 최대 30으로 제한
    
    return [int(median_h), int(median_s), int(median_v)]

def get_white_dominant_color(pixels_hsv):
    """🚨 흰색 홀드 전용 색상 추출"""
    if len(pixels_hsv) == 0:
        return [0, 0, 255]
    
    # 흰색 홀드는 Value가 높고 Saturation이 낮은 픽셀들을 우선적으로 고려
    # Value 기준으로 정렬하여 밝은 픽셀들 우선 선택
    sorted_pixels = sorted(pixels_hsv, key=lambda x: x[2], reverse=True)  # Value 기준 역순 정렬
    
    # 상위 50% 픽셀들만 사용 (가장 밝은 픽셀들)
    bright_pixels = sorted_pixels[:len(sorted_pixels)//2]
    
    if len(bright_pixels) == 0:
        bright_pixels = sorted_pixels[:max(1, len(sorted_pixels)//4)]
    
    # 흰색 홀드의 경우 Value와 Saturation 중심으로 색상 추출
    # Hue는 덜 중요, Value가 높고 Saturation이 낮아야 함
    
    # 1. Value의 중간값 사용 (높게)
    v_values = [p[2] for p in bright_pixels]
    median_v = max(np.median(v_values), 200)  # 최소 200으로 설정
    
    # 2. Saturation은 낮게 설정 (흰색은 채도가 낮음)
    s_values = [p[1] for p in bright_pixels]
    median_s = min(np.median(s_values), 30)  # 최대 30으로 제한
    
    # 3. Hue는 전체 픽셀의 중간값 사용 (흰색은 Hue가 중요하지 않음)
    h_values = [p[0] for p in bright_pixels]
    median_h = np.median(h_values)
    
    return [int(median_h), int(median_s), int(median_v)]

def normalize_brightness_invariant_color(pixels_hsv):
    """🌞 명도 정규화: 어둡고 밝은 같은 색을 동일하게 인식"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # HSV에서 Hue, Saturation만 사용하고 Value는 정규화
    pixels_array = np.array(pixels_hsv)
    
    # 1단계: Value를 128로 정규화 (중간 명도로 통일)
    normalized_pixels = pixels_array.copy()
    normalized_pixels[:, 2] = 128  # Value를 128로 고정
    
    # 2단계: Saturation 보정 (어두운 색의 채도 보정)
    # Value가 낮을 때 Saturation이 과소평가되는 경우 보정
    original_s = pixels_array[:, 1]
    original_v = pixels_array[:, 2]
    
    # 어두운 픽셀의 채도를 보정 (V < 100인 경우)
    dark_mask = original_v < 100
    if np.any(dark_mask):
        # 어두운 픽셀의 채도를 1.5배로 증가
        brightness_factor = 1.5
        normalized_pixels[dark_mask, 1] = np.minimum(255, original_s[dark_mask] * brightness_factor)
    
    # 3단계: 밝은 픽셀의 채도도 보정 (V > 200인 경우)
    bright_mask = original_v > 200
    if np.any(bright_mask):
        # 밝은 픽셀의 채도를 약간 감소
        brightness_factor = 0.8
        normalized_pixels[bright_mask, 1] = original_s[bright_mask] * brightness_factor
    
    return normalized_pixels

def get_hybrid_dominant_color(pixels_hsv):
    """🎯 하이브리드 색상 추출: 색상 유형별 다른 전처리 전략"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    pixels_array = np.array(pixels_hsv)
    
    # 1단계: 색상 유형 분류
    avg_h = np.mean(pixels_array[:, 0])
    avg_s = np.mean(pixels_array[:, 1]) 
    avg_v = np.mean(pixels_array[:, 2])
    
    # 색상 유형 판단
    is_achromatic = avg_s < 30  # 채도가 낮으면 무채색 (흰색, 검정색, 회색)
    is_dark = avg_v < 80        # 어두운 색
    is_bright = avg_v > 180     # 밝은 색
    
    print(f"🔍 색상 분석: H={avg_h:.1f}, S={avg_s:.1f}, V={avg_v:.1f}")
    print(f"   무채색: {is_achromatic}, 어두움: {is_dark}, 밝음: {is_bright}")
    
    # 2단계: 유형별 전처리 전략
    if is_achromatic:
        # 무채색 (흰색, 검정색, 회색) → 명도 정규화 하지 않음
        print("   → 무채색: 기존 방식 사용")
        return get_dominant_color(pixels_hsv)
    
    elif is_dark or is_bright:
        # 어두운/밝은 유채색 → 명도 정규화 적용
        print("   → 어두운/밝은 유채색: 명도 정규화 적용")
        normalized_pixels = normalize_brightness_invariant_color(pixels_hsv)
        
        # K-means로 대표색 추출
        from sklearn.cluster import KMeans
        if len(normalized_pixels) < 3:
            return [int(np.mean(normalized_pixels[:, 0])), 
                    int(np.mean(normalized_pixels[:, 1])), 
                    int(np.mean(normalized_pixels[:, 2]))]
        
        k = min(3, len(normalized_pixels) // 15 + 1)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_pixels)
        
        cluster_sizes = [np.sum(labels == i) for i in range(k)]
        largest_cluster_idx = np.argmax(cluster_sizes)
        
        dominant_hsv = kmeans.cluster_centers_[largest_cluster_idx]
        return [int(dominant_hsv[0]), int(dominant_hsv[1]), int(dominant_hsv[2])]
    
    else:
        # 중간 명도의 유채색 → 기존 방식 사용
        print("   → 중간 명도 유채색: 기존 방식 사용")
        return get_dominant_color(pixels_hsv)

def get_brightness_invariant_dominant_color(pixels_hsv):
    """🌞 명도 무관 색상 추출: 어둡고 밝은 같은 색을 동일하게 인식 (기존 함수)"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # 명도 정규화 적용
    normalized_pixels = normalize_brightness_invariant_color(pixels_hsv)
    
    # 정규화된 픽셀들로 대표색 추출
    # K-means로 클러스터링하여 가장 큰 클러스터의 중심 색상 추출
    from sklearn.cluster import KMeans
    
    if len(normalized_pixels) < 3:
        # 픽셀이 너무 적으면 평균값 사용
        return [int(np.mean(normalized_pixels[:, 0])), 
                int(np.mean(normalized_pixels[:, 1])), 
                int(np.mean(normalized_pixels[:, 2]))]
    
    # K-means 클러스터링 (최대 5개 클러스터)
    k = min(5, len(normalized_pixels) // 10 + 1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized_pixels)
    
    # 가장 큰 클러스터 찾기
    cluster_sizes = [np.sum(labels == i) for i in range(k)]
    largest_cluster_idx = np.argmax(cluster_sizes)
    
    # 가장 큰 클러스터의 중심 색상 반환
    dominant_hsv = kmeans.cluster_centers_[largest_cluster_idx]
    
    return [int(dominant_hsv[0]), int(dominant_hsv[1]), int(dominant_hsv[2])]

def get_robust_dominant_color(pixels_hsv):
    """🚀 극강화된 앙상블 방식: 최고 정확도 색상 추출 + 명도 정규화"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # 🚨 검정색/흰색 홀드 특별 처리
    # Value가 매우 낮은 픽셀들 확인 (검정색 후보)
    low_value_pixels = [p for p in pixels_hsv if p[2] < 50]  # Value < 50
    high_value_pixels = [p for p in pixels_hsv if p[2] > 200 and p[1] < 50]  # Value > 200, Saturation < 50
    
    if len(low_value_pixels) > len(pixels_hsv) * 0.3:  # 30% 이상이 어두운 색상
        # 검정색 홀드로 판단 - 특별 처리
        return get_black_dominant_color(pixels_hsv)
    elif len(high_value_pixels) > len(pixels_hsv) * 0.3:  # 30% 이상이 밝고 채도가 낮은 색상
        # 흰색 홀드로 판단 - 특별 처리
        return get_white_dominant_color(pixels_hsv)
    
    # 1단계: 극도로 엄격한 아웃라이어 제거 (일반 색상용)
    filtered_pixels = remove_outliers(pixels_hsv, percentile=3)  # 3%로 극도로 엄격
    if len(filtered_pixels) < 25:  # 최소 픽셀 수 더 증가
        filtered_pixels = pixels_hsv
    
    # 2단계: 다단계 색상 순도 필터링
    core_pixels = extract_ultra_pure_pixels(filtered_pixels, purity_threshold=0.8)
    if len(core_pixels) < 20:
        core_pixels = extract_high_purity_pixels(filtered_pixels, purity_threshold=0.6)
    
    # 🚨 필터링 후에도 픽셀이 너무 적으면 원본 사용
    if len(core_pixels) < 10:
        print(f"⚠️ 필터링 후 픽셀 부족 ({len(core_pixels)}개) - 원본 사용 (총 {len(pixels_hsv)}개)")
        core_pixels = filtered_pixels
    
    # 최종 안전장치
    if len(core_pixels) == 0:
        print(f"🚨 심각! core_pixels가 비어있음! filtered_pixels: {len(filtered_pixels)}, 원본: {len(pixels_hsv)}")
        core_pixels = pixels_hsv
    
    # 3단계: 8가지 방법으로 대표색 추출
    method1 = get_kmeans_dominant_color(core_pixels, k=5)  # 클러스터 수 더 증가
    method2 = get_histogram_peak_color(core_pixels)
    method3 = get_median_color(core_pixels)
    method4 = get_weighted_mean_color(core_pixels)
    method5 = get_mode_color(core_pixels)
    method6 = get_percentile_color(core_pixels, percentile=80)  # 더 높은 백분위수
    method7 = get_robust_mean_color(core_pixels)  # 새로운 방법
    method8 = get_dominant_hue_color(core_pixels)  # 새로운 방법
    
    # 🚨 [0,0,0] 결과 검증
    candidates = [method1, method2, method3, method4, method5, method6, method7, method8]
    zero_count = sum(1 for c in candidates if c == [0, 0, 0])
    if zero_count > 4:  # 절반 이상이 [0,0,0]이면 문제
        print(f"🚨 앙상블 메서드 중 {zero_count}개가 [0,0,0] 반환!")
        print(f"   core_pixels 길이: {len(core_pixels)}, 샘플: {core_pixels[:3].tolist() if len(core_pixels) >= 3 else []}")
    
    weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05]  # 가중치 재조정
    
    # 4단계: 극도로 엄격한 가중 투표 시스템
    best_candidate = None
    best_score = 0
    
    for i, candidate in enumerate(candidates):
        score = 0
        for j, other in enumerate(candidates):
            # 극도로 극도로 엄격한 유사도 기준
            if colors_are_similar(candidate, other, h_thresh=3, s_thresh=10, v_thresh=10):
                score += weights[j]
        
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    # 5단계: 최종 검증 및 보정
    if best_candidate is not None:
        # 색상 범위 검증 및 보정
        final_color = validate_and_correct_color(best_candidate)
        return final_color
    
    # 모든 방법 실패 시 K-means 결과 사용
    return method1

def extract_ultra_pure_pixels(pixels_hsv, purity_threshold=0.8):
    """🎯 극도로 높은 색상 순도의 픽셀만 추출"""
    if len(pixels_hsv) == 0:
        return pixels_hsv
    
    # 색상 순도 계산 (Saturation과 Value의 곱)
    saturation = pixels_hsv[:, 1] / 255.0
    value = pixels_hsv[:, 2] / 255.0
    color_purity = saturation * value
    
    # 극도로 높은 순도만 선택
    ultra_pure_mask = color_purity >= purity_threshold
    
    if np.sum(ultra_pure_mask) < 15:  # 너무 적으면 임계값 낮춤
        ultra_pure_mask = color_purity >= (purity_threshold * 0.7)
    
    return pixels_hsv[ultra_pure_mask]

def extract_high_purity_pixels(pixels_hsv, purity_threshold=0.7):
    """🎯 높은 색상 순도의 픽셀만 추출"""
    if len(pixels_hsv) == 0:
        return pixels_hsv
    
    # 색상 순도 계산 (Saturation과 Value의 곱)
    saturation = pixels_hsv[:, 1] / 255.0
    value = pixels_hsv[:, 2] / 255.0
    color_purity = saturation * value
    
    # 임계값 이상의 픽셀만 선택
    high_purity_mask = color_purity >= purity_threshold
    
    if np.sum(high_purity_mask) < 10:  # 너무 적으면 임계값 낮춤
        high_purity_mask = color_purity >= (purity_threshold * 0.6)
    
    return pixels_hsv[high_purity_mask]

def get_robust_mean_color(pixels_hsv):
    """🎯 강건한 평균 색상 추출 (아웃라이어 제거)"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # 각 채널별로 아웃라이어 제거 후 평균 계산
    robust_hue = np.median(pixels_hsv[:, 0])  # 중간값 사용
    robust_sat = np.mean(pixels_hsv[:, 1])     # 평균 사용
    robust_val = np.mean(pixels_hsv[:, 2])    # 평균 사용
    
    return [robust_hue, robust_sat, robust_val]

def get_dominant_hue_color(pixels_hsv):
    """🎯 지배적인 Hue 기반 색상 추출"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # Hue 히스토그램에서 가장 빈번한 값 찾기
    hue_hist, hue_bins = np.histogram(pixels_hsv[:, 0], bins=36, range=(0, 180))
    dominant_hue_bin = np.argmax(hue_hist)
    dominant_hue = hue_bins[dominant_hue_bin] + (hue_bins[1] - hue_bins[0]) / 2
    
    # 해당 Hue를 가진 픽셀들의 평균 Saturation과 Value
    hue_mask = (pixels_hsv[:, 0] >= hue_bins[dominant_hue_bin]) & \
               (pixels_hsv[:, 0] < hue_bins[dominant_hue_bin + 1])
    
    if np.sum(hue_mask) > 0:
        avg_sat = np.mean(pixels_hsv[hue_mask, 1])
        avg_val = np.mean(pixels_hsv[hue_mask, 2])
    else:
        avg_sat = np.mean(pixels_hsv[:, 1])
        avg_val = np.mean(pixels_hsv[:, 2])
    
    return [dominant_hue, avg_sat, avg_val]

def get_mode_color(pixels_hsv):
    """🎯 최빈값 기반 색상 추출"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # Hue를 18개 구간으로 양자화
    hue_quantized = np.floor(pixels_hsv[:, 0] / 10).astype(int)
    sat_quantized = np.floor(pixels_hsv[:, 1] / 32).astype(int)
    val_quantized = np.floor(pixels_hsv[:, 2] / 32).astype(int)
    
    # 가장 빈번한 조합 찾기
    mode_hue = np.bincount(hue_quantized).argmax() * 10 + 5
    mode_sat = np.bincount(sat_quantized).argmax() * 32 + 16
    mode_val = np.bincount(val_quantized).argmax() * 32 + 16
    
    return [mode_hue, mode_sat, mode_val]

def get_percentile_color(pixels_hsv, percentile=75):
    """🎯 백분위수 기반 색상 추출"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    h_percentile = np.percentile(pixels_hsv[:, 0], percentile)
    s_percentile = np.percentile(pixels_hsv[:, 1], percentile)
    v_percentile = np.percentile(pixels_hsv[:, 2], percentile)
    
    return [h_percentile, s_percentile, v_percentile]

def validate_and_correct_color(color_hsv):
    """🎯 색상 범위 검증 및 보정"""
    h, s, v = color_hsv
    
    # HSV 범위 검증 및 보정
    h = max(0, min(179, h))
    s = max(0, min(255, s))
    v = max(0, min(255, v))
    
    # 비정상적인 색상 보정
    if s < 30 and v > 200:  # 거의 흰색
        s = 0
        v = 255
    elif v < 30:  # 거의 검은색
        s = 0
        v = 0
    elif s < 10:  # 거의 회색
        s = 0
    
    return [h, s, v]

def get_dominant_color(pixels_hsv, k=3):
    """🎯 상위 밝기 방식: 가장 밝은 픽셀들의 평균 색상 추출"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # 픽셀이 너무 적으면 중앙값 사용
    if len(pixels_hsv) < 10:
        return [int(np.median(pixels_hsv[:, 0])), 
                int(np.median(pixels_hsv[:, 1])), 
                int(np.median(pixels_hsv[:, 2]))]
    
    # 🚀 상위 30% 밝은 픽셀만 사용 (그림자/경계 제외)
    brightness_scores = pixels_hsv[:, 2]  # V 채널
    bright_threshold = np.percentile(brightness_scores, 70)  # 상위 30%
    
    bright_mask = brightness_scores >= bright_threshold
    
    if np.sum(bright_mask) > 10:  # 충분한 밝은 픽셀이 있으면
        pixels_hsv = pixels_hsv[bright_mask]
        print(f"   상위 밝기 픽셀 선별: {len(pixels_hsv)}개 (밝기≥{bright_threshold:.0f})")
    else:
        print(f"   밝은 픽셀 부족, 전체 사용: {len(pixels_hsv)}개")
    
    # 🎯 단순 평균 방식: 밝은 픽셀들의 평균 색상 (더 밝은 결과)
    h_avg = np.mean(pixels_hsv[:, 0])
    s_avg = np.mean(pixels_hsv[:, 1])
    v_avg = np.mean(pixels_hsv[:, 2])
    
    print(f"   밝은 픽셀 평균: HSV({h_avg:.1f}, {s_avg:.1f}, {v_avg:.1f})")
    
    return [int(h_avg), int(s_avg), int(v_avg)]

# -------------------------------
# 📌 픽셀 기반 통계치 추출
# -------------------------------
def calculate_color_stats(image, mask, brightness_normalization=False, 
                          brightness_filter=False, min_brightness=0, max_brightness=100,
                          saturation_filter=False, min_saturation=0):
    """🚀 확장된 색상 통계 추출 - 다중 색상 공간 + 고급 특징 + 명도 정규화 옵션"""
    # 다중 색상 공간 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

    pixels_hsv = hsv_image[mask > 0.5]
    pixels_rgb = rgb_image[mask > 0.5]
    pixels_lab = lab_image[mask > 0.5]
    pixels_yuv = yuv_image[mask > 0.5]
    pixels_xyz = xyz_image[mask > 0.5]
    
    # 🎨 색상 품질 필터링 적용
    if len(pixels_hsv) > 0:
        # 명도 필터링 (V 채널 기준)
        if brightness_filter:
            brightness_mask = (pixels_hsv[:, 2] >= min_brightness * 2.55) & (pixels_hsv[:, 2] <= max_brightness * 2.55)
            pixels_hsv = pixels_hsv[brightness_mask]
            pixels_rgb = pixels_rgb[brightness_mask]
            pixels_lab = pixels_lab[brightness_mask]
            pixels_yuv = pixels_yuv[brightness_mask]
            pixels_xyz = pixels_xyz[brightness_mask]
        
        # 채도 필터링 (S 채널 기준)
        if saturation_filter and len(pixels_hsv) > 0:
            saturation_mask = pixels_hsv[:, 1] >= min_saturation * 2.55
            pixels_hsv = pixels_hsv[saturation_mask]
            pixels_rgb = pixels_rgb[saturation_mask]
            pixels_lab = pixels_lab[saturation_mask]
            pixels_yuv = pixels_yuv[saturation_mask]
            pixels_xyz = pixels_xyz[saturation_mask]
    
    # 필터링 후 픽셀이 부족한 경우 원본 사용
    if len(pixels_hsv) < 10:  # 최소 10개 픽셀 필요
        pixels_hsv = hsv_image[mask > 0.5]
        pixels_rgb = rgb_image[mask > 0.5]
        pixels_lab = lab_image[mask > 0.5]
        pixels_yuv = yuv_image[mask > 0.5]
        pixels_xyz = xyz_image[mask > 0.5]

    # 대표색 추출 (전처리 방법에 따라 선택)
    if brightness_normalization == "하이브리드":
        dominant_hsv = get_hybrid_dominant_color(pixels_hsv)
        print(f"🎯 하이브리드 방식 적용: 원본 HSV 샘플 {len(pixels_hsv)}개")
    elif brightness_normalization == "명도 정규화":
        dominant_hsv = get_brightness_invariant_dominant_color(pixels_hsv)
        print(f"🌞 명도 정규화 적용: 원본 HSV 샘플 {len(pixels_hsv)}개")
    else:
        dominant_hsv = get_dominant_color(pixels_hsv)
        print(f"📊 기존 방식 적용: 원본 HSV 샘플 {len(pixels_hsv)}개")
    
    # 🚨 RGB는 dominant_hsv를 RGB로 직접 변환 (일관성 유지)
    # HSV → RGB 변환으로 통일
    if len(pixels_hsv) == 0:
        print(f"⚠️ 픽셀 없음! pixels_hsv 길이: 0")
        dominant_rgb = [128, 128, 128]  # 회색으로 대체
    else:
        try:
            hsv_arr = np.uint8([[dominant_hsv]])
            rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
            dominant_rgb = [int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2])]
            
            # RGB(0,0,0) 검증
            if dominant_rgb == [0, 0, 0]:
                print(f"⚠️ HSV={dominant_hsv} → RGB(0,0,0) 변환됨! pixels_hsv 길이: {len(pixels_hsv)}")
                print(f"   원본 HSV 샘플: {pixels_hsv[:3].tolist() if len(pixels_hsv) >= 3 else pixels_hsv.tolist()}")
                dominant_rgb = [128, 128, 128]  # 회색으로 대체
        except Exception as e:
            print(f"⚠️ HSV→RGB 변환 오류: {e}, HSV={dominant_hsv}")
            dominant_rgb = [128, 128, 128]  # 회색으로 대체
    
    dominant_lab = get_dominant_color(pixels_lab) if len(pixels_lab) > 0 else [0, 0, 0]
    dominant_yuv = get_dominant_color(pixels_yuv) if len(pixels_yuv) > 0 else [0, 0, 0]
    dominant_xyz = get_dominant_color(pixels_xyz) if len(pixels_xyz) > 0 else [0, 0, 0]
    
    # 기본 통계 계산 (평균, 표준편차, 최솟값, 최댓값)
    hsv_stats = calculate_basic_stats(pixels_hsv)
    rgb_stats = calculate_basic_stats(pixels_rgb)
    lab_stats = calculate_basic_stats(pixels_lab)
    yuv_stats = calculate_basic_stats(pixels_yuv)
    xyz_stats = calculate_basic_stats(pixels_xyz)
    
    # 고급 특징 계산
    advanced_features = calculate_advanced_features(pixels_hsv, pixels_lab, pixels_rgb)

    stats = {
        # 대표색 (5개 색상 공간)
        "dominant_hsv": dominant_hsv,
        "dominant_rgb": dominant_rgb,
        "dominant_lab": dominant_lab,
        "dominant_yuv": dominant_yuv,
        "dominant_xyz": dominant_xyz,
        
        # 기본 통계 (5개 색상 공간)
        "hsv_stats": hsv_stats,
        "rgb_stats": rgb_stats,
        "lab_stats": lab_stats,
        "yuv_stats": yuv_stats,
        "xyz_stats": xyz_stats,
        
        # 고급 특징
        "advanced": advanced_features,
        
        # 호환성을 위한 기존 구조 유지
        "illumination_invariant": advanced_features
    }
    return stats

def calculate_basic_stats(pixels):
    """기본 통계 (평균, 표준편차, 최솟값, 최댓값)"""
    if len(pixels) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 4 * 3 channels
    
    return np.hstack([
        np.mean(pixels, axis=0),
        np.std(pixels, axis=0),
        np.min(pixels, axis=0),
        np.max(pixels, axis=0),
    ]).tolist()

def calculate_advanced_features(pixels_hsv, pixels_lab, pixels_rgb):
    """고급 색상 특징 계산"""
    if len(pixels_hsv) == 0:
        return {
            "lab_ab": [0, 0], "hue_sat": [0, 0], "color_purity": 0.0,
            "hue_variance": 0.0, "saturation_variance": 0.0, "value_variance": 0.0,
            "color_uniformity": 0.0, "contrast": 0.0, "brightness_std": 0.0,
            "hue_dominant_frequency": 0.0, "saturation_consistency": 0.0
        }
    
    # 조명 불변 특징
    lab_a_mean = np.mean(pixels_lab[:, 1])
    lab_b_mean = np.mean(pixels_lab[:, 2])
    hue_mean = np.mean(pixels_hsv[:, 0])
    sat_mean = np.mean(pixels_hsv[:, 1])
    color_purity = sat_mean / 255.0
    
    # 색상 분산 특징
    hue_variance = np.var(pixels_hsv[:, 0])
    saturation_variance = np.var(pixels_hsv[:, 1])
    value_variance = np.var(pixels_hsv[:, 2])
    
    # 색상 균일성 (낮을수록 균일)
    color_uniformity = np.mean([
        hue_variance / 100.0,  # 정규화
        saturation_variance / 100.0,
        value_variance / 100.0
    ])
    
    # 대비 (명도 차이)
    contrast = np.std(pixels_rgb, axis=0).mean() / 255.0
    
    # 밝기 표준편차
    brightness = np.mean(pixels_rgb, axis=1)
    brightness_std = np.std(brightness) / 255.0
    
    # Hue 히스토그램 기반 특징
    hue_hist, _ = np.histogram(pixels_hsv[:, 0], bins=18, range=(0, 180))
    hue_dominant_frequency = np.max(hue_hist) / len(pixels_hsv)
    
    # Saturation 일관성 (높을수록 일관적)
    saturation_consistency = 1.0 - (saturation_variance / 100.0)
    
    return {
        "lab_ab": [float(lab_a_mean), float(lab_b_mean)],
        "hue_sat": [float(hue_mean), float(sat_mean)],
        "color_purity": float(color_purity),
        "hue_variance": float(hue_variance),
        "saturation_variance": float(saturation_variance),
        "value_variance": float(value_variance),
        "color_uniformity": float(color_uniformity),
        "contrast": float(contrast),
        "brightness_std": float(brightness_std),
        "hue_dominant_frequency": float(hue_dominant_frequency),
        "saturation_consistency": float(saturation_consistency)
    }

# -------------------------------
# 📌 Preprocess Pipeline
# -------------------------------
def preprocess(image_input, model_path="/app/holdcheck/roboflow_weights/weights.pt", conf=0.4, brightness_normalization=False, 
               brightness_filter=False, min_brightness=0, max_brightness=100, 
               saturation_filter=False, min_saturation=0, mask_refinement=5, use_clip_ai=False):
    
    # 메모리 사용량 측정 (시작)
    log_memory_usage("Preprocess 시작")
    
    # image_input이 문자열(파일 경로)인지 numpy 배열인지 확인
    if isinstance(image_input, str):
        # 파일 경로인 경우
        original_image = cv2.imread(image_input)
        if original_image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없음: {image_input}")
    else:
        # 이미 numpy 배열인 경우 (이미 로드된 이미지)
        original_image = image_input

    h_img, w_img = original_image.shape[:2]
    padded_image, scale, pad_left, pad_top = resize_with_padding(original_image)
    
    # 메모리 사용량 측정 (이미지 로딩 후)
    log_memory_usage("이미지 로딩 후")

    # 🚀 캐싱된 YOLO 모델 사용 (속도 대폭 향상)
    model = get_yolo_model(model_path)
    
    # 🚀 메모리 최적화: YOLO 해상도를 환경변수로 설정 (기본값: 384)
    yolo_img_size = int(os.getenv("YOLO_IMG_SIZE", "384"))  # 640 → 384 (메모리 절약)
    print(f"📊 YOLO 이미지 크기: {yolo_img_size}")
    
    results = model(padded_image, conf=conf, imgsz=yolo_img_size)[0]
    
    # 메모리 사용량 측정 (YOLO 추론 후)
    log_memory_usage("YOLO 추론 후")

    masks_raw = results.masks.data.cpu().numpy()
    masks = [restore_mask_to_original(m, (h_img, w_img), scale, pad_left, pad_top) for m in masks_raw]

    hold_data = []
    overlay = original_image.copy()

    # 🚀 최적화: 마스크 전처리를 한 번만 수행 (중복 제거)
    if use_clip_ai:
        valid_hold_images = []
        valid_masks = []
        valid_indices = []
        preprocessed_data = {}  # 전처리 결과 캐싱
        
        # 🚨 CRITICAL: 홀드 개수가 너무 많으면 메모리 부족 위험!
        max_holds = int(os.getenv("MAX_HOLDS", "50"))  # 기본값: 50개로 제한
        if len(masks) > max_holds:
            print(f"⚠️  경고: 홀드가 {len(masks)}개 감지되었습니다! (최대 {max_holds}개)")
            print(f"⚠️  메모리 절약을 위해 상위 {max_holds}개만 처리합니다.")
            print(f"⚠️  더 많은 홀드를 처리하려면 MAX_HOLDS 환경변수를 늘려주세요.")
            
            # 면적이 큰 홀드부터 선택 (confidence가 높은 것 우선)
            mask_areas = []
            for mask in masks:
                area = np.sum(mask > 0)
                mask_areas.append(area)
            
            # 면적 기준으로 정렬하고 상위 N개만 선택
            top_indices = np.argsort(mask_areas)[::-1][:max_holds]
            masks = [masks[i] for i in sorted(top_indices)]
            print(f"✅ 상위 {len(masks)}개 홀드 선택 완료")
        
        # 먼저 모든 홀드를 검증하고 수집
        print(f"🔍 홀드 마스크 전처리 중... ({len(masks)}개)")
        for i, mask in enumerate(masks):
            # 🚀 마스크 전처리
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_refined = refine_mask_boundary(mask_uint8, kernel_size=3, iterations=mask_refinement)
            
            contours, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area < 200:
                continue
                
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.1:
                continue
            
            mask_clean = np.zeros_like(mask_refined)
            cv2.fillPoly(mask_clean, [largest_contour], 255)
            
            # 전처리 결과 저장
            preprocessed_data[i] = {
                'mask_refined': mask_refined,
                'largest_contour': largest_contour,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'mask_clean': mask_clean
            }
            
            valid_hold_images.append(original_image)
            valid_masks.append(mask_clean / 255.0)
            valid_indices.append(i)
        
        print(f"✅ 마스크 전처리 완료 ({len(valid_indices)}개 유효)")
        
        # 메모리 사용량 측정 (마스크 전처리 후)
        log_memory_usage("마스크 전처리 후")
        
        # 🚀 배치 처리로 CLIP AI 색상 추출
        if valid_hold_images:
            print(f"🤖 CLIP AI 배치 처리 시작 ({len(valid_hold_images)}개 홀드)")
            
            # 메모리 사용량 측정 (CLIP 처리 전)
            memory_before_clip = log_memory_usage("CLIP 처리 전")
            
            batch_results = extract_colors_with_clip_ai_batch(valid_hold_images, valid_masks)
            
            # 메모리 사용량 측정 (CLIP 처리 후)
            memory_after_clip = log_memory_usage("CLIP 처리 후")
            
            # 메모리 증가량 계산
            clip_memory_increase = memory_after_clip['rss'] - memory_before_clip['rss']
            print(f"📊 CLIP 처리 메모리 사용량: +{clip_memory_increase:.1f}MB")
            
            print(f"✅ CLIP AI 배치 처리 완료")
        else:
            batch_results = []
        
        # 배치 결과를 hold_data에 적용
        batch_idx = 0
        for i, mask in enumerate(masks):
            if i in valid_indices:
                # 배치 처리 결과 사용
                color_name, confidence, rgb, hsv, clip_features = batch_results[batch_idx]
                batch_idx += 1
                
                # 🚀 전처리 결과 재사용 (중복 제거)
                preproc = preprocessed_data[i]
                mask_refined = preproc['mask_refined']
                largest_contour = preproc['largest_contour']
                area = preproc['area']
                perimeter = preproc['perimeter']
                circularity = preproc['circularity']
                mask_clean = preproc['mask_clean']
                
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                stats = {
                    "dominant_rgb": rgb,
                    "dominant_hsv": hsv,
                    "clip_color_name": color_name,
                    "clip_confidence": confidence,
                    "clip_features": clip_features.tolist()
                }
                
                hold_data.append({
                    "id": i,
                    "center": [int(cx), int(cy)],
                    "area": area,
                    "circularity": circularity,
                    **stats,
                    "size": int(np.sum(mask_clean > 0))
                })

                overlay[mask > 0.5] = (0, 255, 0)
                cv2.putText(overlay, f"ID:{i}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            else:
                # 유효하지 않은 홀드는 건너뜀
                continue
    else:
        # 기존 방식 (CLIP AI 사용 안 함)
        for i, mask in enumerate(masks):
            # 🚀 강화된 마스크 전처리
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # 1단계: 마스크 경계 정제
            mask_refined = refine_mask_boundary(mask_uint8, kernel_size=3, iterations=mask_refinement)
            
            # 2단계: 컨투어 기반 품질 검증
            contours, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
                
            # 가장 큰 컨투어 선택
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 더 엄격한 크기 필터링
            if area < 200:  # 최소 크기 증가
                continue
            
            # 3단계: 컨투어 품질 검증
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter == 0:
                continue
            
            # 원형도 검증 (홀드는 대체로 원형에 가까움)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.1:  # 너무 불규칙한 모양 제외
                continue
            
            # 4단계: 최종 마스크 생성
            mask_clean = np.zeros_like(mask_refined)
            cv2.fillPoly(mask_clean, [largest_contour], 255)
            
            # 5단계: 중심점 계산
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # 6단계: 기존 색상 통계 추출
            stats = calculate_color_stats(
                original_image, 
                mask_clean / 255.0, 
                brightness_normalization=brightness_normalization,
                brightness_filter=brightness_filter,
                min_brightness=min_brightness,
                max_brightness=max_brightness,
                saturation_filter=saturation_filter,
                min_saturation=min_saturation
            )
            
            # 🚨 RGB(0,0,0) 검증 및 로그
            if stats.get("dominant_rgb") == [0, 0, 0]:
                print(f"🚨 경고! 홀드 {i}: RGB(0,0,0) 감지!")
                print(f"   - 마스크 픽셀 수: {np.sum(mask_clean > 0)}")
                print(f"   - dominant_hsv: {stats.get('dominant_hsv')}")
                print(f"   - 건너뜀 또는 기본값 설정 필요")
                stats["dominant_rgb"] = [128, 128, 128]  # 회색으로 대체
                stats["dominant_hsv"] = [0, 0, 128]  # 회색 HSV

            hold_data.append({
                "id": i,
                "center": [int(cx), int(cy)],
                "area": area,
                "circularity": circularity,
                **stats,
                "size": int(np.sum(mask_clean > 0))
        })

        overlay[mask > 0.5] = (0, 255, 0)
        cv2.putText(overlay, f"ID:{i}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

    os.makedirs("outputs", exist_ok=True)
    
    # 이미지 입력이 파일 경로인 경우에만 파일명 추출
    if isinstance(image_input, str):
        base_name = os.path.splitext(os.path.basename(image_input))[0]
    else:
        # 이미지 배열인 경우 타임스탬프 사용
        import time
        base_name = f"image_{int(time.time())}"

    cv2.imwrite(f"outputs/{base_name}_preprocessed.png", blended)
    
    # 🚀 JSON 직렬화 가능하도록 데이터 변환
    json_safe_data = convert_to_json_safe(hold_data)
    with open(f"outputs/{base_name}_preprocessed.json", "w", encoding="utf-8") as f:
        json.dump(json_safe_data, f, indent=2, ensure_ascii=False)

    # 메모리 사용량 측정 (완료)
    log_memory_usage("Preprocess 완료")
    
    # 가비지 컬렉션 실행
    gc.collect()
    
    return hold_data, masks