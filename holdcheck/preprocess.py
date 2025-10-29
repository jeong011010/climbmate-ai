import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
import torch
import clip
from PIL import Image

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
# CLIP 모델은 clustering 모듈과 공유 (import 시점에 참조)
_yolo_model = None
_yolo_model_path = None

def get_yolo_model(model_path="/app/holdcheck/roboflow_weights/weights.pt"):
    """🚀 YOLO 모델을 싱글톤으로 로드 (메모리 절약 + 속도 향상)"""
    global _yolo_model, _yolo_model_path
    
    if _yolo_model is None or _yolo_model_path != model_path:
        print(f"🔍 YOLO 모델 로딩 중... ({model_path})")
        _yolo_model = YOLO(model_path)
        _yolo_model_path = model_path
        print(f"✅ YOLO 모델 로딩 완료!")
    
    return _yolo_model

def get_clip_model():
    """🤖 CLIP 모델을 싱글톤으로 로드 (clustering 모듈과 공유)"""
    # clustering 모듈의 전역 캐시를 사용
    import clustering
    
    if clustering._clip_model is None:
        print("🤖 CLIP 모델 로딩 중...")
        clustering._clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=clustering._clip_device)
        clustering._clip_model = (model, preprocess)
        print(f"✅ CLIP 모델 로딩 완료 (Device: {clustering._clip_device})")
    else:
        print(f"✅ CLIP 모델 캐시 사용 (Device: {clustering._clip_device})")
    
    model, preprocess = clustering._clip_model
    return model, preprocess, clustering._clip_device

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
    
    # 🎨 색상 프롬프트 정의 (세분화 + 순서 최적화)
    color_prompts = [
        # 검정/흰색/회색 (무채색 우선)
        "a black climbing hold", "a very dark black climbing hold",
        "a white climbing hold", "a bright white climbing hold",
        "a gray climbing hold", "a light gray climbing hold", "a dark gray climbing hold",
        
        # 주황색 (노란색과 명확히 구분)
        "a bright orange climbing hold", "an orange climbing hold", "a dark orange climbing hold",
        
        # 노란색 (주황색과 구분)
        "a bright yellow climbing hold", "a yellow climbing hold", "a light yellow climbing hold",
        
        # 빨간색
        "a red climbing hold", "a bright red climbing hold", "a dark red climbing hold",
        
        # 초록색 계열 (연두/초록/민트 분리)
        "a lime green climbing hold", "a light lime climbing hold",  # 연두색
        "a green climbing hold", "a dark green climbing hold",  # 초록색
        "a mint green climbing hold", "a turquoise climbing hold",  # 민트/청록
        
        # 파란색
        "a blue climbing hold", "a bright blue climbing hold", "a dark blue climbing hold",
        
        # 보라/핑크
        "a purple climbing hold", "a violet climbing hold",
        "a pink climbing hold", "a magenta climbing hold",
        
        # 갈색
        "a brown climbing hold", "a tan climbing hold"
    ]
    
    # 색상 매핑 (CLIP 프롬프트 → 표준 색상 이름)
    color_map = {
        "black": ["black", "very dark black"],
        "white": ["white", "bright white", "gray", "light gray", "dark gray"],
        "orange": ["orange", "bright orange", "dark orange"],
        "yellow": ["yellow", "bright yellow", "light yellow"],
        "red": ["red", "bright red", "dark red"],
        "lime": ["lime green", "light lime"],  # 🔥 연두색 추가
        "green": ["green", "dark green"],
        "mint": ["mint green", "turquoise"],  # 🔥 민트 추가
        "blue": ["blue", "bright blue", "dark blue"],
        "purple": ["purple", "violet"],
        "pink": ["pink", "magenta"],
        "brown": ["brown", "tan"]
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
    🚀 CLIP AI 배치 처리로 모든 홀드의 색상을 한 번에 추출 (검정색 포함)
    
    Args:
        hold_images: 홀드 이미지 리스트 (BGR)
        masks: 홀드 마스크 리스트 (0/1)
    
    Returns:
        results: 각 홀드별 (color_name, confidence, rgb, hsv, clip_features) 리스트
    """
    if not hold_images:
        return []
    
    print(f"   🚀 CLIP AI로 모든 홀드 색상 분석 중... ({len(hold_images)}개)")
    
    # CLIP 모델 로드 (한 번만!)
    print("   🔄 CLIP 모델 로딩 중... (한 번만!)")
    model, preprocess, device = get_clip_model()
    
    # 🎨 색상 프롬프트 정의 (세분화 + 순서 최적화 - 배치용)
    color_prompts = [
        # 검정/흰색/회색 (무채색 우선)
        "a black climbing hold", "a very dark black climbing hold",
        "a white climbing hold", "a bright white climbing hold",
        "a gray climbing hold", "a light gray climbing hold", "a dark gray climbing hold",
        
        # 주황색 (노란색과 명확히 구분)
        "a bright orange climbing hold", "an orange climbing hold", "a dark orange climbing hold",
        
        # 노란색 (주황색과 구분)
        "a bright yellow climbing hold", "a yellow climbing hold", "a light yellow climbing hold",
        
        # 빨간색
        "a red climbing hold", "a bright red climbing hold", "a dark red climbing hold",
        
        # 초록색 계열 (연두/초록/민트 분리)
        "a lime green climbing hold", "a light lime climbing hold",  # 연두색
        "a green climbing hold", "a dark green climbing hold",  # 초록색
        "a mint green climbing hold", "a turquoise climbing hold",  # 민트/청록
        
        # 파란색
        "a blue climbing hold", "a bright blue climbing hold", "a dark blue climbing hold",
        
        # 보라/핑크
        "a purple climbing hold", "a violet climbing hold",
        "a pink climbing hold", "a magenta climbing hold",
        
        # 갈색
        "a brown climbing hold", "a tan climbing hold"
    ]
    
    # 색상 매핑 (CLIP 프롬프트 → 표준 색상 이름)
    color_map = {
        "black": ["black", "very dark black"],
        "white": ["white", "bright white", "gray", "light gray", "dark gray"],
        "orange": ["orange", "bright orange", "dark orange"],
        "yellow": ["yellow", "bright yellow", "light yellow"],
        "red": ["red", "bright red", "dark red"],
        "lime": ["lime green", "light lime"],  # 🔥 연두색
        "green": ["green", "dark green"],
        "mint": ["mint green", "turquoise"],  # 🔥 민트
        "blue": ["blue", "bright blue", "dark blue"],
        "purple": ["purple", "violet"],
        "pink": ["pink", "magenta"],
        "brown": ["brown", "tan"]
    }
    
    # 텍스트 특징 추출 (한 번만)
    text_tokens = clip.tokenize(color_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 🚀 최적화: 배치 크기를 64로 증가 (속도 우선)
    batch_size = 64
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
        
        # 일반 홀드 처리 (검정색은 이미 제외됨)
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
        
        results.append((orig_idx, color_name, confidence, rgb.tolist(), hsv.tolist(), image_features[i].cpu().numpy()))
    
    # 원래 인덱스 순서로 정렬
    results.sort(key=lambda x: x[0])
    
    # 인덱스 제거하고 반환
    final_results = [(color_name, confidence, rgb, hsv, clip_features) for _, color_name, confidence, rgb, hsv, clip_features in results]
    
    return final_results

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
    """🎨 상식적인 HSV 기반 색상 분류"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    pixels_array = np.array(pixels_hsv)
    
    # 중앙값 사용 (평균보다 outlier에 강함)
    median_h = np.median(pixels_array[:, 0])
    median_s = np.median(pixels_array[:, 1])
    median_v = np.median(pixels_array[:, 2])
    
    print(f"🎨 HSV 중앙값: H={median_h:.1f}, S={median_s:.1f}, V={median_v:.1f}")
    
    # 🔥 1단계: 명도 우선 판단 (검정/흰색은 채도 무관)
    if median_v < 90:
        # 매우 어두움 → 검정 (채도 무관!)
        print(f"   → ⚫ 검정 (V={median_v:.1f} < 90, S={median_s:.1f})")
        return [0, 0, int(min(70, median_v))]
    elif median_v > 200:
        # 매우 밝음 → 흰색 (채도가 낮으면)
        if median_s < 50:
            print(f"   → ⚪ 흰색 (V={median_v:.1f} > 200, S={median_s:.1f} < 50)")
            return [0, 0, 255]
    
    # 🔥 2단계: 채도 기반 무채색 판단 (중간 명도)
    if median_s < 30:
        # 채도가 매우 낮음 → 회색
        print(f"   → ⬜ 회색 (S={median_s:.1f} < 30, V={median_v:.1f})")
        return [0, 0, int(median_v)]
    
    # 🔥 2단계: 유채색 판단 (OpenCV H는 0-180 범위)
    h = median_h
    s = median_s
    v = median_v
    
    # H 범위별 색상 분류
    if (h >= 0 and h < 8) or (h >= 170):  # 빨강 (0-8, 170-180)
        print(f"   → 🔴 빨강 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    elif h >= 8 and h < 18:  # 주황 (8-18)
        print(f"   → 🟠 주황 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    elif h >= 18 and h < 30:  # 노랑 (18-30)
        print(f"   → 🟡 노랑 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    elif h >= 30 and h < 45:  # 연두 (30-45)
        print(f"   → 🟢 연두 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    elif h >= 45 and h < 80:  # 초록 (45-80)
        print(f"   → 🟢 초록 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    elif h >= 80 and h < 95:  # 민트/청록 (80-95)
        print(f"   → 🫧 민트 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    elif h >= 95 and h < 130:  # 파랑 (95-130)
        print(f"   → 🔵 파랑 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    elif h >= 130 and h < 150:  # 보라 (130-150)
        print(f"   → 🟣 보라 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    elif h >= 150 and h < 170:  # 핑크/자홍 (150-170)
        print(f"   → 🩷 핑크 (H={h:.1f})")
        return [int(h), int(s), int(v)]
    
    else:  # 기타 (갈색 등)
        # 채도와 명도로 추가 판단
        if s < 60 and v < 120:
            print(f"   → 🟤 갈색 (H={h:.1f}, S={s:.1f}, V={v:.1f})")
        else:
            print(f"   → ❓ 기타 (H={h:.1f})")
        return [int(h), int(s), int(v)]

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
    """🎯 개선된 방식: 초크 + 벽색 제거 + 채도 기반 색상 추출"""
    if len(pixels_hsv) == 0:
        return [0, 0, 0]
    
    # 픽셀이 너무 적으면 중앙값 사용
    if len(pixels_hsv) < 10:
        return [int(np.median(pixels_hsv[:, 0])), 
                int(np.median(pixels_hsv[:, 1])), 
                int(np.median(pixels_hsv[:, 2]))]
    
    original_count = len(pixels_hsv)
    h_values = pixels_hsv[:, 0]
    s_values = pixels_hsv[:, 1]
    v_values = pixels_hsv[:, 2]
    
    # 🧹 Step 1: 초크 공격적 제거 (V > 160 and S < 60)
    chalk_mask = (v_values > 160) & (s_values < 60)
    non_chalk_pixels = pixels_hsv[~chalk_mask]
    chalk_removed_count = np.sum(chalk_mask)
    
    if len(non_chalk_pixels) > original_count * 0.25:
        pixels_hsv = non_chalk_pixels
        if chalk_removed_count > 0:
            print(f"   🧹 초크 제거: {chalk_removed_count}개 픽셀 제거 ({original_count} → {len(pixels_hsv)})")
    else:
        # 완화된 초크 필터
        chalk_mask_relaxed = (v_values > 200) & (s_values < 40)
        non_chalk_pixels_relaxed = pixels_hsv[~chalk_mask_relaxed]
        if len(non_chalk_pixels_relaxed) > original_count * 0.2:
            pixels_hsv = non_chalk_pixels_relaxed
            print(f"   🧹 완화 필터 적용: {np.sum(chalk_mask_relaxed)}개 제거")
    
    # 🧱 Step 2: 통계적 outlier 제거 (세그먼테이션 오류로 포함된 벽색 자동 제거)
    # 원리: 홀드 픽셀 >> 벽색 픽셀 → 벽색은 outlier!
    # K-means나 DBSCAN 대신 간단한 통계 기반 방식 사용
    
    # 2-1. HSV 각 채널별 중앙값과 표준편차 계산
    h_median = np.median(pixels_hsv[:, 0])
    s_median = np.median(pixels_hsv[:, 1])
    v_median = np.median(pixels_hsv[:, 2])
    
    h_std = np.std(pixels_hsv[:, 0])
    s_std = np.std(pixels_hsv[:, 1])
    v_std = np.std(pixels_hsv[:, 2])
    
    # 2-2. Outlier 판정: 중앙값에서 2σ 이상 벗어난 픽셀 제거
    # H는 원형이므로 특별 처리 (0~180 범위)
    h_diff = np.abs(pixels_hsv[:, 0] - h_median)
    h_diff_circular = np.minimum(h_diff, 180 - h_diff)  # 원형 거리
    
    s_diff = np.abs(pixels_hsv[:, 1] - s_median)
    v_diff = np.abs(pixels_hsv[:, 2] - v_median)
    
    # 2-3. 각 채널에서 2σ 이내인 픽셀만 선택 (너무 다른 색상 제외)
    inlier_mask = (
        (h_diff_circular <= 2 * max(h_std, 15)) &  # H는 최소 15도 허용
        (s_diff <= 2 * max(s_std, 30)) &  # S는 최소 30 허용
        (v_diff <= 2 * max(v_std, 30))    # V는 최소 30 허용
    )
    
    inlier_pixels = pixels_hsv[inlier_mask]
    outlier_removed = original_count - len(inlier_pixels)
    
    # 2-4. 충분한 픽셀이 남아있으면 적용 (outlier가 30% 이하면 적용)
    if len(inlier_pixels) > original_count * 0.7:  # 70% 이상 남아있으면
        pixels_hsv = inlier_pixels
        if outlier_removed > 0:
            print(f"   🧹 Outlier 제거: {outlier_removed}개 픽셀 제거 ({original_count} → {len(pixels_hsv)})")
            print(f"      중앙값 HSV({h_median:.0f}, {s_median:.0f}, {v_median:.0f}) 기준")
    else:
        print(f"   ℹ️ Outlier 너무 많음 ({100-len(inlier_pixels)*100/original_count:.0f}%), 전체 사용")
    
    # 🎨 Step 3: 채도 높은 픽셀 우선 선택 (홀드 본래 색상)
    s_values_filtered = pixels_hsv[:, 1]
    saturation_threshold = np.percentile(s_values_filtered, 70)  # 상위 30%
    
    if saturation_threshold > 40:  # 채도가 충분히 높으면
        high_saturation_mask = s_values_filtered >= saturation_threshold
        high_sat_pixels = pixels_hsv[high_saturation_mask]
        
        if len(high_sat_pixels) > 10:
            pixels_hsv = high_sat_pixels
            print(f"   🎨 고채도 픽셀 선별: {len(pixels_hsv)}개 (채도≥{saturation_threshold:.0f})")
        else:
            print(f"   ⚠️ 고채도 픽셀 부족, 전체 사용")
    else:
        print(f"   ℹ️ 전체 채도 낮음 (무채색 홀드), 중앙값 사용")
    
    # 🎯 Step 4: 중앙값 방식으로 대표 색상 추출
    h_med = np.median(pixels_hsv[:, 0])
    s_med = np.median(pixels_hsv[:, 1])
    v_med = np.median(pixels_hsv[:, 2])
    
    print(f"   💎 최종 대표 색상: HSV({h_med:.1f}, {s_med:.1f}, {v_med:.1f})")
    
    return [int(h_med), int(s_med), int(v_med)]

# -------------------------------
# 📌 픽셀 기반 통계치 추출
# -------------------------------
def calculate_color_stats(image, mask, brightness_normalization=False, 
                          brightness_filter=False, min_brightness=0, max_brightness=100,
                          saturation_filter=False, min_saturation=0):
    """🚀 확장된 색상 통계 추출 - 다중 색상 공간 + 고급 특징 + 명도 정규화 옵션"""
    
    # 🔥 마스크 경계 제거 강화 (배경 픽셀 + 반사광 혼입 방지)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 더 큰 커널
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)  # 2번 반복
    
    # 마스크가 너무 작아지면 약하게 적용
    if np.sum(eroded_mask > 0) < 50:
        # 더 작은 커널로 재시도
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded_mask = cv2.erode(mask.astype(np.uint8), kernel_small, iterations=1)
        
        if np.sum(eroded_mask > 0) < 30:
            eroded_mask = mask
            print("   ⚠️ 마스크가 너무 작아 erosion 스킵")
        else:
            print(f"   ✂️ 마스크 경계 제거 (약함): {np.sum(mask > 0)} → {np.sum(eroded_mask > 0)} 픽셀")
    else:
        print(f"   ✂️ 마스크 경계 제거 (강함): {np.sum(mask > 0)} → {np.sum(eroded_mask > 0)} 픽셀")
    
    # 🔥 명도 보정: 규칙 기반도 원본 이미지 사용 (CLIP과 동일)
    # 색상 왜곡 방지를 위해 명도 보정 비활성화
    image_normalized = image  # 원본 이미지 사용
    
    # 🚫 명도 보정 비활성화됨 (원본 색상 보존)
    print("   ✨ 원본 이미지 사용 (명도 보정 없음)")
    
    if False:  # 명도 보정 코드 비활성화 (참고용으로 보존)
        hold_region = image[eroded_mask > 0.5]
        # LAB 색공간에서 명도 분석
        lab_region = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l_channel = lab_region[:, :, 0].copy()
        
        # 1️⃣ 전체 이미지의 명도 히스토그램 분석
        global_l_values = l_channel.flatten()
        global_l_mean = np.mean(global_l_values)
        global_l_median = np.median(global_l_values)
        global_l_p25 = np.percentile(global_l_values, 25)  # 하위 25%
        global_l_p75 = np.percentile(global_l_values, 75)  # 상위 25%
        
        # 2️⃣ 홀드 영역의 평균 명도
        hold_l_values = l_channel[eroded_mask > 0.5]
        hold_l_mean = np.mean(hold_l_values)
        hold_l_median = np.median(hold_l_values)
        hold_l_std = np.std(hold_l_values)
        
        # 3️⃣ 홀드의 상대적 밝기 백분위 계산
        hold_percentile = np.percentile(global_l_values, (hold_l_median / 255.0) * 100)
        hold_rank = np.sum(global_l_values < hold_l_median) / len(global_l_values) * 100
        
        print(f"   🌍 전체 이미지: 평균={global_l_mean:.1f}, 중앙={global_l_median:.1f}, 25%={global_l_p25:.1f}, 75%={global_l_p75:.1f}")
        print(f"   🎯 홀드: 평균={hold_l_mean:.1f}, 중앙={hold_l_median:.1f}, 백분위={hold_rank:.1f}%")
        
        # 4️⃣ 지각적 색상 보정: 백분위 기반 매핑
        l_channel_corrected = l_channel.copy()
        
        # 🔥 CLIP AI 사용 시 명도 보정 최소화 (원본 색상 보존)
        # 극단적인 경우만 약하게 보정
        if hold_rank > 90:
            # 상위 10% 밝기 → 약간만 밝게
            target_mean = min(200, hold_l_mean + 20)
            print(f"   ⚪ 지각적 판단: 매우 밝음 (상위 {hold_rank:.0f}%) → 약한 보정")
        elif hold_rank < 10:
            # 하위 10% 밝기 → 약간만 어둡게
            target_mean = max(60, hold_l_mean - 20)
            print(f"   ⚫ 지각적 판단: 매우 어두움 (하위 {hold_rank:.0f}%) → 약한 보정")
        else:
            # 나머지는 원본 유지
            target_mean = hold_l_mean
            print(f"   ✨ 원본 명도 유지: {hold_l_mean:.1f} (백분위 {hold_rank:.0f}%)")
        
        # 5️⃣ 명도 정규화 + 표준편차 축소
        mask_indices = eroded_mask > 0.5
        target_std = 30  # 표준편차 목표값 (더 타이트하게)
        
        l_channel_corrected[mask_indices] = np.clip(
            ((hold_l_values - hold_l_mean) / (hold_l_std + 1e-6)) * target_std + target_mean,
            0, 255
        ).astype(np.uint8)
        
        # 6️⃣ 정규화된 L 채널로 이미지 재구성
        lab_corrected = lab_region.copy()
        lab_corrected[:, :, 0] = l_channel_corrected
        image_normalized = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)
        
        print(f"   ✅ 지각적 보정 완료: {hold_l_mean:.1f} → {target_mean:.1f}, 표준편차 {hold_l_std:.1f} → {target_std}")
    else:
        image_normalized = image
    
    # 다중 색상 공간 변환 (정규화된 이미지 사용)
    hsv_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2HSV)
    rgb_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2Lab)
    yuv_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2YUV)
    xyz_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2XYZ)

    pixels_hsv = hsv_image[eroded_mask > 0.5]
    pixels_rgb = rgb_image[eroded_mask > 0.5]
    pixels_lab = lab_image[eroded_mask > 0.5]
    pixels_yuv = yuv_image[eroded_mask > 0.5]
    pixels_xyz = xyz_image[eroded_mask > 0.5]
    
    print(f"   📊 추출된 원본 픽셀: {len(pixels_hsv)}개")
    
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
            
            # 🔥 검정색은 유지! (기존 로직 제거)
            # RGB(0,0,0)이 검정색 홀드의 정확한 색상일 수 있음
            print(f"   ✅ HSV={dominant_hsv} → RGB={dominant_rgb}")
            
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

    # 🚀 캐싱된 YOLO 모델 사용 (속도 대폭 향상)
    model = get_yolo_model(model_path)
    results = model(padded_image, conf=conf)[0]

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
        
        # 🚀 배치 처리로 CLIP AI 색상 추출
        if valid_hold_images:
            print(f"🤖 CLIP AI 배치 처리 시작 ({len(valid_hold_images)}개 홀드)")
            batch_results = extract_colors_with_clip_ai_batch(valid_hold_images, valid_masks)
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
                
                # contour 단순화 (JSON 크기 최적화)
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                contour_points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                
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
                    "contour": contour_points,  # 세그먼테이션 윤곽선
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
            # 🔥 검정색 RGB(0,0,0)은 유효한 색상! (검정 홀드)
            if stats.get("dominant_rgb") == [0, 0, 0] and stats.get("dominant_hsv", [0, 0, 0])[2] > 60:
                # HSV의 V값이 60 이상인데 RGB가 (0,0,0)이면 변환 오류
                print(f"⚠️ 경고! 홀드 {i}: 변환 오류 감지 (HSV V={stats.get('dominant_hsv', [0, 0, 0])[2]} but RGB=0,0,0)")
                stats["dominant_rgb"] = [128, 128, 128]  # 회색으로 대체
                stats["dominant_hsv"] = [0, 0, 128]  # 회색 HSV
            elif stats.get("dominant_rgb") == [0, 0, 0]:
                print(f"⚫ 홀드 {i}: 검정색 감지! RGB(0,0,0) 유지")

            # contour 단순화 (JSON 크기 최적화)
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            contour_points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

            hold_data.append({
                "id": i,
                "center": [int(cx), int(cy)],
                "area": area,
                "circularity": circularity,
                "contour": contour_points,  # 세그먼테이션 윤곽선
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

    return hold_data, masks