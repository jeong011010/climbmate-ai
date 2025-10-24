import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import clip
from PIL import Image
import json
import os
from pathlib import Path

# 🚀 성능 최적화: 전역 캐시
_clip_model = None
_clip_text_features = None
_clip_device = None

# 🎨 룰 기반 색상 분류 캐시
_color_ranges_cache = None
_color_feedback_data = []

def hsv_to_rgb(hsv):
    hsv_arr = np.uint8([[hsv]])
    rgb = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
    return tuple(int(x) for x in rgb)

def transform_rgb_with_axis_weights(rgb):
    """
    🎯 RGB를 1:1:1 대각선 기준 3D 좌표계로 변환
    
    Z축: (0,0,0) → (255,255,255) 1:1:1 대각선 (조명/밝기) - 관대
    X,Y축: Z축을 수직으로 360도 반경 (순수 색상) - 엄격
    """
    r, g, b = rgb
    
    # 1. 1:1:1 대각선 방향 벡터 (정규화)
    diagonal_vector = np.array([1, 1, 1]) / np.sqrt(3)  # (1/√3, 1/√3, 1/√3)
    
    # 2. RGB 벡터
    rgb_vector = np.array([r, g, b])
    
    # 3. Z축 성분 (1:1:1 대각선에 투영) - 관대한 축
    z_component = np.dot(rgb_vector, diagonal_vector)
    
    # 4. 대각선에 수직인 평면으로 투영 (X,Y축 성분) - 엄격한 축
    diagonal_projection = z_component * diagonal_vector
    perpendicular_vector = rgb_vector - diagonal_projection
    
    # 5. 수직 벡터를 X,Y축으로 분해 (임의의 직교 좌표계)
    # X축: (1, -1, 0) 방향 성분
    x_axis = np.array([1, -1, 0]) / np.sqrt(2)
    x_component = np.dot(perpendicular_vector, x_axis)
    
    # Y축: (1, 1, -2) 방향 성분 (X축과 직교)
    y_axis = np.array([1, 1, -2]) / np.sqrt(6)
    y_component = np.dot(perpendicular_vector, y_axis)
    
    return np.array([x_component, y_component, z_component])

def clip_ai_color_clustering(hold_data, vectors, original_image, masks, eps=0.3, use_dbscan=False):
    """
    🤖 CLIP AI 기반 색상 클러스터링 (개선 버전)
    
    Args:
        hold_data: 홀드 데이터 (전처리 단계에서 이미 CLIP 특징 포함 가능)
        vectors: 특징 벡터 (사용 안 함)
        original_image: 원본 이미지
        masks: 홀드 마스크들
        eps: DBSCAN epsilon (CLIP 특징 벡터 거리 기준)
        use_dbscan: True면 DBSCAN으로 클러스터링, False면 직접 색상 매칭
    
    Returns:
        hold_data: 그룹 정보가 추가된 홀드 데이터
    """
    if len(hold_data) == 0:
        return hold_data
    
    print(f"\n🤖 CLIP AI 색상 클러스터링 시작")
    print(f"   홀드 개수: {len(hold_data)}개")
    print(f"   모드: {'DBSCAN (특징 벡터 기반)' if use_dbscan else '직접 색상 매칭'}")
    
    # CLIP 특징 벡터가 이미 있는지 확인
    has_clip_features = all("clip_features" in hold for hold in hold_data)
    
    if not has_clip_features:
        # 🚀 성능 최적화: 전역 CLIP 모델 캐시 사용
        global _clip_model, _clip_text_features, _clip_device
        
        if _clip_model is None:
            print("   🔄 CLIP 모델 로딩 중...")
            _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=_clip_device)
            _clip_model = (model, preprocess)
            print(f"   ✅ CLIP 모델 로딩 완료 (Device: {_clip_device})")
        else:
            print("   ✅ CLIP 모델 캐시 사용 (Device: {})".format(_clip_device))
            model, preprocess = _clip_model
        
        # 각 홀드의 이미지 특징 추출
        print("   🔍 CLIP 특징 벡터 추출 중...")
        for i, hold in enumerate(hold_data):
            mask = masks[hold["id"]].astype(np.uint8) * 255
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0:
                hold["clip_features"] = np.zeros(512).tolist()
                continue
                
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            hold_image = original_image[y_min:y_max+1, x_min:x_max+1]
            hold_pil = Image.fromarray(cv2.cvtColor(hold_image, cv2.COLOR_BGR2RGB))
            
            image_input = preprocess(hold_pil).unsqueeze(0).to(_clip_device)
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            
            hold["clip_features"] = image_feature.squeeze().cpu().numpy().tolist()
    
    # CLIP 특징 벡터 추출
    clip_features = np.array([hold["clip_features"] for hold in hold_data])
    
    if use_dbscan:
        # 🔷 모드 1: DBSCAN으로 자동 클러스터링
        print(f"   🎯 DBSCAN 클러스터링 (eps={eps})")
        from sklearn.metrics.pairwise import cosine_distances
        
        # 코사인 거리 계산
        distances = cosine_distances(clip_features)
        
        # DBSCAN 클러스터링
        dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        labels = dbscan.fit_predict(distances)
        
        # 그룹 할당
        for i, hold in enumerate(hold_data):
            hold["group"] = f"clip_g{labels[i]}"
        
        # 통계
        unique_labels = set(labels)
        print(f"\n✅ CLIP DBSCAN 클러스터링 완료")
        print(f"   그룹 개수: {len(unique_labels)}개")
        for label in sorted(unique_labels):
            count = np.sum(labels == label)
            print(f"   그룹 {label}: {count}개 홀드")
    else:
        # 🔷 모드 2: 색상 프롬프트 직접 매칭 + 검정색 강제 감지
        print("   🎨 색상 프롬프트 매칭 중...")
        
        # 🚀 성능 최적화: 전역 캐시 사용
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 🚀 모든 홀드를 한 번에 CLIP으로 색상 분석
        print("   🎨 모든 홀드를 한 번에 CLIP으로 색상 분석 중...")
        
        if _clip_model is None or _clip_device != device:
            print("   🔄 CLIP 모델 로딩 중...")
            model, preprocess = clip.load("ViT-B/32", device=device)
            _clip_model = (model, preprocess)
            _clip_device = device
            print("   ✅ CLIP 모델 로딩 완료")
        else:
            print("   ⚡ CLIP 모델 캐시 사용")
            model, preprocess = _clip_model
        
        # 🎯 모든 색상 프롬프트 (검정색 포함)
        color_prompts = [
            "a black climbing hold", "a very dark black climbing hold", "a dark black climbing hold",
            "a white climbing hold", "a bright white climbing hold", "a pure white climbing hold",
            "a gray climbing hold", "a light gray climbing hold", "a dark gray climbing hold",
            "an orange climbing hold", "a bright orange climbing hold", "a vivid orange climbing hold",
            "a yellow climbing hold", "a bright yellow climbing hold", "a pure yellow climbing hold",
            "a red climbing hold", "a bright red climbing hold", "a vivid red climbing hold",
            "a pink climbing hold", "a bright pink climbing hold", "a hot pink climbing hold",
            "a blue climbing hold", "a light blue climbing hold", "a sky blue climbing hold",
            "a green climbing hold", "a bright green climbing hold", "a forest green climbing hold",
            "a purple climbing hold", "a bright purple climbing hold", "a violet climbing hold",
            "a brown climbing hold", "a dark brown climbing hold", "a light brown climbing hold"
        ]
        
        color_map = {
            "black": ["black", "very dark black", "dark black"],
            "white": ["white", "bright white", "pure white"],
            "gray": ["gray", "light gray", "dark gray"],
            "orange": ["orange", "bright orange", "vivid orange"],
            "yellow": ["yellow", "bright yellow", "pure yellow"],
            "red": ["red", "bright red", "vivid red"],
            "pink": ["pink", "bright pink", "hot pink"],
            "blue": ["blue", "light blue", "sky blue"],
            "green": ["green", "bright green", "forest green"],
            "purple": ["purple", "bright purple", "violet"],
            "brown": ["brown", "dark brown", "light brown"]
        }
        
        # 텍스트 특징 추출
        if _clip_text_features is None:
            print("   📝 텍스트 특징 추출 중...")
            text_tokens = clip.tokenize(color_prompts).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            _clip_text_features = text_features
            print("   ✅ 텍스트 특징 추출 완료")
        else:
            print("   ⚡ 캐시된 텍스트 특징 사용")
            text_features = _clip_text_features
        
        # 모든 홀드의 CLIP 특징 추출
        print(f"   🖼️ {len(hold_data)}개 홀드의 CLIP 특징 추출 중...")
        batch_size = 32
        all_image_features = []
        valid_indices = []
        
        for batch_start in range(0, len(hold_data), batch_size):
            batch_end = min(batch_start + batch_size, len(hold_data))
            batch_holds = hold_data[batch_start:batch_end]
            
            processed_images = []
            batch_valid_indices = []
            
            for i, hold in enumerate(batch_holds):
                actual_idx = batch_start + i
                mask = masks[hold["id"]].astype(np.uint8) * 255
                y_coords, x_coords = np.where(mask > 0)
                if len(y_coords) == 0:
                    continue
                
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                hold_image = original_image[y_min:y_max+1, x_min:x_max+1]
                hold_pil = Image.fromarray(cv2.cvtColor(hold_image, cv2.COLOR_BGR2RGB))
                
                processed_images.append(preprocess(hold_pil))
                batch_valid_indices.append(actual_idx)
            
            if processed_images:
                image_tensor = torch.stack(processed_images).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                all_image_features.append(image_features.cpu().numpy())
                valid_indices.extend(batch_valid_indices)
        
        if not all_image_features:
            print("   ⚠️ 처리할 홀드가 없습니다.")
            return hold_data
        
        # 모든 이미지 특징 합치기
        all_image_features = np.vstack(all_image_features)
        print(f"   ✅ {len(all_image_features)}개 홀드의 CLIP 특징 추출 완료")
        
        # 유사도 계산 (모든 홀드)
        clip_features_tensor = torch.from_numpy(all_image_features).float().to(device)
        similarities = (clip_features_tensor @ text_features.T).cpu().numpy()
        
        # 색상 그룹 할당 (모든 홀드)
        color_groups = {}
        for i, orig_idx in enumerate(valid_indices):
            best_idx = np.argmax(similarities[i])
            best_prompt = color_prompts[best_idx]
            confidence = similarities[i][best_idx]
            
            # 색상 이름 추출
            color_name = "unknown"
            for color, keywords in color_map.items():
                for keyword in keywords:
                    if keyword in best_prompt:
                        color_name = color
                        break
                if color_name != "unknown":
                    break
            
            # 홀드에 색상 정보 저장
            hold_data[orig_idx]["color_name"] = color_name
            hold_data[orig_idx]["color_confidence"] = confidence
            
            # 그룹핑
            if color_name not in color_groups:
                color_groups[color_name] = []
            color_groups[color_name].append(orig_idx)
        
        # 색상별 그룹 ID 할당
        group_id = 0
        for color, indices in color_groups.items():
            for idx in indices:
                hold_data[idx]["group"] = group_id
            group_id += 1
        
        print(f"   ✅ 색상별 그룹핑 완료: {len(color_groups)}개 그룹")
        for color, indices in color_groups.items():
            print(f"   {color}: {len(indices)}개")
        
        return hold_data
        
        # 🤖 CLIP AI 개선: 모든 홀드에 대해 CLIP AI로 색상 판단
        print("   🤖 CLIP AI 색상 판단 개선 중...")
        
        # 🎯 대폭 확장된 색상 분류 체계 (색상당 5개 이상, 초록/노랑 명확히 구분)
        color_prompts = [
            # ⚫ 검정색 (8개)
            "a black climbing hold",
            "a very dark black climbing hold", 
            "a dark black climbing hold",
            "a charcoal black climbing hold",
            "a jet black climbing hold",
            "a pitch black climbing hold",
            "a coal black climbing hold",
            "a midnight black climbing hold",
            
            # ⚪ 흰색 (8개)
            "a white climbing hold",
            "a bright white climbing hold",
            "a pure white climbing hold",
            "a snow white climbing hold",
            "a pearl white climbing hold",
            "a clean white climbing hold",
            "a chalk white climbing hold",
            "a fresh white climbing hold",
            
            # 🔘 회색 (8개)
            "a gray climbing hold",
            "a light gray climbing hold",
            "a dark gray climbing hold",
            "a medium gray climbing hold",
            "a silver climbing hold",
            "a neutral gray climbing hold",
            "a slate gray climbing hold",
            "a stone gray climbing hold",
            
            # 🟠 주황색 (8개)
            "an orange climbing hold",
            "a bright orange climbing hold",
            "a vivid orange climbing hold",
            "a pumpkin orange climbing hold",
            "a tangerine orange climbing hold",
            "a flame orange climbing hold",
            "a traffic orange climbing hold",
            "a sunset orange climbing hold",
            
            # 🟡 노란색 (10개 - 초록과 명확히 구분)
            "a yellow climbing hold",
            "a bright yellow climbing hold",
            "a pure yellow climbing hold",
            "a lemon yellow climbing hold",
            "a golden yellow climbing hold",
            "a sunshine yellow climbing hold",
            "a canary yellow climbing hold",
            "a banana yellow climbing hold",
            "a mustard yellow climbing hold",
            "a highlighter yellow climbing hold",
            
            # 🔴 빨간색 (8개)
            "a red climbing hold",
            "a bright red climbing hold",
            "a vivid red climbing hold",
            "a cherry red climbing hold",
            "a crimson red climbing hold",
            "a scarlet red climbing hold",
            "a burgundy red climbing hold",
            "a wine red climbing hold",
            
            # 🩷 분홍색 (7개)
            "a pink climbing hold",
            "a bright pink climbing hold",
            "a hot pink climbing hold",
            "a rose pink climbing hold",
            "a coral pink climbing hold",
            "a fuchsia pink climbing hold",
            "a bubblegum pink climbing hold",
            
            # 🔵 파란색 (10개)
            "a blue climbing hold",
            "a bright blue climbing hold",
            "a pure blue climbing hold",
            "a light blue climbing hold",
            "a sky blue climbing hold",
            "a baby blue climbing hold",
            "a royal blue climbing hold",
            "a navy blue climbing hold",
            "a dark blue climbing hold",
            "a cobalt blue climbing hold",
            
            # 🟢 초록색 (12개 - 노랑과 명확히 구분, 다양한 톤)
            "a green climbing hold",
            "a pure green climbing hold",
            "a bright green climbing hold",
            "a vivid green climbing hold",
            "a grass green climbing hold",
            "a forest green climbing hold",
            "a dark green climbing hold",
            "an emerald green climbing hold",
            "a kelly green climbing hold",
            "a jade green climbing hold",
            "an olive green climbing hold",
            "a moss green climbing hold",
            
            # 💚 민트색 (새로 추가 - 6개)
            "a mint climbing hold",
            "a mint green climbing hold",
            "a light mint climbing hold",
            "a fresh mint climbing hold",
            "a turquoise mint climbing hold",
            "a pastel mint climbing hold",
            
            # 🍃 연두색 (새로 추가 - 6개, 노랑/초록 중간)
            "a lime climbing hold",
            "a lime green climbing hold",
            "a bright lime climbing hold",
            "a neon lime climbing hold",
            "a chartreuse climbing hold",
            "a yellow-green climbing hold",
            
            # 🟣 보라색 (7개)
            "a purple climbing hold",
            "a bright purple climbing hold",
            "a dark purple climbing hold",
            "a violet climbing hold",
            "a lavender climbing hold",
            "a lilac climbing hold",
            "a magenta climbing hold",
            
            # 🟤 갈색 (7개)
            "a brown climbing hold",
            "a dark brown climbing hold",
            "a light brown climbing hold",
            "a tan climbing hold",
            "a beige climbing hold",
            "a chocolate brown climbing hold",
            "a coffee brown climbing hold"
        ]
        
        # 🎯 확장된 색상 매핑 (민트/연두 추가, 초록/노랑 명확히 구분)
        color_map = {
            "black": ["black", "very dark black", "dark black", "charcoal", "jet black", "pitch black", 
                     "coal black", "midnight black"],
            
            "white": ["white", "bright white", "pure white", "snow white", "pearl white", "clean white",
                     "chalk white", "fresh white"],
            
            "gray": ["gray", "light gray", "dark gray", "medium gray", "silver", "neutral gray",
                    "slate", "stone"],
            
            "orange": ["orange", "bright orange", "vivid orange", "pumpkin", "tangerine", "flame",
                      "traffic", "sunset"],
            
            "yellow": ["yellow", "bright yellow", "pure yellow", "lemon", "golden", "sunshine", 
                      "canary", "banana", "mustard", "highlighter"],
            
            "red": ["red", "bright red", "vivid red", "cherry", "crimson", "scarlet", "burgundy", "wine"],
            
            "pink": ["pink", "bright pink", "hot pink", "rose pink", "coral pink", "fuchsia", "bubblegum"],
            
            "blue": ["blue", "bright blue", "pure blue", "light blue", "sky blue", "baby blue", 
                    "royal blue", "navy", "dark blue", "cobalt"],
            
            "green": ["green", "pure green", "bright green", "vivid green", "grass", "forest", 
                     "dark green", "emerald", "kelly", "jade", "olive", "moss"],
            
            "mint": ["mint", "mint green", "light mint", "fresh mint", "turquoise mint", "pastel mint"],
            
            "lime": ["lime", "lime green", "bright lime", "neon lime", "chartreuse", "yellow-green"],
            
            "purple": ["purple", "bright purple", "dark purple", "violet", "lavender", "lilac", "magenta"],
            
            "brown": ["brown", "dark brown", "light brown", "tan", "beige", "chocolate", "coffee"]
        }
        
        # 🤖 모든 홀드에 대해 CLIP AI 매칭 수행 (개선된 프롬프트 사용)
        # 🚀 성능 최적화: 텍스트 특징 캐싱
        if _clip_text_features is None:
            print("   🔄 텍스트 특징 추출 중...")
            text_tokens = clip.tokenize(color_prompts).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            _clip_text_features = text_features
            print("   ✅ 텍스트 특징 추출 완료")
        else:
            print("   ⚡ 캐시된 텍스트 특징 사용")
            text_features = _clip_text_features
        
        # 유사도 계산 (모든 홀드)
        clip_features_tensor = torch.from_numpy(clip_features).float().to(device)
        similarities = (clip_features_tensor @ text_features.T).cpu().numpy()
        
        # 색상 그룹 할당 (모든 홀드)
        color_groups = {}
        for i, hold in enumerate(hold_data):
            best_idx = np.argmax(similarities[i])
            best_prompt = color_prompts[best_idx]
            confidence = similarities[i][best_idx]
            
            # 색상 이름 추출 (프롬프트에서 키워드 매칭)
            color_name = "unknown"
            for color, keywords in color_map.items():
                for keyword in keywords:
                    if keyword in best_prompt:
                        color_name = color
                        break
                if color_name != "unknown":
                    break
            
            # 🎯 CLIP AI 결과 후처리 보정 (명확한 오류 수정 - 강화)
            rgb = hold.get("dominant_rgb", [128, 128, 128])
            if len(rgb) >= 3:
                r, g, b = rgb[0], rgb[1], rgb[2]
                avg_brightness = (r + g + b) / 3
                max_rgb = max(r, g, b)
                min_rgb = min(r, g, b)
                channel_diff = max_rgb - min_rgb
                
                # 🚨 Unknown 색상 강제 분류 (RGB 기반 - 민트/연두 추가)
                if color_name == "unknown":
                    print(f"   ⚠️ 홀드 {hold['id']} RGB{tuple(rgb)} - unknown 감지, RGB 기반 재분류 시도")
                    
                    # 민트색 체크 (G > R, B > R, G ≈ B, 밝음)
                    if g > r + 30 and b > r + 30 and abs(g - b) < 30 and avg_brightness > 150:
                        color_name = "mint"
                        confidence = 0.88
                        print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → mint (민트색)")
                    # 연두색 체크 (G가 가장 높고, R > B, R ≈ G)
                    elif g > b + 40 and r > b + 20 and abs(r - g) < 50:
                        color_name = "lime"
                        confidence = 0.88
                        print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → lime (연두색)")
                    # 초록색 체크 (G가 확실히 높음)
                    elif g > r + 30 and g > b + 30:
                        color_name = "green"
                        confidence = 0.90
                        print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → green (초록색)")
                    # 주황색 체크 (RGB(195,118,74) 같은 케이스)
                    elif r > g + 30 and r > b + 50 and g > b:
                        color_name = "orange"
                        confidence = 0.90
                        print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → orange (주황색)")
                    # 노란색 체크 (R ≈ G, 둘 다 B보다 높음)
                    elif r > b + 50 and g > b + 50 and abs(r - g) < 40:
                        color_name = "yellow"
                        confidence = 0.90
                        print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → yellow (노란색)")
                    # 파란색 체크
                    elif b > r + 20 and b > g + 20:
                        color_name = "blue"
                        confidence = 0.90
                        print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → blue (파란색)")
                    # 빨간색 체크
                    elif r > g + 30 and r > b + 30:
                        color_name = "red"
                        confidence = 0.90
                        print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → red (빨간색)")
                    # 무채색 체크
                    elif channel_diff < 15:
                        if avg_brightness > 200:
                            color_name = "white"
                            confidence = 0.95
                            print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → white (밝은 무채색)")
                        elif avg_brightness > 100:
                            color_name = "gray"
                            confidence = 0.95
                            print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → gray (중간 무채색)")
                        else:
                            color_name = "black"
                            confidence = 0.95
                            print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → black (어두운 무채색)")
                    # 갈색 체크 (어두운 주황색)
                    elif r > g + 10 and r > b + 20 and avg_brightness < 150:
                        color_name = "brown"
                        confidence = 0.85
                        print(f"   🔧 RGB 재분류: 홀드 {hold['id']} - unknown → brown (갈색)")
                    else:
                        # 최후의 수단: 가장 높은 채널 기준
                        if r > g and r > b:
                            color_name = "red"
                        elif g > r and g > b:
                            color_name = "green"
                        elif b > r and b > g:
                            color_name = "blue"
                        else:
                            color_name = "gray"
                        confidence = 0.70
                        print(f"   🔧 RGB 재분류 (최종): 홀드 {hold['id']} - unknown → {color_name} (최고 채널 기준)")
                
                # 🖤 검정색 보정: 매우 어두운 색상 (RGB(43,54,72) 같은 케이스)
                elif avg_brightness <= 70 and max_rgb <= 80:
                    if color_name != "black":
                        print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - {color_name} → black (매우 어두움)")
                        color_name = "black"
                        confidence = 0.99
                
                # ⚪ 무채색 보정 (RGB(194,199,198) 같은 케이스 - 채널 차이 < 15)
                elif channel_diff < 15:
                    # 밝은 흰색
                    if avg_brightness > 200:
                        if color_name not in ["white"]:
                            print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - {color_name} → white (밝은 무채색)")
                            color_name = "white"
                            confidence = 0.99
                    # 중간 밝기 회색 (RGB(194,199,198) 같은 케이스)
                    elif avg_brightness > 150:
                        if color_name not in ["white", "gray"]:
                            print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - {color_name} → gray (중간 무채색)")
                            color_name = "gray"
                            confidence = 0.99
                    # 어두운 회색
                    elif avg_brightness > 80:
                        if color_name not in ["gray", "black"]:
                            print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - {color_name} → gray (어두운 무채색)")
                            color_name = "gray"
                            confidence = 0.98
                    # 매우 어두운 검정
                    else:
                        if color_name not in ["black"]:
                            print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - {color_name} → black (어두운 무채색)")
                            color_name = "black"
                            confidence = 0.98
                
                # ⚪ 흰색 보정: 매우 밝은 색상 (평균 밝기 > 200, 채널 차이 < 30)
                elif avg_brightness > 200 and channel_diff < 30:
                    if color_name not in ["white", "gray"]:
                        print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - {color_name} → white (매우 밝음)")
                        color_name = "white"
                        confidence = 0.98
                
                # 🔵 밝은 파란색 보정: 파란색 채널이 높고 밝은 경우
                elif avg_brightness > 180 and b > r + 15 and b > g + 10:
                    if color_name not in ["blue", "white"]:
                        print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - {color_name} → blue (밝은 파란색)")
                        color_name = "blue"
                        confidence = 0.95
                
                # ⚪ 검정색으로 잘못 분류된 밝은 색상 보정
                elif color_name == "black" and avg_brightness > 150:
                    # 하늘색 체크 (RGB(184,223,237) 같은 케이스)
                    if b > r + 10 and b > g + 5:
                        print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - black → blue (하늘색)")
                        color_name = "blue"
                        confidence = 0.95
                    # 밝은 회색/흰색 체크 (RGB(202,199,187) 같은 케이스)
                    elif channel_diff < 30:
                        if avg_brightness > 190:
                            print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - black → white (밝은 흰색)")
                            color_name = "white"
                            confidence = 0.95
                        else:
                            print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - black → gray (밝은 회색)")
                            color_name = "gray"
                            confidence = 0.95
                    # 일반 밝은 색상
                    else:
                        print(f"   🔧 후처리 보정: 홀드 {hold['id']} RGB{tuple(rgb)} - black → white (밝은 색상)")
                        color_name = "white"
                        confidence = 0.95
            
            hold["group"] = f"ai_{color_name}"
            hold["clip_color_name"] = color_name
            hold["clip_confidence"] = float(confidence)
            
            if color_name not in color_groups:
                color_groups[color_name] = []
            color_groups[color_name].append(hold["id"])
            
            print(f"   홀드 {hold['id']}: {color_name} (신뢰도: {confidence:.3f}) - RGB{tuple(rgb)}")
        
        # 🔷 최종 그룹 정보 통합
        final_color_groups = {}
        for hold in hold_data:
            group_name = hold.get("group", "ai_unknown")
            color_name = group_name.replace("ai_", "")
            if color_name not in final_color_groups:
                final_color_groups[color_name] = []
            final_color_groups[color_name].append(hold["id"])
            
            confidence = hold.get("clip_confidence", 0.0)
            print(f"   홀드 {hold['id']}: {color_name} (신뢰도: {confidence:.3f})")
        
        print(f"\n✅ CLIP AI 클러스터링 완료")
        for color, hold_ids in sorted(final_color_groups.items()):
            print(f"   {color}: {len(hold_ids)}개 홀드")
    
    return hold_data

def lighting_invariant_dbscan_clustering(hold_data, vectors, eps=0.3, eps_black_gray=1.0, eps_white=1.0, eps_color=2.0):
    """
    🌟 축별 가중치 조명 불변 클러스터링
    
    xy축 (색상): 엄격한 eps 적용
    z축 (대각선/조명): 관대한 eps 적용
    """
    if len(hold_data) == 0:
        return hold_data
    
    print(f"\n🌟 축별 가중치 조명 불변 클러스터링 시작")
    print(f"   기본 eps: {eps}")
    print(f"   xy축(색상): 엄격한 eps, z축(조명): 관대한 eps")
    
    # HSV → RGB 변환
    rgb_values = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        rgb_values.append(rgb)
    
    rgb_values = np.array(rgb_values)
    
    # 축별 가중치 변환 적용
    transformed_rgb = np.array([transform_rgb_with_axis_weights(rgb) for rgb in rgb_values])
    
    print(f"   원본 RGB 샘플: {rgb_values[:3]}")
    print(f"   변환 RGB 샘플: {transformed_rgb[:3]}")
    
    # 축별 가중치를 적용한 거리 행렬 계산
    from sklearn.metrics.pairwise import euclidean_distances
    
    # 가중치 설정 (z축은 관대하게)
    weights = np.array([1.0, 1.0, 0.3])  # x, y, z 축 가중치
    
    # 가중치 적용된 거리 계산
    weighted_distances = euclidean_distances(transformed_rgb * weights)
    
    # DBSCAN with precomputed distances
    if len(hold_data) == 1:
        labels = np.array([0])
    else:
        dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        labels = dbscan.fit_predict(weighted_distances)
    
    # 그룹 할당 및 변환된 RGB 값 저장
    group_id = 0
    for i, label in enumerate(labels):
        if label == -1:
            hold_data[i]["group"] = f"g{group_id}"
            group_id += 1
        else:
            hold_data[i]["group"] = f"g{group_id + label}"
        
        # 변환된 RGB 값을 저장 (2D 시각화에서 사용)
        hold_data[i]["transformed_rgb"] = rgb_values[i].tolist()
    
    # 그룹 수 계산
    unique_labels = set(labels)
    if -1 in unique_labels:
        group_count = len(unique_labels) - 1  # -1 제외
    else:
        group_count = len(unique_labels)
    
    print(f"   생성된 그룹 수: {group_count}개")
    
    # 통계 출력
    groups = {}
    for hold in hold_data:
        g = hold["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(hold["id"])
    
    print(f"\n✅ 총 {len(groups)}개 그룹 생성")
    for g in sorted(groups.keys()):
        print(f"   {g}: {len(groups[g])}개 홀드")
    
    return hold_data

def create_clip_3d_visualization(hold_data, selected_hold_id=None, eps=None):
    """
    🤖 CLIP 특징 벡터 3D 시각화 (PCA로 차원 축소)
    """
    if len(hold_data) == 0:
        return None
    
    # CLIP 특징 벡터 추출
    clip_features = np.array([hold.get("clip_features", np.zeros(512)) for hold in hold_data])
    
    # PCA로 3D로 축소
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    clip_3d = pca.fit_transform(clip_features)
    
    # 그룹별 색상 매핑
    groups = {}
    for hold in hold_data:
        g = hold.get("group", "unknown")
        if g not in groups:
            groups[g] = []
        groups[g].append(hold["id"])
    
    # 색상 팔레트
    group_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]
    
    # 3D 플롯 생성
    fig = go.Figure()
    
    # 그룹별로 점 추가
    for i, group_id in enumerate(sorted(groups.keys())):
        group_hold_ids = groups[group_id]
        group_mask = [hold["id"] in group_hold_ids for hold in hold_data]
        group_coords = clip_3d[group_mask]
        group_color = group_colors[i % len(group_colors)]
        
        # 호버 텍스트 생성
        hover_texts = []
        for hold in [h for h, m in zip(hold_data, group_mask) if m]:
            color_name = hold.get("clip_color_name", "unknown")
            confidence = hold.get("clip_confidence", 0.0)
            hover_texts.append(
                f"홀드 ID: {hold['id']}<br>"
                f"그룹: {group_id}<br>"
                f"AI 색상: {color_name}<br>"
                f"신뢰도: {confidence:.3f}"
            )
        
        fig.add_trace(go.Scatter3d(
            x=group_coords[:, 0],
            y=group_coords[:, 1],
            z=group_coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color=group_color,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=[str(hold["id"]) for hold in [h for h, m in zip(hold_data, group_mask) if m]],
            textposition="middle center",
            name=f"그룹 {group_id}",
            hovertext=hover_texts,
            hoverinfo='text'
        ))
    
    # 선택된 홀드 강조
    if selected_hold_id is not None:
        for i, hold in enumerate(hold_data):
            if hold["id"] == selected_hold_id:
                fig.add_trace(go.Scatter3d(
                    x=[clip_3d[i, 0]],
                    y=[clip_3d[i, 1]],
                    z=[clip_3d[i, 2]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='yellow',
                        symbol='diamond',
                        line=dict(width=3, color='red')
                    ),
                    name=f"선택된 홀드 {selected_hold_id}",
                    showlegend=True
                ))
                break
    
    # 레이아웃 설정
    fig.update_layout(
        title="🤖 CLIP AI 특징 벡터 3D 공간 (PCA 투영)",
        scene=dict(
            xaxis_title="PC1 (주성분 1)",
            yaxis_title="PC2 (주성분 2)",
            zaxis_title="PC3 (주성분 3)"
        ),
        width=900,
        height=700,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def create_compressed_2d_visualization(hold_data, selected_hold_id=None, eps=None):
    """
    🎨 압축된 2D 분포도 시각화 (실제 클러스터링과 연결)
    각 점에 실제 색상과 그룹 정보 표시
    """
    if len(hold_data) == 0:
        return None
    
    # 클러스터링에서 저장된 변환된 RGB 값 사용
    rgb_values = []
    for hold in hold_data:
        if "transformed_rgb" in hold:
            # 클러스터링에서 저장된 변환된 RGB 값 사용
            rgb_values.append(hold["transformed_rgb"])
        else:
            # 백업: HSV → RGB 변환
            h, s, v = hold["dominant_hsv"]
            hsv_arr = np.uint8([[[h, s, v]]])
            rgb = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
            rgb_values.append(rgb)
    
    rgb_values = np.array(rgb_values)
    
    # 실제 클러스터링에서 사용된 변환 좌표 계산
    transformed_coords = np.array([transform_rgb_with_axis_weights(rgb) for rgb in rgb_values])
    # 2D 시각화를 위해 xy축만 사용
    compressed_coords = transformed_coords[:, :2]  # x, y 축만
    
    # 그룹별 색상 매핑
    groups = {}
    for hold in hold_data:
        g = hold["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(hold["id"])
    
    # 그룹별 색상 생성 (더 많은 색상)
    group_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2',
        '#A9DFBF', '#F9E79F', '#D5A6BD', '#AED6F1', '#A3E4D7'
    ]
    
    # 시각화 생성
    fig = go.Figure()
    
    # 그룹별로 점 추가
    for i, group_id in enumerate(sorted(groups.keys())):
        group_hold_ids = groups[group_id]
        group_mask = [hold["id"] in group_hold_ids for hold in hold_data]
        group_coords = compressed_coords[group_mask]
        group_rgb = rgb_values[group_mask]
        
        # 그룹 색상 선택 (순환)
        group_color = group_colors[i % len(group_colors)]
        
        # 각 홀드별로 실제 RGB 색상으로 점 추가
        for j, (coord, rgb, hold) in enumerate(zip(group_coords, group_rgb, 
                                                   [h for h, m in zip(hold_data, group_mask) if m])):
            fig.add_trace(go.Scatter(
                x=[coord[0]],
                y=[coord[1]],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=f'rgb({rgb[0]},{rgb[1]},{rgb[2]})',  # 실제 RGB 색상
                    line=dict(width=2, color='black'),
                    opacity=0.8
                ),
                text=[str(hold["id"])],
                textposition="middle center",
                name=f"그룹 {group_id}",
                showlegend=(j == 0),  # 그룹당 하나만 범례에 표시
                hovertemplate=f"홀드 ID: {hold['id']}<br>그룹: {group_id}<br>실제 RGB: ({rgb[0]},{rgb[1]},{rgb[2]})<br>변환 좌표: ({coord[0]:.1f}, {coord[1]:.1f})<br>xy축(색상): 엄격, z축(조명): 관대<extra></extra>"
            ))
    
    # 선택된 홀드 강조
    if selected_hold_id is not None:
        for i, hold in enumerate(hold_data):
            if hold["id"] == selected_hold_id:
                fig.add_trace(go.Scatter(
                    x=[compressed_coords[i, 0]],
                    y=[compressed_coords[i, 1]],
                    mode='markers',
                    marker=dict(
                        size=25,
                        color='yellow',
                        symbol='star',
                        line=dict(width=4, color='red')
                    ),
                    name=f"선택된 홀드 {selected_hold_id}",
                    showlegend=True
                ))
                break
    
    # eps 구 표시 (선택된 홀드가 있을 때)
    if selected_hold_id is not None and eps is not None:
        for i, hold in enumerate(hold_data):
            if hold["id"] == selected_hold_id:
                center_x, center_y = compressed_coords[i, 0], compressed_coords[i, 1]
                
                # eps 구를 원으로 표시
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = center_x + eps * np.cos(theta)
                circle_y = center_y + eps * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'eps={eps} 구',
                    showlegend=True
                ))
                break
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"🎨 1:1:1 대각선 기준 2D 분포도 (X,Y축: 엄격, Z축: 관대, eps={eps})",
        xaxis_title="X축: 1:1:1 대각선에 수직 성분 (엄격한 eps)",
        yaxis_title="Y축: 1:1:1 대각선에 수직 성분 (엄격한 eps)",
        width=900,
        height=700,
        showlegend=True,
        hovermode='closest'
    )
    
    # 격자 추가
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_gradient_background_heatmap(fig):
    """🎨 간단한 그라데이션 배경 생성 (사각형 방식)"""
    # 더 큰 간격으로 그라데이션 사각형 생성
    for v in range(0, 256, 16):  # Value: 16씩 간격
        for h in range(0, 180, 9):  # Hue: 9도씩 간격
            # HSV를 RGB로 변환 (Saturation=255로 고정하여 순수한 색상)
            hsv_color = np.array([[[h, 255, v]]], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            color_rgb = f"rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})"
            
            # 배경 사각형 추가
            fig.add_shape(
                type="rect",
                x0=v, y0=h, x1=v+16, y1=h+9,  # Value(가로) × Hue(세로)
                fillcolor=color_rgb,
                line=dict(width=0),
                layer="below"
            )

def recommend_holds(hold_data, vectors, clicked_id, top_n=5, alpha=0.7, beta=0.3, gamma=0.5):
    """홀드 추천 시스템"""
    # 클릭한 홀드 찾기
    clicked_hold = None
    clicked_vector = None
    for i, hold in enumerate(hold_data):
        if hold["id"] == clicked_id:
            clicked_hold = hold
            clicked_vector = vectors[i]
            break
    
    if clicked_hold is None or clicked_vector is None:
        return []
    
    recommendations = []
    
    for i, (hold, vector) in enumerate(zip(hold_data, vectors)):
        if hold["id"] == clicked_id:
            continue  # 자기 자신 제외
        
        # 코사인 유사도 계산
        cos_sim = cosine_similarity([clicked_vector], [vector])[0][0]
        
        # 유클리드 거리 계산
        euclid_dist = np.linalg.norm(clicked_vector - vector)
        
        # 그룹 게이팅: 같은 그룹이면 보너스
        group_bonus = 1.0
        if clicked_hold["group"] is not None and hold["group"] is not None:
            if clicked_hold["group"] == hold["group"]:
                group_bonus = 1.0  # 같은 그룹
            else:
                group_bonus = gamma  # 다른 그룹 패널티
        
        # 최종 점수 계산
        score = (alpha * cos_sim + beta * (1 - euclid_dist)) * group_bonus
        
        recommendations.append((hold["id"], score))
    
    # 점수순으로 정렬하고 상위 N개 반환
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

def build_feature_vectors(hold_data, scaler_option="none", use_illumination_invariant=True):
    """🚀 강화된 특징 벡터 생성 - 색상 + 공간 + 품질 특징"""
    vectors = []
    ids = []
    
    # 이미지 크기 계산 (정규화용)
    centers = [hold["center"] for hold in hold_data]
    areas = [hold.get("area", hold.get("size", 1)) for hold in hold_data]
    
    if centers:
        max_x = max(c[0] for c in centers)
        max_y = max(c[1] for c in centers)
        max_area = max(areas) if areas else 1
    else:
        max_x = max_y = max_area = 1
    
    for hold in hold_data:
        vec_components = []
        
        if use_illumination_invariant and "advanced" in hold:
            # 조명 불변 특징 (핵심)
            advanced = hold["advanced"]
            
            # 1. Lab a*, b* (조명에 가장 덜 민감)
            if "lab_ab" in advanced:
                lab_a, lab_b = advanced["lab_ab"]
                vec_components.extend([
                    (lab_a - 128) / 128.0,  # 정규화
                    (lab_b - 128) / 128.0
                ])
            
            # 2. Hue, Saturation (Value 제외)
            if "hue_sat" in advanced:
                h, s = advanced["hue_sat"]
                h_rad = np.deg2rad(h / 180.0 * 360.0)
                vec_components.extend([
                    np.cos(h_rad),
                    np.sin(h_rad),
                    s / 255.0
                ])
            
            # 3. 색상 순도
            if "color_purity" in advanced:
                vec_components.append(advanced["color_purity"])
            
            # 4. 색상 균일성 (낮을수록 균일한 색상)
            if "color_uniformity" in advanced:
                vec_components.append(advanced["color_uniformity"])
            
        else:
            # 기본 HSV 특징 (하위 호환성)
            h, s, v = hold["dominant_hsv"]
            h_rad = np.deg2rad(h / 180.0 * 360.0)
            vec_components = [np.cos(h_rad), np.sin(h_rad), s / 255.0, v / 255.0]
        
        # 🚀 공간적 특징 추가
        cx, cy = hold["center"]
        area = hold.get("area", hold.get("size", 1))
        
        spatial_features = [
            cx / max_x,  # 정규화된 X 위치
            cy / max_y,  # 정규화된 Y 위치
            area / max_area,  # 정규화된 크기
            hold.get("circularity", 0.5)  # 원형도 (0~1)
        ]
        vec_components.extend(spatial_features)
        
        vectors.append(np.array(vec_components))
        ids.append(hold["id"])
    
    vectors = np.array(vectors)

    if scaler_option == "standard":
        vectors = (vectors - np.mean(vectors, axis=0)) / (np.std(vectors, axis=0) + 1e-8)
    elif scaler_option == "minmax":
        vectors = (vectors - np.min(vectors, axis=0)) / (np.ptp(vectors, axis=0) + 1e-8)
    
    return vectors, ids

def color_specific_classification(hold_data):
    """🚨 색상별 특화 분류 시스템"""
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        
        # 검정색/흰색 특별 처리 제거 - RGB 큐브에서 순수 거리 기반으로 처리
        
        # 회색 홀드 (Saturation 낮음, Value 중간)
        if s < 40 and 30 <= v <= 200:
            hold["color_category"] = "gray"
            hold["group"] = None  # 나중에 회색 전용 군집화
        # 4. 빨간색 계열 (Hue 0-30도, 330-360도)
        elif (0 <= h <= 30) or (330 <= h <= 360):
            if s > 100 and v > 100:  # 진한 빨간색
                hold["color_category"] = "red"
            elif s > 80 and v > 150:  # 핑크색
                hold["color_category"] = "pink"
            else:
                hold["color_category"] = "red_light"
            hold["group"] = None
        # 5. 주황색 계열 (Hue 15-45도)
        elif 15 <= h <= 45:
            hold["color_category"] = "orange"
            hold["group"] = None
        # 6. 노란색 계열 (Hue 45-75도)
        elif 45 <= h <= 75:
            hold["color_category"] = "yellow"
            hold["group"] = None
        # 7. 연두색 계열 (Hue 60-90도)
        elif 60 <= h <= 90:
            hold["color_category"] = "lime_green"
            hold["group"] = None
        # 8. 초록색 계열 (Hue 75-165도)
        elif 75 <= h <= 165:
            hold["color_category"] = "green"
            hold["group"] = None
        # 9. 청록색 계열 (Hue 150-180도)
        elif 150 <= h <= 180:
            hold["color_category"] = "cyan"
            hold["group"] = None
        # 10. 파란색 계열 (Hue 180-240도)
        elif 180 <= h <= 240:
            if h <= 210:  # 하늘색
                hold["color_category"] = "sky_blue"
            else:  # 진한 파란색
                hold["color_category"] = "blue"
            hold["group"] = None
        # 11. 남색 계열 (Hue 240-270도)
        elif 240 <= h <= 270:
            hold["color_category"] = "navy_blue"
            hold["group"] = None
        # 12. 보라색 계열 (Hue 270-330도)
        elif 270 <= h <= 330:
            hold["color_category"] = "purple"
            hold["group"] = None
        # 13. 기타
        else:
            hold["color_category"] = "other"
            hold["group"] = None
    
    return hold_data

def cluster_by_color_category(hold_data, vectors, eps, min_samples, method):
    """🚨 색상 카테고리별 군집화"""
    categories = set(h["color_category"] for h in hold_data)
    current_group_id = 0
    
    for category in categories:
        category_holds = [h for h in hold_data if h["color_category"] == category]
        
        if len(category_holds) <= 1:
            # 홀드가 1개뿐이면 그룹 ID만 할당
            category_holds[0]["group"] = current_group_id
            current_group_id += 1
            continue
        
        # 해당 카테고리의 홀드 인덱스 찾기
        category_indices = [i for i, h in enumerate(hold_data) if h["color_category"] == category]
        category_vectors = vectors[category_indices]
        
        # 🚨 샘플 수에 따른 안전한 군집화
        if len(category_holds) < 2:
            # 홀드가 1개뿐이면 그룹 ID만 할당
            category_holds[0]["group"] = current_group_id
            current_group_id += 1
            continue
        elif len(category_holds) < 4:
            # 홀드가 2-3개뿐이면 각각 별도 그룹으로 처리 (군집화 불필요)
            for i, hold in enumerate(category_holds):
                hold["group"] = current_group_id + i
            current_group_id += len(category_holds)
            continue
        
        # 카테고리별 특화 eps 사용
        category_eps = get_category_specific_eps(category, eps)
        
        if method == "ensemble":
            labels = safe_ensemble_clustering(category_holds, category_vectors, base_eps=category_eps)
        else:
            labels = cosine_dbscan(category_vectors, eps=category_eps, min_samples=min_samples)
        
        # 그룹 ID 할당 (카테고리별로 독립적인 그룹 ID)
        for i, hold in enumerate(category_holds):
            if labels[i] != -1:
                hold["group"] = current_group_id + labels[i]
            else:
                hold["group"] = current_group_id + i  # 노이즈도 별도 그룹
        
        current_group_id += max(labels) + 1 if len(labels) > 0 else 1
    
    return hold_data

def get_category_specific_eps(category, base_eps):
    """🚨 카테고리별 특화 eps 설정"""
    eps_multipliers = {
        "black": 0.1,      # 검정색은 매우 엄격
        "white": 0.1,      # 흰색도 매우 엄격
        "gray": 0.2,       # 회색은 엄격
        "red": 0.3,        # 빨간색은 중간
        "pink": 0.5,       # 핑크는 좀 더 느슨 (빨강과 구분)
        "orange": 0.4,     # 주황색
        "yellow": 0.6,     # 노란색은 더 느슨 (연두와 구분)
        "lime_green": 0.7, # 연두색 (노랑과 구분)
        "green": 0.4,      # 초록색
        "cyan": 0.5,       # 청록색
        "blue": 0.4,       # 파란색
        "sky_blue": 0.6,   # 하늘색 (파랑과 구분)
        "navy_blue": 0.5,  # 남색 (파랑과 구분)
        "purple": 0.4,     # 보라색
        "other": 0.5       # 기타
    }
    
    return base_eps * eps_multipliers.get(category, 0.5)

def simple_rgb_clustering(hold_data, vectors, eps, min_samples):
    """🚨 3D RGB 공간에서 단순한 거리 기반 클러스터링"""
    from sklearn.cluster import DBSCAN
    
    # RGB 특징 벡터 추출 (vectors의 첫 3개 차원이 RGB)
    rgb_vectors = vectors[:, :3]  # R, G, B 값만 사용
    
    # 🚨 여러 eps 값으로 시도하여 최적의 클러스터링 찾기
    best_labels = None
    best_score = -1
    best_eps = eps
    
    # eps 값을 5~50 사이에서 테스트 (더 넓은 범위)
    for test_eps in range(5, min(60, eps * 2), 1):
        clustering = DBSCAN(eps=test_eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(rgb_vectors)
        
        # 클러스터 개수와 노이즈 비율로 점수 계산
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # 이상적인 클러스터 수: 6-10개, 노이즈 최소화 (3D 그래프의 클러스터 수에 맞춤)
        if 6 <= n_clusters <= 10 and n_noise < len(hold_data) * 0.3:
            score = n_clusters - (n_noise / len(hold_data))  # 클러스터 많을수록, 노이즈 적을수록 좋음
            if score > best_score:
                best_score = score
                best_labels = labels
                best_eps = test_eps
    
    # 최적 결과가 없으면 강제로 7-8개 클러스터 생성
    if best_labels is None:
        # 🚨 강제 클러스터링: 매우 작은 eps로 시작해서 클러스터 수 맞출 때까지 증가
        target_clusters = 7  # 3D 그래프의 클러스터 수
        best_labels = None
        
        for force_eps in range(5, 30, 1):
            clustering = DBSCAN(eps=force_eps, min_samples=min_samples, metric='euclidean')
            labels = clustering.fit_predict(rgb_vectors)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # 6-8개 클러스터가 나오면 사용
            if 6 <= n_clusters <= 8:
                best_labels = labels
                break
        
        # 그래도 안되면 기본값 사용
        if best_labels is None:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            best_labels = clustering.fit_predict(rgb_vectors)
    
    # 그룹 ID 할당
    for i, hold in enumerate(hold_data):
        hold["group"] = int(best_labels[i]) if best_labels[i] != -1 else -1
    
    return hold_data

def ultra_simple_rgb_clustering(hold_data, vectors, eps=1.0, min_samples=1, n_clusters=7, method="ensemble"):
    """🎯 사용자 정의 파라미터 기반 3D XYZ 직선 거리 클러스터링"""
    from sklearn.cluster import DBSCAN, KMeans
    import streamlit as st
    import numpy as np
    
    # RGB 특징 벡터 추출 (R, G, B 값만)
    rgb_vectors = vectors[:, :3]
    
    # 🚨 디버깅: RGB 벡터 정보 출력
    st.write(f"🔍 **사용자 정의 클러스터링 디버깅:**")
    st.write(f"- 홀드 수: {len(hold_data)}")
    st.write(f"- RGB 벡터 형태: {rgb_vectors.shape}")
    st.write(f"- RGB 값 범위: R({rgb_vectors[:, 0].min():.1f}-{rgb_vectors[:, 0].max():.1f}), G({rgb_vectors[:, 1].min():.1f}-{rgb_vectors[:, 1].max():.1f}), B({rgb_vectors[:, 2].min():.1f}-{rgb_vectors[:, 2].max():.1f})")
    
    # 🚨 실제 거리 계산해서 문제점 분석
    if len(rgb_vectors) > 1:
        st.write(f"🔍 **실제 거리 분석:**")
        distances = []
        for i in range(len(rgb_vectors)):
            for j in range(i+1, len(rgb_vectors)):
                dist = np.sqrt(np.sum((rgb_vectors[i] - rgb_vectors[j])**2))
                distances.append(dist)
        
        if distances:
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = np.mean(distances)
            st.write(f"- 최소 거리: {min_dist:.1f}")
            st.write(f"- 최대 거리: {max_dist:.1f}")
            st.write(f"- 평균 거리: {avg_dist:.1f}")
            
            # eps와 비교
            if 'eps' in locals():
                st.write(f"- 설정된 eps: {eps}")
                close_pairs = [d for d in distances if d <= eps]
                far_pairs = [d for d in distances if d > eps]
                st.write(f"- eps 이하 거리 쌍: {len(close_pairs)}개")
                st.write(f"- eps 초과 거리 쌍: {len(far_pairs)}개")
    
    best_labels = None
    
    # 🚨 전달된 파라미터 확인
    st.write(f"🔍 **전달된 파라미터 확인:**")
    st.write(f"- method: '{method}'")
    st.write(f"- eps: {eps} (실제 사용값)")
    st.write(f"- min_samples: {min_samples}")
    st.write(f"- n_clusters: {n_clusters}")
    
    # 사용자가 선택한 방법에 따라 클러스터링 실행
    if method == "DBSCAN (eps 조절)":
        st.write(f"- **DBSCAN 클러스터링**")
        st.write(f"- eps: {eps}")
        st.write(f"- min_samples: {min_samples}")
        
        if len(rgb_vectors) < min_samples:
            st.warning(f"⚠️ 홀드 수가 min_samples({min_samples})보다 적습니다. 모든 홀드를 노이즈로 처리합니다.")
            best_labels = np.full(len(rgb_vectors), -1)
        else:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            best_labels = clustering.fit_predict(rgb_vectors)
            
            # 🚨 DBSCAN 상세 분석
            core_samples = clustering.core_sample_indices_
            n_core = len(core_samples)
            n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise = list(best_labels).count(-1)
            
            st.write(f"🔍 **DBSCAN 상세 분석:**")
            st.write(f"- 핵심점(Core Points): {n_core}개")
            st.write(f"- 클러스터: {n_clusters}개")
            st.write(f"- 노이즈: {n_noise}개")
            
            # 각 클러스터별 점의 개수
            cluster_counts = {}
            for label in best_labels:
                if label != -1:
                    cluster_counts[label] = cluster_counts.get(label, 0) + 1
            
            st.write(f"- 클러스터별 점 개수: {cluster_counts}")
            
            # 문제점 분석
            if n_noise > len(rgb_vectors) * 0.5:
                st.warning(f"⚠️ 노이즈가 너무 많습니다 ({n_noise}/{len(rgb_vectors)}) - eps를 늘려보세요")
            elif n_clusters == 1 and len(rgb_vectors) > 5:
                st.warning(f"⚠️ 클러스터가 1개뿐입니다 - eps를 줄여보세요")
            elif n_clusters > len(rgb_vectors) * 0.5:
                st.warning(f"⚠️ 클러스터가 너무 많습니다 ({n_clusters}개) - eps를 늘려보세요")
            
    elif method == "K-Means (k값 조절)":
        st.write(f"- **K-Means 클러스터링**")
        st.write(f"- k (클러스터 수): {n_clusters}")
        
        if len(rgb_vectors) < n_clusters:
            st.warning(f"⚠️ 홀드 수가 k값({n_clusters})보다 적습니다. 모든 홀드를 하나의 그룹으로 처리합니다.")
            best_labels = np.zeros(len(rgb_vectors), dtype=int)
        elif len(rgb_vectors) == 0:
            best_labels = np.array([])
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            best_labels = kmeans.fit_predict(rgb_vectors)
            
    else:  # 기존 방법들 (자동 eps 찾기)
        st.write(f"- **자동 eps 찾기 클러스터링**")
        
        # RGB 값들의 실제 거리를 계산해서 적절한 eps 찾기
        distances = []
        for i in range(len(rgb_vectors)):
            for j in range(i+1, len(rgb_vectors)):
                dist = np.sqrt(np.sum((rgb_vectors[i] - rgb_vectors[j])**2))
                distances.append(dist)
        
        if distances:
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = np.mean(distances)
            st.write(f"- 최소 거리: {min_dist:.1f}")
            st.write(f"- 최대 거리: {max_dist:.1f}")
            st.write(f"- 평균 거리: {avg_dist:.1f}")
            
            eps_candidates = np.linspace(min_dist, avg_dist/2, 20)
            best_score = -1
            
            for test_eps in eps_candidates:
                clustering = DBSCAN(eps=test_eps, min_samples=min_samples, metric='euclidean')
                labels = clustering.fit_predict(rgb_vectors)
                
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters_found >= 2 and n_clusters_found <= 15:
                    noise_ratio = n_noise / len(hold_data)
                    score = (15 - n_clusters_found) * 0.7 + (1 - noise_ratio) * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        eps = test_eps
                        
            if best_labels is None:
                st.write("⚠️ **기본 eps=10.0 사용**")
                clustering = DBSCAN(eps=10.0, min_samples=min_samples, metric='euclidean')
                best_labels = clustering.fit_predict(rgb_vectors)
                eps = 10.0
    
    # 🚨 최종 결과 출력
    if best_labels is not None and len(best_labels) > 0:
        final_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        final_noise = list(best_labels).count(-1)
        st.write(f"🎯 **최종 클러스터링 결과:**")
        st.write(f"- 사용된 파라미터: eps={eps:.1f}" if method == "DBSCAN (eps 조절)" else f"- 사용된 파라미터: k={n_clusters}")
        st.write(f"- 생성된 클러스터: {final_clusters}개")
        st.write(f"- 노이즈: {final_noise}개")
        st.write(f"- 그룹 ID들: {sorted(set(best_labels))}")
        
        # 그룹 ID 할당 (노이즈도 별도 그룹으로 처리)
        for i, hold in enumerate(hold_data):
            if best_labels[i] != -1:
                hold["group"] = int(best_labels[i])
            else:
                hold["group"] = int(len(set(best_labels)) + i)  # 노이즈도 별도 그룹
    else:
        st.write("⚠️ 클러스터링할 홀드가 없습니다.")
    
    return hold_data

def pre_classify_bw(hold_data, v_thresh=40, s_thresh=30, v_high=180):
    """🚀 강화된 검정/흰색 전처리 - 더 엄격한 기준"""
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        
        # 검정색: Value가 매우 낮고 Saturation도 낮음
        if v < v_thresh and s < s_thresh:
            hold["group"] = -2  # 검정색 그룹
        # 흰색: Value가 매우 높고 Saturation이 매우 낮음  
        elif s < s_thresh and v > v_high:
            hold["group"] = -3  # 흰색 그룹
        # 회색: Value 중간, Saturation 낮음
        elif s < s_thresh and v_thresh <= v <= v_high:
            hold["group"] = -4  # 회색 그룹
        else:
            hold["group"] = None  # 일반 색상 그룹
    return hold_data

def enhanced_color_preprocessing(hold_data):
    """🚀 강화된 색상 전처리 - 더 세밀한 색상 분류"""
    # 1. 기본 검정/흰색/회색 분류
    hold_data = pre_classify_bw(hold_data)
    
    # 2. 색상 순도 기반 추가 분류
    for hold in hold_data:
        if hold["group"] is None:  # 아직 분류되지 않은 홀드만
            h, s, v = hold["dominant_hsv"]
            
            # 🚀 색상 순도 계산 (HSV에서)
            # Saturation과 Value가 모두 높으면 순수한 색상
            color_purity = (s / 255.0) * (v / 255.0)
            
            # 🚀 저채도 색상 분류 (회색 계열)
            if s < 50:  # 매우 낮은 채도
                if v < 80:
                    hold["group"] = -5  # 어두운 회색
                elif v > 180:
                    hold["group"] = -6  # 밝은 회색
                else:
                    hold["group"] = -7  # 중간 회색
            
            # 🚀 색상 순도가 낮은 경우 (혼합색)
            elif color_purity < 0.3:  # 채도나 명도가 낮은 경우
                hold["group"] = -8  # 저순도 색상 그룹
    
    return hold_data

def ultra_strict_color_separation(hold_data, hue_thresh=8, sat_thresh=20, val_thresh=40):
    """🚀 초엄격 색상 분리 - 매우 세밀한 기준"""
    # 먼저 강화된 전처리 적용
    hold_data = enhanced_color_preprocessing(hold_data)
    
    # 일반 색상 그룹에 대해 더 엄격한 분리
    for group in set(h["group"] for h in hold_data if h["group"] is not None and h["group"] >= 0):
        group_holds = [h for h in hold_data if h["group"] == group]
        if len(group_holds) <= 1:
            continue
            
        hsv_values = [h["dominant_hsv"] for h in group_holds]
        hue_values = [hsv[0] for hsv in hsv_values]
        sat_values = [hsv[1] for hsv in hsv_values]
        val_values = [hsv[2] for hsv in hsv_values]
        
        # 🚀 초엄격 분리 기준
        max_hue_diff = max(hue_values) - min(hue_values)
        max_sat_diff = max(sat_values) - min(sat_values)
        max_val_diff = max(val_values) - min(val_values)
        
        should_separate = False
        separation_reason = ""
        
        # Hue 차이가 8도 이상이면 분리 (매우 엄격)
        if max_hue_diff > hue_thresh:
            should_separate = True
            separation_reason = f"Hue 차이 {max_hue_diff:.1f}도 (임계값: {hue_thresh})"
        # Saturation 차이가 20 이상이면 분리 (매우 엄격)
        elif max_sat_diff > sat_thresh:
            should_separate = True
            separation_reason = f"Saturation 차이 {max_sat_diff:.1f} (임계값: {sat_thresh})"
        # Value 차이가 40 이상이면 분리 (매우 엄격)
        elif max_val_diff > val_thresh:
            should_separate = True
            separation_reason = f"Value 차이 {max_val_diff:.1f} (임계값: {val_thresh})"
        
        # 홀드가 2개 이상이면 분리 (더 관대한 조건)
        if should_separate and len(group_holds) >= 2:
            # K-means로 2개 그룹으로 분리
            from sklearn.cluster import KMeans
            features = [[hsv[0], hsv[1], hsv[2]] for hsv in hsv_values]
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(features)
            
            new_group = max(h["group"] for h in hold_data if h["group"] is not None) + 1
            for i, hold in enumerate(group_holds):
                if sub_labels[i] == 1:
                    hold["group"] = new_group
            
            print(f"초엄격 분리: 그룹 {group} 분리됨 - {separation_reason}")
    
    return hold_data

def advanced_distance_matrix(vectors, hold_data):
    """🚀 초강화된 거리 함수 - HSV 기반 정교한 색상 거리 + 공간 + 품질"""
    n = len(vectors)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # 1. HSV 기반 정교한 색상 거리 (가중치 0.7로 증가)
            color_dist = calculate_hsv_distance(hold_data[i]["dominant_hsv"], hold_data[j]["dominant_hsv"])
            
            # 2. 공간적 거리 (가중치 0.15로 감소)
            center_i = hold_data[i]["center"]
            center_j = hold_data[j]["center"]
            spatial_dist = np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
            spatial_dist = min(spatial_dist / 300.0, 1.0)  # 정규화 범위 증가
            
            # 3. 크기 유사도 (가중치 0.1)
            area_i = hold_data[i].get("area", hold_data[i].get("size", 1))
            area_j = hold_data[j].get("area", hold_data[j].get("size", 1))
            size_sim = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 1
            size_dist = 1 - size_sim
            
            # 4. 원형도 유사도 (가중치 0.05로 감소)
            circ_i = hold_data[i].get("circularity", 0.5)
            circ_j = hold_data[j].get("circularity", 0.5)
            circ_dist = abs(circ_i - circ_j)
            
            # 🚀 가중 평균 거리 (색상에 더 높은 가중치)
            total_dist = (0.7 * color_dist + 0.15 * spatial_dist + 
                         0.1 * size_dist + 0.05 * circ_dist)
            
            dist_matrix[i][j] = dist_matrix[j][i] = total_dist
    
    return dist_matrix

def calculate_hsv_distance(hsv1, hsv2):
    """🎯 HSV 기반 정교한 색상 거리 계산"""
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2
    
    # 1. Hue 거리 (원형 거리 고려)
    hue_diff = min(abs(h1 - h2), 179 - abs(h1 - h2))  # OpenCV HSV는 0-179
    hue_distance = hue_diff / 179.0  # 정규화 (0-1)
    
    # 2. Saturation 거리
    sat_distance = abs(s1 - s2) / 255.0  # 정규화 (0-1)
    
    # 3. Value 거리
    val_distance = abs(v1 - v2) / 255.0  # 정규화 (0-1)
    
    # 🚀 가중 평균 거리 (Hue에 가장 높은 가중치)
    # Hue가 가장 중요한 색상 특성이므로
    total_distance = (0.6 * hue_distance + 0.25 * sat_distance + 0.15 * val_distance)
    
    return total_distance

def cosine_dbscan(vectors, eps=0.01, min_samples=1):
    """원래 잘 작동하던 코사인 DBSCAN"""
    sim_matrix = cosine_similarity(vectors)
    dist_matrix = 1 - sim_matrix
    dist_matrix = np.clip(dist_matrix, 0, 1)
    clustering = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(dist_matrix)
    return labels

def advanced_dbscan(vectors, hold_data, eps=0.01, min_samples=1):
    """🚀 개선된 DBSCAN - 종합 거리 함수 사용"""
    dist_matrix = advanced_distance_matrix(vectors, hold_data)
    clustering = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(dist_matrix)
    return labels

def hierarchical_clustering(hold_data, vectors, base_eps=0.01):
    """계층적 클러스터링: Hue 대분류 → 세부 분류"""
    from sklearn.cluster import KMeans
    
    # 🚨 샘플 수 체크
    if len(hold_data) < 2:
        return [0] * len(hold_data)
    
    # 1단계: Hue로 대분류 (빨강/노랑/초록/파랑 등)
    hue_values = np.array([hold["dominant_hsv"][0] for hold in hold_data])
    
    # Hue를 원형 좌표로 변환
    hue_rad = np.deg2rad(hue_values / 180.0 * 360.0)
    hue_coords = np.column_stack([np.cos(hue_rad), np.sin(hue_rad)])
    
    # 🚨 안전한 K-means 클러스터 수 결정
    max_clusters = min(8, len(hold_data), len(hold_data) - 1)  # 샘플 수보다 작게
    best_n_clusters = min(3, len(hold_data))  # 최소 3개 또는 샘플 수
    
    if len(hold_data) >= 3:
        try:
            from sklearn.metrics import silhouette_score
            best_score = -1
            for n in range(2, max_clusters + 1):  # 최소 2개 클러스터
                if n < len(hold_data):  # 클러스터 수가 샘플 수보다 작아야 함
                    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                    hue_labels = kmeans.fit_predict(hue_coords)
                    if len(set(hue_labels)) > 1:
                        score = silhouette_score(hue_coords, hue_labels)
                        if score > best_score:
                            best_score = score
                            best_n_clusters = n
        except:
            best_n_clusters = min(2, len(hold_data))
    else:
        best_n_clusters = len(hold_data)  # 샘플 수와 같게
    
    # 🚨 안전한 K-means 실행
    if best_n_clusters >= len(hold_data):
        # 클러스터 수가 샘플 수와 같거나 크면 각각 별도 그룹
        return list(range(len(hold_data)))
    
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    hue_groups = kmeans.fit_predict(hue_coords)
    
    # 2단계: 각 Hue 그룹 내에서 DBSCAN으로 세부 분류
    final_labels = np.full(len(hold_data), -1)
    label_counter = 0
    
    for hue_group_id in range(best_n_clusters):
        group_indices = np.where(hue_groups == hue_group_id)[0]
        if len(group_indices) == 0:
            continue
        
        group_vectors = vectors[group_indices]
        
        # Hue 그룹의 분산에 따라 eps 조정
        hue_variance = np.var([hold_data[i]["dominant_hsv"][0] for i in group_indices])
        adaptive_eps = base_eps * (1 + hue_variance / 100.0)  # 분산이 크면 eps 증가
        adaptive_eps = np.clip(adaptive_eps, 0.005, 0.05)
        
        # DBSCAN 적용
        group_labels = cosine_dbscan(group_vectors, eps=adaptive_eps, min_samples=1)
        
        # 라벨 재할당
        for i, orig_idx in enumerate(group_indices):
            if group_labels[i] != -1:
                final_labels[orig_idx] = label_counter + group_labels[i]
            else:
                final_labels[orig_idx] = -1
        
        if len(group_labels) > 0:
            label_counter += max(group_labels) + 1
    
    return final_labels

def safe_ensemble_clustering(hold_data, vectors, base_eps=0.01):
    """🚨 안전한 앙상블 클러스터링 (샘플 수 체크)"""
    if len(hold_data) < 4:
        # 샘플이 4개 미만이면 각각 별도 그룹으로 처리
        return list(range(len(hold_data)))
    
    return ensemble_clustering(hold_data, vectors, base_eps)

def ensemble_clustering(hold_data, vectors, base_eps=0.01):
    """🚀 강화된 앙상블 클러스터링: 4가지 방법의 가중 투표"""
    # 🚨 샘플 수 체크
    if len(hold_data) < 2:
        return [0] * len(hold_data)
    
    # 방법 1: 기본 DBSCAN (가중치 0.3)
    labels_1 = cosine_dbscan(vectors, eps=base_eps, min_samples=1)
    
    # 방법 2: 계층적 클러스터링 (가중치 0.3)
    labels_2 = hierarchical_clustering(hold_data, vectors, base_eps=base_eps)
    
    # 방법 3: 더 엄격한 DBSCAN (가중치 0.2)
    labels_3 = cosine_dbscan(vectors, eps=base_eps * 0.6, min_samples=1)
    
    # 방법 4: 개선된 DBSCAN (가중치 0.2)
    labels_4 = advanced_dbscan(vectors, hold_data, eps=base_eps * 0.8, min_samples=1)
    
    # 가중 투표: 점수 기반
    final_labels = np.full(len(hold_data), -1)
    
    for i in range(len(hold_data)):
        votes = {}
        
        # 각 방법의 그룹 쌍 수집 (가중치 적용)
        for j in range(len(hold_data)):
            if i == j:
                continue
            
            score = 0
            # 방법 1 (가중치 0.3)
            if labels_1[i] != -1 and labels_1[i] == labels_1[j]:
                score += 0.3
            
            # 방법 2 (가중치 0.3)
            if labels_2[i] != -1 and labels_2[i] == labels_2[j]:
                score += 0.3
            
            # 방법 3 (가중치 0.2)
            if labels_3[i] != -1 and labels_3[i] == labels_3[j]:
                score += 0.2
            
            # 방법 4 (가중치 0.2)
            if labels_4[i] != -1 and labels_4[i] == labels_4[j]:
                score += 0.2
            
            if score > 0:
                votes[j] = score
        
        # 🚀 0.7점 이상 받은 홀드들과 같은 그룹 (적절한 엄격도)
        same_group = [j for j, v in votes.items() if v >= 0.7]
        
        if same_group:
            # 그룹 ID 할당 (가장 작은 인덱스 사용)
            final_labels[i] = min(same_group + [i])
    
    # 라벨 정규화 (0, 1, 2, ...)
    unique_labels = sorted(set(final_labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_map[l] for l in final_labels])
    
    return final_labels

def post_process_groups(hold_data, vectors):
    """🚀 강화된 후처리 규칙 - 다중 기준 적용"""
    # 1. 분리 규칙 강화
    separate_groups_by_multiple_criteria(hold_data)
    
    # 2. 병합 규칙 강화  
    merge_similar_groups_by_multiple_criteria(hold_data)
    
    # 3. 노이즈 제거 및 정리
    cleanup_noise_groups(hold_data)
    
    # 4. 그룹 번호 재정렬
    renumber_groups(hold_data)
    
    return hold_data

def post_process_groups_custom(hold_data, vectors, hue_sep_thresh=25, sat_sep_thresh=50, 
                             val_sep_thresh=80, hue_merge_thresh=10):
    """🚀 사용자 정의 파라미터로 후처리"""
    # 1. 사용자 정의 분리 규칙
    separate_groups_custom(hold_data, hue_sep_thresh, sat_sep_thresh, val_sep_thresh)
    
    # 2. 사용자 정의 병합 규칙
    merge_similar_groups_custom(hold_data, hue_merge_thresh)
    
    # 3. 노이즈 제거 및 정리
    cleanup_noise_groups(hold_data)
    
    # 4. 그룹 번호 재정렬
    renumber_groups(hold_data)
    
    return hold_data

def separate_groups_custom(hold_data, hue_sep_thresh=25, sat_sep_thresh=50, val_sep_thresh=80):
    """🚀 초강화된 사용자 정의 그룹 분리 - 더 정교한 기준"""
    for group in set(h["group"] for h in hold_data if h["group"] is not None and h["group"] >= 0):
        group_holds = [h for h in hold_data if h["group"] == group]
        if len(group_holds) <= 1:  # 홀드가 1개뿐이면 분리하지 않음
            continue
            
        hsv_values = [h["dominant_hsv"] for h in group_holds]
        hue_values = [hsv[0] for hsv in hsv_values]
        sat_values = [hsv[1] for hsv in hsv_values]
        val_values = [hsv[2] for hsv in hsv_values]
        
        # 🚀 다중 분리 기준 적용
        should_separate = False
        separation_reason = ""
        
        # 1. 기본 임계값 기준
        max_hue_diff = max(hue_values) - min(hue_values)
        max_sat_diff = max(sat_values) - min(sat_values)
        max_val_diff = max(val_values) - min(val_values)
        
        if max_hue_diff > hue_sep_thresh:
            should_separate = True
            separation_reason = f"Hue 차이 {max_hue_diff:.1f}도 (임계값: {hue_sep_thresh})"
        elif max_sat_diff > sat_sep_thresh:
            should_separate = True
            separation_reason = f"Saturation 차이 {max_sat_diff:.1f} (임계값: {sat_sep_thresh})"
        elif max_val_diff > val_sep_thresh:
            should_separate = True
            separation_reason = f"Value 차이 {max_val_diff:.1f} (임계값: {val_sep_thresh})"
        
        # 2. 🚀 새로운 분리 기준: 색상 분산이 너무 클 때 (더 엄격)
        hue_variance = np.var(hue_values)
        sat_variance = np.var(sat_values)
        val_variance = np.var(val_values)
        
        if hue_variance > 200:  # Hue 분산이 너무 클 때 (극도로 엄격)
            should_separate = True
            separation_reason = f"Hue 분산 {hue_variance:.1f} (임계값: 200)"
        elif sat_variance > 800:  # Saturation 분산이 너무 클 때 (극도로 엄격)
            should_separate = True
            separation_reason = f"Saturation 분산 {sat_variance:.1f} (임계값: 800)"
        elif val_variance > 1500:  # Value 분산이 너무 클 때 (극도로 엄격)
            should_separate = True
            separation_reason = f"Value 분산 {val_variance:.1f} (임계값: 1500)"
        
        # 3. 🚀 새로운 분리 기준: 극단적인 색상 조합
        bright_dark_mix = False
        color_type_mix = False
        
        # 밝은색과 어두운색 혼합 검사
        bright_count = 0
        dark_count = 0
        colorful_count = 0
        
        for hsv in hsv_values:
            h, s, v = hsv
            # 밝은 색상 (Value > 180)
            if v > 180:
                bright_count += 1
            # 어두운 색상 (Value < 80)
            elif v < 80:
                dark_count += 1
            # 채도가 높은 색상 (Saturation > 100)
            if s > 100:
                colorful_count += 1
        
        # 서로 다른 색상 타입이 섞여있는 경우
        if (bright_count > 0 and dark_count > 0) or (colorful_count > 0 and (bright_count > 0 or dark_count > 0)):
            should_separate = True
            separation_reason = f"다른 색상 타입 혼합 (밝음:{bright_count}, 어둠:{dark_count}, 채도:{colorful_count})"
        
        # 4. 🚀 새로운 분리 기준: 특정 색상 조합 분리
        black_count = sum(1 for hsv in hsv_values if hsv[2] < 50)  # 검정색
        white_count = sum(1 for hsv in hsv_values if hsv[1] < 30 and hsv[2] > 200)  # 흰색
        colorful_count = sum(1 for hsv in hsv_values if hsv[1] > 100 and hsv[2] > 100)  # 채도 높은 색상
        
        if (black_count > 0 and (white_count > 0 or colorful_count > 0)) or (white_count > 0 and colorful_count > 0):
            should_separate = True
            separation_reason = f"특정 색상 조합 분리 (검정:{black_count}, 흰색:{white_count}, 채도:{colorful_count})"
        
        # 🚀 홀드가 2개 이상이면 분리
        if should_separate and len(group_holds) >= 2:
            # 🚀 개선된 K-means 분리 (HSV 거리 기반)
            from sklearn.cluster import KMeans
            
            # HSV 특성을 고려한 특징 벡터 생성
            features = []
            for hsv in hsv_values:
                h, s, v = hsv
                # Hue를 원형 좌표로 변환
                h_rad = np.deg2rad(h * 2)  # OpenCV HSV는 0-179
                h_cos = np.cos(h_rad)
                h_sin = np.sin(h_rad)
                features.append([h_cos, h_sin, s/255.0, v/255.0])
            
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=20, max_iter=300)
            sub_labels = kmeans.fit_predict(features)
            
            new_group = max(h["group"] for h in hold_data if h["group"] is not None) + 1
            for i, hold in enumerate(group_holds):
                if sub_labels[i] == 1:
                    hold["group"] = new_group
            
            print(f"그룹 {group} 분리됨: {separation_reason}")

def merge_similar_groups_custom(hold_data, hue_merge_thresh=10):
    """🎯 사용자 정의 그룹 병합"""
    groups = [h["group"] for h in hold_data if h["group"] is not None and h["group"] >= 0]
    unique_groups = list(set(groups))
    
    for i, g1 in enumerate(unique_groups):
        for g2 in unique_groups[i+1:]:
            g1_holds = [h for h in hold_data if h["group"] == g1]
            g2_holds = [h for h in hold_data if h["group"] == g2]
            
            if not g1_holds or not g2_holds:
                continue
                
            g1_hsv = [h["dominant_hsv"] for h in g1_holds]
            g2_hsv = [h["dominant_hsv"] for h in g2_holds]
            
            g1_hue_avg = np.mean([hsv[0] for hsv in g1_hsv])
            g2_hue_avg = np.mean([hsv[0] for hsv in g2_hsv])
            
            hue_diff = abs(g1_hue_avg - g2_hue_avg)
            
            # 사용자 정의 Hue 차이 임계값으로 병합
            if hue_diff <= hue_merge_thresh:
                # g2를 g1으로 병합
                for hold in hold_data:
                    if hold["group"] == g2:
                        hold["group"] = g1
                print(f"그룹 {g2}를 그룹 {g1}으로 병합 (Hue 차이: {hue_diff:.1f}도, 임계값: {hue_merge_thresh})")

def separate_groups_by_multiple_criteria(hold_data):
    """🚀 강화된 그룹 분리 - 다중 기준 적용"""
    for group in set(h["group"] for h in hold_data if h["group"] is not None and h["group"] >= 0):
        group_holds = [h for h in hold_data if h["group"] == group]
        if len(group_holds) <= 2:
            continue
            
        hsv_values = [h["dominant_hsv"] for h in group_holds]
        hue_values = [hsv[0] for hsv in hsv_values]
        sat_values = [hsv[1] for hsv in hsv_values]
        val_values = [hsv[2] for hsv in hsv_values]
        
        # 1. Hue 차이가 25도 이상이면 분리 (더 엄격)
        max_hue_diff = max(hue_values) - min(hue_values)
        
        # 2. Saturation 차이가 50 이상이면 분리 (새로운 기준)
        max_sat_diff = max(sat_values) - min(sat_values)
        
        # 3. Value 차이가 80 이상이면 분리 (새로운 기준)
        max_val_diff = max(val_values) - min(val_values)
        
        should_separate = False
        separation_reason = ""
        
        if max_hue_diff > 25:
            should_separate = True
            separation_reason = f"Hue 차이 {max_hue_diff:.1f}도"
        elif max_sat_diff > 50:
            should_separate = True
            separation_reason = f"Saturation 차이 {max_sat_diff:.1f}"
        elif max_val_diff > 80:
            should_separate = True
            separation_reason = f"Value 차이 {max_val_diff:.1f}"
        
        if should_separate and len(group_holds) >= 3:  # 최소 홀드 수 감소
            # K-means로 2개 그룹으로 분리
            from sklearn.cluster import KMeans
            features = [[hsv[0], hsv[1], hsv[2]] for hsv in hsv_values]
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(features)
            
            new_group = max(h["group"] for h in hold_data if h["group"] is not None) + 1
            for i, hold in enumerate(group_holds):
                if sub_labels[i] == 1:
                    hold["group"] = new_group
            
            print(f"그룹 {group} 분리됨: {separation_reason}")

def merge_similar_groups_by_multiple_criteria(hold_data):
    """🎯 단순화된 그룹 병합"""
    groups = [h["group"] for h in hold_data if h["group"] is not None and h["group"] >= 0]
    unique_groups = list(set(groups))
    
    for i, g1 in enumerate(unique_groups):
        for g2 in unique_groups[i+1:]:
            g1_holds = [h for h in hold_data if h["group"] == g1]
            g2_holds = [h for h in hold_data if h["group"] == g2]
            
            if not g1_holds or not g2_holds:
                continue
                
            g1_hsv = [h["dominant_hsv"] for h in g1_holds]
            g2_hsv = [h["dominant_hsv"] for h in g2_holds]
            
            g1_hue_avg = np.mean([hsv[0] for hsv in g1_hsv])
            g2_hue_avg = np.mean([hsv[0] for hsv in g2_hsv])
            
            hue_diff = abs(g1_hue_avg - g2_hue_avg)
            
            # Hue 차이가 10도 이하면 병합 (단순화)
            if hue_diff <= 10:
                # g2를 g1로 병합
                for hold in hold_data:
                    if hold["group"] == g2:
                        hold["group"] = g1

def cleanup_noise_groups(hold_data):
    """노이즈 그룹 정리"""
    group_counts = {}
    for hold in hold_data:
        if hold["group"] is not None and hold["group"] >= 0:
            group_counts[hold["group"]] = group_counts.get(hold["group"], 0) + 1
    
    # 홀드가 1개뿐인 그룹들을 가장 유사한 그룹으로 병합
    for group, count in group_counts.items():
        if count == 1:
            single_hold = [h for h in hold_data if h["group"] == group][0]
            
            # 가장 유사한 그룹 찾기
            best_group = None
            best_similarity = float('inf')
            
            for other_group in group_counts:
                if other_group != group and group_counts[other_group] > 1:
                    other_holds = [h for h in hold_data if h["group"] == other_group]
                    other_hsv = [h["dominant_hsv"] for h in other_holds]
                    other_hue_avg = np.mean([hsv[0] for hsv in other_hsv])
                    
                    hue_diff = abs(single_hold["dominant_hsv"][0] - other_hue_avg)
                    if hue_diff < best_similarity:
                        best_similarity = hue_diff
                        best_group = other_group
            
            if best_group is not None and best_similarity < 20:  # 20도 이내면 병합
                single_hold["group"] = best_group

def renumber_groups(hold_data):
    """그룹 번호 재정렬"""
    groups = [h["group"] for h in hold_data if h["group"] is not None and h["group"] >= 0]
    unique_groups = sorted(list(set(groups)))
    
    group_mapping = {old_group: new_group for new_group, old_group in enumerate(unique_groups)}
    
    for hold in hold_data:
        if hold["group"] is not None and hold["group"] >= 0:
            hold["group"] = group_mapping[hold["group"]]

def adaptive_eps_selection(hold_data, vectors):
    """🎯 단순화된 적응적 eps 선택"""
    # 1. 홀드 개수 기반 단순 조정
    hold_count = len(hold_data)
    
    # 2. 기본 eps 설정
    if hold_count <= 20:
        base_eps = 0.008  # 적은 홀드: 더 엄격
    elif hold_count <= 50:
        base_eps = 0.012  # 중간 홀드: 보통
    else:
        base_eps = 0.015  # 많은 홀드: 조금 관대
    
    # 3. Hue 분산 기반 조정 (단순화)
    hue_values = [hsv[0] for hsv in [hold["dominant_hsv"] for hold in hold_data]]
    hue_variance = np.var(hue_values)
    
    if hue_variance > 2000:  # 매우 다양한 색상
        base_eps *= 1.5
    elif hue_variance > 1000:  # 다양한 색상
        base_eps *= 1.2
    
    return min(base_eps, 0.03)  # 최대 0.03으로 제한

def assign_groups(hold_data, vectors, eps=0.01, min_samples=1, method="ensemble"):
    """🚀 개선된 그룹 할당 로직 - 적응적 eps + 앙상블 방식"""
    # 적응적 eps 계산
    adaptive_eps = adaptive_eps_selection(hold_data, vectors)
    final_eps = min(eps, adaptive_eps)  # 더 엄격한 기준 사용
    
    # 검정/흰색 전처리
    hold_data = pre_classify_bw(hold_data)
    mask = [h["group"] is None for h in hold_data]
    sub_vectors = vectors[mask]
    
    if len(sub_vectors) > 0:
        sub_hold_data = [h for h in hold_data if h["group"] is None]
        
        if method == "ensemble":
            # 앙상블 클러스터링
            labels = ensemble_clustering(sub_hold_data, sub_vectors, base_eps=final_eps)
        elif method == "hierarchical":
            # 계층적 클러스터링
            labels = hierarchical_clustering(sub_hold_data, sub_vectors, base_eps=final_eps)
        elif method == "advanced":
            # 개선된 DBSCAN
            labels = advanced_dbscan(sub_vectors, sub_hold_data, eps=final_eps, min_samples=min_samples)
        else:
            # 기본 DBSCAN
            labels = cosine_dbscan(sub_vectors, eps=final_eps, min_samples=min_samples)
        
        # 라벨 할당
        j = 0
        for i, hold in enumerate(hold_data):
            if mask[i]:
                hold["group"] = int(labels[j])
                j += 1
        
        # 후처리
        hold_data = post_process_groups(hold_data, vectors)
    
    return hold_data

def simple_dbscan_clustering(hold_data, vectors, eps=1.0):
    """🎯 순수 RGB 거리 기반 클러스터링 - 논리적 그룹핑"""
    import streamlit as st
    import numpy as np
    
    st.write(f"🔍 **순수 RGB 거리 기반 클러스터링:**")
    st.write(f"- 홀드 수: {len(hold_data)}")
    st.write(f"- eps: {eps}")
    st.write(f"💡 **참고**: RGB(156,39,62)와 RGB(155,43,66)의 거리는 5.74입니다. eps=10이면 같은 그룹이 됩니다.")
    
    # RGB 특징 벡터 추출 (R, G, B 값만)
    rgb_vectors = vectors[:, :3]
    st.write(f"- RGB 벡터 형태: {rgb_vectors.shape}")
    
    # 🚨 순수 거리 기반 클러스터링 구현
    n_holds = len(hold_data)
    groups = [-1] * n_holds  # -1은 미분류
    current_group = 0
    
    # 각 홀드에 대해 처리
    for i in range(n_holds):
        if groups[i] != -1:  # 이미 그룹에 할당됨
            continue
            
        # 현재 홀드와 거리가 eps 이하인 모든 홀드 찾기
        current_group_holds = [i]
        
        # 다른 모든 홀드와의 거리 확인
        for j in range(n_holds):
            if i == j or groups[j] != -1:  # 자기 자신이거나 이미 그룹에 할당됨
                continue
                
            # RGB 유클리드 거리 계산
            dist = np.sqrt(np.sum((rgb_vectors[i] - rgb_vectors[j])**2))
            
            if dist <= eps:
                current_group_holds.append(j)
        
        # 현재 홀드와 연결된 모든 홀드에 같은 그룹 할당
        for hold_idx in current_group_holds:
            groups[hold_idx] = current_group
            
        # 🚨 디버깅: 그룹 정보 출력
        if len(current_group_holds) > 1:
            st.write(f"- **그룹 {current_group}**: {len(current_group_holds)}개 홀드")
            for hold_idx in current_group_holds:
                rgb = rgb_vectors[hold_idx]
                st.write(f"  • 홀드 {hold_data[hold_idx]['id']}: RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})")
            
            # 🚨 거리 검증: 그룹 내 모든 홀드 쌍의 거리 확인
            st.write(f"  🔍 **거리 검증 (eps={eps}):**")
            for i in range(len(current_group_holds)):
                for j in range(i+1, len(current_group_holds)):
                    idx1, idx2 = current_group_holds[i], current_group_holds[j]
                    dist = np.sqrt(np.sum((rgb_vectors[idx1] - rgb_vectors[idx2])**2))
                    rgb1 = rgb_vectors[idx1]
                    rgb2 = rgb_vectors[idx2]
                    status = "✅" if dist <= eps else "❌"
                    st.write(f"    {status} 홀드 {hold_data[idx1]['id']} ↔ 홀드 {hold_data[idx2]['id']}: 거리 {dist:.1f} (RGB({rgb1[0]:.0f},{rgb1[1]:.0f},{rgb1[2]:.0f}) ↔ RGB({rgb2[0]:.0f},{rgb2[1]:.0f},{rgb2[2]:.0f}))")
        
        current_group += 1
    
    # 결과 분석
    unique_groups = set(groups)
    n_clusters = len(unique_groups)
    
    st.write(f"🎯 **클러스터링 결과:**")
    st.write(f"- 클러스터: {n_clusters}개")
    st.write(f"- 그룹 ID들: {sorted(unique_groups)}")
    
    # 그룹별 홀드 수 출력
    for group_id in sorted(unique_groups):
        group_holds = [i for i, g in enumerate(groups) if g == group_id]
        st.write(f"- 그룹 {group_id}: {len(group_holds)}개 홀드")
    
    # 🚨 특별 검증: 검정색과 흰색 확인
    st.write(f"🚨 **검정색/흰색 특별 검증:**")
    black_holds = []
    white_holds = []
    
    for i, hold in enumerate(hold_data):
        rgb = rgb_vectors[i]
        if rgb[0] < 10 and rgb[1] < 10 and rgb[2] < 10:  # 검정색
            black_holds.append((i, hold["id"], groups[i]))
        elif rgb[0] > 245 and rgb[1] > 245 and rgb[2] > 245:  # 흰색
            white_holds.append((i, hold["id"], groups[i]))
    
    if black_holds:
        st.write(f"- 검정색 홀드: {len(black_holds)}개")
        for idx, hold_id, group_id in black_holds:
            rgb = rgb_vectors[idx]
            st.write(f"  • 홀드 {hold_id}: RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f}) → 그룹 {group_id}")
    
    if white_holds:
        st.write(f"- 흰색 홀드: {len(white_holds)}개")
        for idx, hold_id, group_id in white_holds:
            rgb = rgb_vectors[idx]
            st.write(f"  • 홀드 {hold_id}: RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f}) → 그룹 {group_id}")
    
    # 검정색과 흰색이 같은 그룹에 있는지 확인
    black_groups = set(groups[idx] for idx, _, _ in black_holds)
    white_groups = set(groups[idx] for idx, _, _ in white_holds)
    
    if black_groups and white_groups:
        intersection = black_groups & white_groups
        if intersection:
            st.error(f"❌ **문제 발견!** 검정색과 흰색이 같은 그룹에 있습니다: {intersection}")
        else:
            st.success(f"✅ 검정색과 흰색이 다른 그룹에 있습니다 (검정: {black_groups}, 흰색: {white_groups})")
    
    # 그룹 ID 할당
    for i, hold in enumerate(hold_data):
        hold["group"] = int(groups[i])
    
    return hold_data

def custom_color_space_transform(rgb_vector):
    """🎨 커스텀 색상 공간 변환: 주요 색상들을 완전히 다른 영역으로 이동"""
    r, g, b = rgb_vector[0], rgb_vector[1], rgb_vector[2]
    
    # 주요 색상 정의 (RGB 좌표) - 더 많은 색상 추가
    major_colors = {
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'red': [255, 0, 0],
        'green': [0, 255, 0], 
        'blue': [0, 0, 255],
        'yellow': [255, 255, 0],
        'magenta': [255, 0, 255],
        'cyan': [0, 255, 255],
        'orange': [255, 165, 0],
        'purple': [128, 0, 128],
        'pink': [255, 192, 203],
        'lime': [0, 255, 0],
        'navy': [0, 0, 128],
        'gray': [128, 128, 128],
        'brown': [139, 69, 19],
        'olive': [128, 128, 0],
        'teal': [0, 128, 128],
        'maroon': [128, 0, 0],
        'gold': [255, 215, 0],
        'silver': [192, 192, 192]
    }
    
    # 🚀 새로운 접근: 색상을 완전히 다른 좌표계로 매핑
    # 각 주요 색상을 3D 공간의 서로 다른 구역에 배치
    color_zones = {
        'black': [0, 0, 0],           # 원점
        'white': [1000, 1000, 1000],  # 최대값
        'red': [1000, 0, 0],          # X축 최대
        'green': [0, 1000, 0],        # Y축 최대
        'blue': [0, 0, 1000],         # Z축 최대
        'yellow': [1000, 1000, 0],    # X+Y 최대
        'magenta': [1000, 0, 1000],   # X+Z 최대
        'cyan': [0, 1000, 1000],      # Y+Z 최대
        'orange': [1000, 500, 0],     # 중간값
        'purple': [500, 0, 1000],     # 중간값
        'pink': [1000, 250, 250],     # 중간값
        'lime': [250, 1000, 0],       # 중간값
        'navy': [0, 0, 500],          # 중간값
        'gray': [500, 500, 500],      # 중간값
        'brown': [500, 250, 0],       # 중간값
        'olive': [500, 500, 0],       # 중간값
        'teal': [0, 500, 500],        # 중간값
        'maroon': [500, 0, 0],        # 중간값
        'gold': [1000, 750, 0],       # 중간값
        'silver': [750, 750, 750]     # 중간값
    }
    
    # 각 주요 색상까지의 거리 계산
    distances = {}
    for color_name, color_rgb in major_colors.items():
        dist = np.sqrt((r - color_rgb[0])**2 + (g - color_rgb[1])**2 + (b - color_rgb[2])**2)
        distances[color_name] = dist
    
    # 가장 가까운 주요 색상 찾기
    closest_color = min(distances, key=distances.get)
    min_distance = distances[closest_color]
    
    # 🚀 새로운 좌표계로 직접 매핑
    target_zone = color_zones[closest_color]
    
    # 거리에 따른 가중치 (가까울수록 더 확실하게 매핑)
    if min_distance < 30:  # 매우 가까운 경우
        weight = 1.0  # 완전히 해당 구역으로 이동
    elif min_distance < 60:  # 가까운 경우
        weight = 0.8
    elif min_distance < 100:  # 중간 거리
        weight = 0.6
    elif min_distance < 150:  # 중간-먼 거리
        weight = 0.4
    else:  # 먼 경우
        weight = 0.2
    
    # 원본 RGB와 목표 구역을 가중 평균
    new_r = (1 - weight) * r + weight * target_zone[0]
    new_g = (1 - weight) * g + weight * target_zone[1]
    new_b = (1 - weight) * b + weight * target_zone[2]
    
    # 범위 제한 (0-1000)
    new_r = np.clip(new_r, 0, 1000)
    new_g = np.clip(new_g, 0, 1000)
    new_b = np.clip(new_b, 0, 1000)
    
    return [new_r, new_g, new_b]

def perceptual_color_dbscan_clustering(hold_data, vectors, eps=30.0):
    """🎨 지각적 색상 공간 DBSCAN: Lab/LCh + CIEDE2000 거리"""
    import streamlit as st
    import numpy as np
    from sklearn.cluster import DBSCAN
    import cv2
    
    st.write(f"🎨 **지각적 색상 공간 DBSCAN 클러스터링:**")
    st.write(f"- 홀드 수: {len(hold_data)}")
    st.write(f"- eps: {eps}")
    st.write(f"- min_samples: 1")
    st.write(f"- **Lab/LCh 공간 + CIEDE2000 거리 사용**")
    
    # RGB → Lab 변환
    lab_vectors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        # HSV → RGB → Lab 변환 (올바른 형태로 수정)
        hsv_arr = np.uint8([[[h, s, v]]])  # 3차원 배열로 수정
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        rgb_image = np.uint8([[[rgb_arr[0], rgb_arr[1], rgb_arr[2]]]])
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)[0][0]
        lab_vectors.append([lab_image[0], lab_image[1], lab_image[2]])
    
    lab_vectors = np.array(lab_vectors)
    st.write(f"- Lab 벡터 형태: {lab_vectors.shape}")
    st.write(f"- L 범위: {lab_vectors[:, 0].min():.0f}-{lab_vectors[:, 0].max():.0f}")
    st.write(f"- a 범위: {lab_vectors[:, 1].min():.0f}-{lab_vectors[:, 1].max():.0f}")
    st.write(f"- b 범위: {lab_vectors[:, 2].min():.0f}-{lab_vectors[:, 2].max():.0f}")
    
    # CIEDE2000 거리 행렬 계산
    def ciede2000_distance(lab1, lab2):
        """CIEDE2000 색상 차이 계산 (간소화 버전)"""
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2
        
        # 명도 차이에 낮은 가중치, 색조 차이에 높은 가중치
        delta_L = abs(L1 - L2) * 0.3  # 명도 차이 가중치 감소
        delta_a = abs(a1 - a2) * 1.5  # 색조 차이 가중치 증가
        delta_b = abs(b1 - b2) * 1.5  # 색조 차이 가중치 증가
        
        # 유클리드 거리 계산
        distance = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
        return distance
    
    # 거리 행렬 계산
    n_samples = len(lab_vectors)
    distance_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                distance_matrix[i, j] = ciede2000_distance(lab_vectors[i], lab_vectors[j])
    
    st.write(f"- 거리 행렬 계산 완료: {distance_matrix.shape}")
    st.write(f"- 거리 범위: {distance_matrix.min():.2f}-{distance_matrix.max():.2f}")
    
    # precomputed 거리 행렬로 DBSCAN 수행
    dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)
    
    # 결과 분석
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    st.write(f"- 클러스터 수: {n_clusters}개")
    st.write(f"- 노이즈 점: {n_noise}개")
    
    # 홀드에 그룹 할당
    for i, hold in enumerate(hold_data):
        hold["group"] = int(labels[i])
    
    return hold_data

def cylindrical_hsv_dbscan_clustering(hold_data, vectors, eps=30.0):
    """🎨 원통 좌표계 HSV DBSCAN: 색조 중심 군집화"""
    import streamlit as st
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    st.write(f"🎨 **원통 좌표계 HSV DBSCAN 클러스터링:**")
    st.write(f"- 홀드 수: {len(hold_data)}")
    st.write(f"- eps: {eps}")
    st.write(f"- min_samples: 1")
    st.write(f"- **Hue 중심 원통 좌표계 사용**")
    
    # HSV → 원통 좌표계 변환
    cylindrical_vectors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        
        # 원통 좌표계 변환
        # Hue를 각도로, Saturation을 반지름으로, Value를 높이로
        theta = np.radians(h)  # 각도 (라디안)
        r = s  # 반지름 (Saturation)
        z = v  # 높이 (Value)
        
        # 명도 가중치 조절 (Value 영향 최소화)
        z_weighted = z * 0.2  # Value 가중치를 0.2로 감소
        
        # 직교 좌표로 변환
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        cylindrical_vectors.append([x, y, z_weighted])
    
    cylindrical_vectors = np.array(cylindrical_vectors)
    st.write(f"- 원통 좌표 벡터 형태: {cylindrical_vectors.shape}")
    st.write(f"- X 범위: {cylindrical_vectors[:, 0].min():.2f}-{cylindrical_vectors[:, 0].max():.2f}")
    st.write(f"- Y 범위: {cylindrical_vectors[:, 1].min():.2f}-{cylindrical_vectors[:, 1].max():.2f}")
    st.write(f"- Z 범위: {cylindrical_vectors[:, 2].min():.2f}-{cylindrical_vectors[:, 2].max():.2f}")
    
    # 원통 좌표계에서 DBSCAN 수행
    dbscan = DBSCAN(eps=eps, min_samples=1, metric='euclidean')
    labels = dbscan.fit_predict(cylindrical_vectors)
    
    # 결과 분석
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    st.write(f"- 클러스터 수: {n_clusters}개")
    st.write(f"- 노이즈 점: {n_noise}개")
    
    # 홀드에 그룹 할당
    for i, hold in enumerate(hold_data):
        hold["group"] = int(labels[i])
    
    return hold_data

def custom_color_cube_dbscan_clustering(hold_data, vectors, eps=30.0):
    """🎨 커스텀 색상 큐브 DBSCAN: 주요 색상 간 거리 확장"""
    import streamlit as st
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    st.write(f"🎨 **커스텀 색상 큐브 DBSCAN 클러스터링 (강화 버전):**")
    st.write(f"- 홀드 수: {len(hold_data)}")
    st.write(f"- 원본 eps: {eps}")
    
    # 새로운 좌표계에 맞게 eps 조정 (0-1000 범위)
    adjusted_eps = eps * 4  # 0-255 → 0-1000 범위로 확장
    st.write(f"- 조정된 eps: {adjusted_eps} (새로운 좌표계용)")
    st.write(f"- min_samples: 1")
    st.write(f"- **주요 색상들을 완전히 다른 구역으로 분리**")
    
    # HSV→RGB 변환된 값 사용
    rgb_vectors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        rgb = hsv_to_rgb([h, s, v])
        rgb_vectors.append([rgb[0], rgb[1], rgb[2]])
    rgb_vectors = np.array(rgb_vectors)
    
    st.write(f"- 원본 RGB 벡터 형태: {rgb_vectors.shape}")
    st.write(f"- RGB 값 범위: R({rgb_vectors[:, 0].min():.0f}-{rgb_vectors[:, 0].max():.0f}), G({rgb_vectors[:, 1].min():.0f}-{rgb_vectors[:, 1].max():.0f}), B({rgb_vectors[:, 2].min():.0f}-{rgb_vectors[:, 2].max():.0f})")
    
    # 커스텀 색상 공간 변환 적용
    custom_vectors = []
    for rgb_vec in rgb_vectors:
        custom_rgb = custom_color_space_transform(rgb_vec)
        custom_vectors.append(custom_rgb)
    custom_vectors = np.array(custom_vectors)
    
    st.write(f"- 변환 후 RGB 벡터 형태: {custom_vectors.shape}")
    st.write(f"- 변환 후 RGB 값 범위: R({custom_vectors[:, 0].min():.0f}-{custom_vectors[:, 0].max():.0f}), G({custom_vectors[:, 1].min():.0f}-{custom_vectors[:, 1].max():.0f}), B({custom_vectors[:, 2].min():.0f}-{custom_vectors[:, 2].max():.0f})")
    
    # 변환된 공간에서 DBSCAN 수행 (조정된 eps 사용)
    dbscan = DBSCAN(eps=adjusted_eps, min_samples=1, metric='euclidean')
    labels = dbscan.fit_predict(custom_vectors)
    
    # 결과 분석
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    st.write(f"- 클러스터 수: {n_clusters}개")
    st.write(f"- 노이즈 점: {n_noise}개")
    
    # 홀드에 그룹 할당
    for i, hold in enumerate(hold_data):
        hold["group"] = int(labels[i])
    
    return hold_data

def hsv_cube_dbscan_clustering(hold_data, vectors, eps=30.0):
    """🎯 HSV 색상 공간 기반 DBSCAN 클러스터링 - 대각선 교차 문제 해결"""
    import streamlit as st
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    st.write(f"🌈 **HSV 색상 공간 기반 DBSCAN 클러스터링:**")
    st.write(f"- 홀드 수: {len(hold_data)}")
    st.write(f"- eps: {eps}")
    st.write(f"- min_samples: 1")
    
    # HSV 값 직접 사용 (대각선 교차 문제 해결)
    hsv_vectors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        hsv_vectors.append([h, s, v])
    hsv_vectors = np.array(hsv_vectors)
    
    st.write(f"- HSV 벡터 형태: {hsv_vectors.shape}")
    st.write(f"- H 범위: {hsv_vectors[:, 0].min():.0f}-{hsv_vectors[:, 0].max():.0f}°")
    st.write(f"- S 범위: {hsv_vectors[:, 1].min():.0f}-{hsv_vectors[:, 1].max():.0f}")
    st.write(f"- V 범위: {hsv_vectors[:, 2].min():.0f}-{hsv_vectors[:, 2].max():.0f}")
    
    # HSV 공간에서 DBSCAN 수행
    dbscan = DBSCAN(eps=eps, min_samples=1, metric='euclidean')
    labels = dbscan.fit_predict(hsv_vectors)
    
    # 결과 분석
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    st.write(f"- 클러스터 수: {n_clusters}개")
    st.write(f"- 노이즈 점: {n_noise}개")
    
    # 홀드에 그룹 할당
    for i, hold in enumerate(hold_data):
        hold["group"] = int(labels[i])
    
    return hold_data

def rgb_weighted_dbscan_clustering(hold_data, vectors, eps=0.01, weights=[0.5, 0.5, 1.2]):
    """🎯 RGB 축별 가중치 DBSCAN 클러스터링 - 사용자 정의 가중치"""
    import streamlit as st
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    st.write(f"🎯 **RGB 축별 가중치 DBSCAN 클러스터링:**")
    st.write(f"- 홀드 수: {len(hold_data)}")
    st.write(f"- eps: {eps}")
    st.write(f"- min_samples: 1")
    st.write(f"- **축별 가중치: R={weights[0]}, G={weights[1]}, B={weights[2]}**")
    
    # HSV→RGB 변환된 값 사용
    rgb_vectors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        rgb = hsv_to_rgb([h, s, v])
        rgb_vectors.append([rgb[0], rgb[1], rgb[2]])
    rgb_vectors = np.array(rgb_vectors)
    
    st.write(f"- RGB 벡터 형태: {rgb_vectors.shape}")
    st.write(f"- RGB 값 범위: R({rgb_vectors[:, 0].min():.0f}-{rgb_vectors[:, 0].max():.0f}), G({rgb_vectors[:, 1].min():.0f}-{rgb_vectors[:, 1].max():.0f}), B({rgb_vectors[:, 2].min():.0f}-{rgb_vectors[:, 2].max():.0f})")
    
    # 사용자 정의 가중치 적용
    weights_array = np.array(weights)
    
    # 가중치 적용된 거리 계산을 위한 커스텀 메트릭
    def weighted_euclidean_distance(x, y):
        diff = x - y
        weighted_diff = diff * weights_array
        return np.sqrt(np.sum(weighted_diff ** 2))
    
    # 🎯 가중치 적용된 DBSCAN 클러스터링
    dbscan = DBSCAN(eps=eps, min_samples=1, metric=weighted_euclidean_distance)
    labels = dbscan.fit_predict(rgb_vectors)
    
    # 결과 분석
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    st.write(f"- 클러스터 수: {n_clusters}개")
    st.write(f"- 노이즈 점: {n_noise}개")
    
    # 홀드에 그룹 할당
    for i, hold in enumerate(hold_data):
        hold["group"] = int(labels[i])
    
    return hold_data

def rgb_cube_dbscan_clustering(hold_data, vectors, eps=0.01):
    """🎯 3D RGB 큐브 기반 DBSCAN 클러스터링 - 축별 가중치 적용"""
    import streamlit as st
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    st.write(f"🔍 **3D RGB 큐브 기반 DBSCAN 클러스터링 (축별 가중치):**")
    st.write(f"- 홀드 수: {len(hold_data)}")
    st.write(f"- eps: {eps}")
    st.write(f"- min_samples: 1")
    
    # 🚨 HSV→RGB 변환된 값 사용 (3D 큐브와 동일)
    rgb_vectors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        rgb = hsv_to_rgb([h, s, v])  # HSV에서 RGB로 변환
        rgb_vectors.append([rgb[0], rgb[1], rgb[2]])
    rgb_vectors = np.array(rgb_vectors)
    
    st.write(f"- RGB 벡터 형태: {rgb_vectors.shape}")
    st.write(f"- RGB 값 범위: R({rgb_vectors[:, 0].min():.0f}-{rgb_vectors[:, 0].max():.0f}), G({rgb_vectors[:, 1].min():.0f}-{rgb_vectors[:, 1].max():.0f}), B({rgb_vectors[:, 2].min():.0f}-{rgb_vectors[:, 2].max():.0f})")
    
    # 🎯 축별 가중치 적용
    # Blue 라인: 상하 변화가 적음 → 엄격한 eps (가중치 높음)
    # Green/Red 라인: 0,0→255,255 1:1 변화 → 관대한 eps (가중치 낮음)
    weights = np.array([0.5, 0.5, 1.2])  # [R, G, B] 가중치
    st.write(f"- **축별 가중치: R={weights[0]}, G={weights[1]}, B={weights[2]}**")
    st.write(f"- Blue 축(B)에 더 엄격한 가중치 적용")
    
    # 가중치 적용된 거리 계산을 위한 커스텀 메트릭
    def weighted_euclidean_distance(x, y):
        diff = x - y
        weighted_diff = diff * weights
        return np.sqrt(np.sum(weighted_diff ** 2))
    
    # 🚨 RGB 값 검증 (콘솔 출력)
    print(f"\n=== RGB 값 검증 (총 {len(rgb_vectors)}개 홀드) ===")
    zero_count = 0
    for i, rgb in enumerate(rgb_vectors):
        if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
            zero_count += 1
            print(f"⚠️ 홀드 {hold_data[i]['id']}: RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})")
    
    print(f"RGB(0,0,0) 홀드 총 개수: {zero_count}개")
    if zero_count > 5:
        print("🚨 색상 추출에 문제가 있을 수 있습니다!")
    
    # Streamlit에도 간단히 표시
    if zero_count > 5:
        st.warning(f"⚠️ RGB(0,0,0) 홀드가 총 {zero_count}개 있습니다! (콘솔 로그 확인)")
    else:
        st.success(f"✅ RGB(0,0,0) 홀드: {zero_count}개")
    
    # 🎯 가중치 적용된 DBSCAN 클러스터링
    dbscan = DBSCAN(eps=eps, min_samples=1, metric=weighted_euclidean_distance)
    labels = dbscan.fit_predict(rgb_vectors)
    
    # 결과 분석
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    st.write(f"🎯 **DBSCAN 클러스터링 결과:**")
    st.write(f"- 클러스터 수: {n_clusters}개")
    st.write(f"- 노이즈 점: {n_noise}개")
    st.write(f"- 라벨들: {sorted(unique_labels)}")
    
    # 그룹별 홀드 수 및 RGB 값 출력 (콘솔 출력)
    print(f"\n=== DBSCAN 클러스터링 결과 (eps={eps}) ===")
    print(f"클러스터 수: {n_clusters}개")
    print(f"노이즈 점: {n_noise}개")
    print(f"라벨들: {sorted(unique_labels)}")
    
    total_errors = 0
    for label in sorted(unique_labels):
        if label == -1:
            continue
        group_indices = [i for i, l in enumerate(labels) if l == label]
        
        print(f"\n--- 그룹 {label} ({len(group_indices)}개 홀드) ---")
        
        # 그룹 내 RGB 값들 출력 (모든 홀드 표시)
        for i, idx in enumerate(group_indices):
            rgb = rgb_vectors[idx]
            print(f"  홀드 {hold_data[idx]['id']}: RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})")
            
        # 🚨 그룹 내 거리 검증
        if len(group_indices) > 1:
            print(f"  🔍 그룹 {label} 내 거리 검증 (eps={eps}):")
            error_count = 0
            for i in range(len(group_indices)):
                for j in range(i+1, len(group_indices)):
                    idx1, idx2 = group_indices[i], group_indices[j]
                    dist = np.sqrt(np.sum((rgb_vectors[idx1] - rgb_vectors[idx2])**2))
                    status = "✅" if dist <= eps else "❌"
                    print(f"    {status} 홀드 {hold_data[idx1]['id']} ↔ 홀드 {hold_data[idx2]['id']}: 거리 {dist:.3f}")
                    if dist > eps:
                        print(f"      ⚠️ 거리 {dist:.3f} > eps {eps} - 같은 그룹이면 안 됨!")
                        error_count += 1
            
            total_errors += error_count
            if error_count > 0:
                print(f"  🚨 그룹 {label}에 {error_count}개의 거리 오류가 있습니다!")
            else:
                print(f"  ✅ 그룹 {label}의 모든 홀드 거리가 eps 이하입니다!")
    
    print(f"\n=== 총 거리 오류: {total_errors}개 ===")
    
    # Streamlit에도 간단히 표시
    if total_errors > 0:
        st.error(f"🚨 총 {total_errors}개의 거리 오류가 있습니다! (콘솔 로그 확인)")
    else:
        st.success(f"✅ 모든 그룹의 거리가 정상입니다!")
    
    # 노이즈 점들 출력
    if n_noise > 0:
        noise_indices = [i for i, l in enumerate(labels) if l == -1]
        st.write(f"- **노이즈**: {n_noise}개 홀드")
        for i, idx in enumerate(noise_indices[:3]):
            rgb = rgb_vectors[idx]
            st.write(f"  • 홀드 {hold_data[idx]['id']}: RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})")
        if n_noise > 3:
            st.write(f"  ... 외 {n_noise-3}개")
    
    # 그룹 ID 할당
    for i, hold in enumerate(hold_data):
        hold["group"] = int(labels[i])
    
    return hold_data

def plot_2d(hold_data, vectors):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, vec in enumerate(vectors):
        h, s, v = hold_data[i]["dominant_hsv"]
        rgb = np.array(hsv_to_rgb(np.array([h, s, v], dtype=np.uint8))) / 255.0
        ax.scatter(vec[0], vec[1], color=rgb, s=80, edgecolors="k")
        ax.text(vec[0], vec[1], str(hold_data[i]["id"]), fontsize=8, ha="center", va="center", color="black")
    ax.set_title("Hold Clustering - HSV 색상환 좌표 기반")
    return fig

def plot_3d(hold_data, vectors):
    colors = ["rgb" + str(hsv_to_rgb(np.array(h["dominant_hsv"], dtype=np.uint8))) for h in hold_data]
    fig = px.scatter_3d(
        x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
        color=colors,
        text=[str(h["id"]) for h in hold_data],
        title="3D Hold Clustering - HSV 좌표 기반"
    )
    return fig

def create_clustering_visualizations(hold_data, vectors):
    """🚀 그룹핑 결과 시각화 - 다중 차트"""
    if len(hold_data) == 0:
        return None
    
    # 그룹별 색상 매핑
    groups = [h["group"] for h in hold_data if h["group"] is not None]
    unique_groups = sorted(set(groups))
    group_colors = px.colors.qualitative.Set3[:len(unique_groups)]
    group_color_map = {g: group_colors[i % len(group_colors)] for i, g in enumerate(unique_groups)}
    
    # 1. t-SNE 2D 시각화
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(hold_data)-1))
        vectors_2d = tsne.fit_transform(vectors)
    except:
        # t-SNE 실패 시 PCA 사용
        pca = PCA(n_components=2, random_state=42)
        vectors_2d = pca.fit_transform(vectors)
    
    # 2. 그룹별 색상 분포 히스토그램
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['t-SNE 2D 분포', 'Hue 분포', 'Saturation 분포', 'Value 분포'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # t-SNE 2D 플롯 - 실제 홀드 색상으로 표시
    for i, hold in enumerate(hold_data):
        group = hold["group"] if hold["group"] is not None else -1
        
        # 🚀 실제 홀드 색상 계산
        h, s, v = hold["dominant_hsv"]
        actual_color = hsv_to_rgb([h, s, v])
        color_rgb = f"rgb({actual_color[0]}, {actual_color[1]}, {actual_color[2]})"
        
        # 테두리 색상 (그룹 구분용)
        border_color = group_color_map.get(group, "#cccccc")
        
        fig.add_trace(
            go.Scatter(
                x=[vectors_2d[i, 0]], y=[vectors_2d[i, 1]],
                mode='markers+text',
                marker=dict(
                    color=color_rgb,  # 실제 홀드 색상
                    size=15,  # 크기 증가
                    line=dict(color=border_color, width=3),  # 그룹별 테두리
                    opacity=0.8
                ),
                text=[str(hold["id"])],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                name=f"Group {group}",
                showlegend=False,
                hovertemplate=f"<b>홀드 ID: {hold['id']}</b><br>" +
                            f"그룹: {group}<br>" +
                            f"실제 색상: {color_rgb}<br>" +
                            f"HSV: ({h:.0f}, {s:.0f}, {v:.0f})<br>" +
                            f"위치: ({vectors_2d[i, 0]:.2f}, {vectors_2d[i, 1]:.2f})<extra></extra>"
            ),
            row=1, col=1
        )
    
    # HSV 히스토그램
    h_values = [h["dominant_hsv"][0] for h in hold_data]
    s_values = [h["dominant_hsv"][1] for h in hold_data]
    v_values = [h["dominant_hsv"][2] for h in hold_data]
    
    # Hue 히스토그램 (그룹별)
    for group in unique_groups:
        group_h_values = [h["dominant_hsv"][0] for h in hold_data if h["group"] == group]
        if group_h_values:
            fig.add_trace(
                go.Histogram(
                    x=group_h_values,
                    name=f"Group {group}",
                    marker_color=group_color_map[group],
                    opacity=0.7
                ),
                row=1, col=2
            )
    
    # Saturation 히스토그램
    for group in unique_groups:
        group_s_values = [h["dominant_hsv"][1] for h in hold_data if h["group"] == group]
        if group_s_values:
            fig.add_trace(
                go.Histogram(
                    x=group_s_values,
                    name=f"Group {group}",
                    marker_color=group_color_map[group],
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Value 히스토그램
    for group in unique_groups:
        group_v_values = [h["dominant_hsv"][2] for h in hold_data if h["group"] == group]
        if group_v_values:
            fig.add_trace(
                go.Histogram(
                    x=group_v_values,
                    name=f"Group {group}",
                    marker_color=group_color_map[group],
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        title="🎯 그룹핑 결과 시각화",
        height=800,
        showlegend=True
    )
    
    return fig

def create_group_color_palette(hold_data):
    """🎨 그룹별 색상 팔레트 생성"""
    if len(hold_data) == 0:
        return None
    
    groups = [h["group"] for h in hold_data if h["group"] is not None]
    unique_groups = sorted(set(groups))
    
    fig = make_subplots(
        rows=1, cols=len(unique_groups) if len(unique_groups) > 0 else 1,
        subplot_titles=[f"그룹 {g}" for g in unique_groups],
        specs=[[{"type": "scatter"} for _ in unique_groups]]
    )
    
    for i, group in enumerate(unique_groups):
        group_holds = [h for h in hold_data if h["group"] == group]
        
        # 그룹 내 모든 홀드의 색상 표시
        colors = []
        hold_ids = []
        positions = []
        
        for j, hold in enumerate(group_holds):
            h, s, v = hold["dominant_hsv"]
            actual_color = hsv_to_rgb([h, s, v])
            color_rgb = f"rgb({actual_color[0]}, {actual_color[1]}, {actual_color[2]})"
            
            colors.append(color_rgb)
            hold_ids.append(hold["id"])
            positions.append(j)
        
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=[0] * len(positions),
                mode='markers+text',
                marker=dict(
                    color=colors,
                    size=25,
                    line=dict(color="black", width=2),
                    opacity=0.9
                ),
                text=[f"ID:{hid}" for hid in hold_ids],
                textposition="bottom center",
                textfont=dict(size=8, color="black"),
                name=f"Group {group}",
                showlegend=False,
                hovertemplate="<b>홀드 ID: %{text}</b><br>" +
                            f"그룹: {group}<br>" +
                            "색상: %{marker.color}<extra></extra>"
            ),
            row=1, col=i+1
        )
        
        # 축 설정
        fig.update_xaxes(showgrid=False, showticklabels=False, range=[-0.5, len(group_holds)-0.5], row=1, col=i+1)
        fig.update_yaxes(showgrid=False, showticklabels=False, range=[-0.5, 0.5], row=1, col=i+1)
    
    fig.update_layout(
        title="🎨 그룹별 홀드 색상 팔레트",
        height=200,
        showlegend=False
    )
    
    return fig

def create_enhanced_color_distance_matrix(hold_data):
    """🚀 강화된 색상 거리 매트릭스 - HSV 기반 정교한 거리 계산"""
    n = len(hold_data)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            h1, s1, v1 = hold_data[i]["dominant_hsv"]
            h2, s2, v2 = hold_data[j]["dominant_hsv"]
            
            # 🚀 HSV 기반 정교한 거리 계산
            # 1. Hue 거리 (원형 거리 고려)
            hue_diff = min(abs(h1 - h2), 179 - abs(h1 - h2))  # 원형 거리
            hue_distance = hue_diff / 179.0  # 정규화 (0-1)
            
            # 2. Saturation 거리
            sat_distance = abs(s1 - s2) / 255.0  # 정규화 (0-1)
            
            # 3. Value 거리  
            val_distance = abs(v1 - v2) / 255.0  # 정규화 (0-1)
            
            # 🚀 가중 평균 거리 (Hue에 더 높은 가중치)
            total_distance = (0.5 * hue_distance + 0.3 * sat_distance + 0.2 * val_distance)
            
            distance_matrix[i][j] = distance_matrix[j][i] = total_distance
    
    return distance_matrix

def create_color_similarity_heatmap(hold_data, vectors):
    """🔥 홀드 간 색상 유사도 히트맵 - 강화된 버전"""
    if len(hold_data) == 0:
        return None
    
    # 🚀 강화된 색상 거리 매트릭스 사용
    distance_matrix = create_enhanced_color_distance_matrix(hold_data)
    
    # 거리를 유사도로 변환 (거리가 가까울수록 유사도 높음)
    similarity_matrix = 1 - distance_matrix
    
    # 홀드 ID와 그룹 정보 준비
    hold_ids = [str(h["id"]) for h in hold_data]
    hold_groups = [h["group"] if h["group"] is not None else -1 for h in hold_data]
    
    # 🚀 더 넓은 색상 범위 사용 (0.2-1.0)
    min_sim = np.min(similarity_matrix)
    max_sim = np.max(similarity_matrix)
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=hold_ids,
        y=hold_ids,
        colorscale='RdYlBu_r',  # 빨강-노랑-파랑 (역순)
        zmin=min_sim,  # 동적 범위
        zmax=max_sim,  # 동적 범위
        hoverongaps=False,
        hovertemplate='<b>홀드 %{y} ↔ 홀드 %{x}</b><br>' +
                     '색상 유사도: %{z:.3f}<br>' +
                     '그룹 %{y}: ' + str(hold_groups[int('%{y}')-1] if '%{y}'.isdigit() and int('%{y}') <= len(hold_groups) else 'N/A') + '<br>' +
                     '그룹 %{x}: ' + str(hold_groups[int('%{x}')-1] if '%{x}'.isdigit() and int('%{x}') <= len(hold_groups) else 'N/A') + '<br>' +
                     '<extra></extra>',
        colorbar=dict(title="색상 유사도")
    ))
    
    # 그룹별 구분선 추가
    fig.update_layout(
        title="🔥 홀드 간 색상 유사도 매트릭스 (강화된 HSV 거리)",
        xaxis_title="홀드 ID",
        yaxis_title="홀드 ID",
        height=600,
        width=600
    )
    
    return fig

def create_group_similarity_heatmap(hold_data):
    """🎯 그룹 간 색상 유사도 매트릭스"""
    if len(hold_data) == 0:
        return None
    
    # 그룹별 평균 색상 계산
    groups = {}
    for hold in hold_data:
        group = hold["group"]
        if group is not None:
            if group not in groups:
                groups[group] = []
            groups[group].append(hold["dominant_hsv"])
    
    # 각 그룹의 평균 HSV 계산
    group_avg_colors = {}
    for group, hsv_list in groups.items():
        avg_h = np.mean([hsv[0] for hsv in hsv_list])
        avg_s = np.mean([hsv[1] for hsv in hsv_list])
        avg_v = np.mean([hsv[2] for hsv in hsv_list])
        group_avg_colors[group] = (avg_h, avg_s, avg_v)
    
    # 그룹 간 거리 계산
    group_ids = sorted(groups.keys())
    n_groups = len(group_ids)
    group_distance_matrix = np.zeros((n_groups, n_groups))
    
    for i, group1 in enumerate(group_ids):
        for j, group2 in enumerate(group_ids):
            if i != j:
                h1, s1, v1 = group_avg_colors[group1]
                h2, s2, v2 = group_avg_colors[group2]
                
                # HSV 거리 계산
                hue_diff = min(abs(h1 - h2), 179 - abs(h1 - h2))
                hue_distance = hue_diff / 179.0
                sat_distance = abs(s1 - s2) / 255.0
                val_distance = abs(v1 - v2) / 255.0
                
                total_distance = (0.5 * hue_distance + 0.3 * sat_distance + 0.2 * val_distance)
                group_distance_matrix[i][j] = total_distance
    
    # 거리를 유사도로 변환
    group_similarity_matrix = 1 - group_distance_matrix
    
    fig = go.Figure(data=go.Heatmap(
        z=group_similarity_matrix,
        x=[f"그룹 {g}" for g in group_ids],
        y=[f"그룹 {g}" for g in group_ids],
        colorscale='RdYlBu_r',
        zmin=0,
        zmax=1,
        hoverongaps=False,
        hovertemplate='<b>그룹 %{y} ↔ 그룹 %{x}</b><br>' +
                     '그룹 간 색상 유사도: %{z:.3f}<br>' +
                     '<extra></extra>',
        colorbar=dict(title="그룹 간 색상 유사도")
    ))
    
    fig.update_layout(
        title="🎯 그룹 간 색상 유사도 매트릭스",
        xaxis_title="그룹",
        yaxis_title="그룹",
        height=500,
        width=500
    )
    
    return fig

def create_color_picker_style_palette(hold_data):
    """🎨 그라데이션 배경 색상 선택기 - 어두움→밝음(가로) × 색상(세로)"""
    if len(hold_data) == 0:
        return None
    
    # 색상판 배경 생성
    fig = go.Figure()
    
    # 🎨 그라데이션 배경을 Heatmap으로 생성
    create_gradient_background_heatmap(fig)
    
    # 그룹별 색상 매핑
    groups = [h["group"] for h in hold_data if h["group"] is not None]
    unique_groups = sorted(set(groups))
    group_colors = px.colors.qualitative.Set3[:len(unique_groups)]
    group_color_map = {g: group_colors[i % len(group_colors)] for i, g in enumerate(unique_groups)}
    
    # 각 홀드를 정확한 위치에 표시
    for i, hold in enumerate(hold_data):
        h, s, v = hold["dominant_hsv"]
        group = hold["group"] if hold["group"] is not None else -1
        
        # 실제 홀드 색상
        actual_color = hsv_to_rgb([h, s, v])
        color_rgb = f"rgb({actual_color[0]}, {actual_color[1]}, {actual_color[2]})"
        
        # 테두리 색상 (그룹 구분용)
        border_color = group_color_map.get(group, "#cccccc")
        
        fig.add_trace(
            go.Scatter(
                x=[v],  # Value (가로축)
                y=[h],  # Hue (세로축)
                mode='markers+text',
                marker=dict(
                    color=color_rgb,
                    size=12,  # 약간 더 큰 크기로 조정
                    line=dict(color=border_color, width=2),
                    opacity=1.0  # 완전 불투명
                ),
                text=[f"{hold['id']}"],
                textposition="top center",
                textfont=dict(size=8, color="white", family="Arial Black"),
                name=f"Group {group}",
                showlegend=False,
                hovertemplate=f"<b>홀드 ID: {hold['id']}</b><br>" +
                            f"그룹: {group}<br>" +
                            f"실제 색상: {color_rgb}<br>" +
                            f"Hue: {h:.1f}°<br>" +
                            f"Saturation: {s:.1f}<br>" +
                            f"Value: {v:.1f}<extra></extra>"
            )
        )
    
    # 축 설정
    fig.update_layout(
        title="🎨 그라데이션 색상 선택기 - 가로=밝기(어두움→밝음), 세로=색상(빨강→보라)",
        xaxis_title="Value (밝기) - 0(검정) → 255(밝음)",
        yaxis_title="Hue (색상) - 0°(빨강) → 60°(노랑) → 120°(초록) → 180°(보라)",
        xaxis=dict(range=[0, 255], dtick=25, showgrid=True, gridcolor="white", gridwidth=0.5),
        yaxis=dict(range=[0, 179], dtick=15, showgrid=True, gridcolor="white", gridwidth=0.5),
        width=1200,  # 더 넓게
        height=800,  # 더 높게
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    return fig

def create_simple_color_palette(hold_data):
    """🎨 간단한 색상판 시각화 - 올바른 축 배치 (배경 없는 버전)"""
    if len(hold_data) == 0:
        return None
    
    # 색상판 배경 없이 바로 홀드들만 표시
    fig = go.Figure()
    
    # 그룹별 색상 매핑
    groups = [h["group"] for h in hold_data if h["group"] is not None]
    unique_groups = sorted(set(groups))
    group_colors = px.colors.qualitative.Set3[:len(unique_groups)]
    group_color_map = {g: group_colors[i % len(group_colors)] for i, g in enumerate(unique_groups)}
    
    # 각 홀드를 올바른 위치에 표시
    for i, hold in enumerate(hold_data):
        h, s, v = hold["dominant_hsv"]
        group = hold["group"] if hold["group"] is not None else -1
        
        # 실제 홀드 색상
        actual_color = hsv_to_rgb([h, s, v])
        color_rgb = f"rgb({actual_color[0]}, {actual_color[1]}, {actual_color[2]})"
        
        # 테두리 색상 (그룹 구분용)
        border_color = group_color_map.get(group, "#cccccc")
        
        fig.add_trace(
            go.Scatter(
                x=[v],  # Value (가로축) - 어두움→밝음
                y=[h],  # Hue (세로축) - 빨강→보라
                mode='markers+text',
                marker=dict(
                    color=color_rgb,
                    size=10,  # 적당한 크기
                    line=dict(color=border_color, width=2),
                    opacity=1.0  # 완전 불투명
                ),
                text=[f"{hold['id']}"],
                textposition="top center",
                textfont=dict(size=8, color="white", family="Arial Black"),
                name=f"Group {group}",
                showlegend=False,
                hovertemplate=f"<b>홀드 ID: {hold['id']}</b><br>" +
                            f"그룹: {group}<br>" +
                            f"실제 색상: {color_rgb}<br>" +
                            f"Hue: {h:.1f}°<br>" +
                            f"Saturation: {s:.1f}<br>" +
                            f"Value: {v:.1f}<extra></extra>"
            )
        )
    
    # 축 설정
    fig.update_layout(
        title="🎨 간단한 홀드 색상판 - 가로=밝기(어두움→밝음), 세로=색상(빨강→보라)",
        xaxis_title="Value (밝기) - 0(검정) → 255(밝음)",
        yaxis_title="Hue (색상) - 0°(빨강) → 60°(노랑) → 120°(초록) → 180°(보라)",
        xaxis=dict(range=[0, 255], dtick=25, showgrid=True, gridcolor="lightgray", gridwidth=1),
        yaxis=dict(range=[0, 179], dtick=15, showgrid=True, gridcolor="lightgray", gridwidth=1),
        width=1200,  # 더 넓게
        height=800,  # 더 높게
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    return fig

def create_hsv_color_palette(hold_data):
    """🎨 HSV 색상환 기반 색상판 시각화 (간단한 버전)"""
    if len(hold_data) == 0:
        return None
    
    # 색상판 배경 생성 (색상 선택기와 같은 스타일)
    fig = go.Figure()
    
    # 그룹별 색상 매핑
    groups = [h["group"] for h in hold_data if h["group"] is not None]
    unique_groups = sorted(set(groups))
    group_colors = px.colors.qualitative.Set3[:len(unique_groups)]
    group_color_map = {g: group_colors[i % len(group_colors)] for i, g in enumerate(unique_groups)}
    
    # 각 홀드를 색상판에 표시
    for i, hold in enumerate(hold_data):
        h, s, v = hold["dominant_hsv"]
        group = hold["group"] if hold["group"] is not None else -1
        
        # HSV를 0-360, 0-100 범위로 변환
        hue_360 = h * 2  # OpenCV HSV는 0-179 범위
        sat_100 = s * 100 / 255  # OpenCV HSV는 0-255 범위
        
        # 실제 홀드 색상
        actual_color = hsv_to_rgb([h, s, v])
        color_rgb = f"rgb({actual_color[0]}, {actual_color[1]}, {actual_color[2]})"
        
        # 테두리 색상 (그룹 구분용)
        border_color = group_color_map.get(group, "#cccccc")
        
        fig.add_trace(
            go.Scatter(
                x=[hue_360],
                y=[sat_100],
                mode='markers+text',
                marker=dict(
                    color=color_rgb,
                    size=20,
                    line=dict(color=border_color, width=3),
                    opacity=0.8
                ),
                text=[f"ID:{hold['id']}"],
                textposition="top center",
                textfont=dict(size=8, color="black"),
                name=f"Group {group}",
                showlegend=False,
                hovertemplate=f"<b>홀드 ID: {hold['id']}</b><br>" +
                            f"그룹: {group}<br>" +
                            f"실제 색상: {color_rgb}<br>" +
                            f"Hue: {hue_360:.1f}°<br>" +
                            f"Saturation: {sat_100:.1f}%<br>" +
                            f"Value: {v:.1f}<extra></extra>"
            )
        )
    
    # 축 설정
    fig.update_layout(
        title="🎨 HSV 색상환 기반 홀드 색상 분포",
        xaxis_title="Hue (0° - 360°)",
        yaxis_title="Saturation (0% - 100%)",
        xaxis=dict(range=[0, 360], dtick=30),
        yaxis=dict(range=[0, 100], dtick=20),
        width=800,
        height=600
    )
    
    return fig

def create_rgb_color_cube(hold_data):
    """🎨 RGB 3D 색상 큐브 시각화 (원본 색상)"""
    if len(hold_data) == 0:
        return None
    
    # RGB 좌표 준비
    rgb_coords = []
    hold_ids = []
    group_labels = []
    colors = []
    
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        rgb = hsv_to_rgb([h, s, v])
        
        rgb_coords.append(rgb)
        hold_ids.append(hold["id"])
        group_labels.append(hold["group"] if hold["group"] is not None else -1)
        colors.append(f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})")
    
    # 3D 산점도
    fig = go.Figure(data=go.Scatter3d(
        x=[coord[0] for coord in rgb_coords],
        y=[coord[1] for coord in rgb_coords],
        z=[coord[2] for coord in rgb_coords],
        mode='markers+text',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.8,
            line=dict(width=2, color='black')
        ),
        text=[f"ID:{hid}" for hid in hold_ids],
        textposition="top center",
        textfont=dict(size=8, color="black"),
        hovertemplate='<b>홀드 ID: %{text}</b><br>' +
                     '그룹: %{marker.color}<br>' +
                     'RGB: (%{x:.0f}, %{y:.0f}, %{z:.0f})<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title="🎨 RGB 3D 색상 큐브에서의 홀드 분포",
        scene=dict(
            xaxis_title="Red (0-255)",
            yaxis_title="Green (0-255)",
            zaxis_title="Blue (0-255)",
            xaxis=dict(range=[0, 255]),
            yaxis=dict(range=[0, 255]),
            zaxis=dict(range=[0, 255])
        ),
        width=800,
        height=600
    )
    
    return fig

def create_pure_rgb_color_cube(hold_data):
    """🎨 순수 3D RGB 색상 큐브 (그룹 정보 없이 실제 평균 색상만)"""
    if len(hold_data) == 0:
        return None
    
    import plotly.graph_objects as go
    
    # RGB 좌표와 홀드 정보 준비
    rgb_coords = []
    hold_ids = []
    hover_texts = []
    
    for hold in hold_data:
        # HSV에서 RGB로 변환된 값 사용
        h, s, v = hold["dominant_hsv"]
        rgb = hsv_to_rgb([h, s, v])
        
        rgb_coords.append(rgb)
        hold_ids.append(hold["id"])
        
        # 호버 텍스트 (HSV→RGB 변환된 값 표시)
        hover_text = f"홀드 {hold['id']}<br>HSV→RGB: ({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})<br>원본 HSV: ({h:.0f}, {s:.0f}, {v:.0f})"
        hover_texts.append(hover_text)
    
    # 3D 산점도 생성
    fig = go.Figure()
    
    # 🎨 각 홀드를 실제 색상으로 표시 (텍스트 제거, 색상만 강조)
    fig.add_trace(go.Scatter3d(
        x=[coord[0] for coord in rgb_coords],  # Red
        y=[coord[1] for coord in rgb_coords],  # Green  
        z=[coord[2] for coord in rgb_coords],  # Blue
        mode='markers',  # 텍스트 제거, 마커만
        marker=dict(
            size=12,  # 크기 증가
            color=[f'rgb({coord[0]:.0f},{coord[1]:.0f},{coord[2]:.0f})' for coord in rgb_coords],  # 실제 RGB 색상
            opacity=0.9,
            line=dict(width=2, color='rgba(0, 0, 0, 0.5)')  # 검은 테두리로 구분
        ),
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_texts,
        name="홀드들"
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title="🎨 순수 3D RGB 색상 큐브 (HSV→RGB 변환)",
        scene=dict(
            xaxis_title="Red (0-255)",
            yaxis_title="Green (0-255)", 
            zaxis_title="Blue (0-255)",
            xaxis=dict(range=[0, 255], dtick=50, showgrid=True, gridcolor="white", gridwidth=0.5),
            yaxis=dict(range=[0, 255], dtick=50, showgrid=True, gridcolor="white", gridwidth=0.5),
            zaxis=dict(range=[0, 255], dtick=50, showgrid=True, gridcolor="white", gridwidth=0.5),
            aspectmode='cube'
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white")
    )
    
    return fig

def create_lab_color_space_visualization(hold_data, selected_hold_id=None, eps=None):
    """🎨 Lab 색상 공간 시각화"""
    import plotly.graph_objects as go
    import numpy as np
    import cv2
    
    # HSV → Lab 변환
    lab_vectors = []
    rgb_colors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        rgb_image = np.uint8([[[rgb_arr[0], rgb_arr[1], rgb_arr[2]]]])
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)[0][0]
        lab_vectors.append([lab_image[0], lab_image[1], lab_image[2]])
        rgb_colors.append(f"rgb({rgb_arr[0]}, {rgb_arr[1]}, {rgb_arr[2]})")
    
    lab_vectors = np.array(lab_vectors)
    
    # Lab 공간에서 3D 시각화
    fig = go.Figure()
    
    # 그룹별로 색상 구분
    groups = {}
    for i, hold in enumerate(hold_data):
        group_id = hold["group"]
        if group_id not in groups:
            groups[group_id] = {"indices": [], "colors": [], "lab": []}
        groups[group_id]["indices"].append(i)
        groups[group_id]["colors"].append(rgb_colors[i])
        groups[group_id]["lab"].append(lab_vectors[i])
    
    # 각 그룹별로 점 추가
    for group_id, group_data in groups.items():
        lab_points = np.array(group_data["lab"])
        fig.add_trace(go.Scatter3d(
            x=lab_points[:, 1],  # a축 (녹색-빨강)
            y=lab_points[:, 2],  # b축 (파랑-노랑)
            z=lab_points[:, 0],  # L축 (명도)
            mode='markers+text',
            marker=dict(
                size=8,
                color=group_data["colors"],
                opacity=0.8
            ),
            text=[f"H{hold_data[i]['id']}" for i in group_data["indices"]],
            textposition="top center",
            textfont=dict(size=10),
            name=f'그룹 {group_id} ({len(group_data["indices"])}개)',
            hovertemplate='홀드 ID: %{text}<br>L: %{z:.0f}<br>a: %{x:.0f}<br>b: %{y:.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="🎨 Lab 색상 공간 시각화 (L:명도, a:녹색-빨강, b:파랑-노랑)",
        scene=dict(
            xaxis_title="a축 (녹색 ← → 빨강)",
            yaxis_title="b축 (파랑 ← → 노랑)",
            zaxis_title="L축 (명도)",
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    
    return fig

def create_cylindrical_hsv_visualization(hold_data, selected_hold_id=None, eps=None):
    """🎨 원통 좌표계 HSV 시각화"""
    import plotly.graph_objects as go
    import numpy as np
    import cv2
    
    # HSV → 원통 좌표계 변환
    cylindrical_vectors = []
    rgb_colors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        
        # 원통 좌표계 변환
        theta = np.radians(h)
        r = s
        z = v * 0.2  # Value 가중치 감소
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        cylindrical_vectors.append([x, y, z])
        
        # RGB 색상 계산
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        rgb_colors.append(f"rgb({rgb_arr[0]}, {rgb_arr[1]}, {rgb_arr[2]})")
    
    cylindrical_vectors = np.array(cylindrical_vectors)
    
    # 원통 좌표계에서 3D 시각화
    fig = go.Figure()
    
    # 그룹별로 색상 구분
    groups = {}
    for i, hold in enumerate(hold_data):
        group_id = hold["group"]
        if group_id not in groups:
            groups[group_id] = {"indices": [], "colors": [], "coords": []}
        groups[group_id]["indices"].append(i)
        groups[group_id]["colors"].append(rgb_colors[i])
        groups[group_id]["coords"].append(cylindrical_vectors[i])
    
    # 각 그룹별로 점 추가
    for group_id, group_data in groups.items():
        coords = np.array(group_data["coords"])
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],  # X (Saturation * cos(Hue))
            y=coords[:, 1],  # Y (Saturation * sin(Hue))
            z=coords[:, 2],  # Z (Value * 0.2)
            mode='markers+text',
            marker=dict(
                size=8,
                color=group_data["colors"],
                opacity=0.8
            ),
            text=[f"H{hold_data[i]['id']}" for i in group_data["indices"]],
            textposition="top center",
            textfont=dict(size=10),
            name=f'그룹 {group_id} ({len(group_data["indices"])}개)',
            hovertemplate='홀드 ID: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="🎨 원통 좌표계 HSV 시각화 (Hue:각도, Saturation:반지름, Value:높이)",
        scene=dict(
            xaxis_title="X (Saturation × cos(Hue))",
            yaxis_title="Y (Saturation × sin(Hue))",
            zaxis_title="Z (Value × 0.2)",
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    
    return fig

def create_custom_color_space_visualization(hold_data, selected_hold_id=None, eps=None):
    """🎨 커스텀 색상 공간 시각화"""
    import plotly.graph_objects as go
    import numpy as np
    import cv2
    
    # 커스텀 색상 공간 변환
    custom_vectors = []
    rgb_colors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        
        # 커스텀 변환 적용
        custom_rgb = custom_color_space_transform([rgb_arr[0], rgb_arr[1], rgb_arr[2]])
        custom_vectors.append(custom_rgb)
        rgb_colors.append(f"rgb({rgb_arr[0]}, {rgb_arr[1]}, {rgb_arr[2]})")
    
    custom_vectors = np.array(custom_vectors)
    
    # 커스텀 공간에서 3D 시각화
    fig = go.Figure()
    
    # 그룹별로 색상 구분
    groups = {}
    for i, hold in enumerate(hold_data):
        group_id = hold["group"]
        if group_id not in groups:
            groups[group_id] = {"indices": [], "colors": [], "coords": []}
        groups[group_id]["indices"].append(i)
        groups[group_id]["colors"].append(rgb_colors[i])
        groups[group_id]["coords"].append(custom_vectors[i])
    
    # 각 그룹별로 점 추가
    for group_id, group_data in groups.items():
        coords = np.array(group_data["coords"])
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],  # R축 (확장됨)
            y=coords[:, 1],  # G축 (확장됨)
            z=coords[:, 2],  # B축 (확장됨)
            mode='markers+text',
            marker=dict(
                size=8,
                color=group_data["colors"],
                opacity=0.8
            ),
            text=[f"H{hold_data[i]['id']}" for i in group_data["indices"]],
            textposition="top center",
            textfont=dict(size=10),
            name=f'그룹 {group_id} ({len(group_data["indices"])}개)',
            hovertemplate='홀드 ID: %{text}<br>R: %{x:.0f}<br>G: %{y:.0f}<br>B: %{z:.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="🎨 커스텀 색상 공간 시각화 (주요 색상 간 거리 확장)",
        scene=dict(
            xaxis_title="R축 (확장됨)",
            yaxis_title="G축 (확장됨)",
            zaxis_title="B축 (확장됨)",
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    
    return fig

def rgb_to_lch(rgb):
    """RGB → LCh 변환 (Hue wrap 해결)"""
    import cv2
    import numpy as np
    
    # RGB → Lab 변환
    rgb_image = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)[0][0]
    L, a, b = lab_image[0], lab_image[1], lab_image[2]
    
    # Lab → LCh 변환
    C = np.sqrt(a*a + b*b)  # Chroma
    h = np.arctan2(b, a)    # Hue (라디안)
    
    # Hue를 cos, sin으로 변환하여 wrap 문제 해결
    cos_h = np.cos(h)
    sin_h = np.sin(h)
    
    return [L, C*cos_h, C*sin_h, C]

def ciede2000_distance_simple(rgb1, rgb2):
    """간단한 CIEDE2000 거리 계산"""
    import cv2
    import numpy as np
    
    # RGB → Lab 변환
    lab1 = cv2.cvtColor(np.uint8([[[rgb1[0], rgb1[1], rgb1[2]]]]), cv2.COLOR_RGB2Lab)[0][0]
    lab2 = cv2.cvtColor(np.uint8([[[rgb2[0], rgb2[1], rgb2[2]]]]), cv2.COLOR_RGB2Lab)[0][0]
    
    # 간단한 가중치 적용 (L*0.3, a*1.5, b*1.5)
    L1, a1, b1 = lab1[0]*0.3, lab1[1]*1.5, lab1[2]*1.5
    L2, a2, b2 = lab2[0]*0.3, lab2[1]*1.5, lab2[2]*1.5
    
    # 유클리드 거리
    distance = np.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)
    return distance

def lch_cosine_dbscan_clustering(hold_data, vectors, eps=0.3):
    """🎨 LCh 변환 + Cosine 거리 DBSCAN"""
    from sklearn.cluster import DBSCAN
    import numpy as np
    
    # HSV → RGB → LCh 변환
    lch_vectors = []
    rgb_colors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        
        # LCh 변환 (Hue wrap 해결)
        lch = rgb_to_lch([rgb_arr[0], rgb_arr[1], rgb_arr[2]])
        lch_vectors.append(lch)
        rgb_colors.append([rgb_arr[0], rgb_arr[1], rgb_arr[2]])
    
    lch_vectors = np.array(lch_vectors)
    
    # Cosine 거리로 DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=1, metric='cosine').fit(lch_vectors)
    
    # 결과 적용
    for i, hold in enumerate(hold_data):
        hold["group"] = int(clustering.labels_[i])
    
    print(f"🎨 LCh+Cosine 클러스터링 완료: {len(set(clustering.labels_))}개 그룹")
    return hold_data

def ciede2000_mds_dbscan_clustering(hold_data, vectors, eps=0.3):
    """🎨 CIEDE2000 + MDS + DBSCAN"""
    from sklearn.cluster import DBSCAN
    from sklearn.manifold import MDS
    import numpy as np
    
    # HSV → RGB 변환
    rgb_colors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        rgb_colors.append([rgb_arr[0], rgb_arr[1], rgb_arr[2]])
    
    rgb_colors = np.array(rgb_colors)
    
    # CIEDE2000 거리 행렬 계산
    n = len(rgb_colors)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = ciede2000_distance_simple(rgb_colors[i], rgb_colors[j])
    
    # MDS로 2D 변환 (균등한 거리 분포)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(distance_matrix)
    
    # MDS 좌표로 DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=1, metric='euclidean').fit(mds_coords)
    
    # 결과 적용
    for i, hold in enumerate(hold_data):
        hold["group"] = int(clustering.labels_[i])
    
    print(f"🎨 CIEDE2000+MDS 클러스터링 완료: {len(set(clustering.labels_))}개 그룹")
    return hold_data

def create_mds_visualization(hold_data, selected_hold_id=None, eps=None):
    """🎨 MDS 2D 시각화 (균등한 거리 분포)"""
    import plotly.graph_objects as go
    import numpy as np
    import cv2
    from sklearn.manifold import MDS
    
    # HSV → RGB 변환
    rgb_colors = []
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        hsv_arr = np.uint8([[[h, s, v]]])
        rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        rgb_colors.append([rgb_arr[0], rgb_arr[1], rgb_arr[2]])
    
    rgb_colors = np.array(rgb_colors)
    
    # CIEDE2000 거리 행렬 계산
    n = len(rgb_colors)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = ciede2000_distance_simple(rgb_colors[i], rgb_colors[j])
    
    # MDS로 2D 변환
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(distance_matrix)
    
    # 2D 시각화
    fig = go.Figure()
    
    # 그룹별로 색상 구분
    groups = {}
    for i, hold in enumerate(hold_data):
        group_id = hold["group"]
        if group_id not in groups:
            groups[group_id] = {"indices": [], "colors": [], "coords": []}
        groups[group_id]["indices"].append(i)
        groups[group_id]["colors"].append(f"rgb({rgb_colors[i][0]}, {rgb_colors[i][1]}, {rgb_colors[i][2]})")
        groups[group_id]["coords"].append(mds_coords[i])
    
    # 각 그룹별로 점 추가
    for group_id, group_data in groups.items():
        coords = np.array(group_data["coords"])
        fig.add_trace(go.Scatter(
            x=coords[:, 0],  # MDS X축
            y=coords[:, 1],  # MDS Y축
            mode='markers+text',
            marker=dict(
                size=12,
                color=group_data["colors"],
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[f"H{hold_data[i]['id']}" for i in group_data["indices"]],
            textposition="top center",
            textfont=dict(size=10, color='black'),
            name=f'그룹 {group_id} ({len(group_data["indices"])}개)',
            hovertemplate='홀드 ID: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="🎨 MDS 2D 시각화 (CIEDE2000 거리 기반, 균등한 분포)",
        xaxis_title="MDS 차원 1",
        yaxis_title="MDS 차원 2",
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig

def create_rgb_color_cube_with_groups(hold_data, selected_hold_id=None, eps=None):
    """🎯 RGB 3D 색상 큐브 시각화 (그룹별 색상 표시)"""
    if len(hold_data) == 0:
        return None
    
    # 그룹별 색상 정의
    group_colors = [
        'red', 'blue', 'green', 'yellow', 'purple', 'orange', 
        'pink', 'cyan', 'lime', 'magenta', 'brown', 'gray', 'black'
    ]
    
    # RGB 좌표와 그룹 정보 준비
    rgb_coords = []
    hold_ids = []
    group_labels = []
    group_colors_list = []
    hover_texts = []
    
    for hold in hold_data:
        h, s, v = hold["dominant_hsv"]
        rgb = hsv_to_rgb([h, s, v])  # HSV에서 RGB로 변환
        group_id = hold["group"] if hold["group"] is not None else -1
        
        rgb_coords.append(rgb)
        hold_ids.append(hold["id"])
        group_labels.append(group_id)
        
        # 그룹별 색상 할당 (group_id가 문자열인 경우 처리)
        try:
            if isinstance(group_id, str) and group_id.startswith('g'):
                group_num = int(group_id[1:])  # 'g0' -> 0
            else:
                group_num = int(group_id)
            
            if group_num >= 0 and group_num < len(group_colors):
                group_colors_list.append(group_colors[group_num])
            else:
                group_colors_list.append('gray')  # 노이즈나 미분류 그룹
        except (ValueError, TypeError):
            group_colors_list.append('gray')  # 변환 실패 시 회색
        
        # 호버 텍스트 생성 (실제 RGB 값과 HSV 값 모두 표시)
        h, s, v = hold["dominant_hsv"]
        hover_texts.append(f"홀드 ID: {hold['id']}<br>그룹: G{group_id}<br>실제 RGB: ({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})<br>HSV: ({h:.0f}, {s:.0f}, {v:.0f})")
    
    # 3D 산점도 (선택된 홀드가 있으면 투명도 낮춤)
    base_opacity = 0.4 if selected_hold_id is not None else 0.9
    
    fig = go.Figure(data=go.Scatter3d(
        x=[coord[0] for coord in rgb_coords],
        y=[coord[1] for coord in rgb_coords],
        z=[coord[2] for coord in rgb_coords],
        mode='markers',  # 텍스트 제거, 실제 색상만 표시
        marker=dict(
            size=12,  # 크기 증가
            color=[f'rgb({coord[0]:.0f},{coord[1]:.0f},{coord[2]:.0f})' for coord in rgb_coords],  # 실제 RGB 색상
            opacity=base_opacity,
            line=dict(width=2, color='rgba(0, 0, 0, 0.5)')
        ),
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_texts,
        name='모든 홀드'
    ))
    
    # 🚨 선택된 홀드가 있으면 eps 구 표시
    if selected_hold_id is not None and eps is not None:
        # 선택된 홀드 찾기
        selected_hold = None
        selected_idx = None
        for i, hold in enumerate(hold_data):
            if hold["id"] == selected_hold_id:
                selected_hold = hold
                selected_idx = i
                break
        
        if selected_hold is not None:
            # 선택된 홀드의 RGB 좌표
            h, s, v = selected_hold["dominant_hsv"]
            selected_rgb = hsv_to_rgb([h, s, v])
            x_center, y_center, z_center = selected_rgb
            
            # eps 구 생성 (와이어프레임 - 더 가벼운 표현)
            import numpy as np
            
            # 구의 와이어프레임을 그리기 위한 원들 (경도선 + 위도선)
            theta = np.linspace(0, 2 * np.pi, 30)
            phi = np.linspace(0, np.pi, 15)
            
            # 경도선 (세로 원들)
            for i in range(0, 360, 30):  # 30도 간격
                angle = np.radians(i)
                x_line = x_center + eps * np.cos(theta) * np.sin(angle)
                y_line = y_center + eps * np.sin(theta) * np.sin(angle)
                z_line = z_center + eps * np.cos(angle) * np.ones_like(theta)
                
                fig.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode='lines',
                    line=dict(color='rgba(100, 100, 255, 0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # 위도선 (가로 원들)
            for i in range(0, 180, 30):  # 30도 간격
                angle = np.radians(i)
                x_line = x_center + eps * np.cos(theta) * np.sin(angle)
                y_line = y_center + eps * np.sin(theta) * np.sin(angle)
                z_line = z_center + eps * np.cos(angle) * np.ones_like(theta)
                
                fig.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode='lines',
                    line=dict(color='rgba(100, 100, 255, 0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # 선택된 홀드를 더 크게 표시 (밝은 노란색 - 눈에 띄게)
            fig.add_trace(go.Scatter3d(
                x=[x_center],
                y=[y_center],
                z=[z_center],
                mode='markers+text',
                marker=dict(
                    size=18,
                    color='rgba(255, 255, 0, 1.0)',  # 밝은 노란색
                    opacity=1.0,
                    line=dict(width=4, color='black'),
                    symbol='diamond'  # 다이아몬드 모양
                ),
                text=[f"🎯{selected_hold_id}"],
                textposition="top center",
                textfont=dict(size=12, color="black", family="Arial Black"),
                name=f'🎯 선택된 홀드 {selected_hold_id}',
                hovertemplate=f'🎯 선택된 홀드 {selected_hold_id}<br>그룹 G{selected_hold["group"]}<br>RGB({x_center:.0f}, {y_center:.0f}, {z_center:.0f})<extra></extra>'
            ))
            
            # eps 구 안에 있는 홀드들 표시
            inside_holds = []
            outside_holds = []
            for i, hold in enumerate(hold_data):
                if i != selected_idx:
                    h, s, v = hold["dominant_hsv"]
                    hold_rgb = hsv_to_rgb([h, s, v])
                    dist = np.sqrt(np.sum((np.array(selected_rgb) - np.array(hold_rgb))**2))
                    
                    if dist <= eps:
                        inside_holds.append((hold["id"], hold["group"], dist, hold_rgb))
                    else:
                        outside_holds.append((hold["id"], hold["group"], dist, hold_rgb))
            
            # 🚨 검증 정보를 streamlit에 표시 (return 후에 표시되도록 저장)
            import streamlit as st
            st.session_state['eps_validation_info'] = {
                'selected_hold_id': selected_hold["id"],
                'selected_group': selected_hold["group"],
                'selected_rgb': selected_rgb,
                'eps': eps,
                'inside_holds': inside_holds,
                'outside_holds': outside_holds
            }
            
            if inside_holds:
                inside_x = [rgb[0] for _, _, _, rgb in inside_holds]
                inside_y = [rgb[1] for _, _, _, rgb in inside_holds]
                inside_z = [rgb[2] for _, _, _, rgb in inside_holds]
                inside_texts = [f"홀드{hold_id}<br>그룹G{group_id}<br>거리{dist:.2f}" 
                               for hold_id, group_id, dist, _ in inside_holds]
                
                # 같은 그룹과 다른 그룹 구분
                inside_same_group = [(hold_id, group_id, dist, rgb) for hold_id, group_id, dist, rgb in inside_holds 
                                     if group_id == selected_hold["group"]]
                inside_diff_group = [(hold_id, group_id, dist, rgb) for hold_id, group_id, dist, rgb in inside_holds 
                                     if group_id != selected_hold["group"]]
                
                # 같은 그룹 (밝은 초록색)
                if inside_same_group:
                    fig.add_trace(go.Scatter3d(
                        x=[rgb[0] for _, _, _, rgb in inside_same_group],
                        y=[rgb[1] for _, _, _, rgb in inside_same_group],
                        z=[rgb[2] for _, _, _, rgb in inside_same_group],
                        mode='markers+text',
                        marker=dict(
                            size=14,
                            color='rgba(100, 255, 100, 0.9)',  # 밝은 초록색
                            opacity=1.0,
                            line=dict(width=3, color='darkgreen')
                        ),
                        text=[f"{hold_id}" for hold_id, _, _, _ in inside_same_group],
                        textposition="top center",
                        textfont=dict(size=10, color="darkgreen", family="Arial Black"),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=[f"✅홀드{hold_id}<br>그룹G{group_id}<br>거리{dist:.2f}" 
                                  for hold_id, group_id, dist, _ in inside_same_group],
                        name=f'✅ 같은 그룹 ({len(inside_same_group)}개)'
                    ))
                
                # 다른 그룹 (밝은 빨간색 - 경고)
                if inside_diff_group:
                    fig.add_trace(go.Scatter3d(
                        x=[rgb[0] for _, _, _, rgb in inside_diff_group],
                        y=[rgb[1] for _, _, _, rgb in inside_diff_group],
                        z=[rgb[2] for _, _, _, rgb in inside_diff_group],
                        mode='markers+text',
                        marker=dict(
                            size=14,
                            color='rgba(255, 100, 100, 0.9)',  # 밝은 빨간색
                            opacity=1.0,
                            line=dict(width=3, color='darkred')
                        ),
                        text=[f"{hold_id}" for hold_id, _, _, _ in inside_diff_group],
                        textposition="top center",
                        textfont=dict(size=10, color="darkred", family="Arial Black"),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=[f"❌홀드{hold_id}<br>그룹G{group_id}<br>거리{dist:.2f}" 
                                  for hold_id, group_id, dist, _ in inside_diff_group],
                        name=f'❌ 다른 그룹 ({len(inside_diff_group)}개)'
                    ))
    
    fig.update_layout(
        title="🎯 RGB 3D 색상 큐브에서의 그룹핑 결과 (그룹별 색상 표시)",
        scene=dict(
            xaxis_title="Red (0-255)",
            yaxis_title="Green (0-255)",
            zaxis_title="Blue (0-255)",
            xaxis=dict(range=[0, 255]),
            yaxis=dict(range=[0, 255]),
            zaxis=dict(range=[0, 255]),
            aspectmode='cube'  # 축 비율을 정육면체로 고정
        ),
        width=800,
        height=600
    )
    
    return fig

def create_color_category_statistics(hold_data):
    """🎨 색상 카테고리별 통계 정보"""
    categories = set(h.get("color_category", "unknown") for h in hold_data)
    stats = []
    
    for category in sorted(categories):
        category_holds = [h for h in hold_data if h.get("color_category") == category]
        hsv_values = [h["dominant_hsv"] for h in category_holds]
        
        h_values = [hsv[0] for hsv in hsv_values]
        s_values = [hsv[1] for hsv in hsv_values]
        v_values = [hsv[2] for hsv in hsv_values]
        
        stats.append({
            "카테고리": category,
            "홀드수": len(category_holds),
            "홀드ID": [h["id"] for h in category_holds],
            "Hue평균": round(np.mean(h_values), 1) if h_values else 0,
            "Hue범위": round(max(h_values) - min(h_values), 1) if h_values else 0,
            "Sat평균": round(np.mean(s_values), 1) if s_values else 0,
            "Sat범위": round(max(s_values) - min(s_values), 1) if s_values else 0,
            "Val평균": round(np.mean(v_values), 1) if v_values else 0,
            "Val범위": round(max(v_values) - min(v_values), 1) if v_values else 0
        })
    
    return stats

def create_group_statistics(hold_data):
    """📊 그룹별 통계 정보"""
    groups = [h["group"] for h in hold_data if h["group"] is not None and h["group"] >= 0]
    unique_groups = sorted(set(groups))
    
    stats = []
    for group in unique_groups:
        group_holds = [h for h in hold_data if h["group"] == group]
        hsv_values = [h["dominant_hsv"] for h in group_holds]
        
        h_values = [hsv[0] for hsv in hsv_values]
        s_values = [hsv[1] for hsv in hsv_values]
        v_values = [hsv[2] for hsv in hsv_values]
        
        stats.append({
            "group": group,
            "count": len(group_holds),
            "hold_ids": [h["id"] for h in group_holds],
            "hue_mean": np.mean(h_values),
            "hue_std": np.std(h_values),
            "sat_mean": np.mean(s_values),
            "sat_std": np.std(s_values),
            "val_mean": np.mean(v_values),
            "val_std": np.std(v_values),
            "hue_range": max(h_values) - min(h_values),
            "sat_range": max(s_values) - min(s_values),
            "val_range": max(v_values) - min(v_values)
        })
    
    return stats

# ============================================
# 🎯 난이도 및 유형 추정 함수
# ============================================

def estimate_difficulty(hold_data):
    """
    홀드 데이터를 기반으로 문제 난이도를 추정합니다.
    
    Args:
        hold_data: 홀드 정보 리스트
        
    Returns:
        dict: {
            'grade': 난이도 등급 (예: "V3-V4"),
            'grade_level': 난이도 레벨 (예: "중급"),
            'score': 난이도 점수 (0-12),
            'confidence': 신뢰도 (0-1),
            'factors': 판단 근거
        }
    """
    if not hold_data or len(hold_data) == 0:
        return {
            'grade': "알 수 없음",
            'grade_level': "알 수 없음",
            'score': 0,
            'confidence': 0.0,
            'factors': {}
        }
    
    score = 0
    factors = {}
    
    # 1. 홀드 개수 분석
    num_holds = len(hold_data)
    factors['num_holds'] = num_holds
    if num_holds < 8:
        score += 3
        factors['num_holds_impact'] = "적음 (어려움)"
    elif num_holds < 12:
        score += 2
        factors['num_holds_impact'] = "보통"
    else:
        score += 1
        factors['num_holds_impact'] = "많음 (쉬움)"
    
    # 2. 홀드 크기 분석
    areas = [h.get('area', 2000) for h in hold_data]
    avg_area = np.mean(areas)
    factors['avg_hold_size'] = f"{avg_area:.0f}px²"
    
    if avg_area < 1500:
        score += 3
        factors['hold_size_impact'] = "작음 (어려움)"
    elif avg_area < 2500:
        score += 2
        factors['hold_size_impact'] = "중형"
    else:
        score += 1
        factors['hold_size_impact'] = "큼 (쉬움)"
    
    # 3. 홀드 간격 분석
    centers = np.array([h['center'] for h in hold_data])
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)
    
    if distances:
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        factors['avg_distance'] = f"{avg_distance:.0f}px"
        factors['max_distance'] = f"{max_distance:.0f}px"
        
        if avg_distance > 150:
            score += 3
            factors['distance_impact'] = "넓음 (어려움)"
        elif avg_distance > 100:
            score += 2
            factors['distance_impact'] = "보통"
        else:
            score += 1
            factors['distance_impact'] = "좁음 (쉬움)"
    
    # 4. 높이 분포 분석
    heights = [h['center'][1] for h in hold_data]
    height_range = max(heights) - min(heights)
    factors['height_range'] = f"{height_range:.0f}px"
    
    if height_range > 500:
        score += 2
        factors['height_impact'] = "높음 (어려움)"
    elif height_range > 300:
        score += 1
        factors['height_impact'] = "보통"
    else:
        factors['height_impact'] = "낮음 (쉬움)"
    
    # 5. 작은 홀드 비율
    small_holds = [h for h in hold_data if h.get('area', 2000) < 1500]
    small_ratio = len(small_holds) / len(hold_data)
    factors['small_hold_ratio'] = f"{small_ratio * 100:.0f}%"
    
    if small_ratio > 0.6:
        score += 1
        factors['small_hold_impact'] = "많음 (크림프 필요)"
    
    # 점수 → 난이도 매핑
    factors['total_score'] = score
    
    if score <= 4:
        grade = "V0-V1"
        grade_level = "초급"
        confidence = 0.55
    elif score <= 6:
        grade = "V2-V3"
        grade_level = "초중급"
        confidence = 0.65
    elif score <= 8:
        grade = "V4-V5"
        grade_level = "중급"
        confidence = 0.60
    elif score <= 10:
        grade = "V6-V7"
        grade_level = "중고급"
        confidence = 0.50
    else:
        grade = "V8+"
        grade_level = "고급"
        confidence = 0.45
    
    return {
        'grade': grade,
        'grade_level': grade_level,
        'score': score,
        'confidence': confidence,
        'factors': factors
    }


def estimate_climb_type(hold_data):
    """
    홀드 데이터를 기반으로 문제 유형을 추정합니다.
    
    Args:
        hold_data: 홀드 정보 리스트
        
    Returns:
        dict: {
            'types': 유형 리스트 (예: ["밸런스", "다이나믹"]),
            'primary_type': 주요 유형,
            'confidence': 신뢰도 (0-1),
            'characteristics': 특징 설명
        }
    """
    if not hold_data or len(hold_data) < 3:
        return {
            'types': [],
            'primary_type': "알 수 없음",
            'confidence': 0.0,
            'characteristics': {}
        }
    
    types = []
    characteristics = {}
    
    centers = np.array([h['center'] for h in hold_data])
    areas = [h.get('area', 2000) for h in hold_data]
    
    # 1. 수평/수직 분산 비율 (밸런스 vs 캠퍼싱)
    horizontal_std = np.std(centers[:, 0])  # x축
    vertical_std = np.std(centers[:, 1])    # y축
    
    if vertical_std > 0:
        ratio = horizontal_std / vertical_std
        characteristics['horizontal_vertical_ratio'] = f"{ratio:.2f}"
        
        if ratio > 1.5:
            types.append("밸런스")
            characteristics['balance_note'] = "수평 이동이 많은 문제"
        elif ratio < 0.7:
            types.append("캠퍼싱")
            characteristics['campus_note'] = "수직 상승 위주의 문제"
    
    # 2. 홀드 간 최대 거리 (다이나믹)
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)
    
    if distances:
        max_gap = np.max(distances)
        avg_distance = np.mean(distances)
        characteristics['max_gap'] = f"{max_gap:.0f}px"
        
        if max_gap > 200 and max_gap > avg_distance * 1.8:
            types.append("다이나믹")
            characteristics['dynamic_note'] = "큰 점프 구간 존재"
    
    # 3. 작은 홀드 비율 (크림프)
    small_holds = [h for h in hold_data if h.get('area', 2000) < 1500]
    small_ratio = len(small_holds) / len(hold_data)
    characteristics['small_hold_ratio'] = f"{small_ratio * 100:.0f}%"
    
    if small_ratio > 0.5:
        types.append("크림프")
        characteristics['crimp_note'] = "작은 홀드가 많음"
    
    # 4. 홀드 밀집도 (테크니컬)
    if len(hold_data) > 12 and np.mean(areas) < 2000:
        types.append("테크니컬")
        characteristics['technical_note'] = "정교한 움직임 필요"
    
    # 5. 수평 이동 거리
    horizontal_range = np.max(centers[:, 0]) - np.min(centers[:, 0])
    vertical_range = np.max(centers[:, 1]) - np.min(centers[:, 1])
    characteristics['horizontal_range'] = f"{horizontal_range:.0f}px"
    characteristics['vertical_range'] = f"{vertical_range:.0f}px"
    
    if horizontal_range > vertical_range * 1.3:
        if "밸런스" not in types:
            types.append("트래버스")
            characteristics['traverse_note'] = "좌우 이동이 많은 문제"
    
    # 유형이 없으면 "일반"
    if not types:
        types = ["일반"]
        characteristics['general_note'] = "특별한 특징이 없는 균형잡힌 문제"
    
    # 신뢰도 계산 (유형이 많을수록 신뢰도 감소)
    confidence = 0.6 if len(types) <= 2 else 0.5
    
    return {
        'types': types,
        'primary_type': types[0] if types else "알 수 없음",
        'confidence': confidence,
        'characteristics': characteristics
    }


def analyze_problem(hold_data, group_id=None, wall_angle=None):
    """
    🧗‍♀️ AI 기반 클라이밍 문제 분석 (난이도 + 유형 추정)
    
    Args:
        hold_data: 전체 홀드 정보 리스트
        group_id: 분석할 그룹 ID (None이면 전체)
        wall_angle: 벽 각도 ("overhang", "slab", "face") - 사용자 입력
        
    Returns:
        dict: {
            'difficulty': 난이도 정보,
            'climb_type': 유형 정보,
            'statistics': 기본 통계
        }
    """
    # 그룹 필터링
    if group_id is not None:
        filtered_holds = [h for h in hold_data if h.get('group') == group_id]
    else:
        filtered_holds = hold_data
    
    if not filtered_holds or len(filtered_holds) < 3:
        return None
    
    # 🎯 1. 난이도 분석
    difficulty = analyze_difficulty(filtered_holds)
    
    # 🧗‍♀️ 2. 문제 유형 분석
    climb_type = analyze_climbing_type(filtered_holds, wall_angle)
    
    # 📊 3. 기본 통계
    centers = np.array([h['center'] for h in filtered_holds])
    areas = np.array([h.get('area', 2000) for h in filtered_holds])
    
    # 거리 분석
    distances = []
    if len(filtered_holds) > 1:
        for i, h1 in enumerate(filtered_holds):
            for h2 in filtered_holds[i+1:]:
                dist = np.linalg.norm(np.array(h1['center']) - np.array(h2['center']))
                distances.append(dist)
    
    statistics = {
        'num_holds': len(filtered_holds),
        'avg_hold_size': f"{np.mean(areas):.0f}px²",
        'total_height': f"{np.max(centers[:, 1]) - np.min(centers[:, 1]):.0f}px",
        'total_width': f"{np.max(centers[:, 0]) - np.min(centers[:, 0]):.0f}px",
        'avg_distance': f"{np.mean(distances):.0f}px" if distances else "0px",
        'max_distance': f"{np.max(distances):.0f}px" if distances else "0px"
    }
    
    return {
        'difficulty': difficulty,
        'climb_type': climb_type,
        'statistics': statistics
    }

def analyze_difficulty(filtered_holds):
    """🎯 난이도 분석 (개선 버전)"""
    num_holds = len(filtered_holds)
    areas = np.array([h.get('area', 2000) for h in filtered_holds])
    centers = np.array([h['center'] for h in filtered_holds])
    
    # 거리 계산
    distances = []
    consecutive_distances = []  # 인접 홀드 간 거리
    if num_holds > 1:
        # 모든 홀드 간 거리
        for i, h1 in enumerate(filtered_holds):
            for h2 in filtered_holds[i+1:]:
                dist = np.linalg.norm(np.array(h1['center']) - np.array(h2['center']))
                distances.append(dist)
        
        # 높이 순으로 정렬하여 연속 거리 계산
        sorted_holds = sorted(filtered_holds, key=lambda h: h['center'][1], reverse=True)
        for i in range(len(sorted_holds) - 1):
            dist = np.linalg.norm(
                np.array(sorted_holds[i]['center']) - np.array(sorted_holds[i+1]['center'])
            )
            consecutive_distances.append(dist)
    
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    max_distance = max(distances) if distances else 0
    avg_distance = np.mean(distances) if distances else 0
    avg_consecutive_distance = np.mean(consecutive_distances) if consecutive_distances else 0
    
    # 홀드 크기 분산 (일관성)
    area_std = np.std(areas)
    
    # 높이 변화
    heights = [h['center'][1] for h in filtered_holds]
    height_range = max(heights) - min(heights) if num_holds > 1 else 0
    
    # 수평 변화
    horizontal_coords = [h['center'][0] for h in filtered_holds]
    horizontal_range = max(horizontal_coords) - min(horizontal_coords) if num_holds > 1 else 0
    
    difficulty_score = 0
    factors = {}
    
    # 1. 홀드 크기 분석 (가중치 증가)
    small_hold_ratio = len([a for a in areas if a < 1200]) / num_holds
    if min_area < 600 or avg_area < 1000:
        difficulty_score += 5
        hold_size_level = "매우 작음 (크림프)"
        factors['hold_size'] = f"극소형 홀드 (평균 {int(avg_area)}px²)"
    elif avg_area < 1500:
        difficulty_score += 4
        hold_size_level = "작음"
        factors['hold_size'] = f"작은 홀드 (평균 {int(avg_area)}px²)"
    elif avg_area < 2500:
        difficulty_score += 2
        hold_size_level = "보통"
        factors['hold_size'] = f"보통 크기 홀드 (평균 {int(avg_area)}px²)"
    elif avg_area < 4000:
        difficulty_score += 1
        hold_size_level = "큼"
        factors['hold_size'] = f"큰 홀드 (평균 {int(avg_area)}px²)"
    else:
        difficulty_score += 0
        hold_size_level = "매우 큼 (쥬그)"
        factors['hold_size'] = f"매우 큰 홀드 (평균 {int(avg_area)}px²)"
    
    # 2. 연속 홀드 간격 분석 (실제 등반 경로)
    if avg_consecutive_distance > 200:
        difficulty_score += 5
        distance_level = "매우 큰 점프"
        factors['distance'] = f"다이나믹한 큰 점프 (평균 {int(avg_consecutive_distance)}px)"
    elif avg_consecutive_distance > 150:
        difficulty_score += 4
        distance_level = "큰 점프"
        factors['distance'] = f"큰 점프 필요 (평균 {int(avg_consecutive_distance)}px)"
    elif avg_consecutive_distance > 100:
        difficulty_score += 2
        distance_level = "보통 간격"
        factors['distance'] = f"보통 간격 (평균 {int(avg_consecutive_distance)}px)"
    elif avg_consecutive_distance > 60:
        difficulty_score += 1
        distance_level = "좁은 간격"
        factors['distance'] = f"좁은 간격 (평균 {int(avg_consecutive_distance)}px)"
    else:
        difficulty_score += 0
        distance_level = "매우 좁은 간격"
        factors['distance'] = f"매우 좁은 간격 (평균 {int(avg_consecutive_distance)}px)"
    
    # 3. 홀드 개수 분석 (적당한 개수가 적당한 난이도)
    if num_holds < 4:
        difficulty_score += 4
        holds_level = "매우 적음"
        factors['num_holds'] = f"{num_holds}개 - 극소수 홀드로 매우 어려움"
    elif num_holds < 6:
        difficulty_score += 3
        holds_level = "적음"
        factors['num_holds'] = f"{num_holds}개 - 적은 홀드로 어려움"
    elif num_holds < 10:
        difficulty_score += 1
        holds_level = "보통"
        factors['num_holds'] = f"{num_holds}개 - 적당한 개수"
    elif num_holds < 15:
        difficulty_score += 0
        holds_level = "많음"
        factors['num_holds'] = f"{num_holds}개 - 많은 홀드로 쉬움"
    else:
        difficulty_score -= 1
        holds_level = "매우 많음"
        factors['num_holds'] = f"{num_holds}개 - 매우 많은 홀드로 쉬움"
    
    # 4. 높이 변화 분석
    if height_range > 600:
        difficulty_score += 3
        height_level = "매우 큰 변화"
        factors['height'] = f"높이 변화 {int(height_range)}px - 체력 소모 큼"
    elif height_range > 400:
        difficulty_score += 2
        height_level = "큰 변화"
        factors['height'] = f"높이 변화 {int(height_range)}px - 보통"
    elif height_range > 200:
        difficulty_score += 1
        height_level = "보통 변화"
        factors['height'] = f"높이 변화 {int(height_range)}px - 적당함"
    else:
        height_level = "작은 변화"
        factors['height'] = f"높이 변화 {int(height_range)}px - 트래버스"
    
    # 5. 수평 변화 (트래버스)
    if horizontal_range > 500 and height_range < 200:
        difficulty_score += 2
        factors['traverse'] = f"긴 트래버스 (수평 {int(horizontal_range)}px)"
    
    # 6. 홀드 크기 일관성
    if area_std > 1000:
        difficulty_score += 1
        factors['consistency'] = "홀드 크기 편차가 커서 적응 어려움"
    
    # V-등급 매핑 (더 세밀하게)
    difficulty_score = max(0, difficulty_score)  # 음수 방지
    
    if difficulty_score <= 2:
        grade = "V0"
        level = "입문"
    elif difficulty_score <= 4:
        grade = "V1"
        level = "초급"
    elif difficulty_score <= 6:
        grade = "V2"
        level = "초급+"
    elif difficulty_score <= 8:
        grade = "V3"
        level = "초중급"
    elif difficulty_score <= 10:
        grade = "V4"
        level = "중급"
    elif difficulty_score <= 12:
        grade = "V5"
        level = "중급+"
    elif difficulty_score <= 14:
        grade = "V6"
        level = "중고급"
    elif difficulty_score <= 16:
        grade = "V7"
        level = "고급"
    elif difficulty_score <= 18:
        grade = "V8"
        level = "고급+"
    else:
        grade = "V9+"
        level = "전문가"
    
    # 신뢰도 계산 (더 보수적)
    confidence = 0.3 + min(num_holds / 20, 0.3)  # 30% ~ 60%
    
    return {
        "grade": grade,
        "level": level,
        "score": difficulty_score,
        "confidence": confidence,
        "factors": factors,
        "details": {
            "hold_size": hold_size_level,
            "distance": distance_level,
            "num_holds": holds_level,
            "height_change": height_level
        }
    }

def analyze_climbing_type(filtered_holds, wall_angle=None):
    """🧗‍♀️ 클라이밍 문제 유형 분석 (개선 버전)"""
    
    num_holds = len(filtered_holds)
    centers = np.array([h['center'] for h in filtered_holds])
    areas = np.array([h.get('area', 2000) for h in filtered_holds])
    
    # 기본 통계
    horizontal_coords = centers[:, 0]
    vertical_coords = centers[:, 1]
    horizontal_std = np.std(horizontal_coords) if num_holds > 1 else 0
    vertical_std = np.std(vertical_coords) if num_holds > 1 else 0
    horizontal_range = np.ptp(horizontal_coords) if num_holds > 1 else 0
    vertical_range = np.ptp(vertical_coords) if num_holds > 1 else 0
    avg_area = np.mean(areas)
    min_area = np.min(areas)
    
    # 거리 분석
    distances = []
    consecutive_distances = []
    if num_holds > 1:
        # 모든 거리
        for i, h1 in enumerate(filtered_holds):
            for h2 in filtered_holds[i+1:]:
                dist = np.linalg.norm(np.array(h1['center']) - np.array(h2['center']))
                distances.append(dist)
        
        # 연속 거리 (높이 순)
        sorted_holds = sorted(filtered_holds, key=lambda h: h['center'][1], reverse=True)
        for i in range(len(sorted_holds) - 1):
            dist = np.linalg.norm(
                np.array(sorted_holds[i]['center']) - np.array(sorted_holds[i+1]['center'])
            )
            consecutive_distances.append(dist)
    
    max_distance = max(distances) if distances else 0
    avg_distance = np.mean(distances) if distances else 0
    avg_consecutive = np.mean(consecutive_distances) if consecutive_distances else 0
    
    types = []
    characteristics = {}
    confidence_factors = []
    
    # 🎯 1. 이동 패턴 분석
    movement_ratio = horizontal_range / (vertical_range + 1)  # 수평/수직 비율
    
    # 트래버스 (수평 이동)
    if horizontal_range > 400 and vertical_range < 250:
        types.append("트래버스")
        characteristics['traverse'] = f"긴 트래버스 (수평 {int(horizontal_range)}px)"
        confidence_factors.append("traverse_pattern")
    # 수직 등반
    elif vertical_range > 400 and horizontal_range < 250:
        types.append("수직등반")
        characteristics['vertical'] = f"수직 등반 (높이 {int(vertical_range)}px)"
        confidence_factors.append("vertical_pattern")
    # 대각선 이동
    elif movement_ratio > 0.5 and movement_ratio < 2.0:
        types.append("대각선")
        characteristics['diagonal'] = "대각선 이동이 많은 문제"
        confidence_factors.append("diagonal_pattern")
    
    # 🎯 2. 다이나믹 vs 스태틱
    dynamic_score = 0
    static_score = 0
    
    # 다이나믹 (큰 점프)
    if max_distance > 220:
        dynamic_score += 4
        types.append("다이노")
        characteristics['dyno'] = f"매우 큰 점프 (최대 {int(max_distance)}px)"
        confidence_factors.append("dyno")
    elif max_distance > 180:
        dynamic_score += 3
        types.append("다이나믹")
        characteristics['dynamic'] = f"큰 점프 필요 (최대 {int(max_distance)}px)"
        confidence_factors.append("dynamic")
    elif avg_consecutive > 120:
        dynamic_score += 2
        types.append("다이나믹")
        characteristics['dynamic'] = f"다이나믹한 이동 (평균 {int(avg_consecutive)}px)"
        confidence_factors.append("dynamic")
    
    # 스태틱 (정적, 밸런스)
    if dynamic_score == 0 and num_holds >= 6:
        static_score += 2
        types.append("스태틱")
        characteristics['static'] = "정밀한 움직임이 필요한 스태틱 문제"
        confidence_factors.append("static")
    
    # 밸런스
    if movement_ratio > 1.2:
        types.append("밸런스")
        characteristics['balance'] = "수평 이동이 많아 밸런스 중요"
        confidence_factors.append("balance")
    
    # 🔄 3. 특수 동작 분석
    special_moves = []
    
    # 코디네이션 (많은 홀드 + 적당한 거리)
    if num_holds >= 7 and 80 < avg_consecutive < 150:
        special_moves.append("코디네이션")
        characteristics['coordination'] = f"{num_holds}개 홀드로 연속 동작 필요"
        confidence_factors.append("coordination")
    
    # 런지 (큰 점프 + 작은 홀드)
    if max_distance > 180 and min_area < 1500:
        special_moves.append("런지")
        characteristics['lunge'] = "작은 홀드로 긴 점프 필요"
        confidence_factors.append("lunge")
    
    # 캠퍼싱 (수직 + 큰 간격)
    if vertical_range > horizontal_range * 1.3 and avg_consecutive > 100:
        special_moves.append("캠퍼싱")
        characteristics['campusing'] = "수직 상승 위주"
        confidence_factors.append("campusing")
    
    # 🏗️ 4. 홀드 타입 분석
    hold_types = []
    
    # 크림프 (작은 홀드 60% 이상)
    small_holds_ratio = len([a for a in areas if a < 1200]) / num_holds
    if small_holds_ratio > 0.7:
        hold_types.append("크림프 중심")
        characteristics['crimp'] = f"크림프 홀드 {int(small_holds_ratio*100)}%"
        confidence_factors.append("crimp")
    elif small_holds_ratio > 0.4:
        hold_types.append("크림프")
        characteristics['crimp'] = f"크림프 홀드 {int(small_holds_ratio*100)}%"
    
    # 쥬그 (큰 홀드 60% 이상)
    large_holds_ratio = len([a for a in areas if a > 3500]) / num_holds
    if large_holds_ratio > 0.6:
        hold_types.append("쥬그")
        characteristics['jug'] = f"쥬그 홀드 {int(large_holds_ratio*100)}%"
        confidence_factors.append("jug")
    
    # 핀치 (길쭉한 홀드)
    circularities = np.array([h.get('circularity', 0.7) for h in filtered_holds])
    low_circularity_ratio = len([c for c in circularities if c < 0.6]) / num_holds
    if low_circularity_ratio > 0.5:
        hold_types.append("핀치")
        characteristics['pinch'] = f"핀치 홀드 {int(low_circularity_ratio*100)}%"
        confidence_factors.append("pinch")
    
    # 슬로퍼 (볼록한 홀드)
    convexities = np.array([h.get('convexity', 0.5) for h in filtered_holds])
    high_convexity_ratio = len([c for c in convexities if c > 0.7]) / num_holds
    if high_convexity_ratio > 0.4:
        hold_types.append("슬로퍼")
        characteristics['sloper'] = f"슬로퍼 홀드 {int(high_convexity_ratio*100)}%"
        confidence_factors.append("sloper")
    
    # 🏔️ 5. 벽 각도별 특성
    wall_characteristics = {}
    if wall_angle:
        if wall_angle == "overhang":
            wall_characteristics['overhang'] = "오버행 - 체력 소모 큼"
            if dynamic_score > 0:
                types.append("파워풀")
                characteristics['powerful'] = "오버행에서의 다이나믹 - 폭발적 힘 필요"
            else:
                types.append("지구력")
                characteristics['endurance'] = "오버행에서의 지속 - 지구력 중요"
            confidence_factors.append("overhang")
        elif wall_angle == "slab":
            wall_characteristics['slab'] = "슬랩 - 균형과 섬세함"
            if "밸런스" not in types:
                types.append("밸런스")
            types.append("테크니컬")
            characteristics['technical'] = "슬랩에서의 섬세한 발 사용"
            confidence_factors.append("slab")
        elif wall_angle == "face":
            wall_characteristics['face'] = "직벽 - 균형잡힌 난이도"
            confidence_factors.append("face")
    
    # 🎭 6. 주요 유형 결정 (우선순위 기반)
    primary_type = "일반"
    
    if wall_angle == "overhang" and dynamic_score > 2:
        primary_type = "오버행 다이나믹"
    elif wall_angle == "slab":
        primary_type = "슬랩 밸런스"
    elif "다이노" in types:
        primary_type = "다이노"
    elif "런지" in special_moves:
        primary_type = "런지"
    elif "트래버스" in types:
        primary_type = "트래버스"
    elif "수직등반" in types and "캠퍼싱" in special_moves:
        primary_type = "캠퍼싱"
    elif "크림프 중심" in hold_types:
        primary_type = "크림프"
    elif "슬로퍼" in hold_types:
        primary_type = "슬로퍼"
    elif "핀치" in hold_types:
        primary_type = "핀치"
    elif "다이나믹" in types:
        primary_type = "다이나믹"
    elif "밸런스" in types:
        primary_type = "밸런스"
    elif "코디네이션" in special_moves:
        primary_type = "코디네이션"
    elif len(types) > 0:
        primary_type = types[0]
    
    # 신뢰도 계산 (요소가 많을수록 높음)
    confidence = 0.4 + min(len(confidence_factors) * 0.08, 0.4)  # 40% ~ 80%
    
    return {
        "primary_type": primary_type,
        "types": list(set(types + special_moves + hold_types)),
        "confidence": min(confidence, 0.85),
        "characteristics": {**characteristics, **wall_characteristics},
        "analysis": {
            "dynamic_score": dynamic_score,
            "static_score": static_score,
            "special_moves": special_moves,
            "hold_types": hold_types,
            "wall_angle": wall_angle,
            "movement_pattern": {
                "horizontal_range": int(horizontal_range),
                "vertical_range": int(vertical_range),
                "movement_ratio": round(movement_ratio, 2)
            }
        }
    }

# ============================================================================
# 🎨 룰 기반 색상 분류 시스템 (CLIP 대체, 빠른 속도)
# ============================================================================

def load_color_ranges(config_path="holdcheck/color_ranges.json"):
    """색상 범위 설정 파일 로드 (사용자 피드백 반영)"""
    global _color_ranges_cache
    
    if _color_ranges_cache is not None:
        return _color_ranges_cache
    
    # 파일이 있으면 로드
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            _color_ranges_cache = json.load(f)
            print(f"✅ 색상 범위 설정 로드: {config_path}")
            return _color_ranges_cache
    
    # 없으면 기본값 생성
    _color_ranges_cache = get_default_color_ranges_data()
    save_color_ranges(_color_ranges_cache, config_path)
    print(f"✅ 기본 색상 범위 생성: {config_path}")
    return _color_ranges_cache


def save_color_ranges(ranges, config_path="holdcheck/color_ranges.json"):
    """색상 범위 설정 저장"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(ranges, f, indent=2, ensure_ascii=False)
    print(f"💾 색상 범위 저장: {config_path}")


def get_default_color_ranges_data():
    """기본 색상 범위 데이터 (JSON 직렬화 가능)"""
    return {
        "version": "1.0",
        "last_updated": "2025-01-01",
        "feedback_count": 0,
        "colors": {
            "black": {
                "name": "검정색",
                "priority": 1,
                "hsv_ranges": [
                    {"h": [0, 180], "s": [0, 255], "v": [0, 80]}  # 매우 어두움
                ],
                "rgb_conditions": [
                    {"type": "max_value", "threshold": 80},  # max(R,G,B) < 80
                    {"type": "achromatic", "brightness_max": 150, "channel_diff_max": 50}  # 무채색
                ]
            },
            "white": {
                "name": "흰색",
                "priority": 2,
                "hsv_ranges": [
                    {"h": [0, 180], "s": [0, 50], "v": [200, 255]}  # 밝고 채도 낮음
                ],
                "rgb_conditions": [
                    {"type": "min_value", "threshold": 180},  # min(R,G,B) > 180
                ]
            },
            "gray": {
                "name": "회색",
                "priority": 3,
                "hsv_ranges": [
                    {"h": [0, 180], "s": [0, 50], "v": [80, 200]}  # 중간 밝기, 낮은 채도
                ],
                "rgb_conditions": [
                    {"type": "achromatic", "brightness_min": 80, "brightness_max": 180, "channel_diff_max": 50}
                ]
            },
            "red": {
                "name": "빨간색",
                "priority": 4,
                "hsv_ranges": [
                    {"h": [0, 10], "s": [100, 255], "v": [100, 255]},  # 빨강 (0도 근처)
                    {"h": [170, 180], "s": [100, 255], "v": [100, 255]}  # 빨강 (180도 근처)
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "r", "min_value": 150, "diff_threshold": 50}
                ]
            },
            "orange": {
                "name": "주황색",
                "priority": 5,
                "hsv_ranges": [
                    {"h": [10, 25], "s": [100, 255], "v": [100, 255]}  # 주황 (15도 근처)
                ],
                "rgb_conditions": [
                    {"type": "two_channel_high", "channels": ["r", "g"], "r_min": 150, "g_min": 80, "b_max": 120, "r_over_g": True}
                ]
            },
            "yellow": {
                "name": "노란색",
                "priority": 6,
                "hsv_ranges": [
                    {"h": [25, 40], "s": [100, 255], "v": [150, 255]}  # 노랑 (30도 근처)
                ],
                "rgb_conditions": [
                    {"type": "two_channel_high", "channels": ["r", "g"], "r_min": 150, "g_min": 150, "b_max": 150, "similar": True}
                ]
            },
            "green": {
                "name": "초록색",
                "priority": 7,
                "hsv_ranges": [
                    {"h": [40, 75], "s": [100, 255], "v": [100, 255]}  # 초록 (60도 근처) - 민트와 겹침 방지
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "g", "min_value": 100, "diff_threshold": 30}
                ]
            },
            "mint": {
                "name": "민트색",
                "priority": 8,
                "hsv_ranges": [
                    {"h": [75, 105], "s": [100, 255], "v": [150, 255]}  # 청록 (90도 근처) - 범위 확장
                ],
                "rgb_conditions": [
                    {"type": "two_channel_high", "channels": ["g", "b"], "g_min": 150, "b_min": 150, "r_max": 150}
                ]
            },
            "blue": {
                "name": "파란색",
                "priority": 9,
                "hsv_ranges": [
                    {"h": [105, 130], "s": [100, 255], "v": [100, 255]}  # 파랑 (120도 근처) - 민트와 겹침 방지
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "b", "min_value": 100, "diff_threshold": 30}
                ]
            },
            "purple": {
                "name": "보라색",
                "priority": 10,
                "hsv_ranges": [
                    {"h": [130, 160], "s": [100, 255], "v": [100, 255]}  # 보라 (145도 근처)
                ],
                "rgb_conditions": [
                    {"type": "two_channel_high", "channels": ["r", "b"], "r_min": 100, "b_min": 100, "g_diff": 20}
                ]
            },
            "pink": {
                "name": "분홍색",
                "priority": 11,
                "hsv_ranges": [
                    {"h": [160, 170], "s": [50, 150], "v": [180, 255]}  # 분홍 (밝은 빨강)
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "r", "min_value": 180, "g_min": 100, "b_min": 100}
                ]
            },
            "brown": {
                "name": "갈색",
                "priority": 12,
                "hsv_ranges": [
                    {"h": [0, 10], "s": [80, 200], "v": [50, 150]}  # 어두운 주황 - 주황과 겹침 방지
                ],
                "rgb_conditions": [
                    {"type": "dominant_channel", "channel": "r", "min_value": 80, "max_value": 150, "dark": True}
                ]
            }
        }
    }


def rule_based_color_clustering(hold_data, vectors, config_path="holdcheck/color_ranges.json", 
                                confidence_threshold=0.7, use_hsv=True):
    """
    ⚡ 룰 기반 색상 클러스터링 (CLIP 대체, 초고속)
    
    RGB/HSV 색상 범위로 직접 분류 - CLIP보다 10-20배 빠름!
    사용자 피드백으로 정확도 지속 개선 가능
    
    Args:
        hold_data: 홀드 데이터 (dominant_rgb 또는 dominant_hsv 필요)
        vectors: 사용 안 함 (호환성 유지)
        config_path: 색상 범위 설정 파일 경로
        confidence_threshold: 신뢰도 임계값 (낮으면 unknown으로 분류)
        use_hsv: HSV 공간 사용 여부 (더 정확함)
    
    Returns:
        hold_data: 그룹 정보가 추가된 홀드 데이터
    """
    if len(hold_data) == 0:
        return hold_data
    
    import time
    start_time = time.time()
    
    print(f"\n⚡ 룰 기반 색상 클러스터링 시작 (CLIP 없음, 초고속)")
    print(f"   홀드 개수: {len(hold_data)}개")
    print(f"   색상 공간: {'HSV' if use_hsv else 'RGB'}")
    
    # 색상 범위 로드
    ranges_data = load_color_ranges(config_path)
    colors_config = ranges_data["colors"]
    
    # 각 홀드를 색상으로 분류
    color_groups = {}
    classification_details = []
    
    for hold_idx, hold in enumerate(hold_data):
        # RGB/HSV 값 가져오기
        if "dominant_hsv" in hold:
            h, s, v = hold["dominant_hsv"]
        elif "dominant_rgb" in hold:
            rgb = hold["dominant_rgb"]
            hsv_arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
            hsv_bgr = cv2.cvtColor(hsv_arr, cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = hsv_bgr
        else:
            h, s, v = 0, 0, 128  # 기본값
            rgb = [128, 128, 128]
        
        if "dominant_rgb" not in hold:
            hsv_arr = np.uint8([[[h, s, v]]])
            rgb_arr = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
            rgb = rgb_arr.tolist()
        else:
            rgb = hold["dominant_rgb"]
        
        # 색상 분류 (우선순위 순서대로)
        if use_hsv:
            color_name, confidence, matched_rule = classify_color_by_hsv(
                h, s, v, rgb, colors_config
            )
        else:
            color_name, confidence, matched_rule = classify_color_by_rgb(
                rgb, colors_config
            )
        
        # 신뢰도 낮으면 unknown
        if confidence < confidence_threshold:
            color_name = "unknown"
        
        # 홀드에 정보 추가 (CLIP 호환)
        hold["clip_color_name"] = color_name
        hold["clip_confidence"] = confidence
        hold["color_method"] = "rule_based"
        hold["matched_rule"] = matched_rule
        
        # 그룹핑
        if color_name not in color_groups:
            color_groups[color_name] = []
        color_groups[color_name].append(hold)
        
        classification_details.append({
            "hold_id": hold.get("id", hold_idx),
            "rgb": rgb,
            "hsv": [h, s, v],
            "color": color_name,
            "confidence": confidence,
            "rule": matched_rule
        })
    
    # 그룹 ID 할당 (색상 이름 기준 정렬)
    color_order = ["black", "white", "gray", "red", "orange", "yellow", 
                   "lime", "green", "mint", "blue", "purple", "pink", "brown", "unknown"]
    
    group_idx = 0
    for color_name in color_order:
        if color_name in color_groups:
            for hold in color_groups[color_name]:
                hold["group"] = f"g{group_idx}"
            group_idx += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ 룰 기반 클러스터링 완료 (⚡ {elapsed:.2f}초)")
    print(f"   생성된 그룹 수: {len(color_groups)}개")
    for color_name in color_order:
        if color_name in color_groups:
            count = len(color_groups[color_name])
            avg_conf = np.mean([h["clip_confidence"] for h in color_groups[color_name]])
            print(f"   {color_name}: {count}개 홀드 (평균 신뢰도: {avg_conf:.2f})")
    
    return hold_data


def classify_color_simple_hsv(h, s, v):
    """🎨 상식적인 HSV 기반 색상 분류 (명도 우선 판단)"""
    
    # 🔥 1단계: 명도 우선 판단 (검정/흰색은 채도 무관)
    if v < 90:
        # 매우 어두움 → 검정 (채도 무관!)
        return "black", 0.95
    elif v > 200 and s < 50:
        # 매우 밝음 + 낮은 채도 → 흰색
        return "white", 0.95
    
    # 🔥 2단계: 채도 기반 무채색 판단 (중간 명도)
    if s < 30:
        # 채도가 매우 낮음 → 회색
        return "gray", 0.90
    
    # 2단계: 유채색 판단 (OpenCV H는 0-180)
    if (h >= 0 and h < 8) or (h >= 170):
        return "red", 0.90
    elif h >= 8 and h < 18:
        return "orange", 0.90
    elif h >= 18 and h < 30:
        return "yellow", 0.90
    elif h >= 30 and h < 45:
        return "lime", 0.90  # 연두
    elif h >= 45 and h < 80:
        return "green", 0.90
    elif h >= 80 and h < 95:
        return "mint", 0.85  # 민트/청록
    elif h >= 95 and h < 130:
        return "blue", 0.90
    elif h >= 130 and h < 150:
        return "purple", 0.90
    elif h >= 150 and h < 170:
        return "pink", 0.90
    else:
        # 갈색 판단 (낮은 채도 + 낮은 명도)
        if s < 60 and v < 120:
            return "brown", 0.80
        return "unknown", 0.50

def classify_color_by_hsv(h, s, v, rgb, colors_config):
    """HSV 범위 기반 색상 분류 (상식적 분류 우선 사용)"""
    
    # 🔥 먼저 상식적인 HSV 분류 시도
    color_name, confidence = classify_color_simple_hsv(h, s, v)
    if confidence > 0.80:  # 신뢰도가 높으면 바로 반환
        return color_name, confidence, f"Simple HSV: H={h}, S={s}, V={v}"
    
    # 기존 config 기반 분류 (백업)
    sorted_colors = sorted(colors_config.items(), key=lambda x: x[1].get("priority", 999))
    
    for color_name, config in sorted_colors:
        # HSV 범위 체크
        if "hsv_ranges" in config:
            for hsv_range in config["hsv_ranges"]:
                h_min, h_max = hsv_range["h"]
                s_min, s_max = hsv_range["s"]
                v_min, v_max = hsv_range["v"]
                
                # Hue는 원형이므로 특별 처리
                h_match = False
                if h_min <= h_max:
                    h_match = h_min <= h <= h_max
                else:  # 예: [170, 10] (빨강)
                    h_match = h >= h_min or h <= h_max
                
                if h_match and s_min <= s <= s_max and v_min <= v <= v_max:
                    confidence = calculate_confidence_hsv(h, s, v, hsv_range)
                    return color_name, confidence, f"HSV: H={h}, S={s}, V={v}"
        
        # RGB 조건 체크 (보조)
        if "rgb_conditions" in config:
            for condition in config["rgb_conditions"]:
                if check_rgb_condition(rgb, condition):
                    confidence = 0.8  # RGB 조건은 약간 낮은 신뢰도
                    return color_name, confidence, f"RGB: {rgb}"
    
    # 매칭 실패 - 가장 가까운 색상 찾기
    return find_nearest_color_hsv(h, s, v, colors_config)


def classify_color_by_rgb(rgb, colors_config):
    """RGB 조건 기반 색상 분류"""
    r, g, b = rgb
    
    sorted_colors = sorted(colors_config.items(), key=lambda x: x[1].get("priority", 999))
    
    for color_name, config in sorted_colors:
        if "rgb_conditions" in config:
            for condition in config["rgb_conditions"]:
                if check_rgb_condition(rgb, condition):
                    confidence = 0.85
                    return color_name, confidence, f"RGB: {rgb}"
    
    # 매칭 실패
    return "unknown", 0.5, "No match"


def check_rgb_condition(rgb, condition):
    """RGB 조건 체크"""
    r, g, b = rgb
    cond_type = condition.get("type")
    
    if cond_type == "max_value":
        return max(r, g, b) < condition["threshold"]
    
    elif cond_type == "min_value":
        return min(r, g, b) > condition["threshold"]
    
    elif cond_type == "achromatic":
        brightness = max(r, g, b)
        channel_diff = max(r, g, b) - min(r, g, b)
        
        checks = []
        if "brightness_min" in condition:
            checks.append(brightness >= condition["brightness_min"])
        if "brightness_max" in condition:
            checks.append(brightness <= condition["brightness_max"])
        if "channel_diff_max" in condition:
            checks.append(channel_diff < condition["channel_diff_max"])
        
        return all(checks) if checks else False
    
    elif cond_type == "dominant_channel":
        channel = condition["channel"]
        min_val = condition.get("min_value", 0)
        diff_thresh = condition.get("diff_threshold", 30)
        
        channel_val = {"r": r, "g": g, "b": b}[channel]
        other_vals = [v for k, v in {"r": r, "g": g, "b": b}.items() if k != channel]
        
        return (channel_val >= min_val and 
                all(channel_val > ov + diff_thresh for ov in other_vals))
    
    elif cond_type == "two_channel_high":
        channels = condition["channels"]
        vals = {"r": r, "g": g, "b": b}
        
        checks = []
        for ch in channels:
            if f"{ch}_min" in condition:
                checks.append(vals[ch] >= condition[f"{ch}_min"])
            if f"{ch}_max" in condition:
                checks.append(vals[ch] <= condition[f"{ch}_max"])
        
        # 추가 조건
        if condition.get("r_over_g"):
            checks.append(r > g)
        if condition.get("similar"):
            checks.append(abs(r - g) < 50)
        if "g_diff" in condition:
            checks.append(r > g + condition["g_diff"] and b > g + condition["g_diff"])
        
        return all(checks)
    
    return False


def calculate_confidence_hsv(h, s, v, hsv_range):
    """HSV 매칭 신뢰도 계산"""
    h_min, h_max = hsv_range["h"]
    s_min, s_max = hsv_range["s"]
    v_min, v_max = hsv_range["v"]
    
    # 중심에 가까울수록 높은 신뢰도
    h_center = (h_min + h_max) / 2
    s_center = (s_min + s_max) / 2
    v_center = (v_min + v_max) / 2
    
    h_dist = min(abs(h - h_center), 180 - abs(h - h_center)) / 90  # 정규화
    s_dist = abs(s - s_center) / 127.5
    v_dist = abs(v - v_center) / 127.5
    
    # 거리 기반 신뢰도
    avg_dist = (h_dist + s_dist + v_dist) / 3
    confidence = 1.0 - avg_dist * 0.3  # 최대 0.3 감소
    
    return max(0.5, min(1.0, confidence))


def find_nearest_color_hsv(h, s, v, colors_config):
    """가장 가까운 색상 찾기 (폴백)"""
    # 무채색 체크
    if s < 50:
        if v < 80:
            return "black", 0.6, "Fallback: dark achromatic"
        elif v > 180:
            return "white", 0.6, "Fallback: bright achromatic"
        else:
            return "gray", 0.6, "Fallback: mid achromatic"
    
    # Hue 기반 분류
    if h < 10 or h > 170:
        return "red", 0.5, "Fallback: hue range"
    elif 10 <= h < 25:
        return "orange", 0.5, "Fallback: hue range"
    elif 25 <= h < 40:
        return "yellow", 0.5, "Fallback: hue range"
    elif 40 <= h < 80:
        return "green", 0.5, "Fallback: hue range"
    elif 80 <= h < 100:
        return "mint", 0.5, "Fallback: hue range"
    elif 100 <= h < 130:
        return "blue", 0.5, "Fallback: hue range"
    else:
        return "purple", 0.5, "Fallback: hue range"


def save_user_feedback(hold_data, feedback_list, config_path="holdcheck/color_ranges.json"):
    """
    사용자 피드백 저장 및 색상 범위 자동 조정
    
    Args:
        hold_data: 홀드 데이터
        feedback_list: [{"hold_id": 0, "correct_color": "yellow", "predicted_color": "orange"}, ...]
        config_path: 설정 파일 경로
    """
    global _color_feedback_data
    
    print(f"\n📝 사용자 피드백 저장 중... ({len(feedback_list)}개)")
    
    # 피드백 데이터 추가
    _color_feedback_data.extend(feedback_list)
    
    # 색상 범위 로드
    ranges_data = load_color_ranges(config_path)
    
    # 피드백 통계
    feedback_stats = {}
    for fb in feedback_list:
        pred = fb["predicted_color"]
        correct = fb["correct_color"]
        
        if pred != correct:
            key = f"{pred} -> {correct}"
            if key not in feedback_stats:
                feedback_stats[key] = []
            
            # 홀드 찾기
            hold = next((h for h in hold_data if h.get("id") == fb["hold_id"]), None)
            if hold:
                feedback_stats[key].append({
                    "rgb": hold.get("dominant_rgb"),
                    "hsv": hold.get("dominant_hsv")
                })
    
    print(f"   오분류 패턴:")
    for pattern, samples in feedback_stats.items():
        print(f"   {pattern}: {len(samples)}건")
    
    # 색상 범위 자동 조정 (학습)
    adjust_color_ranges(ranges_data, feedback_stats)
    
    # 피드백 카운트 증가
    ranges_data["feedback_count"] += len(feedback_list)
    ranges_data["last_updated"] = str(np.datetime64('now'))
    
    # 저장
    save_color_ranges(ranges_data, config_path)
    
    print(f"✅ 피드백 반영 완료! (총 {ranges_data['feedback_count']}건)")
    
    # 캐시 초기화
    global _color_ranges_cache
    _color_ranges_cache = None


def adjust_color_ranges(ranges_data, feedback_stats):
    """피드백 기반 색상 범위 자동 조정"""
    colors_config = ranges_data["colors"]
    
    for pattern, samples in feedback_stats.items():
        if len(samples) < 3:  # 최소 3개 이상
            continue
        
        pred_color, correct_color = pattern.split(" -> ")
        
        if correct_color not in colors_config:
            continue
        
        # 올바른 색상의 HSV 범위 확장
        hsv_samples = [s["hsv"] for s in samples if s["hsv"]]
        
        if hsv_samples:
            avg_h = np.mean([h for h, s, v in hsv_samples])
            avg_s = np.mean([s for h, s, v in hsv_samples])
            avg_v = np.mean([v for h, s, v in hsv_samples])
            
            print(f"   {correct_color} 범위 확장: H={avg_h:.0f}, S={avg_s:.0f}, V={avg_v:.0f}")
            
            # 범위에 새 샘플 추가 (간단한 방식)
            # 실제로는 더 정교한 클러스터링 필요
            current_ranges = colors_config[correct_color].get("hsv_ranges", [])
            
            # 기존 범위와 겹치지 않으면 새 범위 추가
            new_range = {
                "h": [max(0, int(avg_h - 10)), min(180, int(avg_h + 10))],
                "s": [max(0, int(avg_s - 30)), min(255, int(avg_s + 30))],
                "v": [max(0, int(avg_v - 30)), min(255, int(avg_v + 30))]
            }
            
            # 중복 체크 (간단히)
            is_duplicate = any(
                abs(r["h"][0] - new_range["h"][0]) < 20 for r in current_ranges
            )
            
            if not is_duplicate:
                current_ranges.append(new_range)
                print(f"      새 범위 추가됨!")


def draw_holds_on_image_with_highlights(image, hold_data, bboxes, problems):
    """
    이미지에 홀드를 그리고, 피드백이 있는 홀드는 강조하여 표시합니다.
    
    Args:
        image (np.array): 원본 이미지 (BGR 형식).
        hold_data (list): 각 홀드의 정보 (dict).
        bboxes (list): 각 홀드의 바운딩 박스 (x1, y1, x2, y2).
        problems (dict): hold_id를 키로 하고, 문제가 있는 홀드 정보를 값으로 하는 딕셔너리.
                         예: {0: {"predicted_color": "yellow", "correct_color": "orange"}}
    Returns:
        np.array: 홀드와 강조 표시가 그려진 이미지.
    """
    display_image = image.copy()
    
    # 색상 매핑 (BGR 형식)
    color_map = {
        "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
        "red": (0, 0, 255), "orange": (0, 165, 255), "yellow": (0, 255, 255),
        "green": (0, 255, 0), "mint": (204, 255, 0), "blue": (255, 0, 0),
        "purple": (255, 0, 128), "pink": (204, 102, 255), "brown": (42, 42, 165),
        "unknown": (192, 192, 192) # 회색
    }
    
    for i, hold in enumerate(hold_data):
        if i >= len(bboxes):
            continue # 바운딩 박스가 없는 홀드는 건너뜀
            
        x1, y1, x2, y2 = map(int, bboxes[i])
        
        # 홀드 색상 가져오기 (없으면 unknown)
        color_name = hold.get("clip_color_name", "unknown")
        bbox_color = color_map.get(color_name, color_map["unknown"])
        
        # 문제가 있는 홀드인지 확인
        is_problematic = str(i) in problems # problems keys are strings
        
        if is_problematic:
            # 문제가 있는 홀드는 빨간색 두꺼운 테두리로 강조
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 4) # Red, thick
            # 텍스트도 빨간색으로
            text_color = (0, 0, 255) 
        else:
            # 일반 홀드는 해당 색상 테두리
            cv2.rectangle(display_image, (x1, y1), (x2, y2), bbox_color, 2)
            text_color = bbox_color
            
        # 홀드 ID와 색상 이름 표시
        text = f"ID:{i} {color_name}"
        cv2.putText(display_image, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            
    return display_image


def export_feedback_dataset(output_path="holdcheck/color_feedback_dataset.json"):
    """피드백 데이터를 학습 데이터셋으로 내보내기 (AI 모델 학습용)"""
    global _color_feedback_data
    
    if not _color_feedback_data:
        print("⚠️ 피드백 데이터 없음")
        return
    
    dataset = {
        "version": "1.0",
        "total_samples": len(_color_feedback_data),
        "samples": _color_feedback_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 피드백 데이터셋 내보내기: {output_path} ({len(_color_feedback_data)}개 샘플)")
