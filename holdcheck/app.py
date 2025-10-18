import streamlit as st
import cv2
import json
import numpy as np
from PIL import Image
from preprocess import preprocess
from clustering import build_feature_vectors, recommend_holds, rgb_cube_dbscan_clustering, hsv_cube_dbscan_clustering, rgb_weighted_dbscan_clustering, custom_color_cube_dbscan_clustering, perceptual_color_dbscan_clustering, cylindrical_hsv_dbscan_clustering, lch_cosine_dbscan_clustering, ciede2000_mds_dbscan_clustering, lighting_invariant_dbscan_clustering, clip_ai_color_clustering, create_rgb_color_cube_with_groups, create_pure_rgb_color_cube, create_lab_color_space_visualization, create_cylindrical_hsv_visualization, create_custom_color_space_visualization, create_mds_visualization, create_compressed_2d_visualization, create_clip_3d_visualization, analyze_problem
from streamlit_drawable_canvas import st_canvas

def detect_clicked_hold(click_x, click_y, hold_data, masks):
    """클릭된 좌표에서 홀드 ID를 찾는 함수"""
    for i, hold in enumerate(hold_data):
        if i < len(masks):
            mask = masks[i]
            if (click_y < mask.shape[0] and click_x < mask.shape[1] and 
                mask[click_y, click_x] > 0):
                return hold["id"]
    return None

st.title("🧗‍♂️ ClimbMate: 단순한 홀드 클러스터링")

# 이미지 업로드
uploaded_file = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 로드
    image = Image.open(uploaded_file)
    original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 사이드바 설정 - 완전 단순화
    st.sidebar.header("⚙️ 클러스터링 설정")
    
    # eps만 남기고 모든 복잡한 설정 제거
    # 전처리 방법 선택
    preprocessing_method = st.sidebar.selectbox(
        "🌞 전처리 방법",
        ["하이브리드", "명도 정규화", "기존 방식"],
        help="하이브리드: 무채색은 기존방식, 유채색은 명도정규화, 명도정규화: 어둡고 밝은 같은 색을 동일하게 인식, 기존 방식: 원본 색상 유지"
    )
    
    # 🎨 전처리 강화 설정
    st.sidebar.subheader("🎨 전처리 강화 설정")
    
    # 명도 필터링
    brightness_filter = st.sidebar.checkbox(
        "명도 필터링 활성화", 
        value=True, 
        help="너무 어둡거나 밝은 색상을 필터링합니다"
    )
    
    if brightness_filter:
        min_brightness = st.sidebar.slider(
            "최소 명도", 
            0, 100, 20, 5,
            help="너무 어두운 색상 제외 (0-100)"
        )
        max_brightness = st.sidebar.slider(
            "최대 명도", 
            0, 100, 95, 5,
            help="너무 밝은 색상 제외 (0-100)"
        )
    else:
        min_brightness = 0
        max_brightness = 100
    
    # 채도 필터링
    saturation_filter = st.sidebar.checkbox(
        "채도 필터링 활성화", 
        value=True, 
        help="회색/베이지색 등 무채색을 필터링합니다"
    )
    
    if saturation_filter:
        min_saturation = st.sidebar.slider(
            "최소 채도", 
            0, 100, 15, 5,
            help="무채색 제외 (0-100, 높을수록 더 선명한 색상만)"
        )
    else:
        min_saturation = 0
    
    # 🚀 마스크 정제 강도 (기본값 낮춤 - 속도 우선)
    mask_refinement = st.sidebar.slider(
        "마스크 정제 강도",
        1, 10, 2, 1,
        help="마스크 경계 정제 반복 횟수 (높을수록 더 정확하지만 느림)"
    )
    
    # 클러스터링 방법 선택
    clustering_method = st.sidebar.selectbox(
        "🎨 클러스터링 방법",
        ["🤖 CLIP AI 색상 인식", "🤖 CLIP AI + DBSCAN", "🌟 조명 불변 (1:1:1 대각선 압축)", "CIEDE2000+MDS", "LCh+Cosine 거리", "지각적 색상 공간 (Lab+CIEDE2000)", "원통 좌표계 HSV", "커스텀 색상 큐브", "RGB 축별 가중치", "HSV 색상 공간", "RGB 색상 공간"],
        help="🤖 CLIP AI: AI가 자동으로 색상을 인식하여 직접 그룹핑 (라벨링 불필요), 🤖 CLIP AI + DBSCAN: CLIP 특징 벡터로 DBSCAN 클러스터링, 🌟 조명 불변: 조명 차이를 무시하고 순수 색상만으로 군집화 (검정/흰색 분리 → RGB 대각선 압축), CIEDE2000+MDS: 중간톤 압축 해결 + 균등한 2D 분포, LCh+Cosine: Hue wrap 해결, 지각적 색상 공간: Lab+CIEDE2000 거리로 색조 중심 군집화, 원통 좌표계: Hue 중심 원통 좌표계, 커스텀 색상 큐브: 주요 색상 간 거리 확장, RGB 축별 가중치: Blue축 엄격, Green/Red축 관대, HSV: 색상 라인별 분리, RGB: 기존 방식"
    )
    
    # RGB 축별 가중치 설정 (RGB 축별 가중치 선택 시에만)
    if clustering_method == "RGB 축별 가중치":
        st.sidebar.subheader("🎯 축별 가중치 설정")
        weight_r = st.sidebar.slider("Red 축 가중치", 0.1, 2.0, 1.0, 0.1, help="Red 라인: 관대한 가중치")
        weight_g = st.sidebar.slider("Green 축 가중치", 0.1, 2.0, 1.0, 0.1, help="Green 라인: 관대한 가중치") 
        weight_b = st.sidebar.slider("Blue 축 가중치", 0.1, 2.0, 1.0, 0.1, help="Blue 라인: 엄격한 가중치")
    else:
        weight_r = weight_g = weight_b = 1.0  # 기본값
    
    # 🌟 조명 불변 클러스터링은 단순화됨 (모든 색상 통합 처리)
    eps_black_gray = eps_white = eps_color = 1.0  # 기본값 (사용하지 않음)
    
    eps = st.sidebar.slider(
        "eps (거리 임계값)",
        min_value=5.0,
        max_value=80.0,
        value=40.0,  # 정규화 적용 후 적절한 기본값
        step=1.0,
        help="색상 공간에서 유클리드 거리 임계값. HSV: 30-50 추천, RGB: 20-40 추천"
    )
    
    # 그룹핑 재실행 버튼
    rerun_clustering = st.sidebar.button("🔄 그룹핑 재실행")
    
    # 전처리 (한 번만 실행)
    if 'preprocessed_data' not in st.session_state or rerun_clustering:
        with st.spinner("홀드를 감지 중..."):
            # CLIP AI 모드일 때는 use_clip_ai=True로 전처리
            use_clip_ai = clustering_method.startswith("🤖 CLIP AI")
            
            hold_data_raw, masks = preprocess(
                original_image, 
                brightness_normalization=preprocessing_method,
                brightness_filter=brightness_filter,
                min_brightness=min_brightness,
                max_brightness=max_brightness,
                saturation_filter=saturation_filter,
                min_saturation=min_saturation,
                mask_refinement=mask_refinement,
                use_clip_ai=use_clip_ai
            )
            if hold_data_raw:
                st.session_state.preprocessed_data = {
                    'hold_data_raw': hold_data_raw,
                    'masks': masks,
                    'original_image': original_image
                }
                st.success(f"✅ {len(hold_data_raw)}개의 홀드가 감지되었습니다.")
            else:
                st.error("❌ 홀드를 감지할 수 없습니다.")
                st.stop()
    
    # 클러스터링 (파라미터 변경 시 재실행)
    clustering_key = f"{preprocessing_method}_{clustering_method}_{eps}_{weight_r}_{weight_g}_{weight_b}_{eps_black_gray}_{eps_white}_{eps_color}_{brightness_filter}_{min_brightness}_{max_brightness}_{saturation_filter}_{min_saturation}_{mask_refinement}"
    
    if 'clustering_cache' not in st.session_state or st.session_state.clustering_cache.get('key') != clustering_key or rerun_clustering:
        with st.spinner("그룹핑 중..."):
            processed = st.session_state.preprocessed_data
            hold_data_raw = processed['hold_data_raw']
            masks = processed['masks']
            original_image = processed['original_image']
            
            # 특징 벡터 생성
            vectors, ids = build_feature_vectors(hold_data_raw, scaler_option="none",
                                                use_illumination_invariant=True)
            
            # 🚀 색상 공간 선택에 따른 클러스터링
            if clustering_method == "🤖 CLIP AI 색상 인식":
                hold_data = clip_ai_color_clustering(hold_data_raw, vectors, original_image, masks, eps=eps, use_dbscan=False)
            elif clustering_method == "🤖 CLIP AI + DBSCAN":
                hold_data = clip_ai_color_clustering(hold_data_raw, vectors, original_image, masks, eps=eps, use_dbscan=True)
            elif clustering_method == "🌟 조명 불변 (1:1:1 대각선 압축)":
                hold_data = lighting_invariant_dbscan_clustering(hold_data_raw, vectors, eps=eps, 
                                                                eps_black_gray=eps_black_gray, 
                                                                eps_white=eps_white, 
                                                                eps_color=eps_color)
            elif clustering_method == "지각적 색상 공간 (Lab+CIEDE2000)":
                hold_data = perceptual_color_dbscan_clustering(hold_data_raw, vectors, eps=eps)
            elif clustering_method == "원통 좌표계 HSV":
                hold_data = cylindrical_hsv_dbscan_clustering(hold_data_raw, vectors, eps=eps)
            elif clustering_method == "커스텀 색상 큐브":
                hold_data = custom_color_cube_dbscan_clustering(hold_data_raw, vectors, eps=eps)
            elif clustering_method == "LCh+Cosine 거리":
                hold_data = lch_cosine_dbscan_clustering(hold_data_raw, vectors, eps=eps)
            elif clustering_method == "CIEDE2000+MDS":
                hold_data = ciede2000_mds_dbscan_clustering(hold_data_raw, vectors, eps=eps)
            elif clustering_method == "HSV 색상 공간":
                hold_data = hsv_cube_dbscan_clustering(hold_data_raw, vectors, eps=eps)
            elif clustering_method == "RGB 축별 가중치":
                hold_data = rgb_weighted_dbscan_clustering(hold_data_raw, vectors, eps=eps, weights=[weight_r, weight_g, weight_b])
            else:
                hold_data = rgb_cube_dbscan_clustering(hold_data_raw, vectors, eps=eps)
            
            # 클러스터링 결과를 세션에 저장
            st.session_state.clustering_cache = {
                'key': clustering_key,
                'hold_data': hold_data,
                'vectors': vectors,
                'ids': ids
            }
            
            st.success(f"🎯 그룹핑 완료! {len(set(h['group'] for h in hold_data if h['group'] is not None))}개 그룹 생성")
    else:
        # 캐시된 결과 사용
        hold_data = st.session_state.clustering_cache['hold_data']
        vectors = st.session_state.clustering_cache['vectors']
        masks = st.session_state.preprocessed_data['masks']
        original_image = st.session_state.preprocessed_data['original_image']
        
        group_count = len(set(h['group'] for h in hold_data if h['group'] is not None))
        st.success(f"✅ 캐시된 그룹핑 결과 사용 ({group_count}개 그룹)")
    
    # 간단한 정보 표시
    col1, col2 = st.columns(2)
    with col1:
        st.metric("총 홀드 수", len(hold_data))
    with col2:
        st.metric("생성된 그룹 수", len(set(h['group'] for h in hold_data if h['group'] is not None)))
    
    # 🎨 시각화 (CLIP AI 전용 또는 기존 방식)
    if clustering_method.startswith("🤖 CLIP AI"):
        st.subheader("🤖 CLIP AI 특징 벡터 3D 공간")
        st.write("💡 **CLIP AI가 추출한 512차원 특징 벡터를 PCA로 3D로 축소하여 시각화**")
        
        clip_3d_fig = create_clip_3d_visualization(hold_data, st.session_state.get('selected_hold_id'), eps)
        if clip_3d_fig:
            st.plotly_chart(clip_3d_fig, use_container_width=True)
    else:
        st.subheader("🎨 순수 3D RGB 색상 큐브 (HSV→RGB 변환)")
        st.write("💡 **각 점은 홀드의 HSV에서 RGB로 변환된 색상입니다.**")
        
        pure_rgb_fig = create_pure_rgb_color_cube(hold_data)
        if pure_rgb_fig:
            st.plotly_chart(pure_rgb_fig, use_container_width=True)
    
    # 홀드 선택 (이미지에서 클릭) - 확대/축소 가능
    st.subheader("🎯 홀드 선택 (이미지에서 클릭하세요)")
    st.write("💡 **팁**: 마우스 휠로 확대/축소, 드래그로 이동 가능합니다!")
    
    # 이미지 크기 조정 (캔버스에 맞게)
    display_height = 600
    h, w = original_image.shape[:2]
    display_width = int(w * display_height / h)
    
    # 라벨 마스킹 (테두리 표시)
    overlay = original_image.copy()
    for i, hold in enumerate(hold_data):
        if i < len(masks):
            mask = masks[i].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cy, cx = hold["center"]
        cv2.putText(overlay, f"{hold['id']}", (cx, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 캔버스 생성
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="red",
        background_image=Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="point",
        key="canvas"
    )
    
    # 클릭 이벤트 처리
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            # 가장 최근 클릭 좌표
            last_obj = objects[-1]
            click_x = int(last_obj["left"] * (w / display_width))  # 스케일 조정
            click_y = int(last_obj["top"] * (h / display_height))  # 스케일 조정
            
            clicked_hold_id = detect_clicked_hold(click_x, click_y, hold_data, masks)
            if clicked_hold_id is not None:
                st.session_state.selected_hold_id = clicked_hold_id
                st.success(f"✅ 홀드 {clicked_hold_id} 선택됨!")
    
    # 선택된 홀드가 있으면 추천 표시
    same_group_holds = []  # 전역 스코프에서 초기화
    
    if 'selected_hold_id' in st.session_state:
        selected_hold_id = st.session_state.selected_hold_id
        
        # 같은 그룹 홀드 찾기
        selected_hold = next((h for h in hold_data if h["id"] == selected_hold_id), None)
        
        if selected_hold:
            if selected_hold["group"] is not None:
                same_group_holds = [h for h in hold_data if h["group"] == selected_hold["group"]]
                
                st.subheader(f"🎯 선택된 홀드: {selected_hold_id} (그룹 {selected_hold['group']})")
                st.write(f"**같은 그룹 홀드 수**: {len(same_group_holds)}개")
                
                # 🔍 디버깅: RGB 큐브 그룹핑과 동일한지 확인
                st.info(f"📊 **그룹 {selected_hold['group']} 홀드 ID 목록**: {[h['id'] for h in same_group_holds]}")
                
                # 선택한 홀드의 RGB 값 표시
                h, s, v = selected_hold["dominant_hsv"]
                from clustering import hsv_to_rgb
                rgb = hsv_to_rgb([h, s, v])
                st.write(f"**선택한 홀드 RGB**: ({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})")
    
    # 🤖 AI 분석 결과 바로 표시 (선택한 그룹)
    if len(same_group_holds) >= 3:
        st.write("---")
        st.write("### 🤖 선택한 문제 AI 분석")
        
        # 🏔️ 벽 각도 선택 (사용자 입력)
        col_wall1, col_wall2 = st.columns([1, 2])
        with col_wall1:
            wall_angle = st.selectbox(
                "🏔️ 벽 각도 선택:",
                options=[None, "overhang", "slab", "face"],
                format_func=lambda x: {
                    None: "선택 안함",
                    "overhang": "오버행/루프/케이브 (90°+)",
                    "slab": "슬랩 (90°-)",
                    "face": "직벽 (90°)"
                }[x],
                help="벽 각도 정보를 제공하면 더 정확한 유형 분석이 가능합니다."
            )
        with col_wall2:
            if wall_angle:
                st.info(f"🏔️ **{wall_angle.upper()}** 선택됨 - 더 정확한 분석을 위해 벽 각도 정보를 활용합니다.")
        
        analysis = analyze_problem(hold_data, selected_hold['group'], wall_angle)
        if analysis:
            # 🎯 메인 분석 결과
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 홀드 개수", f"{analysis['statistics']['num_holds']}개")
                st.metric("📏 평균 크기", analysis['statistics']['avg_hold_size'])
                st.metric("📐 최대 간격", analysis['statistics']['max_distance'])
            
            with col2:
                diff = analysis['difficulty']
                confidence_stars = "★" * int(diff['confidence'] * 5) + "☆" * (5 - int(diff['confidence'] * 5))
                st.metric("🎯 추정 난이도", f"{diff['grade']}")
                st.caption(f"**{diff['level']}** • 신뢰도: {confidence_stars}")
                st.metric("📈 난이도 점수", f"{diff['score']}/15")
            
            with col3:
                climb_type = analysis['climb_type']
                st.metric("🏋️ 추정 유형", climb_type['primary_type'])
                st.caption(f"신뢰도: {climb_type['confidence']:.1%}")
                if len(climb_type['types']) > 1:
                    st.write("**복합 유형:**")
                    for t in climb_type['types'][:3]:  # 최대 3개까지만
                        st.write(f"• {t}")
                    if len(climb_type['types']) > 3:
                        st.write(f"• 외 {len(climb_type['types'])-3}개...")
            
            # 📋 상세 분석 (접기/펼치기)
            with st.expander("📋 상세 분석 보기", expanded=False):
                # 난이도 분석
                st.write("**🎯 난이도 분석**")
                factors = diff['factors']
                details = diff['details']
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.write(f"• **홀드 크기**: {details['hold_size']} → {factors['hold_size']}")
                    st.write(f"• **홀드 개수**: {details['num_holds']} → {factors['num_holds']}")
                with col_f2:
                    st.write(f"• **홀드 간격**: {details['distance']} → {factors['distance']}")
                    st.write(f"• **높이 변화**: {details['height_change']} → {factors['height']}")
                
                st.write("")
                
                # 유형 분석
                st.write("**🧗‍♀️ 유형 분석**")
                if climb_type['characteristics']:
                    for key, value in climb_type['characteristics'].items():
                        st.write(f"• **{key}**: {value}")
                
                # 분석 근거
                if 'analysis' in climb_type:
                    analysis_data = climb_type['analysis']
                    st.write("")
                    st.write("**📊 분석 근거**")
                    st.write(f"• 다이나믹 점수: {analysis_data['dynamic_score']}")
                    st.write(f"• 스태틱 점수: {analysis_data['static_score']}")
                    if analysis_data['special_moves']:
                        st.write(f"• 특수 동작: {', '.join(analysis_data['special_moves'])}")
                    if analysis_data['hold_types']:
                        st.write(f"• 홀드 유형: {', '.join(analysis_data['hold_types'])}")
                    if analysis_data['wall_angle']:
                        st.write(f"• 벽 각도: {analysis_data['wall_angle']}")
            
            # 💡 추천 정보
            if climb_type['primary_type'] != "일반":
                st.info(f"💡 **{climb_type['primary_type']}** 유형의 문제입니다. 이 유형에 맞는 클라이밍 기술을 연습해보세요!")
            
            st.write("---")
                
                # 결과 이미지 생성
            highlight_rec = original_image.copy()
            for i, hold in enumerate(hold_data):
                if i < len(masks):
                    mask = masks[i].astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if hold["id"] == selected_hold_id:
                        # 선택한 홀드: 빨간색 굵게
                        cv2.drawContours(highlight_rec, contours, -1, (0, 0, 255), 4)
                    elif hold in same_group_holds:
                        # 같은 그룹: 노란색
                        cv2.drawContours(highlight_rec, contours, -1, (0, 255, 255), 3)
                    else:
                        # 다른 그룹: 회색 얇게
                        cv2.drawContours(highlight_rec, contours, -1, (100, 100, 100), 1)
            
            st.image(cv2.cvtColor(highlight_rec, cv2.COLOR_BGR2RGB),
                    caption=f"그룹 {selected_hold['group']} 결과 (🔴 빨간색: 선택, 🟡 노란색: 같은 그룹, ⚪ 회색: 다른 그룹)",
                    use_column_width=True)
        else:
            st.warning(f"⚠️ 홀드 {selected_hold_id}는 그룹에 속하지 않습니다 (노이즈)")
    
    # 🎯 3D RGB 색상 큐브 시각화 (그룹별 색상 표시) - 항상 표시
    st.subheader("🎯 RGB 3D 색상 큐브에서의 그룹핑 결과")
    st.write("💡 **그룹핑 결과 확인!** 각 포인트의 색상이 할당된 그룹을 나타냅니다.")
    
    # 그룹별 홀드 목록 표시
    groups_dict = {}
    for hold in hold_data:
        if hold["group"] is not None:
            if hold["group"] not in groups_dict:
                groups_dict[hold["group"]] = []
            groups_dict[hold["group"]].append(hold["id"])
    
    st.write("**📊 그룹별 홀드 분포**:")
    for group_id in sorted(groups_dict.keys()):
        st.write(f"- **그룹 {group_id}**: {groups_dict[group_id]} ({len(groups_dict[group_id])}개)")
    
    st.write("---")
    
    # eps 구 검증을 위한 홀드 선택
    st.write("🔍 **eps 구 검증**: 홀드를 선택하면 해당 점 기준으로 eps 거리의 구가 표시됩니다!")
    hold_options = [f"홀드 {hold['id']} (그룹 {hold['group']})" for hold in hold_data]
    selected_hold_for_validation = st.selectbox(
        "검증할 홀드 선택:",
        options=[""] + hold_options,
        key="validation_hold_selector"
    )
    
    # 선택된 홀드 ID 추출
    validation_hold_id = None
    if selected_hold_for_validation and selected_hold_for_validation != "":
        validation_hold_id = int(selected_hold_for_validation.split()[1])  # "홀드 1" -> 1
    
    # 선택된 홀드 ID와 eps 값을 전달 (이미지 클릭 홀드 또는 검증 홀드)
    selected_hold_id_from_image = st.session_state.get('selected_hold_id', None)
    display_hold_id = validation_hold_id if validation_hold_id is not None else selected_hold_id_from_image
    
    # 🎨 클러스터링 방법에 따라 적절한 시각화 선택
    if clustering_method == "🌟 조명 불변 (1:1:1 대각선 압축)":
        visualization_fig = create_rgb_color_cube_with_groups(hold_data, display_hold_id, eps)
        st.write("### 🌟 조명 불변 RGB 큐브 (대각선 압축 적용)")
        st.write("💡 **RGB 1:1:1 대각선 성분(조명)을 압축하여 순수 색상만 표현**")
        
        # 2D 압축 분포도 추가
        compressed_2d_fig = create_compressed_2d_visualization(hold_data, display_hold_id, eps)
        if compressed_2d_fig:
            st.plotly_chart(compressed_2d_fig, use_container_width=True)
            st.write("### 📊 압축된 2D 분포도")
            st.write("💡 **검정/회색은 1차원(대각선), 흰색은 1차원(대각선), 유채색은 2차원(압축된 RGB)**")
    elif clustering_method == "CIEDE2000+MDS":
        visualization_fig = create_mds_visualization(hold_data, display_hold_id, eps)
        st.write("### 🎨 MDS 2D 시각화 (CIEDE2000 거리 기반, 균등한 분포)")
    elif clustering_method == "LCh+Cosine 거리":
        visualization_fig = create_lab_color_space_visualization(hold_data, display_hold_id, eps)
        st.write("### 🎨 LCh 색상 공간 시각화 (Hue wrap 해결)")
    elif clustering_method == "지각적 색상 공간 (Lab+CIEDE2000)":
        visualization_fig = create_lab_color_space_visualization(hold_data, display_hold_id, eps)
        st.write("### 🎨 Lab 색상 공간 시각화")
    elif clustering_method == "원통 좌표계 HSV":
        visualization_fig = create_cylindrical_hsv_visualization(hold_data, display_hold_id, eps)
        st.write("### 🎨 원통 좌표계 HSV 시각화")
    elif clustering_method == "커스텀 색상 큐브":
        visualization_fig = create_custom_color_space_visualization(hold_data, display_hold_id, eps)
        st.write("### 🎨 커스텀 색상 공간 시각화")
    else:
        visualization_fig = create_rgb_color_cube_with_groups(hold_data, display_hold_id, eps)
        st.write("### 🎨 RGB 3D 색상 큐브 시각화")
    
    if visualization_fig:
        st.plotly_chart(visualization_fig, use_container_width=True)
        
        # 📊 그룹핑 결과 상세 정보 (3D 큐브 아래)
        st.write("---")
        st.write("### 📋 그룹별 상세 정보 및 AI 분석")
        
        from clustering import hsv_to_rgb
        
        for group_id in sorted(groups_dict.keys()):
            hold_ids = groups_dict[group_id]
            group_holds = [h for h in hold_data if h["id"] in hold_ids]
            
            # 그룹 대표 색상 (평균)
            avg_rgb = [0, 0, 0]
            for hold in group_holds:
                h, s, v = hold["dominant_hsv"]
                rgb = hsv_to_rgb([h, s, v])
                avg_rgb[0] += rgb[0]
                avg_rgb[1] += rgb[1]
                avg_rgb[2] += rgb[2]
            
            avg_rgb = [int(avg_rgb[0] / len(group_holds)), 
                      int(avg_rgb[1] / len(group_holds)), 
                      int(avg_rgb[2] / len(group_holds))]
            
            # 색상 칩 HTML
            color_chip = f'<div style="display:inline-block; width:20px; height:20px; background-color:rgb({avg_rgb[0]},{avg_rgb[1]},{avg_rgb[2]}); border:1px solid black; margin-right:10px; vertical-align:middle;"></div>'
            
            st.markdown(f"{color_chip} **그룹 {group_id}** ({len(group_holds)}개 홀드)", unsafe_allow_html=True)
            st.write(f"  - **홀드 ID**: {hold_ids}")
            st.write(f"  - **평균 RGB**: ({avg_rgb[0]}, {avg_rgb[1]}, {avg_rgb[2]})")
            
            # 각 홀드의 RGB 좌표
            rgb_coords = []
            for hold in group_holds:
                h, s, v = hold["dominant_hsv"]
                rgb = hsv_to_rgb([h, s, v])
                rgb_coords.append(f"홀드{hold['id']}:({rgb[0]},{rgb[1]},{rgb[2]})")
            st.write(f"  - **RGB 좌표**: {', '.join(rgb_coords[:5])}" + (f" ... 외 {len(rgb_coords)-5}개" if len(rgb_coords) > 5 else ""))
            
            # 🎯 AI 분석 결과 추가
            if len(group_holds) >= 3:  # 최소 3개 이상일 때만 분석
                analysis = analyze_problem(hold_data, group_id)
                if analysis:
                    with st.expander(f"🤖 AI 문제 분석 (그룹 {group_id})", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**🎯 추정 난이도**")
                            diff = analysis['difficulty']
                            confidence_stars = "★" * int(diff['confidence'] * 5) + "☆" * (5 - int(diff['confidence'] * 5))
                            st.write(f"- **등급**: {diff['grade']} ({diff['grade_level']})")
                            st.write(f"- **신뢰도**: {confidence_stars} ({diff['confidence']*100:.0f}%)")
                            st.write(f"- **홀드 개수**: {diff['factors']['num_holds']}개 - {diff['factors']['num_holds_impact']}")
                            st.write(f"- **평균 홀드 크기**: {diff['factors']['avg_hold_size']} - {diff['factors']['hold_size_impact']}")
                            if 'avg_distance' in diff['factors']:
                                st.write(f"- **평균 홀드 간격**: {diff['factors']['avg_distance']} - {diff['factors']['distance_impact']}")
                        
                        with col2:
                            st.write("**🏋️ 추정 유형**")
                            climb_type = analysis['climb_type']
                            st.write(f"- **주요 유형**: {climb_type['primary_type']}")
                            st.write(f"- **모든 유형**: {' + '.join(climb_type['types'])}")
                            st.write(f"- **신뢰도**: {climb_type['confidence']*100:.0f}%")
                            
                            # 특징 설명
                            if climb_type['characteristics']:
                                st.write("**💡 특징**")
                                for key, value in climb_type['characteristics'].items():
                                    if 'note' in key:
                                        st.write(f"  • {value}")
            
            st.write("")
        
        st.write("---")
        
        # 검증 정보 표시
        if display_hold_id is not None and 'eps_validation_info' in st.session_state:
            info = st.session_state['eps_validation_info']
            selected_hold = next((h for h in hold_data if h["id"] == display_hold_id), None)
            if selected_hold:
                st.write(f"🎯 **검증 정보**: 홀드 {info['selected_hold_id']} (그룹 {info['selected_group']}) 기준")
                st.write(f"📏 **RGB 좌표**: ({info['selected_rgb'][0]:.0f}, {info['selected_rgb'][1]:.0f}, {info['selected_rgb'][2]:.0f})")
                st.write(f"📐 **eps 값**: {info['eps']}")
                
                # eps 구 안의 홀드들
                inside_holds = info['inside_holds']
                if inside_holds:
                    st.write(f"✅ **구 안의 홀드들 ({len(inside_holds)}개)**: 거리 ≤ {info['eps']}")
                    
                    # 같은 그룹인지 확인
                    same_group_count = sum(1 for hold_id, group_id, dist, rgb in inside_holds if group_id == info['selected_group'])
                    diff_group_count = len(inside_holds) - same_group_count
                    
                    if diff_group_count == 0:
                        st.success(f"🎉 **모든 구 안의 홀드가 같은 그룹입니다!** (그룹 {info['selected_group']})")
                    else:
                        st.error(f"❌ **문제 발견!** 구 안에 다른 그룹 홀드가 {diff_group_count}개 있습니다!")
                    
                    # 상세 목록
                    for hold_id, group_id, dist, rgb in sorted(inside_holds, key=lambda x: x[2]):
                        status = "✅" if group_id == info['selected_group'] else "❌"
                        st.write(f"  {status} 홀드 {hold_id} (그룹 {group_id}): 거리 {dist:.2f} | RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})")
                else:
                    st.warning(f"⚠️ eps={info['eps']} 구 안에 다른 홀드가 없습니다!")
                
                st.write("💡 **참고**: 3D 그래프를 확대/축소해도 실제 RGB 거리는 변하지 않습니다. 축은 항상 0-255로 고정되어 있습니다.")

else:
    st.info("👆 클라이밍 벽 이미지를 업로드해주세요!")
