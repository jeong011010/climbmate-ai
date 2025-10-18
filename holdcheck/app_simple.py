import streamlit as st
import cv2
import json
import numpy as np
from PIL import Image
from preprocess import preprocess
from clustering import build_feature_vectors, recommend_holds, simple_dbscan_clustering, create_rgb_color_cube_with_groups
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
    eps = st.sidebar.slider(
        "eps (거리 임계값)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help="RGB 공간에서 유클리드 거리 임계값. 작을수록 더 엄격한 그룹핑"
    )
    
    # 그룹핑 재실행 버튼
    rerun_clustering = st.sidebar.button("🔄 그룹핑 재실행")
    
    # 전처리 (한 번만 실행)
    if 'preprocessed_data' not in st.session_state or rerun_clustering:
        with st.spinner("홀드를 감지 중..."):
            hold_data_raw, masks = preprocess(original_image)
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
    clustering_key = f"{eps}"
    
    if 'clustering_cache' not in st.session_state or st.session_state.clustering_cache.get('key') != clustering_key or rerun_clustering:
        with st.spinner("그룹핑 중..."):
            processed = st.session_state.preprocessed_data
            hold_data_raw = processed['hold_data_raw']
            masks = processed['masks']
            original_image = processed['original_image']
            
            # 특징 벡터 생성
            vectors, ids = build_feature_vectors(hold_data_raw, scaler_option="none",
                                                use_illumination_invariant=True)
            
            # 🚀 단순한 DBSCAN 클러스터링
            hold_data = simple_dbscan_clustering(hold_data_raw, vectors, eps=eps)
            
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
        if objects:
            click_x = int(objects[0]["left"])
            click_y = int(objects[0]["top"])
            
            clicked_hold_id = detect_clicked_hold(click_x, click_y, hold_data, masks)
            if clicked_hold_id is not None:
                st.session_state.selected_hold_id = clicked_hold_id
    
    # 선택된 홀드가 있으면 추천 표시
    if 'selected_hold_id' in st.session_state:
        selected_hold_id = st.session_state.selected_hold_id
        st.write(f"선택된 홀드: {selected_hold_id}")
        
        # 같은 그룹 홀드 찾기
        selected_hold = next((h for h in hold_data if h["id"] == selected_hold_id), None)
        if selected_hold and selected_hold["group"] is not None:
            same_group_holds = [h for h in hold_data if h["group"] == selected_hold["group"]]
            
            st.subheader("🎯 같은 그룹 홀드")
            
            # 결과 이미지 생성
            highlight_rec = original_image.copy()
            for hold in hold_data:
                if i < len(masks):
                    mask = masks[i].astype(np.uint8) * 255
                    if hold["id"] == selected_hold_id:
                        cv2.drawContours(highlight_rec, [cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, (0, 0, 255), 3)  # 빨간색
                    elif hold in same_group_holds:
                        cv2.drawContours(highlight_rec, [cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, (0, 255, 255), 2)  # 노란색
                    else:
                        cv2.drawContours(highlight_rec, [cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, (128, 128, 128), 1)  # 회색
            
            st.image(cv2.cvtColor(highlight_rec, cv2.COLOR_BGR2RGB),
                    caption="그룹 결과 (빨간색: 선택한 홀드, 노란색: 같은 그룹, 회색: 기타)")

    # 🎯 3D RGB 색상 큐브 시각화 (그룹별 색상 표시)
    st.subheader("🎯 RGB 3D 색상 큐브에서의 그룹핑 결과")
    st.write("💡 **그룹핑 결과 확인!** 각 포인트의 색상이 할당된 그룹을 나타냅니다.")
    rgb_group_fig = create_rgb_color_cube_with_groups(hold_data)
    if rgb_group_fig:
        st.plotly_chart(rgb_group_fig, use_container_width=True)

else:
    st.info("👆 클라이밍 벽 이미지를 업로드해주세요!")
