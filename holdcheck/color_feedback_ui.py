"""
🎨 색상 분류 피드백 UI - 사용자가 직접 색상을 수정하며 학습시킴
"""

import streamlit as st
import cv2
import numpy as np
from clustering import (
    rule_based_color_clustering,
    save_user_feedback,
    load_color_ranges,
    export_feedback_dataset,
    draw_holds_on_image_with_highlights # <--- Add this import
)
from preprocess import preprocess


def show_feedback_ui():
    """색상 피드백 수집 UI"""
    st.title("🎨 색상 분류 피드백 시스템")
    
    st.markdown("""
    ### 💡 사용 방법
    1. 이미지 업로드
    2. 자동 색상 분류 결과 확인
    3. 잘못 분류된 홀드 수정
    4. **피드백 저장** 버튼 클릭
    5. 다음 분석부터 자동으로 개선된 분류 적용!
    """)
    
    # 이미지 업로드
    uploaded_file = st.file_uploader("암벽 이미지 업로드", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is None:
        st.info("👆 이미지를 업로드하세요")
        return
    
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 세션 초기화
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
    
    # 전처리 (CLIP 없이)
    if st.button("🔍 색상 분석 시작", type="primary"):
        with st.spinner("홀드 검출 및 색상 분석 중..."):
            # 전처리
            hold_data_raw, masks = preprocess(
                image,
                use_clip_ai=False  # CLIP 사용 안 함
            )
            
            if not hold_data_raw:
                st.error("❌ 홀드를 감지하지 못했습니다.")
                return
            
            # 룰 기반 색상 분류
            hold_data = rule_based_color_clustering(
                hold_data_raw,
                None,
                confidence_threshold=0.5  # 낮춰서 더 많이 분류
            )
            
            # 세션에 저장
            st.session_state.hold_data = hold_data
            st.session_state.masks = masks
            st.session_state.image = image
            
        st.success(f"✅ {len(hold_data)}개 홀드 분석 완료!")
        st.rerun()
    
    # 분석 결과가 있으면
    if 'hold_data' in st.session_state:
        hold_data = st.session_state.hold_data
        masks = st.session_state.masks
        image = st.session_state.image
        
        st.markdown("---")
        st.subheader("📊 분류 결과")
        
        # 색상별 통계
        color_groups = {}
        for hold in hold_data:
            color = hold.get('clip_color_name', 'unknown')
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(hold)
        
        cols = st.columns(5)
        for idx, (color, holds) in enumerate(sorted(color_groups.items())):
            with cols[idx % 5]:
                st.metric(color, len(holds))
        
        st.markdown("---")
        st.subheader("🔧 홀드별 색상 수정")
        
        # 색상 선택지
        color_options = ["black", "white", "gray", "red", "orange", "yellow", 
                        "green", "mint", "blue", "purple", "pink", "brown", "unknown"]
        
        # 홀드별 수정 UI
        st.markdown("### 잘못 분류된 홀드를 수정하세요:")
        
        # 피드백 모드에서 문제가 있는 홀드들을 강조 표시
        problems_dict = {}
        for change in feedback_changes:
            problems_dict[str(change['hold_id'])] = {
                "predicted_color": change['predicted_color'],
                "correct_color": change['correct_color']
            }
        
        # 강조 표시가 적용된 이미지 생성
        highlighted_image = draw_holds_on_image_with_highlights(
            image, hold_data, 
            [[hold.get("bbox", [0,0,100,100])[0], hold.get("bbox", [0,0,100,100])[1], 
              hold.get("bbox", [0,0,100,100])[2], hold.get("bbox", [0,0,100,100])[3]] for hold in hold_data],
            problems_dict
        )
        
        st.image(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB), 
                 caption="🔴 빨간 테두리 = 수정된 홀드, 🟡 노란 테두리 = 신뢰도 낮음", 
                 use_container_width=True)
        
        # 홀드별 수정
        st.markdown("### 수정할 홀드 선택:")
        
        # 3열로 표시
        num_cols = 3
        rows = [hold_data[i:i+num_cols] for i in range(0, len(hold_data), num_cols)]
        
        feedback_changes = []
        
        for row in rows:
            cols = st.columns(num_cols)
            for idx, hold in enumerate(row):
                with cols[idx]:
                    hold_id = hold["id"]
                    current_color = hold.get('clip_color_name', 'unknown')
                    confidence = hold.get('clip_confidence', 0)
                    rgb = hold.get('dominant_rgb', [128, 128, 128])
                    
                    # 홀드 정보 표시
                    st.markdown(f"**홀드 ID {hold_id}**")
                    st.markdown(f"현재: `{current_color}` ({confidence:.0%})")
                    st.markdown(f"RGB: {rgb}")
                    
                    # 색상 선택
                    new_color = st.selectbox(
                        "올바른 색상:",
                        options=color_options,
                        index=color_options.index(current_color),
                        key=f"color_{hold_id}"
                    )
                    
                    # 변경 감지
                    if new_color != current_color:
                        feedback_changes.append({
                            "hold_id": hold_id,
                            "correct_color": new_color,
                            "predicted_color": current_color,
                            "rgb": rgb,
                            "hsv": hold.get('dominant_hsv')
                        })
                        st.success(f"✏️ {current_color} → {new_color}")
        
        # 피드백 저장
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 피드백 저장 및 학습", type="primary", disabled=len(feedback_changes)==0):
                if feedback_changes:
                    # 피드백 저장
                    save_user_feedback(hold_data, feedback_changes)
                    
                    # 세션에 누적
                    st.session_state.feedback_data.extend(feedback_changes)
                    
                    st.success(f"✅ {len(feedback_changes)}개 피드백 저장 완료!")
                    st.info("💡 다음 분석부터 개선된 분류가 적용됩니다!")
                    
                    # 재분석
                    st.rerun()
        
        with col2:
            if st.button("📊 누적 피드백 보기"):
                show_feedback_stats()
        
        with col3:
            if st.button("📤 학습 데이터 내보내기"):
                export_feedback_dataset()
                st.success("✅ 데이터셋 내보내기 완료!")
        
        # 통계 표시
        if feedback_changes:
            st.markdown(f"### 수정 대기 중: {len(feedback_changes)}개")
            for change in feedback_changes:
                st.markdown(f"- 홀드 {change['hold_id']}: `{change['predicted_color']}` → `{change['correct_color']}`")


def show_feedback_stats():
    """피드백 통계 표시"""
    if 'feedback_data' not in st.session_state or not st.session_state.feedback_data:
        st.info("아직 피드백 데이터가 없습니다.")
        return
    
    feedback_data = st.session_state.feedback_data
    
    st.subheader(f"📊 누적 피드백: {len(feedback_data)}건")
    
    # 오분류 패턴
    patterns = {}
    for fb in feedback_data:
        key = f"{fb['predicted_color']} → {fb['correct_color']}"
        if key not in patterns:
            patterns[key] = 0
        patterns[key] += 1
    
    st.markdown("### 주요 오분류 패턴:")
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        st.markdown(f"- **{pattern}**: {count}건")
    
    # 색상별 정확도 추정
    st.markdown("### 예상 개선 효과:")
    st.markdown("다음 분석부터 이 패턴들이 자동으로 수정됩니다!")


def show_color_ranges_editor():
    """색상 범위 직접 편집 UI (고급)"""
    st.title("🎨 색상 범위 직접 편집")
    
    st.warning("⚠️ 고급 사용자 전용: 색상 범위를 직접 수정할 수 있습니다.")
    
    # 색상 범위 로드
    ranges_data = load_color_ranges()
    
    st.json(ranges_data)
    
    st.markdown("### 수동 편집:")
    edited_json = st.text_area(
        "JSON 수정 (주의: 잘못된 형식은 오류 발생)",
        value=str(ranges_data),
        height=400
    )
    
    if st.button("💾 수정사항 저장"):
        try:
            import json
            new_ranges = json.loads(edited_json)
            from clustering import save_color_ranges
            save_color_ranges(new_ranges)
            st.success("✅ 색상 범위 업데이트 완료!")
        except Exception as e:
            st.error(f"❌ 오류: {e}")


if __name__ == "__main__":
    # Streamlit 앱 실행
    page = st.sidebar.selectbox(
        "페이지 선택",
        ["피드백 수집", "색상 범위 편집"]
    )
    
    if page == "피드백 수집":
        show_feedback_ui()
    else:
        show_color_ranges_editor()

