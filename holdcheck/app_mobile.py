import streamlit as st
import cv2
import numpy as np
from preprocess import preprocess
from clustering import clip_ai_color_clustering, analyze_problem
import json

# 📱 PWA 설정 및 모바일 최적화
st.set_page_config(
    page_title="ClimbMate - AI 클라이밍 분석",
    page_icon="🧗‍♀️",
    layout="wide",
    initial_sidebar_state="collapsed",  # 모바일에서 사이드바 기본 닫기
    menu_items={
        'About': "ClimbMate - AI 기반 클라이밍 문제 분석 앱"
    }
)

# 📱 모바일 최적화 CSS
st.markdown("""
<style>
    /* 전체 레이아웃 */
    .main {
        padding: 0.5rem !important;
    }
    
    /* 모바일 헤더 */
    .mobile-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
    }
    
    .mobile-header h1 {
        font-size: 1.8rem;
        margin: 0;
        font-weight: 700;
    }
    
    .mobile-header p {
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
    }
    
    /* 카드 스타일 */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #333;
    }
    
    /* 문제 카드 */
    .problem-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .problem-card:active {
        transform: scale(0.98);
    }
    
    .problem-color {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .problem-info {
        display: inline-block;
        vertical-align: middle;
    }
    
    .problem-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
    }
    
    .problem-count {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* 난이도 배지 */
    .difficulty-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-beginner {
        background: #d4edda;
        color: #155724;
    }
    
    .badge-intermediate {
        background: #fff3cd;
        color: #856404;
    }
    
    .badge-advanced {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* 버튼 스타일 */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* 이미지 업로드 영역 */
    .uploadedFile {
        border-radius: 12px !important;
    }
    
    /* 통계 카드 */
    .stat-card {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    
    /* 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 📱 헤더
st.markdown("""
<div class="mobile-header">
    <h1>🧗‍♀️ ClimbMate</h1>
    <p>AI 기반 클라이밍 문제 분석</p>
</div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'selected_problem' not in st.session_state:
    st.session_state.selected_problem = None

# 📸 이미지 업로드
uploaded_file = st.file_uploader(
    "📸 클라이밍 벽 사진 업로드",
    type=['jpg', 'jpeg', 'png'],
    help="홀드가 잘 보이는 클라이밍 벽 사진을 업로드하세요"
)

if uploaded_file:
    # 이미지 읽기
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 이미지 표시 (모바일 최적화)
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # 분석 버튼
    if st.button("🔍 문제 분석 시작", type="primary"):
        with st.spinner("🤖 AI가 홀드를 분석하는 중..."):
            # 전처리
            hold_data_raw, masks = preprocess(
                original_image,
                mask_refinement=2,
                use_clip_ai=True
            )
            
            if not hold_data_raw:
                st.error("❌ 홀드를 감지하지 못했습니다. 다른 사진을 시도해보세요.")
                st.stop()
            
            # 그룹핑
            hold_data = clip_ai_color_clustering(
                hold_data_raw,
                None,
                original_image,
                masks,
                eps=0.3,
                use_dbscan=False
            )
            
            # 그룹별 정리
            problems = {}
            for hold in hold_data:
                group = hold.get('group')
                if group is None:
                    continue
                
                if group not in problems:
                    # 그룹 색상 추출
                    clip_color = hold.get('clip_color_name', 'unknown')
                    rgb = hold.get('dominant_rgb', [128, 128, 128])
                    
                    problems[group] = {
                        'color_name': clip_color,
                        'color_rgb': rgb,
                        'holds': [],
                        'group_id': group
                    }
                
                problems[group]['holds'].append(hold)
            
            # 세션에 저장
            st.session_state.analyzed_data = {
                'problems': problems,
                'hold_data': hold_data,
                'masks': masks,
                'original_image': original_image
            }
            
            st.success(f"✅ {len(problems)}개의 문제를 발견했습니다!")
            st.rerun()

# 분석 결과 표시
if st.session_state.analyzed_data:
    data = st.session_state.analyzed_data
    problems = data['problems']
    
    # 통계 카드
    st.markdown("### 📊 분석 결과")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(problems)}</div>
            <div class="stat-label">문제 수</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_holds = sum(len(p['holds']) for p in problems.values())
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_holds}</div>
            <div class="stat-label">홀드 수</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        valid_problems = [p for p in problems.values() if len(p['holds']) >= 3]
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(valid_problems)}</div>
            <div class="stat-label">분석 가능</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 문제 목록
    st.markdown("### 🎯 발견된 문제들")
    st.caption("문제를 선택하면 상세 분석을 볼 수 있습니다")
    
    for group_id, problem in sorted(problems.items()):
        color_name = problem['color_name']
        rgb = problem['color_rgb']
        hold_count = len(problem['holds'])
        
        # 색상 이모지 매핑
        color_emoji = {
            'black': '⚫', 'white': '⚪', 'gray': '🔘',
            'red': '🔴', 'orange': '🟠', 'yellow': '🟡',
            'green': '🟢', 'blue': '🔵', 'purple': '🟣',
            'pink': '🩷', 'brown': '🟤', 'mint': '💚', 'lime': '🍃'
        }.get(color_name, '⭕')
        
        # 문제 카드
        col_info, col_action = st.columns([3, 1])
        
        with col_info:
            st.markdown(f"""
            <div style="padding: 0.8rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin: 0.5rem 0;">
                <span style="font-size: 1.5rem;">{color_emoji}</span>
                <span style="font-size: 1.1rem; font-weight: 600; margin-left: 0.5rem;">{color_name.upper()}</span>
                <span style="font-size: 0.9rem; color: #666; margin-left: 0.5rem;">({hold_count}개 홀드)</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col_action:
            if st.button("분석", key=f"analyze_{group_id}"):
                st.session_state.selected_problem = group_id
                st.rerun()
    
    # 선택된 문제 상세 분석
    if st.session_state.selected_problem is not None:
        selected_group = st.session_state.selected_problem
        problem = problems[selected_group]
        
        st.markdown("---")
        st.markdown(f"### 🎯 {problem['color_name'].upper()} 문제 상세 분석")
        
        # 벽 각도 선택
        wall_angle = st.radio(
            "🏔️ 벽 각도",
            options=[None, "overhang", "slab", "face"],
            format_func=lambda x: {
                None: "선택 안함",
                "overhang": "오버행 (90°+)",
                "slab": "슬랩 (90°-)",
                "face": "직벽 (90°)"
            }[x],
            horizontal=True
        )
        
        # AI 분석
        analysis = analyze_problem(data['hold_data'], selected_group, wall_angle)
        
        if analysis:
            # 난이도
            diff = analysis['difficulty']
            st.markdown(f"""
            <div class="card">
                <div class="card-header">🎯 난이도</div>
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #667eea;">{diff['grade']}</div>
                    <div style="font-size: 1rem; color: #666; margin-top: 0.5rem;">{diff['level']}</div>
                    <div style="margin-top: 1rem;">{'★' * int(diff['confidence'] * 5)}{'☆' * (5 - int(diff['confidence'] * 5))}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 유형
            climb_type = analysis['climb_type']
            st.markdown(f"""
            <div class="card">
                <div class="card-header">🏋️ 문제 유형</div>
                <div style="padding: 0.5rem 0;">
                    <div style="font-size: 1.3rem; font-weight: 600; color: #333; margin-bottom: 0.8rem;">
                        {climb_type['primary_type']}
                    </div>
            """, unsafe_allow_html=True)
            
            for type_name in climb_type['types'][:4]:
                st.markdown(f"• {type_name}")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # 상세 정보
            with st.expander("📋 상세 분석 보기"):
                stats = analysis['statistics']
                st.write(f"**홀드 개수**: {stats['num_holds']}")
                st.write(f"**평균 크기**: {stats['avg_hold_size']}")
                st.write(f"**최대 간격**: {stats['max_distance']}")
                
                st.write("")
                st.write("**난이도 요인:**")
                for key, value in diff['factors'].items():
                    st.write(f"• {value}")
        
        # 닫기 버튼
        if st.button("← 목록으로 돌아가기"):
            st.session_state.selected_problem = None
            st.rerun()

else:
    # 빈 상태 안내
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #666;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📸</div>
        <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
            클라이밍 벽 사진을 업로드하세요
        </div>
        <div style="font-size: 0.9rem;">
            AI가 홀드를 분석하고 문제를 추천해드립니다
        </div>
    </div>
    """, unsafe_allow_html=True)

# 하단 정보
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #999; font-size: 0.85rem;">
    <div>🧗‍♀️ ClimbMate v1.0</div>
    <div>AI 기반 클라이밍 문제 분석</div>
</div>
""", unsafe_allow_html=True)

