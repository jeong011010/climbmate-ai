import os
import cv2
import numpy as np
import base64
import json
import asyncio
from celery import current_task
from celery_app import celery_app
from holdcheck import preprocess, clustering
from backend.gpt4_analyzer import analyze_with_gpt4_vision, translate_and_enhance_gpt4_result

# 🎯 YOLO 모델 선택 (환경 변수로 설정 가능)
# 'roboflow' (기본) 또는 'alternative' (새 모델 테스트)
YOLO_MODEL = os.getenv('YOLO_MODEL', 'roboflow')

# 모델 경로 설정
if YOLO_MODEL == 'alternative':
    MODEL_PATH_LOCAL = "/Users/kimjazz/Desktop/project/climbmate/holdcheck/roboflow_weights/weights_alternative.pt"
    MODEL_PATH_DOCKER = "/app/holdcheck/roboflow_weights/weights_alternative.pt"
    print(f"🎯 대체 YOLO 모델 사용: weights_alternative.pt")
else:
    MODEL_PATH_LOCAL = "/Users/kimjazz/Desktop/project/climbmate/holdcheck/roboflow_weights/weights.pt"
    MODEL_PATH_DOCKER = "/app/holdcheck/roboflow_weights/weights.pt"
    print(f"🎯 기본 Roboflow 모델 사용: weights.pt")

def convert_to_serializable(obj):
    """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

@celery_app.task(bind=True)
def analyze_image_async(self, image_base64, wall_angle=None):
    """
    전체 이미지 분석 비동기 작업 (YOLO + CLIP + GPT-4)
    
    Args:
        image_base64: Base64 인코딩된 이미지 데이터
        wall_angle: 벽 각도 (overhang, slab, face, null)
    
    Returns:
        dict: 전체 분석 결과
    """
    print("=" * 80)
    print("🚀 analyze_image_async 시작!")
    print(f"   이미지 크기: {len(image_base64) if image_base64 else 0} bytes")
    print(f"   Wall angle: {wall_angle}")
    print("=" * 80)
    
    try:
        # 1단계: 이미지 디코딩 및 YOLO 모델 로딩 (0~40%)
        import time
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'message': '📷 이미지 디코딩 중...', 'step': 'image_decoding'}
        )
        time.sleep(0.5)  # 프론트엔드 폴링을 위한 최소 지연
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'message': '📷 이미지 디코딩 완료', 'step': 'image_decoding_complete'}
        )
        time.sleep(0.5)
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'message': '🧠 YOLO 모델 로딩 중...', 'step': 'yolo_loading'}
        )
        time.sleep(0.5)
        
        # Base64 이미지 디코딩
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            self.update_state(
                state='FAILURE',
                meta={
                    'error': '잘못된 이미지 파일',
                    'message': '이미지를 읽을 수 없습니다. 올바른 이미지 파일인지 확인해주세요.',
                    'error_type': 'INVALID_IMAGE'
                }
            )
            return {
                'status': 'error',
                'error': '잘못된 이미지 파일',
                'message': '이미지를 읽을 수 없습니다.'
            }
        
        # YOLO 모델 로딩 시작 (실제 로딩은 preprocess 내부에서 발생)
        self.update_state(
            state='PROGRESS',
            meta={'progress': 15, 'message': '🧠 YOLO 모델 초기화 중...', 'step': 'yolo_loading'}
        )
        
        # YOLO 홀드 감지
        from holdcheck.preprocess import preprocess
        
        # 선택된 모델 사용
        model_path = MODEL_PATH_DOCKER if os.path.exists(MODEL_PATH_DOCKER) else MODEL_PATH_LOCAL
        print(f"🎯 사용 중인 모델: {model_path}")
        
        self.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'message': '🔍 홀드 감지 실행 중...', 'step': 'yolo_detection'}
        )
        
        hold_data, masks = preprocess(image, model_path=model_path)
        
        # 홀드 감지 완료 (40%까지)
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'message': f'✅ {len(hold_data) if hold_data else 0}개 홀드 감지 완료', 'step': 'detection_complete'}
        )
        time.sleep(0.5)
        
        if not hold_data:
            self.update_state(
                state='FAILURE',
                meta={
                    'error': '홀드 감지 실패',
                    'message': '이미지에서 홀드를 찾을 수 없습니다. 다른 이미지를 시도해주세요.',
                    'error_type': 'NO_HOLDS_DETECTED'
                }
            )
            return {
                'status': 'error',
                'error': '홀드 감지 실패',
                'message': '이미지에서 홀드를 찾을 수 없습니다.'
            }
        
        # hold_data는 홀드 리스트입니다
        holds = hold_data
        total_holds = len(holds)
        
        # 2단계: 색상 분류 (42% ~ 60%)
        self.update_state(
            state='PROGRESS',
            meta={
                'progress': 42,
                'message': f'🎨 색상 분류 시작... (홀드 {total_holds}개)',
                'step': 'color_classification',
                'holds_count': total_holds
            }
        )
        time.sleep(0.5)
        
        from holdcheck.color_classifier import rule_based_color_clustering
        
        colored_holds = rule_based_color_clustering(
            holds,
            None,
            config_path="holdcheck/color_ranges.json"
        )
        
        # 색상 클러스터링 완료 (60%까지)
        self.update_state(
            state='PROGRESS',
            meta={'progress': 60, 'message': f'✅ 색상 클러스터링 완료', 'step': 'color_clustering_complete'}
        )
        time.sleep(0.5)
        
        # 3단계: 문제 분석 (60% ~ 80%)
        self.update_state(
            state='PROGRESS',
            meta={'progress': 65, 'message': '📊 문제 분석 시작...', 'step': 'problem_analysis'}
        )
        time.sleep(0.5)
        
        # 색상별로 그룹핑
        from holdcheck.clustering import analyze_problem
        problems_by_color = {}
        
        for hold in colored_holds:
            # 여러 가능한 색상 필드 확인
            color_name = (hold.get('clip_color_name') or 
                        hold.get('color_name') or 
                        hold.get('group', '').replace('ai_', '') or 
                        'unknown')
            if color_name not in problems_by_color:
                problems_by_color[color_name] = []
            problems_by_color[color_name].append(hold)
        
        # 각 색상 그룹을 문제로 분석
        problems = []
        total_colors = len(problems_by_color)
        processed_colors = 0
        
        for color_name, group_holds in problems_by_color.items():
            if len(group_holds) >= 1:  # 모든 홀드 그룹 표시
                # 진행률 업데이트 (60% ~ 80% 사이)
                processed_colors += 1
                progress = 60 + int((processed_colors / total_colors) * 20)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress, 
                        'message': f'🧩 {color_name} 문제 분석 중... ({processed_colors}/{total_colors})',
                        'step': 'problem_analysis',
                        'problems_count': len(problems)
                    }
                )
                
                # 색상 RGB 추출 (첫 번째 홀드에서)
                color_rgb = group_holds[0].get('dominant_rgb', [128, 128, 128])
                
                # 각 홀드에 bbox, color, individual_color 추가 (contour는 preprocess.py에서 이미 계산됨)
                enriched_holds = []
                for hold in group_holds:
                    hold_id = hold['id']
                    bbox = [0, 0, 0, 0]
                    
                    # 마스크에서 bbox 계산
                    if hold_id < len(masks):
                        mask = masks[hold_id]
                        coords = np.argwhere(mask > 0.5)
                        if len(coords) > 0:
                            y_min, x_min = coords.min(axis=0)
                            y_max, x_max = coords.max(axis=0)
                            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                    
                    # 홀드 정보 추가
                    enriched_hold = {
                        'id': hold['id'],
                        'center': hold['center'],
                        'area': hold['area'],
                        'bbox': bbox,
                        'contour': hold.get('contour', []),  # preprocess.py에서 이미 계산됨
                        'color': color_name,  # 그룹 색상 (문제 색상)
                        'individual_color': hold.get('clip_color_name', 'unknown'),  # 홀드 자체의 실제 색상
                        'rgb': hold.get('dominant_rgb', [128, 128, 128]),
                        'hsv': hold.get('dominant_hsv', [0, 0, 128]),
                        # 🎨 ML 학습용 전체 색상 특징
                        'dominant_lab': hold.get('dominant_lab', [0, 0, 0]),
                        'hsv_stats': hold.get('hsv_stats', {}),
                        'rgb_stats': hold.get('rgb_stats', {}),
                        'lab_stats': hold.get('lab_stats', {})
                    }
                    enriched_holds.append(enriched_hold)
                
                # 규칙 기반 분석 (원본 group_holds 사용)
                analysis = analyze_problem(group_holds, None, wall_angle)
                if analysis:
                    problems.append({
                        'id': f"ai_{color_name}",
                        'color_name': color_name,
                        'color_rgb': color_rgb,
                        'holds': enriched_holds,  # bbox와 색상이 추가된 홀드 사용
                        'hold_count': len(enriched_holds),
                        'analysis': analysis
                    })
        
        # 4단계: GPT-4 분석 (80% ~ 99%)
        total_problems = len(problems)
        self.update_state(
            state='PROGRESS',
            meta={'progress': 80, 'message': f'🤖 GPT-4 분석 시작... ({total_problems}개 문제)', 'step': 'gpt4_analysis'}
        )
        
        # 🚀 모든 문제를 동시에 GPT-4에 요청 (병렬 처리)
        async def analyze_all_problems():
            tasks = []
            for problem in problems:
                task = analyze_with_gpt4_vision(
                    image_base64,
                    problem['holds'],
                    wall_angle
                )
                tasks.append(task)
            
            # 모든 요청을 동시에 처리 (병렬)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과를 각 문제에 적용
            for i, (problem, result) in enumerate(zip(problems, results)):
                # 진행률 업데이트 (80% ~ 99% 사이)
                progress = 80 + int(((i + 1) / total_problems) * 19)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'message': f'🤖 GPT-4 분석 중... ({i + 1}/{total_problems})',
                        'step': 'gpt4_analysis'
                    }
                )
                
                if isinstance(result, Exception):
                    print(f"⚠️ 문제 {i} GPT-4 분석 실패: {result}")
                    problem['gpt4_available'] = False
                else:
                    # 루트 검증 및 한글 번역 적용
                    translated = translate_and_enhance_gpt4_result(result, problem['holds'])
                    
                    # GPT-4 결과를 문제에 병합
                    problem.update(translated)
                    # 프론트엔드 전달용 필드 추가
                    problem['gpt4_secondary_types'] = translated.get('secondary_types', [])
                    problem['gpt4_key_factors'] = translated.get('key_factors', [])
                    problem['gpt4_crux'] = translated.get('crux', '')
                    problem['gpt4_movements'] = translated.get('movements', [])
                    problem['gpt4_challenges'] = translated.get('challenges', [])
                    problem['gpt4_tips'] = translated.get('tips', [])
                    problem['gpt4_comparison'] = translated.get('comparison', '')
                    problem['gpt4_route'] = translated.get('route', [])
        
        # 비동기 함수 실행
        import asyncio
        try:
            asyncio.run(analyze_all_problems())
        except Exception as e:
            print(f"⚠️ GPT-4 병렬 분석 오류: {e}")
            for problem in problems:
                problem['gpt4_available'] = False
        
        # 5단계: 홀드가 표시된 이미지 생성
        # TODO: 임시로 비활성화 (Redis 크기 제한 테스트)
        annotated_base64 = ''  # 빈 문자열로 테스트
        
        # 완료
        self.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'message': '✅ 분석 완료!', 'step': 'complete'}
        )
        
        # 결과 반환 (numpy 타입 변환)
        result = {
            'problems': convert_to_serializable(problems),
            'statistics': {
                'total_holds': len(holds),
                'total_problems': len(problems),
                'analyzable_problems': len(problems)
            },
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}'
        }
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ 분석 실패: {str(e)}")
        print(error_details)
        
        # Celery에게 제대로 된 예외를 던져야 함
        raise Exception(f'분석 실패: {str(e)}')

@celery_app.task(bind=True)
def analyze_colors_with_clip_async(self, image_base64, hold_data):
    """
    CLIP 색상 분석 비동기 작업
    
    Args:
        image_base64: Base64 인코딩된 이미지 데이터
        hold_data: 홀드 감지 결과
    
    Returns:
        list: 색상 분석된 홀드 데이터
    """
    try:
        # 진행률 업데이트: 이미지 디코딩
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'message': '📸 이미지 디코딩 중...', 'step': 'decode'}
        )
        
        # Base64 이미지 디코딩
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            self.update_state(
                state='FAILURE',
                meta={
                    'error': '잘못된 이미지 파일',
                    'message': '이미지를 읽을 수 없습니다.',
                    'error_type': 'INVALID_IMAGE'
                }
            )
            return {
                'status': 'error',
                'error': '잘못된 이미지 파일',
                'message': '이미지를 읽을 수 없습니다.'
            }
        
        # 진행률 업데이트: CLIP 분석 시작
        self.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'message': '🎨 CLIP 색상 분석 중...', 'step': 'clip_analysis'}
        )
        
        # 홀드 데이터를 마스크로 변환 (간단한 구현)
        masks = []
        for hold in hold_data:
            # 홀드 중심점 주변을 마스크로 생성
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            center_x, center_y = int(hold['center'][0]), int(hold['center'][1])
            radius = int(np.sqrt(hold['area']) / 2)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            masks.append(mask)
        
        # 규칙 기반 색상 분석
        colored_holds = clustering.rule_based_color_clustering(
            hold_data,
            None,
            config_path="holdcheck/color_ranges.json"
        )
        
        # 진행률 업데이트: 완료
        self.update_state(
            state='SUCCESS',
            meta={
                'progress': 100, 
                'message': '✅ CLIP 색상 분석 완료', 
                'step': 'complete',
                'result': colored_holds
            }
        )
        
        return colored_holds
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ CLIP 분석 오류: {error_details}")
        
        # 오류 발생 시 상태 업데이트
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'❌ CLIP 분석 실패: {str(e)}',
                'step': 'error',
                'error': str(e),
                'error_type': 'CLIP_ERROR'
            }
        )
        return {
            'status': 'error',
            'error': str(e),
            'message': f'CLIP 색상 분석 실패: {str(e)}'
        }
