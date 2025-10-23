import os
import cv2
import numpy as np
import base64
import json
from celery import current_task
from celery_app import celery_app
from holdcheck import preprocess, clustering
from backend.gpt4_analyzer import analyze_with_gpt4_vision

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
    try:
        # 1단계: 홀드 감지 (YOLO)
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'message': '🔍 홀드 감지 중...', 'step': 'yolo_detection'}
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
                    'message': '이미지를 읽을 수 없습니다. 올바른 이미지 파일인지 확인해주세요.',
                    'error_type': 'INVALID_IMAGE'
                }
            )
            return {
                'status': 'error',
                'error': '잘못된 이미지 파일',
                'message': '이미지를 읽을 수 없습니다.'
            }
        
        # YOLO 홀드 감지
        from holdcheck.preprocess import preprocess
        hold_data, masks = preprocess(image)
        
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
        
        # 2단계: CLIP 색상 분석
        self.update_state(
            state='PROGRESS',
            meta={
                'progress': 30,
                'message': f'🎨 색상 분석 중... (홀드 {len(holds)}개 감지)',
                'step': 'clip_analysis'
            }
        )
        
        from holdcheck.clustering import rule_based_color_clustering
        colored_holds = rule_based_color_clustering(
            holds,
            None,
            config_path="holdcheck/color_ranges.json"
        )
        
        # 3단계: 문제 생성 (색상별 그룹핑)
        self.update_state(
            state='PROGRESS',
            meta={'progress': 50, 'message': '🧩 문제 생성 중...', 'step': 'problem_generation'}
        )
        
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
        for color_name, group_holds in problems_by_color.items():
            if len(group_holds) >= 3:  # 최소 3개 이상
                # 색상 RGB 추출 (첫 번째 홀드에서)
                color_rgb = group_holds[0].get('dominant_rgb', [128, 128, 128])
                
                # 규칙 기반 분석 (group_holds를 직접 전달, group_id는 None)
                analysis = analyze_problem(group_holds, None, wall_angle)
                if analysis:
                    problems.append({
                        'id': f"ai_{color_name}",
                        'color_name': color_name,
                        'color_rgb': color_rgb,
                        'holds': group_holds,
                        'hold_count': len(group_holds),
                        'analysis': analysis
                    })
        
        # 4단계: GPT-4 분석
        self.update_state(
            state='PROGRESS',
            meta={'progress': 70, 'message': '🤖 GPT-4 분석 중...', 'step': 'gpt4_analysis'}
        )
        
        # 각 문제에 GPT-4 분석 추가
        for i, problem in enumerate(problems):
            try:
                # 진행률 업데이트 (70% + 문제별 진행률)
                progress = 70 + int((i + 1) / len(problems) * 25)  # 70% ~ 95%
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'message': f'🤖 GPT-4 분석 중... ({i+1}/{len(problems)})',
                        'step': 'gpt4_analysis'
                    }
                )
                
                gpt4_result = analyze_with_gpt4_vision(
                    image_base64,
                    problem['holds'],
                    wall_angle
                )
                problem.update(gpt4_result)
            except Exception as e:
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
    """
    비동기 이미지 분석 작업
    
    Args:
        image_data: Base64 인코딩된 이미지 데이터
        wall_angle: 벽 각도 (overhang, slab, face, null)
    
    Returns:
        dict: 분석 결과
    """
    try:
        # 진행률 업데이트: 이미지 디코딩
        self.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'message': '📸 이미지 디코딩 중...', 'step': 'decode'}
        )
        
        # Base64 이미지 디코딩
        image_bytes = base64.b64decode(image_data)
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
        
        # 진행률 업데이트: 홀드 감지 시작
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'message': '🔍 홀드 감지 중...', 'step': 'detection'}
        )
        
        # 🚀 최적화: 전처리 (홀드 감지)
        model_path = "/app/holdcheck/roboflow_weights/weights.pt" if os.path.exists("/app/holdcheck/roboflow_weights/weights.pt") else "/Users/kimjazz/Desktop/project/climbmate/holdcheck/roboflow_weights/weights.pt"
        
        hold_data_raw, masks = preprocess(
            image,
            model_path=model_path,
            mask_refinement=1,
            conf=0.4,
            use_clip_ai=True
        )
        
        if not hold_data_raw:
            self.update_state(
                state='FAILURE',
                meta={
                    'error': '홀드 감지 실패',
                    'message': '이미지에서 홀드를 찾을 수 없습니다.',
                    'error_type': 'NO_HOLDS_DETECTED'
                }
            )
            return {
                'status': 'error',
                'error': '홀드 감지 실패',
                'message': '이미지에서 홀드를 찾을 수 없습니다.'
            }
        
        # 진행률 업데이트: 홀드 감지 완료
        self.update_state(
            state='PROGRESS',
            meta={
                'progress': 30, 
                'message': f'✅ {len(hold_data_raw)}개 홀드 감지 완료', 
                'step': 'detection_complete',
                'holds_count': len(hold_data_raw)
            }
        )
        
        # 진행률 업데이트: 색상 그룹핑
        self.update_state(
            state='PROGRESS',
            meta={'progress': 40, 'message': '🎨 색상 분류 중...', 'step': 'clustering'}
        )
        
        # 규칙 기반 색상 그룹핑
        hold_data = clustering.rule_based_color_clustering(
            hold_data_raw,
            None,
            config_path="holdcheck/color_ranges.json"
        )
        
        # 문제 그룹핑
        problems = clustering.group_holds_by_color(hold_data)
        
        # 진행률 업데이트: 색상 분류 완료
        self.update_state(
            state='PROGRESS',
            meta={
                'progress': 60, 
                'message': f'✅ {len(problems)}개 문제 분류 완료', 
                'step': 'clustering_complete',
                'problems_count': len(problems)
            }
        )
        
        # 진행률 업데이트: AI 문제 분석 시작
        self.update_state(
            state='PROGRESS',
            meta={'progress': 70, 'message': '🤖 AI 문제 분석 중...', 'step': 'analysis'}
        )
        
        # GPT-4 분석
        problems_with_analysis = {}
        for color, holds in problems.items():
            if len(holds) >= 3:  # 최소 3개 홀드 이상인 문제만 분석
                try:
                    # 이미지를 Base64로 인코딩
                    _, buffer = cv2.imencode('.jpg', image)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # GPT-4 분석
                    analysis = analyze_with_gpt4_vision(image_base64, holds, wall_angle)
                    
                    problems_with_analysis[color] = {
                        'color_name': color,
                        'color_rgb': holds[0].get('dominant_rgb', [128, 128, 128]),
                        'holds': holds,
                        'hold_count': len(holds),
                        'analysis': analysis
                    }
                except Exception as e:
                    print(f"GPT-4 분석 실패 ({color}): {e}")
                    problems_with_analysis[color] = {
                        'color_name': color,
                        'color_rgb': holds[0].get('dominant_rgb', [128, 128, 128]),
                        'holds': holds,
                        'hold_count': len(holds),
                        'analysis': None
                    }
        
        # 진행률 업데이트: AI 분석 완료
        self.update_state(
            state='PROGRESS',
            meta={'progress': 90, 'message': '✅ AI 분석 완료', 'step': 'analysis_complete'}
        )
        
        # 진행률 업데이트: 결과 정리
        self.update_state(
            state='PROGRESS',
            meta={'progress': 95, 'message': '📊 결과 정리 중...', 'step': 'finalizing'}
        )
        
        # 최종 결과 구성
        problems_list = list(problems_with_analysis.values())
        
        # 통계 계산
        total_holds = len(hold_data_raw)
        total_problems = len(problems_list)
        analyzable_problems = len([p for p in problems_list if p and p.get('hold_count', 0) >= 3])
        
        statistics = {
            "total_holds": total_holds,
            "total_problems": total_problems,
            "analyzable_problems": analyzable_problems
        }
        
        # 주석 달린 이미지 생성
        annotated_image = None
        if masks is not None:
            try:
                overlay = image.copy()
                colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
                ]
                
                for i, mask in enumerate(masks):
                    if i < len(colors):
                        color = colors[i % len(colors)]
                        overlay[mask > 0.5] = color
                
                annotated = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                _, buffer = cv2.imencode('.jpg', annotated)
                annotated_image = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"⚠️ 주석 이미지 생성 실패: {e}")
        
        # 최종 결과
        result = {
            "problems": problems_list,
            "statistics": statistics,
            "hold_data": hold_data,
            "annotated_image_base64": annotated_image
        }
        
        # 진행률 업데이트: 완료
        self.update_state(
            state='SUCCESS',
            meta={
                'progress': 100, 
                'message': '✅ 분석 완료!', 
                'step': 'complete',
                'result': result
            }
        )
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # 로그 제거
        
        # 오류 발생 시 상태 업데이트
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'❌ 분석 실패: {str(e)}',
                'step': 'error',
                'error': str(e),
                'error_type': 'GENERAL_ERROR'
            }
        )
        return {
            'status': 'error',
            'error': str(e),
            'message': f'분석 중 오류가 발생했습니다: {str(e)}'
        }
