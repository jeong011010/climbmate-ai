import os
import cv2
import numpy as np
import base64
import json
from celery import current_task
from celery_app import celery_app
from holdcheck import preprocess, clustering
from backend.gpt4_analyzer import analyze_with_gpt4_vision

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
            raise ValueError("잘못된 이미지 파일")
        
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
        
        # CLIP 색상 분석
        colored_holds = clustering.clip_ai_color_clustering(
            hold_data,
            None,
            image,
            masks,
            eps=0.3,
            use_dbscan=False
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
        # 오류 발생 시 상태 업데이트
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'❌ CLIP 분석 실패: {str(e)}',
                'step': 'error',
                'error': str(e)
            }
        )
        raise e
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
            raise ValueError("잘못된 이미지 파일")
        
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
            raise ValueError("홀드를 감지하지 못했습니다")
        
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
        
        # 색상 그룹핑
        hold_data = clustering.clip_ai_color_clustering(
            hold_data_raw,
            None,
            image,
            masks,
            eps=0.3,
            use_dbscan=False
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
        # 오류 발생 시 상태 업데이트
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0,
                'message': f'❌ 분석 실패: {str(e)}',
                'step': 'error',
                'error': str(e)
            }
        )
        raise e
