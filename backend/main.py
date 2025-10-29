from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import asyncio
import json
import cv2
import numpy as np
import sys
import os
import base64

# 🎯 YOLO 모델 선택 (환경 변수로 설정 가능)
# 'roboflow' 또는 'alternative'
YOLO_MODEL = os.getenv('YOLO_MODEL', 'roboflow')

# holdcheck 모듈 경로 추가
holdcheck_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'holdcheck')
sys.path.insert(0, holdcheck_path)

# backend 모듈 경로 추가
backend_path = os.path.dirname(__file__)
sys.path.insert(0, backend_path)

from preprocess import preprocess
from clustering import clip_ai_color_clustering, analyze_problem

# 데이터베이스 및 분석 모듈 (선택적 로드)
try:
    from database import save_problem, save_user_feedback, get_model_stats, get_training_data, convert_gpt4_to_training_data
    DB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Database 모듈 없음: {e}")
    DB_AVAILABLE = False

try:
    from backend.gpt4_analyzer import analyze_with_gpt4_vision, get_gpt4_status
    GPT4_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GPT-4 모듈 없음: {e}")
    GPT4_AVAILABLE = False

try:
    from hybrid_analyzer import hybrid_analyze, get_analysis_method_stats
    HYBRID_AVAILABLE = True
    print("✅ Hybrid Analyzer 로드 완료")
except ImportError as e:
    print(f"⚠️ Hybrid 모듈 없음: {e}")
    HYBRID_AVAILABLE = False

try:
    from ml_trainer import train_difficulty_model, train_type_model
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML 모듈 없음: {e}")
    ML_AVAILABLE = False

# Pydantic 모델
class FeedbackRequest(BaseModel):
    problem_id: int
    user_difficulty: str
    user_type: str
    user_feedback: str = None

class HoldColorFeedbackRequest(BaseModel):
    problem_id: int
    hold_id: str
    predicted_color: str
    user_color: str
    hold_center: Optional[List[float]] = None
    hold_features: Optional[Dict[str, Any]] = None  # 홀드의 전체 색상 특징 데이터

app = FastAPI(title="ClimbMate API", version="1.0.0")

# 🚀 성능 최적화: 시작 시 CLIP 모델 미리 로딩
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    try:
        print("🚀 서버 시작 완료")
        print("⚡ CLIP 모델은 첫 요청 시 자동 로딩됩니다 (메모리 최적화)")
        # CLIP 모델은 메모리 부족 방지를 위해 첫 요청 시 lazy loading
        # clustering.py와 preprocess.py의 get_clip_model()에서 자동 캐싱
    except Exception as e:
        print(f"⚠️ 서버 시작 실패: {e}")

# CORS 설정 (React 개발 서버용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """헬스체크"""
    return {"status": "ok", "message": "ClimbMate API is running"}

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    wall_angle: str = None
):
    """
    클라이밍 벽 이미지 분석
    
    Parameters:
    - file: 이미지 파일
    - wall_angle: 벽 각도 (overhang, slab, face, null)
    
    Returns:
    - problems: 발견된 문제 목록
    - statistics: 통계 정보
    """
    try:
        # 이미지 읽기 및 크기 최적화
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 원본 이미지 크기 로그
        height, width = image.shape[:2]
        print(f"📸 원본 이미지: {width}x{height}")
        # ⚠️ 1차 리사이즈 제거 - preprocess.py에서 한 번만 리사이즈 (640x640)
        # → 작은 홀드 보존 & 디테일 향상
        
        # 🚀 최적화: 전처리 (홀드 감지)
        print(f"🔍 홀드 감지 시작...")
        # 배포 환경에 따른 모델 경로 설정
        if os.path.exists("/app/holdcheck/roboflow_weights/weights.pt"):
            model_path = "/app/holdcheck/roboflow_weights/weights.pt"  # Docker 환경
        else:
            model_path = "/Users/kimjazz/Desktop/project/climbmate/holdcheck/roboflow_weights/weights.pt"  # 로컬 환경
        
        hold_data_raw, masks = preprocess(
            image,
            model_path=model_path,
            mask_refinement=0,  # 마스크 정제 최소화 (속도 우선)
            conf=0.5,  # 더 확실한 홀드만 (노이즈 감소)
            use_clip_ai=True
        )
        
        if not hold_data_raw:
            return JSONResponse(
                status_code=200,
                content={
                    "problems": [],
                    "statistics": {"total_holds": 0, "total_problems": 0},
                    "message": "홀드를 감지하지 못했습니다."
                }
            )
        
        print(f"✅ {len(hold_data_raw)}개 홀드 감지 완료")
        
        # 그룹핑 (색상 기반)
        print(f"🎨 색상 그룹핑 시작...")
        hold_data = clip_ai_color_clustering(
            hold_data_raw,
            None,
            image,
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
                clip_color = hold.get('clip_color_name', 'unknown')
                rgb = hold.get('dominant_rgb', [128, 128, 128])
                
                problems[group] = {
                    'id': group,
                    'color_name': clip_color,
                    'color_rgb': rgb,
                    'holds': [],
                    'hold_count': 0,
                    'analysis': None
                }
            
            # bbox 계산 (contour는 preprocess.py에서 이미 추출됨)
            hold_id = hold['id']
            bbox = [0, 0, 0, 0]
            
            if hold_id < len(masks):
                mask = masks[hold_id]
                coords = np.argwhere(mask > 0.5)
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            
            problems[group]['holds'].append({
                'id': hold['id'],
                'center': hold['center'],
                'area': hold['area'],
                'bbox': bbox,
                'contour': hold.get('contour', []),  # preprocess.py에서 이미 계산됨
                'color': problems[group]['color_name'],  # 그룹 색상 (문제 색상)
                'individual_color': hold.get('clip_color_name', 'unknown'),  # 홀드 자체의 실제 색상
                'rgb': hold.get('dominant_rgb', [128, 128, 128]),
                'hsv': hold.get('dominant_hsv', [0, 0, 128]),
                # 🎨 ML 학습용 전체 색상 특징
                'dominant_lab': hold.get('dominant_lab', [0, 0, 0]),
                'hsv_stats': hold.get('hsv_stats', {}),
                'rgb_stats': hold.get('rgb_stats', {}),
                'lab_stats': hold.get('lab_stats', {})
            })
        
        # 이미지를 Base64로 인코딩 (GPT-4 및 DB 저장용)
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 홀드 수 업데이트
        for group_id, problem in problems.items():
            problem['hold_count'] = len(problem['holds'])
        
        # 🚀 3개 이상인 문제를 병렬로 분석
        analyzable_problems = [(group_id, problem) for group_id, problem in problems.items() if problem['hold_count'] >= 3]
        
        if analyzable_problems and HYBRID_AVAILABLE:
            print(f"🚀 {len(analyzable_problems)}개 문제 병렬 분석 시작...")
            
            async def analyze_all_problems_parallel():
                tasks = []
                
                for group_id, problem in analyzable_problems:
                    # 규칙 기반 분석
                    rule_analysis = analyze_problem(
                        hold_data,
                        group_id,
                        wall_angle if wall_angle != "null" else None
                    )
                    
                    # 하이브리드 분석 태스크 생성
                    task = hybrid_analyze(
                        image_base64=image_base64,
                        holds_data=problem['holds'],
                        wall_angle=wall_angle if wall_angle != "null" else None,
                        rule_based_analysis=rule_analysis
                    )
                    tasks.append((group_id, problem, task, rule_analysis))
                
                # 모든 분석을 동시에 실행
                hybrid_tasks = [task for _, _, task, _ in tasks]
                results = await asyncio.gather(*hybrid_tasks, return_exceptions=True)
                
                # 결과 적용
                for (group_id, problem, _, rule_analysis), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        print(f"⚠️ 문제 {group_id} 하이브리드 분석 실패: {result}")
                        problem['analysis'] = rule_analysis
                    else:
                        # 하이브리드 결과를 기존 분석 구조에 통합
                        rule_analysis['difficulty']['grade'] = result['difficulty']['grade']
                        rule_analysis['difficulty']['confidence'] = result['difficulty']['confidence']
                        rule_analysis['climb_type']['primary_type'] = result['type']['primary_type']
                        rule_analysis['climb_type']['confidence'] = result['type']['confidence']
                        rule_analysis['analysis_method'] = result['method_used']
                        
                        if 'gpt4_reasoning' in result:
                            rule_analysis['gpt4_reasoning'] = result['gpt4_reasoning']
                        
                        problem['analysis'] = rule_analysis
            
            # 병렬 분석 실행
            try:
                await analyze_all_problems_parallel()
                print(f"✅ {len(analyzable_problems)}개 문제 병렬 분석 완료!")
            except Exception as e:
                print(f"⚠️ 병렬 분석 오류: {e}")
                # 실패 시 순차 처리로 폴백
                for group_id, problem in analyzable_problems:
                    rule_analysis = analyze_problem(hold_data, group_id, wall_angle if wall_angle != "null" else None)
                    problem['analysis'] = rule_analysis
        else:
            # 병렬 처리 없이 규칙 기반만 사용
            for group_id, problem in analyzable_problems:
                print(f"🤖 문제 {group_id} 규칙 기반 분석...")
                rule_analysis = analyze_problem(
                    hold_data,
                    group_id,
                    wall_angle if wall_angle != "null" else None
                )
                problem['analysis'] = rule_analysis
        
        # DB에 저장 (가능한 경우)
        if DB_AVAILABLE:
            for group_id, problem in analyzable_problems:
                if problem.get('analysis'):
                    try:
                        rule_analysis = problem['analysis']
                        gpt4_save_data = {
                            'difficulty': rule_analysis['difficulty']['grade'],
                            'type': rule_analysis['climb_type']['primary_type'],
                            'confidence': rule_analysis['difficulty']['confidence'],
                            'method': rule_analysis.get('analysis_method', 'rule_based'),
                            'reasoning': rule_analysis.get('gpt4_reasoning', '')
                        }
                        
                        problem_id = save_problem(
                            image_base64=image_base64,
                            holds_data=problem['holds'],
                            gpt4_result=gpt4_save_data,
                            wall_angle=wall_angle if wall_angle != "null" else None,
                            image_width=image.shape[1],
                            image_height=image.shape[0],
                            statistics=rule_analysis.get('statistics', {})
                        )
                        problem['db_id'] = problem_id
                        print(f"✅ 문제 {group_id} → DB ID {problem_id}")
                    except Exception as e:
                        print(f"⚠️ DB 저장 실패: {e}")
                        problem['db_id'] = None
        
        print(f"✅ {len(problems)}개 문제 분석 완료")
        
        # 🎨 주석 이미지 생성 (색상별로 홀드 표시)
        annotated_image = image.copy()
        
        # 색상 매핑 (BGR)
        color_map_bgr = {
            'black': (50, 50, 50), 'white': (240, 240, 240), 'gray': (128, 128, 128),
            'red': (0, 0, 255), 'orange': (0, 165, 255), 'yellow': (0, 255, 255),
            'green': (0, 255, 0), 'blue': (255, 0, 0), 'purple': (255, 0, 255),
            'pink': (203, 192, 255), 'brown': (42, 42, 165), 
            'mint': (170, 255, 170), 'lime': (0, 255, 127)
        }
        
        for problem in problems.values():
            color_name = problem['color_name']
            bgr_color = color_map_bgr.get(color_name, (128, 128, 128))
            
            for hold in problem['holds']:
                hold_id = hold['id']
                if hold_id < len(masks):
                    mask = (masks[hold_id] * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_image, contours, -1, bgr_color, 3)
                    
                    # 중심에 번호 표시
                    center = tuple(map(int, hold['center']))
                    cv2.putText(annotated_image, str(hold_id), center, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
        
        # Base64 인코딩
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 통계
        total_holds = len(hold_data_raw)
        analyzable_problems = sum(1 for p in problems.values() if p['hold_count'] >= 3)
        h, w = image.shape[:2]
        
        return JSONResponse(
            status_code=200,
            content={
                "problems": list(problems.values()),
                "statistics": {
                    "total_holds": total_holds,
                    "total_problems": len(problems),
                    "analyzable_problems": analyzable_problems
                },
                "image_width": w,
                "image_height": h,
                "annotated_image_base64": annotated_base64,
                "message": f"{len(problems)}개의 문제를 발견했습니다."
            }
        )
        
    except Exception as e:
        print(f"❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 CLIP 색상 분석 API (서버에서 실행)
class ColorAnalysisRequest(BaseModel):
    holds: list
    image_data_base64: str

@app.post("/api/analyze-colors")
async def analyze_colors_with_clip(request: ColorAnalysisRequest):
    """
    🎨 CLIP 모델로 홀드 색상 분석 (서버에서 실행)
    브라우저: YOLO로 홀드 감지 → 서버: CLIP으로 색상 분석
    """
    try:
        from holdcheck.preprocess import get_clip_model, extract_color_with_clip_ai
        
        # 이미지 디코딩 및 검증
        try:
            image_data = base64.b64decode(request.image_data_base64)
            if len(image_data) < 100:
                raise ValueError("Image data too small")
        except Exception as e:
            print(f"⚠️ Base64 디코딩 실패: {e}")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        from PIL import Image
        import io
        
        # 이미지 로드 및 검증
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            if pil_image.size[0] < 10 or pil_image.size[1] < 10:
                raise ValueError("Image too small")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {e}")
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        colored_holds = []
        
        for hold in request.holds:
            try:
                # 홀드 영역 추출
                x, y, w, h = int(hold['x']), int(hold['y']), int(hold['width']), int(hold['height'])
                
                # 경계 체크
                x = max(0, min(x, image.shape[1] - 1))
                y = max(0, min(y, image.shape[0] - 1))
                w = max(1, min(w, image.shape[1] - x))
                h = max(1, min(h, image.shape[0] - y))
                
                hold_image = image[y:y+h, x:x+w]
                
                if hold_image.size == 0:
                    colored_holds.append({**hold, 'color': 'unknown'})
                    continue
                
                # CLIP으로 색상 분석
                color = extract_color_with_clip_ai(hold_image, None)
                
                colored_holds.append({
                    **hold,
                    'color': color
                })
                
            except Exception as e:
                print(f"⚠️ 홀드 색상 분석 실패: {e}")
                colored_holds.append({
                    **hold,
                    'color': 'unknown'
                })
        
        return {
            "success": True,
            "colored_holds": colored_holds,
            "message": f"✅ CLIP으로 {len(colored_holds)}개 홀드 색상 분석 완료"
        }
        
    except Exception as e:
        print(f"❌ CLIP 색상 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CLIP color analysis failed: {str(e)}")

if DB_AVAILABLE:
    @app.post("/api/feedback")
    async def submit_feedback(feedback: FeedbackRequest):
        """사용자 피드백 저장"""
        try:
            save_user_feedback(
                problem_id=feedback.problem_id,
                user_difficulty=feedback.user_difficulty,
                user_type=feedback.user_type,
                user_feedback=feedback.user_feedback
            )
            
            stats = get_model_stats()
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": "피드백 저장 완료! 감사합니다 🙏",
                    "stats": stats
                }
            )
        except Exception as e:
            print(f"❌ 피드백 저장 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/hold-color-feedback")
    async def submit_hold_color_feedback(feedback: HoldColorFeedbackRequest):
        """🎨 홀드 색상 피드백 저장 (ML 학습용)"""
        try:
            print(f"🎨 홀드 색상 피드백 수신:")
            print(f"  - Problem ID: {feedback.problem_id}")
            print(f"  - Hold ID: {feedback.hold_id}")
            print(f"  - Predicted: {feedback.predicted_color}")
            print(f"  - User Correct: {feedback.user_color}")
            print(f"  - Center: {feedback.hold_center}")
            print(f"  - Features: {len(feedback.hold_features) if feedback.hold_features else 0} keys")
            
            # 🔥 데이터베이스에 홀드 색상 피드백 저장 (ML 학습용)
            if DB_AVAILABLE and feedback.hold_features:
                from database import save_hold_color_feedback
                
                feedback_id = save_hold_color_feedback(
                    problem_id=feedback.problem_id,
                    hold_id=int(feedback.hold_id),
                    hold_center=feedback.hold_center or [0, 0],
                    hold_features=feedback.hold_features,
                    predicted_color=feedback.predicted_color,
                    user_correct_color=feedback.user_color
                )
                
                print(f"✅ 홀드 색상 피드백 ID {feedback_id} 저장 완료!")
                
                # 🤖 ML 재학습 트리거 (비동기)
                try:
                    from database import get_color_training_data
                    training_data = get_color_training_data()
                    
                    if len(training_data) >= 30:  # 30개 이상 피드백이 모이면
                        print(f"🤖 색상 학습 데이터 {len(training_data)}개 확보! ML 재학습 준비 완료")
                        # TODO: 실제 재학습은 별도 스케줄러에서 수행
                except Exception as e:
                    print(f"⚠️ ML 재학습 체크 실패: {e}")
            else:
                print("⚠️ DB 또는 hold_features 없음 - 로깅만 수행")
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": "홀드 색상 피드백이 저장되었습니다! ML 학습에 활용됩니다 🎨🤖",
                    "feedback": {
                        "predicted": feedback.predicted_color,
                        "corrected": feedback.user_color
                    }
                }
            )
        except Exception as e:
            print(f"❌ 홀드 색상 피드백 저장 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/stats")
    async def get_stats():
        """모델 성능 통계 조회"""
        try:
            stats = get_model_stats()
            gpt4_status = get_gpt4_status() if GPT4_AVAILABLE else {'available': False}
            method_stats = get_analysis_method_stats() if HYBRID_AVAILABLE else {}
            
            return JSONResponse(
                status_code=200,
                content={
                    "stats": stats,
                    "gpt4_status": gpt4_status,
                    "method_stats": method_stats
                }
            )
        except Exception as e:
            print(f"❌ 통계 조회 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/color-feedbacks")
    async def get_color_feedbacks():
        """🎨 모든 홀드 색상 피드백 조회"""
        try:
            if not DB_AVAILABLE:
                return JSONResponse(
                    status_code=200,
                    content={"feedbacks": [], "count": 0}
                )
            
            from database import get_all_color_feedbacks
            feedbacks = get_all_color_feedbacks()
            
            return JSONResponse(
                status_code=200,
                content={
                    "feedbacks": feedbacks,
                    "count": len(feedbacks)
                }
            )
        except Exception as e:
            print(f"❌ 피드백 조회 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/color-feedbacks/{feedback_id}/confirm")
    async def confirm_feedback(feedback_id: int):
        """🎨 홀드 색상 피드백 확인 (ML 학습 데이터로 확정)"""
        try:
            if not DB_AVAILABLE:
                raise HTTPException(status_code=503, detail="Database not available")
            
            from database import confirm_color_feedback
            confirm_color_feedback(feedback_id)
            
            return JSONResponse(
                status_code=200,
                content={"message": f"피드백 ID {feedback_id} 확인 완료"}
            )
        except Exception as e:
            print(f"❌ 피드백 확인 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/color-feedbacks/confirm-all")
    async def confirm_all_feedbacks():
        """🎨 모든 미확인 피드백 일괄 확인 (ML 학습 데이터로 확정)"""
        try:
            if not DB_AVAILABLE:
                raise HTTPException(status_code=503, detail="Database not available")
            
            from database import confirm_all_unconfirmed_feedbacks
            count = confirm_all_unconfirmed_feedbacks()
            
            return JSONResponse(
                status_code=200,
                content={"message": f"{count}개 피드백 일괄 확인 완료", "count": count}
            )
        except Exception as e:
            print(f"❌ 일괄 확인 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/color-feedbacks/{feedback_id}")
    async def delete_feedback(feedback_id: int):
        """🎨 홀드 색상 피드백 삭제"""
        try:
            if not DB_AVAILABLE:
                raise HTTPException(status_code=503, detail="Database not available")
            
            from database import delete_color_feedback
            delete_color_feedback(feedback_id)
            
            return JSONResponse(
                status_code=200,
                content={"message": f"피드백 ID {feedback_id} 삭제 완료"}
            )
        except Exception as e:
            print(f"❌ 피드백 삭제 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if ML_AVAILABLE and DB_AVAILABLE:
    @app.get("/api/ml-model-stats")
    async def get_ml_model_stats():
        """🤖 ML 모델 학습 상태 및 성능 조회"""
        try:
            import os
            import pickle
            from collections import Counter
            from database import get_color_training_data
            
            # 모델 파일 존재 여부
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'color_model.pkl')
            encoder_path = os.path.join(os.path.dirname(__file__), 'models', 'color_encoder.pkl')
            
            model_exists = os.path.exists(model_path)
            
            # 학습 데이터 통계
            training_data = get_color_training_data(min_samples=1, confirmed_only=False)
            total_samples = len(training_data)
            
            # 색상별 분포
            color_distribution = Counter([d['correct_color'] for d in training_data])
            
            # predicted vs correct 오분류 통계
            misclassifications = {}
            correct_predictions = 0
            
            for data in training_data:
                predicted = data.get('predicted_color', 'unknown')
                correct = data.get('correct_color', 'unknown')
                
                if predicted == correct:
                    correct_predictions += 1
                elif predicted != 'unknown':  # unknown은 제외
                    key = f"{predicted}→{correct}"
                    misclassifications[key] = misclassifications.get(key, 0) + 1
            
            # 규칙 기반 정확도 (unknown 제외)
            valid_predictions = sum(1 for d in training_data if d.get('predicted_color') != 'unknown')
            rule_based_accuracy = (correct_predictions / valid_predictions * 100) if valid_predictions > 0 else 0
            
            # ML 모델 성능 (파일에서 읽기)
            ml_accuracy = None
            ml_cv_accuracy = None
            
            if model_exists:
                try:
                    # 모델 메타데이터 읽기 (저장되어 있다면)
                    meta_path = os.path.join(os.path.dirname(__file__), 'models', 'color_model_meta.pkl')
                    if os.path.exists(meta_path):
                        with open(meta_path, 'rb') as f:
                            meta = pickle.load(f)
                            ml_accuracy = meta.get('test_accuracy', None)
                            ml_cv_accuracy = meta.get('cv_accuracy', None)
                except:
                    pass
            
            return JSONResponse(
                status_code=200,
                content={
                    "model_exists": model_exists,
                    "model_trained": model_exists,
                    "total_samples": total_samples,
                    "valid_samples": valid_predictions,
                    "rule_based_accuracy": round(rule_based_accuracy, 1),
                    "ml_test_accuracy": round(ml_accuracy * 100, 1) if ml_accuracy else None,
                    "ml_cv_accuracy": round(ml_cv_accuracy * 100, 1) if ml_cv_accuracy else None,
                    "color_distribution": dict(color_distribution),
                    "top_misclassifications": dict(sorted(misclassifications.items(), key=lambda x: -x[1])[:10]),
                    "can_train": total_samples >= 30,
                    "samples_needed": max(0, 30 - total_samples)
                }
            )
        except Exception as e:
            print(f"❌ ML 통계 조회 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/train-color-model")
    async def train_color_model_endpoint():
        """🎨 규칙 기반 색상 범위 자동 조정 (피드백 기반)"""
        try:
            import sqlite3
            import shutil
            from datetime import datetime
            
            # 피드백 데이터 로드
            db_path = os.path.join(os.path.dirname(__file__), 'climbmate.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT predicted_color, user_correct_color, hold_hsv
                FROM color_feedback
                WHERE user_correct_color IS NOT NULL AND user_correct_color != ''
            """)
            
            feedback_data = cursor.fetchall()
            conn.close()
            
            print(f"📊 피드백 데이터 로드: {len(feedback_data)}개")
            
            if len(feedback_data) < 30:
                raise HTTPException(
                    status_code=400,
                    detail=f"피드백 데이터 부족: {len(feedback_data)}개 (최소 30개 필요)"
                )
            
            # HSV 데이터 파싱 및 색상별 분류
            color_hsv_data = {}
            for predicted, correct, hsv_str in feedback_data:
                if not hsv_str:
                    continue
                
                try:
                    # "H,S,V" 형식 파싱
                    h, s, v = map(int, hsv_str.split(','))
                    
                    if correct not in color_hsv_data:
                        color_hsv_data[correct] = []
                    color_hsv_data[correct].append((h, s, v))
                except:
                    continue
            
            # color_ranges.json 로드
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'holdcheck', 'color_ranges.json')
            backup_path = config_path + '.backup'
            
            # 백업 생성
            shutil.copy(config_path, backup_path)
            print(f"💾 백업 생성: {backup_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                ranges_data = json.load(f)
            
            # 색상별 HSV 범위 자동 조정
            updated_colors = []
            for color_name, hsv_list in color_hsv_data.items():
                if color_name not in ranges_data['colors']:
                    continue
                
                if len(hsv_list) < 5:  # 최소 5개 데이터 필요
                    continue
                
                # HSV 범위 계산 (평균 ± 표준편차)
                h_values = [h for h, s, v in hsv_list]
                s_values = [s for h, s, v in hsv_list]
                v_values = [v for h, s, v in hsv_list]
                
                h_min = max(0, int(np.mean(h_values) - 1.5 * np.std(h_values)))
                h_max = min(180, int(np.mean(h_values) + 1.5 * np.std(h_values)))
                s_min = max(0, int(np.mean(s_values) - 1.5 * np.std(s_values)))
                s_max = min(255, int(np.mean(s_values) + 1.5 * np.std(s_values)))
                v_min = max(0, int(np.mean(v_values) - 1.5 * np.std(v_values)))
                v_max = min(255, int(np.mean(v_values) + 1.5 * np.std(v_values)))
                
                # 범위가 너무 좁아지지 않도록 최소 폭 보장
                if h_max - h_min < 10:
                    h_min = max(0, int(np.mean(h_values)) - 5)
                    h_max = min(180, int(np.mean(h_values)) + 5)
                if s_max - s_min < 30:
                    s_min = max(0, int(np.mean(s_values)) - 15)
                    s_max = min(255, int(np.mean(s_values)) + 15)
                if v_max - v_min < 40:
                    v_min = max(0, int(np.mean(v_values)) - 20)
                    v_max = min(255, int(np.mean(v_values)) + 20)
                
                # color_ranges.json 업데이트
                if 'hsv_ranges' in ranges_data['colors'][color_name]:
                    ranges_data['colors'][color_name]['hsv_ranges'][0]['h'] = [h_min, h_max]
                    ranges_data['colors'][color_name]['hsv_ranges'][0]['s'] = [s_min, s_max]
                    ranges_data['colors'][color_name]['hsv_ranges'][0]['v'] = [v_min, v_max]
                    updated_colors.append(f"{color_name}: H[{h_min},{h_max}], S[{s_min},{s_max}], V[{v_min},{v_max}]")
                    print(f"🔄 {color_name}: H[{h_min},{h_max}], S[{s_min},{s_max}], V[{v_min},{v_max}] ({len(hsv_list)}개 샘플)")
            
            # 업데이트된 설정 저장
            ranges_data['last_updated'] = datetime.now().isoformat()
            ranges_data['feedback_count'] = len(feedback_data)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(ranges_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ color_ranges.json 업데이트 완료: {len(updated_colors)}개 색상")
            
            # 🔄 색상 범위 캐시 초기화 (즉시 적용)
            try:
                from clustering import reload_color_ranges
                reload_color_ranges()
                print("✅ 색상 범위 캐시 초기화 완료 - 즉시 적용됩니다")
            except Exception as e:
                print(f"⚠️ 캐시 초기화 경고: {e}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"규칙 기반 색상 범위 자동 조정 완료 - 즉시 적용됩니다 ({len(updated_colors)}개 색상 업데이트)",
                    "updated_colors": updated_colors,
                    "feedback_samples": len(feedback_data),
                    "backup_path": backup_path
                }
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ 규칙 조정 오류: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

if ML_AVAILABLE and DB_AVAILABLE:
    @app.post("/api/train")
    async def train_models():
        """자체 ML 모델 학습"""
        try:
            stats = get_model_stats()
            
            if not stats['ready_for_training']:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": f"최소 50개의 검증된 데이터 필요 (현재: {stats['verified_problems']}개)"
                    }
                )
            
            # 훈련 데이터 로드
            training_data = get_training_data()
            
            # 난이도 모델 학습
            diff_test_acc, diff_cv_acc = train_difficulty_model(training_data)
            
            # 유형 모델 학습
            type_test_acc, type_cv_acc = train_type_model(training_data)
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "모델 학습 완료! 🎉",
                    "results": {
                        "difficulty_model": {
                            "test_accuracy": round(diff_test_acc * 100, 1),
                            "cv_accuracy": round(diff_cv_acc * 100, 1)
                        },
                        "type_model": {
                            "test_accuracy": round(type_test_acc * 100, 1),
                            "cv_accuracy": round(type_cv_acc * 100, 1)
                        },
                        "training_samples": len(training_data)
                    }
                }
            )
        except Exception as e:
            print(f"❌ 모델 학습 오류: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/convert-gpt4")
async def convert_gpt4_to_training():
    """GPT-4 분석 결과를 훈련 데이터로 변환"""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="데이터베이스를 사용할 수 없습니다")
    
    try:
        converted_count = convert_gpt4_to_training_data()
        return {
            "message": f"GPT-4 결과 {converted_count}건을 훈련 데이터로 변환했습니다",
            "converted_count": converted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"변환 실패: {str(e)}")

async def send_progress_update(message: str, progress: int, step: str = None, **kwargs):
    """SSE 진행률 업데이트 전송"""
    data = {
        "message": message,
        "progress": progress,
        "step": step,
        **kwargs
    }
    # JSON 인코딩 시 한글과 특수문자 처리
    json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    # SSE 형식 강화: 각 줄마다 명확한 구분자
    sse_message = f"data: {json_str}\n\n"
    print(f"📡 SSE 전송: {message} ({progress}%) - {len(json_str)} bytes")
    return sse_message

@app.post("/api/analyze-stream")
async def analyze_image_stream(
    file: UploadFile = File(...),
    wall_angle: str = None
):
    """
    비동기 이미지 분석 시작 (작업 큐에 추가)
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # 비동기 작업 큐에 추가
        from backend.ai_tasks import analyze_image_async
        task = analyze_image_async.delay(image_base64, wall_angle)
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "🚀 분석 작업이 시작되었습니다"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"작업 시작 실패: {str(e)}")

@app.get("/api/analyze-status/{task_id}")
async def get_analysis_status(task_id: str):
    """
    분석 작업 상태 확인
    """
    try:
        from backend.ai_tasks import analyze_image_async
        task = analyze_image_async.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'status': task.state,
                'progress': 0,
                'message': '작업 대기 중...'
            }
        elif task.state == 'PROGRESS':
            response = {
                'status': task.state,
                'progress': task.info.get('progress', 0),
                'message': task.info.get('message', ''),
                'step': task.info.get('step', ''),
                **task.info
            }
        elif task.state == 'SUCCESS':
            # 성공 상태이지만 결과가 에러인 경우 처리
            result = task.result
            if isinstance(result, dict) and result.get('status') == 'error':
                response = {
                    'status': 'FAILURE',
                    'progress': 0,
                    'message': result.get('message', '분석 실패'),
                    'error': result.get('error', ''),
                    'error_type': result.get('error_type', 'UNKNOWN')
                }
            else:
                response = {
                    'status': task.state,
                    'progress': 100,
                    'message': '✅ 분석 완료!',
                    'result': result
                }
        else:  # FAILURE
            # task.info가 dict가 아닐 수 있으므로 안전하게 처리
            info = task.info if isinstance(task.info, dict) else {}
            response = {
                'status': 'FAILURE',
                'progress': 0,
                'message': info.get('message', '분석 실패'),
                'error': info.get('error', str(task.info) if task.info else '알 수 없는 오류'),
                'error_type': info.get('error_type', 'UNKNOWN')
            }
        
        return response
        
    except Exception as e:
        import traceback
        print(f"❌ 상태 확인 오류: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"상태 확인 실패: {str(e)}")

@app.post("/api/gpt4-analyze")
async def gpt4_analyze(request: dict):
    """
    GPT-4 문제 분석 API
    """
    try:
        image_base64 = request.get('image_base64')
        holds = request.get('holds')
        wall_angle = request.get('wall_angle')
        
        if not image_base64 or not holds:
            raise HTTPException(status_code=400, detail="이미지와 홀드 데이터가 필요합니다")
        
        # GPT-4 분석 실행
        analysis = analyze_with_gpt4_vision(image_base64, holds, wall_angle)
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-4 분석 실패: {str(e)}")

@app.post("/api/analyze-sync")
async def analyze_image_sync(
    file: UploadFile = File(...),
    wall_angle: str = None
):
    """
    클라이밍 벽 이미지 분석 (실시간 진행률 전송)
    
    Parameters:
    - file: 이미지 파일
    - wall_angle: 벽 각도 (overhang, slab, face, null)
    
    Returns:
    - SSE 스트림으로 실시간 진행률 및 결과 전송
    """
    async def generate():
        try:
            # 1단계: 이미지 업로드
            yield await send_progress_update("📸 이미지 업로드 중...", 5, "upload")
            
            # 이미지 읽기
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                yield await send_progress_update("❌ 잘못된 이미지 파일", 0, "error")
                return
            
            # 2단계: 홀드 감지 시작
            yield await send_progress_update("🔍 홀드 감지 중...", 10, "detection")
            
            # 🚀 최적화: 전처리 (홀드 감지)
            # 배포 환경에 따른 모델 경로 설정
            if os.path.exists("/app/holdcheck/roboflow_weights/weights.pt"):
                model_path = "/app/holdcheck/roboflow_weights/weights.pt"  # Docker 환경
            else:
                model_path = "/Users/kimjazz/Desktop/project/climbmate/holdcheck/roboflow_weights/weights.pt"  # 로컬 환경
            
            hold_data_raw, masks = preprocess(
                image,
                model_path=model_path,
                mask_refinement=1,  # 속도 우선
                conf=0.4,  # 확실한 홀드만
                use_clip_ai=True
            )
            
            if not hold_data_raw:
                yield await send_progress_update("❌ 홀드를 감지하지 못했습니다", 0, "error")
                return
            
            # 홀드 감지 완료
            yield await send_progress_update(f"✅ {len(hold_data_raw)}개 홀드 감지 완료", 30, "detection_complete", holds_count=len(hold_data_raw))
            
            # 3단계: 색상 그룹핑
            yield await send_progress_update("🎨 색상 분류 중...", 40, "clustering")
            
            hold_data = clip_ai_color_clustering(
                hold_data_raw,
                None,
                image,
                masks,
                eps=0.3,
                use_dbscan=False
            )
            
            # 그룹별 정리
            problems = {}
            print(f"🔍 홀드 데이터 분석: {len(hold_data)}개 홀드")
            
            for i, hold in enumerate(hold_data):
                if i < 5:  # 처음 5개만 로그
                    print(f"  홀드 {i}: {type(hold)} - group: {hold.get('group')}")
                
                group = hold.get('group')
                if group is None:
                    continue
                
                if group not in problems:
                    clip_color = hold.get('clip_color_name', 'unknown')
                    rgb = hold.get('dominant_rgb', [128, 128, 128])
                    
                    problems[group] = {
                        'id': group,
                        'color_name': clip_color,
                        'color_rgb': rgb,
                        'holds': [],
                        'hold_count': 0,
                        'analysis': None
                    }
                
                problems[group]['holds'].append({
                    'id': hold['id'],
                    'center': hold['center'],
                    'area': hold['area'],
                    'rgb': hold.get('dominant_rgb', [128, 128, 128])
                })
            
            print(f"🔍 생성된 문제 그룹: {len(problems)}개")
            for group_id, problem in problems.items():
                print(f"  그룹 {group_id}: {len(problem['holds'])}개 홀드")
            
            # 색상 분류 완료
            yield await send_progress_update(f"✅ {len(problems)}개 문제 분류 완료", 60, "clustering_complete", problems_count=len(problems))
            
            # 이미지를 Base64로 인코딩
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 4단계: 문제 분석 (🚀 비동기 병렬 처리)
            yield await send_progress_update("🤖 AI 문제 분석 중... (병렬 처리)", 70, "analysis")
            
            # 홀드 수 업데이트
            for group_id, problem in problems.items():
                problem['hold_count'] = len(problem['holds'])
            
            # 3개 이상인 문제만 분석 (병렬 처리)
            analyzable_problems = [(group_id, problem) for group_id, problem in problems.items() if problem['hold_count'] >= 3]
            
            if analyzable_problems:
                print(f"🚀 {len(analyzable_problems)}개 문제 병렬 분석 시작...")
                
                # 🚀 모든 문제를 동시에 분석
                async def analyze_all_problems_parallel():
                    tasks = []
                    
                    for group_id, problem in analyzable_problems:
                        # 기본 통계 기반 분석
                        rule_analysis = analyze_problem(
                            hold_data,
                            group_id,
                            wall_angle if wall_angle != "null" else None
                        )
                        
                        if rule_analysis is None:
                            rule_analysis = {
                                'difficulty': {'grade': 'V?', 'confidence': 0.0},
                                'climb_type': {'primary_type': '분석 불가', 'confidence': 0.0},
                                'statistics': {}
                            }
                        
                        # 하이브리드 분석 태스크 생성
                        if HYBRID_AVAILABLE:
                            task = hybrid_analyze(
                                image_base64=image_base64,
                                holds_data=problem['holds'],
                                wall_angle=wall_angle if wall_angle != "null" else None,
                                rule_based_analysis=rule_analysis
                            )
                            tasks.append((group_id, problem, task, rule_analysis))
                        else:
                            problem['analysis'] = rule_analysis
                    
                    # 모든 하이브리드 분석을 동시에 실행
                    if tasks:
                        hybrid_tasks = [task for _, _, task, _ in tasks]
                        results = await asyncio.gather(*hybrid_tasks, return_exceptions=True)
                        
                        # 결과 적용
                        for (group_id, problem, _, rule_analysis), result in zip(tasks, results):
                            if isinstance(result, Exception):
                                print(f"⚠️ 문제 {group_id} 하이브리드 분석 실패: {result}")
                                problem['analysis'] = rule_analysis
                            else:
                                # 하이브리드 결과를 기존 분석 구조에 통합
                                rule_analysis['difficulty']['grade'] = result['difficulty']['grade']
                                rule_analysis['difficulty']['confidence'] = result['difficulty']['confidence']
                                rule_analysis['climb_type']['primary_type'] = result['type']['primary_type']
                                rule_analysis['climb_type']['confidence'] = result['type']['confidence']
                                rule_analysis['analysis_method'] = result['method_used']
                                
                                if 'gpt4_reasoning' in result:
                                    rule_analysis['gpt4_reasoning'] = result['gpt4_reasoning']
                                
                                problem['analysis'] = rule_analysis
                                problem['gpt4_reasoning'] = result.get('gpt4_reasoning', '')
                                problem['gpt4_confidence'] = result.get('gpt4_confidence', 0.8)
                                problem['gpt4_secondary_types'] = result.get('gpt4_secondary_types', [])
                                problem['gpt4_key_factors'] = result.get('gpt4_key_factors', [])
                                problem['gpt4_crux'] = result.get('gpt4_crux', '')
                                problem['gpt4_movements'] = result.get('gpt4_movements', [])
                                problem['gpt4_challenges'] = result.get('gpt4_challenges', [])
                                problem['gpt4_tips'] = result.get('gpt4_tips', [])
                                problem['gpt4_comparison'] = result.get('gpt4_comparison', '')
                
                # 병렬 분석 실행
                try:
                    await analyze_all_problems_parallel()
                    print(f"✅ {len(analyzable_problems)}개 문제 병렬 분석 완료!")
                except Exception as e:
                    print(f"⚠️ 병렬 분석 오류: {e}")
                    # 실패 시 순차 처리로 폴백
                    for group_id, problem in analyzable_problems:
                        rule_analysis = analyze_problem(hold_data, group_id, wall_angle if wall_angle != "null" else None)
                        problem['analysis'] = rule_analysis if rule_analysis else {
                            'difficulty': {'grade': 'V?', 'confidence': 0.0},
                            'climb_type': {'primary_type': '분석 불가', 'confidence': 0.0},
                            'statistics': {}
                        }
            
            # 분석 완료
            yield await send_progress_update("✅ AI 분석 완료", 90, "analysis_complete")
            
            # 5단계: 결과 정리
            yield await send_progress_update("📊 결과 정리 중...", 95, "finalizing")
            
            # 문제 목록을 배열로 변환 (None 값 필터링)
            problems_list = [p for p in problems.values() if p is not None]
            
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
                    # 원본 이미지에 홀드 마스크 오버레이
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
                    
                    # 오버레이를 원본에 블렌딩
                    annotated = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                    
                    # Base64로 인코딩
                    _, buffer = cv2.imencode('.jpg', annotated)
                    annotated_image = base64.b64encode(buffer).decode('utf-8')
                except Exception as e:
                    print(f"⚠️ 주석 이미지 생성 실패: {e}")
            
            # 최종 결과 전송
            result = {
                "problems": problems_list,
                "statistics": statistics,
                "hold_data": hold_data,
                "annotated_image_base64": annotated_image
            }
            
            # 결과를 단계별로 전송 (큰 데이터는 청크로 분할)
            print(f"📊 통계 데이터 전송: {statistics}")
            yield await send_progress_update("📊 통계 데이터 전송", 96, "result_stats", statistics=statistics)
            
            # 홀드 데이터에서 프론트엔드에 필요한 데이터만 추출
            def clean_hold_data(holds):
                """프론트엔드 전송용 홀드 데이터 정리 - 필요한 필드만 추출"""
                cleaned = []
                for hold in holds:
                    cleaned_hold = {
                        'id': hold['id'],
                        'center': hold['center'],
                        'area': hold['area'],
                        'rgb': hold.get('dominant_rgb', [128, 128, 128]),
                        'color': hold.get('clip_color_name', 'unknown')
                    }
                    cleaned.append(cleaned_hold)
                return cleaned
            
            hold_data_clean = clean_hold_data(hold_data)
            
            # 홀드 데이터 전송 (첫 커밋 때처럼 제한 없이)
            print(f"🔍 홀드 데이터 전송 시작: {len(hold_data_clean)}개")
            yield await send_progress_update(f"🔍 홀드 데이터 전송 완료", 96, "result_holds", hold_data=hold_data_clean)
            
            # 문제 데이터에서 프론트엔드에 필요한 데이터만 추출
            def clean_problem_data(problems):
                """프론트엔드 전송용 문제 데이터 정리 - 필요한 필드만 추출"""
                cleaned = []
                for problem in problems:
                    # None 체크 추가
                    if problem is None:
                        print("⚠️ None 문제 발견, 건너뜀")
                        continue
                    
                    # 필수 필드 체크
                    if not isinstance(problem, dict):
                        print(f"⚠️ 잘못된 문제 데이터 타입: {type(problem)}")
                        continue
                    
                    analysis = problem.get('analysis', {})
                    difficulty = analysis.get('difficulty', {}) if analysis else {}
                    climb_type = analysis.get('climb_type', {}) if analysis else {}
                    
                    cleaned_problem = {
                        'id': problem.get('id', 'unknown'),
                        'color_name': problem.get('color_name', 'unknown'),
                        'color_rgb': problem.get('color_rgb', [128, 128, 128]),
                        'holds': problem.get('holds', []),
                        'hold_count': problem.get('hold_count', 0),
                        'difficulty': {
                            'grade': difficulty.get('grade', 'V?') if difficulty else 'V?',
                            'level': difficulty.get('level', '미분석') if difficulty else '미분석',
                            'confidence': difficulty.get('confidence', 0.0) if difficulty else 0.0,
                            'factors': difficulty.get('factors', {}) if difficulty else {}
                        },
                        'climb_type': {
                            'primary_type': climb_type.get('primary_type', '일반') if climb_type else '일반',
                            'types': climb_type.get('types', []) if climb_type else [],
                            'confidence': climb_type.get('confidence', 0.0) if climb_type else 0.0
                        },
                        'gpt4_reasoning': problem.get('gpt4_reasoning', ''),
                        'gpt4_confidence': problem.get('gpt4_confidence', 0.0),
                        'gpt4_secondary_types': problem.get('gpt4_secondary_types', []),
                        'gpt4_key_factors': problem.get('gpt4_key_factors', []),
                        'gpt4_crux': problem.get('gpt4_crux', ''),
                        'gpt4_movements': problem.get('gpt4_movements', []),
                        'gpt4_challenges': problem.get('gpt4_challenges', []),
                        'gpt4_tips': problem.get('gpt4_tips', []),
                        'gpt4_comparison': problem.get('gpt4_comparison', '')
                    }
                    cleaned.append(cleaned_problem)
                return cleaned
            
            print(f"🔍 원본 문제 목록: {len(problems_list)}개")
            for i, p in enumerate(problems_list):
                print(f"  문제 {i+1}: {type(p)} - {p is not None}")
                if p is not None and isinstance(p, dict):
                    print(f"    - id: {p.get('id')}, color: {p.get('color_name')}, holds: {len(p.get('holds', []))}")
                    print(f"    - analysis: {type(p.get('analysis'))} - {p.get('analysis') is not None}")
                else:
                    print(f"    - ⚠️ 문제 데이터가 None이거나 dict가 아님!")
            
            problems_clean = clean_problem_data(problems_list)
            
            # 문제 데이터 전송
            print(f"🎯 정리된 문제 데이터 전송: {len(problems_clean)}개")
            for i, problem in enumerate(problems_clean):
                if problem and isinstance(problem, dict):
                    print(f"  문제 {i+1} ({problem.get('color_name', 'unknown')}): difficulty={problem.get('difficulty', {}).get('grade', 'V?')}, type={problem.get('climb_type', {}).get('primary_type', '일반')}")
            yield await send_progress_update("🎯 문제 데이터 전송", 98, "result_problems", problems=problems_clean)
            
            # 이미지 데이터를 작은 청크로 분할하여 전송
            if annotated_image:
                print(f"🖼️ 주석 이미지 전송 시작: {len(annotated_image)}bytes")
                chunk_size = 50000  # 50KB씩 전송
                for i in range(0, len(annotated_image), chunk_size):
                    chunk = annotated_image[i:i+chunk_size]
                    chunk_num = i // chunk_size + 1
                    total_chunks = (len(annotated_image) + chunk_size - 1) // chunk_size
                    print(f"🖼️ 이미지 청크 {chunk_num}/{total_chunks} 전송: {len(chunk)}bytes")
                    yield await send_progress_update(f"🖼️ 이미지 전송 ({chunk_num}/{total_chunks})", 99 + (chunk_num * 0.1), "result_image_chunk", image_chunk=chunk, chunk_info={"current": chunk_num, "total": total_chunks})
            
            # 완료 - 최종 결과 포함
            yield await send_progress_update("✅ 분석 완료!", 100, "complete", 
                                           problems=problems_clean, 
                                           statistics=statistics, 
                                           annotated_image_base64=annotated_image)
            
        except Exception as e:
            print(f"❌ 분석 오류: {e}")
            yield await send_progress_update(f"❌ 분석 실패: {str(e)}", 0, "error")
    
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Cache-Control, Accept",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Transfer-Encoding": "chunked"
    }
    print("📡 SSE 응답 헤더 설정:", headers)
    # SSE 스트림 플러시 강화
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream", 
        headers=headers,
        # 스트림 즉시 전송을 위한 설정
        background=None
    )

@app.get("/api/health")
async def health_check():
    """상태 확인"""
    return {
        "status": "healthy",
        "models": {
            "yolo": "loaded",
            "clip": "loaded"
        }
    }

@app.get("/api/gpt4-status")
async def gpt4_status_check():
    """GPT-4 상태 확인 (디버깅용)"""
    try:
        if not GPT4_AVAILABLE:
            return {
                "available": False,
                "reason": "GPT4_AVAILABLE = False",
                "api_key_set": bool(os.getenv('OPENAI_API_KEY')),
                "details": "GPT-4 모듈을 로드할 수 없습니다"
            }
        
        if not os.getenv('OPENAI_API_KEY'):
            return {
                "available": False,
                "reason": "API 키 없음",
                "api_key_set": False,
                "details": "OPENAI_API_KEY 환경변수가 설정되지 않았습니다"
            }
        
        # GPT-4 상태 확인
        if HYBRID_AVAILABLE:
            from hybrid_analyzer import get_analysis_method_stats
            stats = get_analysis_method_stats()
            return {
                "available": stats.get('gpt4_available', False),
                "reason": "정상",
                "api_key_set": True,
                "details": f"GPT-4 사용 가능: {stats.get('gpt4_available', False)}",
                "recommended_method": stats.get('recommended_method', 'unknown'),
                "hybrid_available": HYBRID_AVAILABLE
            }
        else:
            return {
                "available": False,
                "reason": "하이브리드 분석기 없음",
                "api_key_set": bool(os.getenv('OPENAI_API_KEY')),
                "details": "HYBRID_AVAILABLE = False",
                "hybrid_available": False
            }
            
    except Exception as e:
        return {
            "available": False,
            "reason": f"오류: {str(e)}",
            "api_key_set": bool(os.getenv('OPENAI_API_KEY')),
            "details": f"상태 확인 중 오류 발생: {str(e)}"
        }

@app.post("/api/test-gpt4")
async def test_gpt4():
    """GPT-4 간단 테스트 (디버깅용)"""
    try:
        if not GPT4_AVAILABLE:
            return {
                "success": False,
                "message": "GPT-4 모듈을 사용할 수 없습니다",
                "details": "GPT4_AVAILABLE = False"
            }
        
        if not os.getenv('OPENAI_API_KEY'):
            return {
                "success": False,
                "message": "API 키가 설정되지 않았습니다",
                "details": "OPENAI_API_KEY 환경변수 필요"
            }
        
        # 간단한 테스트 이미지 생성 (1x1 픽셀)
        import numpy as np
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # 회색 이미지
        _, buffer = cv2.imencode('.jpg', test_image)
        test_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 간단한 홀드 데이터
        test_holds = [
            {
                'id': 0,
                'center': [50, 50],
                'area': 1000,
                'color_name': 'blue'
            }
        ]
        
        # GPT-4 테스트 호출
        from backend.gpt4_analyzer import analyze_with_gpt4_vision
        result = analyze_with_gpt4_vision(test_image_base64, test_holds, "face")
        
        return {
            "success": True,
            "message": "GPT-4 테스트 성공",
            "result": result,
            "details": f"난이도: {result.get('difficulty')}, 유형: {result.get('type')}, 신뢰도: {result.get('confidence')}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"GPT-4 테스트 실패: {str(e)}",
            "details": str(e)
        }

# ============================================================================
# 🎨 색상 피드백 API (룰 기반 학습용)
# ============================================================================

class ColorFeedbackRequest(BaseModel):
    feedbacks: list  # [{"hold_id": 0, "predicted_color": "yellow", "correct_color": "orange"}]

@app.post("/api/color-feedback")
async def submit_color_feedback(request: ColorFeedbackRequest):
    """
    사용자 색상 피드백 저장 및 학습
    
    프론트엔드에서 사용자가 수정한 색상을 받아서:
    1. 피드백 데이터 저장
    2. 색상 범위 자동 조정
    3. ML 모델 학습 데이터 축적
    """
    try:
        from clustering import save_user_feedback
        
        feedbacks = request.feedbacks
        
        if not feedbacks or len(feedbacks) == 0:
            raise HTTPException(status_code=400, detail="피드백이 비어있습니다")
        
        print(f"\n📝 색상 피드백 수신: {len(feedbacks)}개")
        
        # 피드백 저장 및 학습
        # Note: hold_data는 실제로는 필요 없지만 호환성을 위해 빈 리스트 전달
        save_user_feedback([], feedbacks)
        
        return {
            "status": "success",
            "message": f"{len(feedbacks)}개의 피드백이 저장되었습니다",
            "feedback_count": len(feedbacks),
            "next_steps": "다음 분석부터 개선된 색상 분류가 적용됩니다"
        }
    
    except Exception as e:
        print(f"❌ 피드백 저장 오류: {e}")
        raise HTTPException(status_code=500, detail=f"피드백 저장 실패: {str(e)}")


@app.get("/api/color-ranges")
async def get_color_ranges():
    """
    현재 색상 범위 설정 조회
    """
    try:
        from clustering import load_color_ranges
        
        ranges_data = load_color_ranges()
        
        return {
            "status": "success",
            "ranges": ranges_data,
            "feedback_count": ranges_data.get("feedback_count", 0)
        }
    
    except Exception as e:
        print(f"❌ 색상 범위 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"색상 범위 조회 실패: {str(e)}")


@app.get("/api/feedback-stats")
async def get_feedback_stats():
    """
    피드백 통계 조회
    """
    try:
        from clustering import load_color_ranges
        
        ranges_data = load_color_ranges()
        feedback_count = ranges_data.get("feedback_count", 0)
        last_updated = ranges_data.get("last_updated", "없음")
        
        # 색상별 범위 개수
        colors = ranges_data.get("colors", {})
        color_stats = {}
        for color_name, config in colors.items():
            hsv_ranges = config.get("hsv_ranges", [])
            color_stats[color_name] = {
                "name": config.get("name", color_name),
                "range_count": len(hsv_ranges),
                "priority": config.get("priority", 999)
            }
        
        return {
            "status": "success",
            "total_feedbacks": feedback_count,
            "last_updated": last_updated,
            "color_stats": color_stats
        }
    
    except Exception as e:
        print(f"❌ 피드백 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # 동시 요청 처리를 위한 워커 설정
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=2,  # 2개 워커로 동시 요청 처리
        loop="asyncio",  # 비동기 루프 최적화
        access_log=True,  # 접근 로그 활성화
        log_level="info"
    )

