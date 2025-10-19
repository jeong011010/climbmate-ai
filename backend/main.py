from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import cv2
import numpy as np
import sys
import os
import base64

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
    from gpt4_analyzer import analyze_with_gpt4_vision, get_gpt4_status
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
        
        # 🚀 이미지 크기 최적화 (속도 향상)
        height, width = image.shape[:2]
        if width > 1200:  # 너무 큰 이미지는 리사이즈
            scale = 1200 / width
            new_width = 1200
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"📐 이미지 리사이즈: {width}x{height} → {new_width}x{new_height}")
        
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
            
            problems[group]['holds'].append({
                'id': hold['id'],
                'center': hold['center'],
                'area': hold['area'],
                'rgb': hold.get('dominant_rgb', [128, 128, 128])
            })
        
        # 이미지를 Base64로 인코딩 (GPT-4 및 DB 저장용)
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 홀드 수 업데이트 및 분석
        for group_id, problem in problems.items():
            problem['hold_count'] = len(problem['holds'])
            
            # 3개 이상인 문제만 분석
            if problem['hold_count'] >= 3:
                print(f"🤖 문제 {group_id} 분석 중...")
                
                # 기본 통계 기반 분석 (백업용)
                rule_analysis = analyze_problem(
                    hold_data,
                    group_id,
                    wall_angle if wall_angle != "null" else None
                )
                
                # 🚀 하이브리드 분석 (가능한 경우)
                print(f"   🔍 HYBRID_AVAILABLE: {HYBRID_AVAILABLE}")
                print(f"   🔍 GPT4_AVAILABLE: {GPT4_AVAILABLE}")
                print(f"   🔍 OPENAI_API_KEY 존재: {bool(os.getenv('OPENAI_API_KEY'))}")
                if HYBRID_AVAILABLE:
                    print(f"   🚀 하이브리드 분석 시작...")
                    hybrid_result = await hybrid_analyze(
                        image_base64=image_base64,
                        holds_data=problem['holds'],
                        wall_angle=wall_angle if wall_angle != "null" else None,
                        rule_based_analysis=rule_analysis
                    )
                    
                    # 하이브리드 결과를 기존 분석 구조에 통합
                    rule_analysis['difficulty']['grade'] = hybrid_result['difficulty']['grade']
                    rule_analysis['difficulty']['confidence'] = hybrid_result['difficulty']['confidence']
                    rule_analysis['climb_type']['primary_type'] = hybrid_result['type']['primary_type']
                    rule_analysis['climb_type']['confidence'] = hybrid_result['type']['confidence']
                    rule_analysis['analysis_method'] = hybrid_result['method_used']
                    
                    if 'gpt4_reasoning' in hybrid_result:
                        rule_analysis['gpt4_reasoning'] = hybrid_result['gpt4_reasoning']
                
                problem['analysis'] = rule_analysis
                
                # DB에 저장 (가능한 경우)
                if DB_AVAILABLE:
                    try:
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
        from ai_tasks import analyze_image_async
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
        from ai_tasks import analyze_image_async
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
            response = {
                'status': task.state,
                'progress': 100,
                'message': '✅ 분석 완료!',
                'result': task.result
            }
        else:  # FAILURE
            response = {
                'status': task.state,
                'progress': 0,
                'message': task.info.get('message', '분석 실패'),
                'error': task.info.get('error', '')
            }
        
        return response
        
    except Exception as e:
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
            
            # 4단계: 문제 분석
            yield await send_progress_update("🤖 AI 문제 분석 중...", 70, "analysis")
            
            # 홀드 수 업데이트 및 분석
            for group_id, problem in problems.items():
                problem['hold_count'] = len(problem['holds'])
                
                # 3개 이상인 문제만 분석
                if problem['hold_count'] >= 3:
                    print(f"🤖 문제 {group_id} 분석 중...")
                    
                    # 기본 통계 기반 분석 (백업용)
                    rule_analysis = analyze_problem(
                        hold_data,
                        group_id,
                        wall_angle if wall_angle != "null" else None
                    )
                    
                    # analyze_problem이 None을 반환할 수 있음
                    if rule_analysis is None:
                        print(f"   ⚠️ 규칙 기반 분석 실패 (홀드 부족): {group_id}")
                        rule_analysis = {
                            'difficulty': {'grade': 'V?', 'confidence': 0.0},
                            'climb_type': {'primary_type': '분석 불가', 'confidence': 0.0},
                            'statistics': {}
                        }
                    
                    # 하이브리드 분석 (GPT-4 + ML)
                    if HYBRID_AVAILABLE:
                        try:
                            print(f"   🔍 하이브리드 분석 시작 - GPT4_AVAILABLE: {GPT4_AVAILABLE}, API_KEY: {bool(os.getenv('OPENAI_API_KEY'))}")
                            hybrid_result = await hybrid_analyze(
                                image_base64=image_base64,
                                holds_data=problem['holds'],
                                wall_angle=wall_angle if wall_angle != "null" else None,
                                rule_based_analysis=rule_analysis
                            )
                            
                            print(f"   🔍 하이브리드 결과: {hybrid_result}")
                            
                            # 하이브리드 결과를 기존 분석 구조에 통합
                            rule_analysis['difficulty']['grade'] = hybrid_result['difficulty']['grade']
                            rule_analysis['difficulty']['confidence'] = hybrid_result['difficulty']['confidence']
                            rule_analysis['climb_type']['primary_type'] = hybrid_result['type']['primary_type']
                            rule_analysis['climb_type']['confidence'] = hybrid_result['type']['confidence']
                            rule_analysis['analysis_method'] = hybrid_result['method_used']
                            
                            if 'gpt4_reasoning' in hybrid_result:
                                rule_analysis['gpt4_reasoning'] = hybrid_result['gpt4_reasoning']
                            
                            problem['analysis'] = rule_analysis
                            problem['gpt4_reasoning'] = hybrid_result.get('gpt4_reasoning', '')
                            problem['gpt4_confidence'] = hybrid_result.get('gpt4_confidence', 0.8)
                            print(f"   🔍 GPT-4 데이터 확인: reasoning='{problem['gpt4_reasoning']}', confidence={problem['gpt4_confidence']}")
                        except Exception as e:
                            print(f"⚠️ 하이브리드 분석 실패, 규칙 기반 사용: {e}")
                            problem['analysis'] = rule_analysis
                    else:
                        print(f"   ⚠️ 하이브리드 분석 사용 불가 - HYBRID_AVAILABLE: {HYBRID_AVAILABLE}")
                        problem['analysis'] = rule_analysis
            
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
                        'gpt4_movements': problem.get('gpt4_movements', []),
                        'gpt4_challenges': problem.get('gpt4_challenges', []),
                        'gpt4_tips': problem.get('gpt4_tips', [])
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
        from gpt4_analyzer import analyze_with_gpt4_vision
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

