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
import psutil
import gc
import torch
import redis
from celery_app import analyze_image_task

# 🚀 메모리 최적화: 스레드 수 제한 (메모리 절약)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except:
    pass

# holdcheck 모듈 경로 추가
holdcheck_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'holdcheck')
sys.path.insert(0, holdcheck_path)

# backend 모듈 경로 추가
backend_path = os.path.dirname(__file__)
sys.path.insert(0, backend_path)

# Redis 연결 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)

from preprocess import preprocess
from clustering import clip_ai_color_clustering, analyze_problem

def get_memory_usage():
    """📊 현재 메모리 사용량 반환 (MB 단위)"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # 실제 메모리 사용량 (MB)
        'vms': memory_info.vms / 1024 / 1024,  # 가상 메모리 사용량 (MB)
        'percent': process.memory_percent(),    # 시스템 메모리 대비 비율
        'available': psutil.virtual_memory().available / 1024 / 1024  # 사용 가능한 메모리 (MB)
    }

def log_memory_usage(stage_name):
    """📊 메모리 사용량 로그 출력"""
    memory = get_memory_usage()
    print(f"📊 [{stage_name}] 메모리 사용량:")
    print(f"   🔸 실제 메모리: {memory['rss']:.1f}MB")
    print(f"   🔸 가상 메모리: {memory['vms']:.1f}MB") 
    print(f"   🔸 사용률: {memory['percent']:.1f}%")
    print(f"   🔸 사용 가능: {memory['available']:.1f}MB")
    return memory

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

# CORS 설정 (React 개발 서버용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 CRITICAL: FastAPI startup event로 모델 사전 로딩 (확실하게!)
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 AI 모델을 사전 로딩하여 첫 요청 시 먹통 방지"""
    print("")
    print("=" * 80)
    print("🚀 AI 모델 사전 로딩 시작... (이 과정은 서버 시작 시 1회만 실행됩니다)")
    print("=" * 80)
    
    try:
        from preprocess import get_yolo_model, get_clip_model
        
        # YOLO 모델 사전 로딩
        print("")
        print("📦 1/2: YOLO 모델 사전 로딩 중...")
        log_memory_usage("YOLO 로딩 전")
        yolo_model = get_yolo_model()
        log_memory_usage("YOLO 로딩 후")
        print("✅ YOLO 모델 로딩 완료!")
        
        # CLIP 모델 사전 로딩 (338MB → 151MB)
        print("")
        print("📦 2/2: CLIP 모델 사전 로딩 중...")
        log_memory_usage("CLIP 로딩 전")
        clip_model, clip_preprocess, clip_device = get_clip_model()
        log_memory_usage("CLIP 로딩 후")
        print("✅ CLIP 모델 로딩 완료!")
        
        # 메모리 정리
        gc.collect()
        
        print("")
        print("=" * 80)
        print("✅ 모든 AI 모델 사전 로딩 완료! 이제 첫 요청부터 빠르게 응답합니다.")
        print("=" * 80)
        log_memory_usage("모델 로딩 완료 후")
        print("")
        
    except Exception as e:
        print("")
        print("=" * 80)
        print(f"⚠️  모델 사전 로딩 실패: {e}")
        print(f"⚠️  첫 요청 시 모델이 로딩됩니다 (느릴 수 있음)")
        print("=" * 80)
        print("")
        import traceback
        traceback.print_exc()

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
    🚀 비동기 클라이밍 벽 이미지 분석 (즉시 응답 + 백그라운드 처리)
    
    Parameters:
    - file: 이미지 파일
    - wall_angle: 벽 각도 (overhang, slab, face, null)
    
    Returns:
    - task_id: 작업 ID (상태 확인용)
    - status: 작업 상태
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        
        # Base64 인코딩
        image_data_base64 = base64.b64encode(contents).decode('utf-8')
        
        # 분석 파라미터 설정
        params = {
            'conf': 0.4,
            'brightness_normalization': False,
            'brightness_filter': False,
            'min_brightness': 0,
            'max_brightness': 100,
            'saturation_filter': False,
            'min_saturation': 0,
            'mask_refinement': 5,
            'use_clip_ai': True
        }
        
        # Celery 작업 시작
        task = analyze_image_task.delay(image_data_base64, params)
        
        print(f"🚀 분석 작업 시작: {task.id}")
        
        return {
            "task_id": task.id,
            "status": "started",
            "message": "AI 분석이 시작되었습니다. /api/analysis-status/{task_id}로 진행상황을 확인하세요."
        }
        
    except Exception as e:
        print(f"❌ 분석 요청 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis request failed: {str(e)}")

@app.post("/api/analyze-stream")
async def analyze_image_stream(
    file: UploadFile = File(...),
    wall_angle: str = Form(None),
    conf: float = Form(0.4),
    brightness_normalization: bool = Form(False),
    brightness_filter: bool = Form(False),
    min_brightness: int = Form(0),
    max_brightness: int = Form(100),
    saturation_filter: bool = Form(False),
    min_saturation: int = Form(0),
    mask_refinement: int = Form(5),
    use_clip_ai: bool = Form(True)
):
    """
    🚀 비동기 스트리밍 분석 (즉시 응답 + 백그라운드 처리)
    
    기존 동기식 스트리밍을 비동기식으로 변경하여 다른 요청을 블록킹하지 않음
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        
        # Base64 인코딩
        image_data_base64 = base64.b64encode(contents).decode('utf-8')
        
        # 분석 파라미터 설정
        params = {
            'conf': conf,
            'brightness_normalization': brightness_normalization,
            'brightness_filter': brightness_filter,
            'min_brightness': min_brightness,
            'max_brightness': max_brightness,
            'saturation_filter': saturation_filter,
            'min_saturation': min_saturation,
            'mask_refinement': mask_refinement,
            'use_clip_ai': use_clip_ai
        }
        
        # Celery 작업 시작
        task = analyze_image_task.delay(image_data_base64, params)
        
        print(f"🚀 스트리밍 분석 작업 시작: {task.id}")
        
        return {
            "task_id": task.id,
            "status": "started",
            "message": "AI 분석이 시작되었습니다. /api/analysis-status/{task_id}로 진행상황을 확인하세요."
        }
        
    except Exception as e:
        print(f"❌ 스트리밍 분석 요청 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming analysis request failed: {str(e)}")

@app.get("/api/analysis-status/{task_id}")
async def get_analysis_status(task_id: str):
    """
    🚀 분석 작업 상태 확인 (Celery 표준 방식)
    
    Parameters:
    - task_id: 작업 ID
    
    Returns:
    - status: 작업 상태 (started, processing, completed, failed)
    - progress: 진행률 (0-100)
    - message: 상태 메시지
    - result: 분석 결과 (완료 시)
    """
    try:
        # Celery 표준 방식으로 작업 상태 조회
        task = analyze_image_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response_data = {
                "task_id": task_id, 
                "status": "pending", 
                "message": "분석 대기 중...", 
                "progress": 0
            }
        elif task.state == 'STARTED':
            meta = task.info or {}
            response_data = {
                "task_id": task_id,
                "status": "started",
                "message": meta.get('message', '분석 진행 중...'),
                "progress": meta.get('progress', 0)
            }
        elif task.state == 'PROGRESS':
            meta = task.info or {}
            response_data = {
                "task_id": task_id,
                "status": "progress",
                "message": meta.get('message', '분석 진행 중...'),
                "progress": meta.get('progress', 0)
            }
        elif task.state == 'SUCCESS':
            result = task.result
            response_data = {
                "task_id": task_id,
                "status": "completed",
                "message": "분석 완료!",
                "progress": 100,
                "result": result
            }
        elif task.state == 'FAILURE':
            response_data = {
                "task_id": task.id,
                "status": "failed",
                "message": f"분석 실패: {str(task.info)}",
                "progress": 100
            }
        else:
            response_data = {
                "task_id": task.id, 
                "status": task.state, 
                "message": "알 수 없는 상태", 
                "progress": 0
            }
        
        return response_data
        
    except Exception as e:
        print(f"❌ 작업 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@app.get("/api/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        # Redis 연결 확인
        redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "memory": get_memory_usage(),
        "timestamp": psutil.time.time()
    }

# 기존 API들 (데이터베이스, 통계 등)은 그대로 유지
@app.get("/api/stats")
async def get_stats():
    """통계 정보 반환"""
    if not DB_AVAILABLE:
        return {"error": "Database not available"}
    
    try:
        stats = get_model_stats()
        return stats
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """사용자 피드백 제출"""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        result = save_user_feedback(
            feedback.problem_id,
            feedback.user_difficulty,
            feedback.user_type,
            feedback.user_feedback
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gpt4-status")
async def get_gpt4_status_endpoint():
    """GPT-4 상태 확인"""
    if not GPT4_AVAILABLE:
        return {"available": False, "message": "GPT-4 module not available"}
    
    try:
        status = get_gpt4_status()
        return status
    except Exception as e:
        return {"available": False, "error": str(e)}

@app.post("/api/gpt4-analyze")
async def gpt4_analyze(file: UploadFile = File(...)):
    """GPT-4 Vision으로 분석"""
    if not GPT4_AVAILABLE:
        raise HTTPException(status_code=500, detail="GPT-4 module not available")
    
    try:
        contents = await file.read()
        result = analyze_with_gpt4_vision(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hybrid-stats")
async def get_hybrid_stats():
    """하이브리드 분석 통계"""
    if not HYBRID_AVAILABLE:
        return {"error": "Hybrid analyzer not available"}
    
    try:
        stats = get_analysis_method_stats()
        return stats
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)