"""
🚀 Celery 워커 설정 - AI 분석 작업을 비동기로 처리
"""
import os
import redis
from celery import Celery
from holdcheck.preprocess import preprocess
import base64
import json
import time
import psutil
import gc

# Redis 연결 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery 앱 생성
celery_app = Celery(
    'climbmate',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['celery_app']
)

# Celery 설정
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Seoul',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5분으로 증가
    task_soft_time_limit=270,  # 4.5분 소프트 제한
    worker_prefetch_multiplier=1,  # 한 번에 하나의 작업만 처리
    task_acks_late=True,  # 작업 완료 후에만 ACK
    worker_max_tasks_per_child=5,  # 메모리 누수 방지 (더 자주 재시작)
    worker_max_memory_per_child=500000,  # 500MB로 증가
    result_expires=3600,  # 결과 1시간 후 만료
)

# Redis 클라이언트 (작업 상태 저장용)
redis_client = redis.from_url(REDIS_URL)

def get_memory_usage():
    """메모리 사용량 측정"""
    process = psutil.Process()
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    return {
        'rss': round(memory_info.rss / 1024 / 1024, 1),  # MB
        'vms': round(memory_info.vms / 1024 / 1024, 1),  # MB
        'percent': round(system_memory.percent, 1),
        'available': round(system_memory.available / 1024 / 1024, 1)  # MB
    }

def log_memory_usage(stage_name):
    """메모리 사용량 로깅"""
    memory = get_memory_usage()
    print(f"📊 [{stage_name}] 메모리 사용량:")
    print(f"   🔸 실제 메모리: {memory['rss']}MB ({memory['percent']}%)")
    print(f"   🔸 가상 메모리: {memory['vms']}MB")
    print(f"   🔸 사용 가능: {memory['available']}MB")
    
    # 메모리 사용률 경고
    if memory['percent'] > 90:
        print(f"🚨 경고: 메모리 사용률이 {memory['percent']}%입니다!")
    elif memory['percent'] > 80:
        print(f"⚠️  주의: 메모리 사용률이 {memory['percent']}%입니다!")

@celery_app.task(bind=True)
def analyze_image_task(self, image_data_base64, params):
    """
    🚀 AI 이미지 분석 작업 (Celery 워커에서 실행)
    
    Args:
        image_data_base64: Base64 인코딩된 이미지 데이터
        params: 분석 파라미터 딕셔너리
    
    Returns:
        분석 결과 딕셔너리
    """
    task_id = self.request.id
    print(f"")
    print(f"=" * 80)
    print(f"🚀 AI 분석 작업 시작 (Task ID: {task_id})")
    print(f"=" * 80)
    
    try:
        # 작업 상태 업데이트
        redis_client.setex(f"task_status:{task_id}", 300, json.dumps({
            'status': 'started',
            'progress': 0,
            'message': 'AI 분석 시작...',
            'started_at': time.time()
        }))
        
        log_memory_usage("작업 시작")
        
        # Base64 디코딩
        print("📷 이미지 데이터 디코딩 중...")
        image_data = base64.b64decode(image_data_base64)
        
        # PIL Image로 변환
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_data))
        
        redis_client.setex(f"task_status:{task_id}", 300, json.dumps({
            'status': 'processing',
            'progress': 10,
            'message': '이미지 전처리 중...',
            'started_at': time.time()
        }))
        
        log_memory_usage("이미지 로딩 후")
        
        # AI 분석 실행
        print("🤖 AI 분석 실행 중...")
        redis_client.setex(f"task_status:{task_id}", 300, json.dumps({
            'status': 'processing',
            'progress': 30,
            'message': 'YOLO 모델로 홀드 감지 중...',
            'started_at': time.time()
        }))
        
        result = preprocess(
            image,
            model_path=params.get('model_path', '/app/holdcheck/roboflow_weights/weights.pt'),
            conf=params.get('conf', 0.4),
            brightness_normalization=params.get('brightness_normalization', False),
            brightness_filter=params.get('brightness_filter', False),
            min_brightness=params.get('min_brightness', 0),
            max_brightness=params.get('max_brightness', 100),
            saturation_filter=params.get('saturation_filter', False),
            min_saturation=params.get('min_saturation', 0),
            mask_refinement=params.get('mask_refinement', 5),
            use_clip_ai=params.get('use_clip_ai', True)
        )
        
        redis_client.setex(f"task_status:{task_id}", 300, json.dumps({
            'status': 'processing',
            'progress': 80,
            'message': '결과 후처리 중...',
            'started_at': time.time()
        }))
        
        log_memory_usage("분석 완료 후")
        
        # 결과 저장
        redis_client.setex(f"task_result:{task_id}", 3600, json.dumps(result))
        
        # 완료 상태 업데이트
        redis_client.setex(f"task_status:{task_id}", 300, json.dumps({
            'status': 'completed',
            'progress': 100,
            'message': '분석 완료!',
            'started_at': time.time(),
            'completed_at': time.time(),
            'result': result
        }))
        
        print(f"✅ AI 분석 완료! (Task ID: {task_id})")
        log_memory_usage("작업 완료")
        
        # 메모리 정리
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"❌ AI 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 상태 업데이트
        redis_client.setex(f"task_status:{task_id}", 300, json.dumps({
            'status': 'failed',
            'progress': 0,
            'message': f'분석 실패: {str(e)}',
            'started_at': time.time(),
            'error': str(e)
        }))
        
        raise e

@celery_app.task
def cleanup_old_tasks():
    """오래된 작업 데이터 정리"""
    try:
        # 1시간 이상 된 작업 데이터 삭제
        keys = redis_client.keys("task_*")
        for key in keys:
            ttl = redis_client.ttl(key)
            if ttl == -1:  # TTL이 설정되지 않은 경우
                redis_client.expire(key, 3600)  # 1시간 후 만료
        print(f"🧹 작업 데이터 정리 완료: {len(keys)}개 키 처리")
    except Exception as e:
        print(f"❌ 작업 데이터 정리 실패: {e}")

if __name__ == '__main__':
    celery_app.start()
