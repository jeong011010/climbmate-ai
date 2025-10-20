from celery import Celery
import os

# Redis 브로커 설정
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Celery 앱 생성
celery_app = Celery(
    'climbmate_ai',
    broker=redis_url,
    backend=redis_url,
    include=['backend.ai_tasks']
)

# Celery 설정
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    result_extended=True,  # 확장된 결과 메타데이터 활성화
    timezone='Asia/Seoul',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10분 타임아웃
    task_soft_time_limit=540,  # 9분 소프트 타임아웃
    worker_prefetch_multiplier=1,  # 동시 처리 작업 수 제한
    task_acks_late=True,  # 작업 완료 후 ACK
    worker_disable_rate_limits=False,
    result_expires=3600,  # 결과는 1시간 후 삭제
    result_backend_transport_options={'master_name': 'mymaster'},  # Redis Sentinel 옵션 (선택)
)

if __name__ == '__main__':
    celery_app.start()