#!/usr/bin/env python3

# 🌐 간단한 웹 기반 메모리 모니터링 서버
# 사용법: python3 web_monitor.py
# 브라우저에서 http://localhost:8080 접속

from flask import Flask, render_template_string
import psutil
import json
import time
import threading
from datetime import datetime

app = Flask(__name__)

# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚀 ClimbMate 메모리 모니터링</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .critical { border-left: 5px solid #e74c3c; }
        .warning { border-left: 5px solid #f39c12; }
        .ok { border-left: 5px solid #27ae60; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .value { font-size: 24px; font-weight: bold; }
        .label { color: #666; font-size: 14px; }
        .progress-bar { width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; transition: width 0.3s ease; }
        .progress-critical { background: #e74c3c; }
        .progress-warning { background: #f39c12; }
        .progress-ok { background: #27ae60; }
        h1 { color: #2c3e50; text-align: center; }
        .timestamp { text-align: center; color: #7f8c8d; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ClimbMate 메모리 모니터링</h1>
        <div class="timestamp">마지막 업데이트: {{ timestamp }}</div>
        
        <div class="card {{ system_status }}">
            <h2>💾 시스템 메모리</h2>
            <div class="metric">
                <div class="value" style="color: {{ system_color }}">{{ system_percent }}%</div>
                <div class="label">사용률</div>
            </div>
            <div class="metric">
                <div class="value">{{ system_used }}</div>
                <div class="label">사용량</div>
            </div>
            <div class="metric">
                <div class="value">{{ system_total }}</div>
                <div class="label">총 메모리</div>
            </div>
            <div class="metric">
                <div class="value">{{ system_available }}</div>
                <div class="label">사용 가능</div>
            </div>
            <div style="margin-top: 15px;">
                <div class="progress-bar">
                    <div class="progress-fill {{ system_progress_class }}" style="width: {{ system_percent }}%"></div>
                </div>
            </div>
        </div>
        
        <div class="card {{ swap_status }}">
            <h2>🔄 Swap 메모리</h2>
            <div class="metric">
                <div class="value" style="color: {{ swap_color }}">{{ swap_percent }}%</div>
                <div class="label">사용률</div>
            </div>
            <div class="metric">
                <div class="value">{{ swap_used }}</div>
                <div class="label">사용량</div>
            </div>
            <div class="metric">
                <div class="value">{{ swap_total }}</div>
                <div class="label">총 Swap</div>
            </div>
            <div style="margin-top: 15px;">
                <div class="progress-bar">
                    <div class="progress-fill {{ swap_progress_class }}" style="width: {{ swap_percent }}%"></div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🐳 Docker 컨테이너</h2>
            {% for container in containers %}
            <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                <strong>{{ container.name }}</strong><br>
                메모리: {{ container.memory }} ({{ container.memory_percent }}%)<br>
                CPU: {{ container.cpu }}<br>
                상태: {{ container.status }}
            </div>
            {% endfor %}
        </div>
        
        <div class="card">
            <h2>📊 시스템 정보</h2>
            <div class="metric">
                <div class="value">{{ cpu_percent }}%</div>
                <div class="label">CPU 사용률</div>
            </div>
            <div class="metric">
                <div class="value">{{ load_avg }}</div>
                <div class="label">Load Average</div>
            </div>
            <div class="metric">
                <div class="value">{{ disk_percent }}%</div>
                <div class="label">디스크 사용률</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

def get_system_info():
    """시스템 정보 수집"""
    # 메모리 정보
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    # CPU 정보
    cpu_percent = psutil.cpu_percent(interval=1)
    load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
    
    # 디스크 정보
    disk = psutil.disk_usage('/')
    
    return {
        'memory': memory,
        'swap': swap,
        'cpu_percent': cpu_percent,
        'load_avg': f"{load_avg:.2f}",
        'disk_percent': f"{(disk.used / disk.total) * 100:.1f}"
    }

def get_docker_containers():
    """Docker 컨테이너 정보 수집 (실제로는 subprocess 사용)"""
    import subprocess
    try:
        result = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}'], 
                              capture_output=True, text=True, timeout=5)
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line and 'climbmate' in line:
                parts = line.split('\t')
                if len(parts) >= 4:
                    containers.append({
                        'name': parts[0],
                        'memory': parts[1],
                        'memory_percent': parts[2],
                        'cpu': parts[3],
                        'status': 'Running'
                    })
        return containers
    except:
        return [{'name': 'Docker 정보를 가져올 수 없음', 'memory': 'N/A', 'memory_percent': 'N/A', 'cpu': 'N/A', 'status': 'N/A'}]

@app.route('/')
def index():
    """메인 페이지"""
    info = get_system_info()
    containers = get_docker_containers()
    
    # 메모리 상태 결정
    memory_percent = info['memory'].percent
    if memory_percent > 90:
        system_status = 'critical'
        system_color = '#e74c3c'
        system_progress_class = 'progress-critical'
    elif memory_percent > 80:
        system_status = 'warning'
        system_color = '#f39c12'
        system_progress_class = 'progress-warning'
    else:
        system_status = 'ok'
        system_color = '#27ae60'
        system_progress_class = 'progress-ok'
    
    # Swap 상태 결정
    swap_percent = info['swap'].percent
    if swap_percent > 50:
        swap_status = 'warning'
        swap_color = '#f39c12'
        swap_progress_class = 'progress-warning'
    else:
        swap_status = 'ok'
        swap_color = '#27ae60'
        swap_progress_class = 'progress-ok'
    
    return render_template_string(HTML_TEMPLATE,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        system_percent=f"{memory_percent:.1f}",
        system_used=f"{info['memory'].used / 1024 / 1024 / 1024:.1f}GB",
        system_total=f"{info['memory'].total / 1024 / 1024 / 1024:.1f}GB",
        system_available=f"{info['memory'].available / 1024 / 1024 / 1024:.1f}GB",
        system_status=system_status,
        system_color=system_color,
        system_progress_class=system_progress_class,
        swap_percent=f"{swap_percent:.1f}",
        swap_used=f"{info['swap'].used / 1024 / 1024 / 1024:.1f}GB",
        swap_total=f"{info['swap'].total / 1024 / 1024 / 1024:.1f}GB",
        swap_status=swap_status,
        swap_color=swap_color,
        swap_progress_class=swap_progress_class,
        containers=containers,
        cpu_percent=f"{info['cpu_percent']:.1f}",
        load_avg=info['load_avg'],
        disk_percent=info['disk_percent']
    )

if __name__ == '__main__':
    print("🌐 메모리 모니터링 웹 서버 시작...")
    print("📱 브라우저에서 http://localhost:8080 접속")
    print("종료하려면 Ctrl+C를 누르세요")
    app.run(host='0.0.0.0', port=8080, debug=False)
