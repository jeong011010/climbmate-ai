#!/bin/bash

# 🐳 Docker 컨테이너 전용 메모리 모니터링
# 사용법: ./monitor_docker.sh

echo "🐳 Docker 컨테이너 메모리 모니터링 시작..."
echo "종료하려면 Ctrl+C를 누르세요"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

while true; do
    timestamp=$(date '+%H:%M:%S')
    
    # 화면 지우기
    clear
    
    echo -e "${BLUE}🐳 Docker 컨테이너 메모리 모니터링 - $timestamp${NC}"
    echo "=================================================="
    echo ""
    
    # 모든 컨테이너 메모리 사용량
    echo -e "${BLUE}📊 모든 컨테이너 메모리 사용량:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" 2>/dev/null || echo "Docker가 실행되지 않음"
    
    echo ""
    echo -e "${BLUE}📈 ClimbMate 컨테이너 상세 정보:${NC}"
    
    # ClimbMate 관련 컨테이너만 필터링
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(climbmate|backend|frontend|nginx)" || echo "ClimbMate 컨테이너가 실행되지 않음"
    
    echo ""
    echo -e "${BLUE}💾 컨테이너별 메모리 사용량:${NC}"
    
    # 각 컨테이너의 메모리 사용량을 개별적으로 표시
    for container in $(docker ps --format "{{.Names}}" | grep climbmate); do
        memory_info=$(docker stats --no-stream --format "{{.MemUsage}}" $container 2>/dev/null)
        memory_percent=$(docker stats --no-stream --format "{{.MemPerc}}" $container 2>/dev/null)
        
        if [ ! -z "$memory_info" ]; then
            # 메모리 사용률에 따른 색상
            percent_num=$(echo $memory_percent | sed 's/%//')
            if (( $(echo "$percent_num > 80" | bc -l) )); then
                color=$RED
            elif (( $(echo "$percent_num > 60" | bc -l) )); then
                color=$YELLOW
            else
                color=$GREEN
            fi
            
            echo -e "   ${color}$container: $memory_info ($memory_percent)${NC}"
        fi
    done
    
    echo ""
    echo -e "${BLUE}🔄 시스템 리소스:${NC}"
    echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% 사용"
    echo "   Load: $(uptime | awk -F'load average:' '{print $2}')"
    
    echo ""
    echo "다음 업데이트까지 3초 대기..."
    
    sleep 3
done
