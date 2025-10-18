#!/bin/bash

# 🚀 실시간 메모리 모니터링 스크립트
# 사용법: ./monitor_memory.sh

echo "📊 실시간 메모리 모니터링 시작..."
echo "종료하려면 Ctrl+C를 누르세요"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

while true; do
    # 현재 시간
    timestamp=$(date '+%H:%M:%S')
    
    # 시스템 메모리 정보
    memory_info=$(free -h | grep Mem)
    total=$(echo $memory_info | awk '{print $2}')
    used=$(echo $memory_info | awk '{print $3}')
    available=$(echo $memory_info | awk '{print $7}')
    percent=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    # Swap 정보
    swap_info=$(free -h | grep Swap)
    swap_used=$(echo $swap_info | awk '{print $3}')
    swap_total=$(echo $swap_info | awk '{print $2}')
    
    # Docker 컨테이너 메모리 사용량
    docker_memory=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null | grep climbmate || echo "No containers")
    
    # 메모리 사용률에 따른 색상
    if (( $(echo "$percent > 90" | bc -l) )); then
        color=$RED
        status="🔴 CRITICAL"
    elif (( $(echo "$percent > 80" | bc -l) )); then
        color=$YELLOW
        status="🟡 WARNING"
    else
        color=$GREEN
        status="🟢 OK"
    fi
    
    # 화면 지우기
    clear
    
    echo -e "${BLUE}📊 실시간 메모리 모니터링 - $timestamp${NC}"
    echo "=================================================="
    echo ""
    
    # 시스템 메모리
    echo -e "${color}💾 시스템 메모리: $used / $total (${percent}%) $status${NC}"
    echo -e "   사용 가능: $available"
    echo ""
    
    # Swap 메모리
    if [ "$swap_used" != "0B" ]; then
        echo -e "${YELLOW}🔄 Swap 사용: $swap_used / $swap_total${NC}"
    else
        echo -e "${GREEN}🔄 Swap 사용: 없음${NC}"
    fi
    echo ""
    
    # Docker 컨테이너 메모리
    echo -e "${BLUE}🐳 Docker 컨테이너 메모리:${NC}"
    if [ "$docker_memory" != "No containers" ]; then
        echo "$docker_memory" | while read line; do
            if [[ $line == *"climbmate"* ]]; then
                echo -e "   $line"
            fi
        done
    else
        echo "   컨테이너가 실행되지 않음"
    fi
    echo ""
    
    # 메모리 사용량 히스토그램 (간단한 ASCII)
    echo -e "${BLUE}📈 메모리 사용량 히스토그램:${NC}"
    bars=$(($(echo "$percent" | cut -d. -f1) / 5))
    for i in $(seq 1 20); do
        if [ $i -le $bars ]; then
            echo -n "█"
        else
            echo -n "░"
        fi
    done
    echo " ${percent}%"
    echo ""
    
    # 경고 메시지
    if (( $(echo "$percent > 90" | bc -l) )); then
        echo -e "${RED}⚠️  경고: 메모리 사용량이 90%를 초과했습니다!${NC}"
        echo -e "${RED}   백엔드 컨테이너가 OOM으로 종료될 수 있습니다.${NC}"
    elif (( $(echo "$percent > 80" | bc -l) )); then
        echo -e "${YELLOW}⚠️  주의: 메모리 사용량이 80%를 초과했습니다.${NC}"
    fi
    
    echo ""
    echo "다음 업데이트까지 2초 대기..."
    
    sleep 2
done
