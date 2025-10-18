#!/bin/bash

# 🚀 ClimbMate 자동 정리 스크립트
# 디스크 공간 부족 시 자동으로 정리

echo "🧹 ClimbMate 자동 정리 시작..."

# Docker 정리
echo "📦 Docker 리소스 정리 중..."
docker system prune -a -f
docker builder prune -a -f
docker volume prune -f
docker network prune -f

# 로그 파일 정리
echo "📝 로그 파일 정리 중..."
sudo find /var/lib/docker/containers/ -name "*.log" -size +50M -delete 2>/dev/null || true
sudo journalctl --vacuum-time=3d 2>/dev/null || true

# 임시 파일 정리
echo "🗑️ 임시 파일 정리 중..."
sudo rm -rf /tmp/* 2>/dev/null || true
sudo rm -rf /var/tmp/* 2>/dev/null || true
sudo apt clean 2>/dev/null || true

# 패키지 캐시 정리
echo "📦 패키지 캐시 정리 중..."
sudo apt autoremove -y 2>/dev/null || true

# 디스크 사용량 확인
echo "📊 정리 후 디스크 사용량:"
df -h /

echo "✅ 자동 정리 완료!"
echo "💡 정기적으로 실행하려면: crontab -e 에서 다음 추가:"
echo "   0 2 * * * /path/to/cleanup.sh"
