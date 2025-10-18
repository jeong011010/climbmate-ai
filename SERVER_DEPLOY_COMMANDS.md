# 🚀 서버 배포 명령어 (단계별 실행)

## 1️⃣ pip3 설치
```bash
sudo apt update
sudo apt install -y python3-pip
```

## 2️⃣ 변환 의존성 설치
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install ultralytics
pip3 install git+https://github.com/openai/CLIP.git
pip3 install ftfy regex
pip3 install onnx onnxruntime onnxscript
```

## 3️⃣ 모델 변환 실행
```bash
python3 convert_models_to_onnx.py
```

## 4️⃣ 프론트엔드 재빌드 (PWA 파일 크기 제한 수정됨)
```bash
docker compose build frontend
docker compose up -d
```

## 5️⃣ 서비스 상태 확인
```bash
docker compose ps
docker compose logs frontend
```

---

## 🔍 예상 결과

### 모델 변환 성공 시:
```
================================================================================
📊 변환 결과 요약
================================================================================
  YOLO: ✅ 성공
  CLIP: ✅ 성공
  Info: ✅ 성공
================================================================================

🎉 모든 모델 변환 완료!
```

### 프론트엔드 빌드 성공 시:
```
✓ built in 7.70s
```

---

## ⚠️ 문제 해결

### pip3 설치 실패 시:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
```

### 모델 변환 실패 시:
```bash
# 메모리 부족 시 스왑 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Docker 빌드 실패 시:
```bash
# 디스크 공간 확인
df -h

# Docker 정리
docker system prune -a -f
```

