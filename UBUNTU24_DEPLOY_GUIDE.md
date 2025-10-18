# 🚀 Ubuntu 24.04 서버 배포 명령어 (가상환경 사용)

## 1️⃣ 필요한 패키지 설치
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-full
```

## 2️⃣ 가상환경 생성 및 활성화
```bash
cd ~/climbmate-ai
python3 -m venv model_converter_env
source model_converter_env/bin/activate
```

## 3️⃣ 가상환경에서 의존성 설치
```bash
# 가상환경이 활성화되면 프롬프트에 (model_converter_env) 표시됨
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
pip install ftfy regex
pip install onnx onnxruntime onnxscript
```

## 4️⃣ 모델 변환 실행
```bash
python convert_models_to_onnx.py
```

## 5️⃣ 가상환경 비활성화
```bash
deactivate
```

## 6️⃣ 프론트엔드 재빌드
```bash
docker compose build frontend
docker compose up -d
```

---

## 🔍 예상 출력

### 가상환경 활성화 시:
```bash
ubuntu@ip-172-31-12-99:~/climbmate-ai$ source model_converter_env/bin/activate
(model_converter_env) ubuntu@ip-172-31-12-99:~/climbmate-ai$ 
```

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

---

## ⚠️ 문제 해결

### 가상환경 생성 실패 시:
```bash
sudo apt install -y python3-full python3-dev
python3 -m venv model_converter_env --clear
```

### 메모리 부족 시:
```bash
# 스왑 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 메모리 확인
free -h
```

### Docker 빌드 실패 시:
```bash
# 디스크 공간 확인
df -h

# Docker 정리
docker system prune -a -f
```

---

## 📝 참고사항

- **가상환경**: 시스템 Python과 분리된 독립 환경
- **활성화**: `source model_converter_env/bin/activate`
- **비활성화**: `deactivate`
- **재사용**: 다음에도 `source model_converter_env/bin/activate`로 활성화 가능
