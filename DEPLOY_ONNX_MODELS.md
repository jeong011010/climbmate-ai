# 🚀 ONNX 모델 서버 배포 가이드

## 📋 개요

ONNX 모델 파일이 440MB로 GitHub 업로드 제한(100MB)을 초과하므로, **서버에서 직접 변환**해야 합니다.

---

## 🔧 서버에서 모델 변환

### **1. 서버 접속**

```bash
ssh ubuntu@3.38.94.104
cd ~/climbmate-ai
```

### **2. 최신 코드 가져오기**

```bash
git pull origin main
```

### **3. 변환 의존성 설치**

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install ultralytics
pip3 install git+https://github.com/openai/CLIP.git
pip3 install ftfy regex
pip3 install onnx onnxruntime onnxscript
```

### **4. 모델 변환 실행**

```bash
python3 convert_models_to_onnx.py
```

**예상 출력**:
```
================================================================================
🚀 ClimbMate AI 모델 ONNX 변환
================================================================================

================================================================================
🔄 YOLO 모델 → ONNX 변환
================================================================================
📂 커스텀 모델 사용: holdcheck/roboflow_weights/weights.pt
🔄 ONNX 변환 중... (img_size=640)
✅ YOLO ONNX 변환 완료!
📦 파일: frontend/public/models/yolo.onnx (104MB)

================================================================================
🔄 CLIP 모델 → ONNX 변환
================================================================================
📦 CLIP 모델 로드 중... (ViT-B/32)
🔄 Visual Encoder ONNX 변환 중...
✅ CLIP ONNX 변환 완료! (336MB)

================================================================================
📝 모델 메타데이터 생성
================================================================================
✅ 모델 정보 생성: frontend/public/models/model_info.json

================================================================================
📊 변환 결과 요약
================================================================================
  YOLO: ✅ 성공
  CLIP: ✅ 성공
  Info: ✅ 성공
================================================================================

🎉 모든 모델 변환 완료!
```

### **5. 변환된 모델 확인**

```bash
ls -lh frontend/public/models/

# 예상 출력:
# -rw-r--r-- 1 ubuntu ubuntu 954K Oct 19 05:33 clip.onnx
# -rw-r--r-- 1 ubuntu ubuntu 335M Oct 19 05:33 clip.onnx.data
# -rw-r--r-- 1 ubuntu ubuntu 677B Oct 19 05:33 model_info.json
# -rw-r--r-- 1 ubuntu ubuntu 104M Oct 19 05:32 yolo.onnx
```

### **6. 프론트엔드 재빌드**

```bash
docker compose build frontend
docker compose up -d
```

---

## 🎯 테스트

### **웹사이트 접속**

```
https://climbmate.store
```

### **브라우저 콘솔 확인** (F12)

성공 시:
```
🚀 클라이언트 사이드 AI 분석 시작...
📦 ONNX Runtime 로딩 중...
✅ ONNX Runtime 로드 완료
🚀 AI 모델 다운로드 및 로딩 시작...
⏳ 처음 사용 시 440MB 다운로드 (이후에는 캐시 사용)
  📦 YOLO 모델 다운로드 중... (104MB)
  ✅ YOLO 모델 로드 완료
  📦 CLIP 모델 다운로드 중... (336MB)
  ✅ CLIP 모델 로드 완료
🎉 실제 YOLO + CLIP AI 모델 로드 완료!
🔍 YOLO로 홀드 감지 중...
✅ YOLO: 15개 홀드 감지 완료
🎨 CLIP으로 색상 분석 중...
✅ CLIP: 색상 분석 완료
✅ 클라이언트 사이드 분석 완료!
```

---

## ⚠️ 주의사항

### **첫 사용 시**
- 440MB 모델 다운로드 (1-3분 소요)
- 이후 브라우저 캐시에서 즉시 로드

### **브라우저 요구사항**
- 최신 Chrome, Firefox, Safari, Edge
- WebAssembly 지원 필수
- 최소 1GB RAM 권장 (PC/태블릿)

### **모바일**
- 최신 스마트폰: 작동 가능 (다운로드 시간 김)
- 구형 폰: 메모리 부족 가능

---

## 🔍 문제 해결

### **모델 로드 실패**

브라우저 콘솔에서:
```
⚠️ YOLO 모델 로드 실패: Failed to fetch
⚠️ CLIP 모델 로드 실패: Failed to fetch
```

**해결**:
```bash
# 서버에서 모델 파일 확인
ls -lh frontend/public/models/

# Nginx 설정 확인 (정적 파일 서빙)
docker compose logs nginx | grep models
```

### **메모리 부족**

브라우저 크래시 시:
- 다른 탭 닫기
- 브라우저 재시작
- 메모리 많은 기기에서 테스트

---

## 📊 성능 예상

### **PC/노트북**
- 모델 다운로드: 1-3분 (첫 사용)
- 분석 속도: 2-5초
- 재방문: 즉시 로드 (캐시)

### **태블릿**
- 모델 다운로드: 2-5분
- 분석 속도: 5-10초

### **모바일**
- 모델 다운로드: 3-10분
- 분석 속도: 10-30초

---

## ✅ 완료 체크리스트

- [ ] 서버에서 `convert_models_to_onnx.py` 실행
- [ ] `frontend/public/models/` 에 ONNX 파일 생성 확인
- [ ] 프론트엔드 재빌드
- [ ] 웹사이트에서 테스트
- [ ] 브라우저 콘솔에서 모델 로드 확인
- [ ] 실제 분석 결과 확인

---

## 🎉 결과

성공하면:
- ✅ 서버 메모리 부담 제로
- ✅ 무제한 동시 사용자
- ✅ 커스텀 YOLO + CLIP 정확도 유지
- ✅ 개인정보 보호 (이미지가 서버로 전송 안 됨)

