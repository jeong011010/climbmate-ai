# 🚀 TensorFlow.js 모델 변환 가이드

## 📋 개요

이 가이드는 PyTorch YOLO + CLIP 모델을 TensorFlow.js로 변환하여 사용자 브라우저에서 실행하는 방법을 설명합니다.

---

## 🔧 변환 과정

### **1. 의존성 설치 (로컬)**

```bash
pip install -r requirements_converter.txt
```

### **2. 모델 변환 실행**

```bash
python convert_models_to_tfjs.py
```

### **3. 변환 결과**

성공 시:
```
frontend/public/models/
├── yolo/
│   ├── model.json
│   └── group1-shard1of1.bin
└── clip/
    ├── model.json
    └── group1-shard1of1.bin
```

---

## 🎯 작동 방식

### **변환 파이프라인**

```
PyTorch (.pt)
    ↓
TorchScript
    ↓
ONNX (.onnx)
    ↓
TensorFlow (SavedModel)
    ↓
TensorFlow.js (model.json + .bin)
```

### **브라우저에서 실행**

```javascript
// 사용자가 이미지 업로드
→ TensorFlow.js 로드
→ YOLO 모델 로드 (/models/yolo/model.json)
→ CLIP 모델 로드 (/models/clip/model.json)
→ 홀드 감지 (YOLO)
→ 색상 분석 (CLIP)
→ 문제 생성
→ 결과 표시
```

---

## ⚠️ 주의사항

### **모델 크기**
- YOLO: ~50MB
- CLIP: ~150MB
- **총 200MB를 사용자가 다운로드**

### **브라우저 요구사항**
- 최신 Chrome, Firefox, Safari, Edge
- WebGL 지원 필수
- 최소 2GB RAM 권장

### **모바일**
- 최신 스마트폰: 작동 가능
- 구형 폰: 메모리 부족 가능

---

## 🚀 배포

### **변환 후 프론트엔드 배포**

```bash
# 1. 변환된 모델 확인
ls frontend/public/models/yolo/
ls frontend/public/models/clip/

# 2. 프론트엔드 빌드
cd frontend
npm run build

# 3. 배포
docker compose build frontend
docker compose up -d
```

---

## 🔍 문제 해결

### **변환 실패 시**

모델 변환이 실패하면 자동으로 **모의 모드**로 전환됩니다:
- YOLO 없음 → 랜덤 홀드 생성
- CLIP 없음 → 랜덤 색상 할당
- 정확도는 낮지만 기능 테스트 가능

### **모델 로드 실패 시**

브라우저 콘솔에서 확인:
```javascript
// 성공
✅ YOLO 모델 로드 완료
✅ CLIP 모델 로드 완료
🎉 실제 AI 모델 로드 완료!

// 실패
⚠️ YOLO 모델 로드 실패, 모의 모드로 전환
⚠️ CLIP 모델 로드 실패, 모의 모드로 전환
⚠️ 모의 모드로 실행됩니다.
```

---

## 📊 성능 비교

| 방식 | 서버 부담 | 정확도 | 속도 (PC) | 속도 (모바일) |
|------|-----------|--------|-----------|---------------|
| **서버 AI** | 높음 | ⭐⭐⭐⭐⭐ | ⚡⚡ | ⚡⚡ |
| **TensorFlow.js** | 없음 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | ⚡⚡ |
| **모의 모드** | 없음 | ⭐⭐ | ⚡⚡⚡ | ⚡⚡⚡ |

---

## 💡 팁

### **첫 로딩 개선**

사용자 경험 향상을 위해:
1. 로딩 프로그레스 바 표시
2. 모델 다운로드 상태 표시
3. 캐싱으로 재방문 시 빠른 로드

### **메모리 최적화**

```javascript
// 사용 후 텐서 메모리 해제
tensor.dispose();

// 주기적으로 가비지 컬렉션
tf.disposeVariables();
```

---

## ✅ 완료 체크리스트

- [ ] `requirements_converter.txt` 설치
- [ ] `convert_models_to_tfjs.py` 실행
- [ ] `frontend/public/models/` 확인
- [ ] 프론트엔드 빌드 및 배포
- [ ] 브라우저에서 테스트
- [ ] 모바일에서 테스트

---

## 🔗 참고

- [TensorFlow.js 문서](https://www.tensorflow.org/js)
- [ONNX 변환 가이드](https://github.com/onnx/onnx-tensorflow)
- [tf2onnx 문서](https://github.com/onnx/tensorflow-onnx)

