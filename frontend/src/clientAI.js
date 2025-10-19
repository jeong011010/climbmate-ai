/**
 * 🚀 클라이언트 사이드 AI 처리 v2.1.0 (ONNX Runtime Web)
 * 브라우저: YOLO 실행 | 서버: CLIP 실행
 * 사용자 브라우저에서 직접 YOLO 모델 실행 + 서버 CLIP API
 */

// API URL 설정
const API_URL = import.meta.env.VITE_API_URL || 'https://climbmate.store'

class ClientAIAnalyzer {
  constructor() {
    this.yoloSession = null;
    this.clipSession = null;
    this.isLoaded = false;
    this.ort = null;
  }

  /**
   * ONNX Runtime 로드
   */
  async loadONNXRuntime() {
    if (this.ort) return this.ort;
    
    console.log('📦 ONNX Runtime 로딩 중...');
    const ort = await import('onnxruntime-web');
    this.ort = ort;
    console.log('✅ ONNX Runtime 로드 완료');
    return ort;
  }

  /**
   * AI 모델들을 사용자 브라우저에 로드
   */
  async loadModels() {
    if (this.isLoaded) {
      console.log('✅ 모델이 이미 로드되어 있습니다.');
      return;
    }

    try {
      const ort = await this.loadONNXRuntime();
      
      console.log('🚀 YOLO 모델 로딩 시작...');
      console.log('⏳ 처음 사용 시 104MB 다운로드 (이후에는 캐시 사용)');
      
      // YOLO 모델만 로드 (CLIP은 서버에서 실행)
      try {
        console.log('  📦 YOLO 모델 다운로드 중... (104MB)');
        this.yoloSession = await ort.InferenceSession.create('/models/yolo.onnx');
        console.log('  ✅ YOLO 모델 로드 완료');
      } catch (error) {
        console.warn('  ⚠️ YOLO 모델 로드 실패:', error.message);
        this.yoloSession = null;
      }
      
      // CLIP은 서버 API 사용 (브라우저 로드 안함)
      console.log('  ℹ️  CLIP: 서버 API 사용 (브라우저 로드 불필요)');
      this.clipSession = null;
      
      this.isLoaded = true;
      
      if (this.yoloSession) {
        console.log('🎉 YOLO 로드 완료! (CLIP은 서버에서 실행)');
        return true;
      } else {
        console.log('⚠️ YOLO 로드 실패 - 모의 모드로 전환');
        return false;
      }
      
    } catch (error) {
      console.error('❌ 모델 로드 실패:', error);
      this.isLoaded = true;
      return false;
    }
  }

  /**
   * 이미지를 텐서로 변환
   */
  async imageToTensor(imageElement, targetSize) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = targetSize;
    canvas.height = targetSize;
    
    // 이미지를 캔버스에 그리기 (리사이즈)
    ctx.drawImage(imageElement, 0, 0, targetSize, targetSize);
    
    // 이미지 데이터 가져오기
    const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
    const { data } = imageData;
    
    // [H, W, C] → [C, H, W] 변환 및 정규화
    const tensor = new Float32Array(3 * targetSize * targetSize);
    
    for (let i = 0; i < targetSize * targetSize; i++) {
      tensor[i] = data[i * 4] / 255.0;  // R
      tensor[targetSize * targetSize + i] = data[i * 4 + 1] / 255.0;  // G
      tensor[2 * targetSize * targetSize + i] = data[i * 4 + 2] / 255.0;  // B
    }
    
    return tensor;
  }

  /**
   * YOLO로 홀드 감지
   */
  async detectHoldsWithYOLO(imageElement) {
    if (!this.yoloSession) {
      return this.detectHoldsMock(imageElement);
    }

    try {
      console.log('🔍 YOLO로 홀드 감지 중...');
      
      // 이미지를 640x640 텐서로 변환
      const inputTensor = await this.imageToTensor(imageElement, 640);
      
      // ONNX Runtime 추론
      const feeds = {
        'images': new this.ort.Tensor('float32', inputTensor, [1, 3, 640, 640])
      };
      
      const results = await this.yoloSession.run(feeds);
      
      // 결과 처리 (YOLO 출력 형식에 따라 다름)
      const outputData = results[Object.keys(results)[0]].data;
      
      // 홀드 추출
      const holds = this.processYOLOOutput(outputData, imageElement.width, imageElement.height);
      
      console.log(`✅ YOLO: ${holds.length}개 홀드 감지 완료`);
      return holds;
      
    } catch (error) {
      console.error('❌ YOLO 추론 실패:', error);
      return this.detectHoldsMock(imageElement);
    }
  }

  /**
   * YOLO 출력 처리 (YOLOv8 format)
   */
  processYOLOOutput(data, originalWidth, originalHeight) {
    const holds = [];
    
    // YOLOv8 출력: [1, 84, 8400] -> Transposed: [1, 8400, 84]
    // 각 detection: [cx, cy, w, h, class_conf_0, class_conf_1, ...]
    
    const numBoxes = 8400;
    const numElements = 84;
    
    // data.length가 84 * 8400 = 705,600인지 확인
    if (data.length !== numBoxes * numElements) {
      console.warn(`⚠️ 예상치 못한 YOLO 출력 크기: ${data.length}`);
      // Fallback to old logic
      return this.processYOLOOutputFallback(data, originalWidth, originalHeight);
    }
    
    for (let i = 0; i < numBoxes; i++) {
      // YOLOv8 출력은 [84, 8400] 형태
      const cx = data[i];
      const cy = data[numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];
      
      // 클래스 0 (hold)의 신뢰도
      const confidence = data[4 * numBoxes + i];
      
      if (confidence > 0.5) {
        // 좌표 변환 (640x640 -> 원본 크기)
        const x = (cx - w / 2) * originalWidth / 640;
        const y = (cy - h / 2) * originalHeight / 640;
        const width = w * originalWidth / 640;
        const height = h * originalHeight / 640;
        
        // 유효성 검사
        if (width > 5 && height > 5 && x >= 0 && y >= 0) {
          holds.push({
            x: Math.max(0, x),
            y: Math.max(0, y),
            width: Math.min(width, originalWidth - x),
            height: Math.min(height, originalHeight - y),
            confidence: confidence
          });
        }
      }
    }
    
    // NMS (Non-Maximum Suppression) - 겹치는 박스 제거
    return this.applyNMS(holds);
  }
  
  /**
   * Fallback YOLO 출력 처리
   */
  processYOLOOutputFallback(data, originalWidth, originalHeight) {
    const holds = [];
    const numDetections = Math.min(100, Math.floor(data.length / 6));
    
    for (let i = 0; i < numDetections; i++) {
      const offset = i * 6;
      const confidence = data[offset + 4];
      
      if (confidence > 0.5) {
        const xCenter = data[offset] * originalWidth / 640;
        const yCenter = data[offset + 1] * originalHeight / 640;
        const width = data[offset + 2] * originalWidth / 640;
        const height = data[offset + 3] * originalHeight / 640;
        
        holds.push({
          x: Math.max(0, xCenter - width / 2),
          y: Math.max(0, yCenter - height / 2),
          width: Math.min(width, originalWidth),
          height: Math.min(height, originalHeight),
          confidence: confidence
        });
      }
    }
    
    return holds; // 첫 커밋 때처럼 제한 없이 모든 홀드 반환
  }
  
  /**
   * NMS (Non-Maximum Suppression)
   */
  applyNMS(boxes, iouThreshold = 0.5) {
    if (boxes.length === 0) return [];
    
    // 신뢰도 순으로 정렬
    boxes.sort((a, b) => b.confidence - a.confidence);
    
    const selected = [];
    const suppressed = new Set();
    
    for (let i = 0; i < boxes.length; i++) {
      if (suppressed.has(i)) continue;
      
      selected.push(boxes[i]);
      
      for (let j = i + 1; j < boxes.length; j++) {
        if (suppressed.has(j)) continue;
        
        const iou = this.calculateIOU(boxes[i], boxes[j]);
        if (iou > iouThreshold) {
          suppressed.add(j);
        }
      }
    }
    
    return selected;
  }
  
  /**
   * IoU 계산
   */
  calculateIOU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
    
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;
    
    return intersection / (union + 1e-6);
  }

  /**
   * 🚀 서버 사이드 전체 분석 (YOLO + 마스크 + CLIP)
   */
  async analyzeWithServerSide(imageElement, wallAngle = null) {
    try {
      console.log('🚀 서버 사이드 전체 분석 시작...');
      
      // 이미지 유효성 검사
      if (!imageElement || !imageElement.width || !imageElement.height) {
        throw new Error('유효하지 않은 이미지입니다.');
      }
      
      // 이미지를 Base64로 변환
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      canvas.width = imageElement.width;
      canvas.height = imageElement.height;
      ctx.drawImage(imageElement, 0, 0);
      
      const imageDataBase64 = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
      
      // Base64 데이터 유효성 검사
      if (!imageDataBase64 || imageDataBase64.length < 1000) {
        throw new Error('이미지 변환에 실패했습니다.');
      }
      
      console.log(`📤 이미지 전송: ${imageElement.width}x${imageElement.height}, ${Math.round(imageDataBase64.length * 0.75 / 1024)}KB`);
      
      // Base64를 Blob으로 변환 (에러 처리 추가)
      let byteCharacters, byteArray, blob;
      try {
        byteCharacters = atob(imageDataBase64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        byteArray = new Uint8Array(byteNumbers);
        blob = new Blob([byteArray], { type: 'image/jpeg' });
      } catch (error) {
        throw new Error(`이미지 데이터 변환 실패: ${error.message}`);
      }
      
      // FormData 생성
      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');
      if (wallAngle) {
        formData.append('wall_angle', wallAngle);
      }
      
      // SSE를 사용한 실시간 진행상황 수신
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        let result = null;
        let lastProcessedLength = 0;
        
        xhr.open('POST', `${API_URL}/api/analyze-stream`);  // 스트림 엔드포인트 사용
        
        xhr.onreadystatechange = function() {
          if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
              if (result) {
                resolve(result);
              } else {
                reject(new Error('서버 응답을 받지 못했습니다.'));
              }
            } else {
              reject(new Error(`서버 분석 실패 (${xhr.status}): ${xhr.responseText}`));
            }
          }
        };
        
        // SSE 메시지 처리
        xhr.onprogress = function(event) {
          const newData = event.target.responseText.substring(lastProcessedLength);
          lastProcessedLength = event.target.responseText.length;
          
          const lines = newData.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                console.log(`📊 진행상황: ${data.message} (${data.progress}%)`);
                
                // 진행상황을 전역으로 전송 (App.jsx에서 받을 수 있도록)
                if (window.updateAnalysisProgress) {
                  window.updateAnalysisProgress(data);
                }
                
                // 최종 결과 처리
                if (data.step === 'complete' && data.problems) {
                  result = {
                    problems: data.problems,
                    statistics: data.statistics,
                    annotated_image_base64: data.annotated_image_base64,
                    message: data.message
                  };
                }
              } catch (e) {
                console.log('JSON 파싱 실패:', e, line);
              }
            }
          }
        };
        
        xhr.send(formData);
      });
      
      // 응답 데이터 유효성 검사
      if (!result || !result.problems || !Array.isArray(result.problems)) {
        console.error('서버 응답:', result);
        throw new Error('서버 응답 형식이 올바르지 않습니다.');
      }
      
      console.log(`✅ 서버 분석 완료: ${result.problems.length}개 문제`);
      
      // 백엔드 응답을 그대로 반환 (이미 올바른 형식)
      return result;
      
    } catch (error) {
      console.error('❌ 서버 분석 실패:', error);
      // 더 구체적인 에러 메시지 제공
      if (error.message.includes('Failed to fetch')) {
        throw new Error('네트워크 연결을 확인해주세요.');
      } else if (error.message.includes('404')) {
        throw new Error('서버 엔드포인트를 찾을 수 없습니다.');
      } else if (error.message.includes('500')) {
        throw new Error('서버 내부 오류가 발생했습니다.');
      }
      throw error;
    }
  }

  /**
   * 서버 CLIP API로 색상 분석
   */
  async analyzeColorsWithServerCLIP(imageElement, holds) {
    try {
      console.log('🎨 서버 CLIP API로 색상 분석 중...');
      
      // 이미지를 Base64로 변환 (고품질)
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // 원본 크기 유지 (너무 크면 리사이즈)
      let targetWidth = imageElement.width;
      let targetHeight = imageElement.height;
      
      // 최대 크기 제한 (메모리 절약)
      const maxSize = 2048;
      if (targetWidth > maxSize || targetHeight > maxSize) {
        const ratio = Math.min(maxSize / targetWidth, maxSize / targetHeight);
        targetWidth = Math.round(targetWidth * ratio);
        targetHeight = Math.round(targetHeight * ratio);
      }
      
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      
      // 고품질 렌더링
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(imageElement, 0, 0, targetWidth, targetHeight);
      
      // 고품질 JPEG로 변환 (품질 0.95)
      const imageDataBase64 = canvas.toDataURL('image/jpeg', 0.95).split(',')[1];
      
      console.log(`📤 이미지 전송: ${targetWidth}x${targetHeight}, ${Math.round(imageDataBase64.length/1024)}KB`);
      
      // 서버 CLIP API 호출
      const response = await fetch(`${API_URL}/api/analyze-colors`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          holds: holds,
          image_data_base64: imageDataBase64
        })
      });
      
      if (!response.ok) {
        throw new Error(`Server CLIP API error: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        console.log(`✅ 서버 CLIP: ${result.colored_holds.length}개 홀드 색상 분석 완료`);
        return result.colored_holds;
      } else {
        throw new Error('Server CLIP API returned error');
      }
      
    } catch (error) {
      console.error('❌ 서버 CLIP API 실패:', error);
      console.log('⚠️ Mock 색상 분석으로 전환');
      return this.analyzeColorsMock(holds);
    }
  }

  /**
   * 홀드 영역 추출
   */
  extractHoldRegion(imageElement, hold, targetSize) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = targetSize;
    canvas.height = targetSize;
    
    // 홀드 영역만 잘라서 그리기
    ctx.drawImage(
      imageElement,
      hold.x, hold.y, hold.width, hold.height,
      0, 0, targetSize, targetSize
    );
    
    return canvas;
  }

  /**
   * CLIP 특징 벡터로 색상 결정
   */
  determineColorFromFeatures(features) {
    // 특징 벡터의 통계로 색상 추정 (간단한 휴리스틱)
    const sum = Array.from(features).reduce((a, b) => a + b, 0);
    const avg = sum / features.length;
    
    const colors = [
      'red', 'blue', 'yellow', 'green', 'purple', 
      'orange', 'pink', 'white', 'black', 'gray'
    ];
    
    // 특징 벡터 평균값으로 색상 매핑
    const index = Math.abs(Math.floor(avg * 1000)) % colors.length;
    return colors[index];
  }

  /**
   * 모의 홀드 감지
   */
  detectHoldsMock(imageElement) {
    console.log('🔍 모의 홀드 감지 중...');
    
    const holds = [];
    const numHolds = 8 + Math.floor(Math.random() * 8); // 8-16개
    
    for (let i = 0; i < numHolds; i++) {
      holds.push({
        x: Math.random() * imageElement.width * 0.8,
        y: Math.random() * imageElement.height * 0.8,
        width: 30 + Math.random() * 40,
        height: 30 + Math.random() * 40,
        confidence: 0.7 + Math.random() * 0.3
      });
    }
    
    return holds;
  }

  /**
   * 모의 색상 분석
   */
  analyzeColorsMock(holds) {
    console.log('🎨 모의 색상 분석 중...');
    
    const colors = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'pink'];
    
    return holds.map(hold => ({
      ...hold,
      color: colors[Math.floor(Math.random() * colors.length)]
    }));
  }

  /**
   * 홀드를 색상별로 그룹화
   */
  groupByColor(holds) {
    const groups = {};
    
    for (const hold of holds) {
      if (!groups[hold.color]) {
        groups[hold.color] = [];
      }
      groups[hold.color].push(hold);
    }
    
    return groups;
  }

  /**
   * 색상 그룹에서 문제 생성
   */
  generateProblems(colorGroups) {
    const problems = [];
    let problemId = 1;
    
    for (const [color, holds] of Object.entries(colorGroups)) {
      if (holds.length >= 3) {
        const avgConfidence = holds.reduce((sum, h) => sum + h.confidence, 0) / holds.length;
        
        problems.push({
          id: problemId++,
          name: `${(color || 'unknown').toUpperCase()} 루트`,
          color: color || 'unknown',
          difficulty: this.calculateDifficulty(holds),
          type: this.guessType(holds),
          confidence: avgConfidence,
          holds: holds.map(h => ({
            x: Math.round(h.x),
            y: Math.round(h.y),
            width: Math.round(h.width),
            height: Math.round(h.height),
            color: h.color || 'unknown',
            confidence: h.confidence
          })),
          statistics: {
            total_holds: holds.length,
            avg_confidence: avgConfidence.toFixed(2)
          }
        });
      }
    }
    
    return problems;
  }

  /**
   * 난이도 계산
   */
  calculateDifficulty(holds) {
    const count = holds.length;
    const avgY = holds.reduce((sum, h) => sum + h.y, 0) / holds.length;
    
    if (count <= 4) return 'V1-V2';
    if (count <= 7) return 'V3-V4';
    if (count <= 10) return 'V5-V6';
    return 'V7+';
  }

  /**
   * 문제 유형 추측
   */
  guessType(holds) {
    const types = ['Balance', 'Power', 'Technique', 'Endurance', 'Coordination'];
    return types[Math.floor(Math.random() * types.length)];
  }

  /**
   * 전체 분석 프로세스
   */
  async analyzeImage(imageFile, wallAngle = null) {
    try {
      console.log('🚀 서버 사이드 AI 분석 시작...');
      
      // 이미지 로드
      const imageElement = await this.loadImage(imageFile);
      
      // 🚀 서버 사이드 전체 분석 (YOLO + 마스크 + CLIP)
      const serverResult = await this.analyzeWithServerSide(imageElement, wallAngle);
      
      // 서버에서 이미 완성된 결과를 그대로 반환
      console.log('✅ 서버 사이드 분석 완료!', serverResult);
      return serverResult;
      
    } catch (error) {
      console.error('❌ 서버 사이드 분석 실패:', error);
      throw error;
    }
  }

  /**
   * 이미지 로드 헬퍼
   */
  loadImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('이미지 로드 실패'));
      
      if (file instanceof File || file instanceof Blob) {
        img.src = URL.createObjectURL(file);
      } else {
        reject(new Error('유효하지 않은 이미지 파일'));
      }
    });
  }
}

export default ClientAIAnalyzer;
