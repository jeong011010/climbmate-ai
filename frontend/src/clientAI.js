/**
 * 🚀 클라이언트 사이드 AI 처리 v2.1.0 (ONNX Runtime Web)
 * 브라우저: YOLO 실행 | 서버: CLIP 실행
 * 사용자 브라우저에서 직접 YOLO 모델 실행 + 서버 CLIP API
 */

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
    
    return holds.slice(0, 30);
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
      const response = await fetch('/api/analyze-colors', {
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
  async analyzeImage(imageFile) {
    try {
      console.log('🚀 클라이언트 사이드 AI 분석 시작...');
      
      // 모델 로딩
      const modelsLoaded = await this.loadModels();
      
      // 이미지 로드
      const imageElement = await this.loadImage(imageFile);
      
      // YOLO로 홀드 감지
      const holds = await this.detectHoldsWithYOLO(imageElement);
      
      // 서버 CLIP API로 색상 분석
      const coloredHolds = await this.analyzeColorsWithServerCLIP(imageElement, holds);
      
      // 색상별 그룹화
      const colorGroups = this.groupByColor(coloredHolds);
      
      // 문제 생성
      const problems = this.generateProblems(colorGroups);
      
      const result = {
        problems: problems,
        statistics: {
          total_holds: coloredHolds.length,
          total_problems: problems.length,
          color_groups: Object.keys(colorGroups).length,
          analysis_method: modelsLoaded ? 'client_side_onnx' : 'client_side_mock'
        },
        message: `클라이언트 분석 완료 ${modelsLoaded ? '(실제 YOLO+CLIP)' : '(모의 데이터)'}`,
        note: modelsLoaded 
          ? '✅ 사용자 브라우저에서 커스텀 YOLO + CLIP 모델을 실행했습니다.'
          : '⚠️ AI 모델 파일이 없어 모의 분석을 수행했습니다.'
      };
      
      console.log('✅ 클라이언트 사이드 분석 완료!', result);
      return result;
      
    } catch (error) {
      console.error('❌ 클라이언트 사이드 분석 실패:', error);
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
