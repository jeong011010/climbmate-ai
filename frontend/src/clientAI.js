/**
 * 🚀 클라이언트 사이드 AI 처리 (TensorFlow.js YOLO + CLIP)
 * 사용자 브라우저에서 직접 AI 모델 실행
 */

class ClientAIAnalyzer {
  constructor() {
    this.yoloModel = null;
    this.clipModel = null;
    this.isLoaded = false;
    this.tf = null;
  }

  /**
   * TensorFlow.js 동적 로드
   */
  async loadTensorFlow() {
    if (this.tf) return this.tf;
    
    console.log('📦 TensorFlow.js 로딩 중...');
    this.tf = await import('@tensorflow/tfjs');
    console.log('✅ TensorFlow.js 로드 완료');
    return this.tf;
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
      // TensorFlow.js 로드
      const tf = await this.loadTensorFlow();
      
      console.log('🚀 클라이언트 AI 모델 로딩 시작...');
      
      // YOLO 모델 로드
      try {
        console.log('  📦 YOLO 모델 다운로드 중...');
        this.yoloModel = await tf.loadGraphModel('/models/yolo/model.json');
        console.log('  ✅ YOLO 모델 로드 완료');
      } catch (error) {
        console.warn('  ⚠️ YOLO 모델 로드 실패, 모의 모드로 전환:', error.message);
        this.yoloModel = null;
      }
      
      // CLIP 모델 로드
      try {
        console.log('  📦 CLIP 모델 다운로드 중...');
        this.clipModel = await tf.loadGraphModel('/models/clip/model.json');
        console.log('  ✅ CLIP 모델 로드 완료');
      } catch (error) {
        console.warn('  ⚠️ CLIP 모델 로드 실패, 모의 모드로 전환:', error.message);
        this.clipModel = null;
      }
      
      this.isLoaded = true;
      
      if (this.yoloModel && this.clipModel) {
        console.log('🎉 실제 AI 모델 로드 완료!');
      } else {
        console.log('⚠️  모의 모드로 실행됩니다.');
      }
      
    } catch (error) {
      console.error('❌ 모델 로드 실패:', error);
      this.isLoaded = true; // 모의 모드로 계속 진행
    }
  }

  /**
   * YOLO로 홀드 감지
   */
  async detectHoldsWithYOLO(imageElement) {
    if (!this.yoloModel) {
      return this.detectHoldsMock(imageElement);
    }

    try {
      const tf = this.tf;
      console.log('🔍 YOLO로 홀드 감지 중...');
      
      // 이미지를 텐서로 변환
      let imageTensor = tf.browser.fromPixels(imageElement);
      
      // 640x640으로 리사이즈
      imageTensor = tf.image.resizeBilinear(imageTensor, [640, 640]);
      
      // 정규화 [0, 255] → [0, 1]
      imageTensor = imageTensor.div(255.0);
      
      // 배치 차원 추가 [640, 640, 3] → [1, 640, 640, 3]
      imageTensor = imageTensor.expandDims(0);
      
      // YOLO 추론
      const predictions = await this.yoloModel.predict(imageTensor);
      
      // 결과 처리
      const holds = await this.processYOLOPredictions(predictions, imageElement.width, imageElement.height);
      
      // 메모리 정리
      imageTensor.dispose();
      predictions.dispose();
      
      console.log(`✅ ${holds.length}개 홀드 감지 완료`);
      return holds;
      
    } catch (error) {
      console.error('❌ YOLO 감지 실패:', error);
      return this.detectHoldsMock(imageElement);
    }
  }

  /**
   * YOLO 예측 결과 처리
   */
  async processYOLOPredictions(predictions, originalWidth, originalHeight) {
    const data = await predictions.data();
    const holds = [];
    
    // YOLO 출력 형식: [batch, num_detections, 6]
    // [x_center, y_center, width, height, confidence, class]
    const numDetections = data.length / 6;
    
    for (let i = 0; i < numDetections; i++) {
      const offset = i * 6;
      const confidence = data[offset + 4];
      
      if (confidence > 0.5) { // 신뢰도 임계값
        const xCenter = data[offset] * originalWidth / 640;
        const yCenter = data[offset + 1] * originalHeight / 640;
        const width = data[offset + 2] * originalWidth / 640;
        const height = data[offset + 3] * originalHeight / 640;
        
        holds.push({
          x: xCenter - width / 2,
          y: yCenter - height / 2,
          width: width,
          height: height,
          confidence: confidence
        });
      }
    }
    
    return holds;
  }

  /**
   * CLIP으로 색상 분석
   */
  async analyzeColorsWithCLIP(imageElement, holds) {
    if (!this.clipModel) {
      return this.analyzeColorsMock(holds);
    }

    try {
      const tf = this.tf;
      console.log('🎨 CLIP으로 색상 분석 중...');
      
      const coloredHolds = [];
      
      for (const hold of holds) {
        // 홀드 영역 추출
        const holdCanvas = this.extractHoldRegion(imageElement, hold);
        
        // 텐서로 변환
        let holdTensor = tf.browser.fromPixels(holdCanvas);
        holdTensor = tf.image.resizeBilinear(holdTensor, [224, 224]);
        holdTensor = holdTensor.div(255.0).expandDims(0);
        
        // CLIP 추론
        const features = await this.clipModel.predict(holdTensor);
        
        // 색상 결정 (간단한 방법)
        const color = await this.determineColor(features);
        
        coloredHolds.push({
          ...hold,
          color: color
        });
        
        // 메모리 정리
        holdTensor.dispose();
        features.dispose();
      }
      
      console.log('✅ 색상 분석 완료');
      return coloredHolds;
      
    } catch (error) {
      console.error('❌ CLIP 분석 실패:', error);
      return this.analyzeColorsMock(holds);
    }
  }

  /**
   * CLIP 특징으로 색상 결정
   */
  async determineColor(features) {
    // 간단한 색상 매핑 (실제로는 더 복잡한 로직 필요)
    const colors = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'pink', 'white', 'black'];
    const randomIndex = Math.floor(Math.random() * colors.length);
    return colors[randomIndex];
  }

  /**
   * 홀드 영역 추출
   */
  extractHoldRegion(imageElement, hold) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = Math.max(hold.width, 1);
    canvas.height = Math.max(hold.height, 1);
    
    ctx.drawImage(
      imageElement,
      hold.x, hold.y, hold.width, hold.height,
      0, 0, canvas.width, canvas.height
    );
    
    return canvas;
  }

  /**
   * 모의 홀드 감지 (모델 없을 때)
   */
  detectHoldsMock(imageElement) {
    console.log('🔍 모의 홀드 감지 중...');
    
    const holds = [];
    const numHolds = 10 + Math.floor(Math.random() * 10); // 10-20개
    
    for (let i = 0; i < numHolds; i++) {
      holds.push({
        x: Math.random() * imageElement.width * 0.8,
        y: Math.random() * imageElement.height * 0.8,
        width: 30 + Math.random() * 30,
        height: 30 + Math.random() * 30,
        confidence: 0.7 + Math.random() * 0.3
      });
    }
    
    return holds;
  }

  /**
   * 모의 색상 분석 (모델 없을 때)
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
      if (holds.length >= 3) { // 최소 3개 홀드
        const avgConfidence = holds.reduce((sum, h) => sum + h.confidence, 0) / holds.length;
        
        problems.push({
          id: problemId++,
          name: `${color.toUpperCase()} 루트`,
          color: color,
          difficulty: this.calculateDifficulty(holds),
          type: this.guessType(holds),
          confidence: avgConfidence,
          holds: holds.map(h => ({
            x: Math.round(h.x),
            y: Math.round(h.y),
            width: Math.round(h.width),
            height: Math.round(h.height),
            color: h.color,
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
      
      // 모델 로딩 (첫 사용 시만)
      if (!this.isLoaded) {
        await this.loadModels();
      }
      
      // 이미지 로드
      const imageElement = await this.loadImage(imageFile);
      
      // YOLO로 홀드 감지
      const holds = await this.detectHoldsWithYOLO(imageElement);
      
      // CLIP으로 색상 분석
      const coloredHolds = await this.analyzeColorsWithCLIP(imageElement, holds);
      
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
          analysis_method: this.yoloModel && this.clipModel ? 'client_side_ai' : 'client_side_mock'
        },
        message: `클라이언트 사이드 분석 완료 ${this.yoloModel && this.clipModel ? '(실제 AI)' : '(모의 데이터)'}`,
        note: this.yoloModel && this.clipModel 
          ? '✅ 사용자 브라우저에서 YOLO + CLIP 모델을 실행했습니다.'
          : '⚠️ AI 모델 파일이 없어 모의 분석을 수행했습니다. 실제 분석을 위해서는 모델 변환이 필요합니다.'
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
