/**
 * 🚀 클라이언트 사이드 AI 처리
 * 사용자 브라우저에서 직접 AI 모델 실행 (TensorFlow.js 의존성 없이 임시 구현)
 */
// import * as tf from '@tensorflow/tfjs'; // TODO: TensorFlow.js 의존성 추가 후 활성화

class ClientAIAnalyzer {
  constructor() {
    this.yoloModel = null;
    this.clipModel = null;
    this.isLoaded = false;
  }

  /**
   * AI 모델들을 사용자 브라우저에 로드
   */
  async loadModels() {
    try {
      console.log('🚀 클라이언트 AI 모델 로딩 시작...');
      
      // YOLO 모델 로드 (WebAssembly 버전)
      this.yoloModel = await tf.loadLayersModel('/models/yolo-wasm/model.json');
      console.log('✅ YOLO 모델 로드 완료');
      
      // CLIP 모델 로드 (WebAssembly 버전)
      this.clipModel = await tf.loadLayersModel('/models/clip-wasm/model.json');
      console.log('✅ CLIP 모델 로드 완료');
      
      this.isLoaded = true;
      console.log('🎉 모든 AI 모델 로드 완료!');
      
    } catch (error) {
      console.error('❌ 클라이언트 AI 모델 로드 실패:', error);
      throw error;
    }
  }

  /**
   * 사용자 브라우저에서 직접 홀드 감지
   */
  async detectHolds(imageElement) {
    if (!this.isLoaded) {
      await this.loadModels();
    }

    try {
      console.log('🔍 클라이언트에서 홀드 감지 중...');
      
      // 이미지를 텐서로 변환
      const imageTensor = tf.browser.fromPixels(imageElement);
      const resized = tf.image.resizeBilinear(imageTensor, [416, 416]);
      const normalized = resized.div(255.0);
      const batched = normalized.expandDims(0);

      // YOLO 모델로 홀드 감지
      const predictions = this.yoloModel.predict(batched);
      
      // 결과 처리
      const holds = await this.processYOLOPredictions(predictions);
      
      // 메모리 정리
      imageTensor.dispose();
      resized.dispose();
      normalized.dispose();
      batched.dispose();
      predictions.dispose();
      
      console.log(`✅ ${holds.length}개 홀드 감지 완료`);
      return holds;
      
    } catch (error) {
      console.error('❌ 홀드 감지 실패:', error);
      throw error;
    }
  }

  /**
   * 사용자 브라우저에서 직접 색상 분석
   */
  async analyzeColors(imageElement, holds) {
    if (!this.isLoaded) {
      await this.loadModels();
    }

    try {
      console.log('🎨 클라이언트에서 색상 분석 중...');
      
      const colorGroups = [];
      
      for (const hold of holds) {
        // 홀드 영역 추출
        const holdImage = this.extractHoldRegion(imageElement, hold);
        
        // CLIP으로 색상 분석
        const color = await this.analyzeColorWithCLIP(holdImage);
        
        // 색상 그룹에 추가
        this.addToColorGroup(colorGroups, color, hold);
        
        // 메모리 정리
        holdImage.dispose();
      }
      
      console.log(`✅ ${colorGroups.length}개 색상 그룹 분석 완료`);
      return colorGroups;
      
    } catch (error) {
      console.error('❌ 색상 분석 실패:', error);
      throw error;
    }
  }

  /**
   * 전체 분석 프로세스 (사용자 브라우저에서 실행)
   */
  async analyzeImage(imageFile) {
    try {
      console.log('🚀 클라이언트 사이드 AI 분석 시작...');
      
      // 이미지 로드
      const imageElement = await this.loadImage(imageFile);
      
      // 홀드 감지
      const holds = await this.detectHolds(imageElement);
      
      // 색상 분석
      const colorGroups = await this.analyzeColors(imageElement, holds);
      
      // 문제 생성
      const problems = this.generateProblems(colorGroups);
      
      const result = {
        problems: problems,
        statistics: {
          total_holds: holds.length,
          total_problems: problems.length,
          color_groups: colorGroups
        },
        analysis_method: 'client_side_ai'
      };
      
      console.log('✅ 클라이언트 사이드 분석 완료!');
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
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * YOLO 예측 결과 처리
   */
  async processYOLOPredictions(predictions) {
    // YOLO 출력을 홀드 좌표로 변환
    const boxes = predictions[0].dataSync();
    const scores = predictions[1].dataSync();
    
    const holds = [];
    for (let i = 0; i < boxes.length; i += 4) {
      if (scores[i / 4] > 0.5) { // 신뢰도 임계값
        holds.push({
          x: boxes[i],
          y: boxes[i + 1],
          width: boxes[i + 2],
          height: boxes[i + 3],
          confidence: scores[i / 4]
        });
      }
    }
    
    return holds;
  }

  /**
   * CLIP으로 색상 분석
   */
  async analyzeColorWithCLIP(imageTensor) {
    // CLIP 모델로 색상 분석
    const features = this.clipModel.predict(imageTensor);
    
    // 색상 텍스트와 비교
    const colorTexts = ['red', 'blue', 'yellow', 'green', 'purple', 'black', 'white'];
    const similarities = [];
    
    for (const colorText of colorTexts) {
      const textEmbedding = await this.encodeText(colorText);
      const similarity = tf.losses.cosineDistance(features, textEmbedding);
      similarities.push({
        color: colorText,
        similarity: similarity.dataSync()[0]
      });
    }
    
    // 가장 유사한 색상 반환
    const bestMatch = similarities.reduce((prev, current) => 
      prev.similarity > current.similarity ? prev : current
    );
    
    return bestMatch.color;
  }

  /**
   * 텍스트 인코딩
   */
  async encodeText(text) {
    // 간단한 텍스트 인코딩 (실제로는 CLIP의 텍스트 인코더 사용)
    const tokens = text.split('').map(char => char.charCodeAt(0));
    return tf.tensor(tokens);
  }

  /**
   * 홀드 영역 추출
   */
  extractHoldRegion(imageElement, hold) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = hold.width;
    canvas.height = hold.height;
    
    ctx.drawImage(
      imageElement,
      hold.x, hold.y, hold.width, hold.height,
      0, 0, hold.width, hold.height
    );
    
    return tf.browser.fromPixels(canvas);
  }

  /**
   * 색상 그룹에 홀드 추가
   */
  addToColorGroup(colorGroups, color, hold) {
    let group = colorGroups.find(g => g.color === color);
    if (!group) {
      group = { color, holds: [] };
      colorGroups.push(group);
    }
    group.holds.push(hold);
  }

  /**
   * 문제 생성
   */
  generateProblems(colorGroups) {
    const problems = [];
    
    // 각 색상 그룹으로 문제 생성
    colorGroups.forEach(group => {
      if (group.holds.length >= 3) {
        problems.push({
          name: `${group.color} 루트`,
          difficulty: this.calculateDifficulty(group.holds),
          holds: group.holds,
          color: group.color
        });
      }
    });
    
    return problems;
  }

  /**
   * 난이도 계산
   */
  calculateDifficulty(holds) {
    // 홀드 개수와 분포로 난이도 계산
    const count = holds.length;
    if (count <= 5) return '쉬움';
    if (count <= 10) return '보통';
    return '어려움';
  }
}

export default ClientAIAnalyzer;
