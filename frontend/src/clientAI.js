/**
 * 🚀 클라이언트 사이드 AI 처리 (간단한 색상 기반 분석)
 * 사용자 브라우저에서 직접 이미지 분석
 * TensorFlow.js 대신 Canvas API로 간단한 분석 수행
 */

class ClientAIAnalyzer {
  constructor() {
    this.isLoaded = false;
  }

  /**
   * 모델 로딩 시뮬레이션 (실제로는 Canvas API만 사용)
   */
  async loadModels() {
    console.log('🚀 클라이언트 AI 준비 중...');
    
    // 간단한 지연 (로딩 시뮬레이션)
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    this.isLoaded = true;
    console.log('✅ 클라이언트 AI 준비 완료!');
  }

  /**
   * 이미지에서 색상 기반 홀드 감지 (간단한 버전)
   */
  async detectHolds(imageElement) {
    console.log('🔍 홀드 감지 중...');
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // 이미지 크기 조정
    const maxSize = 640;
    let width = imageElement.naturalWidth || imageElement.width;
    let height = imageElement.naturalHeight || imageElement.height;
    
    if (width > maxSize || height > maxSize) {
      const scale = maxSize / Math.max(width, height);
      width *= scale;
      height *= scale;
    }
    
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(imageElement, 0, 0, width, height);
    
    // 이미지 데이터 가져오기
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    // 색상 기반 홀드 감지 (간단한 알고리즘)
    const holds = this.detectColorRegions(data, width, height);
    
    console.log(`✅ ${holds.length}개 홀드 감지 완료`);
    return holds;
  }

  /**
   * 색상 영역 감지 (간단한 알고리즘)
   */
  detectColorRegions(data, width, height) {
    const holds = [];
    const visited = new Set();
    const minHoldSize = 100; // 최소 픽셀 수
    
    // 그리드 샘플링 (성능 최적화)
    const step = 20;
    
    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const idx = (y * width + x) * 4;
        const key = `${x},${y}`;
        
        if (visited.has(key)) continue;
        
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        
        // 채도가 높은 색상만 감지 (홀드는 보통 밝은 색)
        const brightness = (r + g + b) / 3;
        const saturation = Math.max(r, g, b) - Math.min(r, g, b);
        
        if (saturation > 30 && brightness > 50) {
          // 홀드 후보 발견
          const hold = {
            x: x,
            y: y,
            width: step * 3,
            height: step * 3,
            color: this.rgbToColorName(r, g, b),
            rgb: { r, g, b },
            confidence: Math.min(saturation / 100, 1.0)
          };
          
          holds.push(hold);
          visited.add(key);
        }
      }
    }
    
    // 최대 20개 홀드로 제한
    return holds.slice(0, 20);
  }

  /**
   * RGB를 색상 이름으로 변환
   */
  rgbToColorName(r, g, b) {
    const colors = [
      { name: 'red', r: 255, g: 0, b: 0 },
      { name: 'blue', r: 0, g: 0, b: 255 },
      { name: 'yellow', r: 255, g: 255, b: 0 },
      { name: 'green', r: 0, g: 255, b: 0 },
      { name: 'purple', r: 128, g: 0, b: 128 },
      { name: 'orange', r: 255, g: 165, b: 0 },
      { name: 'pink', r: 255, g: 192, b: 203 },
      { name: 'white', r: 255, g: 255, b: 255 },
      { name: 'black', r: 0, g: 0, b: 0 }
    ];
    
    let minDist = Infinity;
    let bestColor = 'unknown';
    
    for (const color of colors) {
      const dist = Math.sqrt(
        Math.pow(r - color.r, 2) +
        Math.pow(g - color.g, 2) +
        Math.pow(b - color.b, 2)
      );
      
      if (dist < minDist) {
        minDist = dist;
        bestColor = color.name;
      }
    }
    
    return bestColor;
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
            x: h.x,
            y: h.y,
            width: h.width,
            height: h.height,
            color: h.color
          })),
          statistics: {
            total_holds: holds.length,
            avg_confidence: avgConfidence
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
      
      // 홀드 감지
      const holds = await this.detectHolds(imageElement);
      
      // 색상별 그룹화
      const colorGroups = this.groupByColor(holds);
      
      // 문제 생성
      const problems = this.generateProblems(colorGroups);
      
      const result = {
        problems: problems,
        statistics: {
          total_holds: holds.length,
          total_problems: problems.length,
          color_groups: Object.keys(colorGroups).length,
          analysis_method: 'client_side_simple'
        },
        message: '클라이언트 사이드 간단 분석 완료 (색상 기반)',
        note: '이 분석은 사용자 브라우저에서 간단한 색상 분석으로 수행되었습니다. 서버 AI보다 정확도는 낮지만, 서버 부담이 없고 빠릅니다.'
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
