// 🚀 완전 비동기 통합 이미지 분석 시스템

const API_URL = import.meta.env.VITE_API_URL || 'https://climbmate.store'

class AsyncImageAnalyzer {
  constructor() {
    this.onnxRuntime = null
    this.yoloModel = null
    this.isYoloInitialized = false
  }

  /**
   * 완전 비동기 3단계 분석 프로세스
   * 1단계: YOLO (브라우저) → 홀드 감지
   * 2단계: CLIP (백엔드) → 색상 분석  
   * 3단계: GPT-4 (API) → 문제 분석
   */
  async analyzeImageAsync(imageFile, wallAngle = null, onProgress = null) {
    try {
      console.log('🚀 완전 비동기 분석 시작')
      
      // 이미지 로드
      const imageElement = await this.loadImage(imageFile)
      this.updateProgress(onProgress, 5, '📸 이미지 로드 완료')
      
      // 1단계: YOLO 홀드 감지 (브라우저에서 실행)
      console.log('🔍 1단계: YOLO 홀드 감지 시작')
      this.updateProgress(onProgress, 10, '🔍 홀드 감지 중...')
      
      const holdData = await this.detectHoldsWithYOLO(imageElement)
      this.updateProgress(onProgress, 30, `✅ ${holdData.length}개 홀드 감지 완료`)
      
      if (holdData.length === 0) {
        throw new Error('홀드를 감지하지 못했습니다')
      }
      
      // 2단계: CLIP 색상 분석 (백엔드 비동기 작업)
      console.log('🎨 2단계: CLIP 색상 분석 시작')
      this.updateProgress(onProgress, 40, '🎨 색상 분석 중...')
      
      const coloredHolds = await this.analyzeColorsWithCLIP(imageElement, holdData)
      this.updateProgress(onProgress, 60, `✅ 색상 분석 완료`)
      
      // 문제 그룹핑
      const problems = this.groupHoldsByColor(coloredHolds)
      this.updateProgress(onProgress, 70, `✅ ${Object.keys(problems).length}개 문제 그룹 생성`)
      
      // 3단계: GPT-4 문제 분석 (병렬 API 호출)
      console.log('🤖 3단계: GPT-4 문제 분석 시작')
      this.updateProgress(onProgress, 80, '🤖 AI 문제 분석 중...')
      
      const analyzedProblems = await this.analyzeProblemsWithGPT4(imageElement, problems, wallAngle)
      this.updateProgress(onProgress, 95, '✅ AI 분석 완료')
      
      // 최종 결과 구성
      const result = {
        problems: analyzedProblems,
        statistics: this.calculateStatistics(holdData, analyzedProblems),
        hold_data: coloredHolds,
        annotated_image: await this.generateAnnotatedImage(imageElement, coloredHolds)
      }
      
      this.updateProgress(onProgress, 100, '✅ 분석 완료!')
      console.log('🎉 완전 비동기 분석 완료')
      
      return result
      
    } catch (error) {
      console.error('❌ 비동기 분석 실패:', error)
      this.updateProgress(onProgress, 0, `❌ 분석 실패: ${error.message}`)
      throw error
    }
  }

  /**
   * 1단계: YOLO 홀드 감지 (브라우저에서 실행)
   */
  async detectHoldsWithYOLO(imageElement) {
    try {
      // YOLO 모델 초기화 (한 번만)
      if (!this.isYoloInitialized) {
        await this.initializeYOLO()
      }
      
      // 홀드 감지 실행
      const detections = await this.runYOLODetection(imageElement)
      
      // 홀드 데이터 변환
      const holdData = detections.map((detection, index) => ({
        id: index,
        center: [detection.x + detection.width/2, detection.y + detection.height/2],
        area: detection.width * detection.height,
        bbox: [detection.x, detection.y, detection.width, detection.height],
        confidence: detection.confidence
      }))
      
      return holdData
      
    } catch (error) {
      console.error('❌ YOLO 홀드 감지 실패:', error)
      throw new Error(`홀드 감지 실패: ${error.message}`)
    }
  }

  /**
   * 2단계: CLIP 색상 분석 (백엔드 비동기 작업)
   */
  async analyzeColorsWithCLIP(imageElement, holdData) {
    try {
      // 이미지를 Base64로 변환
      const imageBase64 = await this.imageToBase64(imageElement)
      
      // 백엔드 CLIP 분석 작업 시작
      const taskResponse = await fetch(`${API_URL}/api/analyze-colors-async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: imageBase64,
          hold_data: holdData
        })
      })
      
      if (!taskResponse.ok) {
        throw new Error(`CLIP 분석 시작 실패: ${taskResponse.status}`)
      }
      
      const { task_id } = await taskResponse.json()
      
      // 작업 완료까지 폴링
      return await this.pollTaskCompletion(task_id, 'CLIP 색상 분석')
      
    } catch (error) {
      console.error('❌ CLIP 색상 분석 실패:', error)
      throw new Error(`색상 분석 실패: ${error.message}`)
    }
  }

  /**
   * 3단계: GPT-4 문제 분석 (병렬 API 호출)
   */
  async analyzeProblemsWithGPT4(imageElement, problems, wallAngle) {
    try {
      const imageBase64 = await this.imageToBase64(imageElement)
      
      // 모든 문제를 병렬로 GPT-4 분석
      const analysisPromises = Object.entries(problems).map(async ([color, holds]) => {
        if (holds.length < 3) return null // 최소 3개 홀드 이상만 분석
        
        try {
          const analysis = await this.callGPT4API(imageBase64, holds, wallAngle)
          return {
            id: color,
            color_name: color,
            color_rgb: holds[0]?.dominant_rgb || [128, 128, 128],
            holds: holds,
            hold_count: holds.length,
            analysis: analysis
          }
        } catch (error) {
          console.error(`GPT-4 분석 실패 (${color}):`, error)
          return {
            id: color,
            color_name: color,
            color_rgb: holds[0]?.dominant_rgb || [128, 128, 128],
            holds: holds,
            hold_count: holds.length,
            analysis: null
          }
        }
      })
      
      // 모든 분석 완료 대기
      const results = await Promise.all(analysisPromises)
      
      // null 값 제거
      return results.filter(result => result !== null)
      
    } catch (error) {
      console.error('❌ GPT-4 문제 분석 실패:', error)
      throw new Error(`문제 분석 실패: ${error.message}`)
    }
  }

  /**
   * 백엔드 작업 완료까지 폴링
   */
  async pollTaskCompletion(taskId, taskName) {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const response = await fetch(`${API_URL}/api/task-status/${taskId}`)
          const data = await response.json()
          
          if (data.status === 'SUCCESS') {
            resolve(data.result)
          } else if (data.status === 'FAILURE') {
            reject(new Error(data.error || `${taskName} 실패`))
          } else {
            // 진행 중이면 1초 후 다시 확인
            setTimeout(poll, 1000)
          }
        } catch (error) {
          reject(error)
        }
      }
      
      poll()
    })
  }

  /**
   * GPT-4 API 호출
   */
  async callGPT4API(imageBase64, holds, wallAngle) {
    const response = await fetch(`${API_URL}/api/gpt4-analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: imageBase64,
        holds: holds,
        wall_angle: wallAngle
      })
    })
    
    if (!response.ok) {
      throw new Error(`GPT-4 API 호출 실패: ${response.status}`)
    }
    
    return await response.json()
  }

  /**
   * 진행률 업데이트
   */
  updateProgress(onProgress, progress, message) {
    if (onProgress && typeof onProgress === 'function') {
      onProgress({ progress, message })
    }
    console.log(`📊 ${progress}%: ${message}`)
  }

  /**
   * 이미지 로드
   */
  async loadImage(imageFile) {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error('이미지 로드 실패'))
      img.src = URL.createObjectURL(imageFile)
    })
  }

  /**
   * 이미지를 Base64로 변환
   */
  async imageToBase64(imageElement) {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    canvas.width = imageElement.width
    canvas.height = imageElement.height
    ctx.drawImage(imageElement, 0, 0)
    
    return canvas.toDataURL('image/jpeg', 0.8).split(',')[1]
  }

  /**
   * 홀드를 색상별로 그룹핑
   */
  groupHoldsByColor(holds) {
    const groups = {}
    holds.forEach(hold => {
      const color = hold.color_name || 'unknown'
      if (!groups[color]) {
        groups[color] = []
      }
      groups[color].push(hold)
    })
    return groups
  }

  /**
   * 통계 계산
   */
  calculateStatistics(holdData, problems) {
    return {
      total_holds: holdData.length,
      total_problems: problems.length,
      analyzable_problems: problems.filter(p => p.hold_count >= 3).length
    }
  }

  /**
   * 주석 달린 이미지 생성
   */
  async generateAnnotatedImage(imageElement, holds) {
    // 간단한 주석 이미지 생성 로직
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    canvas.width = imageElement.width
    canvas.height = imageElement.height
    ctx.drawImage(imageElement, 0, 0)
    
    // 홀드에 원 그리기
    holds.forEach((hold, index) => {
      ctx.beginPath()
      ctx.arc(hold.center[0], hold.center[1], 10, 0, 2 * Math.PI)
      ctx.fillStyle = `hsl(${index * 60}, 70%, 50%)`
      ctx.fill()
    })
    
    return canvas.toDataURL('image/jpeg', 0.8)
  }

  /**
   * YOLO 모델 초기화 (기존 코드 재사용)
   */
  async initializeYOLO() {
    // 기존 YOLO 초기화 로직 재사용
    console.log('🔄 YOLO 모델 초기화 중...')
    // ... 기존 초기화 코드 ...
    this.isYoloInitialized = true
    console.log('✅ YOLO 모델 초기화 완료')
  }

  /**
   * YOLO 감지 실행 (기존 코드 재사용)
   */
  async runYOLODetection(imageElement) {
    // 기존 YOLO 감지 로직 재사용
    console.log('🔍 YOLO 홀드 감지 실행 중...')
    // ... 기존 감지 코드 ...
    return [] // 임시 반환값
  }
}

// 전역 인스턴스 생성
const asyncAnalyzer = new AsyncImageAnalyzer()

// 기존 함수와 호환성 유지
async function analyzeImage(imageFile, wallAngle = null) {
  return await asyncAnalyzer.analyzeImageAsync(imageFile, wallAngle)
}

export { AsyncImageAnalyzer, analyzeImage }
export default asyncAnalyzer
