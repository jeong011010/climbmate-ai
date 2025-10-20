// 🚀 클라이언트 사이드 AI 분석 (비동기 작업 큐 방식)

const API_URL = import.meta.env.VITE_API_URL || 'https://climbmate.store'

class ClientAI {
  constructor() {
    this.onnxRuntime = null
    this.yoloModel = null
    this.isInitialized = false
  }

  /**
   * 이미지 분석 (비동기 작업 큐 방식)
   */
  async analyzeImage(imageElement, wallAngle = null) {
    return await this.analyzeWithServerSide(imageElement, wallAngle)
  }

  /**
   * 서버 사이드 전체 분석 (내부 구현)
   */
  async analyzeWithServerSide(imageElement, wallAngle = null) {
    try {
      console.log('🚀 서버 사이드 전체 분석 시작...')
      
      // 이미지를 Base64로 변환
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      canvas.width = imageElement.width
      canvas.height = imageElement.height
      ctx.drawImage(imageElement, 0, 0)
      
      const imageData = canvas.toDataURL('image/jpeg', 0.8)
      const base64Data = imageData.split(',')[1]
      
      // Blob 생성
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'image/jpeg' })
      
      // FormData 생성
      const formData = new FormData()
      formData.append('file', blob, 'image.jpg')
      if (wallAngle) {
        formData.append('wall_angle', wallAngle)
      }
      
      console.log('📦 FormData 생성 완료:', {
        fileSize: blob.size,
        fileType: blob.type,
        wallAngle: wallAngle
      })
      
      // 🚀 비동기 작업 큐 방식으로 분석
      return new Promise(async (resolve, reject) => {
        try {
          // 1단계: 분석 작업 시작
          console.log('🚀 비동기 분석 작업 시작')
          
          const startResponse = await fetch(`${API_URL}/api/analyze-stream`, {
            method: 'POST',
            body: formData
          })
          
          if (!startResponse.ok) {
            throw new Error(`작업 시작 실패: ${startResponse.status}`)
          }
          
          const startData = await startResponse.json()
          const taskId = startData.task_id
          console.log('✅ 작업 시작됨, Task ID:', taskId)
          
          // 2단계: 진행률 폴링
          const pollStatus = async () => {
            try {
              const statusResponse = await fetch(`${API_URL}/api/analyze-status/${taskId}`)
              if (!statusResponse.ok) {
                throw new Error(`상태 확인 실패: ${statusResponse.status}`)
              }
              
              const statusData = await statusResponse.json()
              console.log('📊 진행률:', statusData.progress + '%', statusData.message)
              
              // UI 업데이트
              if (typeof window.updateAnalysisProgress === 'function') {
                window.updateAnalysisProgress({
                  message: statusData.message,
                  progress: statusData.progress,
                  step: statusData.step
                })
              }
              
              if (statusData.status === 'SUCCESS') {
                console.log('✅ 분석 완료!')
                resolve(statusData.result)
              } else if (statusData.status === 'FAILURE') {
                reject(new Error(statusData.message || '분석 실패'))
              } else {
                // 진행 중이면 1초 후 다시 확인
                setTimeout(pollStatus, 1000)
              }
    } catch (error) {
              reject(error)
            }
          }
          
          // 폴링 시작
          pollStatus()
          
        } catch (error) {
          reject(error)
        }
      })
      
    } catch (error) {
      console.error('❌ 서버 분석 실패:', error)
      throw error
    }
  }

  /**
   * 전체 분석 프로세스
   */
  async analyzeImage(imageFile, wallAngle = null) {
    try {
      console.log('🚀 서버 사이드 AI 분석 시작...')
      
      // 이미지 로드
      const imageElement = await this.loadImage(imageFile)
      
      // 🚀 서버 사이드 전체 분석 (비동기 작업 큐)
      const serverResult = await this.analyzeWithServerSide(imageElement, wallAngle)
      
      // 서버에서 이미 완성된 결과를 그대로 반환
      console.log(`✅ 서버 분석 완료: ${serverResult.problems?.length || 0}개 문제`)
      return serverResult
      
    } catch (error) {
      console.error('❌ 서버 사이드 분석 실패:', error)
      throw error
    }
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
}

// 전역 인스턴스 생성
const clientAI = new ClientAI()

// 기존 함수들 유지 (호환성)
async function analyzeImage(imageFile, wallAngle = null) {
  return await clientAI.analyzeImage(imageFile, wallAngle)
}

export { ClientAI, analyzeImage }
export default ClientAI
