import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'https://climbmate.store'

/**
 * 이미지 분석 API
 */
export const analyzeImage = async (imageFile, options = {}) => {
  // 이 함수는 나중에 App.jsx에서 이동할 예정
  console.log('analyzeImage API - 준비 중')
}

/**
 * GPT-4 상태 확인
 */
export const checkGpt4Status = async () => {
  try {
    const response = await axios.get(`${API_URL}/api/gpt4-status`)
    return response.data
  } catch (error) {
    console.error('❌ GPT-4 상태 확인 실패:', error)
    throw error
  }
}

/**
 * GPT-4 테스트
 */
export const testGpt4 = async () => {
  try {
    const response = await axios.post(`${API_URL}/api/test-gpt4`)
    return response.data
  } catch (error) {
    console.error('❌ GPT-4 테스트 실패:', error)
    throw error
  }
}

/**
 * GPT-4 결과를 훈련 데이터로 변환
 */
export const convertGpt4ToTraining = async () => {
  try {
    const response = await axios.post(`${API_URL}/api/convert-gpt4`)
    return response.data
  } catch (error) {
    console.error('❌ GPT-4 변환 실패:', error)
    throw error
  }
}

