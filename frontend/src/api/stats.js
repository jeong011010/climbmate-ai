import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'https://climbmate.store'

/**
 * 통계 데이터 조회
 */
export const getStats = async () => {
  try {
    const response = await axios.get(`${API_URL}/api/stats`)
    return response.data
  } catch (error) {
    console.log('통계 API 사용 불가 (정상)')
    throw error
  }
}

/**
 * 모델 재훈련
 */
export const trainModel = async () => {
  try {
    await axios.post(`${API_URL}/api/train`)
  } catch (error) {
    console.error('❌ 모델 훈련 실패:', error)
    throw error
  }
}

