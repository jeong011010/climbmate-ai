import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'https://climbmate.store'

/**
 * 색상 피드백 목록 조회
 */
export const getColorFeedbacks = async (limit = 300, offset = 0) => {
  try {
    const response = await axios.get(`${API_URL}/api/color-feedbacks`, {
      params: { limit, offset }
    })
    return response.data
  } catch (error) {
    console.error('❌ 피드백 조회 실패:', error)
    throw error
  }
}

/**
 * 색상 피드백 확인 (ML 학습용으로 확정)
 */
export const confirmColorFeedback = async (feedbackId) => {
  try {
    await axios.post(`${API_URL}/api/color-feedbacks/${feedbackId}/confirm`)
  } catch (error) {
    console.error('❌ 피드백 확인 실패:', error)
    throw error
  }
}

/**
 * 모든 미확인 색상 피드백 일괄 확인
 */
export const confirmAllColorFeedbacks = async () => {
  try {
    const response = await axios.post(`${API_URL}/api/color-feedbacks/confirm-all`)
    return response.data
  } catch (error) {
    console.error('❌ 일괄 확인 실패:', error)
    throw error
  }
}

/**
 * 색상 피드백 삭제
 */
export const deleteColorFeedback = async (feedbackId) => {
  try {
    await axios.delete(`${API_URL}/api/color-feedbacks/${feedbackId}`)
  } catch (error) {
    console.error('❌ 피드백 삭제 실패:', error)
    throw error
  }
}

/**
 * 색상 피드백 전체 삭제
 */
export const deleteAllColorFeedbacks = async () => {
  try {
    const response = await axios.delete(`${API_URL}/api/color-feedbacks`)
    return response.data
  } catch (error) {
    console.error('❌ 전체 피드백 삭제 실패:', error)
    throw error
  }
}

/**
 * 색상 피드백 JSON 추출
 */
export const exportColorFeedbacks = async () => {
  try {
    const response = await axios.get(`${API_URL}/api/color-feedbacks/export`)
    return response.data
  } catch (error) {
    console.error('❌ 피드백 내보내기 실패:', error)
    throw error
  }
}

/**
 * 색상 ML 모델 학습
 */
export const trainColorModel = async () => {
  try {
    const response = await axios.post(`${API_URL}/api/train-color-model`)
    return response.data
  } catch (error) {
    console.error('❌ ML 학습 실패:', error)
    throw error
  }
}

/**
 * 문제 피드백 제출
 */
export const submitProblemFeedback = async (feedbackData) => {
  try {
    const response = await axios.post(`${API_URL}/api/feedback`, feedbackData)
    return response.data
  } catch (error) {
    console.error('❌ 피드백 제출 실패:', error)
    throw error
  }
}

/**
 * 홀드 색상 피드백 제출
 */
export const submitHoldColorFeedback = async (feedbackData) => {
  try {
    await axios.post(`${API_URL}/api/hold-color-feedback`, feedbackData)
  } catch (error) {
    console.error('❌ 홀드 색상 피드백 제출 실패:', error)
    throw error
  }
}

