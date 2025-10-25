import * as api from '../api'

/**
 * 피드백 관련 로직 Hook
 */
export const useFeedback = ({
  setModelStats,
  setColorFeedbacks,
  setFeedbacksLoading,
  colorFeedbacks
}) => {
  
  // 통계 로드
  const loadStats = async () => {
    try {
      const data = await api.getStats()
      setModelStats(data.stats)
    } catch {
      // API가 없으면 무시 (선택적 기능)
    }
  }

  // 🎨 색상 피드백 목록 로드
  const loadColorFeedbacks = async () => {
    setFeedbacksLoading(true)
    try {
      const data = await api.getColorFeedbacks()
      setColorFeedbacks(data.feedbacks || [])
      console.log(`✅ 피드백 ${data.count}개 로드 완료`)
    } catch (error) {
      console.error('피드백 로드 실패:', error)
      alert('피드백 목록을 불러오는데 실패했습니다.')
    } finally {
      setFeedbacksLoading(false)
    }
  }

  // 🎨 색상 피드백 확인 (ML 학습용으로 확정)
  const confirmFeedback = async (feedbackId) => {
    if (!confirm('이 피드백을 ML 학습 데이터로 확정하시겠습니까?')) {
      return
    }

    try {
      await api.confirmColorFeedback(feedbackId)
      alert('피드백이 확인되었습니다! ML 학습에 사용됩니다.')
      loadColorFeedbacks() // 목록 새로고침
    } catch (error) {
      console.error('피드백 확인 실패:', error)
      alert('피드백 확인에 실패했습니다.')
    }
  }

  // 🎨 색상 피드백 삭제
  const deleteFeedback = async (feedbackId) => {
    if (!confirm('이 피드백을 삭제하시겠습니까?')) {
      return
    }

    try {
      await api.deleteColorFeedback(feedbackId)
      alert('피드백이 삭제되었습니다.')
      loadColorFeedbacks() // 목록 새로고침
    } catch (error) {
      console.error('피드백 삭제 실패:', error)
      alert('피드백 삭제에 실패했습니다.')
    }
  }

  // 🤖 ML 학습 실행
  const trainColorModel = async () => {
    const confirmedCount = colorFeedbacks.filter(f => f.confirmed).length
    
    if (confirmedCount < 30) {
      alert(`⚠️ 확인된 피드백이 부족합니다.\n\n현재: ${confirmedCount}개\n필요: 30개 이상`)
      return
    }

    if (!confirm(`🎓 ML 색상 분류 모델을 학습하시겠습니까?\n\n확인된 피드백: ${confirmedCount}개`)) {
      return
    }

    try {
      setFeedbacksLoading(true)
      const data = await api.trainColorModel()
      alert(`✅ ML 학습 완료!\n\n정확도: ${(data.test_accuracy * 100).toFixed(1)}%\nCross-validation: ${(data.cv_accuracy * 100).toFixed(1)}%`)
      loadColorFeedbacks() // 목록 새로고침
    } catch (error) {
      console.error('ML 학습 실패:', error)
      alert(`❌ ML 학습 실패: ${error.response?.data?.detail || error.message}`)
    } finally {
      setFeedbacksLoading(false)
    }
  }

  // GPT-4 상태 확인 (디버깅용)
  const checkGpt4Status = async () => {
    try {
      console.log('🔍 GPT-4 상태 확인 중...')
      const status = await api.checkGpt4Status()
      console.log('📊 GPT-4 상태:', status)
      
      let message = `🤖 GPT-4 상태 확인\n\n`
      message += `✅ 사용 가능: ${status.available ? '예' : '아니오'}\n`
      message += `🔑 API 키: ${status.api_key_set ? '설정됨' : '없음'}\n`
      message += `📋 이유: ${status.reason}\n`
      message += `📝 상세: ${status.details}\n`
      
      if (status.recommended_method) {
        message += `🎯 권장 방법: ${status.recommended_method}\n`
      }
      
      alert(message)
    } catch (error) {
      console.error('GPT-4 상태 확인 실패:', error)
      alert(`❌ GPT-4 상태 확인 실패: ${error.message}`)
    }
  }

  // GPT-4 간단 테스트 (디버깅용)
  const testGpt4 = async () => {
    try {
      console.log('🧪 GPT-4 테스트 시작...')
      const result = await api.testGpt4()
      console.log('🧪 GPT-4 테스트 결과:', result)
      
      let message = `🧪 GPT-4 테스트 결과\n\n`
      message += `✅ 성공: ${result.success ? '예' : '아니오'}\n`
      message += `📝 메시지: ${result.message}\n`
      message += `📋 상세: ${result.details}\n`
      
      if (result.success && result.result) {
        message += `\n🎯 분석 결과:\n`
        message += `- 난이도: ${result.result.difficulty}\n`
        message += `- 유형: ${result.result.type}\n`
        message += `- 신뢰도: ${result.result.confidence}\n`
        if (result.result.reasoning) {
          message += `- 분석: ${result.result.reasoning}\n`
        }
      }
      
      alert(message)
    } catch (error) {
      console.error('GPT-4 테스트 실패:', error)
      alert(`❌ GPT-4 테스트 실패: ${error.message}`)
    }
  }

  // GPT-4 결과를 훈련 데이터로 변환
  const convertGpt4ToTraining = async () => {
    try {
      const data = await api.convertGpt4ToTraining()
      alert(`✅ ${data.message}`)
      loadStats() // 통계 새로고침
    } catch (error) {
      alert(`❌ 변환 실패: ${error.response?.data?.detail || error.message}`)
    }
  }

  return {
    loadStats,
    loadColorFeedbacks,
    confirmFeedback,
    deleteFeedback,
    trainColorModel,
    checkGpt4Status,
    testGpt4,
    convertGpt4ToTraining
  }
}

