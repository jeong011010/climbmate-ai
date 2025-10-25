import { useEffect } from 'react'

/**
 * 분석 히스토리 관리 Hook
 */
export const useHistory = (analysisHistory, setAnalysisHistory, preview, wallAngle) => {
  
  const loadAnalysisHistory = () => {
    const history = JSON.parse(localStorage.getItem('climbmate_history') || '[]')
    setAnalysisHistory(history)
  }

  // 컴포넌트 마운트 시 데이터 로드
  useEffect(() => {
    loadAnalysisHistory()
  }, [])

  // 분석 결과를 히스토리에 저장
  const saveToHistory = (analysisResult) => {
    if (!analysisResult || !analysisResult.problems) return
    
    // 이미지 데이터 제거하여 용량 절약
    const compressedResult = {
      ...analysisResult,
      // 이미지 데이터 제거 (용량 절약)
      annotated_image: undefined,
      // 문제별 이미지 데이터도 제거
      problems: analysisResult.problems?.map(problem => ({
        ...problem,
        annotated_image: undefined
      }))
    }
    
    const historyItem = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      image: preview, // 썸네일은 유지 (작은 크기)
      result: compressedResult,
      wallAngle: wallAngle
    }
    
    try {
      const newHistory = [historyItem, ...analysisHistory.slice(0, 19)] // 최대 20개로 줄임
      setAnalysisHistory(newHistory)
      localStorage.setItem('climbmate_history', JSON.stringify(newHistory))
      console.log('✅ 히스토리 저장 완료')
    } catch (error) {
      console.error('❌ 히스토리 저장 실패:', error)
      // 스토리지 용량 초과 시 오래된 항목들 삭제
      try {
        const reducedHistory = [historyItem, ...analysisHistory.slice(0, 9)] // 최대 10개로 더 줄임
        setAnalysisHistory(reducedHistory)
        localStorage.setItem('climbmate_history', JSON.stringify(reducedHistory))
        console.log('✅ 히스토리 저장 완료 (용량 절약)')
      } catch (retryError) {
        console.error('❌ 히스토리 저장 완전 실패:', retryError)
        // 히스토리 저장 실패해도 분석 결과는 표시
      }
    }
  }

  return {
    loadAnalysisHistory,
    saveToHistory
  }
}

