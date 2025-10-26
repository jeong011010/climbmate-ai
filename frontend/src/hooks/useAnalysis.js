/**
 * 이미지 분석 관련 Hook
 */
export const useAnalysis = ({
  image,
  wallAngle,
  setLoading,
  setLoadingProgress,
  setDetectedHolds,
  setDetectedProblems,
  setCurrentAnalysisStep,
  setResult,
  saveToHistory
}) => {
  
  // 🚀 클라이언트 사이드 AI 분석 (기본 분석 방법)
  const analyzeImage = async () => {
    if (!image) return

    setLoading(true)
    setLoadingProgress(0)
    setDetectedHolds(0)
    setDetectedProblems(0)
    setCurrentAnalysisStep('')
    setResult(null)

    // 진행률 시뮬레이션을 위한 변수 (cleanup을 위해 함수 스코프에 선언)
    let currentProgress = 5
    let targetProgress = 5
    let progressInterval = null
    
    try {
      console.log('🚀 클라이언트 사이드 AI 분석 시작...')
      
      // 실시간 진행상황 업데이트 함수 등록
      window.updateAnalysisProgress = (data) => {
        // 목표 진행률 설정
        targetProgress = data.progress
        
        // 메시지 즉시 업데이트
        setCurrentAnalysisStep(data.message)
        
        // 특정 단계에서 추가 정보 표시
        if (data.holds_count) {
          setDetectedHolds(data.holds_count)
        }
        if (data.problems_count) {
          setDetectedProblems(data.problems_count)
        }
        
        // 진행률 시뮬레이션 시작 (부드러운 증가)
        if (progressInterval) {
          clearInterval(progressInterval)
        }
        
        progressInterval = setInterval(() => {
          if (currentProgress < targetProgress) {
            currentProgress += 1
            setLoadingProgress(currentProgress)
          } else {
            currentProgress = targetProgress
            setLoadingProgress(targetProgress)
            clearInterval(progressInterval)
            progressInterval = null
          }
        }, 100) // 100ms마다 1%씩 증가
      }

      // 클라이언트 AI 분석기 로드
      const { default: ClientAIAnalyzer } = await import('../clientAI.js')
      const analyzer = new ClientAIAnalyzer()

      // 초기 상태만 설정 (이후 SSE에서 실시간 업데이트)
      setCurrentAnalysisStep('서버로 이미지 전송 중...')
      setLoadingProgress(5)

      // 사용자 브라우저에서 직접 분석 (SSE로 실시간 진행상황 수신)
      const clientResult = await analyzer.analyzeImage(image, wallAngle)

      // 분석 완료 후 최종 상태 설정 (100%까지 부드럽게)
      targetProgress = 100
      setCurrentAnalysisStep('✅ 분석 완료!')
      
      // 100%까지 부드럽게 증가한 후 완료 처리
      await new Promise(resolve => {
        const finalInterval = setInterval(() => {
          if (currentProgress < 100) {
            currentProgress += 2  // 조금 빠르게
            setLoadingProgress(currentProgress)
          } else {
            clearInterval(finalInterval)
            setLoadingProgress(100)
            resolve()
          }
        }, 50)
      })
      
      setLoading(false)
      setResult(clientResult)

      // 통계 업데이트
      if (clientResult.statistics) {
        setDetectedHolds(clientResult.statistics.total_holds || 0)
        setDetectedProblems(clientResult.statistics.total_problems || 0)
      }

      // 히스토리에 저장
      saveToHistory(clientResult)

      // 전역 함수 및 interval 정리
      if (progressInterval) {
        clearInterval(progressInterval)
      }
      delete window.updateAnalysisProgress

      console.log('✅ 클라이언트 사이드 분석 완료:', clientResult)

    } catch (error) {
      console.error('❌ 클라이언트 사이드 분석 실패:', error)
      
      // interval 정리
      if (progressInterval) {
        clearInterval(progressInterval)
      }
      
      setLoading(false)
      setCurrentAnalysisStep('분석 실패')
      
      // 에러 타입별 구체적인 메시지 제공
      let errorMessage = '분석 중 오류가 발생했습니다.';
      if (error.message.includes('네트워크')) {
        errorMessage = '네트워크 연결을 확인해주세요.';
      } else if (error.message.includes('메모리') || error.message.includes('메모리가 부족')) {
        errorMessage = '브라우저 메모리가 부족합니다. 다른 탭을 닫고 다시 시도해주세요.';
      } else if (error.message.includes('지원하지 않')) {
        errorMessage = '브라우저가 AI 모델을 지원하지 않습니다. Chrome 또는 Firefox 최신 버전을 사용해주세요.';
      } else if (error.message.includes('404')) {
        errorMessage = '서버를 찾을 수 없습니다. 잠시 후 다시 시도해주세요.';
      } else if (error.message.includes('500')) {
        errorMessage = '서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.';
      } else {
        errorMessage = error.message || '알 수 없는 오류가 발생했습니다.';
      }
      
      alert(`❌ 클라이언트 사이드 분석 실패: ${errorMessage}`)
    }
  }

  return {
    analyzeImage
  }
}

