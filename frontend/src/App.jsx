import axios from 'axios'
import { useEffect, useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'https://climbmate.store'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [detectedHolds, setDetectedHolds] = useState(0)
  const [detectedProblems, setDetectedProblems] = useState(0)
  const [currentAnalysisStep, setCurrentAnalysisStep] = useState('')
  const [result, setResult] = useState(null)
  const [selectedProblem, setSelectedProblem] = useState(null)
  const [selectedHold, setSelectedHold] = useState(null) // 선택된 홀드
  const [wallAngle, setWallAngle] = useState(null)
  const [annotatedImage, setAnnotatedImage] = useState(null)
  const [showImageModal, setShowImageModal] = useState(false)
  const [showControlPanel, setShowControlPanel] = useState(false)
  const [showFeedbackModal, setShowFeedbackModal] = useState(false)
  const [colorFeedbackProblem, setColorFeedbackProblem] = useState(null)
  const [feedbackDifficulty, setFeedbackDifficulty] = useState('')
  const [feedbackType, setFeedbackType] = useState('')
  const [feedbackText, setFeedbackText] = useState('')
  const [showHoldFeedbackModal, setShowHoldFeedbackModal] = useState(false)
  const [holdColorFeedback, setHoldColorFeedback] = useState('')
  const [modelStats, setModelStats] = useState(null)
  
  // 새로운 상태들
  const [analysisHistory, setAnalysisHistory] = useState([])
  const [favorites, setFavorites] = useState([])
  const [currentView, setCurrentView] = useState('analyze') // 'analyze', 'history', 'favorites', 'stats'
  const [compareMode, setCompareMode] = useState(false)
  const [selectedForCompare, setSelectedForCompare] = useState([])
  
  // Ref
  const imageRef = useRef(null)
  const [imageLoaded, setImageLoaded] = useState(false)

  // 통계 로드
  const loadStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/stats`)
      setModelStats(response.data.stats)
    } catch {
      // API가 없으면 무시 (선택적 기능)
      console.log('통계 API 사용 불가 (정상)')
    }
  }

  // GPT-4 상태 확인 (디버깅용)
  const checkGpt4Status = async () => {
    try {
      console.log('🔍 GPT-4 상태 확인 중...')
      const response = await axios.get(`${API_URL}/api/gpt4-status`)
      console.log('📊 GPT-4 상태:', response.data)
      
      const status = response.data
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
      const response = await axios.post(`${API_URL}/api/test-gpt4`)
      console.log('🧪 GPT-4 테스트 결과:', response.data)
      
      const result = response.data
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

  // 분석 히스토리 로드
  const loadAnalysisHistory = () => {
    const history = JSON.parse(localStorage.getItem('climbmate_history') || '[]')
    setAnalysisHistory(history)
  }

  // 즐겨찾기 로드
  const loadFavorites = () => {
    const favs = JSON.parse(localStorage.getItem('climbmate_favorites') || '[]')
    setFavorites(favs)
  }

  // 컴포넌트 마운트 시 데이터 로드
  useEffect(() => {
    loadAnalysisHistory()
    loadFavorites()
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

  // 즐겨찾기 추가/제거
  const toggleFavorite = (problemId) => {
    const isFavorited = favorites.includes(problemId)
    let newFavorites
    
    if (isFavorited) {
      newFavorites = favorites.filter(id => id !== problemId)
    } else {
      newFavorites = [...favorites, problemId]
    }
    
    setFavorites(newFavorites)
    localStorage.setItem('climbmate_favorites', JSON.stringify(newFavorites))
  }

  // GPT-4 결과를 훈련 데이터로 변환
  const convertGpt4ToTraining = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/convert-gpt4`)
      alert(`✅ ${response.data.message}`)
      loadStats() // 통계 새로고침
    } catch (error) {
      alert(`❌ 변환 실패: ${error.response?.data?.detail || error.message}`)
    }
  }

  // 문제 비교 기능
  const toggleCompareMode = () => {
    setCompareMode(!compareMode)
    setSelectedForCompare([])
  }

  const toggleProblemForCompare = (problemId) => {
    if (selectedForCompare.includes(problemId)) {
      setSelectedForCompare(selectedForCompare.filter(id => id !== problemId))
    } else if (selectedForCompare.length < 3) {
      setSelectedForCompare([...selectedForCompare, problemId])
    }
  }


  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(file)
      setResult(null)
      setSelectedProblem(null)
      setSelectedHold(null)
      setAnnotatedImage(null)
      setShowControlPanel(true)
      setImageLoaded(false)
    }
  }

  const handleCameraCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        } 
      })
      
      const video = document.createElement('video')
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      
      video.srcObject = stream
      video.play()
      
      // 카메라 모달 생성
      const modal = document.createElement('div')
      modal.className = 'fixed inset-0 bg-black bg-opacity-90 z-50 flex flex-col items-center justify-center'
      modal.innerHTML = `
        <div class="bg-white p-4 rounded-t-2xl w-full max-w-md">
          <video class="w-full rounded-lg" autoplay playsinline></video>
          <div class="flex gap-4 mt-4">
            <button id="capture-btn" class="flex-1 bg-primary-500 text-white py-3 rounded-xl font-semibold">
              📸 촬영
            </button>
            <button id="cancel-btn" class="flex-1 bg-gray-300 text-gray-700 py-3 rounded-xl font-semibold">
              취소
            </button>
          </div>
        </div>
      `
      
      document.body.appendChild(modal)
      const videoEl = modal.querySelector('video')
      videoEl.srcObject = stream
      
      modal.querySelector('#capture-btn').onclick = () => {
        canvas.width = videoEl.videoWidth
        canvas.height = videoEl.videoHeight
        ctx.drawImage(videoEl, 0, 0)
        
        canvas.toBlob((blob) => {
          const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })
          setImage(file)
          setPreview(URL.createObjectURL(blob))
          setResult(null)
          setSelectedProblem(null)
          setSelectedHold(null)
          setAnnotatedImage(null)
          setShowControlPanel(true)
          setImageLoaded(false)
          
          stream.getTracks().forEach(track => track.stop())
          document.body.removeChild(modal)
        }, 'image/jpeg', 0.9)
      }
      
      modal.querySelector('#cancel-btn').onclick = () => {
        stream.getTracks().forEach(track => track.stop())
        document.body.removeChild(modal)
      }
      
    } catch (error) {
      console.error('카메라 접근 실패:', error)
      alert('카메라에 접근할 수 없습니다. 파일 업로드를 사용해주세요.')
    }
  }


  // 🚀 클라이언트 사이드 AI 분석 (기본 분석 방법)
  const analyzeImage = async () => {
    if (!image) return

    setLoading(true)
    setLoadingProgress(0)
    setDetectedHolds(0)
    setDetectedProblems(0)
    setCurrentAnalysisStep('')
    setResult(null)

    try {
      console.log('🚀 클라이언트 사이드 AI 분석 시작...')
      
      // 실시간 진행상황 업데이트 함수 등록
      window.updateAnalysisProgress = (data) => {
        setCurrentAnalysisStep(data.message)
        setLoadingProgress(data.progress)
        
        // 특정 단계에서 추가 정보 표시
        if (data.holds_count) {
          setDetectedHolds(data.holds_count)
        }
        if (data.problems_count) {
          setDetectedProblems(data.problems_count)
        }
      }

      // 클라이언트 AI 분석기 로드
      const { default: ClientAIAnalyzer } = await import('./clientAI.js')
      const analyzer = new ClientAIAnalyzer()

      // 초기 상태만 설정 (이후 SSE에서 실시간 업데이트)
      setCurrentAnalysisStep('서버로 이미지 전송 중...')
      setLoadingProgress(5)

      // 사용자 브라우저에서 직접 분석 (SSE로 실시간 진행상황 수신)
      const clientResult = await analyzer.analyzeImage(image, wallAngle)

      // 분석 완료 후 최종 상태 설정
      setLoading(false)
      setLoadingProgress(100)
      setCurrentAnalysisStep('✅ 분석 완료!')
      setResult(clientResult)

      // 통계 업데이트
      if (clientResult.statistics) {
        setDetectedHolds(clientResult.statistics.total_holds || 0)
        setDetectedProblems(clientResult.statistics.total_problems || 0)
      }

      // 히스토리에 저장
      saveToHistory(clientResult)

      // 전역 함수 정리
      delete window.updateAnalysisProgress

      console.log('✅ 클라이언트 사이드 분석 완료:', clientResult)

    } catch (error) {
      console.error('❌ 클라이언트 사이드 분석 실패:', error)
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


  const submitFeedback = async () => {
    if (!selectedProblem || !selectedProblem.db_id) {
      alert('문제 ID를 찾을 수 없습니다.')
      return
    }

    if (!feedbackDifficulty || !feedbackType) {
      alert('난이도와 유형을 모두 선택해주세요.')
      return
    }

    try {
      const response = await axios.post(`${API_URL}/api/feedback`, {
        problem_id: selectedProblem.db_id,
        user_difficulty: feedbackDifficulty,
        user_type: feedbackType,
        user_feedback: feedbackText
      })

      alert(response.data.message)
      setModelStats(response.data.stats)
      setShowFeedbackModal(false)
      setFeedbackDifficulty('')
      setFeedbackType('')
      setFeedbackText('')
      
      // 통계 다시 로드
      loadStats()
    } catch (error) {
      console.error('피드백 제출 실패:', error)
      alert('피드백 제출에 실패했습니다.')
    }
  }

  // 홀드 색상 피드백 제출
  const submitHoldColorFeedback = async () => {
    if (!selectedHold || !selectedProblem || !selectedProblem.db_id) {
      alert('홀드 또는 문제 정보를 찾을 수 없습니다.')
      return
    }

    if (!holdColorFeedback) {
      alert('색상을 선택해주세요.')
      return
    }

    try {
      await axios.post(`${API_URL}/api/hold-color-feedback`, {
        problem_id: selectedProblem.db_id,
        hold_id: selectedHold.id || `${selectedHold.center[0]}_${selectedHold.center[1]}`,
        predicted_color: selectedHold.individual_color || selectedHold.color,
        user_color: holdColorFeedback,
        hold_center: selectedHold.center
      })

      alert('홀드 색상 피드백이 제출되었습니다. 감사합니다!')
      setShowHoldFeedbackModal(false)
      setHoldColorFeedback('')
      
      // 통계 다시 로드
      loadStats()
    } catch (error) {
      console.error('홀드 색상 피드백 제출 실패:', error)
      alert('홀드 색상 피드백 제출에 실패했습니다.')
    }
  }

  const handleImageClick = (e) => {
    if (!result || !result.problems) return
    
    e.preventDefault() // 기본 동작 방지
    
    const rect = e.target.getBoundingClientRect()
    
    // 터치 이벤트와 마우스 이벤트 모두 지원
    // onTouchEnd에서는 e.touches가 비어있으므로 e.changedTouches 사용
    let clientX, clientY
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX
      clientY = e.touches[0].clientY
    } else if (e.changedTouches && e.changedTouches.length > 0) {
      clientX = e.changedTouches[0].clientX
      clientY = e.changedTouches[0].clientY
    } else {
      clientX = e.clientX
      clientY = e.clientY
    }
    
    const x = clientX - rect.left
    const y = clientY - rect.top
    
    // 이미지의 실제 크기 (원본 이미지 기준)
    const img = e.target
    const scaleX = img.naturalWidth / rect.width
    const scaleY = img.naturalHeight / rect.height
    
    // 클릭 위치를 원본 이미지 좌표로 변환
    const realX = x * scaleX
    const realY = y * scaleY
    
    // 클릭 위치에서 가장 가까운 홀드와 문제 찾기
    let closestProblem = null
    let closestHold = null
    let minDistance = Infinity
    
    result.problems?.forEach(problem => {
      problem.holds?.forEach(hold => {
        if (!hold.center) return
        
        const holdX = hold.center[0]
        const holdY = hold.center[1]
        const distance = Math.sqrt(Math.pow(realX - holdX, 2) + Math.pow(realY - holdY, 2))
        
        if (distance < minDistance && distance < 150) { // 150px 반경 내
          minDistance = distance
          closestProblem = problem
          closestHold = hold
        }
      })
    })
    
    if (closestProblem) {
      setSelectedProblem(closestProblem)
      setSelectedHold(closestHold)
    } else {
      setSelectedHold(null)
    }
  }

  const colorEmoji = {
    black: '⚫', white: '⚪', gray: '🔘',
    red: '🔴', orange: '🟠', yellow: '🟡',
    green: '🟢', blue: '🔵', purple: '🟣',
    pink: '🩷', brown: '🟤', mint: '💚', lime: '🍃'
  }

  // 히스토리 뷰 컴포넌트
  const HistoryView = () => (
    <div className="w-full px-2 sm:px-4">
      <div className="glass-card p-4 sm:p-6">
        <h2 className="text-xl sm:text-2xl font-bold mb-4 text-slate-800">📚 분석 히스토리</h2>
        {analysisHistory.length === 0 ? (
          <p className="text-slate-600 text-center py-8">아직 분석한 문제가 없습니다.</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            {analysisHistory.map((item) => (
              <div key={item.id} className="glass-card p-4 hover:shadow-lg transition-shadow">
                <img 
                  src={item.image} 
                  alt="분석된 문제" 
                  className="w-full h-32 object-cover rounded-lg mb-3"
                />
                <div className="text-sm text-slate-600 mb-2">
                  {new Date(item.timestamp).toLocaleString()}
                </div>
                <div className="space-y-1">
                  {item.result.problems?.map((problem, idx) => (
                    <div key={idx} className="flex justify-between items-center">
                      <span className="text-sm font-medium">{problem.color}</span>
                      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        {problem.difficulty} {problem.type}
                      </span>
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => {
                    setResult(item.result)
                    setPreview(item.image)
                    setCurrentView('analyze')
                  }}
                  className="w-full mt-3 glass-button text-sm py-2"
                >
                  다시 보기
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )

  // 즐겨찾기 뷰 컴포넌트
  const FavoritesView = () => (
    <div className="w-full px-2 sm:px-4">
      <div className="glass-card p-4 sm:p-6">
        <h2 className="text-xl sm:text-2xl font-bold mb-4 text-slate-800">⭐ 즐겨찾기</h2>
        {favorites.length === 0 ? (
          <p className="text-slate-600 text-center py-8">즐겨찾기한 문제가 없습니다.</p>
        ) : (
          <div className="space-y-3">
            {favorites.map((problemId) => {
              const historyItem = analysisHistory.find(item => 
                item.result.problems?.some(p => p.id === problemId)
              )
              if (!historyItem) return null
              
              const problem = historyItem.result.problems.find(p => p.id === problemId)
              return (
                <div key={problemId} className="glass-card p-4 flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <img 
                      src={historyItem.image} 
                      alt="즐겨찾기 문제" 
                      className="w-16 h-16 object-cover rounded-lg"
                    />
                    <div>
                      <div className="font-medium">{problem.color}</div>
                      <div className="text-sm text-slate-600">
                        {problem.difficulty} {problem.type}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => toggleFavorite(problemId)}
                    className="text-yellow-500 hover:text-yellow-600"
                  >
                    ⭐
                  </button>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )

  // 문제 비교 뷰 컴포넌트
  const CompareView = () => {
    const selectedProblems = result?.problems?.filter(p => selectedForCompare.includes(p.id)) || []
    
    return (
      <div className="w-full px-2 sm:px-4">
        <div className="glass-card p-4 sm:p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl sm:text-2xl font-bold text-slate-800">🔍 문제 비교</h2>
            <button
              onClick={toggleCompareMode}
              className="glass-button px-4 py-2 text-sm"
            >
              비교 모드 종료
            </button>
          </div>
          
          {selectedProblems.length === 0 ? (
            <p className="text-slate-600 text-center py-8">
              비교할 문제를 선택해주세요. (최대 3개)
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {selectedProblems.map((problem) => (
                <div key={problem.id} className="glass-card p-4">
                  <div className="text-center mb-3">
                    <span className="text-3xl">{colorEmoji[problem.color_name] || '⭕'}</span>
                    <h3 className="text-lg font-bold mt-2">{(problem.color_name || 'UNKNOWN').toUpperCase()}</h3>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-slate-600">난이도:</span>
                      <span className="font-bold text-blue-600">{problem.difficulty}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-slate-600">유형:</span>
                      <span className="font-bold text-green-600">{problem.type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-slate-600">홀드 수:</span>
                      <span className="font-bold">{problem.hold_count}개</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-slate-600">분석 방법:</span>
                      <span className="font-bold">{problem.gpt4_reasoning ? 'GPT-4 AI' : '규칙 기반'}</span>
                    </div>
                  </div>
                  
                  {problem.gpt4_reasoning && (
                    <div className="mt-3 p-2 bg-blue-50 rounded text-xs text-slate-700">
                      <strong>AI 분석:</strong> {problem.gpt4_reasoning}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  // 통계 뷰 컴포넌트
  const StatsView = () => (
    <div className="w-full px-2 sm:px-4">
      <div className="glass-card p-4 sm:p-6">
        <h2 className="text-xl sm:text-2xl font-bold mb-4 text-slate-800">📊 통계</h2>
        
        {modelStats ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="glass-card p-4 text-center">
                <div className="text-2xl font-bold text-blue-600">{modelStats.total_problems}</div>
                <div className="text-sm text-slate-600">전체 문제</div>
              </div>
              <div className="glass-card p-4 text-center">
                <div className="text-2xl font-bold text-green-600">{modelStats.verified_problems}</div>
                <div className="text-sm text-slate-600">검증된 문제</div>
              </div>
            </div>
            
            {modelStats.verified_problems > 0 && (
              <div className="glass-card p-4">
                <h3 className="font-bold mb-2">AI 모델 성능</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>난이도 정확도:</span>
                    <span className="font-bold text-blue-600">
                      {Math.round(modelStats.difficulty_accuracy * 100)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>유형 정확도:</span>
                    <span className="font-bold text-green-600">
                      {Math.round(modelStats.type_accuracy * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
            
            <div className="flex flex-col sm:flex-row gap-2">
              <button
                onClick={convertGpt4ToTraining}
                className="glass-button flex-1 py-2 text-xs sm:text-sm"
              >
                <span className="hidden sm:inline">🤖 GPT-4 결과를 훈련 데이터로 변환</span>
                <span className="sm:hidden">🤖 GPT-4 변환</span>
              </button>
              <button
                onClick={() => {
                  axios.post(`${API_URL}/api/train`)
                    .then(() => alert('모델 훈련 완료!'))
                    .catch(err => alert(`훈련 실패: ${err.message}`))
                }}
                className="glass-button flex-1 py-2 text-xs sm:text-sm"
              >
                <span className="hidden sm:inline">🎯 모델 재훈련</span>
                <span className="sm:hidden">🎯 재훈련</span>
              </button>
            </div>
          </div>
        ) : (
          <p className="text-slate-600 text-center py-8">통계 데이터를 불러올 수 없습니다.</p>
        )}
      </div>
    </div>
  )

  return (
    <div className="w-full min-h-screen flex flex-col items-center">
      {/* 헤더 (상단 고정) */}
      <div className="fixed top-0 left-0 right-0 bg-white/95 backdrop-blur-md border-b border-slate-200 shadow-sm z-40">
        <div className="w-full px-2 sm:px-4 py-2 sm:py-3">
          <div className="text-center text-slate-800">
            <h1 className="text-xl sm:text-3xl font-extrabold gradient-text">
              🧗‍♀️ ClimbMate
            </h1>
            <p className="text-xs sm:text-sm opacity-70 font-medium">
              AI 기반 클라이밍 문제 분석
            </p>
           {modelStats && modelStats.verified_problems > 0 && (
             <div className="mt-3 text-xs text-slate-600">
               📊 학습 데이터: {modelStats.verified_problems}개
               {modelStats.ready_for_training && (
                 <span className="ml-2 text-green-600 font-bold">✅ AI 학습 가능</span>
               )}
             </div>
           )}
           
           {/* GPT-4 디버깅 버튼 (개발용) */}
           <div className="mt-2 flex justify-center gap-2">
             <button
               onClick={checkGpt4Status}
               className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors"
             >
               🔍 GPT-4 상태
             </button>
             <button
               onClick={testGpt4}
               className="px-3 py-1 text-xs bg-green-100 text-green-700 rounded-full hover:bg-green-200 transition-colors"
             >
               🧪 GPT-4 테스트
             </button>
           </div>
          </div>
        </div>
      </div>

      {/* 메인 컨텐츠 영역 */}
      <div className="w-full pt-24 pb-20 px-2 sm:px-4">
        {preview && (
           <div className="relative mb-4 w-full max-w-[900px] mx-auto">
             <div className="relative w-full flex justify-center">
               <div className="relative" style={{ display: 'inline-block' }}>
                 <img 
                   ref={imageRef}
                   src={annotatedImage || preview} 
                   alt="Climbing Wall" 
                   className={`max-h-[500px] object-contain rounded-2xl shadow-2xl border border-white/20 ${
                     result ? 'cursor-pointer hover:opacity-90 transition-opacity' : ''
                   }`}
                   onClick={result ? handleImageClick : undefined}
                   onTouchEnd={result ? handleImageClick : undefined}
                   onDoubleClick={result ? () => setShowImageModal(true) : undefined}
                   onLoad={() => {
                     setImageLoaded(true)
                     console.log('Image loaded:', imageRef.current?.naturalWidth, 'x', imageRef.current?.naturalHeight)
                   }}
                 />
                 
                 {/* SVG 오버레이 - 선택된 문제의 홀드들 강조 */}
                 {result && selectedProblem && imageRef.current && imageLoaded && (() => {
                   const img = imageRef.current
                   const rect = img.getBoundingClientRect()
                   const scaleX = rect.width / img.naturalWidth
                   const scaleY = rect.height / img.naturalHeight
                   
                   console.log('SVG Rendering:', {
                     naturalWidth: img.naturalWidth,
                     naturalHeight: img.naturalHeight,
                     displayWidth: rect.width,
                     displayHeight: rect.height,
                     scaleX,
                     scaleY,
                     holdsCount: selectedProblem.holds?.length
                   })
                   
                   return (
                     <svg
                       className="absolute top-0 left-0 pointer-events-none"
                       style={{
                         width: rect.width + 'px',
                         height: rect.height + 'px'
                       }}
                       viewBox={`0 0 ${img.naturalWidth} ${img.naturalHeight}`}
                       preserveAspectRatio="xMidYMid meet"
                     >
                       {selectedProblem.holds?.map((hold, idx) => {
                         if (!hold.bbox || hold.bbox.length !== 4) {
                           console.log('Hold missing bbox:', hold)
                           return null
                         }
                         const [x1, y1, x2, y2] = hold.bbox
                         const isSelectedHold = selectedHold && 
                           selectedHold.center && hold.center &&
                           selectedHold.center[0] === hold.center[0] && 
                           selectedHold.center[1] === hold.center[1]
                         
                         console.log(`Hold ${idx}:`, {
                           bbox: [x1, y1, x2, y2],
                           center: hold.center,
                           isSelected: isSelectedHold
                         })
                         
                         return (
                           <g key={idx}>
                             {/* 홀드 바운딩 박스 */}
                             <rect
                               x={x1}
                               y={y1}
                               width={x2 - x1}
                               height={y2 - y1}
                               fill="none"
                               stroke={isSelectedHold ? "#FFD700" : "#00FF00"}
                               strokeWidth={isSelectedHold ? "6" : "3"}
                               strokeDasharray={isSelectedHold ? "8,4" : "none"}
                               opacity="0.8"
                             >
                               {isSelectedHold && (
                                 <animate
                                   attributeName="opacity"
                                   values="0.8;1;0.8"
                                   dur="1s"
                                   repeatCount="indefinite"
                                 />
                               )}
                             </rect>
                             
                             {/* 홀드 중심점 */}
                             {hold.center && (
                               <circle
                                 cx={hold.center[0]}
                                 cy={hold.center[1]}
                                 r={isSelectedHold ? "10" : "6"}
                                 fill={isSelectedHold ? "#FFD700" : "#00FF00"}
                                 opacity="0.9"
                               >
                                 {isSelectedHold && (
                                   <animate
                                     attributeName="r"
                                     values="10;14;10"
                                     dur="1s"
                                     repeatCount="indefinite"
                                   />
                                 )}
                               </circle>
                             )}
                           </g>
                         )
                       })}
                     </svg>
                   )
                 })()}
                 
                 {result && selectedProblem && (
                   <div className="absolute top-4 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-full text-base font-bold shadow-lg animate-pulse-slow z-10">
                     {colorEmoji[selectedProblem.color_name] || '⭕'} {(selectedProblem.color_name || 'UNKNOWN').toUpperCase()} 선택됨
                   </div>
                 )}
               </div>
             </div>
           </div>
         )}

        {/* 메인 컨텐츠 */}
        {currentView === 'analyze' && (
          <>
            {/* 히어로 섹션 (이미지 없을 때) */}
            {!preview && !loading && (
              <div className="text-center w-full max-w-2xl mx-auto mb-8">
                <div className="glass-card p-8 sm:p-12 mb-6">
                  <div className="text-6xl sm:text-8xl mb-6 animate-bounce-slow">🧗‍♀️</div>
                  <h2 className="text-2xl sm:text-3xl font-bold gradient-text mb-4">
                    AI가 클라이밍 문제를 분석합니다
                  </h2>
                  <p className="text-sm sm:text-base text-slate-600 mb-6">
                    클라이밍 벽 사진을 업로드하면 AI가 홀드를 감지하고<br className="hidden sm:block"/>
                    난이도와 유형을 자동으로 분석해드립니다
                  </p>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
                    <div className="glass-card p-4">
                      <div className="text-3xl mb-2">🎯</div>
                      <div className="font-bold text-slate-800">정확한 분석</div>
                      <div className="text-xs text-slate-600">GPT-4 Vision 기반</div>
                    </div>
                    <div className="glass-card p-4">
                      <div className="text-3xl mb-2">⚡</div>
                      <div className="font-bold text-slate-800">빠른 처리</div>
                      <div className="text-xs text-slate-600">실시간 홀드 감지</div>
                    </div>
                    <div className="glass-card p-4">
                      <div className="text-3xl mb-2">📊</div>
                      <div className="font-bold text-slate-800">상세 정보</div>
                      <div className="text-xs text-slate-600">난이도/유형 제공</div>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col sm:flex-row gap-4 w-full max-w-md mx-auto">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    id="file-input"
                    className="hidden"
                  />
                  
                  <label 
                    htmlFor="file-input" 
                    className="glass-button flex-1 inline-flex items-center gap-2 px-6 py-4 text-slate-800 rounded-2xl text-base font-semibold cursor-pointer shadow-lg justify-center hover:shadow-xl hover:scale-105 transition-all"
                  >
                    📁 사진 업로드
                  </label>
                  
                  <button
                    onClick={handleCameraCapture}
                    className="glass-button flex-1 inline-flex items-center gap-2 px-6 py-4 text-slate-800 rounded-2xl text-base font-semibold cursor-pointer shadow-lg justify-center hover:shadow-xl hover:scale-105 transition-all"
                  >
                    📸 촬영
                  </button>
                </div>
              </div>
            )}

         {/* 로딩 */}
         {loading && (
           <div className="glass-card text-center p-12 my-8 w-full">
             <div className="relative w-24 h-24 mx-auto mb-8">
               <div className="w-24 h-24 border-6 border-primary-500/10 border-t-primary-500 border-r-purple-600 rounded-full animate-spin shadow-lg"></div>
               <div className="absolute inset-0 flex items-center justify-center">
                 <span className="text-lg font-bold gradient-text">{Math.round(loadingProgress)}%</span>
               </div>
             </div>
             <p className="text-xl gradient-text font-bold mb-3 animate-pulse">{currentAnalysisStep}</p>
             
             {/* 홀드/문제 개수 표시 */}
             {(detectedHolds > 0 || detectedProblems > 0) && (
               <div className="flex justify-center gap-6 text-sm text-slate-500 mb-3">
                 {detectedHolds > 0 && (
                   <span>🎯 홀드 {detectedHolds}개</span>
                 )}
                 {detectedProblems > 0 && (
                   <span>🎨 문제 {detectedProblems}개</span>
                 )}
               </div>
             )}
             
             <p className="text-base text-slate-600 font-medium">AI가 열심히 분석 중...</p>
             <div className="flex justify-center gap-1 mt-4">
               <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
               <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
               <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
             </div>
           </div>
         )}

        {/* 결과 */}
        {result && (
          <div className="w-full">
            {/* 통계 */}
            <div className="flex flex-row gap-4 mx-auto mb-6 w-full justify-center items-center">
              <div className="glass-card p-6 rounded-2xl text-center shadow-lg transition-all duration-300 flex-1 min-w-[80px] max-w-[120px] hover:translate-y-[-5px] hover:shadow-xl">
                <div className="text-3xl font-extrabold gradient-text mb-1">{result.statistics.total_problems}</div>
                <div className="text-xs text-slate-600 font-semibold">문제 수</div>
              </div>
              <div className="glass-card p-6 rounded-2xl text-center shadow-lg transition-all duration-300 flex-1 min-w-[80px] max-w-[120px] hover:translate-y-[-5px] hover:shadow-xl">
                <div className="text-3xl font-extrabold gradient-text mb-1">{result.statistics.total_holds}</div>
                <div className="text-xs text-slate-600 font-semibold">홀드 수</div>
              </div>
              <div className="glass-card p-6 rounded-2xl text-center shadow-lg transition-all duration-300 flex-1 min-w-[80px] max-w-[120px] hover:translate-y-[-5px] hover:shadow-xl">
                <div className="text-3xl font-extrabold gradient-text mb-1">{result.statistics.analyzable_problems}</div>
                <div className="text-xs text-slate-600 font-semibold">분석 가능</div>
              </div>
             </div>

             {/* 선택된 홀드 상세 */}
             {selectedHold && selectedProblem && (
               <div className="glass-card p-5 mx-auto mb-4 w-full shadow-lg border-2 border-yellow-400">
                 <div className="flex items-center justify-between mb-3">
                   <h3 className="text-xl text-slate-800 font-bold flex items-center gap-2">
                     <span className="text-3xl">🎯</span>
                     선택된 홀드
                   </h3>
                   <button
                     onClick={() => {
                       setSelectedHold(null)
                     }}
                     className="px-3 py-1 text-slate-600 hover:text-slate-800 text-sm"
                   >
                     ✕
                   </button>
                 </div>
                 
                {/* 문제 색상 */}
                <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-xl shadow-md mb-4 border-2 border-purple-200">
                  <h4 className="text-xs mb-2 text-slate-600 font-semibold text-center">🎨 문제 그룹 색상</h4>
                  <div className="flex items-center justify-center gap-2">
                    <span className="text-4xl">{colorEmoji[selectedHold.color] || '⭕'}</span>
                    <span className="text-xl font-bold gradient-text">{(selectedHold.color || 'UNKNOWN').toUpperCase()}</span>
                  </div>
                  <p className="text-xs text-slate-500 text-center mt-2">이 홀드가 속한 문제의 색상</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-md">
                    <h4 className="text-xs mb-2 text-slate-600 font-semibold text-center">💎 홀드 실제 색상</h4>
                    <div className="flex flex-col items-center justify-center gap-1">
                      <span className="text-3xl">{colorEmoji[selectedHold.individual_color] || '⭕'}</span>
                      <span className="text-sm font-bold text-slate-800">{(selectedHold.individual_color || 'UNKNOWN').toUpperCase()}</span>
                    </div>
                    <p className="text-xs text-slate-500 text-center mt-2">AI 감지 색상</p>
                  </div>
                  
                  <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-md">
                    <h4 className="text-xs mb-2 text-slate-600 font-semibold text-center">📍 위치</h4>
                    <div className="text-sm text-slate-700 text-center space-y-1">
                      <div className="font-mono">X: {selectedHold.center ? Math.round(selectedHold.center[0]) : 'N/A'}</div>
                      <div className="font-mono">Y: {selectedHold.center ? Math.round(selectedHold.center[1]) : 'N/A'}</div>
                    </div>
                  </div>
                </div>
                 
                 <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-3 rounded-xl border border-yellow-200">
                   <h4 className="text-sm mb-2 text-slate-800 font-bold text-center">💬 색상 피드백</h4>
                   <p className="text-xs text-slate-600 mb-3 text-center">
                     AI가 예측한 색상이 맞나요? 피드백을 주시면 더 정확해집니다!
                   </p>
                   <button
                     onClick={() => setShowHoldFeedbackModal(true)}
                     className="w-full px-4 py-2 bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-xl text-sm font-semibold shadow-md hover:shadow-lg transition-all"
                   >
                     🎨 색상 피드백 제출
                   </button>
                 </div>
               </div>
             )}

             {/* 선택된 문제 상세 */}
             {selectedProblem && selectedProblem.difficulty && (
               <div className="glass-card p-6 mx-auto mb-6 w-full text-center shadow-lg">
                 <div className="flex justify-between items-center mb-4">
                   <h3 className="text-2xl text-slate-800 font-extrabold flex-1">
                     {colorEmoji[selectedProblem.color_name] || '⭕'} {(selectedProblem.color_name || 'UNKNOWN').toUpperCase()} 문제
                   </h3>
                   <button
                     onClick={() => setShowFeedbackModal(true)}
                     className="px-4 py-2 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-xl text-sm font-semibold shadow-md hover:shadow-lg transition-all"
                   >
                     📝 피드백
                   </button>
                 </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-white/80 backdrop-blur-sm p-5 rounded-xl shadow-md transition-all duration-300 hover:translate-y-[-3px] hover:shadow-lg">
                    <h4 className="text-sm mb-3 text-slate-600 font-semibold text-center">🎯 난이도</h4>
                    <div className="text-3xl font-extrabold gradient-text text-center mb-1">{selectedProblem.difficulty || 'V?'}</div>
                    <div className="text-sm text-slate-600 mb-2 font-medium text-center">{selectedProblem.type || '미분석'}</div>
                    <div className="text-xl text-yellow-400 text-center drop-shadow-sm">
                      {'★'.repeat(Math.floor((selectedProblem.difficulty?.confidence || 0) * 5))}
                      {'☆'.repeat(5 - Math.floor((selectedProblem.difficulty?.confidence || 0) * 5))}
                    </div>
                  </div>

                  <div className="bg-white/80 backdrop-blur-sm p-5 rounded-xl shadow-md transition-all duration-300 hover:translate-y-[-3px] hover:shadow-lg">
                    <h4 className="text-sm mb-3 text-slate-600 font-semibold text-center">🏋️ 유형</h4>
                    <div className="text-lg font-bold text-slate-800 mb-2 text-center">{selectedProblem.type || '일반'}</div>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {selectedProblem.climb_type?.types?.slice(0, 3).map((type, idx) => (
                        <span key={idx} className="px-3 py-1 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-full text-xs font-semibold shadow-md">
                          {type}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-md mb-4">
                  <h4 className="text-sm mb-3 text-slate-800 font-bold text-center">📊 문제 정보</h4>
                  <div className="flex justify-between items-center py-2 border-b border-slate-200 text-sm transition-all duration-200 hover:bg-white/50 hover:px-4 hover:rounded-lg hover:-mx-4">
                    <span className="text-slate-600 font-medium">홀드 개수:</span>
                    <span className="font-bold gradient-text">{selectedProblem.hold_count}개</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-200 text-sm transition-all duration-200 hover:bg-white/50 hover:px-4 hover:rounded-lg hover:-mx-4">
                    <span className="text-slate-600 font-medium">난이도:</span>
                    <span className="font-bold gradient-text">{selectedProblem.difficulty || 'V?'}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-200 text-sm transition-all duration-200 hover:bg-white/50 hover:px-4 hover:rounded-lg hover:-mx-4">
                    <span className="text-slate-600 font-medium">유형:</span>
                    <span className="font-bold gradient-text">{selectedProblem.type || '일반'}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 text-sm transition-all duration-200 hover:bg-white/50 hover:px-4 hover:rounded-lg hover:-mx-4">
                    <span className="text-slate-600 font-medium">분석 방법:</span>
                    <span className="font-bold gradient-text">{selectedProblem.reasoning ? 'GPT-4 AI' : '규칙 기반'}</span>
                  </div>
                </div>

                {/* GPT-4 AI 분석 */}
                {selectedProblem.reasoning && (
                  <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-xl shadow-md border-2 border-blue-200">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-2xl">🤖</span>
                      <h4 className="text-sm text-slate-800 font-bold">GPT-4 AI 상세 분석</h4>
                      <span className="ml-auto text-xs bg-blue-500 text-white px-2 py-1 rounded-full">
                        신뢰도: {Math.round((selectedProblem.gpt4_confidence || 0.8) * 100)}%
                      </span>
                    </div>
                    
                    {/* 간소화된 분석 내용 */}
                    <div className="text-sm text-slate-700 leading-relaxed whitespace-pre-line">
                      {selectedProblem.reasoning}
                    </div>
                  </div>
                )}

                {/* 추가 팁 */}
                {!selectedProblem.reasoning && (
                  <div className="bg-yellow-50 p-4 rounded-xl shadow-md border-2 border-yellow-200">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xl">💡</span>
                      <h4 className="text-sm text-slate-800 font-bold">분석 팁</h4>
                    </div>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      이 문제는 규칙 기반으로 분석되었습니다. 더 정확한 분석을 위해 GPT-4를 활성화하거나 피드백을 제공해주세요!
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* 문제 목록 (축약) */}
            <div className="glass-card p-4 sm:p-8 mx-auto mb-8 w-full text-center shadow-lg">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg sm:text-xl font-bold text-slate-800">
                  📋 전체 문제 목록 ({result.problems?.length || 0}개)
                </h2>
                <button
                  onClick={toggleCompareMode}
                  className={`glass-button px-3 sm:px-4 py-2 text-xs sm:text-sm ${
                    compareMode ? 'bg-blue-500 text-white' : ''
                  }`}
                >
                  {compareMode ? '🔍 비교 모드' : '👁️ 비교 모드'}
                </button>
              </div>
              
              <div className="flex flex-col gap-4">
                {result.problems?.map((problem) => (
                  <div 
                    key={problem.id} 
                    className={`p-6 bg-white/80 backdrop-blur-sm rounded-2xl cursor-pointer transition-all duration-300 border-2 shadow-md ${
                      selectedProblem?.id === problem.id 
                        ? 'bg-gradient-to-r from-primary-500 to-purple-600 text-white border-transparent shadow-lg translate-y-[-3px]' 
                        : 'border-white/30 hover:translate-y-[-3px] hover:shadow-lg hover:bg-white/95'
                    }`}
                    onClick={() => setSelectedProblem(problem)}
                  >
                    <div className="flex items-center mb-3">
                      <span className="text-3xl mr-5">{colorEmoji[problem.color_name] || '⭕'}</span>
                      <div className="flex-1">
                        <div className={`text-xl font-bold mb-1 ${
                          selectedProblem?.id === problem.id ? 'text-white' : 'text-slate-800'
                        }`}>
                          {(problem.color_name || 'UNKNOWN').toUpperCase()}
                        </div>
                        <div className={`text-base font-medium ${
                          selectedProblem?.id === problem.id ? 'text-white' : 'text-slate-600'
                        }`}>
                          {problem.hold_count}개 홀드
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            toggleFavorite(problem.id)
                          }}
                          className={`text-2xl transition-colors ${
                            favorites.includes(problem.id) 
                              ? 'text-yellow-500' 
                              : 'text-gray-400 hover:text-yellow-500'
                          }`}
                        >
                          {favorites.includes(problem.id) ? '⭐' : '☆'}
                        </button>
                        
                        {compareMode && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              toggleProblemForCompare(problem.id)
                            }}
                            className={`text-xl transition-colors ${
                              selectedForCompare.includes(problem.id)
                                ? 'text-blue-500'
                                : 'text-gray-400 hover:text-blue-500'
                            }`}
                          >
                            {selectedForCompare.includes(problem.id) ? '🔍' : '👁️'}
                          </button>
                        )}
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setColorFeedbackProblem(problem)
                          }}
                          className="text-xl text-gray-400 hover:text-purple-500 transition-colors"
                          title="색상 피드백"
                        >
                          🎨
                        </button>
                        
                        {selectedProblem?.id === problem.id && (
                          <span className="text-3xl text-white animate-bounce-slow">✓</span>
                        )}
                      </div>
                    </div>
                    
                    {/* GPT-4 분석 결과 표시 */}
                    {problem.analysis && (
                      <div className={`mt-3 p-3 rounded-lg border ${
                        selectedProblem?.id === problem.id 
                          ? 'bg-white/20 border-white/30' 
                          : 'bg-blue-50 border-blue-200'
                      }`}>
                        <div className={`text-sm ${
                          selectedProblem?.id === problem.id ? 'text-white' : 'text-blue-800'
                        }`}>
                          <div className="flex justify-between items-center mb-2">
                            <strong>🤖 AI 분석:</strong>
                            <span className="font-bold">
                              {problem.difficulty || 'V?'} | {problem.type || '일반'}
                            </span>
                          </div>
                          {problem.reasoning && (
                            <div className="text-xs opacity-90">
                              {problem.reasoning}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <button 
              className="w-full py-4 glass-button text-primary-600 border-primary-500/30 rounded-2xl text-lg font-bold cursor-pointer mb-8 mx-auto flex items-center justify-center text-center transition-all duration-300 hover:translate-y-[-2px] hover:bg-gradient-to-r hover:from-primary-500 hover:to-purple-600 hover:text-white hover:border-transparent hover:shadow-lg active:translate-y-0"
              onClick={() => {
                setResult(null)
                setPreview(null)
                setImage(null)
                setSelectedProblem(null)
                setAnnotatedImage(null)
                setShowControlPanel(false)
              }}
            >
              ← 새로운 사진 업로드
            </button>
          </div>
        )}
          </>
        )}

        {/* 히스토리 뷰 */}
        {currentView === 'history' && <HistoryView />}

        {/* 즐겨찾기 뷰 */}
        {currentView === 'favorites' && <FavoritesView />}

        {/* 통계 뷰 */}
        {currentView === 'stats' && <StatsView />}

        {/* 비교 뷰 */}
        {compareMode && <CompareView />}

         {/* 이미지 확대 모달 */}
         {showImageModal && (
           <div className="fixed top-0 left-0 w-full h-full bg-black/90 flex items-center justify-center z-[1000] p-4" onClick={() => setShowImageModal(false)}>
             <img 
               src={annotatedImage || preview} 
               alt="Climbing Wall - 확대보기" 
               className="max-w-full max-h-full rounded-xl shadow-2xl"
               onClick={(e) => e.stopPropagation()}
             />
             <button 
               className="absolute top-4 right-4 bg-white/90 border-none rounded-full w-10 h-10 text-2xl cursor-pointer flex items-center justify-center text-slate-800 transition-all duration-300 hover:bg-white hover:scale-110"
               onClick={() => setShowImageModal(false)}
             >
               ×
             </button>
           </div>
         )}

         {/* 피드백 모달 */}
         {showFeedbackModal && selectedProblem && (
           <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[1000] p-4" onClick={() => setShowFeedbackModal(false)}>
             <div className="glass-card p-6 max-w-md w-full" onClick={(e) => e.stopPropagation()}>
               <h3 className="text-2xl font-extrabold gradient-text mb-4 text-center">
                 📝 피드백 제공하기
               </h3>
               <p className="text-sm text-slate-600 mb-6 text-center">
                 실제 난이도와 유형을 알려주시면<br/>
                 AI가 더 정확해집니다! 🙏
               </p>

               {/* 난이도 선택 */}
               <div className="mb-6">
                 <label className="block text-sm font-bold text-slate-700 mb-3">
                   🎯 실제 난이도
                 </label>
                 <div className="grid grid-cols-5 gap-2">
                   {['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10'].map(grade => (
                     <button
                       key={grade}
                       onClick={() => setFeedbackDifficulty(grade)}
                       className={`p-2 rounded-lg text-sm font-semibold transition-all ${
                         feedbackDifficulty === grade
                           ? 'bg-gradient-to-r from-primary-500 to-purple-600 text-white shadow-lg'
                           : 'bg-white/80 text-slate-600 hover:bg-white'
                       }`}
                     >
                       {grade}
                     </button>
                   ))}
                 </div>
               </div>

               {/* 유형 선택 */}
               <div className="mb-6">
                 <label className="block text-sm font-bold text-slate-700 mb-3">
                   🏋️ 실제 유형
                 </label>
                 <div className="grid grid-cols-2 gap-2">
                   {['다이나믹', '스태틱', '밸런스', '크림프', '슬로퍼', '트래버스', '캠퍼싱', '런지', '다이노', '코디네이션'].map(type => (
                     <button
                       key={type}
                       onClick={() => setFeedbackType(type)}
                       className={`p-3 rounded-lg text-sm font-semibold transition-all ${
                         feedbackType === type
                           ? 'bg-gradient-to-r from-primary-500 to-purple-600 text-white shadow-lg'
                           : 'bg-white/80 text-slate-600 hover:bg-white'
                       }`}
                     >
                       {type}
                     </button>
                   ))}
                 </div>
               </div>

               {/* 추가 의견 */}
               <div className="mb-6">
                 <label className="block text-sm font-bold text-slate-700 mb-2">
                   💬 추가 의견 (선택)
                 </label>
                 <textarea
                   value={feedbackText}
                   onChange={(e) => setFeedbackText(e.target.value)}
                   placeholder="예: 실제로는 발 사용이 중요해서 더 어려웠어요"
                   className="w-full p-3 rounded-lg border-2 border-gray-200 focus:border-primary-500 outline-none resize-none"
                   rows={3}
                 />
               </div>

               {/* 버튼 */}
               <div className="flex gap-3">
                 <button
                   onClick={() => setShowFeedbackModal(false)}
                   className="flex-1 py-3 bg-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-300 transition-all"
                 >
                   취소
                 </button>
                 <button
                   onClick={submitFeedback}
                   className="flex-1 py-3 bg-gradient-to-r from-pink-400 to-red-400 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all"
                 >
                   💾 저장
                 </button>
               </div>

               {/* 통계 표시 */}
               {modelStats && (
                 <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg text-xs">
                   <div className="flex justify-between items-center mb-2">
                     <span className="text-slate-600">📊 전체 데이터:</span>
                     <span className="font-bold text-slate-800">{modelStats.total_problems}개</span>
                   </div>
                   <div className="flex justify-between items-center mb-2">
                     <span className="text-slate-600">✅ 검증된 데이터:</span>
                     <span className="font-bold text-green-600">{modelStats.verified_problems}개</span>
                   </div>
                   {modelStats.verified_problems > 0 && (
                     <div className="mt-3 pt-3 border-t border-slate-200">
                       <div className="flex justify-between items-center">
                         <span className="text-slate-600">🎯 GPT-4 정확도:</span>
                         <span className="font-bold text-purple-600">
                           {modelStats.gpt4_difficulty_accuracy}%
                         </span>
                       </div>
                     </div>
                   )}
                   {modelStats.ready_for_training && (
                     <div className="mt-3 p-2 bg-green-100 rounded text-green-700 font-bold text-center">
                       ✨ 자체 AI 학습 가능!
                     </div>
                   )}
                 </div>
               )}
             </div>
           </div>
         )}

         {/* 홀드 색상 피드백 모달 */}
         {showHoldFeedbackModal && selectedHold && (
           <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[1000] p-4" onClick={() => setShowHoldFeedbackModal(false)}>
             <div className="glass-card p-6 max-w-md w-full" onClick={(e) => e.stopPropagation()}>
               <h3 className="text-2xl font-extrabold gradient-text mb-4 text-center">
                 🎨 홀드 색상 피드백
               </h3>
               <p className="text-sm text-slate-600 mb-6 text-center">
                 AI가 예측한 색상이 맞나요?<br/>
                 정확한 색상을 알려주시면 AI가 더 똑똑해집니다! 🙏
               </p>

               {/* 현재 예측된 색상 */}
               <div className="mb-6 space-y-3">
                 <div className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl border-2 border-purple-200">
                   <div className="text-center">
                     <p className="text-xs text-slate-600 mb-2">문제 그룹 색상</p>
                     <div className="flex items-center justify-center gap-2">
                       <span className="text-4xl">{colorEmoji[selectedHold.color] || '⭕'}</span>
                       <span className="text-xl font-bold gradient-text">{(selectedHold.color || 'UNKNOWN').toUpperCase()}</span>
                     </div>
                   </div>
                 </div>
                 
                 <div className="p-4 bg-gradient-to-r from-blue-50 to-green-50 rounded-xl border-2 border-blue-200">
                   <div className="text-center">
                     <p className="text-xs text-slate-600 mb-2">AI가 감지한 홀드 실제 색상</p>
                     <div className="flex items-center justify-center gap-2">
                       <span className="text-4xl">{colorEmoji[selectedHold.individual_color] || '⭕'}</span>
                       <span className="text-xl font-bold gradient-text">{(selectedHold.individual_color || 'UNKNOWN').toUpperCase()}</span>
                     </div>
                   </div>
                 </div>
               </div>

               {/* 실제 색상 선택 */}
               <div className="mb-6">
                 <label className="block text-sm font-bold text-slate-700 mb-3 text-center">
                   🎯 실제 홀드 색상
                 </label>
                 <div className="grid grid-cols-3 gap-3">
                   {Object.keys(colorEmoji).map(color => (
                     <button
                       key={color}
                       onClick={() => setHoldColorFeedback(color)}
                       className={`p-4 rounded-xl text-center transition-all ${
                         holdColorFeedback === color
                           ? 'bg-gradient-to-r from-primary-500 to-purple-600 text-white shadow-lg scale-105'
                           : 'bg-white/80 text-slate-600 hover:bg-white hover:scale-105'
                       }`}
                     >
                       <div className="text-3xl mb-1">{colorEmoji[color]}</div>
                       <div className="text-xs font-semibold">{color.toUpperCase()}</div>
                     </button>
                   ))}
                 </div>
               </div>

               {/* 버튼 */}
               <div className="flex gap-3">
                 <button
                   onClick={() => {
                     setShowHoldFeedbackModal(false)
                     setHoldColorFeedback('')
                   }}
                   className="flex-1 py-3 bg-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-300 transition-all"
                 >
                   취소
                 </button>
                 <button
                   onClick={submitHoldColorFeedback}
                   className="flex-1 py-3 bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all"
                 >
                   💾 저장
                 </button>
               </div>
             </div>
           </div>
         )}

        {/* 단계별 컴포넌트 */}
        
        {/* 2단계: 이미지 업로드 후 - 벽 각도 선택 + 분석 시작 */}
        {showControlPanel && !loading && !result && (
          <div className="glass-card p-6 my-8 w-full">
            <div className="space-y-6">
              {/* 벽 각도 선택 */}
              <div>
                <p className="font-bold text-slate-800 text-lg mb-4 text-center">
                  🏔️ 벽 각도 선택 (선택사항)
                </p>
                <div className="grid grid-cols-3 gap-3">
                  <button
                    className={`border-2 border-gray-300 glass-button p-4 rounded-2xl text-sm font-semibold cursor-pointer transition-all duration-300 text-slate-600 flex items-center justify-center ${
                      wallAngle === 'overhang' 
                        ? 'bg-gradient-to-r from-primary-500 to-purple-600 text-white border-transparent shadow-lg' 
                        : 'hover:translate-y-[-2px] hover:shadow-lg hover:bg-white/95'
                    }`}
                    onClick={() => setWallAngle(wallAngle === 'overhang' ? null : 'overhang')}
                  >
                    오버행
                  </button>
                  <button
                    className={`border-2 border-gray-300 glass-button p-4 rounded-2xl text-sm font-semibold cursor-pointer transition-all duration-300 text-slate-600 flex items-center justify-center ${
                      wallAngle === 'slab' 
                        ? 'bg-gradient-to-r from-primary-500 to-purple-600 text-white border-transparent shadow-lg' 
                        : 'hover:translate-y-[-2px] hover:shadow-lg hover:bg-white/95'
                    }`}
                    onClick={() => setWallAngle(wallAngle === 'slab' ? null : 'slab')}
                  >
                    슬랩
                  </button>
                  <button
                    className={`border-2 border-gray-300 glass-button p-4 rounded-2xl text-sm font-semibold cursor-pointer transition-all duration-300 text-slate-600 flex items-center justify-center ${
                      wallAngle === 'face' 
                        ? 'bg-gradient-to-r from-primary-500 to-purple-600 text-white border-transparent shadow-lg' 
                        : 'hover:translate-y-[-2px] hover:shadow-lg hover:bg-white/95'
                    }`}
                    onClick={() => setWallAngle(wallAngle === 'face' ? null : 'face')}
                  >
                    직벽
                  </button>
                </div>
              </div>

              {/* 분석 시작 버튼 */}
              <button
                className="w-full py-4 bg-gradient-to-r from-pink-400 to-red-400 text-white border-none rounded-2xl text-xl font-bold cursor-pointer transition-all duration-300 flex items-center justify-center gap-3 text-center shadow-lg hover:translate-y-[-3px] hover:shadow-xl hover:from-pink-500 hover:to-red-500 active:translate-y-[-1px]"
                onClick={analyzeImage}
              >
                🔍 문제 분석 시작
              </button>
            </div>
          </div>
        )}

        {/* 하단 네비게이션 바 */}
        <div className="fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-md border-t border-slate-200 shadow-lg z-50">
          <div className="grid grid-cols-4 max-w-screen-lg mx-auto">
            <button
              onClick={() => setCurrentView('analyze')}
              className={`flex flex-col items-center justify-center py-3 transition-all ${
                currentView === 'analyze'
                  ? 'text-blue-600'
                  : 'text-slate-600 hover:text-blue-500'
              }`}
            >
              <span className="text-2xl mb-1">📸</span>
              <span className="text-xs font-medium">분석</span>
            </button>
            
            <button
              onClick={() => setCurrentView('history')}
              className={`flex flex-col items-center justify-center py-3 transition-all ${
                currentView === 'history'
                  ? 'text-blue-600'
                  : 'text-slate-600 hover:text-blue-500'
              }`}
            >
              <span className="text-2xl mb-1">📚</span>
              <span className="text-xs font-medium">히스토리</span>
            </button>
            
            <button
              onClick={() => setCurrentView('favorites')}
              className={`flex flex-col items-center justify-center py-3 transition-all ${
                currentView === 'favorites'
                  ? 'text-blue-600'
                  : 'text-slate-600 hover:text-blue-500'
              }`}
            >
              <span className="text-2xl mb-1">⭐</span>
              <span className="text-xs font-medium">즐겨찾기</span>
            </button>
            
            <button
              onClick={() => setCurrentView('stats')}
              className={`flex flex-col items-center justify-center py-3 transition-all ${
                currentView === 'stats'
                  ? 'text-blue-600'
                  : 'text-slate-600 hover:text-blue-500'
              }`}
            >
              <span className="text-2xl mb-1">📊</span>
              <span className="text-xs font-medium">통계</span>
            </button>
          </div>
        </div>
      </div>

      {/* 색상 피드백 모달 */}
      {colorFeedbackProblem && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[100] p-4"
             onClick={() => setColorFeedbackProblem(null)}>
          <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl"
               onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold text-slate-800">🎨 색상 피드백</h3>
              <button
                onClick={() => setColorFeedbackProblem(null)}
                className="text-3xl text-slate-400 hover:text-slate-600 transition-colors"
              >
                ×
              </button>
            </div>
            
            <div className="mb-6">
              <p className="text-slate-600 mb-4">
                현재 색상: <span className="font-bold text-lg">{colorFeedbackProblem.color_name}</span>
              </p>
              <p className="text-sm text-slate-500">올바른 색상을 선택해주세요:</p>
            </div>
            
            <div className="grid grid-cols-3 gap-3 mb-6">
              {['red', 'orange', 'yellow', 'green', 'mint', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'gray'].map(color => (
                <button
                  key={color}
                  onClick={async () => {
                    try {
                      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/color-feedback`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          hold_ids: colorFeedbackProblem.holds.map(h => h.id),
                          predicted_color: colorFeedbackProblem.color_name,
                          correct_color: color
                        })
                      })
                      
                      if (response.ok) {
                        alert('피드백이 제출되었습니다! 감사합니다 🎉')
                        setColorFeedbackProblem(null)
                      } else {
                        alert('피드백 제출에 실패했습니다.')
                      }
                    } catch (error) {
                      console.error('피드백 제출 오류:', error)
                      alert('피드백 제출 중 오류가 발생했습니다.')
                    }
                  }}
                  className="py-3 px-4 rounded-xl border-2 border-slate-200 hover:border-purple-500 hover:bg-purple-50 transition-all font-medium text-slate-700 hover:text-purple-700"
                >
                  {colorEmoji[color] || '⭕'} {color}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App