import axios from 'axios'
import { useEffect, useState } from 'react'
import FavoritesModal from './components/FavoritesModal'
import Header from './components/Header'
import HeroSection from './components/HeroSection'
import HistoryModal from './components/HistoryModal'
import Loading from './components/Loading'
import OnboardingModal from './components/OnboardingModal'

const API_URL = 'http://localhost:8000'

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState({ message: '', progress: 0 })
  const [result, setResult] = useState(null)
  const [selectedProblem, setSelectedProblem] = useState(null)
  const [wallAngle, setWallAngle] = useState(null)
  const [annotatedImage, setAnnotatedImage] = useState(null)
  const [showImageModal, setShowImageModal] = useState(false)
  const [showBottomSheet, setShowBottomSheet] = useState(false)
  const [isSheetExpanded, setIsSheetExpanded] = useState(false)
  const [showFeedbackModal, setShowFeedbackModal] = useState(false)
  const [feedbackDifficulty, setFeedbackDifficulty] = useState('')
  const [feedbackType, setFeedbackType] = useState('')
  const [feedbackText, setFeedbackText] = useState('')
  const [modelStats, setModelStats] = useState(null)
  const [analysisHistory, setAnalysisHistory] = useState([])
  const [favorites, setFavorites] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [showFavorites, setShowFavorites] = useState(false)
  const [compareMode, setCompareMode] = useState(false)
  const [selectedForCompare, setSelectedForCompare] = useState([])
  const [showOnboarding, setShowOnboarding] = useState(false)

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

  // 컴포넌트 마운트 시 통계 로드 및 온보딩 체크
  useEffect(() => {
    loadStats()
    
    // 온보딩 체크
    const hasSeenOnboarding = localStorage.getItem('climbmate-onboarding')
    if (!hasSeenOnboarding) {
      setShowOnboarding(true)
    }
  }, [])

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(file)
      setResult(null)
      setSelectedProblem(null)
      setAnnotatedImage(null)
      setShowBottomSheet(true)
      setIsSheetExpanded(true)
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
          setAnnotatedImage(null)
          setShowBottomSheet(true)
          setIsSheetExpanded(true)
          
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


  const analyzeImage = async () => {
    if (!image) return

    setLoading(true)
    setIsSheetExpanded(false)
    
    // 로딩 단계별 메시지 (진행률 기반)
    const loadingSteps = [
      { progress: 0, message: '📸 이미지 업로드 중...' },
      { progress: 5, message: '🔍 YOLO 모델 로딩 중...' },
      { progress: 10, message: '🎯 홀드 감지 중...' },
      { progress: 30, message: '✂️ 홀드 영역 추출 중...' },
      { progress: 40, message: '🎨 색상 전처리 중...' },
      { progress: 50, message: '🤖 CLIP AI 분석 중...' },
      { progress: 60, message: '📊 홀드 그룹화 중...' },
      { progress: 80, message: '🧮 통계 계산 중...' },
      { progress: 95, message: '📈 난이도 분석 중...' }
    ]
    
    let currentProgress = 0
    const startTime = Date.now()
    let progressInterval = null
    
    // 진행률 시뮬레이션 (실제 응답 시간에 따라 조절)
    const updateProgress = () => {
      const elapsed = Date.now() - startTime
      
      // 시간에 따라 진행률 증가 (처음엔 빠르게, 나중엔 느리게)
      if (elapsed < 5000) {
        currentProgress = Math.min(currentProgress + 2, 15)
      } else if (elapsed < 10000) {
        currentProgress = Math.min(currentProgress + 1.5, 35)
      } else if (elapsed < 20000) {
        currentProgress = Math.min(currentProgress + 1, 75)
      } else {
        currentProgress = Math.min(currentProgress + 0.5, 95)
      }
      
      // 현재 진행률에 맞는 메시지 찾기
      const currentStep = [...loadingSteps]
        .reverse()
        .find(step => currentProgress >= step.progress)
      
      if (currentStep) {
        setLoadingStep({ message: currentStep.message, progress: currentProgress })
      }
    }
    
    progressInterval = setInterval(updateProgress, 500)
    
    const formData = new FormData()
    formData.append('file', image)
    if (wallAngle) formData.append('wall_angle', wallAngle)

    try {
      const response = await axios.post(`${API_URL}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      // 진행률 업데이트 중지
      if (progressInterval) clearInterval(progressInterval)
      
      setLoadingStep({ message: '✅ 분석 완료!', progress: 100 })
      
      // 완료 메시지를 잠깐 보여주고 결과 표시
      setTimeout(() => {
        setResult(response.data)
        
        // 📚 분석 결과를 히스토리에 저장
        const historyItem = {
          id: Date.now(),
          timestamp: new Date().toLocaleString('ko-KR'),
          image: preview,
          result: response.data,
          wallAngle: wallAngle
        }
        setAnalysisHistory(prev => [historyItem, ...prev.slice(0, 19)]) // 최대 20개 저장
        
        // 주석 달린 이미지 생성 (색상별로 홀드 표시)
        if (response.data.annotated_image_base64) {
          setAnnotatedImage(`data:image/jpeg;base64,${response.data.annotated_image_base64}`)
        }
        
        setLoading(false)
        setLoadingStep('')
      }, 500)
      
    } catch (error) {
      // 진행률 업데이트 중지
      if (progressInterval) clearInterval(progressInterval)
      
      console.error('분석 실패:', error)
      setLoadingStep('❌ 분석 실패')
      setTimeout(() => {
        alert('분석에 실패했습니다. 다시 시도해주세요.')
        setLoading(false)
        setLoadingStep('')
      }, 1000)
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

  // 즐겨찾기 추가/제거
  const toggleFavorite = (problem) => {
    const isFavorite = favorites.some(fav => fav.id === problem.id)
    if (isFavorite) {
      setFavorites(prev => prev.filter(fav => fav.id !== problem.id))
    } else {
      setFavorites(prev => [...prev, { ...problem, addedAt: new Date().toLocaleString('ko-KR') }])
    }
  }

  // 히스토리에서 분석 결과 로드
  const loadFromHistory = (historyItem) => {
    setResult(historyItem.result)
    setSelectedProblem(historyItem.result.problems[0])
    setPreview(historyItem.image)
    setWallAngle(historyItem.wallAngle)
    setAnnotatedImage(null)
    setShowHistory(false)
    setShowBottomSheet(true)
  }

  // 비교 모드 토글
  const toggleCompareMode = () => {
    setCompareMode(!compareMode)
    setSelectedForCompare([])
  }

  // 비교 대상 선택/해제
  const toggleForCompare = (problem) => {
    if (selectedForCompare.includes(problem.id)) {
      setSelectedForCompare(prev => prev.filter(id => id !== problem.id))
    } else if (selectedForCompare.length < 3) {
      setSelectedForCompare(prev => [...prev, problem.id])
    }
  }

  const handleImageClick = (e) => {
    if (!result || !result.problems) return
    
    const rect = e.target.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * 100
    const y = ((e.clientY - rect.top) / rect.height) * 100
    
    // 클릭 위치에서 가장 가까운 홀드 찾기
    let closestProblem = null
    let minDistance = Infinity
    
    result.problems.forEach(problem => {
      problem.holds.forEach(hold => {
        const holdX = (hold.center[0] / result.image_width) * 100
        const holdY = (hold.center[1] / result.image_height) * 100
        const distance = Math.sqrt(Math.pow(x - holdX, 2) + Math.pow(y - holdY, 2))
        
        if (distance < minDistance && distance < 10) { // 10% 반경 내
          minDistance = distance
          closestProblem = problem
        }
      })
    })
    
    if (closestProblem) {
      setSelectedProblem(closestProblem)
    }
  }

  const colorEmoji = {
    black: '⚫', white: '⚪', gray: '🔘',
    red: '🔴', orange: '🟠', yellow: '🟡',
    green: '🟢', blue: '🔵', purple: '🟣',
    pink: '🩷', brown: '🟤', mint: '💚', lime: '🍃'
  }

  return (
    <div className="w-full py-4 px-4 min-h-screen flex flex-col items-center">
      {/* 헤더 */}
      <Header
        modelStats={modelStats}
        analysisHistory={analysisHistory}
        favorites={favorites}
        compareMode={compareMode}
        onShowHistory={() => setShowHistory(!showHistory)}
        onShowFavorites={() => setShowFavorites(!showFavorites)}
        onToggleCompare={toggleCompareMode}
        onShowOnboarding={() => setShowOnboarding(true)}
      />
         {preview && (
           <div className="relative mb-4">
             <img 
               src={annotatedImage || preview} 
               alt="Climbing Wall" 
               className={`w-full max-h-[500px] object-contain rounded-2xl mx-auto shadow-2xl border border-white/20 block ${
                 result ? 'cursor-pointer hover:opacity-90 transition-opacity' : ''
               }`}
               onClick={result ? handleImageClick : undefined}
               onDoubleClick={result ? () => setShowImageModal(true) : undefined}
             />
             {result && selectedProblem && (
               <div className="absolute top-4 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-full text-base font-bold shadow-lg animate-pulse-slow">
                 {colorEmoji[selectedProblem.color_name]} {selectedProblem.color_name.toUpperCase()} 선택됨
               </div>
             )}
             {result && (
               <p className="text-center text-xs mt-2 text-slate-600">
                 💡 홀드 클릭으로 문제 선택 | 더블클릭으로 확대
               </p>
             )}
           </div>
         )}

        {/* 이미지 업로드 */}
        {!result && !loading && !preview && (
          <HeroSection
            modelStats={modelStats}
            onFileSelect={handleImageUpload}
            onCameraCapture={handleCameraCapture}
          />
        )}

         {/* 로딩 */}
         {loading && <Loading loadingStep={loadingStep} />}

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

             {/* 선택된 문제 상세 */}
             {selectedProblem && selectedProblem.analysis && (
               <div className="glass-card p-6 mx-auto mb-6 w-full text-center shadow-lg">
                 <div className="flex justify-between items-center mb-4">
                   <h3 className="text-2xl text-slate-800 font-extrabold flex-1">
                     {colorEmoji[selectedProblem.color_name]} {selectedProblem.color_name.toUpperCase()} 문제
                   </h3>
                   <button
                     onClick={() => setShowFeedbackModal(true)}
                     className="px-4 py-2 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-xl text-sm font-semibold shadow-md hover:shadow-lg transition-all"
                   >
                     📝 피드백
                   </button>
                 </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div className="bg-white/80 backdrop-blur-sm p-5 rounded-xl shadow-md transition-all duration-300 hover:translate-y-[-3px] hover:shadow-lg">
                    <h4 className="text-sm mb-3 text-slate-600 font-semibold text-center">🎯 난이도</h4>
                    <div className="text-3xl font-extrabold gradient-text text-center mb-1">{selectedProblem.analysis.difficulty.grade}</div>
                    <div className="text-sm text-slate-600 mb-2 font-medium text-center">{selectedProblem.analysis.difficulty.level}</div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${selectedProblem.analysis.difficulty.confidence * 100}%` }}
                      ></div>
                    </div>
                    <div className="text-xs text-center text-gray-500">
                      신뢰도: {Math.round(selectedProblem.analysis.difficulty.confidence * 100)}%
                    </div>
                  </div>

                  <div className="bg-white/80 backdrop-blur-sm p-5 rounded-xl shadow-md transition-all duration-300 hover:translate-y-[-3px] hover:shadow-lg">
                    <h4 className="text-sm mb-3 text-slate-600 font-semibold text-center">🏋️ 유형</h4>
                    <div className="text-lg font-bold text-slate-800 mb-2 text-center">{selectedProblem.analysis.climb_type.primary_type}</div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${selectedProblem.analysis.climb_type.confidence * 100}%` }}
                      ></div>
                    </div>
                    <div className="text-xs text-center text-gray-500 mb-3">
                      신뢰도: {Math.round(selectedProblem.analysis.climb_type.confidence * 100)}%
                    </div>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {selectedProblem.analysis.climb_type.types.slice(0, 3).map((type, idx) => (
                        <span key={idx} className="px-3 py-1 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-full text-xs font-semibold shadow-md">
                          {type}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                
                {/* 상세 통계 */}
                <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl">
                  <h4 className="font-bold text-slate-800 mb-3 text-center">📊 상세 통계</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <div className="text-center">
                      <p className="text-gray-600">홀드 수</p>
                      <p className="font-bold text-lg text-slate-800">{selectedProblem.hold_count}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-600">평균 크기</p>
                      <p className="font-bold text-lg text-slate-800">{Math.round(selectedProblem.avg_hold_size)}px²</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-600">최대 간격</p>
                      <p className="font-bold text-lg text-slate-800">{selectedProblem.max_hold_distance}px</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-600">평균 간격</p>
                      <p className="font-bold text-lg text-slate-800">{Math.round(selectedProblem.avg_hold_distance)}px</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-md">
                  <h4 className="text-sm mb-3 text-slate-800 font-bold text-center">📊 상세 정보</h4>
                  <div className="flex justify-between items-center py-2 border-b border-slate-200 text-sm transition-all duration-200 hover:bg-white/50 hover:px-4 hover:rounded-lg hover:-mx-4">
                    <span className="text-slate-600 font-medium">홀드 개수:</span>
                    <span className="font-bold gradient-text">{selectedProblem.hold_count}개</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-200 text-sm transition-all duration-200 hover:bg-white/50 hover:px-4 hover:rounded-lg hover:-mx-4">
                    <span className="text-slate-600 font-medium">평균 크기:</span>
                    <span className="font-bold gradient-text">{selectedProblem.analysis.statistics.avg_hold_size}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 text-sm transition-all duration-200 hover:bg-white/50 hover:px-4 hover:rounded-lg hover:-mx-4">
                    <span className="text-slate-600 font-medium">최대 간격:</span>
                    <span className="font-bold gradient-text">{selectedProblem.analysis.statistics.max_distance}</span>
                  </div>
                </div>
              </div>
            )}

            {/* 문제 목록 (축약) */}
            <details className="glass-card p-8 mx-auto mb-8 w-full text-center shadow-lg">
              <summary className="font-bold text-slate-800 cursor-pointer text-xl p-4 rounded-xl transition-all duration-300 list-none flex items-center justify-center gap-2 hover:bg-white/80 hover:translate-y-[-2px]">
                📋 전체 문제 목록 ({result.problems.length}개)
              </summary>
              <div className="mt-6 flex flex-col gap-4">
                {result.problems.map((problem) => (
                  <div 
                    key={problem.id} 
                    className={`flex items-center p-6 bg-white/80 backdrop-blur-sm rounded-2xl transition-all duration-300 border-2 shadow-md ${
                      compareMode && selectedForCompare.includes(problem.id)
                        ? 'ring-2 ring-purple-500 bg-purple-50 border-purple-300'
                        : selectedProblem?.id === problem.id 
                          ? 'bg-gradient-to-r from-primary-500 to-purple-600 text-white border-transparent shadow-lg translate-y-[-3px]' 
                          : 'border-white/30 hover:translate-y-[-3px] hover:shadow-lg hover:bg-white/95'
                    } ${!compareMode ? 'cursor-pointer' : ''}`}
                    onClick={!compareMode ? () => setSelectedProblem(problem) : undefined}
                  >
                    <span className="text-3xl mr-5">{colorEmoji[problem.color_name] || '⭕'}</span>
                    <div className="flex-1">
                      <div className={`text-xl font-bold mb-1 ${
                        selectedProblem?.id === problem.id ? 'text-white' : 'text-slate-800'
                      }`}>
                        {problem.color_name.toUpperCase()}
                      </div>
                      <div className={`text-base font-medium ${
                        selectedProblem?.id === problem.id ? 'text-white' : 'text-slate-600'
                      }`}>
                        {problem.hold_count}개 홀드
                      </div>
                    </div>
                    
                    {/* 액션 버튼들 */}
                    <div className="flex items-center space-x-2">
                      {/* 즐겨찾기 버튼 */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          toggleFavorite(problem)
                        }}
                        className={`text-xl transition-colors ${
                          favorites.some(fav => fav.id === problem.id)
                            ? 'text-yellow-500' 
                            : 'text-gray-400 hover:text-yellow-400'
                        }`}
                      >
                        ⭐
                      </button>
                      
                      {/* 비교 모드 버튼 */}
                      {compareMode && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            toggleForCompare(problem)
                          }}
                          className={`px-3 py-1 rounded-full text-xs font-semibold transition-all ${
                            selectedForCompare.includes(problem.id)
                              ? 'bg-purple-500 text-white'
                              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                          }`}
                        >
                          {selectedForCompare.includes(problem.id) ? '선택됨' : '선택'}
                        </button>
                      )}
                      
                      {/* 선택 표시 */}
                      {!compareMode && selectedProblem?.id === problem.id && (
                        <span className="text-3xl text-white animate-bounce-slow">✓</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </details>

            {/* 비교 모드 - 선택된 문제들 비교 */}
            {compareMode && selectedForCompare.length > 0 && (
              <div className="glass-card p-6 mb-6">
                <h3 className="text-xl font-bold text-slate-800 mb-4 text-center">
                  ⚖️ 문제 비교 ({selectedForCompare.length}개 선택됨)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {selectedForCompare.map(problemId => {
                    const problem = result.problems.find(p => p.id === problemId)
                    if (!problem) return null
                    
                    return (
                      <div key={problemId} className="bg-white/80 rounded-xl p-4 border border-purple-200">
                        <div className="text-center mb-3">
                          <span className="text-2xl mr-2">{colorEmoji[problem.color_name] || '⭕'}</span>
                          <span className="font-bold text-slate-800">{problem.color_name.toUpperCase()}</span>
                        </div>
                        
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">난이도:</span>
                            <span className="font-semibold text-blue-600">{problem.difficulty.grade}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">유형:</span>
                            <span className="font-semibold text-green-600">{problem.climb_type.primary_type}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">홀드 수:</span>
                            <span className="font-semibold">{problem.hold_count}개</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">신뢰도:</span>
                            <span className="font-semibold text-purple-600">
                              {Math.round(problem.difficulty.confidence * 100)}%
                            </span>
                          </div>
                        </div>
                        
                        <button
                          onClick={() => toggleForCompare(problem)}
                          className="w-full mt-3 py-2 bg-red-100 text-red-600 rounded-lg text-sm font-medium hover:bg-red-200 transition-colors"
                        >
                          비교에서 제거
                        </button>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            <button 
              className="w-full py-4 glass-button text-primary-600 border-primary-500/30 rounded-2xl text-lg font-bold cursor-pointer mb-8 mx-auto flex items-center justify-center text-center transition-all duration-300 hover:translate-y-[-2px] hover:bg-gradient-to-r hover:from-primary-500 hover:to-purple-600 hover:text-white hover:border-transparent hover:shadow-lg active:translate-y-0"
              onClick={() => {
                setResult(null)
                setPreview(null)
                setImage(null)
                setSelectedProblem(null)
                setAnnotatedImage(null)
                setIsSheetExpanded(false)
              }}
            >
              ← 새로운 사진 업로드
            </button>
          </div>
        )}

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

        {/* 바텀시트 */}
        {showBottomSheet && (
          <>
            {/* 오버레이 (확장 시에만) */}
            {isSheetExpanded && (
              <div 
                className="fixed inset-0 bg-black bg-opacity-30 z-30 transition-opacity duration-300"
                onClick={() => setIsSheetExpanded(false)}
              />
            )}
            
            <div 
              className={`fixed inset-x-0 bottom-0 bg-white rounded-t-3xl shadow-2xl z-40 transition-all duration-500 ease-out`}
              style={{
                transform: isSheetExpanded 
                  ? 'translateY(0)' 
                  : 'translateY(calc(100% - 30px))'
              }}
            >
              {/* 드래그 핸들 */}
              <div 
                className="flex justify-center pt-3 pb-3 cursor-pointer"
                onClick={() => setIsSheetExpanded(!isSheetExpanded)}
              >
                <div className="w-20 h-1.5 bg-gray-300 rounded-full hover:bg-gray-400 transition-colors shadow-sm"></div>
              </div>
              
              <div className="px-6 pb-8 space-y-6">
                {/* 확장 시에만 보이는 옵션들 */}
                {isSheetExpanded && (
                  <div className="space-y-6 animate-fadeIn">
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
                    
                    {/* 추가 여백 */}
                    <div className="pb-4"></div>
                  </div>
                )}
              </div>
      </div>
          </>
        )}

      {/* 히스토리 모달 */}
      <HistoryModal
        show={showHistory}
        onClose={() => setShowHistory(false)}
        history={analysisHistory}
        onLoadHistory={loadFromHistory}
      />

      {/* 즐겨찾기 모달 */}
      <FavoritesModal
        show={showFavorites}
        onClose={() => setShowFavorites(false)}
        favorites={favorites}
        onToggleFavorite={toggleFavorite}
        colorEmoji={colorEmoji}
      />

      {/* 온보딩 모달 */}
      <OnboardingModal
        show={showOnboarding}
        onClose={() => setShowOnboarding(false)}
      />
    </div>
  )
}

export default App