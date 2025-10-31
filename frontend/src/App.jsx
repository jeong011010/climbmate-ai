import { useRef, useState } from 'react'
import * as api from './api'
import AnalyzeLayout from './components/AnalyzeLayout'
import Header from './components/Header'
import Modals from './components/Modals'
import Navigation from './components/Navigation'
import { colorEmoji } from './constants'
import { useAnalysis } from './hooks/useAnalysis'
import { useFeedback } from './hooks/useFeedback'
import { useHistory } from './hooks/useHistory'
import { useImageUpload } from './hooks/useImageUpload'
import ComparePage from './pages/ComparePage'
import FeedbacksPage from './pages/FeedbacksPage'
import HistoryPage from './pages/HistoryPage'
import LoadingPage from './pages/LoadingPage'
import MainPage from './pages/MainPage'
import StatsPage from './pages/StatsPage'

function App() {
  // ===== 상태 관리 =====
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [detectedHolds, setDetectedHolds] = useState(0)
  const [detectedProblems, setDetectedProblems] = useState(0)
  const [currentAnalysisStep, setCurrentAnalysisStep] = useState('')
  const [result, setResult] = useState(null)
  const [selectedProblem, setSelectedProblem] = useState(null)
  const [selectedHold, setSelectedHold] = useState(null)
  const [wallAngle, setWallAngle] = useState(null)
  const [annotatedImage, setAnnotatedImage] = useState(null)
  const [showImageModal, setShowImageModal] = useState(false)
  const [showFeedbackModal, setShowFeedbackModal] = useState(false)
  const [feedbackDifficulty, setFeedbackDifficulty] = useState('')
  const [feedbackType, setFeedbackType] = useState('')
  const [feedbackText, setFeedbackText] = useState('')
  const [showHoldFeedbackModal, setShowHoldFeedbackModal] = useState(false)
  const [holdColorFeedback, setHoldColorFeedback] = useState('')
  const [modelStats, setModelStats] = useState(null)
  const [analysisHistory, setAnalysisHistory] = useState([])
  const [currentView, setCurrentView] = useState('analyze')
  const [compareMode, setCompareMode] = useState(false)
  const [selectedForCompare, setSelectedForCompare] = useState([])
  const [colorFeedbacks, setColorFeedbacks] = useState([])
  const [feedbacksLoading, setFeedbacksLoading] = useState(false)
  
  // Ref
  const imageRef = useRef(null)
  const [imageLoaded, setImageLoaded] = useState(false)

  // ===== Custom Hooks =====
  const { handleImageUpload, handleCameraCapture } = useImageUpload({
    setImage,
    setPreview,
    setResult,
    setSelectedProblem,
    setSelectedHold,
    setAnnotatedImage,
    setShowControlPanel: () => {},
    setImageLoaded
  })

  const { saveToHistory } = useHistory(analysisHistory, setAnalysisHistory, preview, wallAngle)

  const {
    loadStats,
    loadColorFeedbacks,
    confirmFeedback,
    confirmAllFeedbacks,
    deleteFeedback,
    deleteAllFeedbacks,
    exportColorFeedbacks,
    trainColorModel,
    checkGpt4Status,
    testGpt4,
    convertGpt4ToTraining
  } = useFeedback({
    setModelStats,
    setColorFeedbacks,
    setFeedbacksLoading,
    colorFeedbacks
  })

  const { analyzeImage } = useAnalysis({
    image,
    wallAngle,
    setLoading,
    setLoadingProgress,
    setDetectedHolds,
    setDetectedProblems,
    setCurrentAnalysisStep,
    setResult,
    saveToHistory
  })

  // ===== Helper Functions =====
  const toggleCompareMode = () => {
    setCompareMode(!compareMode)
    setSelectedForCompare([])
  }

  // 향후 비교 기능 확장 시 사용
  // const toggleProblemForCompare = (problemId) => {
  //   if (selectedForCompare.includes(problemId)) {
  //     setSelectedForCompare(selectedForCompare.filter(id => id !== problemId))
  //   } else if (selectedForCompare.length < 3) {
  //     setSelectedForCompare([...selectedForCompare, problemId])
  //   }
  // }

  const submitFeedback = async () => {
    if (!selectedProblem || !selectedProblem.db_id || !feedbackDifficulty || !feedbackType) {
      alert('문제 ID를 찾을 수 없거나 필수 정보가 누락되었습니다.')
      return
    }
    try {
      const data = await api.submitProblemFeedback({
        problem_id: selectedProblem.db_id,
        user_difficulty: feedbackDifficulty,
        user_type: feedbackType,
        user_feedback: feedbackText
      })
      alert(data.message)
      setModelStats(data.stats)
      setShowFeedbackModal(false)
      setFeedbackDifficulty('')
      setFeedbackType('')
      setFeedbackText('')
      loadStats()
    } catch (error) {
      console.error('피드백 제출 실패:', error)
      alert('피드백 제출에 실패했습니다.')
    }
  }

  const submitHoldColorFeedback = async () => {
    if (!selectedHold || !selectedProblem || !holdColorFeedback) {
      alert('필수 정보가 누락되었습니다.')
      return
    }
    try {
      const holdFeatures = {
        dominant_rgb: selectedHold.rgb || [128, 128, 128],
        dominant_hsv: selectedHold.hsv || [0, 0, 128],
        dominant_lab: selectedHold.dominant_lab || [0, 0, 0],
        hsv_stats: selectedHold.hsv_stats || {},
        rgb_stats: selectedHold.rgb_stats || {},
        lab_stats: selectedHold.lab_stats || {},
        area: selectedHold.area || 0,
        circularity: selectedHold.circularity || 0
      }
      await api.submitHoldColorFeedback({
        problem_id: selectedProblem.db_id || 0,
        hold_id: String(selectedHold.id || `${selectedHold.center[0]}_${selectedHold.center[1]}`),
        predicted_color: selectedHold.individual_color || selectedHold.color,
        user_color: holdColorFeedback,
        hold_center: selectedHold.center,
        hold_features: holdFeatures
      })
      alert('홀드 색상 피드백이 제출되었습니다! ML 학습에 활용됩니다 🤖')
      setShowHoldFeedbackModal(false)
      setHoldColorFeedback('')
      loadStats()
    } catch (error) {
      console.error('홀드 색상 피드백 제출 실패:', error)
      alert('홀드 색상 피드백 제출에 실패했습니다.')
    }
  }

  const handleImageClick = (e) => {
    if (!result || !result.problems || !imageRef.current) return
    
    e.preventDefault()
    
    // 터치 이벤트와 마우스 이벤트 모두 지원
    const clientX = e.type === 'touchend' 
      ? e.changedTouches[0].clientX 
      : e.clientX
    const clientY = e.type === 'touchend' 
      ? e.changedTouches[0].clientY 
      : e.clientY
    
    const rect = e.target.getBoundingClientRect()
    const clickX = ((clientX - rect.left) / rect.width) * imageRef.current.naturalWidth
    const clickY = ((clientY - rect.top) / rect.height) * imageRef.current.naturalHeight

    let closestHold = null
    let closestDistance = Infinity
    let closestProblem = null

    for (const problem of result.problems) {
      if (!problem.holds) continue
      
      for (const hold of problem.holds) {
        const distance = Math.sqrt(
          Math.pow(clickX - hold.center[0], 2) + 
          Math.pow(clickY - hold.center[1], 2)
        )
        
        if (distance < closestDistance) {
          closestDistance = distance
          closestHold = { ...hold, color: problem.color_name }
          closestProblem = problem
        }
      }
    }

    if (closestProblem) {
      setSelectedProblem(closestProblem)
      setSelectedHold(closestHold)
    } else {
      setSelectedHold(null)
    }
  }

  // ===== 렌더링 =====
  return (
    <div className="w-full min-h-screen flex flex-col items-center">
      {/* 헤더 */}
      <Header 
        modelStats={modelStats}
        checkGpt4Status={checkGpt4Status}
        testGpt4={testGpt4}
      />

      {/* 메인 컨텐츠 영역 */}
      <div className="w-full pt-36 pb-20 px-2 sm:px-4">
        {currentView === 'analyze' && (
          <>
            {/* 분석 결과 화면 */}
            {preview && result && (
              <AnalyzeLayout
                preview={preview}
                annotatedImage={annotatedImage}
                result={result}
                selectedProblem={selectedProblem}
                selectedHold={selectedHold}
                imageLoaded={imageLoaded}
                setImageLoaded={setImageLoaded}
                setSelectedHold={setSelectedHold}
                setShowHoldFeedbackModal={setShowHoldFeedbackModal}
                setShowFeedbackModal={setShowFeedbackModal}
                setShowImageModal={setShowImageModal}
                handleImageClick={handleImageClick}
                imageRef={imageRef}
                onProblemSelect={(problem) => {
                  setSelectedProblem(problem)
                  setSelectedHold(null)
                }}
                colorEmoji={colorEmoji}
              />
            )}

            {/* 메인 페이지 (초기 화면 + 업로드 후) */}
            {!loading && !result && (
              <MainPage
                preview={preview}
                handleImageUpload={handleImageUpload}
                handleCameraCapture={handleCameraCapture}
                analyzeImage={analyzeImage}
                wallAngle={wallAngle}
                setWallAngle={setWallAngle}
              />
            )}

            {/* 로딩 페이지 */}
            {loading && (
              <LoadingPage
                preview={preview}
                loadingProgress={loadingProgress}
                currentAnalysisStep={currentAnalysisStep}
                detectedHolds={detectedHolds}
                detectedProblems={detectedProblems}
              />
            )}
          </>
        )}

        {/* 히스토리 페이지 */}
        {currentView === 'history' && (
          <HistoryPage 
            analysisHistory={analysisHistory}
            setResult={setResult}
            setPreview={setPreview}
            setCurrentView={setCurrentView}
          />
        )}

        {/* 통계 페이지 */}
        {currentView === 'stats' && (
          <StatsPage 
            modelStats={modelStats}
            convertGpt4ToTraining={convertGpt4ToTraining}
          />
        )}
        
        {/* 피드백 관리 페이지 */}
        {currentView === 'feedbacks' && (
          <FeedbacksPage 
            feedbacksLoading={feedbacksLoading}
            colorFeedbacks={colorFeedbacks}
            loadColorFeedbacks={loadColorFeedbacks}
            trainColorModel={trainColorModel}
            confirmFeedback={confirmFeedback}
            confirmAllFeedbacks={confirmAllFeedbacks}
            deleteFeedback={deleteFeedback}
            deleteAllFeedbacks={deleteAllFeedbacks}
            exportColorFeedbacks={exportColorFeedbacks}
          />
        )}

        {/* 비교 페이지 */}
        {compareMode && (
          <ComparePage 
            result={result}
            selectedForCompare={selectedForCompare}
            toggleCompareMode={toggleCompareMode}
            colorEmoji={colorEmoji}
          />
        )}

        {/* 모달들 */}
        <Modals
          showImageModal={showImageModal}
          setShowImageModal={setShowImageModal}
          annotatedImage={annotatedImage}
          preview={preview}
          showFeedbackModal={showFeedbackModal}
          setShowFeedbackModal={setShowFeedbackModal}
          selectedProblem={selectedProblem}
          feedbackDifficulty={feedbackDifficulty}
          setFeedbackDifficulty={setFeedbackDifficulty}
          feedbackType={feedbackType}
          setFeedbackType={setFeedbackType}
          feedbackText={feedbackText}
          setFeedbackText={setFeedbackText}
          submitFeedback={submitFeedback}
          showHoldFeedbackModal={showHoldFeedbackModal}
          setShowHoldFeedbackModal={setShowHoldFeedbackModal}
          selectedHold={selectedHold}
          holdColorFeedback={holdColorFeedback}
          setHoldColorFeedback={setHoldColorFeedback}
          submitHoldColorFeedback={submitHoldColorFeedback}
          colorEmoji={colorEmoji}
          modelStats={modelStats}
        />
      </div>

      {/* 하단 네비게이션 */}
      <Navigation 
        currentView={currentView}
        setCurrentView={setCurrentView}
        loadColorFeedbacks={loadColorFeedbacks}
      />
    </div>
  )
}

export default App

