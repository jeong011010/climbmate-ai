import { useEffect, useState } from 'react'

// 원형 게이지 컴포넌트
const CircularGauge = ({ value, label, color = "blue" }) => {
  const percentage = Math.min(100, Math.max(0, value || 0))
  const circumference = 2 * Math.PI * 45
  const strokeDashoffset = circumference - (percentage / 100) * circumference
  
  const colorMap = {
    blue: { stroke: "stroke-blue-500", text: "text-blue-600" },
    green: { stroke: "stroke-green-500", text: "text-green-600" },
    orange: { stroke: "stroke-orange-500", text: "text-orange-600" },
    red: { stroke: "stroke-red-500", text: "text-red-600" }
  }
  
  const colors = colorMap[color] || colorMap.blue
  
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="transform -rotate-90" width="96" height="96">
          {/* 배경 원 */}
          <circle
            cx="48"
            cy="48"
            r="45"
            stroke="currentColor"
            strokeWidth="8"
            fill="none"
            className="text-slate-200"
          />
          {/* 진행 원 */}
          <circle
            cx="48"
            cy="48"
            r="45"
            stroke="currentColor"
            strokeWidth="8"
            fill="none"
            className={colors.stroke}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`text-2xl font-bold ${colors.text}`}>
            {percentage.toFixed(0)}%
          </span>
        </div>
      </div>
      <span className="text-sm text-slate-600 mt-2 text-center">{label}</span>
    </div>
  )
}

const FeedbacksPage = ({ 
  feedbacksLoading, 
  colorFeedbacks, 
  loadColorFeedbacks,
  trainColorModel,
  confirmFeedback,
  confirmAllFeedbacks,
  deleteFeedback
}) => {
  const [mlStats, setMlStats] = useState(null)
  const [statsLoading, setStatsLoading] = useState(false)
  
  // ML 통계 로드
  const loadMlStats = async () => {
    setStatsLoading(true)
    try {
      const response = await fetch('/api/ml-model-stats')
      const data = await response.json()
      setMlStats(data)
    } catch (error) {
      console.error('ML 통계 로드 실패:', error)
    } finally {
      setStatsLoading(false)
    }
  }
  
  // 페이지 로드 시 & 피드백 변경 시 통계 로드
  useEffect(() => {
    loadMlStats()
  }, [colorFeedbacks.length])
  
  return (
    <div className="w-full px-2 sm:px-4">
      <div className="glass-card p-4 sm:p-6">
        <div className="flex justify-between items-center mb-4 flex-wrap gap-2">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-800">🎨 색상 피드백 관리</h2>
          <div className="flex gap-2">
            {colorFeedbacks.filter(f => !f.confirmed).length > 0 && (
              <button
                onClick={confirmAllFeedbacks}
                disabled={feedbacksLoading}
                className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg hover:shadow-lg transition-all font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                ✅ 전체 확인
              </button>
            )}
            <button
              onClick={() => { loadColorFeedbacks(); loadMlStats(); }}
              className="px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all"
            >
              🔄 새로고침
            </button>
          </div>
        </div>
        
        {/* ML 모델 상태 대시보드 */}
        {mlStats && (
          <div className="mb-6 p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-200 shadow-lg">
            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
              🤖 ML 모델 성능 대시보드
              {mlStats.model_exists && <span className="text-sm px-2 py-1 bg-green-500 text-white rounded-full">✓ 학습됨</span>}
              {!mlStats.model_exists && <span className="text-sm px-2 py-1 bg-slate-400 text-white rounded-full">미학습</span>}
            </h3>
            
            {/* 원형 게이지 */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 mb-6">
              <CircularGauge 
                value={mlStats.rule_based_accuracy} 
                label="규칙 기반 정확도" 
                color="blue"
              />
              {mlStats.ml_test_accuracy && (
                <CircularGauge 
                  value={mlStats.ml_test_accuracy} 
                  label="ML 테스트 정확도" 
                  color="green"
                />
              )}
              {mlStats.ml_cv_accuracy && (
                <CircularGauge 
                  value={mlStats.ml_cv_accuracy} 
                  label="ML CV 정확도" 
                  color="orange"
                />
              )}
              <CircularGauge 
                value={(mlStats.total_samples / 100) * 100} 
                label={`학습 데이터 ${mlStats.total_samples}개`}
                color={mlStats.can_train ? "green" : "red"}
              />
            </div>
            
            {/* 색상별 분포 */}
            {mlStats.color_distribution && Object.keys(mlStats.color_distribution).length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-bold text-slate-700 mb-2">📊 색상별 데이터 분포</h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2">
                  {Object.entries(mlStats.color_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([color, count]) => (
                      <div key={color} className="px-3 py-2 bg-white rounded-lg border border-slate-200 text-center">
                        <div className="text-xs text-slate-600 uppercase">{color}</div>
                        <div className="text-lg font-bold text-slate-800">{count}</div>
                      </div>
                    ))}
                </div>
              </div>
            )}
            
            {/* 오분류 Top 5 */}
            {mlStats.top_misclassifications && Object.keys(mlStats.top_misclassifications).length > 0 && (
              <div>
                <h4 className="text-sm font-bold text-slate-700 mb-2">⚠️ 자주 틀리는 색상 조합 (Top 5)</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                  {Object.entries(mlStats.top_misclassifications)
                    .slice(0, 5)
                    .map(([combination, count]) => (
                      <div key={combination} className="px-3 py-2 bg-red-50 rounded-lg border border-red-200 text-sm">
                        <span className="font-mono text-red-700">{combination}</span>
                        <span className="ml-2 text-red-600 font-bold">{count}회</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}

        {feedbacksLoading ? (
          <p className="text-slate-600 text-center py-8">피드백 로딩 중...</p>
        ) : colorFeedbacks.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-slate-600 mb-4">아직 피드백이 없습니다.</p>
            <p className="text-slate-500 text-sm">홀드를 클릭하고 색상 피드백을 제출해보세요!</p>
          </div>
        ) : (
          <>
            <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <p className="text-sm text-blue-800">
                  📊 총 <span className="font-bold text-lg">{colorFeedbacks.length}</span>개의 피드백
                  <span className="mx-2">|</span>
                  ✅ 확인됨: <span className="font-bold">{colorFeedbacks.filter(f => f.confirmed).length}</span>개
                  <span className="mx-2">|</span>
                  ⏳ 대기 중: <span className="font-bold">{colorFeedbacks.filter(f => !f.confirmed).length}</span>개
                  {colorFeedbacks.filter(f => f.confirmed).length >= 30 && (
                    <span className="ml-2 text-green-600 font-semibold">
                      🤖 ML 학습 가능!
                    </span>
                  )}
                </p>
                
                {colorFeedbacks.filter(f => f.confirmed).length >= 30 && (
                  <button
                    onClick={trainColorModel}
                    disabled={feedbacksLoading}
                    className="px-6 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg hover:shadow-lg transition-all font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {feedbacksLoading ? '⏳ 학습 중...' : '🤖 ML 학습 시작'}
                  </button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {colorFeedbacks.map((feedback) => (
                <div key={feedback.id} className={`p-4 rounded-xl shadow-lg hover:shadow-xl transition-all ${
                  feedback.confirmed 
                    ? 'bg-blue-50 border-2 border-blue-400' 
                    : 'bg-white border-2 border-slate-200'
                }`}>
                  {/* 확인 배지 */}
                  {feedback.confirmed && (
                    <div className="mb-2 px-2 py-1 bg-blue-500 text-white text-xs rounded-full inline-block">
                      ✅ ML 학습용 확정
                    </div>
                  )}
                  
                  {/* AI 예측 vs 사용자 정답 */}
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-8 h-8 rounded-full border-2 border-slate-300"
                        style={{
                          backgroundColor: `rgb(${feedback.rgb[0]}, ${feedback.rgb[1]}, ${feedback.rgb[2]})`
                        }}
                      />
                      <span className="text-xs font-mono text-slate-600">
                        RGB({feedback.rgb[0]},{feedback.rgb[1]},{feedback.rgb[2]})
                      </span>
                    </div>
                  </div>

                  {/* AI 예측 */}
                  <div className="mb-2 p-2 bg-red-50 rounded-lg border border-red-200">
                    <p className="text-xs text-red-600 mb-1">AI 예측 ❌</p>
                    <p className="font-bold text-red-800">{(feedback.predicted_color || 'unknown').toUpperCase()}</p>
                  </div>

                  {/* 사용자 정답 */}
                  <div className="mb-3 p-2 bg-green-50 rounded-lg border border-green-200">
                    <p className="text-xs text-green-600 mb-1">사용자 정답 ✅</p>
                    <p className="font-bold text-green-800">{(feedback.user_correct_color || 'unknown').toUpperCase()}</p>
                  </div>

                  {/* 상세 정보 */}
                  <div className="text-xs text-slate-500 mb-3 space-y-1">
                    <p>📍 위치: ({Math.round(feedback.center[0])}, {Math.round(feedback.center[1])})</p>
                    <p>🎨 HSV: ({Math.round(feedback.hsv[0])}, {Math.round(feedback.hsv[1])}, {Math.round(feedback.hsv[2])})</p>
                    <p>🕐 {new Date(feedback.created_at).toLocaleString('ko-KR')}</p>
                  </div>

                  {/* 액션 버튼 */}
                  <div className="flex gap-2">
                    {!feedback.confirmed ? (
                      <>
                        <button
                          onClick={() => confirmFeedback(feedback.id)}
                          className="flex-1 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-all text-sm font-semibold"
                        >
                          ✅ 확인
                        </button>
                        <button
                          onClick={() => deleteFeedback(feedback.id)}
                          className="flex-1 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-all text-sm"
                        >
                          🗑️ 삭제
                        </button>
                      </>
                    ) : (
                      <div className="flex-1 py-2 bg-blue-500 text-white rounded-lg text-center text-sm font-semibold">
                        ✅ 확인됨 (ML 학습용)
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default FeedbacksPage

