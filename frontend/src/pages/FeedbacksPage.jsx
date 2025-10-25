const FeedbacksPage = ({ 
  feedbacksLoading, 
  colorFeedbacks, 
  loadColorFeedbacks,
  trainColorModel,
  confirmFeedback,
  deleteFeedback
}) => {
  return (
    <div className="w-full px-2 sm:px-4">
      <div className="glass-card p-4 sm:p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-800">🎨 색상 피드백 관리</h2>
          <button
            onClick={loadColorFeedbacks}
            className="px-4 py-2 bg-gradient-to-r from-purple-500 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all"
          >
            🔄 새로고침
          </button>
        </div>

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

