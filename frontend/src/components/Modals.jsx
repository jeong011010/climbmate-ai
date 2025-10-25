const Modals = ({
  showImageModal,
  setShowImageModal,
  annotatedImage,
  preview,
  showFeedbackModal,
  setShowFeedbackModal,
  selectedProblem,
  feedbackDifficulty,
  setFeedbackDifficulty,
  feedbackType,
  setFeedbackType,
  feedbackText,
  setFeedbackText,
  submitFeedback,
  showHoldFeedbackModal,
  setShowHoldFeedbackModal,
  selectedHold,
  holdColorFeedback,
  setHoldColorFeedback,
  submitHoldColorFeedback,
  colorEmoji
}) => {
  return (
    <>
      {/* 이미지 확대 모달 */}
      {showImageModal && (
        <div className="fixed top-0 left-0 w-full h-full bg-black/90 flex items-center justify-center z-[1000] p-2 sm:p-4" onClick={() => setShowImageModal(false)}>
          <img 
            src={annotatedImage || preview} 
            alt="Climbing Wall - 확대보기" 
            className="max-w-full max-h-full rounded-xl shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
          <button 
            className="absolute top-2 right-2 sm:top-4 sm:right-4 bg-white/90 border-none rounded-full w-8 h-8 sm:w-10 sm:h-10 text-xl sm:text-2xl cursor-pointer flex items-center justify-center text-slate-800 transition-all duration-300 hover:bg-white hover:scale-110"
            onClick={() => setShowImageModal(false)}
          >
            ×
          </button>
        </div>
      )}

      {/* 피드백 모달 */}
      {showFeedbackModal && selectedProblem && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[1000] p-2 sm:p-4" onClick={() => setShowFeedbackModal(false)}>
          <div className="glass-card p-4 sm:p-6 max-w-md w-full max-h-[95vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
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
                {['크림프', '파워', '지구력', '밸런스', '다이나믹', '슬랩'].map(type => (
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
              <label className="block text-sm font-bold text-slate-700 mb-3">
                💬 추가 의견 (선택사항)
              </label>
              <textarea
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                className="w-full p-3 border-2 border-slate-200 rounded-xl resize-none focus:border-primary-500 focus:outline-none transition-all"
                rows="3"
                placeholder="난이도나 유형에 대한 추가 의견을 적어주세요..."
              />
            </div>

            {/* 버튼 */}
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowFeedbackModal(false)
                  setFeedbackDifficulty('')
                  setFeedbackType('')
                  setFeedbackText('')
                }}
                className="flex-1 py-3 bg-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-300 transition-all"
              >
                취소
              </button>
              <button
                onClick={submitFeedback}
                className="flex-1 py-3 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all"
              >
                💾 저장
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 홀드 색상 피드백 모달 */}
      {showHoldFeedbackModal && selectedHold && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[1000] p-2 sm:p-4" onClick={() => setShowHoldFeedbackModal(false)}>
          <div className="glass-card p-4 sm:p-6 max-w-md w-full max-h-[95vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
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
                  <p className="text-xs text-slate-600 mb-3">AI가 감지한 홀드 실제 색상</p>
                  <div className="flex flex-col items-center justify-center gap-2">
                    {/* 실제 RGB 색상 원형 표시 */}
                    <div 
                      className="w-20 h-20 rounded-full shadow-lg border-4 border-white"
                      style={{
                        backgroundColor: selectedHold.rgb ? 
                          `rgb(${selectedHold.rgb[0]}, ${selectedHold.rgb[1]}, ${selectedHold.rgb[2]})` : 
                          '#808080'
                      }}
                    />
                    <div className="text-xs font-mono text-slate-600">
                      {selectedHold.rgb ? 
                        `RGB(${selectedHold.rgb[0]}, ${selectedHold.rgb[1]}, ${selectedHold.rgb[2]})` : 
                        'N/A'}
                    </div>
                    <span className="text-sm font-bold gradient-text">{(selectedHold.individual_color || 'UNKNOWN').toUpperCase()}</span>
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
    </>
  )
}

export default Modals

