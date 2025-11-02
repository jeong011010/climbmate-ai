const LoadingPage = ({ 
  preview, 
  loadingProgress, 
  currentAnalysisStep,
  detectedHolds,
  detectedProblems 
}) => {
  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start">
      {/* 좌측: 이미지 (로딩 중에도 표시) */}
      <div className="w-full lg:w-1/2">
        {preview && (
          <div className="relative">
            <img 
              src={preview} 
              alt="Uploading" 
              className="w-full max-h-[400px] lg:max-h-[600px] object-contain rounded-2xl shadow-2xl"
            />
          </div>
        )}
      </div>
      
      {/* 우측: 로딩 상태 */}
      <div className="w-full lg:w-1/2">
        <div className="glass-card text-center p-8">
          <div className="relative w-20 h-20 mx-auto mb-6">
            <div className="w-20 h-20 border-6 border-primary-500/10 border-t-primary-500 border-r-purple-600 rounded-full animate-spin shadow-lg"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-base font-bold gradient-text">{Math.round(loadingProgress)}%</span>
            </div>
          </div>
          <p className="text-lg gradient-text font-bold mb-3 animate-pulse">{currentAnalysisStep}</p>
          
          {/* 상세 진행 단계 타임라인 */}
          <div className="bg-white/50 rounded-lg p-4 mb-4">
            <div className="space-y-2 text-xs text-left">
              {[
                { range: [0, 10], label: '📷 이미지 디코딩', icon: '📷' },
                { range: [10, 30], label: '🧠 YOLO 모델 로딩', icon: '🧠', slow: true },
                { range: [30, 40], label: '🔍 홀드 감지 실행', icon: '🔍' },
                { range: [40, 42], label: '✅ 홀드 감지 완료', icon: '✅' },
                { range: [42, 60], label: '🎨 색상 분류 & 클러스터링', icon: '🎨', slow: true },
                { range: [60, 80], label: '📊 문제 분석', icon: '📊' },
                { range: [80, 100], label: '🤖 GPT-4 분석', icon: '🤖', slow: true }
              ].map((stage, idx) => {
                const isActive = loadingProgress >= stage.range[0] && loadingProgress < stage.range[1]
                const isDone = loadingProgress >= stage.range[1]
                return (
                  <div key={idx} className="flex items-center gap-2">
                    <span className={`text-base transition-all ${
                      isActive ? 'scale-125 animate-pulse' : isDone ? 'opacity-50' : 'opacity-30'
                    }`}>
                      {isActive ? '▶' : isDone ? '✓' : '○'}
                    </span>
                    <span className={`flex-1 font-medium transition-all ${
                      isActive ? 'text-blue-600 font-bold' : isDone ? 'text-green-600 line-through' : 'text-slate-400'
                    }`}>
                      {stage.label}
                      {stage.slow && isActive && <span className="ml-1 text-[10px] text-orange-500">(시간 소요)</span>}
                    </span>
                    <span className="text-[10px] text-slate-400">{stage.range[0]}~{stage.range[1]}%</span>
                  </div>
                )
              })}
            </div>
          </div>
          
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
          
          <p className="text-sm text-slate-600 font-medium">AI가 열심히 분석 중...</p>
          <div className="flex justify-center gap-1 mt-4">
            <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
            <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
            <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LoadingPage

