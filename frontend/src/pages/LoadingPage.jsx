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

